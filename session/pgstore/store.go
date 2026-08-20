// Package pgstore implements session.Store and session.FencedStore on
// PostgreSQL. It owns a small, self-contained schema
// (agentkit_sessions/agentkit_events/agentkit_compactions) so the library has
// no coupling to any host application's tables. Host apps may alternatively
// map session.Store onto their own event log (e.g. wzhooh's chat_events) via
// WithEventsTable; fenced single-writer operations require the native schema.
//
// Durability design (H-DB-*):
//   - Sequence allocation is a per-session row counter reserved and inserted
//     in ONE transaction (never SELECT MAX(seq)+1, which races).
//   - Every fenced mutation carries lease_token + lease_until>now() and the
//     monotonic execution fence captured at claim time.
//   - Appends are idempotent per event_id: retries return canonical seqs
//     without duplicating events or leaving gaps.
//   - Recover() closes orphaned turns and marks dangling tool calls
//     TOOL_OUTCOME_UNKNOWN (no blind retry of side-effecting work).
package pgstore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MIZUDINOV/awesome-go-agents/session"
)

var (
	// ErrNoTenant is returned when a fenced operation is attempted without a
	// configured tenant id (tenant isolation is mandatory for the native
	// schema, H-SEC-001/H-SEC-013).
	ErrNoTenant = errors.New("pgstore: tenant id required for fenced operations")
	// ErrNotNative is returned when a fenced operation is attempted on a
	// remapped (host) table that lacks the lease/tenant columns.
	ErrNotNative = errors.New("pgstore: fenced operations require the native agentkit schema (not WithEventsTable remapping)")
)

// safeIdentifier restricts interpolated table names to plain identifiers
// (H-DB-012): no quoting gymnastics, no user-controlled SQL.
var safeIdentifier = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

// Store is a PostgreSQL-backed session.Store / FencedStore.
type Store struct {
	db       *pgxpool.Pool
	events   string // table name, default "agentkit_events"
	sessions string // table name, default "agentkit_sessions"
	tenant   string // configured tenant id (native isolation)
	native   bool   // native schema (lease/tenant columns present)
}

// New returns a Store bound to pool in native mode. Native reads and writes
// require WithTenant so an unconfigured store cannot access shared tables.
func New(db *pgxpool.Pool) *Store {
	return &Store{db: db, events: "agentkit_events", sessions: "agentkit_sessions", native: true}
}

// WithEventsTable overrides the default events table name (for host-app
// remapping). Only plain identifiers are accepted. Remapped stores are not
// native: fenced operations are rejected.
func (s *Store) WithEventsTable(name string) *Store {
	if !safeIdentifier.MatchString(name) {
		panic(fmt.Sprintf("pgstore: unsafe events table name %q", name))
	}
	s.events = name
	s.native = false
	return s
}

// WithSessionsTable overrides the default sessions table name.
func (s *Store) WithSessionsTable(name string) *Store {
	if !safeIdentifier.MatchString(name) {
		panic(fmt.Sprintf("pgstore: unsafe sessions table name %q", name))
	}
	s.sessions = name
	return s
}

// WithTenant pins the tenant id applied to every native query.
func (s *Store) WithTenant(tenantID string) *Store {
	s.tenant = strings.TrimSpace(tenantID)
	return s
}

func (s *Store) requireNativeAndTenant() error {
	if !s.native {
		return ErrNotNative
	}
	if s.tenant == "" {
		return ErrNoTenant
	}
	return nil
}

func (s *Store) requireTenantForNativeQuery() error {
	if s.native && s.tenant == "" {
		return ErrNoTenant
	}
	return nil
}

// Append persists events with atomic per-session sequence allocation. This is
// the lease-less compatibility path (session.Store contract); the durable
// loop should use AppendFenced under a lease.
func (s *Store) Append(ctx context.Context, sessionID string, events []session.Event) (uint64, error) {
	if err := s.requireTenantForNativeQuery(); err != nil {
		return 0, err
	}
	if len(events) == 0 {
		seq, err := s.Sequence(ctx, sessionID)
		return seq, err
	}
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return 0, err
		}
	}
	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return 0, fmt.Errorf("pgstore: begin append: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var base uint64
	if s.native {
		// Ensure the session row exists, then reserve a contiguous block in
		// the same transaction: rollback rolls back the counter too.
		if _, err := tx.Exec(ctx, fmt.Sprintf(`INSERT INTO %s(session_id,tenant_id) VALUES($1,$2) ON CONFLICT (session_id) DO NOTHING`, s.sessions),
			sessionID, s.tenant); err != nil {
			return 0, fmt.Errorf("pgstore: ensure session: %w", err)
		}
		err = tx.QueryRow(ctx, fmt.Sprintf(`UPDATE %s SET next_seq=next_seq+$2, updated_at=now() WHERE session_id=$1 RETURNING next_seq-$2`, s.sessions),
			sessionID, len(events)).Scan(&base)
		if err != nil {
			return 0, fmt.Errorf("pgstore: reserve sequence block: %w", err)
		}
	} else {
		err = tx.QueryRow(ctx, fmt.Sprintf(`SELECT COALESCE(MAX(seq),0) FROM %s WHERE session_id=$1`, s.events), sessionID).Scan(&base)
		if err != nil {
			return 0, fmt.Errorf("pgstore: read sequence: %w", err)
		}
	}
	next := base
	for i := range events {
		next++
		event := events[i]
		if err := insertEventTx(ctx, tx, s.events, s.tenant, sessionID, next, event); err != nil {
			return 0, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("pgstore: commit append: %w", err)
	}
	return next, nil
}

// AppendFenced appends under a valid lease, idempotent per event ID.
func (s *Store) AppendFenced(ctx context.Context, lease session.Lease, events []session.Event) (uint64, error) {
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return 0, err
		}
	}
	if err := s.requireNativeAndTenant(); err != nil {
		return 0, err
	}
	if len(events) == 0 {
		return s.Sequence(ctx, lease.SessionID)
	}
	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return 0, fmt.Errorf("pgstore: begin fenced append: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	if err := s.checkLeaseTx(ctx, tx, lease); err != nil {
		return 0, err
	}

	// Dedup already-committed events by event id (idempotent retry).
	existing := map[string]uint64{}
	ids := eventIDs(events)
	if len(ids) > 0 {
		rows, err := tx.Query(ctx, fmt.Sprintf(`SELECT event_id, seq FROM %s WHERE session_id=$1 AND event_id = ANY($2)`, s.events), lease.SessionID, ids)
		if err != nil {
			return 0, fmt.Errorf("pgstore: lookup idempotency keys: %w", err)
		}
		for rows.Next() {
			var id string
			var seq uint64
			if err := rows.Scan(&id, &seq); err != nil {
				rows.Close()
				return 0, fmt.Errorf("pgstore: scan idempotency row: %w", err)
			}
			existing[id] = seq
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return 0, fmt.Errorf("pgstore: iterate idempotency rows: %w", err)
		}
	}

	var base uint64
	var newEvents []session.Event
	var lastSeq uint64
	for i := range events {
		if events[i].ID != "" {
			if seq, ok := existing[events[i].ID]; ok {
				if seq > lastSeq {
					lastSeq = seq
				}
				continue
			}
		}
		newEvents = append(newEvents, events[i])
		if events[i].ID != "" {
			// De-duplicate repeated IDs within the same input batch as well as
			// IDs already committed in the database.
			existing[events[i].ID] = 0
		}
	}
	if len(newEvents) > 0 {
		if err := tx.QueryRow(ctx, fmt.Sprintf(`UPDATE %s SET next_seq=next_seq+$2, updated_at=now() WHERE session_id=$1 RETURNING next_seq-$2`, s.sessions),
			lease.SessionID, len(newEvents)).Scan(&base); err != nil {
			return 0, fmt.Errorf("pgstore: reserve fenced sequence block: %w", err)
		}
		next := base
		for i := range newEvents {
			next++
			if err := insertEventTx(ctx, tx, s.events, s.tenant, lease.SessionID, next, newEvents[i]); err != nil {
				return 0, err
			}
			lastSeq = next
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("pgstore: commit fenced append: %w", err)
	}
	if lastSeq == 0 {
		return s.Sequence(ctx, lease.SessionID)
	}
	return lastSeq, nil
}

// AppendFencedCommitted is the canonical-batch variant used by new AgentKit
// loops. The legacy method remains available for host adapters. The lease
// makes the post-commit lookup safe from another writer for this session.
func (s *Store) AppendFencedCommitted(ctx context.Context, lease session.Lease, events []session.Event) (session.CommittedBatch, error) {
	if err := s.requireNativeAndTenant(); err != nil {
		return session.CommittedBatch{}, err
	}
	if len(events) == 0 {
		return session.CommittedBatch{}, nil
	}
	prepared := make([]session.Event, len(events))
	for i := range events {
		prepared[i] = events[i].Clone()
		if err := prepared[i].Validate(); err != nil {
			return session.CommittedBatch{}, err
		}
		if prepared[i].SessionID != "" && prepared[i].SessionID != lease.SessionID {
			return session.CommittedBatch{}, fmt.Errorf("pgstore: event %s belongs to %q, not %q", prepared[i].ID, prepared[i].SessionID, lease.SessionID)
		}
		prepared[i].Normalize()
		if prepared[i].ID == "" {
			prepared[i].ID = randomToken()
		}
	}
	if _, err := s.AppendFenced(ctx, lease, prepared); err != nil {
		return session.CommittedBatch{}, err
	}
	all, err := s.loadByIDs(ctx, lease.SessionID, eventIDs(prepared))
	if err != nil {
		return session.CommittedBatch{}, err
	}
	byID := make(map[string]session.Event, len(all))
	for _, event := range all {
		if event.ID != "" {
			byID[event.ID] = event
		}
	}
	committed := make([]session.Event, 0, len(prepared))
	for _, event := range prepared {
		if canonical, ok := byID[event.ID]; ok {
			committed = append(committed, canonical.Clone())
		}
	}
	return session.CommittedBatch{Events: committed}, nil
}

// AppendCommitted is the lease-less compatibility form. New durable loops
// should use AppendFencedCommitted so a host-owned fence protects the append.
func (s *Store) AppendCommitted(ctx context.Context, sessionID string, events []session.Event) (session.CommittedBatch, error) {
	if err := s.requireTenantForNativeQuery(); err != nil {
		return session.CommittedBatch{}, err
	}
	if len(events) == 0 {
		return session.CommittedBatch{}, nil
	}
	prepared := make([]session.Event, len(events))
	for i := range events {
		prepared[i] = events[i].Clone()
		if err := prepared[i].Validate(); err != nil {
			return session.CommittedBatch{}, err
		}
		if prepared[i].SessionID != "" && prepared[i].SessionID != sessionID {
			return session.CommittedBatch{}, fmt.Errorf("pgstore: event %s belongs to %q, not %q", prepared[i].ID, prepared[i].SessionID, sessionID)
		}
		prepared[i].Normalize()
		if prepared[i].ID == "" {
			prepared[i].ID = randomToken()
		}
	}
	if _, err := s.Append(ctx, sessionID, prepared); err != nil {
		return session.CommittedBatch{}, err
	}
	all, err := s.loadByIDs(ctx, sessionID, eventIDs(prepared))
	if err != nil {
		return session.CommittedBatch{}, err
	}
	byID := make(map[string]session.Event, len(all))
	for _, event := range all {
		if event.ID != "" {
			byID[event.ID] = event
		}
	}
	committed := make([]session.Event, 0, len(prepared))
	for _, event := range prepared {
		if canonical, ok := byID[event.ID]; ok {
			committed = append(committed, canonical.Clone())
		}
	}
	return session.CommittedBatch{Events: committed}, nil
}

// ClaimLease acquires the session lease, bumping the execution fence.
func (s *Store) ClaimLease(ctx context.Context, sessionID, owner string, ttl time.Duration, tenantID string) (session.Lease, error) {
	if err := s.requireNativeAndTenant(); err != nil {
		return session.Lease{}, err
	}
	if tenantID != "" && tenantID != s.tenant {
		return session.Lease{}, fmt.Errorf("pgstore: tenant mismatch (configured %q, requested %q)", s.tenant, tenantID)
	}
	if ttl <= 0 {
		ttl = 30 * time.Second
	}
	token := randomToken()
	var fence uint64
	err := s.db.QueryRow(ctx, fmt.Sprintf(`INSERT INTO %s(session_id,tenant_id,next_seq,lease_owner,lease_token,lease_until,execution_fence,updated_at)
VALUES($1,$2,0,$3,$4,now()+$5,1,now())
ON CONFLICT (session_id) DO UPDATE
SET lease_owner=$3, lease_token=$4, lease_until=now()+$5, execution_fence=%s.execution_fence+1, updated_at=now()
WHERE (%s.lease_until IS NULL OR %s.lease_until < now()) AND %s.tenant_id=$2
RETURNING %s.execution_fence`, s.sessions, s.sessions, s.sessions, s.sessions, s.sessions, s.sessions),
		sessionID, s.tenant, owner, token, ttl).Scan(&fence)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return session.Lease{}, fmt.Errorf("%w: session %s", session.ErrLeaseHeld, sessionID)
		}
		return session.Lease{}, fmt.Errorf("pgstore: claim lease: %w", err)
	}
	return session.Lease{
		SessionID: sessionID, Owner: owner, Token: token, Fence: fence,
		ExpiresAt: time.Now().Add(ttl),
	}, nil
}

// RenewLease extends the caller's lease.
func (s *Store) RenewLease(ctx context.Context, lease session.Lease) (session.Lease, error) {
	if err := s.requireNativeAndTenant(); err != nil {
		return session.Lease{}, err
	}
	var fence uint64
	err := s.db.QueryRow(ctx, fmt.Sprintf(`UPDATE %s SET lease_until=now()+$5, updated_at=now()
WHERE session_id=$1 AND tenant_id=$2 AND lease_token=$3 AND execution_fence=$4 AND lease_until>now()
RETURNING execution_fence`, s.sessions),
		lease.SessionID, s.tenant, lease.Token, lease.Fence, 30*time.Second).Scan(&fence)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return session.Lease{}, fmt.Errorf("%w: session %s", session.ErrLeaseLost, lease.SessionID)
		}
		return session.Lease{}, fmt.Errorf("pgstore: renew lease: %w", err)
	}
	lease.Fence = fence
	lease.ExpiresAt = time.Now().Add(30 * time.Second)
	return lease, nil
}

// ReleaseLease releases the lease if it still belongs to the caller.
func (s *Store) ReleaseLease(ctx context.Context, lease session.Lease) error {
	if err := s.requireNativeAndTenant(); err != nil {
		return err
	}
	_, err := s.db.Exec(ctx, fmt.Sprintf(`UPDATE %s SET lease_owner=NULL, lease_token=NULL, lease_until=NULL, updated_at=now()
WHERE session_id=$1 AND tenant_id=$2 AND lease_token=$3`, s.sessions),
		lease.SessionID, s.tenant, lease.Token)
	if err != nil {
		return fmt.Errorf("pgstore: release lease: %w", err)
	}
	return nil
}

// Recover reconciles an interrupted session under the lease, distinguishing
// calls that reached dispatch from calls that were still only admitted.
func (s *Store) Recover(ctx context.Context, lease session.Lease) (*session.RecoveryReport, error) {
	if err := s.requireNativeAndTenant(); err != nil {
		return nil, err
	}
	events, err := s.Load(ctx, lease.SessionID, 0, 0)
	if err != nil {
		return nil, err
	}
	report := &session.RecoveryReport{}
	openTurns := 0
	openTurnID := ""
	openRunID := ""
	for _, e := range events {
		if e.Type == session.EventTurnStart {
			openTurns++
			openTurnID = e.TurnID
			openRunID = e.RunID
		} else if e.Type == session.EventTurnEnd {
			openTurns--
			if openTurns == 0 {
				openTurnID = ""
				openRunID = ""
			}
		}
	}
	var recovery []session.Event
	recovery = append(recovery, session.InterruptedDraftEvents(events, lease.SessionID)...)
	stepEnds := session.InterruptedStepEndEvents(events, lease.SessionID)
	report.StepsClosed = len(stepEnds)
	if openTurns > 0 {
		report.TurnClosed = true
	}
	callIDs := make([]string, 0)
	callEvents := make(map[string]session.Event)
	seen := map[string]bool{}
	resultIDs := map[string]bool{}
	pendingApproval := map[string]bool{}
	resolvedApproval := map[string]bool{}
	dispatched := map[string]bool{}
	for _, e := range events {
		switch e.Type {
		case session.EventToolCall:
			var p struct {
				CallID string `json:"call_id"`
			}
			_ = json.Unmarshal(e.Data, &p)
			if p.CallID != "" && !seen[p.CallID] {
				seen[p.CallID] = true
				callIDs = append(callIDs, p.CallID)
			}
			if p.CallID != "" {
				callEvents[p.CallID] = e
			}
		case session.EventToolDispatched, session.EventToolRunning:
			if e.CallID != "" {
				dispatched[e.CallID] = true
			}
		case session.EventAssistantMessage:
			var p struct {
				ToolCalls []session.ToolCall `json:"tool_calls"`
			}
			if json.Unmarshal(e.Data, &p) == nil {
				for _, call := range p.ToolCalls {
					if call.CallID == "" {
						continue
					}
					if !seen[call.CallID] {
						seen[call.CallID] = true
						callIDs = append(callIDs, call.CallID)
					}
					if _, exists := callEvents[call.CallID]; !exists {
						callEvents[call.CallID] = session.Event{RunID: e.RunID, TurnID: e.TurnID, StepID: e.StepID, Data: session.ToolCallPayload(call.CallID, call.Name, call.Arguments)}
					}
				}
			}
		case session.EventToolResult:
			var p struct {
				CallID string `json:"call_id"`
			}
			_ = json.Unmarshal(e.Data, &p)
			if p.CallID != "" {
				resultIDs[p.CallID] = true
			}
		case session.EventApprovalRequested:
			if e.CallID != "" {
				pendingApproval[e.CallID] = true
			}
		case session.EventApprovalResolved:
			var p session.ApprovalResolvedPayload
			if json.Unmarshal(e.Data, &p) == nil && p.CallID != "" {
				pendingApproval[p.CallID] = false
				resolvedApproval[p.CallID] = true
			}
		}
	}
	for _, callID := range callIDs {
		if !resultIDs[callID] && !pendingApproval[callID] && (!resolvedApproval[callID] || dispatched[callID]) {
			report.DanglingCalls = append(report.DanglingCalls, callID)
			callEvent := callEvents[callID]
			var callPayload session.ToolCall
			_ = json.Unmarshal(callEvent.Data, &callPayload)
			code := "TOOL_OUTCOME_UNKNOWN"
			if !dispatched[callID] {
				code = "ABORTED_BEFORE_DISPATCH"
			}
			recovery = append(recovery, session.Event{
				Type: session.EventToolResult, SessionID: lease.SessionID, RunID: callEvent.RunID,
				TurnID: callEvent.TurnID, StepID: callEvent.StepID, CallID: callID,
				ID: "recover:" + callID,
				Data: mustJSON(map[string]any{
					"call_id": callID, "name": callPayload.Name, "is_error": true, "code": code,
					"output": map[string]any{},
				}),
			})
		}
	}
	// Settle tool results before closing the step and turn so the durable
	// lifecycle remains tool/result -> step/end -> turn/end.
	recovery = append(recovery, stepEnds...)
	if openTurns > 0 {
		recovery = append(recovery, session.Event{
			ID: "recover:turn-end:" + openTurnID, RunID: openRunID, TurnID: openTurnID,
			Type: session.EventTurnEnd, SessionID: lease.SessionID,
			Data: mustJSON(map[string]any{"reason": "interrupted"}),
		})
	}
	if len(recovery) > 0 {
		if _, err := s.AppendFenced(ctx, lease, recovery); err != nil {
			return nil, err
		}
		report.EventsAppended = len(recovery)
	}
	return report, nil
}

// SaveCompactionCheckpoint records a durable compaction ledger entry.
func (s *Store) SaveCompactionCheckpoint(ctx context.Context, lease session.Lease, record session.CompactionCheckpoint) error {
	if err := s.requireNativeAndTenant(); err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return fmt.Errorf("pgstore: begin compaction checkpoint: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err := s.checkLeaseTx(ctx, tx, lease); err != nil {
		return err
	}
	summary, err := json.Marshal(map[string]any{
		"generation": record.Generation, "transaction_id": record.TransactionID,
		"through_seq": record.ThroughSeq, "shadowed_seqs": record.ShadowedSeqs,
		"summary": record.Summary, "source_fingerprint": record.SourceFingerprint,
	})
	if err != nil {
		return fmt.Errorf("pgstore: encode checkpoint: %w", err)
	}
	_, err = tx.Exec(ctx, `INSERT INTO agentkit_compactions(session_id,generation,transaction_id,through_seq,shadowed_seqs,summary_json,summary_sha256,source_fingerprint,created_at)
VALUES($1,$2,$3,$4,$5,$6,$7,$8,now())
ON CONFLICT (session_id,generation) DO UPDATE
SET transaction_id=EXCLUDED.transaction_id, through_seq=EXCLUDED.through_seq,
    shadowed_seqs=EXCLUDED.shadowed_seqs, summary_json=EXCLUDED.summary_json,
    summary_sha256=EXCLUDED.summary_sha256, source_fingerprint=EXCLUDED.source_fingerprint, created_at=now()`,
		record.SessionID, record.Generation, record.TransactionID, record.ThroughSeq,
		record.ShadowedSeqs, summary, record.SummarySHA256, record.SourceFingerprint)
	if err != nil {
		return fmt.Errorf("pgstore: upsert compaction checkpoint: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("pgstore: commit compaction checkpoint: %w", err)
	}
	return nil
}

func (s *Store) checkLeaseTx(ctx context.Context, tx pgx.Tx, lease session.Lease) error {
	var fence uint64
	err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT execution_fence FROM %s
WHERE session_id=$1 AND tenant_id=$2 AND lease_token=$3 AND lease_until>now()`, s.sessions),
		lease.SessionID, s.tenant, lease.Token).Scan(&fence)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("%w: session %s", session.ErrLeaseLost, lease.SessionID)
		}
		return fmt.Errorf("pgstore: check lease: %w", err)
	}
	if fence != lease.Fence {
		return fmt.Errorf("%w: session %s (fence %d != %d)", session.ErrLeaseLost, lease.SessionID, lease.Fence, fence)
	}
	return nil
}

func insertEventTx(ctx context.Context, tx pgx.Tx, table, tenant, sessionID string, seq uint64, event session.Event) error {
	event = event.Clone()
	event.Normalize()
	data := event.Data
	if len(data) == 0 {
		data = json.RawMessage("{}")
	}
	_, err := tx.Exec(ctx, fmt.Sprintf(`INSERT INTO %s(session_id,tenant_id,seq,event_id,type,format_version,timestamp,data,surface,source_seqs,run_id,turn_id,step_id,call_id)
VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)`, table),
		sessionID, tenant, seq, nullableText(event.ID), string(event.Type), event.NormalizedFormatVersion(),
		event.Timestamp, data, event.Type.Surface(), event.SourceSeqs,
		event.RunID, event.TurnID, event.StepID, event.CallID)
	if err != nil {
		return fmt.Errorf("pgstore: insert event: %w", err)
	}
	return nil
}

func nullableText(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func eventIDs(events []session.Event) []string {
	ids := make([]string, 0, len(events))
	seen := map[string]bool{}
	for _, e := range events {
		if e.ID != "" && !seen[e.ID] {
			seen[e.ID] = true
			ids = append(ids, e.ID)
		}
	}
	return ids
}

func (s *Store) Load(ctx context.Context, sessionID string, afterSeq uint64, limit int) ([]session.Event, error) {
	if err := s.requireTenantForNativeQuery(); err != nil {
		return nil, err
	}
	args := []any{sessionID, afterSeq}
	predicate := ""
	if s.native && s.tenant != "" {
		predicate = ` AND tenant_id=$3`
		args = append(args, s.tenant)
	}
	query := fmt.Sprintf(`SELECT seq,type,timestamp,data,source_seqs%s FROM %s WHERE session_id=$1%s AND seq>$2 ORDER BY seq`,
		s.nativeSelectSuffix(), s.events, predicate)
	if limit > 0 {
		query += fmt.Sprintf(` LIMIT $%d`, len(args)+1)
		args = append(args, limit)
	}
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("pgstore: load events: %w", err)
	}
	defer rows.Close()
	return scanEvents(rows, s.native, sessionID)
}

func (s *Store) loadByIDs(ctx context.Context, sessionID string, ids []string) ([]session.Event, error) {
	if err := s.requireTenantForNativeQuery(); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, nil
	}
	args := []any{sessionID, ids}
	predicate := ""
	if s.native && s.tenant != "" {
		predicate = ` AND tenant_id=$3`
		args = append(args, s.tenant)
	}
	suffix := s.nativeSelectSuffix()
	if !s.native {
		suffix = ",event_id"
	}
	query := fmt.Sprintf(`SELECT seq,type,timestamp,data,source_seqs%s FROM %s WHERE session_id=$1 AND event_id = ANY($2)%s ORDER BY seq`, suffix, s.events, predicate)
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("pgstore: load events by id: %w", err)
	}
	defer rows.Close()
	if s.native {
		return scanEvents(rows, true, sessionID)
	}
	return scanEventsWithIDs(rows, sessionID)
}

func (s *Store) Tail(ctx context.Context, sessionID string, limit int) ([]session.Event, error) {
	if err := s.requireTenantForNativeQuery(); err != nil {
		return nil, err
	}
	if limit <= 0 {
		return s.Load(ctx, sessionID, 0, 0)
	}
	args := []any{sessionID, limit}
	predicate := ""
	if s.native && s.tenant != "" {
		predicate = ` AND tenant_id=$3`
		args = append(args, s.tenant)
	}
	query := fmt.Sprintf(`SELECT seq,type,timestamp,data,source_seqs%s FROM (
    SELECT seq,type,timestamp,data,source_seqs%s FROM %s WHERE session_id=$1%s ORDER BY seq DESC LIMIT $2
) t ORDER BY seq`, s.nativeSelectSuffix(), s.nativeSelectSuffix(), s.events, predicate)
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("pgstore: load tail: %w", err)
	}
	defer rows.Close()
	return scanEvents(rows, s.native, sessionID)
}

func (s *Store) Sequence(ctx context.Context, sessionID string) (uint64, error) {
	if err := s.requireTenantForNativeQuery(); err != nil {
		return 0, err
	}
	if s.native {
		var seq uint64
		err := s.db.QueryRow(ctx, fmt.Sprintf(`SELECT COALESCE((SELECT next_seq FROM %s WHERE session_id=$1 AND tenant_id=$2),0)`, s.sessions),
			sessionID, s.tenant).Scan(&seq)
		if err != nil {
			return 0, fmt.Errorf("pgstore: sequence: %w", err)
		}
		return seq, nil
	}
	var seq uint64
	err := s.db.QueryRow(ctx, fmt.Sprintf(`SELECT COALESCE(MAX(seq),0) FROM %s WHERE session_id=$1`, s.events), sessionID).Scan(&seq)
	if err != nil {
		return 0, fmt.Errorf("pgstore: sequence: %w", err)
	}
	return seq, nil
}

// nativeSelectSuffix lists the additional columns present only in the native
// schema (empty for remapped tables).
func (s *Store) nativeSelectSuffix() string {
	if s.native {
		return `,event_id,format_version,run_id,turn_id,step_id,call_id`
	}
	return ""
}

func scanEvents(rows pgx.Rows, native bool, sessionID string) ([]session.Event, error) {
	var events []session.Event
	for rows.Next() {
		var e session.Event
		var eventType string
		var ts time.Time
		var data []byte
		var sourceSeqs []uint64
		if native {
			var eventID *string
			var formatVersion int
			var runID, turnID, stepID, callID string
			if err := rows.Scan(&e.Seq, &eventType, &ts, &data, &sourceSeqs,
				&eventID, &formatVersion, &runID, &turnID, &stepID, &callID); err != nil {
				return nil, fmt.Errorf("pgstore: scan event: %w", err)
			}
			if eventID != nil {
				e.ID = *eventID
			}
			e.FormatVersion = formatVersion
			e.RunID, e.TurnID, e.StepID, e.CallID = runID, turnID, stepID, callID
		} else {
			if err := rows.Scan(&e.Seq, &eventType, &ts, &data, &sourceSeqs); err != nil {
				return nil, fmt.Errorf("pgstore: scan event: %w", err)
			}
		}
		e.Type = session.EventType(eventType)
		e.SessionID = sessionID
		e.Timestamp = ts
		e.Data = json.RawMessage(data)
		e.SourceSeqs = sourceSeqs
		e.Surface = e.Type.Surface()
		events = append(events, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgstore: iterate events: %w", err)
	}
	return events, nil
}

func scanEventsWithIDs(rows pgx.Rows, sessionID string) ([]session.Event, error) {
	var events []session.Event
	for rows.Next() {
		var e session.Event
		var eventID *string
		var eventType string
		var ts time.Time
		var data []byte
		var sourceSeqs []uint64
		if err := rows.Scan(&e.Seq, &eventType, &ts, &data, &sourceSeqs, &eventID); err != nil {
			return nil, fmt.Errorf("pgstore: scan event: %w", err)
		}
		if eventID != nil {
			e.ID = *eventID
		}
		e.Type = session.EventType(eventType)
		e.SessionID = sessionID
		e.Timestamp = ts
		e.Data = json.RawMessage(data)
		e.SourceSeqs = sourceSeqs
		e.Surface = e.Type.Surface()
		e.Normalize()
		events = append(events, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgstore: iterate events: %w", err)
	}
	return events, nil
}

func randomToken() string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf[:])
}

func mustJSON(v any) json.RawMessage {
	encoded, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("pgstore: marshal payload: %v", err))
	}
	return encoded
}

// Migration is the SQL to create the self-contained schema. It is additive and
// idempotent: existing agentkit_events tables gain the new columns via
// ALTER TABLE ... IF NOT EXISTS, no destructive rewrite (H-DB-013/H-DB-014).
// Host apps mount it into their migrate chain, or remap Store onto their own
// table via WithEventsTable (fenced ops then require the native schema).
const Migration = `
CREATE TABLE IF NOT EXISTS agentkit_sessions (
    session_id text PRIMARY KEY,
    tenant_id text NOT NULL DEFAULT '',
    next_seq bigint NOT NULL DEFAULT 0 CHECK (next_seq >= 0),
    format_version int NOT NULL DEFAULT 1,
    lease_owner text,
    lease_token text,
    lease_until timestamptz,
    execution_fence bigint NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agentkit_events (
    session_id text NOT NULL,
    seq bigint NOT NULL CHECK (seq > 0),
    type text NOT NULL,
    timestamp timestamptz NOT NULL DEFAULT now(),
    data jsonb NOT NULL DEFAULT '{}'::jsonb,
    surface boolean NOT NULL DEFAULT false,
    source_seqs bigint[] NOT NULL DEFAULT '{}',
    PRIMARY KEY (session_id, seq)
);

-- Additive upgrade of pre-existing agentkit_events tables.
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS event_id text;
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS format_version int NOT NULL DEFAULT 1;
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT '';
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS run_id text NOT NULL DEFAULT '';
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS turn_id text NOT NULL DEFAULT '';
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS step_id text NOT NULL DEFAULT '';
ALTER TABLE agentkit_events ADD COLUMN IF NOT EXISTS call_id text NOT NULL DEFAULT '';

CREATE UNIQUE INDEX IF NOT EXISTS agentkit_events_event_id
    ON agentkit_events (session_id, event_id) WHERE event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS agentkit_events_tenant_session
    ON agentkit_events (tenant_id, session_id);
CREATE INDEX IF NOT EXISTS agentkit_sessions_session_seq
    ON agentkit_sessions (session_id);

CREATE TABLE IF NOT EXISTS agentkit_compactions (
    session_id text NOT NULL,
    generation bigint NOT NULL,
    transaction_id text NOT NULL,
    through_seq bigint NOT NULL DEFAULT 0,
    shadowed_seqs bigint[] NOT NULL DEFAULT '{}',
    summary_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    summary_sha256 text NOT NULL DEFAULT '',
    source_fingerprint text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, generation)
);
`
