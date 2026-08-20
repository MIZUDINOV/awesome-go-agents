package session

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Sentinel errors for durable single-writer semantics. Stores return these
// (possibly wrapped) so the agent loop can classify them without
// provider/store-specific knowledge.
var (
	// ErrLeaseLost means the worker's lease token no longer matches or has
	// expired: writes must stop, never silently proceed (H-DB-008).
	ErrLeaseLost = errors.New("session: lease lost or expired")
	// ErrLeaseHeld means another worker currently holds the session lease.
	ErrLeaseHeld = errors.New("session: lease held by another worker")
)

// Lease is a durable single-writer lease over one session. A worker must hold
// a valid lease (token + expiry + execution fence) for every fenced mutation;
// a lost lease is never silently overwritten.
type Lease struct {
	SessionID string
	Owner     string
	// Token is the opaque lease credential returned by ClaimLease.
	Token string
	// Fence is the monotonic execution fence captured at claim time. Every
	// committed write must see the same fence; a replaced worker's stale
	// writes are rejected.
	Fence uint64
	// ExpiresAt bounds the lease; RenewLease extends it.
	ExpiresAt time.Time
}

// RecoveryReport summarises what Recover synthesized for a session.
type RecoveryReport struct {
	// TurnClosed is true when an orphaned turn/start was closed with an
	// interrupted turn/end.
	TurnClosed bool
	// DanglingCalls are tool/call ids whose outcome was unknown at crash time.
	// They received synthetic TOOL_OUTCOME_UNKNOWN results (never blind retry
	// of side-effecting work, H-RECOVERY-003/H-ANTI-009).
	DanglingCalls []string
	// EventsAppended is the number of recovery events persisted.
	EventsAppended int
}

// FencedStore extends Store with durable single-writer semantics: leases,
// fences, idempotent fenced appends, recovery, and durable compaction
// checkpoints. MemoryStore implements it with an in-memory lease; pgstore
// implements it against PostgreSQL.
type FencedStore interface {
	Store

	// ClaimLease acquires the session lease, bumping the execution fence.
	// Returns ErrLeaseHeld when another worker holds an unexpired lease.
	// tenantID must be non-empty for stores that enforce tenancy.
	ClaimLease(ctx context.Context, sessionID, owner string, ttl time.Duration, tenantID string) (Lease, error)

	// RenewLease extends the caller's lease; ErrLeaseLost if invalid/expired.
	RenewLease(ctx context.Context, lease Lease) (Lease, error)

	// AppendFenced appends events under the lease. It is idempotent per
	// event ID: a retried batch returns the canonical sequences without
	// duplicating events or leaving gaps (H-DB-011).
	AppendFenced(ctx context.Context, lease Lease, events []Event) (uint64, error)

	// ReleaseLease releases the lease if it still belongs to the caller.
	ReleaseLease(ctx context.Context, lease Lease) error

	// Recover reconciles an interrupted session under the lease: closes
	// orphaned turns and synthesizes TOOL_OUTCOME_UNKNOWN results for dangling
	// tool calls. All recovery writes are fenced and idempotent.
	Recover(ctx context.Context, lease Lease) (*RecoveryReport, error)

	// SaveCompactionCheckpoint durably records a compaction checkpoint
	// (summary text, exact shadowed seqs, fingerprints) alongside the event
	// log for audit and drift detection. The event log remains the truth.
	SaveCompactionCheckpoint(ctx context.Context, lease Lease, record CompactionCheckpoint) error
}

// FencedBatchStore is the stronger additive form of AppendFenced. It returns
// canonical committed copies so an execution loop can publish exactly what
// was durably assigned without a racy tail re-read.
type FencedBatchStore interface {
	FencedStore
	AppendFencedCommitted(ctx context.Context, lease Lease, events []Event) (CommittedBatch, error)
}

// CompactionCheckpoint is the durable compaction ledger entry.
type CompactionCheckpoint struct {
	SessionID         string
	Generation        uint64
	TransactionID     string
	ThroughSeq        uint64
	ShadowedSeqs      []uint64
	Summary           string
	SummarySHA256     string
	SourceFingerprint string
}

// ClaimLease acquires an in-memory lease (MemoryStore). Fence increments so a
// stale lease from a previous claim can never pass a fenced write.
func (s *MemoryStore) ClaimLease(_ context.Context, sessionID, owner string, ttl time.Duration, tenantID string) (Lease, error) {
	if ttl <= 0 {
		ttl = 30 * time.Second
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.leases == nil {
		s.leases = make(map[string]Lease)
	}
	if existing, ok := s.leases[sessionID]; ok && time.Now().Before(existing.ExpiresAt) {
		return Lease{}, fmt.Errorf("%w: session %s", ErrLeaseHeld, sessionID)
	}
	token := randomToken()
	lease := Lease{
		SessionID: sessionID, Owner: owner, Token: token,
		Fence:     s.fences[sessionID] + 1,
		ExpiresAt: time.Now().Add(ttl),
	}
	if s.fences == nil {
		s.fences = make(map[string]uint64)
	}
	s.fences[sessionID] = lease.Fence
	s.leases[sessionID] = lease
	return lease, nil
}

// RenewLease extends an in-memory lease if the token matches.
func (s *MemoryStore) RenewLease(_ context.Context, lease Lease) (Lease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.leases[lease.SessionID]
	if !ok || current.Token != lease.Token || current.Fence != lease.Fence || !time.Now().Before(current.ExpiresAt) {
		return Lease{}, fmt.Errorf("%w: session %s", ErrLeaseLost, lease.SessionID)
	}
	current.ExpiresAt = time.Now().Add(30 * time.Second)
	s.leases[lease.SessionID] = current
	return current, nil
}

// AppendFenced appends under the in-memory lease, idempotent per event ID:
// retried events return the canonical tail seq without duplication.
func (s *MemoryStore) AppendFenced(ctx context.Context, lease Lease, events []Event) (uint64, error) {
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return 0, err
		}
		if events[i].SessionID != "" && events[i].SessionID != lease.SessionID {
			return 0, fmt.Errorf("session: event %s belongs to %q, not %q", events[i].ID, events[i].SessionID, lease.SessionID)
		}
	}
	if err := s.checkFence(lease); err != nil {
		return 0, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	current := s.leases[lease.SessionID]
	if current.Token != lease.Token || current.Fence != lease.Fence || !time.Now().Before(current.ExpiresAt) {
		return 0, fmt.Errorf("%w: session %s", ErrLeaseLost, lease.SessionID)
	}
	existing := map[string]bool{}
	for _, e := range s.events[lease.SessionID] {
		if e.ID != "" {
			existing[e.ID] = true
		}
	}
	next := s.next[lease.SessionID] + 1
	stored := make([]Event, 0, len(events))
	last := s.next[lease.SessionID]
	for i := range events {
		e := events[i].Clone()
		if e.SessionID == "" {
			e.SessionID = lease.SessionID
		}
		if e.ID != "" && existing[e.ID] {
			continue
		}
		e.Seq = next
		e.Normalize()
		stored = append(stored, e)
		next++
		last = e.Seq
	}
	if len(stored) > 0 {
		s.next[lease.SessionID] = last
		s.events[lease.SessionID] = append(s.events[lease.SessionID], stored...)
	}
	return last, nil
}

func (s *MemoryStore) AppendFencedCommitted(ctx context.Context, lease Lease, events []Event) (CommittedBatch, error) {
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return CommittedBatch{}, err
		}
		if events[i].SessionID != "" && events[i].SessionID != lease.SessionID {
			return CommittedBatch{}, fmt.Errorf("session: event %s belongs to %q, not %q", events[i].ID, events[i].SessionID, lease.SessionID)
		}
	}
	if err := s.checkFence(lease); err != nil {
		return CommittedBatch{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.leases[lease.SessionID]
	if !ok || current.Token != lease.Token || current.Fence != lease.Fence || !time.Now().Before(current.ExpiresAt) {
		return CommittedBatch{}, fmt.Errorf("%w: session %s", ErrLeaseLost, lease.SessionID)
	}
	existing := map[string]Event{}
	for _, event := range s.events[lease.SessionID] {
		if event.ID != "" {
			existing[event.ID] = event
		}
	}
	committed := make([]Event, 0, len(events))
	next := s.next[lease.SessionID] + 1
	for i := range events {
		e := events[i].Clone()
		if e.SessionID == "" {
			e.SessionID = lease.SessionID
		}
		if e.ID != "" {
			if prior, found := existing[e.ID]; found {
				committed = append(committed, prior.Clone())
				continue
			}
		} else {
			e.ID = randomToken()
		}
		e.Seq = next
		e.Normalize()
		committed = append(committed, e.Clone())
		s.events[lease.SessionID] = append(s.events[lease.SessionID], e)
		existing[e.ID] = e
		next++
	}
	if len(committed) > 0 {
		s.next[lease.SessionID] = next - 1
	}
	return CommittedBatch{Events: committed}, nil
}

// ReleaseLease releases the in-memory lease if it belongs to the caller.
func (s *MemoryStore) ReleaseLease(_ context.Context, lease Lease) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if current, ok := s.leases[lease.SessionID]; ok && current.Token == lease.Token {
		delete(s.leases, lease.SessionID)
	}
	return nil
}

// Recover reconciles an interrupted session in memory: closes orphaned turns
// and marks dangling tool calls unknown.
func (s *MemoryStore) Recover(ctx context.Context, lease Lease) (*RecoveryReport, error) {
	report := &RecoveryReport{}
	s.mu.RLock()
	all := s.events[lease.SessionID]
	s.mu.RUnlock()

	openTurns := 0
	openTurnID := ""
	for _, e := range all {
		if e.Type == EventTurnStart {
			openTurns++
			openTurnID = e.TurnID
		} else if e.Type == EventTurnEnd {
			openTurns--
			if openTurns == 0 {
				openTurnID = ""
			}
		}
	}
	var recovery []Event
	recovery = append(recovery, InterruptedDraftEvents(all, lease.SessionID)...)
	if openTurns > 0 {
		report.TurnClosed = true
		recovery = append(recovery, Event{
			ID: "recover:turn-end:" + openTurnID, TurnID: openTurnID,
			Type: EventTurnEnd, SessionID: lease.SessionID,
			Data: mustJSON(map[string]any{"reason": "interrupted"}),
		})
	}
	dangling := danglingCallIDs(all)
	for _, callID := range dangling {
		report.DanglingCalls = append(report.DanglingCalls, callID)
		recovery = append(recovery, Event{
			ID:   "recover:unknown:" + callID,
			Type: EventToolResult, SessionID: lease.SessionID, CallID: callID,
			Data: mustJSON(map[string]any{
				"call_id": callID, "is_error": true, "code": "TOOL_OUTCOME_UNKNOWN",
				"output": map[string]any{},
			}),
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

// SaveCompactionCheckpoint records a checkpoint ledger entry in memory.
func (s *MemoryStore) SaveCompactionCheckpoint(_ context.Context, lease Lease, record CompactionCheckpoint) error {
	if err := s.checkFence(lease); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.checkpoints == nil {
		s.checkpoints = make(map[string][]CompactionCheckpoint)
	}
	s.checkpoints[record.SessionID] = append(s.checkpoints[record.SessionID], record)
	return nil
}

func (s *MemoryStore) checkFence(lease Lease) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	current, ok := s.leases[lease.SessionID]
	if !ok || s.fences[lease.SessionID] != lease.Fence || current.Token != lease.Token || current.Fence != lease.Fence || !time.Now().Before(current.ExpiresAt) {
		return fmt.Errorf("%w: session %s (stale fence)", ErrLeaseLost, lease.SessionID)
	}
	return nil
}

func danglingCallIDs(events []Event) []string {
	resultIDs := make(map[string]bool)
	pendingApproval := make(map[string]bool)
	callIDs := make([]string, 0)
	seen := make(map[string]bool)
	for _, e := range events {
		switch e.Type {
		case EventToolCall:
			var payload struct {
				CallID string `json:"call_id"`
			}
			_ = decodeJSON(e.Data, &payload)
			if payload.CallID != "" && !seen[payload.CallID] {
				seen[payload.CallID] = true
				callIDs = append(callIDs, payload.CallID)
			}
		case EventAssistantMessage:
			var payload struct {
				ToolCalls []ToolCall `json:"tool_calls"`
			}
			_ = decodeJSON(e.Data, &payload)
			for _, call := range payload.ToolCalls {
				if call.CallID != "" && !seen[call.CallID] {
					seen[call.CallID] = true
					callIDs = append(callIDs, call.CallID)
				}
			}
		case EventToolResult:
			var payload struct {
				CallID string `json:"call_id"`
			}
			_ = decodeJSON(e.Data, &payload)
			if payload.CallID != "" {
				resultIDs[payload.CallID] = true
			}
		case EventApprovalRequested:
			if e.CallID != "" {
				pendingApproval[e.CallID] = true
			}
		case EventApprovalResolved:
			var payload ApprovalResolvedPayload
			if decodeJSON(e.Data, &payload) == nil && payload.CallID != "" {
				pendingApproval[payload.CallID] = false
			}
		}
	}
	var dangling []string
	for _, id := range callIDs {
		if !resultIDs[id] && !pendingApproval[id] {
			dangling = append(dangling, id)
		}
	}
	return dangling
}

func decodeJSON(data []byte, out any) error {
	if len(data) == 0 {
		return nil
	}
	return json.Unmarshal(data, out)
}

func randomToken() string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf[:])
}
