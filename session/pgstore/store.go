// Package pgstore implements session.Store on PostgreSQL. It owns a small,
// self-contained schema (agentkit_sessions/agentkit_events) so the library has
// no coupling to any host application's tables. Host apps may alternatively
// map session.Store onto their own event log (e.g. wzhooh's chat_events).
package pgstore

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MIZUDINOV/awesome-go-agents/session"
)

// Store is a PostgreSQL-backed session.Store.
type Store struct {
	db     *pgxpool.Pool
	events string // table name, default "agentkit_events"
}

// New returns a Store bound to pool. sessionID is stored as text.
func New(db *pgxpool.Pool) *Store {
	return &Store{db: db, events: "agentkit_events"}
}

// WithEventsTable overrides the default table name (for host-app remapping).
func (s *Store) WithEventsTable(name string) *Store {
	s.events = name
	return s
}

func (s *Store) Append(ctx context.Context, sessionID string, events []session.Event) (uint64, error) {
	if len(events) == 0 {
		seq, err := s.Sequence(ctx, sessionID)
		return seq, err
	}
	tx, err := s.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return 0, fmt.Errorf("pgstore: begin append: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var base uint64
	err = tx.QueryRow(ctx, fmt.Sprintf(`SELECT COALESCE(MAX(seq),0) FROM %s WHERE session_id=$1`, s.events), sessionID).Scan(&base)
	if err != nil {
		return 0, fmt.Errorf("pgstore: read sequence: %w", err)
	}
	next := base
	for _, event := range events {
		next++
		data := event.Data
		if len(data) == 0 {
			data = json.RawMessage("{}")
		}
		_, err := tx.Exec(ctx, fmt.Sprintf(`INSERT INTO %s(session_id,seq,type,timestamp,data,surface,source_seqs) VALUES($1,$2,$3,$4,$5,$6,$7)`, s.events),
			sessionID, next, string(event.Type), event.Timestamp, data, event.Type.Surface(), event.SourceSeqs)
		if err != nil {
			return 0, fmt.Errorf("pgstore: insert event: %w", err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("pgstore: commit append: %w", err)
	}
	return next, nil
}

func (s *Store) Load(ctx context.Context, sessionID string, afterSeq uint64, limit int) ([]session.Event, error) {
	query := fmt.Sprintf(`SELECT seq,type,timestamp,data,source_seqs FROM %s WHERE session_id=$1 AND seq>$2 ORDER BY seq`, s.events)
	args := []any{sessionID, afterSeq}
	if limit > 0 {
		query += ` LIMIT $3`
		args = append(args, limit)
	}
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("pgstore: load events: %w", err)
	}
	defer rows.Close()
	events, err := scanEvents(rows)
	if err != nil {
		return nil, err
	}
	return events, nil
}

func (s *Store) Tail(ctx context.Context, sessionID string, limit int) ([]session.Event, error) {
	if limit <= 0 {
		return s.Load(ctx, sessionID, 0, 0)
	}
	rows, err := s.db.Query(ctx, fmt.Sprintf(`SELECT seq,type,timestamp,data,source_seqs FROM (
    SELECT seq,type,timestamp,data,source_seqs FROM %s WHERE session_id=$1 ORDER BY seq DESC LIMIT $2
) t ORDER BY seq`, s.events), sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("pgstore: load tail: %w", err)
	}
	defer rows.Close()
	return scanEvents(rows)
}

func (s *Store) Sequence(ctx context.Context, sessionID string) (uint64, error) {
	var seq uint64
	err := s.db.QueryRow(ctx, fmt.Sprintf(`SELECT COALESCE(MAX(seq),0) FROM %s WHERE session_id=$1`, s.events), sessionID).Scan(&seq)
	if err != nil {
		return 0, fmt.Errorf("pgstore: sequence: %w", err)
	}
	return seq, nil
}

func scanEvents(rows pgx.Rows) ([]session.Event, error) {
	var events []session.Event
	for rows.Next() {
		var e session.Event
		var eventType string
		var ts time.Time
		var data []byte
		var sourceSeqs []uint64
		if err := rows.Scan(&e.Seq, &eventType, &ts, &data, &sourceSeqs); err != nil {
			return nil, fmt.Errorf("pgstore: scan event: %w", err)
		}
		e.Type = session.EventType(eventType)
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

// Migration is the SQL to create the self-contained schema. Host apps mount it
// into their migrate chain, or remap Store onto their own table via
// WithEventsTable.
const Migration = `
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
CREATE INDEX IF NOT EXISTS agentkit_events_session_seq
    ON agentkit_events (session_id, seq);
`
