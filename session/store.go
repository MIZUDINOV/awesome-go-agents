package session

import (
	"context"
	"encoding/json"
	"sync"
)

// Store persists the append-only event log. The real deployment uses
// PGSessionStore (PostgreSQL); MemoryStore is used in tests and embedded runs.
type Store interface {
	// Append persists events and returns the seq assigned to the last one.
	Append(ctx context.Context, sessionID string, events []Event) (uint64, error)
	// Load returns events with seq > afterSeq, up to limit (0 = all).
	Load(ctx context.Context, sessionID string, afterSeq uint64, limit int) ([]Event, error)
	// Tail returns the most recent events (newest last), up to limit.
	Tail(ctx context.Context, sessionID string, limit int) ([]Event, error)
	// Sequence returns the highest assigned seq for the session (0 if none).
	Sequence(ctx context.Context, sessionID string) (uint64, error)
}

// MemoryStore is an in-memory, concurrency-safe Store for tests and embedded
// use. It does not survive restarts.
type MemoryStore struct {
	mu       sync.RWMutex
	next     map[string]uint64
	events   map[string][]Event
}

// NewMemoryStore returns an empty MemoryStore.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		next:   make(map[string]uint64),
		events: make(map[string][]Event),
	}
}

func (s *MemoryStore) Append(_ context.Context, sessionID string, events []Event) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	next := s.next[sessionID] + 1
	for i := range events {
		events[i].Seq = next
		next++
	}
	s.next[sessionID] = next - 1
	s.events[sessionID] = append(s.events[sessionID], events...)
	return next - 1, nil
}

func (s *MemoryStore) Load(_ context.Context, sessionID string, afterSeq uint64, limit int) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	all := s.events[sessionID]
	var out []Event
	for _, e := range all {
		if e.Seq > afterSeq {
			out = append(out, cloneEvent(e))
			if limit > 0 && len(out) >= limit {
				break
			}
		}
	}
	return out, nil
}

func (s *MemoryStore) Tail(_ context.Context, sessionID string, limit int) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	all := s.events[sessionID]
	if limit <= 0 || len(all) <= limit {
		out := make([]Event, len(all))
		for i, e := range all {
			out[i] = cloneEvent(e)
		}
		return out, nil
	}
	start := len(all) - limit
	out := make([]Event, limit)
	for i := range out {
		out[i] = cloneEvent(all[start+i])
	}
	return out, nil
}

func (s *MemoryStore) Sequence(_ context.Context, sessionID string) (uint64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.next[sessionID], nil
}

func cloneEvent(e Event) Event {
	e.Data = append(json.RawMessage(nil), e.Data...)
	if e.SourceSeqs != nil {
		e.SourceSeqs = append([]uint64(nil), e.SourceSeqs...)
	}
	return e
}

// ensure Store is satisfied
var _ Store = (*MemoryStore)(nil)
