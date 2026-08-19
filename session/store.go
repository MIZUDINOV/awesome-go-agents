package session

import (
	"context"
	"sync"
)

// Store persists the append-only event log. The real deployment uses
// PGSessionStore (PostgreSQL); MemoryStore is used in tests and embedded runs.
//
// Contract: Append atomically assigns strictly increasing per-session sequence
// numbers to the batch (contiguous, no gaps within one batch) and returns the
// seq of the last event. The input slice is never mutated; callers read back
// per-event seqs from the returned last seq assuming contiguity.
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
// use. It does not survive restarts. Events are deep-copied into internal
// storage so later caller mutations never leak into committed history
// (H-SESSION-004).
type MemoryStore struct {
	mu     sync.RWMutex
	next   map[string]uint64
	events map[string][]Event

	// Fenced single-writer state.
	leases      map[string]Lease
	fences      map[string]uint64
	checkpoints map[string][]CompactionCheckpoint
}

// NewMemoryStore returns an empty MemoryStore.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		next:   make(map[string]uint64),
		events: make(map[string][]Event),
		leases: make(map[string]Lease),
		fences: make(map[string]uint64),
	}
}

func (s *MemoryStore) Append(_ context.Context, sessionID string, events []Event) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.events == nil {
		s.next = make(map[string]uint64)
		s.events = make(map[string][]Event)
	}
	next := s.next[sessionID] + 1
	stored := make([]Event, 0, len(events))
	for i := range events {
		e := events[i].Clone()
		e.Seq = next
		e.FormatVersion = e.NormalizedFormatVersion()
		stored = append(stored, e)
		next++
	}
	s.next[sessionID] = next - 1
	s.events[sessionID] = append(s.events[sessionID], stored...)
	return next - 1, nil
}

func (s *MemoryStore) Load(_ context.Context, sessionID string, afterSeq uint64, limit int) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	all := s.events[sessionID]
	var out []Event
	for _, e := range all {
		if e.Seq > afterSeq {
			out = append(out, e.Clone())
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
			out[i] = e.Clone()
		}
		return out, nil
	}
	start := len(all) - limit
	out := make([]Event, limit)
	for i := range out {
		out[i] = all[start+i].Clone()
	}
	return out, nil
}

func (s *MemoryStore) Sequence(_ context.Context, sessionID string) (uint64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.next[sessionID], nil
}

// ensure Store is satisfied
var _ Store = (*MemoryStore)(nil)
