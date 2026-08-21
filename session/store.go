package session

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CommittedBatch is the canonical result of one atomic append. Every event in
// the batch has its assigned sequence, stable ID and commit timestamp filled
// by the store. Consumers must publish this copy, never a pre-commit input.
type CommittedBatch struct {
	Events []Event
}

// BatchStore is an optional additive capability for stores that can return
// committed events directly. Store remains source-compatible with older
// adapters while new loops prefer this stronger contract.
type BatchStore interface {
	Store
	AppendCommitted(ctx context.Context, sessionID string, events []Event) (CommittedBatch, error)
}

// Store persists the append-only event log. Hosts provide the durable backend;
// MemoryStore is used in tests and embedded runs.
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
	if len(events) == 0 {
		return s.Sequence(context.Background(), sessionID)
	}
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return 0, err
		}
		if events[i].SessionID != "" && events[i].SessionID != sessionID {
			return 0, fmt.Errorf("session: event %s belongs to %q, not %q", events[i].ID, events[i].SessionID, sessionID)
		}
	}
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
		if e.SessionID == "" {
			e.SessionID = sessionID
		}
		e.Seq = next
		e.Normalize()
		stored = append(stored, e)
		next++
	}
	s.next[sessionID] = next - 1
	s.events[sessionID] = append(s.events[sessionID], stored...)
	return next - 1, nil
}

func (s *MemoryStore) AppendCommitted(_ context.Context, sessionID string, events []Event) (CommittedBatch, error) {
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return CommittedBatch{}, err
		}
		if events[i].SessionID != "" && events[i].SessionID != sessionID {
			return CommittedBatch{}, fmt.Errorf("session: event %s belongs to %q, not %q", events[i].ID, events[i].SessionID, sessionID)
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.events == nil {
		s.next = make(map[string]uint64)
		s.events = make(map[string][]Event)
	}
	committed := make([]Event, 0, len(events))
	existing := make(map[string]Event)
	for _, prior := range s.events[sessionID] {
		if prior.ID != "" {
			existing[prior.ID] = prior
		}
	}
	next := s.next[sessionID] + 1
	for i := range events {
		e := events[i].Clone()
		if e.SessionID == "" {
			e.SessionID = sessionID
		}
		if e.ID == "" {
			e.ID = randomToken()
		}
		if prior, found := existing[e.ID]; found {
			committed = append(committed, prior.Clone())
			continue
		}
		e.Seq = next
		e.Normalize()
		if e.Timestamp.IsZero() {
			e.Timestamp = time.Now().UTC()
		}
		committed = append(committed, e.Clone())
		s.events[sessionID] = append(s.events[sessionID], e)
		existing[e.ID] = e
		s.next[sessionID] = next
		next++
	}
	return CommittedBatch{Events: committed}, nil
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
