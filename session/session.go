package session

import (
	"context"
	"fmt"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Session is the durable handle over an append-only event log. Append appends
// events; DeriveMessages projects them into the model-facing history.
type Session struct {
	ID    string
	store Store

	// surface uses the zero spec unless configured.
	mu      sync.RWMutex
	surface *Surface
}

// NewSession returns a Session backed by store.
func NewSession(id string, store Store) *Session {
	return &Session{ID: id, store: store, surface: NewSurface(SurfaceSpec{})}
}

// WithSurface configures the projection spec used by DeriveMessages. Immutable
// after the first call that matters; call before running the loop.
func (s *Session) WithSurface(spec SurfaceSpec) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.surface = NewSurface(spec)
	return s
}

// Append persists one event, assigning it the next seq.
func (s *Session) Append(ctx context.Context, event Event) (uint64, error) {
	if event.Type == "" {
		return 0, fmt.Errorf("session: append requires an event type")
	}
	return s.store.Append(ctx, s.ID, []Event{event})
}

// AppendAll persists multiple events atomically (best effort per store).
func (s *Session) AppendAll(ctx context.Context, events []Event) (uint64, error) {
	return s.store.Append(ctx, s.ID, events)
}

// Load returns all events younger than afterSeq (0 = from the beginning).
func (s *Session) Load(ctx context.Context, afterSeq uint64) ([]Event, error) {
	return s.store.Load(ctx, s.ID, afterSeq, 0)
}

// Events returns all events from the beginning.
func (s *Session) Events(ctx context.Context) ([]Event, error) {
	return s.store.Load(ctx, s.ID, 0, 0)
}

// Sequence returns the highest assigned seq.
func (s *Session) Sequence(ctx context.Context) (uint64, error) {
	return s.store.Sequence(ctx, s.ID)
}

// DeriveMessages projects the durable events into the model history.
func (s *Session) DeriveMessages(ctx context.Context) ([]*llm.Message, error) {
	s.mu.RLock()
	surface := s.surface
	s.mu.RUnlock()
	events, err := s.store.Load(ctx, s.ID, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("session: load events: %w", err)
	}
	return surface.DeriveMessages(events)
}
