package session

import (
	"context"
	"fmt"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Session is the durable handle over an append-only event log. Append appends
// events; DeriveMessages projects them into the model-facing history.
//
// The active session/surface is held in memory by the worker between steps;
// the store is the durable recovery source (H-DB-001). DeriveMessages serves
// from the in-memory tail after the first hydration instead of re-reading the
// whole log on every model step (H-ANTI-011).
type Session struct {
	ID    string
	store Store

	// surface uses the zero spec unless configured.
	mu      sync.RWMutex
	surface *Surface

	// cached is the in-memory tail of the log (ascending seq), initialized on
	// first read from the store (hydration) and extended on Append.
	cached    []Event
	cachedSeq uint64
	hydrated  bool
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

// Append validates and persists one event, assigning it the next seq.
func (s *Session) Append(ctx context.Context, event Event) (uint64, error) {
	return s.AppendAll(ctx, []Event{event})
}

// AppendAll validates and persists multiple events atomically (best effort per
// store). Invalid events abort the whole batch before anything is written
// (H-SESSION-004: payloads are validated and copied before commit).
func (s *Session) AppendAll(ctx context.Context, events []Event) (uint64, error) {
	committed, err := s.AppendCommitted(ctx, events)
	if err != nil {
		return 0, err
	}
	if len(committed.Events) == 0 {
		return s.Sequence(ctx)
	}
	return committed.Events[len(committed.Events)-1].Seq, nil
}

// AppendCommitted returns the canonical copies assigned by a batch-capable
// store. Older adapters are supported by the legacy contiguous append/load
// fallback; new adapters should implement BatchStore atomically.
func (s *Session) AppendCommitted(ctx context.Context, events []Event) (CommittedBatch, error) {
	batchInput := make([]Event, len(events))
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return CommittedBatch{}, err
		}
		batchInput[i] = events[i].Clone()
		batchInput[i].Normalize()
	}
	if batchStore, ok := s.store.(BatchStore); ok {
		batch, err := batchStore.AppendCommitted(ctx, s.ID, batchInput)
		if err != nil {
			return CommittedBatch{}, err
		}
		canonical := make([]Event, len(batch.Events))
		for i, event := range batch.Events {
			canonical[i] = event.Clone()
		}
		s.mu.Lock()
		if s.hydrated {
			for _, event := range canonical {
				if event.Seq <= s.cachedSeq {
					continue
				}
				s.cached = append(s.cached, event.Clone())
				s.cachedSeq = event.Seq
			}
		}
		s.mu.Unlock()
		return CommittedBatch{Events: canonical}, nil
	}
	last, err := s.store.Append(ctx, s.ID, batchInput)
	if err != nil {
		return CommittedBatch{}, err
	}
	after := uint64(0)
	if last > uint64(len(events)) {
		after = last - uint64(len(events))
	}
	loaded, err := s.store.Load(ctx, s.ID, after, 0)
	if err != nil {
		return CommittedBatch{}, err
	}
	s.mu.Lock()
	if s.hydrated {
		for _, event := range loaded {
			if event.Seq > s.cachedSeq {
				s.cached = append(s.cached, event.Clone())
				s.cachedSeq = event.Seq
			}
		}
	}
	s.mu.Unlock()
	return CommittedBatch{Events: loaded}, nil
}

// CompactionSummary durably records a compaction transaction
// (compaction/start + compaction/summary + compaction/end) as one atomic
// batch. generation increases monotonically; shadowedSeqs is the exact
// shadowed list. The raw history is never deleted (H-COMPACT-001);
// compaction only replaces the model-visible projection.
func (s *Session) CompactionSummary(ctx context.Context, generation uint64, transactionID string, throughSeq uint64, shadowedSeqs []uint64, summary, fingerprint string) (uint64, error) {
	if transactionID == "" {
		return 0, fmt.Errorf("session: compaction requires a transaction id")
	}
	if len(shadowedSeqs) == 0 && throughSeq == 0 {
		return 0, fmt.Errorf("session: compaction covers no events")
	}
	common := Event{SessionID: s.ID}
	start := common
	start.Type = EventCompactionStart
	start.Data = CompactionStartPayload(generation, transactionID, shadowedSeqs)

	sum := common
	sum.Type = EventCompactionSummary
	sum.Data = CompactionSummaryPayload(generation, transactionID, throughSeq, shadowedSeqs, summary, fingerprint)
	sum.SourceSeqs = append([]uint64(nil), shadowedSeqs...)

	end := common
	end.Type = EventCompactionEnd
	end.Data = CompactionEndPayload(generation, transactionID)

	return s.AppendAll(ctx, []Event{start, sum, end})
}

// Load returns all events younger than afterSeq (0 = from the beginning).
// This is an explicit fresh read from the store (recovery/resume path); normal
// DeriveMessages serves from the in-memory tail.
func (s *Session) Load(ctx context.Context, afterSeq uint64) ([]Event, error) {
	events, err := s.store.Load(ctx, s.ID, afterSeq, 0)
	if err != nil {
		return nil, err
	}
	cloned := make([]Event, len(events))
	for i, event := range events {
		if err := event.Validate(); err != nil {
			return nil, err
		}
		cloned[i] = event.Clone()
	}
	return cloned, nil
}

// Events returns all events from the beginning (fresh from the store).
func (s *Session) Events(ctx context.Context) ([]Event, error) {
	return s.Load(ctx, 0)
}

// Sequence returns the highest assigned seq.
func (s *Session) Sequence(ctx context.Context) (uint64, error) {
	return s.store.Sequence(ctx, s.ID)
}

// DeriveMessages projects the durable events into the model history from the
// in-memory tail (hydrated once from the store).
func (s *Session) DeriveMessages(ctx context.Context) ([]*llm.Message, error) {
	msgs, _, err := s.Project(ctx)
	return msgs, err
}

// Project derives messages plus projection metadata (generation, shadowed
// seqs, summary, fingerprint) from the in-memory tail.
func (s *Session) Project(ctx context.Context) ([]*llm.Message, *Projection, error) {
	if err := s.hydrate(ctx); err != nil {
		return nil, nil, err
	}
	s.mu.RLock()
	surface := s.surface
	events := make([]Event, len(s.cached))
	for i, e := range s.cached {
		events[i] = e.Clone()
	}
	s.mu.RUnlock()
	return surface.Project(events)
}

// Refresh re-hydrates the in-memory tail from the store (used after resume or
// an external recovery pass).
func (s *Session) Refresh(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	events, err := s.store.Load(ctx, s.ID, 0, 0)
	if err != nil {
		return fmt.Errorf("session: refresh tail: %w", err)
	}
	cached := make([]Event, len(events))
	for i, e := range events {
		if err := e.Validate(); err != nil {
			return fmt.Errorf("session: refresh event %d: %w", e.Seq, err)
		}
		cached[i] = e.Clone()
	}
	s.cached = cached
	if len(cached) > 0 {
		s.cachedSeq = cached[len(cached)-1].Seq
	} else {
		s.cachedSeq = 0
	}
	s.hydrated = true
	return nil
}

func (s *Session) hydrate(ctx context.Context) error {
	s.mu.RLock()
	if s.hydrated {
		s.mu.RUnlock()
		return nil
	}
	s.mu.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hydrated {
		return nil
	}
	events, err := s.store.Load(ctx, s.ID, 0, 0)
	if err != nil {
		return fmt.Errorf("session: hydrate tail: %w", err)
	}
	// Validate the whole log before serving it: unknown mandatory event types
	// and unsupported versions fail the replay explicitly (H-SESSION-007).
	cached := make([]Event, 0, len(events))
	for _, e := range events {
		if err := ValidateType(e.Type, e.FormatVersion); err != nil {
			return err
		}
		if err := e.Validate(); err != nil {
			return err
		}
		cached = append(cached, e.Clone())
	}
	s.cached = cached
	if len(cached) > 0 {
		s.cachedSeq = cached[len(cached)-1].Seq
	}
	s.hydrated = true
	return nil
}
