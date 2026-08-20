package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/session"
)

func appendDurable(ctx context.Context, store Store, hub *EventHub, sessionID string, event session.Event) (session.Event, error) {
	if event.ID == "" {
		return session.Event{}, fmt.Errorf("agent: durable event id is required")
	}
	if batchStore, ok := store.(session.BatchStore); ok {
		batch, err := batchStore.AppendCommitted(ctx, sessionID, []session.Event{event})
		if err != nil {
			return session.Event{}, err
		}
		if len(batch.Events) == 0 {
			return session.Event{}, fmt.Errorf("agent: committed event %s not returned", event.ID)
		}
		committed := batch.Events[0].Clone()
		if hub != nil {
			hub.Publish(committed)
		}
		return committed, nil
	}
	last, err := store.Append(ctx, sessionID, []session.Event{event})
	if err != nil {
		return session.Event{}, err
	}
	committed := event.Clone()
	committed.SessionID = sessionID
	committed.Seq = last
	committed.Normalize()
	if hub != nil {
		hub.Publish(committed)
	}
	return committed, nil
}

// ErrSubscriberLagged means a live subscriber did not consume its bounded
// notification buffer. The durable session log remains authoritative; callers
// must reconnect from their last acknowledged sequence.
var ErrSubscriberLagged = errors.New("agent: event subscriber lagged")

// Notification is a live-only lifecycle/control event. Durable session
// events use Subscription and can always be replayed by sequence.
type Notification struct {
	Type      string
	SessionID string
	RunID     string
	CallID    string
	Data      json.RawMessage
}

const (
	NotificationAgentCreated          = "agent/created"
	NotificationAgentStatus           = "agent/status"
	NotificationAgentRunFailed        = "agent/run_failed"
	NotificationToolApprovalRequested = "tool/approval_requested"
)

type NotificationSubscription struct {
	hub  *EventHub
	ch   chan Notification
	done chan struct{}
}

func (s *NotificationSubscription) Next(ctx context.Context) (Notification, error) {
	select {
	case notification, ok := <-s.ch:
		if !ok {
			return Notification{}, context.Canceled
		}
		return notification, nil
	case <-s.done:
		return Notification{}, context.Canceled
	case <-ctx.Done():
		return Notification{}, ctx.Err()
	}
}

func (s *NotificationSubscription) Close() {
	s.hub.removeNotification(s)
}

// EventFilter selects durable events for a subscription. Returning true keeps
// the event; nil keeps every event.
type EventFilter func(session.Event) bool

// EventHub is the process-local live notification fan-out. It deliberately
// never blocks the agent loop: durable append happens first, then a slow
// observer is disconnected and can replay from its cursor.
type EventHub struct {
	mu               sync.Mutex
	nextID           uint64
	limit            int
	subs             map[uint64]*eventSubscription
	notificationSubs map[*NotificationSubscription]struct{}
}

// NewEventHub creates a bounded live event hub. A non-positive buffer uses 64.
func NewEventHub(buffer int) *EventHub {
	if buffer <= 0 {
		buffer = 64
	}
	return &EventHub{limit: buffer, subs: make(map[uint64]*eventSubscription), notificationSubs: make(map[*NotificationSubscription]struct{})}
}

func (h *EventHub) SubscribeNotifications(ctx context.Context) *NotificationSubscription {
	h.mu.Lock()
	ch := make(chan Notification, h.limit)
	sub := &NotificationSubscription{hub: h, ch: ch, done: make(chan struct{})}
	h.notificationSubs[sub] = struct{}{}
	h.mu.Unlock()
	if done := ctx.Done(); done != nil {
		go func() {
			select {
			case <-done:
				sub.Close()
			case <-sub.done:
			}
		}()
	}
	return sub
}

func (h *EventHub) removeNotification(sub *NotificationSubscription) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.notificationSubs[sub]; ok {
		delete(h.notificationSubs, sub)
		close(sub.ch)
		close(sub.done)
	}
}

func (h *EventHub) PublishNotification(notification Notification) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for sub := range h.notificationSubs {
		select {
		case sub.ch <- Notification{Type: notification.Type, SessionID: notification.SessionID, RunID: notification.RunID, CallID: notification.CallID, Data: append(json.RawMessage(nil), notification.Data...)}:
		default:
			delete(h.notificationSubs, sub)
			close(sub.ch)
			close(sub.done)
		}
	}
}

type eventSubscription struct {
	id     uint64
	hub    *EventHub
	filter EventFilter
	ch     chan session.Event
	done   chan struct{}
	lagged bool
	// nextSeq is the next durable sequence expected by this subscriber. Live
	// commits can arrive out of order when independent library commands append
	// concurrently; pending keeps them bounded until the missing sequence is
	// published or replay establishes the cursor.
	nextSeq uint64
	pending map[uint64]session.Event
}

// Subscription delivers committed durable events in sequence order. The
// channel is receive-only so only the hub owns closure.
type Subscription struct {
	sub     *eventSubscription
	backlog []session.Event
	index   int
	// lastSeq is the caller's acknowledged cursor. deliveredSeq is kept
	// separately so reconnecting before Acknowledge redelivers the event.
	lastSeq      uint64
	deliveredSeq uint64
	mu           sync.Mutex
}

// Subscribe starts a cursor subscription. Replay is loaded before returning,
// while live events are buffered from the moment the subscription is created,
// so the replay/live handoff has no gap.
func (h *EventHub) subscribe(ctx context.Context, after uint64, filter EventFilter) *Subscription {
	h.mu.Lock()
	h.nextID++
	sub := &eventSubscription{id: h.nextID, hub: h, filter: filter, ch: make(chan session.Event, h.limit), done: make(chan struct{}), nextSeq: after + 1, pending: make(map[uint64]session.Event)}
	h.subs[sub.id] = sub
	h.mu.Unlock()
	result := &Subscription{sub: sub, lastSeq: after, deliveredSeq: after}
	go func() {
		select {
		case <-ctx.Done():
			sub.close()
		case <-sub.done:
		}
	}()
	return result
}

// SubscribeCursor registers the live side first, then loads replay through
// loader. Register-before-load closes the race between the cursor snapshot and
// a concurrently committed event.
func (h *EventHub) SubscribeCursor(ctx context.Context, after uint64, filter EventFilter, loader func() ([]session.Event, error)) (*Subscription, error) {
	result := h.subscribe(ctx, after, filter)
	var replay []session.Event
	if loader != nil {
		var err error
		replay, err = loader()
		if err != nil {
			result.Close()
			return nil, err
		}
	}
	filtered := make([]session.Event, 0, len(replay))
	maxReplay := after
	for _, event := range replay {
		if err := event.Validate(); err != nil {
			result.Close()
			return nil, err
		}
		if event.Seq <= after {
			continue
		}
		if event.Seq > maxReplay {
			maxReplay = event.Seq
		}
		if filter != nil && !filter(event) {
			continue
		}
		filtered = append(filtered, event.Clone())
	}
	sort.SliceStable(filtered, func(i, j int) bool { return filtered[i].Seq < filtered[j].Seq })
	result.backlog = filtered
	// Replay is delivered from the subscription backlog, not through the live
	// channel. Advance the live cursor under the hub lock and flush any events
	// that raced with the store read.
	h.mu.Lock()
	_, active := h.subs[result.sub.id]
	if active {
		if maxReplay+1 > result.sub.nextSeq {
			result.sub.nextSeq = maxReplay + 1
		}
		for seq := range result.sub.pending {
			if seq < result.sub.nextSeq {
				delete(result.sub.pending, seq)
			}
		}
		h.flushPendingLocked(result.sub)
	}
	h.mu.Unlock()
	return result, nil
}

// Next returns the next replayed or live event. It de-duplicates an event that
// was concurrently present in both the replay batch and live buffer.
func (s *Subscription) Next(ctx context.Context) (session.Event, error) {
	for {
		s.mu.Lock()
		if s.index >= len(s.backlog) {
			s.mu.Unlock()
			break
		}
		event := s.backlog[s.index]
		s.index++
		if event.Seq <= s.deliveredSeq {
			s.mu.Unlock()
			continue
		}
		s.deliveredSeq = event.Seq
		s.mu.Unlock()
		return event.Clone(), nil
	}
	for {
		select {
		case event, ok := <-s.sub.ch:
			if !ok {
				s.sub.hub.mu.Lock()
				lagged := s.sub.lagged
				s.sub.hub.mu.Unlock()
				if lagged {
					return session.Event{}, ErrSubscriberLagged
				}
				return session.Event{}, context.Canceled
			}
			s.mu.Lock()
			if event.Seq <= s.deliveredSeq {
				s.mu.Unlock()
				continue
			}
			s.deliveredSeq = event.Seq
			s.mu.Unlock()
			return event.Clone(), nil
		case <-ctx.Done():
			return session.Event{}, ctx.Err()
		}
	}
}

// Close removes the subscription and releases its channel.
func (s *Subscription) Close() { s.sub.close() }

// Acknowledge advances the reconnect cursor without requiring callers to
// consume an event again. Cursors never move backwards.

func (s *Subscription) Acknowledge(seq uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if seq > s.deliveredSeq {
		seq = s.deliveredSeq
	}
	if seq > s.lastSeq {
		s.lastSeq = seq
	}
}
func (s *Subscription) Cursor() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastSeq
}

func (s *eventSubscription) close() {
	s.hub.mu.Lock()
	defer s.hub.mu.Unlock()
	s.hub.terminateLocked(s, false)
}

func (h *EventHub) terminateLocked(sub *eventSubscription, lagged bool) {
	if _, active := h.subs[sub.id]; !active {
		return
	}
	delete(h.subs, sub.id)
	sub.lagged = lagged
	close(sub.done)
	close(sub.ch)
}

// Publish sends one already-committed event to observers. A full observer is
// closed with ErrSubscriberLagged semantics; the producer is never blocked.
func (h *EventHub) Publish(event session.Event) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, sub := range h.subs {
		candidate := event.Clone()
		if candidate.Seq == 0 {
			if sub.filter != nil && !sub.filter(candidate) {
				continue
			}
			if !h.enqueueLocked(sub, candidate) {
				h.terminateLocked(sub, true)
			}
			continue
		}
		if candidate.Seq < sub.nextSeq {
			continue
		}
		if candidate.Seq > sub.nextSeq {
			sub.pending[candidate.Seq] = candidate
			if len(sub.pending) > h.limit {
				h.terminateLocked(sub, true)
			}
			continue
		}
		if sub.filter != nil && !sub.filter(candidate) {
			sub.nextSeq++
			h.flushPendingLocked(sub)
			continue
		}
		if !h.enqueueLocked(sub, candidate) {
			h.terminateLocked(sub, true)
			continue
		}
		sub.nextSeq++
		h.flushPendingLocked(sub)
	}
}

func (h *EventHub) enqueueLocked(sub *eventSubscription, event session.Event) bool {
	select {
	case sub.ch <- event:
		return true
	default:
		return false
	}
}

func (h *EventHub) flushPendingLocked(sub *eventSubscription) {
	for {
		event, ok := sub.pending[sub.nextSeq]
		if !ok {
			return
		}
		if sub.filter != nil && !sub.filter(event) {
			delete(sub.pending, sub.nextSeq)
			sub.nextSeq++
			continue
		}
		if !h.enqueueLocked(sub, event) {
			h.terminateLocked(sub, true)
			return
		}
		delete(sub.pending, sub.nextSeq)
		sub.nextSeq++
	}
}
