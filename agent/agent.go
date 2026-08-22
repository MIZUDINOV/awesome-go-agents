package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

// AgentStatus is the externally observable lifecycle of a stateful agent.
type AgentStatus string

const (
	StatusIdle     AgentStatus = "idle"
	StatusRunning  AgentStatus = "running"
	StatusDisposed AgentStatus = "disposed"
)

var (
	ErrAgentDisposed = errors.New("agent: agent is disposed")
	ErrAgentBusy     = errors.New("agent: operation is already running")
)

// CancelOptions controls whether pending inbox work survives cancellation.
type CancelOptions struct{ KeepInbox bool }

// Agent is the public stateful handle. Implementations own a session and
// serialize turns; callers interact through commands and subscriptions rather
// than reaching into the concrete loop.
type Agent interface {
	ID() string
	Session() SessionView
	ToolScope() tools.Runtime
	ToolCatalog() tools.Catalog
	ScopedContext() tools.ToolRunContext
	Status() AgentStatus
	FollowUp(ctx context.Context, text string) error
	Send(ctx context.Context, text string) error
	Steer(ctx context.Context, text string) error
	Inject(ctx context.Context, text string) error
	Submit(ctx context.Context, input Input) (*Result, error)
	Cancel(ctx context.Context, opts CancelOptions) error
	WhenIdle(ctx context.Context) error
	DecideApproval(ctx context.Context, callID string, approved bool) error
	Approve(ctx context.Context, callID string) error
	Reject(ctx context.Context, callID string) error
	ResumeTool(ctx context.Context, request ToolResume) error
	Subscribe(ctx context.Context, after uint64, filter EventFilter) (*Subscription, error)
	SubscribeNotifications(ctx context.Context) *NotificationSubscription
	Run(ctx context.Context, input string) (*Result, error)
	Dispose(ctx context.Context) error
	Close(ctx context.Context) error
}

// SessionView exposes the durable session projection without exposing the
// lease-less append methods owned by Loop.
type SessionView struct{ session *session.Session }

func (v SessionView) ID() string {
	if v.session == nil {
		return ""
	}
	return v.session.ID
}

func (v SessionView) Events(ctx context.Context) ([]session.Event, error) {
	if v.session == nil {
		return nil, ErrAgentDisposed
	}
	return v.session.Events(ctx)
}

func (v SessionView) Load(ctx context.Context, afterSeq uint64) ([]session.Event, error) {
	if v.session == nil {
		return nil, ErrAgentDisposed
	}
	return v.session.Load(ctx, afterSeq)
}

func (v SessionView) Sequence(ctx context.Context) (uint64, error) {
	if v.session == nil {
		return 0, ErrAgentDisposed
	}
	return v.session.Sequence(ctx)
}

func (v SessionView) DeriveMessages(ctx context.Context) ([]*llm.Message, error) {
	if v.session == nil {
		return nil, ErrAgentDisposed
	}
	return v.session.DeriveMessages(ctx)
}

func (v SessionView) Project(ctx context.Context) ([]*llm.Message, *session.Projection, error) {
	if v.session == nil {
		return nil, nil, ErrAgentDisposed
	}
	return v.session.Project(ctx)
}

type inboxKind uint8

const (
	inboxFollowUp inboxKind = iota + 1
	inboxSteer
	inboxInject
)

type inboxItem struct {
	id   string
	kind inboxKind
	text string
}

// Input is the host-safe durable input boundary. It preserves the caller's
// idempotency identity and distinguishes user, steering, and injected context.
type Input struct {
	ID   string
	Type session.EventType
	Text string
}

// Handle is the default in-process Agent implementation. The durable session
// log remains owned by Loop/Store; this handle owns only live command routing
// and worker lifecycle.
type Handle struct {
	loop    *Loop
	session *session.Session

	mu          sync.Mutex
	status      AgentStatus
	inbox       []inboxItem
	wake        chan struct{}
	changed     chan struct{}
	stop        chan struct{}
	done        chan struct{}
	runCancel   context.CancelFunc
	approval    *approvalBroker
	disposeOnce sync.Once
}

var _ Agent = (*Handle)(nil)

// NewAgent creates a stateful handle and starts its inbox worker. A nil loop is
// rejected because the handle cannot provide a useful zero-value runtime.
func NewAgent(loop *Loop) (*Handle, error) {
	if loop == nil || loop.Store == nil || loop.Tools == nil || loop.Chat == nil {
		return nil, fmt.Errorf("agent: loop dependencies are required")
	}
	loop.mu.Lock()
	if loop.agentAttached {
		loop.mu.Unlock()
		return nil, fmt.Errorf("agent: loop already has an attached agent")
	}
	if loop.Config.EventHub == nil {
		loop.Config.EventHub = NewEventHub(64)
	}
	scopedTools := loop.Tools
	if root, ok := loop.Tools.(*tools.Registry); ok {
		scopedTools = root.NewScope()
	}
	loop.agentAttached = true
	loop.mu.Unlock()
	releaseAttachment := func() {
		loop.mu.Lock()
		loop.agentAttached = false
		loop.mu.Unlock()
	}
	h := &Handle{loop: loop, session: session.NewSession(loop.SessionID, loop.Store), status: StatusIdle, wake: make(chan struct{}, 1), changed: make(chan struct{}, 1), stop: make(chan struct{}), done: make(chan struct{})}
	if err := h.restoreInbox(context.Background()); err != nil {
		releaseAttachment()
		return nil, fmt.Errorf("agent: restore inbox: %w", err)
	}
	h.approval = newApprovalBroker(loop.Config.EventHub, loop.Store, loop.SessionID)
	if err := h.approval.restore(context.Background()); err != nil {
		releaseAttachment()
		return nil, fmt.Errorf("agent: restore approvals: %w", err)
	}
	loop.mu.Lock()
	loop.Tools = scopedTools
	if loop.Config.NextStep == nil {
		loop.Config.NextStep = h.claimNextStepInputs
	}
	if loop.Config.RequeueStep == nil {
		loop.Config.RequeueStep = h.requeueStepInputs
	}
	loop.mu.Unlock()
	h.approval.setResume(h.resumeApprovedTool)
	if setter, ok := scopedTools.(interface{ SetApprovalService(tools.ApprovalService) }); ok {
		setter.SetApprovalService(h.approval)
	}
	go h.worker()
	if h.hasWakeableInbox() {
		h.wake <- struct{}{}
	}
	loop.Config.EventHub.PublishNotification(Notification{Type: NotificationAgentCreated, SessionID: loop.SessionID})
	return h, nil
}

func (h *Handle) hasWakeableInbox() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, item := range h.inbox {
		if item.kind == inboxFollowUp {
			return true
		}
	}
	return false
}

func (h *Handle) claimNextStepInputs() []StepInput {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.status == StatusDisposed || len(h.inbox) == 0 {
		return nil
	}
	inputs := make([]StepInput, 0)
	remaining := make([]inboxItem, 0, len(h.inbox))
	for _, item := range h.inbox {
		if item.kind == inboxSteer || item.kind == inboxInject {
			typ := session.EventInjectedContext
			if item.kind == inboxSteer {
				typ = session.EventSteeringMessage
			}
			inputs = append(inputs, StepInput{ID: item.id, Type: typ, Text: item.text})
			continue
		}
		remaining = append(remaining, item)
	}
	h.inbox = remaining
	if len(inputs) > 0 {
		h.signalChanged()
	}
	return inputs
}

func (h *Handle) requeueStepInputs(inputs []StepInput) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.status == StatusDisposed {
		return
	}
	items := make([]inboxItem, 0, len(inputs))
	for _, input := range inputs {
		if input.ID == "" || input.Text == "" {
			continue
		}
		kind := inboxInject
		if input.Type == session.EventSteeringMessage {
			kind = inboxSteer
		}
		items = append(items, inboxItem{id: input.ID, kind: kind, text: input.Text})
	}
	h.inbox = append(items, h.inbox...)
	h.signalChanged()
}

// New is the concise constructor alias used by applications embedding the
// library; NewAgent remains explicit for callers migrating from Loop.
func New(loop *Loop) (*Handle, error) { return NewAgent(loop) }

// RegisterAgent is the lifecycle-oriented constructor name used by hosts.
func RegisterAgent(loop *Loop) (*Handle, error) { return NewAgent(loop) }

func (h *Handle) ID() string { return h.loop.SessionID }

func (h *Handle) Session() SessionView     { return SessionView{session: h.session} }
func (h *Handle) ToolScope() tools.Runtime { return h.loop.Tools }
func (h *Handle) ToolCatalog() tools.Catalog {
	if catalog, ok := h.loop.Tools.(tools.Catalog); ok {
		return catalog
	}
	return nil
}
func (h *Handle) ScopedContext() tools.ToolRunContext {
	return tools.ToolRunContext{SessionID: h.ID(), Vars: cloneVars(h.loop.Config.Vars), Sandbox: h.loop.Config.Sandbox, Artifacts: h.loop.Config.Artifacts, Runtime: h.loop.Tools}
}

func cloneVars(vars map[string]any) map[string]any {
	if vars == nil {
		return nil
	}
	out := make(map[string]any, len(vars))
	for key, value := range vars {
		out[key] = value
	}
	return out
}

func (h *Handle) Status() AgentStatus {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.status
}

func (h *Handle) FollowUp(ctx context.Context, text string) error {
	return h.enqueue(ctx, inboxFollowUp, text, true)
}

func (h *Handle) Send(ctx context.Context, text string) error { return h.FollowUp(ctx, text) }

func (h *Handle) Steer(ctx context.Context, text string) error {
	// Steering belongs to the next model step; it is consumed by an already
	// active turn and does not open a fresh turn by itself.
	return h.enqueue(ctx, inboxSteer, text, false)
}

func (h *Handle) Inject(ctx context.Context, text string) error {
	// Injection is next-step context and intentionally does not wake an idle
	// agent; a host can add it while a turn is running or pair it with a
	// follow-up that supplies the wake-up boundary.
	return h.enqueue(ctx, inboxInject, text, false)
}

// Submit runs a host-owned durable input under the loop's claimed lease. It
// deliberately bypasses the in-process inbox, whose convenience methods are
// intended for interactive callers and use lease-less queue projections.
func (h *Handle) Submit(ctx context.Context, input Input) (*Result, error) {
	if input.Text == "" || input.ID == "" {
		return nil, fmt.Errorf("agent: input id and text are required")
	}
	if input.Type != session.EventUserMessage && input.Type != session.EventSteeringMessage && input.Type != session.EventInjectedContext {
		return nil, fmt.Errorf("agent: unsupported input event type %q", input.Type)
	}
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return nil, ErrAgentDisposed
	}
	if h.status == StatusRunning {
		h.mu.Unlock()
		return nil, ErrAgentBusy
	}
	h.status = StatusRunning
	runCtx, cancel := context.WithCancel(ctx)
	h.runCancel = cancel
	h.signalChanged()
	h.mu.Unlock()
	h.emitStatus(StatusRunning)
	defer func() {
		cancel()
		h.mu.Lock()
		h.runCancel = nil
		disposed := h.status == StatusDisposed
		if !disposed {
			h.status = StatusIdle
		}
		h.signalChanged()
		h.mu.Unlock()
		if !disposed {
			h.emitStatus(StatusIdle)
		}
	}()
	return h.loop.RunInputWithID(runCtx, input.Type, input.ID, input.Text)
}

func (h *Handle) enqueue(ctx context.Context, kind inboxKind, text string, wake bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if text == "" {
		return fmt.Errorf("agent: input text is required")
	}
	item := inboxItem{id: newInboxID(), kind: kind, text: text}
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return ErrAgentDisposed
	}
	if err := h.appendInbox(ctx, session.EventInboxQueued, item); err != nil {
		h.mu.Unlock()
		return err
	}
	h.inbox = append(h.inbox, item)
	h.signalChanged()
	if wake {
		select {
		case h.wake <- struct{}{}:
		default:
		}
	}
	h.mu.Unlock()
	return nil
}

func (h *Handle) Cancel(ctx context.Context, opts CancelOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return ErrAgentDisposed
	}
	if h.runCancel != nil {
		h.runCancel()
	}
	if !opts.KeepInbox {
		pending := append([]inboxItem(nil), h.inbox...)
		h.inbox = nil
		h.signalChanged()
		h.mu.Unlock()
		var discardErr error
		for _, item := range pending {
			if err := h.appendInbox(ctx, session.EventInboxDiscarded, item); err != nil && discardErr == nil {
				discardErr = err
			}
		}
		return discardErr
	}
	h.signalChanged()
	h.mu.Unlock()
	return nil
}

func (h *Handle) WhenIdle(ctx context.Context) error {
	for {
		h.mu.Lock()
		idle := h.status == StatusIdle && len(h.inbox) == 0
		h.mu.Unlock()
		if idle {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-h.done:
			return ErrAgentDisposed
		case <-h.changed:
		}
	}
}

// DecideApproval resolves a pending PolicyAsk call and resumes its executor.
func (h *Handle) DecideApproval(ctx context.Context, callID string, approved bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if callID == "" {
		return fmt.Errorf("agent: approval call id is required")
	}
	h.mu.Lock()
	disposed := h.status == StatusDisposed
	h.mu.Unlock()
	if disposed {
		return ErrAgentDisposed
	}
	return h.approval.Decide(ctx, callID, approved)
}

func (h *Handle) Approve(ctx context.Context, callID string) error {
	return h.DecideApproval(ctx, callID, true)
}

func (h *Handle) Reject(ctx context.Context, callID string) error {
	return h.DecideApproval(ctx, callID, false)
}

// ResumeTool completes a deferred tool under the loop's claimed lease.
func (h *Handle) ResumeTool(ctx context.Context, request ToolResume) error {
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return ErrAgentDisposed
	}
	if h.status == StatusRunning {
		h.mu.Unlock()
		return ErrAgentBusy
	}
	h.status = StatusRunning
	runCtx, cancel := context.WithCancel(ctx)
	h.runCancel = cancel
	h.signalChanged()
	h.mu.Unlock()
	h.emitStatus(StatusRunning)
	defer func() {
		cancel()
		h.mu.Lock()
		h.runCancel = nil
		disposed := h.status == StatusDisposed
		if !disposed {
			h.status = StatusIdle
		}
		h.signalChanged()
		h.mu.Unlock()
		if !disposed {
			h.emitStatus(StatusIdle)
		}
	}()
	return h.loop.ResumeTool(runCtx, request)
}

// ResumeDeferredTools executes the durable deferred calls after their
// host-owned prerequisite is ready. The handle keeps the same single-run
// status guard as Submit and ResumeTool.
func (h *Handle) ResumeDeferredTools(ctx context.Context, requests []DeferredToolResume) error {
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return ErrAgentDisposed
	}
	if h.status == StatusRunning {
		h.mu.Unlock()
		return ErrAgentBusy
	}
	h.status = StatusRunning
	runCtx, cancel := context.WithCancel(ctx)
	h.runCancel = cancel
	h.signalChanged()
	h.mu.Unlock()
	h.emitStatus(StatusRunning)
	defer func() {
		cancel()
		h.mu.Lock()
		h.runCancel = nil
		disposed := h.status == StatusDisposed
		if !disposed {
			h.status = StatusIdle
		}
		h.signalChanged()
		h.mu.Unlock()
		if !disposed {
			h.emitStatus(StatusIdle)
		}
	}()
	return h.loop.ResumeDeferredTools(runCtx, requests)
}

func (h *Handle) resumeApprovedTool(ctx context.Context, request tools.ApprovalRequest, approved bool) error {
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return ErrAgentDisposed
	}
	if h.status == StatusRunning {
		h.mu.Unlock()
		return ErrAgentBusy
	}
	h.status = StatusRunning
	runCtx, cancel := context.WithCancel(ctx)
	h.runCancel = cancel
	h.signalChanged()
	h.mu.Unlock()
	h.emitStatus(StatusRunning)
	defer func() {
		cancel()
		h.mu.Lock()
		h.runCancel = nil
		disposed := h.status == StatusDisposed
		if !disposed {
			h.status = StatusIdle
		}
		h.signalChanged()
		h.mu.Unlock()
		if disposed {
			return
		}
		h.emitStatus(StatusIdle)
		pending := h.hasWakeableInbox()
		if pending {
			select {
			case h.wake <- struct{}{}:
			default:
			}
		}
	}()
	return h.loop.ResumeApprovedTool(runCtx, request, approved)
}

func (h *Handle) Subscribe(ctx context.Context, after uint64, filter EventFilter) (*Subscription, error) {
	return h.loop.Subscribe(ctx, after, filter)
}

func (h *Handle) SubscribeNotifications(ctx context.Context) *NotificationSubscription {
	return h.loop.Config.EventHub.SubscribeNotificationsFor(ctx, h.ID())
}

// Run is the compatibility one-shot API. Stateful callers should use
// FollowUp/WhenIdle and consume the event subscription.
func (h *Handle) Run(ctx context.Context, input string) (*Result, error) {
	h.mu.Lock()
	if h.status == StatusDisposed {
		h.mu.Unlock()
		return nil, ErrAgentDisposed
	}
	if h.status == StatusRunning {
		h.mu.Unlock()
		return nil, ErrAgentBusy
	}
	h.status = StatusRunning
	runCtx, cancel := context.WithCancel(ctx)
	h.runCancel = cancel
	h.signalChanged()
	h.mu.Unlock()
	h.emitStatus(StatusRunning)
	defer func() {
		_ = h.session.Refresh(context.Background())
		cancel()
		h.mu.Lock()
		h.runCancel = nil
		disposed := h.status == StatusDisposed
		if !disposed {
			h.status = StatusIdle
		}
		h.signalChanged()
		pending := h.hasWakeableInbox()
		h.mu.Unlock()
		if disposed {
			return
		}
		h.emitStatus(StatusIdle)
		if pending {
			select {
			case h.wake <- struct{}{}:
			default:
			}
		}
	}()
	return h.loop.Run(runCtx, input)
}

func (h *Handle) worker() {
	defer close(h.done)
	for {
		select {
		case <-h.stop:
			return
		case <-h.wake:
		}
		for {
			h.mu.Lock()
			wakeIndex := h.nextWakeableIndexLocked()
			if h.status == StatusDisposed || h.status != StatusIdle || wakeIndex < 0 {
				h.mu.Unlock()
				break
			}
			item := h.inbox[wakeIndex]
			h.inbox = append(h.inbox[:wakeIndex], h.inbox[wakeIndex+1:]...)
			h.status = StatusRunning
			runCtx, cancel := context.WithCancel(context.Background())
			h.runCancel = cancel
			h.signalChanged()
			h.mu.Unlock()
			h.emitStatus(StatusRunning)

			inputType := session.EventUserMessage
			switch item.kind {
			case inboxSteer:
				inputType = session.EventSteeringMessage
			case inboxInject:
				inputType = session.EventInjectedContext
			}
			if err := h.appendInbox(runCtx, session.EventInboxClaimed, item); err != nil {
				cancel()
				h.mu.Lock()
				h.status = StatusIdle
				h.runCancel = nil
				h.inbox = append([]inboxItem{item}, h.inbox...)
				h.signalChanged()
				h.mu.Unlock()
				h.emitStatus(StatusIdle)
				continue
			}
			_, runErr := h.loop.RunInputWithID(runCtx, inputType, item.id, item.text)
			_ = h.session.Refresh(context.Background())
			if runErr == nil {
				if completeErr := h.appendInbox(context.Background(), session.EventInboxCompleted, item); completeErr != nil {
					runErr = completeErr
					h.requeueAfterRun(item)
				}
			} else {
				// A failed or cancelled fenced run has not consumed the inbox
				// command. Keep it durable and in memory for a later retry.
				h.requeueAfterRun(item)
			}
			if runErr != nil && !errors.Is(runErr, context.Canceled) && !errors.Is(runErr, context.DeadlineExceeded) {
				h.loop.Config.EventHub.PublishNotification(Notification{
					Type:      NotificationAgentRunFailed,
					SessionID: h.ID(),
					Data:      mustJSON(map[string]any{"error": runErr.Error()}),
				})
			}
			cancel()
			h.mu.Lock()
			h.runCancel = nil
			disposed := h.status == StatusDisposed
			if !disposed {
				h.status = StatusIdle
			}
			h.signalChanged()
			h.mu.Unlock()
			if !disposed {
				h.emitStatus(StatusIdle)
			}
			if runErr != nil {
				// Leave the failed command pending without entering a hot
				// retry loop. A later wake or a restarted handle can retry it.
				break
			}
		}
	}
}

func (h *Handle) nextWakeableIndexLocked() int {
	for index, item := range h.inbox {
		if item.kind == inboxFollowUp {
			return index
		}
	}
	return -1
}

func (h *Handle) appendInbox(ctx context.Context, typ session.EventType, item inboxItem) error {
	return h.appendInboxEvent(ctx, typ, item, item.id+":"+string(typ))
}

func (h *Handle) appendInboxEvent(ctx context.Context, typ session.EventType, item inboxItem, eventID string) error {
	data := session.InboxPayloadJSON(item.id, inboxKindName(item.kind), item.text)
	_, err := appendDurable(ctx, h.loop.Store, h.loop.Config.EventHub, h.loop.SessionID, session.Event{
		ID: eventID, SessionID: h.loop.SessionID, Type: typ, Data: data,
	})
	return err
}

func (h *Handle) requeueAfterRun(item inboxItem) {
	eventID := fmt.Sprintf("%s:%s:%d", item.id, session.EventInboxRequeued, atomic.AddUint64(&inboxCounter, 1))
	_ = h.appendInboxEvent(context.Background(), session.EventInboxRequeued, item, eventID)
	h.mu.Lock()
	if h.status != StatusDisposed {
		h.inbox = append([]inboxItem{item}, h.inbox...)
		h.signalChanged()
	}
	h.mu.Unlock()
}

func inboxKindName(kind inboxKind) string {
	switch kind {
	case inboxSteer:
		return "steer"
	case inboxInject:
		return "inject"
	default:
		return "follow_up"
	}
}

func inboxKindFromName(name string) inboxKind {
	switch name {
	case "steer":
		return inboxSteer
	case "inject":
		return inboxInject
	default:
		return inboxFollowUp
	}
}

func newInboxID() string {
	return fmt.Sprintf("inbox-%d-%d", atomic.AddUint64(&inboxCounter, 1), time.Now().UnixNano())
}

var inboxCounter uint64

func (h *Handle) restoreInbox(ctx context.Context) error {
	events, err := h.loop.Store.Load(ctx, h.loop.SessionID, 0, 0)
	if err != nil {
		return err
	}
	queued := map[string]inboxItem{}
	queuedSeq := map[string]uint64{}
	terminal := map[string]bool{}
	requeued := map[string]bool{}
	inputSeen := map[string]bool{}
	inputRun := map[string]string{}
	runEnded := map[string]bool{}
	failedRuns := map[string]bool{}
	for _, event := range events {
		if event.Type == session.EventUserMessage || event.Type == session.EventSteeringMessage || event.Type == session.EventInjectedContext {
			var payload struct {
				InboxID string `json:"inbox_id"`
			}
			if json.Unmarshal(event.Data, &payload) == nil && payload.InboxID != "" {
				inputSeen[payload.InboxID] = true
				inputRun[payload.InboxID] = event.RunID
			}
		}
		if event.Type == session.EventTurnEnd && event.RunID != "" {
			runEnded[event.RunID] = true
			var payload struct {
				Reason string `json:"reason"`
			}
			if json.Unmarshal(event.Data, &payload) == nil {
				switch payload.Reason {
				case "error", "cancelled", "context_overflow":
					failedRuns[event.RunID] = true
				}
			}
		}
		if event.Type == session.EventRequestError && event.RunID != "" {
			failedRuns[event.RunID] = true
		}
		switch event.Type {
		case session.EventInboxQueued, session.EventInboxRequeued:
			var payload session.InboxPayload
			if json.Unmarshal(event.Data, &payload) == nil && payload.ItemID != "" {
				queued[payload.ItemID] = inboxItem{id: payload.ItemID, kind: inboxKindFromName(payload.Kind), text: payload.Text}
				queuedSeq[payload.ItemID] = event.Seq
				requeued[payload.ItemID] = event.Type == session.EventInboxRequeued
				terminal[payload.ItemID] = false
			}
		case session.EventInboxCompleted, session.EventInboxDiscarded:
			var payload session.InboxPayload
			if json.Unmarshal(event.Data, &payload) == nil {
				terminal[payload.ItemID] = true
				requeued[payload.ItemID] = false
			}
		}
	}
	ids := make([]string, 0, len(queued))
	for id := range queued {
		ids = append(ids, id)
	}
	sort.SliceStable(ids, func(i, j int) bool {
		if queuedSeq[ids[i]] == queuedSeq[ids[j]] {
			return ids[i] < ids[j]
		}
		return queuedSeq[ids[i]] < queuedSeq[ids[j]]
	})
	for _, id := range ids {
		item := queued[id]
		if terminal[id] {
			continue
		}
		if inputSeen[id] && runEnded[inputRun[id]] && !failedRuns[inputRun[id]] && !requeued[id] {
			// A completed turn already consumed this input. Seal the queue
			// lifecycle without replaying the user message.
			if err := h.appendInbox(ctx, session.EventInboxCompleted, item); err != nil {
				return err
			}
			continue
		}
		// A queued item whose input was written but whose turn has no end is
		// an interrupted run. Requeue it; the next RunInputWithID is fenced
		// and idempotent for the existing input event.
		h.inbox = append(h.inbox, item)
	}
	return nil
}

func (h *Handle) Dispose(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	var discardErr error
	h.disposeOnce.Do(func() {
		h.mu.Lock()
		h.status = StatusDisposed
		pending := append([]inboxItem(nil), h.inbox...)
		h.inbox = nil
		if h.runCancel != nil {
			h.runCancel()
		}
		close(h.stop)
		h.signalChanged()
		h.mu.Unlock()
		for _, item := range pending {
			if err := h.appendInbox(context.Background(), session.EventInboxDiscarded, item); err != nil && discardErr == nil {
				discardErr = err
			}
		}
		h.emitStatus(StatusDisposed)
	})
	select {
	case <-h.done:
		return discardErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (h *Handle) Close(ctx context.Context) error { return h.Dispose(ctx) }

func (h *Handle) signalChanged() {
	select {
	case h.changed <- struct{}{}:
	default:
	}
}

func (h *Handle) emitStatus(status AgentStatus) {
	h.loop.Config.EventHub.PublishNotification(Notification{Type: NotificationAgentStatus, SessionID: h.ID(), Data: mustJSON(map[string]any{"status": status})})
}

type approvalBroker struct {
	mu        sync.Mutex
	pending   map[string]chan bool
	requested map[string]bool
	decisions map[string]bool
	requests  map[string]tools.ApprovalRequest
	lease     *session.Lease
	hub       *EventHub
	store     Store
	sessionID string
	resume    func(context.Context, tools.ApprovalRequest, bool) error
}

func newApprovalBroker(hub *EventHub, store Store, sessionID string) *approvalBroker {
	return &approvalBroker{pending: make(map[string]chan bool), requested: make(map[string]bool), decisions: make(map[string]bool), requests: make(map[string]tools.ApprovalRequest), hub: hub, store: store, sessionID: sessionID}
}

func (b *approvalBroker) setResume(resume func(context.Context, tools.ApprovalRequest, bool) error) {
	b.mu.Lock()
	b.resume = resume
	b.mu.Unlock()
}

func (b *approvalBroker) SetLease(lease session.Lease) {
	b.mu.Lock()
	copyLease := lease
	b.lease = &copyLease
	b.mu.Unlock()
}

func (b *approvalBroker) Approve(ctx context.Context, request tools.ApprovalRequest) (bool, error) {
	decision := make(chan bool, 1)
	b.mu.Lock()
	if approved, decided := b.decisions[request.CallID]; decided {
		b.mu.Unlock()
		return approved, nil
	}
	if _, exists := b.pending[request.CallID]; exists {
		b.mu.Unlock()
		return false, fmt.Errorf("agent: approval already pending for %s", request.CallID)
	}
	b.pending[request.CallID] = decision
	b.requested[request.CallID] = true
	b.requests[request.CallID] = request
	b.mu.Unlock()
	if err := b.append(ctx, session.EventApprovalRequested, request.RunID, request.TurnID, request.StepID, request.CallID, mustJSON(request)); err != nil {
		b.mu.Lock()
		delete(b.pending, request.CallID)
		delete(b.requested, request.CallID)
		delete(b.requests, request.CallID)
		b.mu.Unlock()
		return false, err
	}
	b.hub.PublishNotification(Notification{Type: NotificationToolApprovalRequested, SessionID: request.SessionID, RunID: request.RunID, CallID: request.CallID, Data: mustJSON(request)})
	defer func() {
		b.mu.Lock()
		delete(b.pending, request.CallID)
		b.mu.Unlock()
	}()
	select {
	case approved := <-decision:
		return approved, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func (b *approvalBroker) Decide(ctx context.Context, callID string, approved bool) error {
	b.mu.Lock()
	decision, ok := b.pending[callID]
	requested := b.requested[callID]
	previousDecision, alreadyDecided := b.decisions[callID]
	request := b.requests[callID]
	resume := b.resume
	b.mu.Unlock()
	if alreadyDecided {
		if previousDecision != approved {
			return fmt.Errorf("agent: approval %s was already resolved", callID)
		}
		if decision == nil && requested && resume != nil && request.CallID != "" {
			return resume(ctx, request, previousDecision)
		}
		return nil
	}
	if !ok && !requested {
		return fmt.Errorf("agent: no approval pending for %s", callID)
	}
	if err := b.append(ctx, session.EventApprovalResolved, request.RunID, request.TurnID, request.StepID, callID, session.ApprovalResolvedJSON(callID, approved)); err != nil {
		return err
	}
	b.mu.Lock()
	b.decisions[callID] = approved
	b.mu.Unlock()
	if decision != nil {
		decision <- approved
		return nil
	}
	if resume != nil && requested {
		return resume(ctx, request, approved)
	}
	return nil
}

func (b *approvalBroker) restore(ctx context.Context) error {
	events, err := b.store.Load(ctx, b.sessionID, 0, 0)
	if err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, event := range events {
		switch event.Type {
		case session.EventApprovalRequested:
			if event.CallID != "" {
				b.requested[event.CallID] = true
				var request tools.ApprovalRequest
				if json.Unmarshal(event.Data, &request) == nil {
					b.requests[event.CallID] = request
				}
			}
		case session.EventApprovalResolved:
			var payload session.ApprovalResolvedPayload
			if json.Unmarshal(event.Data, &payload) == nil && payload.CallID != "" {
				b.requested[payload.CallID] = true
				b.decisions[payload.CallID] = payload.Approved
			}
		}
	}
	return nil
}

func (b *approvalBroker) append(ctx context.Context, typ session.EventType, runID, turnID, stepID, callID string, data json.RawMessage) error {
	id := fmt.Sprintf("approval-%s-%s", callID, typ)
	b.mu.Lock()
	lease := b.lease
	if lease != nil {
		copyLease := *lease
		lease = &copyLease
	}
	b.mu.Unlock()
	if lease != nil {
		if batchStore, ok := b.store.(session.FencedBatchStore); ok {
			batch, err := batchStore.AppendFencedCommitted(ctx, *lease, []session.Event{{ID: id, SessionID: b.sessionID, RunID: runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: typ, Data: data}})
			if err == nil {
				for _, event := range batch.Events {
					b.hub.Publish(event)
				}
			}
			return err
		}
		if fenced, ok := b.store.(session.FencedStore); ok {
			if _, err := fenced.AppendFenced(ctx, *lease, []session.Event{{ID: id, SessionID: b.sessionID, RunID: runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: typ, Data: data}}); err != nil {
				return err
			}
			committed, err := b.store.Load(ctx, b.sessionID, 0, 0)
			if err != nil {
				return err
			}
			for _, event := range committed {
				if event.ID == id {
					b.hub.Publish(event)
					return nil
				}
			}
			return fmt.Errorf("agent: committed approval event %s not found", id)
		}
	}
	_, err := appendDurable(ctx, b.store, b.hub, b.sessionID, session.Event{ID: id, SessionID: b.sessionID, RunID: runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: typ, Data: data})
	return err
}

func mustJSON(value any) json.RawMessage {
	data, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(`{}`)
	}
	return data
}
