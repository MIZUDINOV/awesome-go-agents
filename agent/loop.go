// Package agent implements the durable, resumable agent loop.
//
// Responsibilities (H-THEORY, §7):
//
//   - The loop owns orchestration: it persists every model step and every tool
//     invocation to a durable event log BEFORE the next model call, so a crash
//     never loses the provenance of a side effect.
//   - Durable log != model surface != tool runtime. The loop appends events to
//     the log, projects them into model messages, and executes tools through a
//     Registry — never editing a mutable history in place.
//   - Single-writer: every mutation is a fenced append under a session lease;
//     a lost/expired lease aborts the run (ErrLeaseLost) instead of writing
//     blindly. Recovery closes orphaned turns and settles calls from their
//     durable dispatch barrier (never a blind retry of side-effecting work).
//   - Cancellation has priority over recovery/compaction.
//   - Context overflow handling order: prune -> max-safe compaction -> bounded
//     retry, and only after the surface advances.
//
// Tool calls are scheduler-owned: the provider may return parallel calls,
// but bodies only overlap after durable call barriers and only when their
// definitions explicitly classify them as concurrency-safe.
package agent

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	memctx "github.com/MIZUDINOV/awesome-go-agents/context"
	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

// Sentinel loop errors.
var (
	// ErrLeaseLost propagates when the worker's lease was taken by another
	// worker; all writes must stop (H-DB-008).
	ErrLeaseLost = errors.New("agent: lease lost or expired")
	// ErrBusy is returned when another worker holds the session lease.
	ErrBusy = errors.New("agent: session is busy (lease held by another worker)")
	// ErrContextOverflow is returned when the request exceeds context capacity
	// even after max-safe compaction.
	ErrContextOverflow = errors.New("agent: context overflow after max-safe compaction")
	// ErrStopped is returned when the run was cancelled mid-turn.
	ErrStopped    = errors.New("agent: run stopped (cancelled)")
	ErrToolLimit  = errors.New("agent: tool-call limit exceeded")
	ErrTokenLimit = errors.New("agent: token limit exceeded")
)

// Chat is the provider-neutral model seam.
type Chat interface {
	Generate(ctx context.Context, req *llm.Request, cb llm.StreamCallback) (*llm.Response, error)
	Name() string
}

type capabilityChat interface {
	Capabilities(context.Context, string) (llm.Capabilities, error)
}

type requestSnapshotChat interface {
	RequestSnapshot(*llm.Request) (json.RawMessage, error)
}

// Store is the durable single-writer seam the loop persists to.
type Store interface {
	session.FencedStore
}

// Compactor produces the durable summary + fingerprint over a region of the
// log during compaction. When nil, the loop uses a deterministic bounded
// summary and still advances the durable surface.
type Compactor interface {
	Compact(ctx context.Context, generation uint64, events []session.Event, throughSeq uint64) (summary, fingerprint string, err error)
}

// Config controls the loop.
type Config struct {
	Model        string
	Owner        string
	SystemPrompt string

	MaxStepsPerTurn   int
	MaxToolCalls      int
	MaxTokens         int64
	MaxTotalTokens    int64
	MaxWallTime       time.Duration
	MaxContextRetries int

	Stream bool

	// ProviderConfig is opaque JSON embedded in llm.Request.Config.
	ProviderConfig json.RawMessage

	LeaseTTL time.Duration
	// ClaimedLease binds the loop to a lease already owned by the host. When
	// set, Run never claims, renews, or releases it. This is required when the
	// host owns the canonical run fence (as Wzhooh does).
	ClaimedLease *session.Lease

	ContextWindow         int64
	MaxOutput             int64
	CompactThresholdRatio float64
	PruneHead, PruneTail  int
	// RetainTailEvents is the minimum recent model-visible tail that compaction
	// must preserve verbatim, including the current user/steering input.
	RetainTailEvents int
	Compactor        Compactor

	// Vars are merged into every tool ExecContext (host bindings such as the
	// sandbox working directory under "cwd").
	Vars map[string]any
	// Sandbox is the host-bound admission authority for this claimed run.
	Sandbox integration.Sandbox
	// Artifacts is the optional durable spill port for complete tool output.
	Artifacts integration.ArtifactStore
	// EventHub receives committed durable events without blocking the loop.
	EventHub *EventHub
	// ContextBuilder makes prompt assembly and token estimation one immutable
	// operation. Nil keeps the legacy SystemPrompt path.
	ContextBuilder   *memctx.Builder
	Instructions     []memctx.Section
	ToolGuidance     []memctx.Section
	RuntimeContext   []memctx.Section
	WorkspaceContext []memctx.Section
	// NextStep supplies durable steering/injected inputs claimed immediately
	// before a model step. The public Agent handle installs this callback;
	// standalone Loop users may leave it nil.
	NextStep func() []StepInput
	// RequeueStep receives claimed steering items if the durable append is
	// interrupted before they become model-visible.
	RequeueStep func([]StepInput)
}

// StepInput is a durable steering/context item inserted at the next model
// step, without reopening the current turn.
type StepInput struct {
	ID   string
	Type session.EventType
	Text string
}

// DefaultConfig returns the review-checklist reference defaults.
func DefaultConfig() Config {
	return Config{
		MaxStepsPerTurn: 6, MaxTokens: 4096,
		MaxToolCalls: 64, MaxTotalTokens: 1000000, MaxContextRetries: 1, MaxWallTime: 30 * time.Minute,
		LeaseTTL:      30 * time.Second,
		ContextWindow: 128000, MaxOutput: 4096,
		CompactThresholdRatio: 0.80, PruneHead: 4096, PruneTail: 1024,
		RetainTailEvents: 8,
	}
}

// Loop runs a durable session against the seams.
type Loop struct {
	SessionID string
	Store     Store
	Tools     tools.Runtime
	Chat      Chat
	Config    Config

	mu            sync.Mutex
	events        []session.Event
	agentAttached bool
}

// NewLoop builds a Loop. cfg is normalized with reference defaults.
func NewLoop(sessionID string, store Store, registry tools.Runtime, chat Chat, cfg Config) *Loop {
	def := DefaultConfig()
	if cfg.MaxStepsPerTurn <= 0 {
		cfg.MaxStepsPerTurn = def.MaxStepsPerTurn
	}
	if cfg.MaxToolCalls <= 0 {
		cfg.MaxToolCalls = def.MaxToolCalls
	}
	if cfg.MaxTotalTokens <= 0 {
		cfg.MaxTotalTokens = def.MaxTotalTokens
	}
	if cfg.MaxWallTime <= 0 {
		cfg.MaxWallTime = def.MaxWallTime
	}
	if cfg.MaxContextRetries <= 0 {
		cfg.MaxContextRetries = def.MaxContextRetries
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = def.MaxTokens
	}
	if cfg.LeaseTTL <= 0 {
		cfg.LeaseTTL = def.LeaseTTL
	}
	if cfg.ContextWindow <= 0 {
		cfg.ContextWindow = def.ContextWindow
	}
	if cfg.MaxOutput <= 0 {
		cfg.MaxOutput = def.MaxOutput
	}
	if cfg.CompactThresholdRatio <= 0 {
		cfg.CompactThresholdRatio = def.CompactThresholdRatio
	}
	if cfg.PruneHead <= 0 {
		cfg.PruneHead = def.PruneHead
	}
	if cfg.PruneTail <= 0 {
		cfg.PruneTail = def.PruneTail
	}
	if cfg.RetainTailEvents <= 0 {
		cfg.RetainTailEvents = def.RetainTailEvents
	}
	if cfg.EventHub == nil {
		cfg.EventHub = NewEventHub(64)
	}
	return &Loop{SessionID: sessionID, Store: store, Tools: registry, Chat: chat, Config: cfg}
}

// Subscribe returns a replay-then-live view of this loop's durable session.
// The caller's cursor is inclusive-exclusive: events with seq <= after are
// skipped, and the subscription resumes at the next committed event.
func (l *Loop) Subscribe(ctx context.Context, after uint64, filter EventFilter) (*Subscription, error) {
	l.mu.Lock()
	if l.Config.EventHub == nil {
		l.Config.EventHub = NewEventHub(64)
	}
	hub := l.Config.EventHub
	l.mu.Unlock()
	sessionFilter := func(event session.Event) bool {
		if event.SessionID != l.SessionID {
			return false
		}
		return filter == nil || filter(event)
	}
	return hub.SubscribeCursor(ctx, after, sessionFilter, func() ([]session.Event, error) {
		replay, err := l.Store.Load(ctx, l.SessionID, after, 0)
		if err != nil {
			return nil, fmt.Errorf("agent: load event subscription replay: %w", err)
		}
		return replay, nil
	})
}

// Result is the outcome of a Run.
type Result struct {
	Input          string
	Messages       []*llm.Message
	Turns          int
	FinishedReason string
	Interrupted    bool
	CompactCount   int
}

// Run executes one user input against the durable session.
func (l *Loop) Run(ctx context.Context, input string) (*Result, error) {
	return l.RunInput(ctx, session.EventUserMessage, input)
}

// RunInput executes one queued input with an explicit durable message kind.
// Steering and injected context are model-surface messages but remain
// distinguishable in the canonical event log.
func (l *Loop) RunInput(ctx context.Context, inputType session.EventType, input string) (*Result, error) {
	return l.RunInputWithID(ctx, inputType, "", input)
}

// ResumeApprovedTool completes a durable approval that survived a worker
// restart. It is deliberately limited to calls without a dispatch barrier;
// once dispatch was durable, recovery must classify the outcome as unknown
// instead of risking a duplicate side effect.
func (l *Loop) ResumeApprovedTool(ctx context.Context, request tools.ApprovalRequest, approved bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if request.CallID == "" || request.ToolName == "" {
		return fmt.Errorf("agent: approval request is incomplete")
	}
	var lease session.Lease
	externalLease := l.Config.ClaimedLease != nil
	if externalLease {
		lease = *l.Config.ClaimedLease
		if lease.SessionID != l.SessionID || lease.Token == "" || lease.Fence == 0 {
			return ErrLeaseLost
		}
	} else {
		claimed, err := l.Store.ClaimLease(ctx, l.SessionID, l.Config.Owner, l.Config.LeaseTTL, "")
		if err != nil {
			return err
		}
		lease = claimed
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		defer func() { _ = l.Store.ReleaseLease(cleanupCtx, lease) }()
	}
	resumeCtx := ctx
	stopHeartbeat := func() error { return nil }
	if !externalLease {
		resumeCtx, stopHeartbeat = l.leaseHeartbeat(ctx, lease)
		defer func() { _ = stopHeartbeat() }()
	}
	if err := l.refresh(resumeCtx); err != nil {
		return err
	}
	var callEvent session.Event
	resultExists, dispatched, stepEnded, turnEnded := false, false, false, false
	for _, event := range l.snapshotEvents() {
		if event.Type == session.EventToolCall && event.CallID == request.CallID {
			callEvent = event
		}
		if event.Type == session.EventToolResult && event.CallID == request.CallID {
			resultExists = true
		}
		if (event.Type == session.EventToolDispatched || event.Type == session.EventToolRunning) && event.CallID == request.CallID {
			dispatched = true
		}
		if event.Type == session.EventStepEnd && event.TurnID == request.TurnID && event.StepID == request.StepID {
			stepEnded = true
		}
		if event.Type == session.EventTurnEnd && event.TurnID == request.TurnID {
			turnEnded = true
		}
	}
	if resultExists && turnEnded {
		return nil
	}
	if dispatched {
		return fmt.Errorf("agent: approved tool %s was already dispatched", request.CallID)
	}
	if callEvent.ID == "" {
		return fmt.Errorf("agent: approval call %s was not found", request.CallID)
	}
	runID, turnID, stepID := callEvent.RunID, callEvent.TurnID, callEvent.StepID
	if runID == "" {
		runID = request.RunID
	}
	if turnID == "" {
		turnID = "turn-resume-" + request.CallID
	}
	if stepID == "" {
		stepID = turnID + "-step-00"
	}
	for _, event := range l.snapshotEvents() {
		if event.Type == session.EventStepEnd && event.TurnID == turnID && event.StepID == stepID {
			stepEnded = true
		}
		if event.Type == session.EventTurnEnd && event.TurnID == turnID {
			turnEnded = true
		}
	}
	if turnEnded {
		return fmt.Errorf("agent: approval result belongs to an already closed turn")
	}
	em := &emitter{sessionID: l.SessionID, runID: runID}
	persistCtx := durableContext(resumeCtx)
	concludesTurn := false
	if !approved {
		if !resultExists {
			if err := l.appendToolResultErr(persistCtx, lease, em, request.CallID, request.ToolName, tools.ErrApprovalDenied, turnID, stepID); err != nil {
				return err
			}
		}
	} else if !resultExists {
		onDispatch := func(dispatchCtx context.Context, name, callID string) error {
			if err := l.append(dispatchCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: session.EventToolDispatched, Data: strJSON("dispatched")}); err != nil {
				return err
			}
			return l.append(dispatchCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: session.EventToolRunning, Data: strJSON("running")})
		}
		calls := []tools.Call{{Name: request.ToolName, CallID: request.CallID, Input: append(json.RawMessage(nil), request.Arguments...)}}
		outcomes := l.Tools.RunBatch(resumeCtx, tools.ExecContext{SessionID: l.SessionID, RunID: runID, TurnID: turnID, StepID: stepID, Vars: mergeVars(l.Config.Vars), Sandbox: l.Config.Sandbox, Artifacts: l.Config.Artifacts, Lease: &lease, OnDispatch: onDispatch}, calls)
		if err := validateToolOutcomes(calls, outcomes); err != nil {
			return err
		}
		var err error
		concludesTurn, err = l.appendToolOutcome(persistCtx, lease, em, outcomes[0], turnID, stepID)
		if err != nil {
			return err
		}
	}
	if !stepEnded {
		if err := l.append(persistCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("approval_resolved")}); err != nil {
			return err
		}
	}
	if concludesTurn {
		if err := l.appendTurnEnd(persistCtx, lease, em, turnID, "tool_concluded"); err != nil {
			return err
		}
		l.finish(&Result{}, "tool_concluded")
		return nil
	}
	_, runErr := l.runTurnSteps(resumeCtx, lease, stopHeartbeat, em, turnID, &Result{}, nextStepIndex(stepID), nil)
	return runErr
}

// RunInputWithID is the durable inbox-aware variant. inputID is used only as
// a correlation key in the user payload; the event envelope still receives a
// distinct id so fenced idempotency cannot confuse an inbox record with the
// model-visible message.
func (l *Loop) RunInputWithID(ctx context.Context, inputType session.EventType, inputID, input string) (*Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, ErrStopped
	}
	if inputType != session.EventUserMessage && inputType != session.EventSteeringMessage && inputType != session.EventInjectedContext {
		return nil, fmt.Errorf("agent: unsupported input event type %q", inputType)
	}
	cfg := l.Config
	if cfg.MaxWallTime > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.MaxWallTime)
		defer cancel()
	}

	// Single-writer admission. A host-provided claim is authoritative: AgentKit
	// must not create a competing renewal loop or release that host lease.
	var lease session.Lease
	var err error
	externalLease := cfg.ClaimedLease != nil
	if externalLease {
		lease = *cfg.ClaimedLease
		if lease.SessionID != l.SessionID || lease.Token == "" || lease.Fence == 0 {
			return nil, ErrLeaseLost
		}
	} else {
		lease, err = l.Store.ClaimLease(ctx, l.SessionID, cfg.Owner, cfg.LeaseTTL, "")
		if err != nil {
			if errors.Is(err, session.ErrLeaseHeld) {
				return nil, ErrBusy
			}
			return nil, err
		}
	}
	loopCtx := ctx
	stopHeartbeat := func() error { return nil }
	if !externalLease {
		loopCtx, stopHeartbeat = l.leaseHeartbeat(ctx, lease)
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		defer func() {
			_ = stopHeartbeat()
			_ = l.Store.ReleaseLease(cleanupCtx, lease)
		}()
	}

	// Recovery runs BEFORE any new work: close an orphaned turn from a crash
	// and mark dangling tool calls TOOL_OUTCOME_UNKNOWN (never re-execute).
	recoveryAfter, sequenceErr := l.Store.Sequence(loopCtx, l.SessionID)
	recovery, err := l.Store.Recover(loopCtx, lease)
	if err != nil {
		return nil, err
	}
	if sequenceErr == nil && recovery != nil && recovery.EventsAppended > 0 && l.Config.EventHub != nil {
		if recovered, loadErr := l.Store.Load(loopCtx, l.SessionID, recoveryAfter, 0); loadErr == nil {
			for _, event := range recovered {
				l.Config.EventHub.Publish(event)
			}
		}
	}
	if err := l.refresh(loopCtx); err != nil {
		return nil, err
	}

	runID := newRunID()
	em := &emitter{sessionID: l.SessionID, runID: runID}
	turnID := fmt.Sprintf("turn-%04d", l.nextTurnNumber())
	if err := l.append(loopCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: runID, TurnID: turnID, Type: session.EventTurnStart, Data: strJSON("start")}); err != nil {
		return nil, err
	}

	// Durable request preamble (diagnostics, non-surface).
	requestContext := em.next(session.EventRequestContext, map[string]any{
		"model": cfg.Model, "context_window": cfg.ContextWindow, "max_output": cfg.MaxOutput,
	})
	requestContext.TurnID = turnID
	if err := l.append(loopCtx, lease, requestContext); err != nil {
		return nil, err
	}
	initialInput := &StepInput{ID: inputID, Type: inputType, Text: input}
	return l.runTurnSteps(loopCtx, lease, stopHeartbeat, em, turnID, &Result{Input: input}, 0, initialInput)
}

func (l *Loop) runTurnSteps(ctx context.Context, lease session.Lease, stopHeartbeat func() error, em *emitter, turnID string, result *Result, startStep int, initialInput *StepInput) (*Result, error) {
	cfg := l.Config
	if startStep < 0 {
		startStep = 0
	}
	compacted := 0
	overflowRetries := 0
	toolCallCount := 0
	totalTokens := int64(0)
	for step := startStep; step < cfg.MaxStepsPerTurn; step++ {
		if err := ctx.Err(); err != nil {
			if heartbeatErr := stopHeartbeat(); heartbeatErr != nil {
				return nil, ErrLeaseLost
			}
			if endErr := l.appendTurnEnd(durableContext(ctx), lease, em, turnID, "cancelled"); endErr != nil {
				return nil, endErr
			}
			l.finish(result, "interrupted")
			return result, ErrStopped
		}
		var inputs []StepInput
		if cfg.NextStep != nil {
			inputs = cfg.NextStep()
		}
		stepAttempt := 0
		var done bool
		var reason string
		var err error
		for {
			done, reason, err = l.step(ctx, lease, em, turnID, step, stepAttempt, initialInput, inputs, &compacted, &toolCallCount, &totalTokens)
			initialInput = nil
			if err == nil {
				break
			}
			if ctx.Err() != nil {
				if heartbeatErr := stopHeartbeat(); heartbeatErr != nil {
					return nil, ErrLeaseLost
				}
			}
			if errors.Is(err, ErrContextOverflow) {
				_ = l.appendTurnEnd(durableContext(ctx), lease, em, turnID, "context_overflow")
				return nil, ErrContextOverflow
			}
			if isOverflow(err) {
				if overflowRetries >= cfg.MaxContextRetries {
					_ = l.appendTurnEnd(durableContext(ctx), lease, em, turnID, "context_overflow")
					return nil, ErrContextOverflow
				}
				overflowRetries++
				var compactErr error
				if cfg.Compactor != nil {
					_, compactErr = l.compact(ctx, lease)
				} else {
					msgs, _, projectErr := l.project(ctx)
					if projectErr != nil {
						compactErr = projectErr
					} else {
						_, compactErr = l.compactFallback(ctx, lease, msgs)
					}
				}
				if compactErr == nil && ctx.Err() == nil {
					compacted++
					stepAttempt++
					continue
				} else {
					_ = l.appendTurnEnd(durableContext(ctx), lease, em, turnID, "context_overflow")
					return nil, ErrContextOverflow
				}
			} else if errors.Is(err, ErrStopped) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				_ = l.appendTurnEnd(context.Background(), lease, em, turnID, "cancelled")
				l.finish(result, "interrupted")
				return result, ErrStopped
			} else {
				if endErr := l.appendTurnEnd(context.Background(), lease, em, turnID, "error"); endErr != nil {
					return nil, endErr
				}
				return nil, err
			}
		}
		result.Turns = 1
		result.CompactCount = compacted
		if done {
			if endErr := l.appendTurnEnd(ctx, lease, em, turnID, reason); endErr != nil {
				return nil, endErr
			}
			l.finish(result, reason)
			return result, nil
		}
	}
	if endErr := l.appendTurnEnd(ctx, lease, em, turnID, "turn_limit"); endErr != nil {
		return nil, endErr
	}
	l.finish(result, "turn_limit")
	return result, nil
}

func (l *Loop) appendStepInputs(ctx context.Context, lease session.Lease, em *emitter, turnID string, step int, inputs []StepInput) error {
	stepID := fmt.Sprintf("%s-step-%02d", turnID, step)
	for _, input := range inputs {
		if input.ID == "" || input.Text == "" {
			continue
		}
		typ := input.Type
		if typ != session.EventSteeringMessage && typ != session.EventInjectedContext {
			typ = session.EventInjectedContext
		}
		payload := session.InboxPayloadJSON(input.ID, map[session.EventType]string{session.EventSteeringMessage: "steer", session.EventInjectedContext: "inject"}[typ], input.Text)
		if err := l.append(ctx, lease, session.Event{ID: input.ID + ":claimed", SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventInboxClaimed, Data: payload}); err != nil {
			return err
		}
		if err := l.append(ctx, lease, session.Event{ID: input.ID + ":input", SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: typ, Data: session.UserTextWithInbox(input.Text, input.ID)}); err != nil {
			return err
		}
		if err := l.append(ctx, lease, session.Event{ID: input.ID + ":completed", SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventInboxCompleted, Data: payload}); err != nil {
			return err
		}
	}
	return nil
}

func stepIDFor(turnID string, step, attempt int) string {
	id := fmt.Sprintf("%s-step-%02d", turnID, step)
	if attempt > 0 {
		id += fmt.Sprintf("-retry-%02d", attempt)
	}
	return id
}

func nextStepIndex(stepID string) int {
	marker := strings.LastIndex(stepID, "-step-")
	if marker < 0 {
		return 0
	}
	var step int
	if _, err := fmt.Sscanf(stepID[marker+len("-step-"):], "%d", &step); err != nil {
		return 0
	}
	return step + 1
}

func (l *Loop) nextTurnNumber() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	number := 0
	for _, event := range l.events {
		if event.Type == session.EventTurnStart {
			number++
		}
	}
	return number
}

// finish projects the final surface into the result.
func (l *Loop) finish(result *Result, reason string) {
	msgs, _, err := l.project(context.Background())
	if err == nil {
		result.Messages = msgs
	}
	result.FinishedReason = reason
}

func (l *Loop) resolvedSystemPrompt() (string, error) {
	prompt := l.Config.SystemPrompt
	if provider, ok := l.Tools.(interface{ CodeGuidance() (string, error) }); ok {
		guidance, err := provider.CodeGuidance()
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(guidance) != "" {
			if strings.TrimSpace(prompt) != "" {
				prompt += "\n\n"
			}
			prompt += guidance
		}
	}
	return prompt, nil
}

// step runs ONE model call plus its ordered scheduled tool executions. Returns
// done=true when the assistant produced no further tool calls.
func (l *Loop) step(ctx context.Context, lease session.Lease, em *emitter, turnID string, step, attempt int, initialInput *StepInput, inputs []StepInput, compacted *int, toolCallCount *int, totalTokens *int64) (done bool, reason string, err error) {
	cfg := l.Config
	systemPrompt, guidanceErr := l.resolvedSystemPrompt()
	if guidanceErr != nil {
		return false, "", guidanceErr
	}

	stepID := stepIDFor(turnID, step, attempt)
	if err := l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, Type: session.EventStepStart, TurnID: turnID, StepID: stepID, Data: strJSON("step_start")}); err != nil {
		return false, "", err
	}
	stepEnded := false
	defer func() {
		if err != nil && !stepEnded {
			_ = l.append(durableContext(ctx), lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("error")})
		}
	}()
	if attempt == 0 {
		if initialInput != nil {
			inputID := em.id()
			if initialInput.ID != "" {
				inputID = initialInput.ID + ":input"
			}
			if err := l.append(ctx, lease, session.Event{ID: inputID, SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: initialInput.Type, Data: session.UserTextWithInbox(initialInput.Text, initialInput.ID)}); err != nil {
				return false, "", err
			}
		}
		if len(inputs) > 0 {
			if err := l.appendStepInputs(ctx, lease, em, turnID, step, inputs); err != nil {
				if l.Config.RequeueStep != nil {
					l.Config.RequeueStep(inputs)
				}
				return false, "", err
			}
		}
	}

	var resolvedCapabilities *llm.Capabilities
	if provider, ok := l.Chat.(capabilityChat); ok {
		capabilities, capabilityErr := provider.Capabilities(ctx, cfg.Model)
		if capabilityErr != nil {
			return false, "", fmt.Errorf("agent: resolve provider capabilities: %w", capabilityErr)
		}
		resolvedCapabilities = &capabilities
	}
	over, preflightErr := l.preflight(ctx, lease, compacted, resolvedCapabilities)
	if preflightErr != nil {
		return false, "", preflightErr
	}
	if over {
		return false, "", ErrContextOverflow
	}

	// Project the durable history into model messages.
	msgs, _, err := l.project(ctx)
	if err != nil {
		return false, "", err
	}

	var reqValue llm.Request
	if cfg.ContextBuilder != nil {
		snapshot, buildErr := cfg.ContextBuilder.Build(memctx.BuildInput{Model: cfg.Model, System: []memctx.Section{{Title: "system", Content: systemPrompt}}, Instructions: cfg.Instructions, ToolGuidance: cfg.ToolGuidance, Runtime: cfg.RuntimeContext, Workspace: cfg.WorkspaceContext, Tools: l.Tools.ModelTools(), Messages: msgs, MaxTokens: cfg.MaxTokens, Config: cfg.ProviderConfig, Stream: cfg.Stream, Capabilities: resolvedCapabilities, ParallelTools: true})
		if buildErr != nil {
			return false, "", buildErr
		}
		reqValue = snapshot.Request
	} else {
		reqValue = llm.Request{Model: cfg.Model, System: []llm.Message{{Role: llm.RoleSystem, Parts: []llm.Part{{Type: llm.PartText, Text: systemPrompt}}}}, Messages: deref(msgs), Tools: l.Tools.ModelTools(), MaxTokens: cfg.MaxTokens, Config: append(json.RawMessage(nil), cfg.ProviderConfig...), Stream: cfg.Stream}
	}
	req := &reqValue
	if resolvedCapabilities != nil {
		req.Capabilities = resolvedCapabilities
	}
	if req.Capabilities != nil && !req.Capabilities.SupportsStreaming {
		req.Stream = false
	}
	parallel := true
	if req.Capabilities != nil && !req.Capabilities.SupportsParallelToolCalls {
		parallel = false
	}
	req.ParallelToolCalls = &parallel
	if err := l.appendRequestHeader(ctx, lease, em, turnID, stepID, req); err != nil {
		return false, "", err
	}

	var streamedText, streamedReasoning strings.Builder
	streamedCalls := make(map[string]session.ToolCall)
	streamedCallOrder := make([]string, 0)
	streamedMedia := make([]session.MediaBlock, 0)
	// streamedParts is the provider order, unlike the legacy aggregate
	// builders above. It is the source for the final typed block assembly so a
	// reasoning/media block cannot be reordered around text or tool calls.
	streamedParts := make([]llm.Part, 0)
	streamedPartCallIndex := make(map[string]int)
	var streamedResponse *llm.Response
	var streamMu sync.Mutex
	streamCallback := func(streamCtx context.Context, event llm.StreamEvent) error {
		streamMu.Lock()
		defer streamMu.Unlock()
		if streamCtx == nil {
			streamCtx = ctx
		}
		if err := streamCtx.Err(); err != nil {
			return err
		}
		if event.Err != nil {
			return event.Err
		}
		var payload json.RawMessage
		switch event.Type {
		case llm.StreamEventText:
			streamedText.WriteString(event.Text)
			streamedParts = append(streamedParts, llm.Part{Type: llm.PartText, Text: event.Text})
			payload = session.AssistantChunkPayload("text", event.Text, "")
		case llm.StreamEventReasoning:
			streamedReasoning.WriteString(event.Reasoning)
			streamedParts = append(streamedParts, llm.Part{Type: llm.PartReasoning, Reasoning: event.Reasoning})
			payload = session.AssistantChunkPayload("reasoning", event.Reasoning, "")
		case llm.StreamEventToolCall:
			if event.ToolCall != nil {
				prior, exists := streamedCalls[event.ToolCall.CallID]
				if !exists {
					streamedCallOrder = append(streamedCallOrder, event.ToolCall.CallID)
				}
				arguments := append(json.RawMessage(nil), event.ToolCall.Arguments...)
				// Providers may emit either an accumulated argument object or
				// raw fragments. Preserve the accumulated value in both cases.
				if exists && len(prior.Arguments) > 0 && !json.Valid(arguments) {
					arguments = append(append(json.RawMessage(nil), prior.Arguments...), arguments...)
				}
				name := event.ToolCall.Name
				if name == "" {
					name = prior.Name
				}
				streamedCalls[event.ToolCall.CallID] = session.ToolCall{CallID: event.ToolCall.CallID, Name: name, Arguments: arguments}
				part := llm.Part{Type: llm.PartToolCall, ToolCall: &llm.ToolCallRequest{CallID: event.ToolCall.CallID, Name: name, Arguments: append(json.RawMessage(nil), arguments...)}}
				if index, exists := streamedPartCallIndex[event.ToolCall.CallID]; exists {
					streamedParts[index] = part
				} else {
					streamedPartCallIndex[event.ToolCall.CallID] = len(streamedParts)
					streamedParts = append(streamedParts, part)
				}
				payload = session.AssistantToolCallChunkPayload(event.ToolCall)
			}
		case llm.StreamEventMedia:
			if event.Media != nil {
				streamedMedia = append(streamedMedia, session.MediaBlock{MediaType: event.Media.MediaType, URL: event.Media.URL, Data: base64.StdEncoding.EncodeToString(event.Media.Data)})
				media := *event.Media
				media.Data = append([]byte(nil), event.Media.Data...)
				streamedParts = append(streamedParts, llm.Part{Type: llm.PartMedia, Media: &media})
			}
			payload = session.AssistantMediaChunkPayload(event.Media)
		case llm.StreamEventDone:
			streamedResponse = event.Response
			return nil
		}
		if len(payload) == 0 {
			return nil
		}
		return l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventAssistantChunk, Data: payload})
	}
	var callback llm.StreamCallback
	if req.Stream {
		callback = streamCallback
	}
	resp, err := l.Chat.Generate(ctx, req, callback)
	if err != nil {
		streamStarted := streamedText.Len() > 0 || streamedReasoning.Len() > 0 || len(streamedCalls) > 0 || len(streamedMedia) > 0
		if streamStarted {
			calls := make([]session.ToolCall, 0, len(streamedCallOrder))
			for _, callID := range streamedCallOrder {
				calls = append(calls, streamedCalls[callID])
			}
			draftData := session.AssistantContentFromParts(streamedParts, true)
			if len(streamedParts) == 0 {
				draftData = session.AssistantContentWithMedia(streamedText.String(), streamedReasoning.String(), calls, streamedMedia, true)
			}
			draft := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventAssistantMessage, Data: draftData}
			if draftErr := l.append(durableContext(ctx), lease, draft); draftErr != nil {
				return false, "", draftErr
			}
		}
		code := modelErrorCode(err)
		if appendErr := l.append(durableContext(ctx), lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventRequestError, Data: session.RequestErrorPayload(code, err.Error(), streamStarted)}); appendErr != nil {
			return false, "", appendErr
		}
		return false, "", err
	}
	if resp == nil {
		resp = streamedResponse
	}
	streamHasContent := streamedText.Len() > 0 || streamedReasoning.Len() > 0 || len(streamedMedia) > 0 || len(streamedCalls) > 0
	if resp == nil || resp.Message == nil || (len(resp.Message.Parts) == 0 && streamHasContent) {
		if streamedText.Len() > 0 || streamedReasoning.Len() > 0 || len(streamedMedia) > 0 || len(streamedCalls) > 0 {
			parts := append([]llm.Part(nil), streamedParts...)
			resp = &llm.Response{Message: &llm.Message{Role: llm.RoleAssistant, Parts: parts}, FinishReason: llm.FinishReasonStop}
		} else {
			return false, "", fmt.Errorf("agent: provider returned no message")
		}
	}

	if len(streamedMedia) == 0 {
		for _, part := range resp.Message.Parts {
			if part.Type == llm.PartMedia && part.Media != nil {
				streamedMedia = append(streamedMedia, session.MediaBlock{MediaType: part.Media.MediaType, URL: part.Media.URL, Data: base64.StdEncoding.EncodeToString(part.Media.Data)})
			}
		}
	}
	finalParts := chooseStreamParts(resp.Message.Parts, streamedParts)
	finalParts, calls, err := normalizeToolCallParts(finalParts, em)
	if err != nil {
		return false, "", err
	}
	if cfg.MaxToolCalls > 0 && *toolCallCount+len(calls) > cfg.MaxToolCalls {
		return false, "", ErrToolLimit
	}
	*toolCallCount += len(calls)

	if err := l.appendUsage(ctx, lease, em, turnID, stepID, resp); err != nil {
		return false, "", err
	}
	if resp.Usage != nil {
		*totalTokens += resp.Usage.TotalTokens()
		if cfg.MaxTotalTokens > 0 && *totalTokens > cfg.MaxTotalTokens {
			return false, "", ErrTokenLimit
		}
	}

	if len(calls) == 0 {
		// Final assistant reply; no further tools -> turn completes.
		ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID,
			Type: session.EventAssistantMessage, Data: session.AssistantContentFromPartsWithMetadata(finalParts, resp.Message.Metadata, false)}
		end := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("end")}
		if err := l.appendBatch(ctx, lease, []session.Event{ev, end}); err != nil {
			return false, "", err
		}
		stepEnded = true
		return true, string(resp.FinishReason), nil
	}

	// Persist the assistant message WITH its tool calls durably (the calls
	// must survive a crash so Recover can mark unknown outcomes rather than
	// re-execute).
	asst := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID,
		Type: session.EventAssistantMessage, Data: session.AssistantContentFromPartsWithMetadata(finalParts, resp.Message.Metadata, false)}
	if err := l.append(ctx, lease, asst); err != nil {
		return false, "", err
	}

	// Persist every call barrier before any body starts, then allow only
	// explicitly parallel-safe calls to overlap. Result commits stay in model
	// order regardless of completion order.
	batch := make([]tools.Call, 0, len(calls))
	for _, call := range calls {
		callID := call.CallID
		if callID == "" {
			callID = em.id()
		}
		callEv := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
			Type: session.EventToolCall, Data: session.ToolCallPayload(callID, call.Name, call.Arguments)}
		if err := l.append(ctx, lease, callEv); err != nil {
			return false, "", err
		}
		if err := l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: session.EventToolAdmitted, Data: strJSON("admitted")}); err != nil {
			return false, "", err
		}
		batch = append(batch, tools.Call{Name: call.Name, CallID: callID, Input: append(json.RawMessage(nil), call.Arguments...)})
	}
	ecVars := mergeVars(cfg.Vars)
	// No execution-world call has started before RunBatch. If cancellation won
	// here, settle every queued call in model order with a durable, explicit
	// ABORTED_BEFORE_DISPATCH result rather than recording a false dispatch.
	if err := ctx.Err(); err != nil {
		persistCtx := durableContext(ctx)
		for _, call := range batch {
			if appendErr := l.appendToolResultErr(persistCtx, lease, em, call.CallID, call.Name, tools.ErrAbortedBeforeDispatch, turnID, stepID); appendErr != nil {
				return false, "", appendErr
			}
		}
		if appendErr := l.append(persistCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("cancelled_before_dispatch")}); appendErr == nil {
			stepEnded = true
		}
		return false, "", ErrStopped
	}
	onDispatch := func(dispatchCtx context.Context, name, callID string) error {
		if err := l.append(dispatchCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: session.EventToolDispatched, Data: strJSON("dispatched")}); err != nil {
			return err
		}
		return l.append(dispatchCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID, Type: session.EventToolRunning, Data: strJSON("running")})
	}
	outcomes := l.Tools.RunBatch(ctx, tools.ExecContext{SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Vars: ecVars, Sandbox: cfg.Sandbox, Artifacts: cfg.Artifacts, Lease: &lease, OnDispatch: onDispatch}, batch)
	if err := validateToolOutcomes(batch, outcomes); err != nil {
		return false, "", err
	}
	// Tool bodies may have started before the caller cancelled the run. Their
	// terminal outcomes are part of the durable recovery boundary and therefore
	// must be committed with a context that is no longer cancelled; otherwise a
	// started side effect would be left as an indistinguishable dangling call.
	persistCtx := durableContext(ctx)
	concludesTurn := false
	for _, outcome := range outcomes {
		concludes, appendErr := l.appendToolOutcome(persistCtx, lease, em, outcome, turnID, stepID)
		if appendErr != nil {
			return false, "", appendErr
		}
		concludesTurn = concludesTurn || concludes
	}
	if err := l.append(persistCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("tools_done")}); err != nil {
		return false, "", err
	}
	stepEnded = true
	if ctx.Err() != nil {
		return false, "", ErrStopped
	}
	if concludesTurn {
		return true, "tool_concluded", nil
	}
	// Turn continues (another model call next iteration).
	return false, "tool_calls", nil
}

func (l *Loop) appendToolOutcome(ctx context.Context, lease session.Lease, em *emitter, outcome tools.Outcome, turnID, stepID string) (bool, error) {
	if outcome.Err != nil {
		if outcome.Result != nil {
			if err := l.appendToolFailureResult(ctx, lease, em, outcome.Result, outcome.Err, turnID, stepID); err != nil {
				return false, err
			}
			return false, nil
		}
		if err := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, outcome.Err, turnID, stepID); err != nil {
			return false, err
		}
		return false, nil
	}
	if outcome.Result == nil {
		err := fmt.Errorf("tool %s returned no result", outcome.Call.Name)
		if appendErr := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, err, turnID, stepID); appendErr != nil {
			return false, appendErr
		}
		return false, nil
	}
	canonicalEncoded, err := json.Marshal(outcome.Result.Canonical)
	if err != nil {
		if appendErr := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, fmt.Errorf("encode tool content: %w", err), turnID, stepID); appendErr != nil {
			return false, appendErr
		}
		return false, nil
	}
	modelFacing := any(outcome.Result.Canonical)
	if outcome.Result.ModelFacing != nil {
		modelFacing = outcome.Result.ModelFacing
	}
	contentEncoded, err := json.Marshal(modelFacing)
	if err != nil {
		if appendErr := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, fmt.Errorf("encode tool content: %w", err), turnID, stepID); appendErr != nil {
			return false, appendErr
		}
		return false, nil
	}
	if err := l.appendToolResult(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, canonicalEncoded, contentEncoded, outcome.Result.UI, outcome.Result.Code, outcome.Result.Content, outcome.Result.AdditionalContexts, outcome.Result.ConcludesTurn, turnID, stepID); err != nil {
		return false, err
	}
	return outcome.Result.ConcludesTurn, nil
}

func (l *Loop) appendToolFailureResult(ctx context.Context, lease session.Lease, em *emitter, result *tools.Result, fallback error, turnID, stepID string) error {
	code, recovery := toolFailure(fallback)
	if result.Code != "" {
		code = result.Code
	}
	if result.Failure != nil && result.Failure.Message != "" {
		recovery = result.Failure.Message
	}
	modelFacing := result.ModelFacing
	if modelFacing == nil {
		modelFacing = map[string]any{"error": map[string]any{"code": code, "message": recovery}}
	}
	output, err := json.Marshal(modelFacing)
	if err != nil {
		return fmt.Errorf("encode tool failure output: %w", err)
	}
	content := output
	if len(result.Content) > 0 {
		content, err = json.Marshal(result.Content)
		if err != nil {
			return fmt.Errorf("encode tool failure content: %w", err)
		}
	}
	data := session.ToolResultStructuredPayloadWithOutput(result.CallID, result.Name, output, content, result.UI, code, true, result.Content, result.AdditionalContexts, false)
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: result.CallID,
		Type: session.EventToolResult, Data: data, SourceSeqs: l.assistantSourceSeqs(result.CallID, turnID, stepID)}
	return l.append(ctx, lease, ev)
}

func validateToolOutcomes(calls []tools.Call, outcomes []tools.Outcome) error {
	if len(outcomes) != len(calls) {
		return fmt.Errorf("agent: tool runtime returned %d outcomes for %d calls", len(outcomes), len(calls))
	}
	for index := range calls {
		if outcomes[index].Call.CallID != calls[index].CallID || outcomes[index].Call.Name != calls[index].Name {
			return fmt.Errorf("agent: tool runtime reordered or lost call %q at outcome %d", calls[index].CallID, index)
		}
	}
	return nil
}

func (l *Loop) appendToolResult(ctx context.Context, lease session.Lease, em *emitter, callID, name string, output, content json.RawMessage, meta map[string]any, code string, blocks []session.ContentBlock, contexts []llm.Message, concludesTurn bool, turnID, stepID string) error {
	data := session.ToolResultStructuredPayloadWithOutput(callID, name, output, content, meta, code, false, blocks, contexts, concludesTurn)
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
		Type: session.EventToolResult, Data: data, SourceSeqs: l.assistantSourceSeqs(callID, turnID, stepID)}
	return l.append(ctx, lease, ev)
}

func (l *Loop) appendToolResultErr(ctx context.Context, lease session.Lease, em *emitter, callID, name string, execErr error, turnID, stepID string) error {
	code, recovery := toolFailure(execErr)
	data, _ := json.Marshal(map[string]any{
		"call_id": callID, "name": name, "is_error": true,
		"error":   map[string]any{"code": code, "message": recovery},
		"output":  map[string]any{"error": map[string]any{"code": code, "message": recovery}},
		"content": map[string]any{"error": map[string]any{"code": code, "message": recovery}},
		"code":    code,
	})
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
		Type: session.EventToolResult, Data: data, SourceSeqs: l.assistantSourceSeqs(callID, turnID, stepID)}
	return l.append(ctx, lease, ev)
}

func (l *Loop) assistantSourceSeqs(callID, turnID, stepID string) []uint64 {
	for _, event := range l.snapshotEvents() {
		if event.Type != session.EventAssistantMessage || event.TurnID != turnID || event.StepID != stepID {
			continue
		}
		var payload struct {
			ToolCalls []session.ToolCall `json:"tool_calls"`
		}
		if json.Unmarshal(event.Data, &payload) != nil {
			continue
		}
		for _, call := range payload.ToolCalls {
			if call.CallID == callID {
				return []uint64{event.Seq}
			}
		}
	}
	return nil
}

func toolFailure(err error) (code, recovery string) {
	var denial *integration.Denial
	if errors.As(err, &denial) {
		code := denial.Code
		if code == "" {
			code = "SANDBOX_DENIED"
		}
		return code, denial.Reason
	}
	switch {
	case errors.Is(err, integration.ErrStaleVersion):
		return "FS_STALE_VERSION", "The file changed after it was read. Read it again before editing."
	case errors.Is(err, integration.ErrNotObserved):
		return "FS_NOT_OBSERVED", "Read the file before editing it."
	case errors.Is(err, integration.ErrAlreadyExists):
		return "FS_ALREADY_EXISTS", "The target file already exists."
	case errors.Is(err, integration.ErrAmbiguousMatch):
		return "FS_AMBIGUOUS_MATCH", "The edit matched multiple locations; narrow the old text or use replace_all."
	case errors.Is(err, integration.ErrTargetOutsideRoot):
		return "SANDBOX_DENIED_PATH", "The requested path is outside the sandbox."
	case errors.Is(err, tools.ErrToolNotFound):
		return "UNKNOWN_TOOL", "The requested tool is unavailable in this session. Use one of the visible tools."
	case errors.Is(err, tools.ErrInvalidArguments):
		return "INVALID_ARGS", "The tool arguments do not match its schema. Correct them and retry."
	case errors.Is(err, tools.ErrInvalidOutput):
		return "INVALID_TOOL_OUTPUT", "The tool returned an invalid result. Retry with a narrower request or another tool."
	case errors.Is(err, tools.ErrToolTimeout):
		return "TOOL_TIMEOUT", "The tool timed out. Narrow the request or retry."
	case errors.Is(err, tools.ErrAbortedBeforeDispatch):
		return "ABORTED_BEFORE_DISPATCH", "The run was cancelled before this tool started."
	case errors.Is(err, tools.ErrPolicyDenied):
		return "POLICY_DENIED", "The tool call was denied by runtime policy."
	case errors.Is(err, tools.ErrApprovalDenied):
		return "APPROVAL_DENIED", "The tool call was not approved."
	case errors.Is(err, context.Canceled):
		return "ABORTED", "The tool was cancelled before it completed."
	case errors.Is(err, context.DeadlineExceeded):
		return "ABORTED", "The tool was cancelled before it completed."
	default:
		return "TOOL_FAILED", "The tool failed; retry or choose another tool."
	}
}

func modelErrorCode(err error) string {
	var providerErr *llm.Error
	if errors.As(err, &providerErr) {
		if providerErr.Code != "" {
			return providerErr.Code
		}
		switch providerErr.Kind {
		case llm.ErrorKindContextOverflow:
			return "CONTEXT_OVERFLOW"
		case llm.ErrorKindAuth:
			return "PROVIDER_AUTH"
		case llm.ErrorKindRateLimit:
			return "PROVIDER_RATE_LIMIT"
		case llm.ErrorKindNetwork:
			return "PROVIDER_NETWORK"
		}
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return "MODEL_ABORTED"
	}
	return "MODEL_REQUEST_FAILED"
}

// leaseHeartbeat renews the standalone FencedStore lease while model and tool
// work is in flight. Host adapters use their own worker heartbeat and must not
// call Run; they inject cancellation through their claimed-run entrypoint.
func (l *Loop) leaseHeartbeat(ctx context.Context, lease session.Lease) (context.Context, func() error) {
	loopCtx, cancel := context.WithCancel(ctx)
	interval := l.Config.LeaseTTL / 3
	if interval <= 0 {
		interval = 10 * time.Second
	}
	if interval > 10*time.Second {
		interval = 10 * time.Second
	}
	var mu sync.Mutex
	var renewalErr error
	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		current := lease
		for {
			select {
			case <-done:
				return
			case <-loopCtx.Done():
				return
			case <-ticker.C:
				renewed, err := l.Store.RenewLease(context.Background(), current)
				if err != nil {
					mu.Lock()
					renewalErr = err
					mu.Unlock()
					cancel()
					return
				}
				current = renewed
			}
		}
	}()
	var once sync.Once
	return loopCtx, func() error {
		once.Do(func() { close(done); cancel(); <-stopped })
		mu.Lock()
		defer mu.Unlock()
		return renewalErr
	}
}

func (l *Loop) appendRequestHeader(ctx context.Context, lease session.Lease, em *emitter, turnID, stepID string, req *llm.Request) error {
	toolSchemas := make([]string, 0, len(req.Tools))
	for _, tool := range req.Tools {
		encoded, err := json.Marshal(tool)
		if err != nil {
			return fmt.Errorf("encode tool schema: %w", err)
		}
		toolSchemas = append(toolSchemas, string(encoded))
	}
	requestBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("encode request header: %w", err)
	}
	configHash := fmt.Sprintf("%x", sha256.Sum256(req.Config))
	requestSnapshot := requestBytes
	if snapshotter, ok := l.Chat.(requestSnapshotChat); ok {
		requestSnapshot, err = snapshotter.RequestSnapshot(req)
		if err != nil {
			return fmt.Errorf("resolve provider request snapshot: %w", err)
		}
	}
	requestHash := fmt.Sprintf("%x", sha256.Sum256(requestSnapshot))
	systemSections := make([]string, 0, len(req.System))
	for _, message := range req.System {
		for _, part := range message.Parts {
			if part.Type == llm.PartText {
				systemSections = append(systemSections, part.Text)
			}
		}
	}
	if req.Capabilities != nil {
		data := session.RequestHeaderPayloadWithSnapshot(req.Model, l.Chat.Name(), systemSections, toolSchemas, configHash, requestHash, []llm.Capabilities{*req.Capabilities}, requestSnapshot)
		return l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventRequestHeader, Data: data})
	}
	data := session.RequestHeaderPayloadWithSnapshot(req.Model, l.Chat.Name(), systemSections, toolSchemas, configHash, requestHash, nil, requestSnapshot)
	return l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventRequestHeader, Data: data})
}

func (l *Loop) appendUsage(ctx context.Context, lease session.Lease, em *emitter, turnID, stepID string, resp *llm.Response) error {
	if resp.Usage == nil {
		return nil
	}
	data := mustMarshal(map[string]any{
		"input_tokens": resp.Usage.InputTokens, "output_tokens": resp.Usage.OutputTokens,
		"total_tokens": resp.Usage.TotalTokens(),
	})
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventUsage, Data: data}
	return l.append(ctx, lease, ev)
}

func (l *Loop) appendTurnEnd(ctx context.Context, lease session.Lease, em *emitter, turnID, reason string) error {
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, Type: session.EventTurnEnd, Data: mustMarshal(map[string]any{"reason": reason})}
	return l.append(ctx, lease, ev)
}

// preflight measures pressure; if over threshold it prunes or compacts so the
// surface advances. Returns true when still over capacity after the attempt.
func (l *Loop) preflight(ctx context.Context, lease session.Lease, compacted *int, capabilities *llm.Capabilities) (bool, error) {
	cfg := l.Config
	msgs, _, err := l.project(ctx)
	if err != nil {
		return false, err
	}
	systemPrompt, promptErr := l.resolvedSystemPrompt()
	if promptErr != nil {
		return false, promptErr
	}
	var usage int64
	if cfg.ContextBuilder != nil {
		snapshot, buildErr := cfg.ContextBuilder.Build(memctx.BuildInput{
			Model: cfg.Model, System: []memctx.Section{{Title: "system", Content: systemPrompt}},
			Instructions: cfg.Instructions, ToolGuidance: cfg.ToolGuidance, Runtime: cfg.RuntimeContext,
			Workspace: cfg.WorkspaceContext, Tools: l.Tools.ModelTools(), Messages: msgs,
			MaxTokens: cfg.MaxTokens, Config: cfg.ProviderConfig, Stream: cfg.Stream,
			Capabilities: capabilities, ParallelTools: true,
		})
		if buildErr != nil {
			return false, buildErr
		}
		usage = snapshot.TokenEstimate
	} else {
		usage = estimateTokens(msgs) + estimateTokens([]*llm.Message{{Parts: []llm.Part{{Type: llm.PartText, Text: systemPrompt}}}}) + estimateToolTokens(l.Tools.ModelTools())
	}
	contextWindow, maxOutput := cfg.ContextWindow, cfg.MaxOutput
	if capabilities != nil {
		if capabilities.ContextWindow > 0 {
			contextWindow = capabilities.ContextWindow
		}
		if capabilities.MaxOutput > 0 {
			maxOutput = capabilities.MaxOutput
		}
	}
	window := contextWindow - maxOutput
	threshold := int64(float64(window) * cfg.CompactThresholdRatio)
	if threshold < 1 {
		threshold = 1
	}
	if window <= 0 || usage < threshold {
		return false, nil
	}

	// Over threshold: one max-safe compaction when a compactor exists.
	if cfg.Compactor != nil && l.eventCount() > 0 {
		if _, cerr := l.compact(ctx, lease); cerr == nil {
			*compacted++
			return false, nil
		}
	}
	// Without an LLM compactor, still advance the durable surface: summarize a
	// deterministic bounded projection and retain the mandatory recent tail.
	if l.eventCount() > cfg.RetainTailEvents {
		if _, cerr := l.compactFallback(ctx, lease, msgs); cerr == nil {
			*compacted++
			return false, nil
		}
	}
	return true, nil
}

func (l *Loop) compactFallback(ctx context.Context, lease session.Lease, msgs []*llm.Message) (uint64, error) {
	events := l.snapshotEvents()
	generation := l.nextGeneration(ctx)
	shadowed := shadowedSeqsRetainingTail(events, l.Config.RetainTailEvents)
	if len(shadowed) == 0 {
		return 0, fmt.Errorf("agent: no compactable surface before retained tail")
	}
	pruned := make([]*llm.Message, 0, len(msgs))
	for _, message := range msgs {
		if message != nil {
			pruned = append(pruned, message.Clone())
		}
	}
	caps := memctx.DefaultPruneCaps()
	if l.Config.PruneHead > 0 {
		caps.HeadChars = l.Config.PruneHead
	}
	if l.Config.PruneTail > 0 {
		caps.TailChars = l.Config.PruneTail
	}
	caps.PruneMessages(pruned)
	var summaryBuilder strings.Builder
	for _, message := range pruned {
		encoded, _ := json.Marshal(message)
		if summaryBuilder.Len()+len(encoded)+1 > 12000 {
			break
		}
		summaryBuilder.Write(encoded)
		summaryBuilder.WriteByte('\n')
	}
	summary := summaryBuilder.String()
	if summary == "" {
		summary = "Earlier context was compacted; the recent tail is retained verbatim."
	}
	fingerprint := fmt.Sprintf("%x", sha256.Sum256([]byte(summary)))
	throughSeq := shadowed[len(shadowed)-1]
	transactionID := fmt.Sprintf("compact-%d", generation)
	start := session.Event{ID: l.emitterIDFor("compact-start", generation), SessionID: l.SessionID, Type: session.EventCompactionStart, Data: session.CompactionStartPayload(generation, transactionID, shadowed)}
	sum := session.Event{ID: l.emitterIDFor("compact-summary", generation), SessionID: l.SessionID, Type: session.EventCompactionSummary, Data: session.CompactionSummaryPayload(generation, transactionID, throughSeq, shadowed, summary, fingerprint), SourceSeqs: append([]uint64(nil), shadowed...)}
	end := session.Event{ID: l.emitterIDFor("compact-end", generation), SessionID: l.SessionID, Type: session.EventCompactionEnd, Data: session.CompactionEndPayload(generation, transactionID)}
	if err := l.appendBatch(ctx, lease, []session.Event{start, sum, end}); err != nil {
		return 0, err
	}
	if err := l.Store.SaveCompactionCheckpoint(ctx, lease, session.CompactionCheckpoint{SessionID: l.SessionID, Generation: generation, TransactionID: transactionID, ThroughSeq: throughSeq, ShadowedSeqs: shadowed, Summary: summary, SummarySHA256: fingerprint}); err != nil {
		return 0, err
	}
	return l.cachedLast(), nil
}

// compact performs one durable compaction: ask the Compactor to summarize the
// log, then append compaction/start|summary|end and record the checkpoint.
// Raw history is never deleted (H-COMPACT-001).
func (l *Loop) compact(ctx context.Context, lease session.Lease) (uint64, error) {
	if l.Config.Compactor == nil {
		return 0, fmt.Errorf("agent: compaction not configured")
	}
	events := l.snapshotEvents()
	generation := l.nextGeneration(ctx)
	shadowed := shadowedSeqsRetainingTail(events, l.Config.RetainTailEvents)
	if len(shadowed) == 0 {
		return 0, fmt.Errorf("agent: no compactable surface before retained tail")
	}
	throughSeq := shadowed[len(shadowed)-1]
	region := make([]session.Event, 0, len(events))
	for _, event := range events {
		if event.Seq <= throughSeq {
			region = append(region, event)
		}
	}

	summary, fingerprint, err := l.Config.Compactor.Compact(ctx, generation, region, throughSeq)
	if err != nil {
		return 0, err
	}
	transactionID := fmt.Sprintf("compact-%d", generation)
	start := session.Event{ID: l.emitterIDFor("compact-start", generation), SessionID: l.SessionID, Type: session.EventCompactionStart, Data: session.CompactionStartPayload(generation, transactionID, shadowed)}
	sum := session.Event{ID: l.emitterIDFor("compact-summary", generation), SessionID: l.SessionID, Type: session.EventCompactionSummary,
		Data: session.CompactionSummaryPayload(generation, transactionID, throughSeq, shadowed, summary, fingerprint), SourceSeqs: append([]uint64(nil), shadowed...)}
	end := session.Event{ID: l.emitterIDFor("compact-end", generation), SessionID: l.SessionID, Type: session.EventCompactionEnd, Data: session.CompactionEndPayload(generation, transactionID)}

	if err := l.appendBatch(ctx, lease, []session.Event{start, sum, end}); err != nil {
		return 0, err
	}
	if err := l.Store.SaveCompactionCheckpoint(ctx, lease, session.CompactionCheckpoint{
		SessionID: l.SessionID, Generation: generation, TransactionID: transactionID,
		ThroughSeq: throughSeq, ShadowedSeqs: shadowed, Summary: summary, SummarySHA256: fingerprint,
	}); err != nil {
		return 0, err
	}
	return l.cachedLast(), nil
}

func (l *Loop) append(ctx context.Context, lease session.Lease, ev session.Event) error {
	return l.appendBatch(ctx, lease, []session.Event{ev})
}

// appendBatch persists events fenced + idempotent, then re-reads the tail to
// fold canonical seqs into the in-memory cache.
func (l *Loop) appendBatch(ctx context.Context, lease session.Lease, events []session.Event) error {
	if lease.Token == "" {
		return ErrLeaseLost
	}
	batchInput := make([]session.Event, len(events))
	for i := range events {
		batchInput[i] = events[i].Clone()
		batchInput[i].Normalize()
	}
	var committed []session.Event
	if batchStore, ok := l.Store.(session.FencedBatchStore); ok {
		batch, err := batchStore.AppendFencedCommitted(ctx, lease, batchInput)
		if err != nil {
			if errors.Is(err, session.ErrLeaseLost) {
				return ErrLeaseLost
			}
			return err
		}
		committed = batch.Events
	} else if _, err := l.Store.AppendFenced(ctx, lease, batchInput); err != nil {
		if errors.Is(err, session.ErrLeaseLost) {
			return ErrLeaseLost
		}
		return err
	}
	// Fold the freshly-persisted canonical batch into the cache. Legacy stores
	// are supported by a tail read, but failure is returned rather than silently
	// losing the publication.
	if committed == nil {
		after := l.cachedLast()
		loaded, err := l.Store.Load(ctx, l.SessionID, after, 0)
		if err != nil {
			return fmt.Errorf("agent: load committed event tail: %w", err)
		}
		committed = loaded
	}
	if len(committed) > 0 {
		l.mu.Lock()
		after := uint64(0)
		if len(l.events) > 0 {
			after = l.events[len(l.events)-1].Seq
		}
		for _, e := range committed {
			if e.Seq <= after {
				continue
			}
			l.events = append(l.events, e.Clone())
		}
		l.mu.Unlock()
		if l.Config.EventHub != nil {
			for _, e := range committed {
				if e.Seq > after {
					l.Config.EventHub.Publish(e)
				}
			}
		}
	}
	return nil
}

// refresh rebuilds the in-memory tail from the log.
func (l *Loop) refresh(ctx context.Context) error {
	events, err := l.Store.Load(ctx, l.SessionID, 0, 0)
	if err != nil {
		return err
	}
	for _, event := range events {
		if err := event.Validate(); err != nil {
			return fmt.Errorf("agent: invalid durable event: %w", err)
		}
	}
	l.mu.Lock()
	cached := make([]session.Event, len(events))
	for i, e := range events {
		cached[i] = e.Clone()
	}
	l.events = cached
	l.mu.Unlock()
	return nil
}

func (l *Loop) snapshotEvents() []session.Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]session.Event, len(l.events))
	copy(out, l.events)
	return out
}

func (l *Loop) eventCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.events)
}

func (l *Loop) cachedLast() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.events) == 0 {
		return 0
	}
	return l.events[len(l.events)-1].Seq
}

func (l *Loop) nextGeneration(ctx context.Context) uint64 {
	_, proj, err := l.project(ctx)
	if err == nil && proj != nil {
		return proj.Generation + 1
	}
	return 1
}

func (l *Loop) project(ctx context.Context) ([]*llm.Message, *session.Projection, error) {
	events := l.snapshotEvents()
	return session.NewSurface(session.SurfaceSpec{}).Project(events)
}

// emitterIDFor builds a stable event id for a code-known event.
func (l *Loop) emitterIDFor(kind string, n uint64) string {
	return fmt.Sprintf("ev:%s:%s:%s:%d", l.SessionID, l.Config.Owner, kind, n)
}

// ---------------------------------------------------------------------------
// helpers

func strJSON(s string) json.RawMessage {
	return mustMarshal(map[string]any{"text": s})
}

func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("agent: marshal: %v", err))
	}
	return b
}

// deref converts a slice of *llm.Message to a slice of values for the
// provider request (Request.Messages is []Message).
func deref(msgs []*llm.Message) []llm.Message {
	out := make([]llm.Message, 0, len(msgs))
	for _, m := range msgs {
		if m != nil {
			out = append(out, *m)
		}
	}
	return out
}

func isOverflow(err error) bool { return llm.IsContextOverflow(err) }

// durableContext preserves terminal recovery records when a provider returns
// after the caller has cancelled its request context. Store implementations
// remain responsible for their own bounded I/O timeouts.
func durableContext(ctx context.Context) context.Context {
	return context.WithoutCancel(ctx)
}

func estimateTokens(msgs []*llm.Message) int64 {
	var total int64
	for _, m := range msgs {
		if m == nil {
			continue
		}
		for _, part := range m.Parts {
			// Count the serialized provider-neutral part rather than only text
			// fields. Tool results and media can dominate context pressure and
			// must trigger pruning/compaction before the provider rejects them.
			encoded, err := json.Marshal(part)
			if err != nil {
				continue
			}
			tokens := int64((len(encoded) + 3) / 4)
			if tokens < 1 {
				tokens = 1
			}
			total += tokens
		}
	}
	return total
}

func estimateToolTokens(tools []*llm.ToolDefinition) int64 {
	encoded, err := json.Marshal(tools)
	if err != nil {
		return 0
	}
	return int64((len(encoded) + 3) / 4)
}

func chooseStreamParts(response, streamed []llm.Part) []llm.Part {
	if len(streamed) == 0 {
		return response
	}
	for _, part := range streamed {
		if part.Type != llm.PartToolCall || part.ToolCall == nil {
			continue
		}
		if part.ToolCall.CallID == "" || part.ToolCall.Name == "" || !json.Valid(part.ToolCall.Arguments) {
			return response
		}
	}
	return streamed
}

func normalizeToolCallParts(parts []llm.Part, em *emitter) ([]llm.Part, []llm.ToolCallRequest, error) {
	normalized := make([]llm.Part, len(parts))
	copy(normalized, parts)
	calls := make([]llm.ToolCallRequest, 0)
	seen := make(map[string]struct{})
	for index := range normalized {
		part := normalized[index]
		if part.Type != llm.PartToolCall || part.ToolCall == nil {
			continue
		}
		call := *part.ToolCall
		if call.CallID == "" {
			call.CallID = em.id()
		}
		if _, exists := seen[call.CallID]; exists {
			return nil, nil, fmt.Errorf("agent: provider returned duplicate tool call id %q", call.CallID)
		}
		seen[call.CallID] = struct{}{}
		call.Arguments = append(json.RawMessage(nil), part.ToolCall.Arguments...)
		normalized[index].ToolCall = &call
		calls = append(calls, call)
	}
	return normalized, calls, nil
}

func shadowedSeqsRetainingTail(events []session.Event, retain int) []uint64 {
	surface := make([]session.Event, 0, len(events))
	for _, event := range events {
		if event.Type.Surface() {
			surface = append(surface, event)
		}
	}
	// A fresh request must remain usable even when history is short: preserve
	// the configured tail where possible, but always leave at least one newest
	// surface event verbatim rather than refusing every compaction.
	if retain >= len(surface) {
		retain = len(surface) - 1
	}
	if retain < 1 || len(surface) <= retain {
		return nil
	}
	cut := len(surface) - retain
	// Never split a visible assistant/tool-result pair at the compaction
	// boundary. Expand the verbatim tail backwards until every retained tool
	// result has its originating assistant message and every retained assistant
	// tool call has its result (when one already exists in the log).
	for changed := true; changed; {
		changed = false
		retained := surface[cut:]
		retainedSet := make(map[uint64]bool, len(retained))
		for _, event := range retained {
			retainedSet[event.Seq] = true
		}
		for _, event := range retained {
			if event.Type == session.EventToolResult {
				for _, source := range event.SourceSeqs {
					if index := surfaceIndex(surface, source); index >= 0 && index < cut {
						cut, changed = index, true
					}
				}
			}
			if event.Type == session.EventAssistantMessage {
				var payload struct {
					ToolCalls []session.ToolCall `json:"tool_calls"`
				}
				if json.Unmarshal(event.Data, &payload) == nil {
					for _, call := range payload.ToolCalls {
						for _, candidate := range surface {
							if candidate.Type == session.EventToolResult && candidate.CallID == call.CallID && !retainedSet[candidate.Seq] {
								if index := surfaceIndex(surface, candidate.Seq); index >= 0 && index < cut {
									cut, changed = index, true
								}
							}
						}
					}
				}
			}
		}
	}
	out := make([]uint64, 0, cut)
	for _, event := range surface[:cut] {
		out = append(out, event.Seq)
	}
	return out
}

func surfaceIndex(events []session.Event, seq uint64) int {
	for i, event := range events {
		if event.Seq == seq {
			return i
		}
	}
	return -1
}

// mergeVars returns a copy of the configured Vars (never mutates the source).
func mergeVars(src map[string]any) map[string]any {
	out := make(map[string]any, len(src)+1)
	for k, v := range src {
		out[k] = v
	}
	return out
}

type emitter struct {
	sessionID string
	runID     string
	mu        sync.Mutex
	n         int
}

func (e *emitter) next(typ session.EventType, payload map[string]any) session.Event {
	data, _ := json.Marshal(payload)
	return session.Event{ID: e.id(), SessionID: e.sessionID, RunID: e.runID, Type: typ, Data: data}
}

func (e *emitter) id() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.n++
	return fmt.Sprintf("ev:%s:%s:%d", e.sessionID, e.runID, e.n)
}

func newRunID() string {
	n := atomic.AddUint64(&runCounter, 1)
	return fmt.Sprintf("run-%d-%d", n, time.Now().UnixNano())
}

var runCounter uint64
