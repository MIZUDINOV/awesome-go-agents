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
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	memctx "github.com/MIZUDINOV/awesome-go-agents/context"
	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/skill"
	"github.com/MIZUDINOV/awesome-go-agents/skilltool"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

const CurrentSkillSchemaVersion = "agentkit-skills-v2"

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
	ErrStopped                  = errors.New("agent: run stopped (cancelled)")
	ErrToolLimit                = errors.New("agent: tool-call limit exceeded")
	ErrTokenLimit               = errors.New("agent: token limit exceeded")
	ErrToolDeferred             = errors.New("agent: tool execution deferred")
	ErrDeferredExecutionStarted = errors.New("agent: deferred tool execution already started")
)

type SkillInvocationError struct {
	Code string
	Name string
	Err  error
}

func (e *SkillInvocationError) Error() string {
	if e == nil {
		return "agent: skill invocation failed"
	}
	return fmt.Sprintf("agent: skill invocation %q failed: %v", e.Name, e.Err)
}

func (e *SkillInvocationError) Unwrap() error { return e.Err }

// Chat is the provider-neutral model seam.
type Chat interface {
	Generate(ctx context.Context, req *llm.Request, cb llm.StreamCallback) (*llm.Response, error)
	Name() string
}

type toolContinuation uint8

const (
	toolContinue toolContinuation = iota
	toolConclude
	toolDeferred
)

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

// Compactor produces the durable summary + fingerprint over the exact source
// events selected for replacement. The list is intentionally not a prefix:
// compaction must never shadow unrelated events that happen to fall between
// two selected sequences.
type Compactor interface {
	Compact(ctx context.Context, generation uint64, events []session.Event, sourceSeqs []uint64) (summary, fingerprint string, err error)
}

// Config controls the loop.
type Config struct {
	Model         string
	Owner         string
	SystemPrompt  string
	PromptVersion string

	MaxStepsPerTurn   int
	MaxToolCalls      int
	MaxTokens         int64
	MaxTotalTokens    int64
	MaxWallTime       time.Duration
	MaxContextRetries int

	Stream bool
	// ToolChoice controls model tool selection for this loop. Empty preserves
	// the provider's default automatic selection.
	ToolChoice llm.ToolChoice

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
	Policy           RunPolicy

	// Vars are merged into every tool ToolRunContext (host bindings such as the
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
	// Skills is an optional run-scoped runtime. Pinned runtimes preserve the
	// exact catalog and bodies selected by the host at run creation.
	Skills *skill.Runtime
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

// ToolResume completes a deferred tool execution. Result is validated against
// the durable call identity before it can be appended to the session.
type ToolResume struct {
	CallID    string
	ResumeKey string
	Result    *tools.Result
	Err       error
}

// DeferredToolResume asks AgentKit to execute the original deferred tool
// after its host-owned prerequisite is ready. The original call and argument
// bytes are loaded from the durable event log; callers cannot substitute a
// different tool body or input during continuation.
type DeferredToolResume struct {
	CallID    string `json:"call_id"`
	ResumeKey string `json:"resume_key"`
}

// ResumeTool appends the result of a deferred tool and resumes the same model
// turn. The host may supply ClaimedLease; otherwise the loop owns a temporary
// lease exactly as it does for RunInputWithID.
func (l *Loop) ResumeTool(ctx context.Context, request ToolResume) error {
	return l.resumeTool(ctx, request, false)
}

// ResumeMaterializedTool resumes a deferred call whose host-owned result is
// already durable. The separate method preserves ToolResume source shape for
// callers using legacy unkeyed literals.
func (l *Loop) ResumeMaterializedTool(ctx context.Context, request ToolResume) error {
	return l.ResumeMaterializedTools(ctx, []ToolResume{request})
}

// ResumeMaterializedTools appends host-owned outcomes for every deferred call
// in one model tool step before continuing the turn. A shared prerequisite may
// unblock or fail several calls together; exposing a partial result batch to a
// provider would break its tool-call/result pairing contract.
func (l *Loop) ResumeMaterializedTools(ctx context.Context, requests []ToolResume) error {
	resumes := make([]deferredToolBatchResume, 0, len(requests))
	for _, request := range requests {
		if request.CallID == "" || request.ResumeKey == "" || request.Result == nil {
			return fmt.Errorf("agent: materialized deferred tool call, resume key, and result are required")
		}
		if request.Result.CallID != "" && request.Result.CallID != request.CallID {
			return fmt.Errorf("agent: deferred tool result call id mismatch")
		}
		resumes = append(resumes, deferredToolBatchResume{
			resume:       DeferredToolResume{CallID: request.CallID, ResumeKey: request.ResumeKey},
			result:       request.Result,
			err:          request.Err,
			materialized: true,
		})
	}
	return l.resumeDeferredToolBatch(ctx, resumes)
}

type deferredToolBatchResume struct {
	resume       DeferredToolResume
	result       *tools.Result
	err          error
	materialized bool
}

func (l *Loop) resumeTool(ctx context.Context, request ToolResume, materialized bool) error {
	if request.CallID == "" || request.ResumeKey == "" {
		return fmt.Errorf("agent: deferred tool call and resume key are required")
	}
	if materialized && request.Result == nil {
		return fmt.Errorf("agent: materialized deferred result is required")
	}
	if request.Result != nil && request.Result.CallID != "" && request.Result.CallID != request.CallID {
		return fmt.Errorf("agent: deferred tool result call id mismatch")
	}
	var lease session.Lease
	externalLease := l.Config.ClaimedLease != nil
	stopHeartbeat := func() error { return nil }
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
		var loopCtx context.Context
		loopCtx, stopHeartbeat = l.leaseHeartbeat(ctx, lease)
		ctx = loopCtx
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		defer func() { _ = l.Store.ReleaseLease(cleanupCtx, lease) }()
		defer func() { _ = stopHeartbeat() }()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := l.recoverSession(ctx, lease); err != nil {
		return err
	}
	if err := l.refresh(ctx); err != nil {
		return err
	}
	var deferred session.Event
	var toolName string
	resumeStarted := false
	resumeMaterialized := false
	var resultEvent session.Event
	resultExists, stepEnded, turnEnded := false, false, false
	for _, event := range l.snapshotEvents() {
		if event.Type != session.EventToolDeferred || event.CallID != request.CallID {
			if event.Type == session.EventToolResumeStarted && event.CallID == request.CallID {
				resumeStarted = true
				var payload struct {
					Materialized bool `json:"materialized"`
				}
				if json.Unmarshal(event.Data, &payload) == nil {
					resumeMaterialized = payload.Materialized
				}
			}
			if event.Type == session.EventToolResult && event.CallID == request.CallID {
				resultEvent = event
				resultExists = true
			}
			continue
		}
		var payload struct {
			Name      string `json:"name"`
			ResumeKey string `json:"resume_key"`
		}
		if json.Unmarshal(event.Data, &payload) == nil && payload.ResumeKey == request.ResumeKey {
			deferred = event
			toolName = payload.Name
		}
	}
	if deferred.Type != session.EventToolDeferred {
		return fmt.Errorf("agent: deferred tool %s was not found", request.CallID)
	}
	if toolName == "" {
		return fmt.Errorf("agent: deferred tool %s has no durable tool name", request.CallID)
	}
	if err := validateToolResultIdentity(request.Result, request.CallID, toolName); err != nil {
		return err
	}
	for _, event := range l.snapshotEvents() {
		if event.Type == session.EventStepEnd && event.TurnID == deferred.TurnID && event.StepID == deferred.StepID {
			stepEnded = true
		}
		if event.Type == session.EventTurnEnd && event.TurnID == deferred.TurnID {
			turnEnded = true
		}
	}
	if resultExists {
		if turnEnded {
			return nil
		}
		if hasUnresolvedDeferredToolInStep(l.snapshotEvents(), deferred.TurnID, deferred.StepID) {
			return ErrToolDeferred
		}
		var payload struct {
			ConcludesTurn bool `json:"concludes_turn"`
		}
		_ = json.Unmarshal(resultEvent.Data, &payload)
		em := &emitter{sessionID: l.SessionID, runID: deferred.RunID, n: l.eventCount()}
		if !stepEnded {
			if err := l.append(durableContext(ctx), lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: deferred.RunID, TurnID: deferred.TurnID, StepID: deferred.StepID, CallID: request.CallID, Type: session.EventStepEnd, Data: strJSON("deferred_resolved")}); err != nil {
				return err
			}
		}
		if payload.ConcludesTurn {
			return l.appendTurnEnd(durableContext(ctx), lease, em, deferred.TurnID, "tool_concluded")
		}
		_, err := l.runTurnSteps(ctx, lease, stopHeartbeat, em, deferred.TurnID, &Result{}, nextStepIndex(deferred.StepID), nil)
		return err
	}
	if resumeStarted {
		if !materialized || !resumeMaterialized {
			return ErrDeferredExecutionStarted
		}
	}
	em := &emitter{sessionID: l.SessionID, runID: deferred.RunID, n: l.eventCount()}
	// Ordinary resumes cross a side-effect barrier before executing. Materialized
	// host results may safely retry after this marker because their result is
	// already durable outside the AgentKit process.
	if !resumeStarted {
		if err := l.append(durableContext(ctx), lease, session.Event{
			ID: em.id(), SessionID: l.SessionID, RunID: deferred.RunID, TurnID: deferred.TurnID, StepID: deferred.StepID,
			CallID: request.CallID, Type: session.EventToolResumeStarted,
			Data: session.ToolResumeStartedPayloadWithMode(request.CallID, toolName, request.ResumeKey, materialized),
		}); err != nil {
			return err
		}
	}
	result := request.Result
	if result == nil {
		result = &tools.Result{Name: toolName, CallID: request.CallID}
	}
	continuation, err := l.appendToolOutcomeWithMaterialized(durableContext(ctx), lease, em, tools.Outcome{
		Call: tools.Call{Name: toolName, CallID: request.CallID}, Result: result, Err: request.Err,
	}, deferred.TurnID, deferred.StepID, materialized)
	if err != nil {
		return err
	}
	if hasUnresolvedDeferredToolInStep(l.snapshotEvents(), deferred.TurnID, deferred.StepID) {
		return ErrToolDeferred
	}
	if err := l.append(durableContext(ctx), lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: deferred.RunID, TurnID: deferred.TurnID, StepID: deferred.StepID, CallID: request.CallID, Type: session.EventStepEnd, Data: strJSON("deferred_resolved")}); err != nil {
		return err
	}
	if continuation == toolDeferred {
		return ErrToolDeferred
	}
	if continuation == toolConclude {
		return l.appendTurnEnd(durableContext(ctx), lease, em, deferred.TurnID, "tool_concluded")
	}
	_, err = l.runTurnSteps(ctx, lease, stopHeartbeat, em, deferred.TurnID, &Result{}, nextStepIndex(deferred.StepID), nil)
	return err
}

// ResumeDeferredTools executes deferred calls through the original registry
// after a host-owned prerequisite becomes ready. All calls from the same
// model tool batch are completed before the loop starts another model step.
// This is the continuation path for lazy workspace tools; it does not create
// a steering message or a second model turn.
func (l *Loop) ResumeDeferredTools(ctx context.Context, requests []DeferredToolResume) error {
	resumes := make([]deferredToolBatchResume, 0, len(requests))
	for _, request := range requests {
		resumes = append(resumes, deferredToolBatchResume{resume: request})
	}
	return l.resumeDeferredToolBatch(ctx, resumes)
}

func (l *Loop) resumeDeferredToolBatch(ctx context.Context, requests []deferredToolBatchResume) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	for _, request := range requests {
		if request.resume.CallID == "" || request.resume.ResumeKey == "" {
			return fmt.Errorf("agent: deferred tool call and resume key are required")
		}
	}

	var lease session.Lease
	externalLease := l.Config.ClaimedLease != nil
	stopHeartbeat := func() error { return nil }
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
		var loopCtx context.Context
		loopCtx, stopHeartbeat = l.leaseHeartbeat(ctx, lease)
		ctx = loopCtx
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		defer func() { _ = l.Store.ReleaseLease(cleanupCtx, lease) }()
		defer func() { _ = stopHeartbeat() }()
	}
	if _, err := l.recoverSession(ctx, lease); err != nil {
		return err
	}
	if err := l.refresh(ctx); err != nil {
		return err
	}
	events := l.snapshotEvents()
	if len(requests) == 0 {
		for _, request := range pendingDeferredToolResumes(events) {
			requests = append(requests, deferredToolBatchResume{resume: request})
		}
	}
	if len(requests) == 0 {
		return nil
	}
	ordered := append([]deferredToolBatchResume(nil), requests...)
	sort.SliceStable(ordered, func(i, j int) bool {
		left, _, _, _, _ := deferredToolRecord(events, ordered[i].resume)
		right, _, _, _, _ := deferredToolRecord(events, ordered[j].resume)
		if left.Seq == 0 || right.Seq == 0 {
			return false
		}
		return left.Seq < right.Seq
	})
	requests = ordered

	var first session.Event
	concludes := false
	for _, request := range requests {
		callEvent, deferred, resultExists, resumeStarted, resumeMaterialized := deferredToolRecord(events, request.resume)
		if deferred.Type != session.EventToolDeferred {
			return fmt.Errorf("agent: deferred tool %s was not found", request.resume.CallID)
		}
		if first.Type == "" {
			first = deferred
		} else if deferred.RunID != first.RunID || deferred.TurnID != first.TurnID || deferred.StepID != first.StepID {
			return fmt.Errorf("agent: deferred tool %s is not in the resumed model tool batch", request.resume.CallID)
		}
		if resultExists {
			continue
		}
		if resumeStarted && (!request.materialized || !resumeMaterialized) {
			return ErrDeferredExecutionStarted
		}
		var call session.ToolCall
		if callEvent.Type == session.EventToolCall && json.Unmarshal(callEvent.Data, &call) != nil {
			return fmt.Errorf("agent: deferred tool %s has invalid call payload", request.resume.CallID)
		}
		if call.CallID == "" || call.Name == "" {
			return fmt.Errorf("agent: deferred tool %s has no durable call", request.resume.CallID)
		}
		if request.materialized {
			if request.result == nil {
				return fmt.Errorf("agent: materialized deferred result is required")
			}
			if err := validateToolResultIdentity(request.result, call.CallID, call.Name); err != nil {
				return err
			}
		}
	}

	first = session.Event{}
	for _, request := range requests {
		callEvent, deferred, resultExists, resumeStarted, resumeMaterialized := deferredToolRecord(events, request.resume)
		if deferred.Type != session.EventToolDeferred {
			return fmt.Errorf("agent: deferred tool %s was not found", request.resume.CallID)
		}
		if first.Type == "" {
			first = deferred
		} else if deferred.RunID != first.RunID || deferred.TurnID != first.TurnID || deferred.StepID != first.StepID {
			return fmt.Errorf("agent: deferred tool %s is not in the resumed model tool batch", request.resume.CallID)
		}
		if resultExists {
			continue
		}
		if resumeStarted && (!request.materialized || !resumeMaterialized) {
			return ErrDeferredExecutionStarted
		}
		var call session.ToolCall
		if callEvent.Type == session.EventToolCall && json.Unmarshal(callEvent.Data, &call) != nil {
			return fmt.Errorf("agent: deferred tool %s has invalid call payload", request.resume.CallID)
		}
		if call.CallID == "" || call.Name == "" {
			return fmt.Errorf("agent: deferred tool %s has no durable call", request.resume.CallID)
		}
		em := &emitter{sessionID: l.SessionID, runID: deferred.RunID, n: l.eventCount()}
		if !resumeStarted {
			if err := l.append(durableContext(ctx), lease, session.Event{
				ID: em.id(), SessionID: l.SessionID, RunID: deferred.RunID, TurnID: deferred.TurnID, StepID: deferred.StepID,
				CallID: request.resume.CallID, Type: session.EventToolResumeStarted,
				Data: session.ToolResumeStartedPayloadWithMode(request.resume.CallID, call.Name, request.resume.ResumeKey, request.materialized),
			}); err != nil {
				return err
			}
		}
		result, runErr := request.result, request.err
		if !request.materialized {
			outcomes := l.Tools.RunBatch(ctx, tools.ToolRunContext{
				SessionID: l.SessionID, RunID: deferred.RunID, TurnID: deferred.TurnID, StepID: deferred.StepID,
				CallID: request.resume.CallID, Vars: mergeVars(l.Config.Vars), Sandbox: l.Config.Sandbox,
				Artifacts: l.Config.Artifacts, Lease: &lease,
				// tool/resume_started is the side-effect barrier. A second dispatch
				// event would make recovery treat the same call as a new invocation.
				OnDispatch: func(context.Context, string, string) error { return nil },
			}, []tools.Call{{Name: call.Name, CallID: request.resume.CallID, Input: call.Arguments}})
			if len(outcomes) == 1 {
				result, runErr = outcomes[0].Result, outcomes[0].Err
			} else {
				runErr = fmt.Errorf("agent: deferred tool %s did not produce one outcome", request.resume.CallID)
			}
		}
		continuation, err := l.appendToolOutcomeWithMaterialized(durableContext(ctx), lease, em, tools.Outcome{
			Call:   tools.Call{Name: call.Name, CallID: request.resume.CallID, Input: append(json.RawMessage(nil), call.Arguments...)},
			Result: result, Err: runErr,
		}, deferred.TurnID, deferred.StepID, request.materialized)
		if err != nil {
			return err
		}
		concludes = concludes || continuation == toolConclude
		events = l.snapshotEvents()
	}

	if first.Type == "" {
		return nil
	}
	events = l.snapshotEvents()
	if hasUnresolvedDeferredToolInStep(events, first.TurnID, first.StepID) {
		return ErrToolDeferred
	}
	stepEnded, turnEnded := false, false
	for _, event := range events {
		if event.Type == session.EventStepEnd && event.TurnID == first.TurnID && event.StepID == first.StepID {
			stepEnded = true
		}
		if event.Type == session.EventTurnEnd && event.TurnID == first.TurnID {
			turnEnded = true
		}
	}
	if turnEnded {
		return nil
	}
	for _, request := range requests {
		concludes = concludes || toolResultConcludesTurn(events, request.resume.CallID)
	}
	em := &emitter{sessionID: l.SessionID, runID: first.RunID, n: l.eventCount()}
	if !stepEnded {
		if err := l.append(durableContext(ctx), lease, session.Event{
			ID: em.id(), SessionID: l.SessionID, RunID: first.RunID, TurnID: first.TurnID, StepID: first.StepID,
			Type: session.EventStepEnd, Data: strJSON("deferred_resolved"),
		}); err != nil {
			return err
		}
	}
	if concludes {
		return l.appendTurnEnd(durableContext(ctx), lease, em, first.TurnID, "tool_concluded")
	}
	_, err := l.runTurnSteps(ctx, lease, stopHeartbeat, em, first.TurnID, &Result{}, nextStepIndex(first.StepID), nil)
	return err
}

func toolResultConcludesTurn(events []session.Event, callID string) bool {
	for _, event := range events {
		if event.Type != session.EventToolResult || event.CallID != callID {
			continue
		}
		var payload struct {
			ConcludesTurn bool `json:"concludes_turn"`
		}
		return json.Unmarshal(event.Data, &payload) == nil && payload.ConcludesTurn
	}
	return false
}

func validateToolResultIdentity(result *tools.Result, callID, name string) error {
	if result == nil {
		return nil
	}
	if result.CallID != "" && result.CallID != callID {
		return fmt.Errorf("agent: deferred tool result call id mismatch")
	}
	if result.Name != "" && result.Name != name {
		return fmt.Errorf("agent: deferred tool result tool name mismatch")
	}
	return nil
}

func bindToolResultIdentity(result *tools.Result, call tools.Call) (*tools.Result, error) {
	if err := validateToolResultIdentity(result, call.CallID, call.Name); err != nil {
		return nil, err
	}
	if result == nil || (result.CallID != "" && result.Name != "") {
		return result, nil
	}
	bound := *result
	if bound.CallID == "" {
		bound.CallID = call.CallID
	}
	if bound.Name == "" {
		bound.Name = call.Name
	}
	return &bound, nil
}

func hasUnresolvedDeferredToolInStep(events []session.Event, turnID, stepID string) bool {
	requests := make(map[string]DeferredToolResume)
	for _, event := range events {
		if event.Type != session.EventToolDeferred || event.TurnID != turnID || event.StepID != stepID || event.CallID == "" {
			continue
		}
		var payload struct {
			ResumeKey string `json:"resume_key"`
		}
		if json.Unmarshal(event.Data, &payload) == nil && payload.ResumeKey != "" {
			requests[event.CallID] = DeferredToolResume{CallID: event.CallID, ResumeKey: payload.ResumeKey}
		}
	}
	for _, request := range requests {
		_, deferred, resultExists, _, _ := deferredToolRecord(events, request)
		if deferred.Type == session.EventToolDeferred && !resultExists {
			return true
		}
	}
	return false
}

func deferredToolRecord(events []session.Event, request DeferredToolResume) (call, deferred session.Event, resultExists, resumeStarted, resumeMaterialized bool) {
	for _, event := range events {
		if event.CallID != request.CallID {
			continue
		}
		switch event.Type {
		case session.EventToolCall:
			call = event
		case session.EventToolDeferred:
			var payload struct {
				ResumeKey string `json:"resume_key"`
			}
			if json.Unmarshal(event.Data, &payload) == nil && payload.ResumeKey == request.ResumeKey {
				deferred, resultExists, resumeStarted = event, false, false
			}
		case session.EventToolResumeStarted:
			if deferred.Type == session.EventToolDeferred {
				resumeStarted = true
				var payload struct {
					Materialized bool `json:"materialized"`
				}
				if json.Unmarshal(event.Data, &payload) == nil {
					resumeMaterialized = payload.Materialized
				}
			}
		case session.EventToolResult:
			if deferred.Type == session.EventToolDeferred {
				resultExists = true
			}
		}
	}
	return call, deferred, resultExists, resumeStarted, resumeMaterialized
}

func pendingDeferredToolResumes(events []session.Event) []DeferredToolResume {
	type state struct {
		key     string
		pending bool
		started bool
	}
	states := make(map[string]state)
	order := make([]string, 0)
	for _, event := range events {
		if event.CallID == "" {
			continue
		}
		current, exists := states[event.CallID]
		if !exists {
			order = append(order, event.CallID)
		}
		switch event.Type {
		case session.EventToolDeferred:
			var payload struct {
				ResumeKey string `json:"resume_key"`
			}
			if json.Unmarshal(event.Data, &payload) == nil && payload.ResumeKey != "" {
				states[event.CallID] = state{key: payload.ResumeKey, pending: true}
			}
		case session.EventToolResumeStarted:
			current.started = true
			states[event.CallID] = current
		case session.EventToolResult:
			current.pending, current.started = false, false
			states[event.CallID] = current
		}
	}
	requests := make([]DeferredToolResume, 0)
	for _, callID := range order {
		current := states[callID]
		if current.pending && !current.started {
			requests = append(requests, DeferredToolResume{CallID: callID, ResumeKey: current.key})
		}
	}
	return requests
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
	if callEvent.Type != session.EventToolCall {
		return fmt.Errorf("agent: approval call %s was not found", request.CallID)
	}
	var call session.ToolCall
	if err := json.Unmarshal(callEvent.Data, &call); err != nil || call.CallID != request.CallID || call.Name == "" {
		return fmt.Errorf("agent: approval call %s has invalid durable call payload", request.CallID)
	}
	if call.Name != request.ToolName {
		return fmt.Errorf("agent: approval tool name mismatch")
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
	em := &emitter{sessionID: l.SessionID, runID: runID, n: l.eventCount()}
	persistCtx := durableContext(resumeCtx)
	continuation := toolContinue
	if !approved {
		if !resultExists {
			if err := l.appendToolResultErr(persistCtx, lease, em, request.CallID, call.Name, tools.ErrApprovalDenied, turnID, stepID); err != nil {
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
		calls := []tools.Call{{Name: call.Name, CallID: request.CallID, Input: append(json.RawMessage(nil), call.Arguments...)}}
		outcomes := l.Tools.RunBatch(resumeCtx, tools.ToolRunContext{SessionID: l.SessionID, RunID: runID, TurnID: turnID, StepID: stepID, Vars: mergeVars(l.Config.Vars), Sandbox: l.Config.Sandbox, Artifacts: l.Config.Artifacts, Lease: &lease, OnDispatch: onDispatch}, calls)
		if err := validateToolOutcomes(calls, outcomes); err != nil {
			return err
		}
		var err error
		continuation, err = l.appendToolOutcome(persistCtx, lease, em, outcomes[0], turnID, stepID)
		if err != nil {
			return err
		}
	}
	if !stepEnded {
		if err := l.append(persistCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("approval_resolved")}); err != nil {
			return err
		}
	}
	if continuation == toolDeferred {
		return ErrToolDeferred
	}
	if continuation == toolConclude {
		if err := l.appendTurnEnd(persistCtx, lease, em, turnID, "tool_concluded"); err != nil {
			return err
		}
		l.finish(&Result{}, "tool_concluded")
		return nil
	}
	_, runErr := l.runTurnSteps(resumeCtx, lease, stopHeartbeat, em, turnID, &Result{}, nextStepIndex(stepID), nil)
	return runErr
}

func (l *Loop) recoverSession(ctx context.Context, lease session.Lease) (*session.RecoveryReport, error) {
	recoveryAfter, sequenceErr := l.Store.Sequence(ctx, l.SessionID)
	recovery, err := l.Store.Recover(ctx, lease)
	if err != nil {
		return nil, err
	}
	if sequenceErr == nil && recovery != nil && recovery.EventsAppended > 0 && l.Config.EventHub != nil {
		if recovered, loadErr := l.Store.Load(ctx, l.SessionID, recoveryAfter, 0); loadErr == nil {
			for _, event := range recovered {
				l.Config.EventHub.Publish(event)
			}
		}
	}
	return recovery, nil
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
	if _, err := l.recoverSession(loopCtx, lease); err != nil {
		return nil, err
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
	startedAt := time.Now()
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
		if cfg.Policy != nil {
			decision, policyErr := cfg.Policy.BeforeStep(ctx, StepSnapshot{
				SessionID:     l.SessionID,
				RunID:         em.runID,
				TurnID:        turnID,
				Model:         cfg.Model,
				StepIndex:     step,
				ToolCallCount: toolCallCount,
				TotalTokens:   totalTokens,
				Elapsed:       time.Since(startedAt),
			})
			if policyErr != nil {
				_ = l.appendTurnEnd(durableContext(ctx), lease, em, turnID, "policy_error")
				return nil, policyErr
			}
			switch decision {
			case StepContinue:
			case StepStop:
				if endErr := l.appendTurnEnd(durableContext(ctx), lease, em, turnID, "policy_stop"); endErr != nil {
					return nil, endErr
				}
				l.finish(result, "policy_stop")
				return result, ErrPolicyStopped
			case StepWait:
				return nil, ErrPolicyWaiting
			default:
				return nil, fmt.Errorf("agent: unknown run policy decision %d", decision)
			}
		}
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
			} else if errors.Is(err, ErrToolDeferred) {
				return nil, err
			} else if errors.Is(err, ErrPolicyWaiting) {
				_ = l.appendTurnEnd(context.Background(), lease, em, turnID, "policy_wait")
				return nil, err
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
		input.Type = typ
		if err := l.appendSkillAwareInput(ctx, lease, em, turnID, stepID, input); err != nil {
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
		// A deferred tool leaves the step open until its original call is
		// resumed. Closing it here would put step/end before tool/result and
		// force recovery to invent a second step boundary.
		if err != nil && !stepEnded && !errors.Is(err, ErrToolDeferred) {
			_ = l.append(durableContext(ctx), lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("error")})
		}
	}()
	if attempt == 0 {
		if initialInput != nil {
			_, explicitSkill := parseSkillCommand(initialInput.Text)
			if explicitSkill && initialInput.Type == session.EventUserMessage && l.Config.Skills != nil {
				if err := l.prepareSkills(ctx, lease, em, turnID, stepID); err != nil {
					return false, "", err
				}
			}
			if err := l.appendInitialInput(ctx, lease, em, turnID, stepID, *initialInput); err != nil {
				return false, "", err
			}
		}
		if len(inputs) > 0 {
			if hasExplicitSkillInput(inputs) && l.Config.Skills != nil {
				if err := l.prepareSkills(ctx, lease, em, turnID, stepID); err != nil {
					return false, "", err
				}
			}
			if err := l.appendStepInputs(ctx, lease, em, turnID, step, inputs); err != nil {
				if l.Config.RequeueStep != nil {
					l.Config.RequeueStep(inputs)
				}
				return false, "", err
			}
		}
	}
	if err := l.prepareSkills(ctx, lease, em, turnID, stepID); err != nil {
		return false, "", err
	}

	var resolvedCapabilities *llm.Capabilities
	if provider, ok := l.Chat.(capabilityChat); ok {
		capabilities, capabilityErr := provider.Capabilities(ctx, cfg.Model)
		if capabilityErr != nil {
			return false, "", fmt.Errorf("agent: resolve provider capabilities: %w", capabilityErr)
		}
		resolvedCapabilities = &capabilities
	}
	contextWindow, maxOutput := cfg.ContextWindow, cfg.MaxOutput
	if resolvedCapabilities != nil {
		if resolvedCapabilities.ContextWindow > 0 {
			contextWindow = resolvedCapabilities.ContextWindow
		}
		if resolvedCapabilities.MaxOutput > 0 {
			maxOutput = resolvedCapabilities.MaxOutput
		}
	}
	if err := l.append(ctx, lease, session.Event{
		ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID,
		Type: session.EventRequestContext,
		Data: session.RequestContextPayloadWithCapabilities(cfg.Model, contextWindow, maxOutput, resolvedCapabilities),
	}); err != nil {
		return false, "", err
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
		snapshot, buildErr := cfg.ContextBuilder.Build(memctx.BuildInput{Model: cfg.Model, System: []memctx.Section{{Title: "system", Content: systemPrompt}}, Instructions: cfg.Instructions, ToolGuidance: cfg.ToolGuidance, Runtime: cfg.RuntimeContext, Workspace: cfg.WorkspaceContext, Tools: l.Tools.ModelTools(), Messages: msgs, MaxTokens: cfg.MaxTokens, Config: cfg.ProviderConfig, Stream: cfg.Stream, Capabilities: resolvedCapabilities, ParallelTools: resolvedCapabilities != nil && resolvedCapabilities.SupportsParallelToolCalls})
		if buildErr != nil {
			return false, "", buildErr
		}
		reqValue = snapshot.Request
	} else {
		reqValue = llm.Request{Model: cfg.Model, System: []llm.Message{{Role: llm.RoleSystem, Parts: []llm.Part{{Type: llm.PartText, Text: systemPrompt}}}}, Messages: deref(msgs), Tools: l.Tools.ModelTools(), MaxTokens: cfg.MaxTokens, Config: append(json.RawMessage(nil), cfg.ProviderConfig...), Stream: cfg.Stream}
	}
	req := &reqValue
	if cfg.ToolChoice != "" {
		req.ToolChoice = cfg.ToolChoice
	}
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
		var providerErr *llm.Error
		if errors.As(err, &providerErr) && providerErr.StreamStarted {
			streamStarted = true
		}
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
		if appendErr := l.append(durableContext(ctx), lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventRequestError, Data: session.RequestErrorPayloadWithMetadata(code, err.Error(), streamStarted, llm.IsRetryable(err), requestMetadataFromError(err))}); appendErr != nil {
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
	outcomes := l.Tools.RunBatch(ctx, tools.ToolRunContext{SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Vars: ecVars, Sandbox: cfg.Sandbox, Artifacts: cfg.Artifacts, Lease: &lease, OnDispatch: onDispatch}, batch)
	if err := validateToolOutcomes(batch, outcomes); err != nil {
		return false, "", err
	}
	// Tool bodies may have started before the caller cancelled the run. Their
	// terminal outcomes are part of the durable recovery boundary and therefore
	// must be committed with a context that is no longer cancelled; otherwise a
	// started side effect would be left as an indistinguishable dangling call.
	persistCtx := durableContext(ctx)
	continuation := toolContinue
	for _, outcome := range outcomes {
		concludes, appendErr := l.appendToolOutcome(persistCtx, lease, em, outcome, turnID, stepID)
		if appendErr != nil {
			return false, "", appendErr
		}
		if concludes == toolDeferred {
			continuation = toolDeferred
		} else if concludes == toolConclude && continuation == toolContinue {
			continuation = toolConclude
		}
	}
	if continuation == toolDeferred {
		return false, "tool_deferred", ErrToolDeferred
	}
	if err := l.append(persistCtx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("tools_done")}); err != nil {
		return false, "", err
	}
	stepEnded = true
	if ctx.Err() != nil {
		return false, "", ErrStopped
	}
	if continuation == toolConclude {
		return true, "tool_concluded", nil
	}
	// Turn continues (another model call next iteration).
	return false, "tool_calls", nil
}

func (l *Loop) appendToolOutcome(ctx context.Context, lease session.Lease, em *emitter, outcome tools.Outcome, turnID, stepID string) (toolContinuation, error) {
	return l.appendToolOutcomeWithMaterialized(ctx, lease, em, outcome, turnID, stepID, false)
}

func (l *Loop) appendToolOutcomeWithMaterialized(ctx context.Context, lease session.Lease, em *emitter, outcome tools.Outcome, turnID, stepID string, materialized bool) (toolContinuation, error) {
	if outcome.Result != nil {
		bound, err := bindToolResultIdentity(outcome.Result, outcome.Call)
		if err != nil {
			return toolContinue, err
		}
		outcome.Result = bound
	}
	if outcome.Err != nil {
		if outcome.Result != nil {
			if err := l.appendToolFailureResult(ctx, lease, em, outcome.Result, outcome.Err, turnID, stepID, materialized); err != nil {
				return toolContinue, err
			}
			return toolContinue, nil
		}
		if err := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, outcome.Err, turnID, stepID); err != nil {
			return toolContinue, err
		}
		return toolContinue, nil
	}
	if outcome.Result == nil {
		err := fmt.Errorf("tool %s returned no result", outcome.Call.Name)
		if appendErr := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, err, turnID, stepID); appendErr != nil {
			return toolContinue, appendErr
		}
		return toolContinue, nil
	}
	continuation := toolContinue
	if outcome.Result.Continuation == tools.ToolDeferred {
		if outcome.Result.ResumeKey == "" {
			return toolContinue, fmt.Errorf("agent: deferred tool %s requires resume key", outcome.Call.CallID)
		}
		modelFacing := any(outcome.Result.Canonical)
		if outcome.Result.ModelFacing != nil {
			modelFacing = outcome.Result.ModelFacing
		}
		contentEncoded, err := json.Marshal(modelFacing)
		if err != nil {
			return toolContinue, err
		}
		data := session.ToolDeferredPayload(outcome.Call.Name, outcome.Result.ResumeKey, outcome.Result.WaitingReason, contentEncoded, outcome.Result.UI)
		if err := l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: outcome.Call.CallID, Type: session.EventToolDeferred, Data: data}); err != nil {
			return toolContinue, err
		}
		return toolDeferred, nil
	}
	modelFacing := any(outcome.Result.Canonical)
	if outcome.Result.ModelFacing != nil {
		modelFacing = outcome.Result.ModelFacing
	}
	contentEncoded, err := json.Marshal(modelFacing)
	if err != nil {
		if appendErr := l.appendToolResultErr(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, fmt.Errorf("encode tool content: %w", err), turnID, stepID); appendErr != nil {
			return toolContinue, appendErr
		}
		return toolContinue, nil
	}
	if outcome.Result.Continuation == tools.ToolConclude || outcome.Result.ConcludesTurn {
		continuation = toolConclude
	}
	if err := l.appendToolResult(ctx, lease, em, outcome.Call.CallID, outcome.Call.Name, contentEncoded, outcome.Result.UI, outcome.Result.Code, outcome.Result.Content, outcome.Result.AdditionalContexts, outcome.Result.ConcludesTurn, materialized, turnID, stepID); err != nil {
		return toolContinue, err
	}
	if outcome.Call.Name == "skill" {
		if loaded, ok := outcome.Result.Canonical.(skilltool.Output); ok && loaded.Activated {
			payload := session.SkillInvocationEventPayload{
				SchemaVersion: CurrentSkillSchemaVersion,
				Name:          loaded.Name,
				Provider:      loaded.Provider,
				Source:        loaded.Source,
				Version:       loaded.Version,
				ContentHash:   loaded.ContentHash,
				Origin:        "model",
			}
			if err := l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: outcome.Call.CallID, Type: session.EventSkillLoaded, Data: session.SkillInvocationPayload(payload)}); err != nil {
				return toolContinue, err
			}
			if l.Config.Skills != nil {
				l.Config.Skills.MarkLoaded(loaded.Name)
			}
		}
	}
	return continuation, nil
}

func (l *Loop) appendToolFailureResult(ctx context.Context, lease session.Lease, em *emitter, result *tools.Result, fallback error, turnID, stepID string, materialized bool) error {
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
	data := session.ToolResultStructuredPayloadWithMaterialized(result.CallID, result.Name, content, result.UI, code, true, result.Content, result.AdditionalContexts, false, materialized)
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

func (l *Loop) appendToolResult(ctx context.Context, lease session.Lease, em *emitter, callID, name string, content json.RawMessage, meta map[string]any, code string, blocks []session.ContentBlock, contexts []llm.Message, concludesTurn, materialized bool, turnID, stepID string) error {
	data := session.ToolResultStructuredPayloadWithMaterialized(callID, name, content, meta, code, false, blocks, contexts, concludesTurn, materialized)
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
		Type: session.EventToolResult, Data: data, SourceSeqs: l.assistantSourceSeqs(callID, turnID, stepID)}
	return l.append(ctx, lease, ev)
}

func (l *Loop) appendToolResultErr(ctx context.Context, lease session.Lease, em *emitter, callID, name string, execErr error, turnID, stepID string) error {
	code, recovery := toolFailure(execErr)
	data, err := json.Marshal(map[string]any{
		"call_id": callID, "name": name, "is_error": true,
		"error":   map[string]any{"code": code, "message": recovery},
		"content": map[string]any{"error": map[string]any{"code": code, "message": recovery}},
		"code":    code,
	})
	if err != nil {
		return fmt.Errorf("encode tool error result: %w", err)
	}
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
	systemSections := make([]string, 0, len(req.System))
	for _, message := range req.System {
		for _, part := range message.Parts {
			if part.Type == llm.PartText {
				systemSections = append(systemSections, part.Text)
			}
		}
	}
	promptHash := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(systemSections, ""))))
	toolsBytes, err := json.Marshal(req.Tools)
	if err != nil {
		return fmt.Errorf("encode tool schemas: %w", err)
	}
	toolsHash := fmt.Sprintf("%x", sha256.Sum256(toolsBytes))
	requestHash := fmt.Sprintf("%x", sha256.Sum256(requestSnapshot))
	metadata := session.RequestHeaderMetadata{PromptVersion: l.Config.PromptVersion, PromptHash: promptHash, ToolsHash: toolsHash}
	if l.Config.Skills != nil {
		if snapshot, ok := l.Config.Skills.Snapshot(); ok {
			metadata.SkillCatalogHash = l.Config.Skills.Catalog().Hash
			metadata.SkillSnapshotHash = snapshot.SnapshotHash
			metadata.SkillSchemaVersion = CurrentSkillSchemaVersion
		}
	}
	if req.Capabilities != nil {
		data := session.RequestHeaderPayloadWithSnapshotAndMetadata(req.Model, l.Chat.Name(), systemSections, toolSchemas, configHash, requestHash, []llm.Capabilities{*req.Capabilities}, metadata, requestSnapshot)
		return l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventRequestHeader, Data: data})
	}
	data := session.RequestHeaderPayloadWithSnapshotAndMetadata(req.Model, l.Chat.Name(), systemSections, toolSchemas, configHash, requestHash, nil, metadata, requestSnapshot)
	return l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventRequestHeader, Data: data})
}

func requestMetadataFromError(err error) *llm.RequestMetadata {
	var providerErr *llm.Error
	if !errors.As(err, &providerErr) || providerErr.Metadata == nil {
		return nil
	}
	metadata := *providerErr.Metadata
	return &metadata
}

func (l *Loop) appendUsage(ctx context.Context, lease session.Lease, em *emitter, turnID, stepID string, resp *llm.Response) error {
	if resp.Usage == nil {
		return nil
	}
	data := mustMarshal(map[string]any{
		"input_tokens": resp.Usage.InputTokens, "output_tokens": resp.Usage.OutputTokens,
		"cached_tokens": resp.Usage.CachedTokens, "cache_write_tokens": resp.Usage.CacheWriteTokens,
		"total_tokens": resp.Usage.TotalTokens(), "cost_usd": resp.Usage.CostUSD,
		"model": resp.Model, "provider": resp.Provider,
		"provider_response_id": resp.ProviderResponseID, "request_id": resp.RequestID,
		"latency_ms": resp.Latency.Milliseconds(),
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
			Capabilities: capabilities, ParallelTools: capabilities != nil && capabilities.SupportsParallelToolCalls,
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
	_, projection, err := l.project(ctx)
	if err != nil {
		return 0, err
	}
	shadowed := shadowedSeqsRetainingTail(events, l.Config.RetainTailEvents, projection.ShadowedSeqs)
	if len(shadowed) == 0 {
		return 0, fmt.Errorf("agent: no compactable surface before retained tail")
	}
	transactionID := fmt.Sprintf("compact-%d", generation)
	if err := l.append(ctx, lease, session.Event{
		ID: l.emitterIDFor("compact-start", generation), SessionID: l.SessionID,
		Type: session.EventCompactionStart, Data: session.CompactionStartPayload(generation, transactionID, shadowed),
	}); err != nil {
		return 0, err
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
	blocks := []session.ContentBlock{session.TextBlock(summary)}
	if err := l.appendCompactionCommit(ctx, lease, generation, transactionID, shadowed, summary, blocks, fingerprint); err != nil {
		return 0, err
	}
	if err := l.Store.SaveCompactionCheckpoint(ctx, lease, session.CompactionCheckpoint{SessionID: l.SessionID, Generation: generation, TransactionID: transactionID, ThroughSeq: shadowed[len(shadowed)-1], ShadowedSeqs: shadowed, Summary: summary, SummaryBlocks: blocks, SummarySHA256: fingerprint, SourceFingerprint: fingerprint}); err != nil {
		return 0, err
	}
	return l.cachedLast(), nil
}

// compact performs one durable compaction: append the start marker, ask the
// Compactor to summarize the log, then append the summary, durable surface
// replacement and end marker.
// Raw history is never deleted (H-COMPACT-001).
func (l *Loop) compact(ctx context.Context, lease session.Lease) (uint64, error) {
	if l.Config.Compactor == nil {
		return 0, fmt.Errorf("agent: compaction not configured")
	}
	events := l.snapshotEvents()
	generation := l.nextGeneration(ctx)
	_, projection, err := l.project(ctx)
	if err != nil {
		return 0, err
	}
	shadowed := shadowedSeqsRetainingTail(events, l.Config.RetainTailEvents, projection.ShadowedSeqs)
	if len(shadowed) == 0 {
		return 0, fmt.Errorf("agent: no compactable surface before retained tail")
	}
	selected := make(map[uint64]bool, len(shadowed))
	for _, seq := range shadowed {
		selected[seq] = true
	}
	region := make([]session.Event, 0, len(shadowed))
	for _, event := range events {
		if selected[event.Seq] {
			region = append(region, event)
		}
	}

	transactionID := fmt.Sprintf("compact-%d", generation)
	if err := l.append(ctx, lease, session.Event{
		ID: l.emitterIDFor("compact-start", generation), SessionID: l.SessionID,
		Type: session.EventCompactionStart, Data: session.CompactionStartPayload(generation, transactionID, shadowed),
	}); err != nil {
		return 0, err
	}
	summary, fingerprint, err := l.Config.Compactor.Compact(ctx, generation, region, append([]uint64(nil), shadowed...))
	if err != nil {
		return 0, err
	}
	blocks := []session.ContentBlock{session.TextBlock(summary)}
	if err := l.appendCompactionCommit(ctx, lease, generation, transactionID, shadowed, summary, blocks, fingerprint); err != nil {
		return 0, err
	}
	if err := l.Store.SaveCompactionCheckpoint(ctx, lease, session.CompactionCheckpoint{
		SessionID: l.SessionID, Generation: generation, TransactionID: transactionID,
		ThroughSeq: shadowed[len(shadowed)-1], ShadowedSeqs: shadowed, Summary: summary, SummaryBlocks: blocks, SummarySHA256: fingerprint, SourceFingerprint: fingerprint,
	}); err != nil {
		return 0, err
	}
	return l.cachedLast(), nil
}

func (l *Loop) appendCompactionCommit(ctx context.Context, lease session.Lease, generation uint64, transactionID string, shadowed []uint64, summary string, blocks []session.ContentBlock, fingerprint string) error {
	seqs := append([]uint64(nil), shadowed...)
	sum := session.Event{ID: l.emitterIDFor("compact-summary", generation), SessionID: l.SessionID,
		Type: session.EventCompactionSummary, Data: session.CompactionSummaryPayloadWithBlocks(generation, transactionID, shadowed[len(shadowed)-1], seqs, summary, blocks, fingerprint), SourceSeqs: append([]uint64(nil), seqs...)}
	replacement := session.Event{ID: l.emitterIDFor("compact-surface", generation), SessionID: l.SessionID,
		Type: session.EventUserMessage, Data: session.CompactionSurfaceReplacementPayload(generation, transactionID, seqs, summary, blocks, fingerprint), SourceSeqs: append([]uint64(nil), seqs...)}
	end := session.Event{ID: l.emitterIDFor("compact-end", generation), SessionID: l.SessionID,
		Type: session.EventCompactionEnd, Data: session.CompactionEndPayload(generation, transactionID)}
	return l.appendBatch(ctx, lease, []session.Event{sum, replacement, end})
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

func (l *Loop) appendInitialInput(ctx context.Context, lease session.Lease, em *emitter, turnID, stepID string, input StepInput) error {
	return l.appendSkillAwareInput(ctx, lease, em, turnID, stepID, input)
}

func (l *Loop) appendSkillAwareInput(ctx context.Context, lease session.Lease, em *emitter, turnID, stepID string, input StepInput) error {
	inputID := em.id()
	if input.ID != "" {
		inputID = input.ID + ":input"
	}
	name, command := parseSkillCommand(input.Text)
	if (input.Type != session.EventUserMessage && input.Type != session.EventSteeringMessage) || !command || l.Config.Skills == nil {
		return l.append(ctx, lease, session.Event{ID: inputID, SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: input.Type, Data: session.UserTextWithInbox(input.Text, input.ID)})
	}
	if !skill.IsName(name) {
		return l.append(ctx, lease, session.Event{ID: inputID, SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: input.Type, Data: session.UserTextWithInbox(input.Text, input.ID)})
	}
	definition, err := l.Config.Skills.GetUser(ctx, name)
	if err != nil {
		if errors.Is(err, skill.ErrSkillNotFound) || errors.Is(err, skill.ErrPolicyDenied) || errors.Is(err, skill.ErrAlreadyLoaded) {
			return l.append(ctx, lease, session.Event{ID: inputID, SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: input.Type, Data: session.UserTextWithInbox(input.Text, input.ID)})
		}
		return classifySkillInvocation(name, err)
	}
	payload := session.SkillInvocationEventPayload{
		SchemaVersion: CurrentSkillSchemaVersion,
		Name:          definition.Name,
		Provider:      definition.Provider,
		Source:        definition.Source,
		Version:       definition.Version,
		ContentHash:   definition.ContentHash,
		Origin:        "user",
		Text:          skilltool.Render(definition),
	}
	if err := l.append(ctx, lease, session.Event{ID: inputID, SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventSkillInvocation, Data: session.SkillInvocationPayload(payload)}); err != nil {
		return err
	}
	l.Config.Skills.MarkUserLoaded(name)
	return nil
}

func hasExplicitSkillInput(inputs []StepInput) bool {
	for _, input := range inputs {
		if input.Type == session.EventSteeringMessage {
			if _, command := parseSkillCommand(input.Text); command {
				return true
			}
		}
	}
	return false
}

func parseSkillCommand(text string) (string, bool) {
	fields := strings.Fields(strings.TrimSpace(text))
	if len(fields) == 0 || fields[0] != "/skill" {
		return "", false
	}
	if len(fields) != 2 {
		return "", true
	}
	return fields[1], true
}

func classifySkillInvocation(name string, err error) error {
	code := "SKILL_LOAD_FAILED"
	switch {
	case errors.Is(err, skill.ErrSkillNotFound):
		code = "SKILL_NOT_FOUND"
	case errors.Is(err, skill.ErrPolicyDenied):
		code = "SKILL_POLICY_DENIED"
	case errors.Is(err, skill.ErrPinnedMismatch), errors.Is(err, skill.ErrProviderDisposed):
		code = "SKILL_PIN_MISMATCH"
	case errors.Is(err, skill.ErrAlreadyLoaded):
		code = "SKILL_ALREADY_LOADED"
	}
	return &SkillInvocationError{Code: code, Name: name, Err: err}
}

func (l *Loop) prepareSkills(ctx context.Context, lease session.Lease, em *emitter, turnID, stepID string) error {
	runtime := l.Config.Skills
	if runtime == nil || !hasModelSkillTool(l.Tools.ModelTools()) {
		return nil
	}
	if _, err := runtime.Refresh(ctx); err != nil && !errors.Is(err, skill.ErrIncompleteCatalog) {
		return fmt.Errorf("agent: refresh skills: %w", err)
	}
	snapshot, ok := runtime.Snapshot()
	if !ok {
		return nil
	}
	catalog := runtime.Catalog()
	events := l.snapshotEvents()
	_, projection, err := l.project(ctx)
	if err != nil {
		return err
	}
	shadowedSeqs := make(map[uint64]struct{}, len(projection.ShadowedSeqs))
	for _, seq := range projection.ShadowedSeqs {
		shadowedSeqs[seq] = struct{}{}
	}
	var previous session.SkillCatalogEventPayload
	var previousSeq uint64
	for _, event := range events {
		if event.Type == session.EventSkillInvocation {
			var invocation session.SkillInvocationEventPayload
			_, isShadowed := shadowedSeqs[event.Seq]
			if !isShadowed && json.Unmarshal(event.Data, &invocation) == nil && invocation.Origin == "user" && skill.IsName(invocation.Name) && len(invocation.ContentHash) == sha256.Size*2 {
				runtime.MarkUserLoaded(invocation.Name)
			}
		}
		if event.Type == session.EventSkillLoaded && visibleSkillActivation(events, shadowedSeqs, event.CallID) {
			var invocation session.SkillInvocationEventPayload
			if json.Unmarshal(event.Data, &invocation) == nil && invocation.Origin == "model" && skill.IsName(invocation.Name) && len(invocation.ContentHash) == sha256.Size*2 {
				runtime.MarkLoaded(invocation.Name)
			}
		}
		if event.Type != session.EventSkillCatalog {
			continue
		}
		var payload session.SkillCatalogEventPayload
		if json.Unmarshal(event.Data, &payload) == nil && validSkillCatalogPayload(payload) && event.Seq >= previousSeq {
			previous, previousSeq = payload, event.Seq
		}
	}
	if len(catalog.Skills) == 0 && previousSeq == 0 {
		return nil
	}
	shadowed := false
	for _, seq := range projection.ShadowedSeqs {
		if seq == previousSeq {
			shadowed = true
			break
		}
	}
	if previousSeq > 0 && previous.CatalogHash == catalog.Hash && !shadowed {
		return nil
	}
	summaries := make([]session.SkillSummaryRef, 0, len(catalog.Skills))
	for _, summary := range catalog.Skills {
		summaries = append(summaries, session.SkillSummaryRef{Name: summary.Name, Description: summary.Description})
	}
	payload := session.SkillCatalogEventPayload{
		SchemaVersion: CurrentSkillSchemaVersion,
		Complete:      catalog.Complete,
		CatalogHash:   catalog.Hash,
		SnapshotHash:  snapshot.SnapshotHash,
		SnapshotID:    snapshot.SnapshotHash,
		Update:        previousSeq > 0,
		Text:          runtime.RenderCatalog(previousSeq > 0),
		Skills:        summaries,
	}
	return l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventSkillCatalog, Data: session.SkillCatalogPayload(payload)})
}

func validSkillCatalogPayload(payload session.SkillCatalogEventPayload) bool {
	if payload.SchemaVersion == "" || len(payload.CatalogHash) != sha256.Size*2 || payload.Text == "" {
		return false
	}
	for _, summary := range payload.Skills {
		if !skill.IsName(summary.Name) || strings.TrimSpace(summary.Description) == "" {
			return false
		}
	}
	return true
}

func hasModelSkillTool(definitions []*llm.ToolDefinition) bool {
	for _, definition := range definitions {
		if definition != nil && definition.Name == "skill" {
			return true
		}
	}
	return false
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

func shadowedSeqsRetainingTail(events []session.Event, retain int, alreadyShadowed []uint64) []uint64 {
	shadowed := make(map[uint64]bool, len(alreadyShadowed))
	for _, seq := range alreadyShadowed {
		shadowed[seq] = true
	}
	surface := make([]session.Event, 0, len(events))
	for _, event := range events {
		if event.Type.Surface() && !shadowed[event.Seq] {
			surface = append(surface, event)
		}
	}
	protected := protectedSkillSurfaceSeqs(surface)
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
		if !protected[event.Seq] {
			out = append(out, event.Seq)
		}
	}
	return out
}

// Skill instructions are durable operational context, not ordinary chat
// prose. Preserve explicit injections and the complete assistant/result group
// of model skill loads so compaction cannot keep only an activation marker.
func protectedSkillSurfaceSeqs(surface []session.Event) map[uint64]bool {
	protected := make(map[uint64]bool)
	for _, event := range surface {
		if event.Type == session.EventSkillInvocation {
			protected[event.Seq] = true
		}
		if event.Type == session.EventToolResult && toolResultIsSkillActivation(event.Data) {
			protected[event.Seq] = true
			for _, source := range event.SourceSeqs {
				protected[source] = true
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, event := range surface {
			if event.Type != session.EventToolResult {
				continue
			}
			linked := protected[event.Seq]
			for _, source := range event.SourceSeqs {
				linked = linked || protected[source]
			}
			if !linked {
				continue
			}
			if !protected[event.Seq] {
				protected[event.Seq], changed = true, true
			}
			for _, source := range event.SourceSeqs {
				if !protected[source] {
					protected[source], changed = true, true
				}
			}
		}
	}
	return protected
}

func toolResultIsSkillActivation(data json.RawMessage) bool {
	var payload struct {
		Name    string          `json:"name"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(data, &payload) != nil || payload.Name != "skill" {
		return false
	}
	var content string
	return json.Unmarshal(payload.Content, &content) == nil && strings.HasPrefix(strings.TrimSpace(content), "<skill_content ")
}

func visibleSkillActivation(events []session.Event, shadowed map[uint64]struct{}, callID string) bool {
	if callID == "" {
		return false
	}
	for _, event := range events {
		if event.Type != session.EventToolResult || event.CallID != callID || !toolResultIsSkillActivation(event.Data) {
			continue
		}
		_, isShadowed := shadowed[event.Seq]
		return !isShadowed
	}
	return false
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
