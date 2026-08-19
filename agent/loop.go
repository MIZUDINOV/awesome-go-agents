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
//     blindly. Recovery closes orphaned turns and marks dangling tool calls
//     TOOL_OUTCOME_UNKNOWN (never a blind retry of side-effecting work).
//   - Cancellation has priority over recovery/compaction.
//   - Context overflow handling order: prune -> max-safe compaction -> bounded
//     retry, and only after the surface advances.
//
// The loop forces sequential tool execution (ParallelToolCalls=false) and
// reads tool calls from the model response, so it never executes tools inline
// in a way that breaks the fencing/ordering contract the durable scheduler
// relies on.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	memctx "github.com/MIZUDINOV/awesome-go-agents/context"
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
	ErrStopped = errors.New("agent: run stopped (cancelled)")
)

// Chat is the provider-neutral model seam.
type Chat interface {
	Generate(ctx context.Context, req *llm.Request, cb llm.StreamCallback) (*llm.Response, error)
	Name() string
}

// Store is the durable single-writer seam the loop persists to.
type Store interface {
	session.FencedStore
}

// Compactor produces the durable summary + fingerprint over a region of the
// log during compaction. When nil, compaction is disabled and the loop falls
// back to pruning the projected messages on overflow.
type Compactor interface {
	Compact(ctx context.Context, generation uint64, events []session.Event, throughSeq uint64) (summary, fingerprint string, err error)
}

// Config controls the loop.
type Config struct {
	Model        string
	Owner        string
	SystemPrompt string

	MaxTurns        int
	MaxStepsPerTurn int
	MaxTokens       int64

	Stream bool

	// ProviderConfig is opaque JSON embedded in llm.Request.Config. The loop
	// forces parallel_tool_calls=false on top of it.
	ProviderConfig json.RawMessage

	LeaseTTL time.Duration

	ContextWindow         int64
	MaxOutput             int64
	CompactThresholdRatio float64
	PruneHead, PruneTail  int
	Compactor             Compactor

	// Vars are merged into every tool ExecContext (host bindings such as the
	// sandbox working directory under "cwd").
	Vars map[string]any
}

// DefaultConfig returns the review-checklist reference defaults.
func DefaultConfig() Config {
	return Config{
		MaxTurns: 10, MaxStepsPerTurn: 6, MaxTokens: 4096,
		LeaseTTL:      30 * time.Second,
		ContextWindow: 128000, MaxOutput: 4096,
		CompactThresholdRatio: 0.80, PruneHead: 4096, PruneTail: 1024,
	}
}

// Loop runs a durable session against the seams.
type Loop struct {
	SessionID string
	Store     Store
	Tools     *tools.Registry
	Chat      Chat
	Config    Config

	mu     sync.Mutex
	events []session.Event
}

// NewLoop builds a Loop. cfg is normalized with reference defaults.
func NewLoop(sessionID string, store Store, registry *tools.Registry, chat Chat, cfg Config) *Loop {
	def := DefaultConfig()
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = def.MaxTurns
	}
	if cfg.MaxStepsPerTurn <= 0 {
		cfg.MaxStepsPerTurn = def.MaxStepsPerTurn
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
	return &Loop{SessionID: sessionID, Store: store, Tools: registry, Chat: chat, Config: cfg}
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
	cfg := l.Config

	// Single-writer admission. If another worker holds the session, refuse
	// rather than split-brain the log.
	lease, err := l.Store.ClaimLease(ctx, l.SessionID, cfg.Owner, cfg.LeaseTTL, "")
	if err != nil {
		if errors.Is(err, session.ErrLeaseHeld) {
			return nil, ErrBusy
		}
		return nil, err
	}
	defer func() { _ = l.Store.ReleaseLease(ctx, lease) }()

	// Recovery runs BEFORE any new work: close an orphaned turn from a crash
	// and mark dangling tool calls TOOL_OUTCOME_UNKNOWN (never re-execute).
	if _, err := l.Store.Recover(ctx, lease); err != nil {
		return nil, err
	}
	if err := l.refresh(ctx); err != nil {
		return nil, err
	}

	runID := newRunID()
	em := &emitter{sessionID: l.SessionID, runID: runID}

	// Durable request preamble (diagnostics, non-surface).
	if err := l.append(ctx, lease, em.next(session.EventRequestContext, map[string]any{
		"model": cfg.Model, "context_window": cfg.ContextWindow, "max_output": cfg.MaxOutput,
	})); err != nil {
		return nil, err
	}
	userEv := session.Event{
		ID: em.id(), SessionID: l.SessionID, RunID: runID, Type: session.EventUserMessage,
		Data: session.UserText(input),
	}
	if err := l.append(ctx, lease, userEv); err != nil {
		return nil, err
	}

	result := &Result{Input: input}
	compacted := 0

	for turn := 0; turn < cfg.MaxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			l.finish(result, "interrupted")
			return result, ErrStopped
		}

		// Context-overflow preflight: prune -> compact -> bounded retry.
		over, err := l.preflight(ctx, lease, &compacted)
		if err != nil {
			return nil, err
		}
		if over {
			return nil, ErrContextOverflow
		}

		done, reason, err := l.step(ctx, lease, em, turn)
		if err != nil {
			if isOverflow(err) {
				// Provider hit the context limit mid-step: one max-safe
				// compaction, then retry once. Cancellation wins over recovery.
				if _, cerr := l.compact(ctx, lease); cerr == nil && ctx.Err() == nil && l.Config.Compactor != nil {
					compacted++
					done, reason, err = l.step(ctx, lease, em, turn)
					if err != nil {
						if isOverflow(err) {
							return nil, ErrContextOverflow
						}
						return nil, err
					}
				} else {
					return nil, ErrContextOverflow
				}
			} else if errors.Is(err, ErrStopped) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				l.finish(result, "interrupted")
				return result, ErrStopped
			} else {
				return nil, err
			}
		}
		result.Turns = turn + 1
		result.CompactCount = compacted
		if done {
			_ = l.appendTurnEnd(ctx, lease, em, reason)
			l.finish(result, reason)
			return result, nil
		}
	}
	_ = l.appendTurnEnd(ctx, lease, em, "turn_limit")
	l.finish(result, "turn_limit")
	return result, nil
}

// finish projects the final surface into the result.
func (l *Loop) finish(result *Result, reason string) {
	msgs, _, err := l.project(context.Background())
	if err == nil {
		result.Messages = msgs
	}
	result.FinishedReason = reason
}

// step runs ONE model call plus its (sequential) tool executions. Returns
// done=true when the assistant produced no further tool calls.
func (l *Loop) step(ctx context.Context, lease session.Lease, em *emitter, turn int) (bool, string, error) {
	cfg := l.Config

	turnID := fmt.Sprintf("turn-%04d", turn)
	stepID := fmt.Sprintf("turn-%04d-step-%02d", turn, 0)
	_ = l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, Type: session.EventTurnStart, TurnID: turnID, Data: strJSON("start")})
	_ = l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, Type: session.EventStepStart, TurnID: turnID, StepID: stepID, Data: strJSON("step_start")})

	// Project the durable history into model messages.
	msgs, _, err := l.project(ctx)
	if err != nil {
		return false, "", err
	}

	req := &llm.Request{
		Model:     cfg.Model,
		System:    []llm.Message{{Role: llm.RoleSystem, Parts: []llm.Part{{Type: llm.PartText, Text: cfg.SystemPrompt}}}},
		Messages:  deref(msgs),
		Tools:     l.Tools.ModelTools(),
		MaxTokens: cfg.MaxTokens,
		Config:    cfg.ProviderConfig,
		Stream:    cfg.Stream,
	}
	parallel := false
	req.ParallelToolCalls = &parallel

	resp, err := l.Chat.Generate(ctx, req, nil)
	if err != nil {
		return false, "", err
	}
	if resp == nil || resp.Message == nil {
		return false, "", fmt.Errorf("agent: provider returned no message")
	}

	calls := resp.Message.ToolCalls()
	text := resp.Message.Text()
	reasoning := resp.Message.Reasoning()

	_ = l.appendUsage(ctx, lease, em, turnID, stepID, resp)

	if len(calls) == 0 {
		// Final assistant reply; no further tools -> turn completes.
		ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID,
			Type: session.EventAssistantMessage, Data: session.AssistantContent(text, reasoning, nil)}
		end := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, Type: session.EventStepEnd, Data: strJSON("end")}
		if err := l.appendBatch(ctx, lease, []session.Event{ev, end}); err != nil {
			return false, "", err
		}
		return true, string(resp.FinishReason), nil
	}

	// Persist the assistant message WITH its tool calls durably (the calls
	// must survive a crash so Recover can mark unknown outcomes rather than
	// re-execute).
	asst := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID,
		Type: session.EventAssistantMessage, Data: session.AssistantContent(text, reasoning, toDurableCalls(calls))}
	if err := l.append(ctx, lease, asst); err != nil {
		return false, "", err
	}

	// Execute tools sequentially. Each invocation:
	//   tool/call barrier (persisted BEFORE the side effect, H-RUNTIME-008)
	//   -> execute -> tool/result (durable).
	executed := 0
	for _, call := range calls {
		if err := ctx.Err(); err != nil {
			return false, "", ErrStopped
		}
		callID := call.CallID
		if callID == "" {
			callID = em.id()
		}
		callEv := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
			Type: session.EventToolCall, Data: session.ToolCallPayload(callID, call.Name, call.Arguments)}
		if err := l.append(ctx, lease, callEv); err != nil {
			return false, "", err
		}
		// The tool/call is durable before any side effect executes.

		ecVars := mergeVars(cfg.Vars)
		out, runErr := l.Tools.Run(ctx, tools.ExecContext{SessionID: l.SessionID, RunID: em.runID, Vars: ecVars}, call.Name, callID, call.Arguments)
		if runErr != nil {
			// A tool pipeline failure (validation, sandbox denial, timeout) is
			// still a durable tool/result outcome; the call is never left dangling.
			if err := l.appendToolResultErr(ctx, lease, em, callID, call.Name, runErr, turnID, stepID); err != nil {
				return false, "", err
			}
		} else {
			// Persist the model-facing result (the durable output the surface
			// re-renders on the next step), falling back to the canonical value.
			output := any(out.Canonical)
			if out.ModelFacing != nil {
				output = out.ModelFacing
			}
			encoded, _ := json.Marshal(output)
			if err := l.appendToolResult(ctx, lease, em, callID, call.Name, encoded, turnID, stepID); err != nil {
				return false, "", err
			}
		}
		executed++
		if executed >= cfg.MaxStepsPerTurn {
			break
		}
	}
	_ = l.append(ctx, lease, session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, Type: session.EventStepEnd, Data: strJSON("tools_done")})
	// Turn continues (another model call next iteration).
	return false, "tool_calls", nil
}

func toDurableCalls(calls []llm.ToolCallRequest) []session.ToolCall {
	out := make([]session.ToolCall, 0, len(calls))
	for _, c := range calls {
		out = append(out, session.ToolCall{CallID: c.CallID, Name: c.Name, Arguments: c.Arguments})
	}
	return out
}

func (l *Loop) appendToolResult(ctx context.Context, lease session.Lease, em *emitter, callID, name string, output json.RawMessage, turnID, stepID string) error {
	data, _ := json.Marshal(map[string]any{"call_id": callID, "name": name, "output": output})
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
		Type: session.EventToolResult, Data: data}
	return l.append(ctx, lease, ev)
}

func (l *Loop) appendToolResultErr(ctx context.Context, lease session.Lease, em *emitter, callID, name string, execErr error, turnID, stepID string) error {
	data, _ := json.Marshal(map[string]any{
		"call_id": callID, "name": name, "is_error": true, "error": execErr.Error(), "output": map[string]any{},
	})
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, RunID: em.runID, TurnID: turnID, StepID: stepID, CallID: callID,
		Type: session.EventToolResult, Data: data}
	return l.append(ctx, lease, ev)
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

func (l *Loop) appendTurnEnd(ctx context.Context, lease session.Lease, em *emitter, reason string) error {
	ev := session.Event{ID: em.id(), SessionID: l.SessionID, Type: session.EventTurnEnd, Data: mustMarshal(map[string]any{"reason": reason})}
	return l.append(ctx, lease, ev)
}

// preflight measures pressure; if over threshold it prunes or compacts so the
// surface advances. Returns true when still over capacity after the attempt.
func (l *Loop) preflight(ctx context.Context, lease session.Lease, compacted *int) (bool, error) {
	cfg := l.Config
	msgs, _, err := l.project(ctx)
	if err != nil {
		return false, err
	}
	usage := estimateTokens(msgs) + estimateTokens([]*llm.Message{{Parts: []llm.Part{{Type: llm.PartText, Text: cfg.SystemPrompt}}}})
	window := cfg.ContextWindow - cfg.MaxOutput
	if window <= 0 || usage < int64(float64(cfg.ContextWindow)*cfg.CompactThresholdRatio) {
		return false, nil
	}

	// Over threshold: one max-safe compaction when a compactor exists.
	if cfg.Compactor != nil && l.eventCount() > 0 {
		if _, cerr := l.compact(ctx, lease); cerr == nil {
			*compacted++
			return false, nil
		}
	}
	// No compactor (or compaction failed): prune oversized tool results in the
	// projected request as a last resort.
	if cfg.PruneHead > 0 {
		if pruned := memctx.DefaultPruneCaps().PruneMessages(msgs); pruned > 0 {
			return false, nil
		}
	}
	return true, nil
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
	throughSeq := l.lastSeq(events)

	summary, fingerprint, err := l.Config.Compactor.Compact(ctx, generation, events, throughSeq)
	if err != nil {
		return 0, err
	}
	transactionID := fmt.Sprintf("compact-%d", generation)
	shadowed := shadowedSeqs(events, throughSeq)

	start := session.Event{ID: l.emitterIDFor("compact-start", generation), SessionID: l.SessionID, Type: session.EventCompactionStart, Data: session.CompactionStartPayload(generation, transactionID, shadowed)}
	sum := session.Event{ID: l.emitterIDFor("compact-summary", generation), SessionID: l.SessionID, Type: session.EventCompactionSummary,
		Data: session.CompactionSummaryPayload(generation, transactionID, throughSeq, shadowed, summary, fingerprint), SourceSeqs: append([]uint64(nil), shadowed...)}
	end := session.Event{ID: l.emitterIDFor("compact-end", generation), SessionID: l.SessionID, Type: session.EventCompactionEnd, Data: session.CompactionEndPayload(generation, transactionID)}

	if err := l.appendBatch(ctx, lease, []session.Event{start, sum, end}); err != nil {
		return 0, err
	}
	_ = l.Store.SaveCompactionCheckpoint(ctx, lease, session.CompactionCheckpoint{
		SessionID: l.SessionID, Generation: generation, TransactionID: transactionID,
		ThroughSeq: throughSeq, ShadowedSeqs: shadowed, Summary: summary, SummarySHA256: fingerprint,
	})
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
	for i := range events {
		events[i].Normalize()
	}
	if _, err := l.Store.AppendFenced(ctx, lease, events); err != nil {
		if errors.Is(err, session.ErrLeaseLost) {
			return ErrLeaseLost
		}
		return err
	}
	// Fold the freshly-persisted tail into the cache.
	after := l.cachedLast()
	loaded, err := l.Store.Load(ctx, l.SessionID, after, 0)
	if err == nil && len(loaded) > 0 {
		l.mu.Lock()
		for _, e := range loaded {
			l.events = append(l.events, e.Clone())
		}
		l.mu.Unlock()
	}
	return nil
}

// refresh rebuilds the in-memory tail from the log.
func (l *Loop) refresh(ctx context.Context) error {
	events, err := l.Store.Load(ctx, l.SessionID, 0, 0)
	if err != nil {
		return err
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

func (l *Loop) lastSeq(events []session.Event) uint64 {
	if len(events) == 0 {
		return 0
	}
	return events[len(events)-1].Seq
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

func estimateTokens(msgs []*llm.Message) int64 {
	var total int64
	for _, m := range msgs {
		if m == nil {
			continue
		}
		for _, part := range m.Parts {
			switch part.Type {
			case llm.PartText:
				total += int64((len([]rune(part.Text)) + 3) / 4)
			case llm.PartReasoning:
				total += int64((len([]rune(part.Reasoning)) + 3) / 4)
			}
		}
	}
	return total
}

func shadowedSeqs(events []session.Event, throughSeq uint64) []uint64 {
	var out []uint64
	for _, e := range events {
		if e.Seq <= throughSeq && e.Type.Surface() {
			out = append(out, e.Seq)
		}
	}
	return out
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
	n         int
}

func (e *emitter) next(typ session.EventType, payload map[string]any) session.Event {
	data, _ := json.Marshal(payload)
	return session.Event{ID: e.id(), Type: typ, Data: data}
}

func (e *emitter) id() string {
	e.n++
	return fmt.Sprintf("ev:%s:%s:%d", e.sessionID, e.runID, e.n)
}

func newRunID() string {
	n := atomic.AddUint64(&runCounter, 1)
	return fmt.Sprintf("run-%d-%d", n, time.Now().UnixNano())
}

var runCounter uint64
