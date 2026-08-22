// Package tools implements the agent's tool registry and execution pipeline.
// A Definition carries the model-facing surface (name, description, input
// schema) AND the runtime concerns (executor, renderers, concurrency, timeout)
// that the llm package intentionally excludes.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
)

// ToolRunContext carries the call identity and host-owned execution ports a
// tool may need beyond its decoded arguments. Signal mirrors the cancellation
// source for channel-oriented integrations.
type ToolRunContext struct {
	// SessionID / RunID are durable correlation identifiers.
	SessionID    string
	RunID        string
	TurnID       string
	StepID       string
	RootCallID   string
	ParentCallID string
	CallID       string
	Signal       <-chan struct{}
	// Vars carries arbitrary host-provided bindings (tool-agnostic).
	Vars map[string]any
	// Sandbox is the runtime authority boundary. Registry admission happens
	// before a body starts; execution adapters still enforce resource checks.
	Sandbox integration.Sandbox
	// Artifacts is the optional durable spill port for complete output that is
	// too large for the model-facing render.
	Artifacts integration.ArtifactStore
	// Lease is the current fenced session claim. Approval brokers may use it
	// to persist request/decision events without bypassing single-writer
	// fencing.
	Lease *session.Lease
	// Runtime is the owning scoped registry. Code Mode runtimes must use this
	// seam for generated SDK calls instead of dispatching directly to a backend.
	Runtime Runtime
	// OnDispatch is invoked after policy/approval/guards and immediately before
	// the executor can cause an external side effect.
	OnDispatch func(context.Context, string, string) error
	control    *toolRunControl
}

type toolRunControl struct {
	mu            sync.Mutex
	continuation  ToolContinuation
	resumeKey     string
	waitingReason string
	concluded     bool
}

// DeferContext marks the current call as waiting for a host-resumed operation.
func (c ToolRunContext) DeferContext(resumeKey, waitingReason string) {
	if c.control == nil {
		return
	}
	c.control.mu.Lock()
	c.control.continuation = ToolDeferred
	c.control.resumeKey = resumeKey
	c.control.waitingReason = waitingReason
	c.control.mu.Unlock()
}

// ConcludeTurn marks the current call as the terminal tool call for this turn.
func (c ToolRunContext) ConcludeTurn() {
	if c.control == nil {
		return
	}
	c.control.mu.Lock()
	c.control.concluded = true
	c.control.continuation = ToolConclude
	c.control.mu.Unlock()
}

func (c ToolRunContext) directive() (ToolContinuation, string, string, bool) {
	if c.control == nil {
		return ToolContinue, "", "", false
	}
	c.control.mu.Lock()
	defer c.control.mu.Unlock()
	return c.control.continuation, c.control.resumeKey, c.control.waitingReason, c.control.concluded
}

// Executor runs a tool and returns the canonical structured result.
type Executor func(ctx context.Context, ec ToolRunContext, input json.RawMessage) (any, error)

// Renderer converts the original arguments and canonical result into a
// model-facing textual or JSON representation. If nil, the canonical value is
// JSON-marshalled. Raw args keep rendering deterministic without forcing the
// registry to know a tool's input type.
type Renderer func(args json.RawMessage, canonical any) (any, error)

// Presenter converts the original arguments and canonical result into UI
// presentation metadata. If nil, no UI metadata is produced.
type Presenter func(args json.RawMessage, canonical any) (map[string]any, error)

// ContentRenderer emits provider-neutral content blocks for model output.
type ContentRenderer func(args json.RawMessage, canonical any) ([]session.ContentBlock, error)

// ContentFinalizer derives final model content from validated canonical output.
// It is pure and cannot mutate Result or lifecycle state.
type ContentFinalizer func(args json.RawMessage, canonical any) ([]session.ContentBlock, error)

// CallPresenter derives compact metadata for a tool call before execution.
type CallPresenter func(args json.RawMessage) (map[string]any, error)

// ContinuationResolver derives lifecycle state from the validated canonical
// result. Lifecycle is registry-owned; content finalizers cannot mutate it.
type ContinuationResolver func(args json.RawMessage, canonical any) (ToolContinuation, string, string, error)

// PromptSection is tool-owned guidance kept separate from the model catalog.
type PromptSection struct {
	Title   string
	Content string
}

// PresentationMode controls how the model receives the catalog.
type PresentationMode string

const (
	PresentationNative PresentationMode = "native"
	PresentationCode   PresentationMode = "code"
	PresentationBoth   PresentationMode = "both"
)

// CodeRuntime is the only execution seam for generated Code Mode programs.
// Implementations must route SDK calls back through the owning Registry.
type CodeRuntime interface {
	ExecuteCode(context.Context, ToolRunContext, string, string) (any, error)
}

// StaticInputSchema is a convenience for tools whose input schema is a fixed
// JSON Schema object already encoded.
type StaticInputSchema struct{ Schema json.RawMessage }

// Definition describes one tool. InputSchema and OutputSchema must be JSON
// Schema objects for arguments and canonical results respectively.
type Definition struct {
	Name        string
	Description string
	Version     string
	Guidance    []PromptSection

	// InputSchema is the JSON Schema object the model sees for arguments.
	InputSchema json.RawMessage
	// OutputSchema validates the canonical result and is mandatory at register.
	OutputSchema json.RawMessage

	// typed*Schema are populated by DefineTool. They keep a typed definition's
	// runtime schema and its Go type contract from drifting apart.
	typedInputSchema  json.RawMessage
	typedOutputSchema json.RawMessage

	// Execute implements the tool. Required.
	Execute Executor
	// RenderModel optionally shapes what the model sees as the result.
	RenderModel Renderer
	// RenderContent optionally emits provider-neutral content blocks.
	RenderContent       ContentRenderer
	PresentCall         CallPresenter
	PresentResult       Presenter
	FinalizeContent     ContentFinalizer
	ResolveContinuation ContinuationResolver

	// ConcurrencySafe marks the tool safe to run in parallel with other tools.
	ConcurrencySafe bool
	// MutatesWorkspace marks the tool as side-effecting (used for scheduling).
	MutatesWorkspace bool
	// Timeout bounds a single execution. Zero uses the registry default.
	Timeout time.Duration
	// ConcurrencySafeFor classifies a validated call. Only an exact true lets
	// the scheduler overlap it with another parallel-safe call.
	ConcurrencySafeFor func(json.RawMessage) bool
}

// IsConcurrencySafe is the scheduler-facing classification hook. It accepts
// the immutable, already-validated argument bytes and returns true only when
// this call is explicitly safe to overlap with another call.
func (d *Definition) IsConcurrencySafe(input json.RawMessage) bool {
	if d == nil {
		return false
	}
	if d.ConcurrencySafeFor != nil {
		var safe bool
		func() {
			defer func() { _ = recover() }()
			safe = d.ConcurrencySafeFor(append(json.RawMessage(nil), input...))
		}()
		return safe
	}
	return d.ConcurrencySafe
}

// DefineToolOptions is the typed authoring surface. It compiles into the
// erased Definition so providers never receive runtime callbacks or bindings.
// When a schema override is omitted, it is generated from the corresponding
// Go type, keeping typed decoding and model validation on one source of truth.
type DefineToolOptions[I any, O any] struct {
	Name                  string
	Description           string
	Version               string
	Guidance              []PromptSection
	InputSchema           json.RawMessage
	OutputSchema          json.RawMessage
	Timeout               time.Duration
	MutatesWorkspace      bool
	ConcurrencySafe       func(I) bool
	Execute               func(context.Context, ToolRunContext, I) (O, error)
	RenderModel           func(O) (any, error)
	RenderModelWithArgs   func(I, O) (any, error)
	RenderContent         func(O) ([]session.ContentBlock, error)
	RenderContentWithArgs func(I, O) ([]session.ContentBlock, error)
	PresentCall           func(I) (map[string]any, error)
	PresentResult         func(O) (map[string]any, error)
	PresentResultWithArgs func(I, O) (map[string]any, error)
	FinalizeContent       func(O) ([]session.ContentBlock, error)
	ResolveContinuation   func(I, O) (ToolContinuation, string, string, error)
}

// DefineTool converts a typed definition into the runtime representation.
func DefineTool[I any, O any](opts DefineToolOptions[I, O]) *Definition {
	expectedInputSchema := generatedSchema[I]()
	expectedOutputSchema := generatedSchema[O]()
	inputSchema := append(json.RawMessage(nil), opts.InputSchema...)
	if len(inputSchema) == 0 {
		inputSchema = append(json.RawMessage(nil), expectedInputSchema...)
	}
	outputSchema := append(json.RawMessage(nil), opts.OutputSchema...)
	if len(outputSchema) == 0 {
		outputSchema = append(json.RawMessage(nil), expectedOutputSchema...)
	}
	definition := &Definition{
		Name: opts.Name, Description: opts.Description, Version: opts.Version,
		Guidance:    append([]PromptSection(nil), opts.Guidance...),
		InputSchema: inputSchema, OutputSchema: outputSchema,
		typedInputSchema: expectedInputSchema,
		Timeout:          opts.Timeout, MutatesWorkspace: opts.MutatesWorkspace,
		ConcurrencySafeFor: func(input json.RawMessage) bool {
			var args I
			return opts.ConcurrencySafe != nil && json.Unmarshal(input, &args) == nil && opts.ConcurrencySafe(args)
		},
		Execute: func(ctx context.Context, ec ToolRunContext, input json.RawMessage) (any, error) {
			if opts.Execute == nil {
				return nil, ErrInvalidArguments
			}
			var args I
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, err
			}
			return opts.Execute(ctx, ec, args)
		},
	}
	definition.RenderModel = func(input json.RawMessage, canonical any) (any, error) {
		var args I
		if err := json.Unmarshal(input, &args); err != nil {
			return nil, ErrInvalidArguments
		}
		value, ok := canonical.(O)
		if !ok {
			return nil, ErrInvalidOutput
		}
		if opts.RenderModelWithArgs != nil {
			return opts.RenderModelWithArgs(args, value)
		}
		if opts.RenderModel != nil {
			return opts.RenderModel(value)
		}
		return canonical, nil
	}
	if opts.RenderContent != nil || opts.RenderContentWithArgs != nil {
		definition.RenderContent = func(input json.RawMessage, canonical any) ([]session.ContentBlock, error) {
			var args I
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, ErrInvalidArguments
			}
			value, ok := canonical.(O)
			if !ok {
				return nil, ErrInvalidOutput
			}
			if opts.RenderContentWithArgs != nil {
				return opts.RenderContentWithArgs(args, value)
			}
			return opts.RenderContent(value)
		}
	}
	if opts.FinalizeContent != nil {
		definition.FinalizeContent = func(input json.RawMessage, canonical any) ([]session.ContentBlock, error) {
			var args I
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, ErrInvalidArguments
			}
			value, ok := canonical.(O)
			if !ok {
				return nil, ErrInvalidOutput
			}
			return opts.FinalizeContent(value)
		}
	}
	if opts.PresentCall != nil {
		definition.PresentCall = func(input json.RawMessage) (map[string]any, error) {
			var args I
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, ErrInvalidArguments
			}
			return opts.PresentCall(args)
		}
	}
	if opts.PresentResult != nil || opts.PresentResultWithArgs != nil {
		definition.PresentResult = func(input json.RawMessage, canonical any) (map[string]any, error) {
			var args I
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, ErrInvalidArguments
			}
			value, ok := canonical.(O)
			if !ok {
				return nil, ErrInvalidOutput
			}
			if opts.PresentResultWithArgs != nil {
				return opts.PresentResultWithArgs(args, value)
			}
			return opts.PresentResult(value)
		}
	}
	definition.ResolveContinuation = func(input json.RawMessage, canonical any) (ToolContinuation, string, string, error) {
		if opts.ResolveContinuation == nil {
			return ToolContinue, "", "", nil
		}
		var args I
		if err := json.Unmarshal(input, &args); err != nil {
			return ToolContinue, "", "", ErrInvalidArguments
		}
		value, ok := canonical.(O)
		if !ok {
			return ToolContinue, "", "", ErrInvalidOutput
		}
		return opts.ResolveContinuation(args, value)
	}
	definition.typedOutputSchema = expectedOutputSchema
	return definition
}

func generatedSchema[T any]() json.RawMessage {
	schema, err := FromStruct[T]()
	if err != nil {
		panic(fmt.Sprintf("tools: generate typed schema for %T: %v", *new(T), err))
	}
	return schema
}

// Result is the three-part tool outcome per the DSH model:
// Canonical (execution-local), ModelFacing (what goes to the model), UI
// (presentation). Only the model-facing content, UI metadata and lifecycle
// outcome cross the durable session boundary.
type Result struct {
	// Name/CallID bind the result to the originating model tool call.
	Name   string      `json:"name"`
	CallID string      `json:"call_id"`
	Kind   OutcomeKind `json:"kind"`

	// Canonical is the structured execution value (may not be persisted to the
	// LLM history).
	Canonical any `json:"canonical,omitempty"`
	// ModelFacing is what the model sees (text or structured).
	ModelFacing any `json:"model_facing,omitempty"`
	// UI is presentation metadata (diffs, paths).
	UI map[string]any `json:"ui,omitempty"`
	// CallPresentation is compact metadata derived from arguments before
	// execution; UI is reserved for the completed result.
	CallPresentation   map[string]any         `json:"call_presentation,omitempty"`
	Meta               map[string]any         `json:"meta,omitempty"`
	Content            []session.ContentBlock `json:"content,omitempty"`
	AdditionalContexts []llm.Message          `json:"additional_contexts,omitempty"`
	Continuation       ToolContinuation       `json:"continuation,omitempty"`
	ResumeKey          string                 `json:"resume_key,omitempty"`
	WaitingReason      string                 `json:"waiting_reason,omitempty"`
	// ConcludesTurn is retained for source compatibility while hosts migrate
	// to Continuation=ToolConclude. The AgentKit loop normalizes both forms.
	ConcludesTurn bool `json:"concludes_turn,omitempty"`
	// Code is a stable non-empty outcome code for failed tool calls.
	Code    string   `json:"code,omitempty"`
	Failure *Failure `json:"failure,omitempty"`
	Frozen  bool     `json:"frozen"`
}

// Freeze returns an isolated, immutable-by-convention result snapshot. The
// registry never returns its internal maps to callers.
func (r *Result) Freeze() *Result {
	if r == nil {
		return nil
	}
	out := *r
	out.Frozen = true
	if r.CallPresentation != nil {
		out.CallPresentation = make(map[string]any, len(r.CallPresentation))
		for key, value := range r.CallPresentation {
			out.CallPresentation[key] = cloneResultValue(value)
		}
	}
	out.Canonical = cloneResultValue(r.Canonical)
	out.ModelFacing = cloneResultValue(r.ModelFacing)
	if r.UI != nil {
		out.UI = make(map[string]any, len(r.UI))
		for key, value := range r.UI {
			out.UI[key] = cloneResultValue(value)
		}
	}
	if r.Meta != nil {
		out.Meta = make(map[string]any, len(r.Meta))
		for key, value := range r.Meta {
			out.Meta[key] = cloneResultValue(value)
		}
	}
	if r.Content != nil {
		out.Content = make([]session.ContentBlock, len(r.Content))
		for i, block := range r.Content {
			out.Content[i] = block
			out.Content[i].Payload = append(json.RawMessage(nil), block.Payload...)
			if block.Media != nil {
				media := *block.Media
				out.Content[i].Media = &media
			}
			if block.ToolCall != nil {
				call := *block.ToolCall
				call.Arguments = append(json.RawMessage(nil), block.ToolCall.Arguments...)
				out.Content[i].ToolCall = &call
			}
			if block.ToolResult != nil {
				result := *block.ToolResult
				result.Content = append(json.RawMessage(nil), block.ToolResult.Content...)
				out.Content[i].ToolResult = &result
			}
		}
	}
	if r.AdditionalContexts != nil {
		out.AdditionalContexts = make([]llm.Message, len(r.AdditionalContexts))
		for i := range r.AdditionalContexts {
			if clone := r.AdditionalContexts[i].Clone(); clone != nil {
				out.AdditionalContexts[i] = *clone
			}
		}
	}
	if r.Failure != nil {
		failure := *r.Failure
		if r.Failure.Meta != nil {
			failure.Meta = make(map[string]any, len(r.Failure.Meta))
			for key, value := range r.Failure.Meta {
				failure.Meta[key] = cloneResultValue(value)
			}
		}
		out.Failure = &failure
	}
	return &out
}

// cloneResultValue copies JSON-shaped and typed values used by canonical,
// model/UI projections without changing the concrete type of tool results.
func cloneResultValue(value any) any {
	if value == nil {
		return nil
	}
	state := &cloneResultState{seen: make(map[cloneResultVisit]reflect.Value)}
	cloned := cloneResultReflect(reflect.ValueOf(value), state)
	if !cloned.IsValid() || !cloned.CanInterface() {
		return nil
	}
	return cloned.Interface()
}

type cloneResultVisit struct {
	typ  reflect.Type
	kind reflect.Kind
	ptr  uintptr
}

type cloneResultState struct {
	seen map[cloneResultVisit]reflect.Value
}

func cloneResultReflect(value reflect.Value, state *cloneResultState) reflect.Value {
	if !value.IsValid() {
		return reflect.Value{}
	}
	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := cloneResultReflect(value.Elem(), state)
		out := reflect.New(value.Type()).Elem()
		if cloned.IsValid() && cloned.Type().AssignableTo(value.Type()) {
			out.Set(cloned)
		} else if cloned.IsValid() && cloned.Type().Implements(value.Type()) {
			out.Set(cloned)
		}
		return out
	case reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := cloneResultVisit{typ: value.Type(), kind: value.Kind(), ptr: value.Pointer()}
		if cloned, ok := state.seen[visit]; ok {
			return cloned
		}
		out := reflect.New(value.Type().Elem())
		state.seen[visit] = out
		setClonedField(out.Elem(), cloneResultReflect(value.Elem(), state))
		return out
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := cloneResultVisit{typ: value.Type(), kind: value.Kind(), ptr: value.Pointer()}
		if cloned, ok := state.seen[visit]; ok {
			return cloned
		}
		out := reflect.MakeMapWithSize(value.Type(), value.Len())
		state.seen[visit] = out
		iter := value.MapRange()
		for iter.Next() {
			key := cloneResultReflect(iter.Key(), state)
			item := cloneResultReflect(iter.Value(), state)
			if key.IsValid() && item.IsValid() {
				out.SetMapIndex(key, item)
			}
		}
		return out
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := cloneResultVisit{typ: value.Type(), kind: value.Kind(), ptr: value.Pointer()}
		if visit.ptr != 0 {
			if cloned, ok := state.seen[visit]; ok {
				return cloned
			}
		}
		out := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		if visit.ptr != 0 {
			state.seen[visit] = out
		}
		for i := 0; i < value.Len(); i++ {
			setClonedField(out.Index(i), cloneResultReflect(value.Index(i), state))
		}
		return out
	case reflect.Array:
		out := reflect.New(value.Type()).Elem()
		for i := 0; i < value.Len(); i++ {
			setClonedField(out.Index(i), cloneResultReflect(value.Index(i), state))
		}
		return out
	case reflect.Struct:
		out := reflect.New(value.Type()).Elem()
		out.Set(value)
		for i := 0; i < value.NumField(); i++ {
			if !value.Field(i).CanInterface() || !out.Field(i).CanSet() {
				continue
			}
			setClonedField(out.Field(i), cloneResultReflect(value.Field(i), state))
		}
		return out
	default:
		return value
	}
}

func setClonedField(destination, source reflect.Value) {
	if !destination.IsValid() || !destination.CanSet() || !source.IsValid() {
		return
	}
	if source.Type().AssignableTo(destination.Type()) {
		destination.Set(source)
	}
}

// IsToolNotFound sentinel returned by Registry.Run for unknown tools.
