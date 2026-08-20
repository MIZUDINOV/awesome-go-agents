// Package tools implements the agent's tool registry and execution pipeline.
// A Definition carries the model-facing surface (name, description, input
// schema) AND the runtime concerns (executor, renderers, concurrency, timeout)
// that the llm package intentionally excludes.
package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
)

// ExecContext carries request-scoped context a tool may need beyond the raw
// arguments. Host apps populate it (run id, workspace, sandbox binding).
type ExecContext struct {
	// SessionID / RunID are durable correlation identifiers.
	SessionID string
	RunID     string
	TurnID    string
	StepID    string
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
}

// Executor runs a tool and returns the canonical structured result.
type Executor func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error)

// Renderer converts a canonical result into a model-facing textual or JSON
// representation. If nil, the canonical value is JSON-marshalled.
type Renderer func(canonical any) (any, error)

// Presenter converts a canonical result into UI presentation metadata.
// If nil, no UI metadata is produced.
type Presenter func(canonical any) (map[string]any, error)

// Finalizer applies a definition-owned final content invariant after the
// canonical outcome has been validated and rendered.
type Finalizer func(*Result) error

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
	ExecuteCode(context.Context, ExecContext, string, string) (any, error)
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

	// InputSchema is the JSON Schema object the model sees for arguments.
	InputSchema json.RawMessage
	// OutputSchema validates the canonical result and is mandatory at register.
	OutputSchema json.RawMessage

	// Execute implements the tool. Required.
	Execute Executor
	// RenderModel optionally shapes what the model sees as the result.
	RenderModel Renderer
	// PresentUI optionally shapes what the UI sees.
	PresentUI       Presenter
	FinalizeContent Finalizer

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
type DefineToolOptions[I any, O any] struct {
	Name             string
	Description      string
	Version          string
	InputSchema      json.RawMessage
	OutputSchema     json.RawMessage
	Timeout          time.Duration
	MutatesWorkspace bool
	ConcurrencySafe  func(I) bool
	Execute          func(context.Context, ExecContext, I) (O, error)
	RenderModel      func(O) (any, error)
	PresentUI        func(O) (map[string]any, error)
	FinalizeContent  func(*Result) error
}

// DefineTool converts a typed definition into the runtime representation.
func DefineTool[I any, O any](opts DefineToolOptions[I, O]) *Definition {
	inputSchema := append(json.RawMessage(nil), opts.InputSchema...)
	if len(inputSchema) == 0 {
		inputSchema = json.RawMessage(`{"type":"object","additionalProperties":false}`)
	}
	outputSchema := append(json.RawMessage(nil), opts.OutputSchema...)
	if len(outputSchema) == 0 {
		outputSchema = append(json.RawMessage(nil), AnyOutputSchema...)
	}
	return &Definition{
		Name: opts.Name, Description: opts.Description, Version: opts.Version,
		InputSchema: inputSchema, OutputSchema: outputSchema,
		Timeout: opts.Timeout, MutatesWorkspace: opts.MutatesWorkspace,
		ConcurrencySafeFor: func(input json.RawMessage) bool {
			var args I
			return opts.ConcurrencySafe != nil && json.Unmarshal(input, &args) == nil && opts.ConcurrencySafe(args)
		},
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) {
			if opts.Execute == nil {
				return nil, ErrInvalidArguments
			}
			var args I
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, err
			}
			return opts.Execute(ctx, ec, args)
		},
		RenderModel: func(canonical any) (any, error) {
			if opts.RenderModel == nil {
				return canonical, nil
			}
			value, ok := canonical.(O)
			if !ok {
				return nil, ErrInvalidOutput
			}
			return opts.RenderModel(value)
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			if opts.PresentUI == nil {
				return nil, nil
			}
			value, ok := canonical.(O)
			if !ok {
				return nil, ErrInvalidOutput
			}
			return opts.PresentUI(value)
		},
		FinalizeContent: opts.FinalizeContent,
	}
}

// Result is the three-part tool outcome per the DSH model:
// Canonical (durable), ModelFacing (what goes to the model), UI (presentation).
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
	UI                 map[string]any         `json:"ui,omitempty"`
	Meta               map[string]any         `json:"meta,omitempty"`
	Content            []session.ContentBlock `json:"content,omitempty"`
	AdditionalContexts []llm.Message          `json:"additional_contexts,omitempty"`
	ConcludesTurn      bool                   `json:"concludes_turn,omitempty"`
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
