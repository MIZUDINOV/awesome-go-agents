// Package tools implements the agent's tool registry and execution pipeline.
// A Definition carries the model-facing surface (name, description, input
// schema) AND the runtime concerns (executor, renderers, concurrency, timeout)
// that the llm package intentionally excludes.
package tools

import (
	"context"
	"encoding/json"
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

// Definition describes one tool. InputSchema must be a JSON Schema object for
// the arguments; OutputSchema, when set, validates the canonical result.
type Definition struct {
	Name        string
	Description string
	Version     string

	// InputSchema is the JSON Schema object the model sees for arguments.
	InputSchema json.RawMessage
	// OutputSchema optionally validates the canonical result.
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
		outputSchema = json.RawMessage(`{}`)
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

// cloneResultValue copies the JSON-shaped values used by canonical/model/UI
// projections without changing the concrete type of typed tool results.
func cloneResultValue(value any) any {
	switch typed := value.(type) {
	case nil:
		return nil
	case json.RawMessage:
		return append(json.RawMessage(nil), typed...)
	case []byte:
		return append([]byte(nil), typed...)
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = cloneResultValue(item)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = cloneResultValue(item)
		}
		return out
	case []string:
		return append([]string(nil), typed...)
	case map[string]string:
		return map[string]string(typed)
	default:
		return value
	}
}

// IsToolNotFound sentinel returned by Registry.Run for unknown tools.
