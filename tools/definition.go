// Package tools implements the agent's tool registry and execution pipeline.
// A Definition carries the model-facing surface (name, description, input
// schema) AND the runtime concerns (executor, renderers, concurrency, timeout)
// that the llm package intentionally excludes.
package tools

import (
	"context"
	"encoding/json"
	"time"
)

// ExecContext carries request-scoped context a tool may need beyond the raw
// arguments. Host apps populate it (run id, workspace, sandbox binding).
type ExecContext struct {
	// SessionID / RunID are durable correlation identifiers.
	SessionID string
	RunID     string
	// Vars carries arbitrary host-provided bindings (tool-agnostic).
	Vars map[string]any
}

// Executor runs a tool and returns the canonical structured result.
type Executor func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error)

// Renderer converts a canonical result into a model-facing textual or JSON
// representation. If nil, the canonical value is JSON-marshalled.
type Renderer func(canonical any) (any, error)

// Presenter converts a canonical result into UI presentation metadata.
// If nil, no UI metadata is produced.
type Presenter func(canonical any) (map[string]any, error)

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
	PresentUI Presenter

	// ConcurrencySafe marks the tool safe to run in parallel with other tools.
	ConcurrencySafe bool
	// MutatesWorkspace marks the tool as side-effecting (used for scheduling).
	MutatesWorkspace bool
	// Timeout bounds a single execution. Zero uses the registry default.
	Timeout time.Duration
}

// Result is the three-part tool outcome per the DSH model:
// Canonical (durable), ModelFacing (what goes to the model), UI (presentation).
type Result struct {
	// Name/CallID bind the result to the originating model tool call.
	Name   string
	CallID string

	// Canonical is the structured execution value (may not be persisted to the
	// LLM history).
	Canonical any
	// ModelFacing is what the model sees (text or structured).
	ModelFacing any
	// UI is presentation metadata (diffs, paths).
	UI map[string]any
}

// IsToolNotFound sentinel returned by Registry.Run for unknown tools.
