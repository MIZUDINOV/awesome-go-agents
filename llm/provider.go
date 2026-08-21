package llm

import (
	"context"
	"encoding/json"
	"time"
)

// ToolChoice controls how the model selects tools when Tools are present.
type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// Request is a provider-neutral generation request. Config carries
// provider-specific options (temperature, reasoning, routing, ...) as opaque
// JSON; each provider decodes only what it recognises.
type Request struct {
	Model string `json:"model"`
	// System holds system-level instructions, usually one RoleSystem message.
	System []Message `json:"system,omitempty"`
	// Messages is the conversation history in display order.
	Messages []Message `json:"messages,omitempty"`
	// Tools are the model-facing tool definitions.
	Tools []*ToolDefinition `json:"tools,omitempty"`
	// ToolChoice forces tool usage. Empty means "auto" when Tools exist.
	ToolChoice ToolChoice `json:"tool_choice,omitempty"`
	// ParallelToolCalls, when set, requests parallel tool calls. The durable
	// scheduler still commits every result in model order and only overlaps
	// definitions explicitly classified concurrency-safe.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
	// MaxTokens bounds the completion. Zero means provider default.
	MaxTokens int64 `json:"max_tokens,omitempty"`
	// Config is provider-specific configuration encoded as JSON.
	Config json.RawMessage `json:"config,omitempty"`
	// Stream requests streaming transport when true.
	Stream bool `json:"stream,omitempty"`
	// StructuredOutputSchema, when set, requests JSON output constrained to the
	// given JSON Schema. Supported by providers that implement it.
	StructuredOutputSchema json.RawMessage `json:"structured_output_schema,omitempty"`
	// StructuredStrict requests provider-side strict schema adherence.
	StructuredStrict bool `json:"structured_strict,omitempty"`
	// Capabilities is the resolved provider/model snapshot used by the loop;
	// providers may ignore it on the wire.
	Capabilities *Capabilities `json:"capabilities,omitempty"`
}

// FinishReason is the provider-neutral completion classification.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonOther         FinishReason = "other"
)

// Response is the completed generation result. Message is the full assistant
// reply (text, reasoning, tool calls). Raw preserves the provider envelope for
// debugging and replay.
type Response struct {
	Message      *Message
	FinishReason FinishReason
	Usage        *Usage
	Model        string
	Provider     string
	// ProviderResponseID / RequestID are correlation identifiers surfaced in
	// telemetry and recovery.
	ProviderResponseID string
	RequestID          string
	// Latency is the total wall time of the call.
	Latency time.Duration
	// Raw is the provider-specific envelope retained for debugging and replay.
	Raw json.RawMessage
}

// Provider streams a completion. cb is invoked per event; for non-streaming
// requests cb may be nil and a single Response is returned. Providers must
// always return a final Response unless the request failed.
type Provider interface {
	Name() string
	Generate(ctx context.Context, req *Request, cb StreamCallback) (*Response, error)
	Capabilities(ctx context.Context, model string) (Capabilities, error)
}
