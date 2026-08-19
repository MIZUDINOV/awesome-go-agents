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
	Model string
	// System holds system-level instructions, usually one RoleSystem message.
	System []Message
	// Messages is the conversation history in display order.
	Messages []Message
	// Tools are the model-facing tool definitions.
	Tools []*ToolDefinition
	// ToolChoice forces tool usage. Empty means "auto" when Tools exist.
	ToolChoice ToolChoice
	// ParallelToolCalls, when set, requests parallel tool calls. The durable
	// scheduler keeps this false so tool ordering and fencing are
	// deterministic (H-SCHED-001); nil defers to the provider/config default.
	ParallelToolCalls *bool
	// MaxTokens bounds the completion. Zero means provider default.
	MaxTokens int64
	// Config is provider-specific configuration (e.g. GenerateConfig for
	// OpenRouter) encoded as JSON.
	Config json.RawMessage
	// Stream requests streaming transport when true.
	Stream bool
	// StructuredOutputSchema, when set, requests JSON output constrained to the
	// given JSON Schema. Supported by providers that implement it.
	StructuredOutputSchema json.RawMessage
	// StructuredStrict requests provider-side strict schema adherence.
	StructuredStrict bool
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
	// Raw is the provider-specific envelope (OpenRouter metadata, usage, id).
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
