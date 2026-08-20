package llm

// Capabilities describes the fixed or configured capacity of a model. It is
// resolved per (provider, model) and drives compaction thresholds rather than
// a single hard-coded agent constant.
type Capabilities struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`

	// ContextWindow is the model's maximum context (input) window in tokens.
	ContextWindow int64 `json:"context_window"`
	// MaxOutput is the model's maximum completion length in tokens.
	MaxOutput int64 `json:"max_output"`

	SupportsTools             bool `json:"supports_tools"`
	SupportsMedia             bool `json:"supports_media"`
	SupportsSystem            bool `json:"supports_system"`
	SupportsReasoning         bool `json:"supports_reasoning"`
	SupportsStreaming         bool `json:"supports_streaming"`
	SupportsParallelToolCalls bool `json:"supports_parallel_tool_calls"`
	SupportsStructuredOutput  bool `json:"supports_structured_output"`
}
