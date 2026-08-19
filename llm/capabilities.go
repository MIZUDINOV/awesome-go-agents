package llm

// Capabilities describes the fixed or configured capacity of a model. It is
// resolved per (provider, model) and drives compaction thresholds rather than
// a single hard-coded agent constant.
type Capabilities struct {
	Provider string
	Model    string

	// ContextWindow is the model's maximum context (input) window in tokens.
	ContextWindow int64
	// MaxOutput is the model's maximum completion length in tokens.
	MaxOutput int64

	SupportsTools   bool
	SupportsMedia   bool
	SupportsSystem  bool
	SupportsReasoning bool
}
