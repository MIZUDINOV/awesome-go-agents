package openrouter

import "github.com/MIZUDINOV/awesome-go-agents/llm"

// Catalog maps model ids to capability metadata. This is the production source
// of truth for compaction thresholds; it replaces hard-coded agent constants.
type Catalog map[string]llm.Capabilities

// CapabilitiesFor returns a resolver that looks up model metadata and falls
// back to static defaults for unknown models.
func (cat Catalog) CapabilitiesFor(model string) llm.Capabilities {
	if caps, ok := cat[model]; ok {
		return caps
	}
	return llm.Capabilities{
		Provider: ProviderName, Model: model,
		ContextWindow: 1_000_000, MaxOutput: 256_000,
		SupportsTools: true, SupportsMedia: true, SupportsSystem: true, SupportsReasoning: true,
	}
}

// Model returns a capability entry.
func (cat Catalog) Model(id string, window, maxOutput int64) llm.Capabilities {
	caps := llm.Capabilities{
		Provider: ProviderName, Model: id,
		ContextWindow: window, MaxOutput: maxOutput,
		SupportsTools: true, SupportsMedia: true, SupportsSystem: true, SupportsReasoning: true,
	}
	cat[id] = caps
	return caps
}
