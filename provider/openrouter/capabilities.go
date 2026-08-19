package openrouter

import (
	"errors"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Sentinel errors for capability resolution.
var (
	// ErrNoCapabilitySource is returned when no capability source is
	// configured on the Client. Production wiring must provide a ModelCatalog
	// or a resolver; the client never assumes a hard-coded context window.
	ErrNoCapabilitySource = errors.New("openrouter: no model capability source configured (set ModelCatalog or CapabilitiesFor)")
)

// ErrUnknownModel reports a model id that is not present in the configured
// capability catalog. Callers must fail visibly rather than assume safe
// defaults for an unknown route (H-ANTI-017).
type ErrUnknownModel struct {
	Model string
}

func (e *ErrUnknownModel) Error() string {
	return "openrouter: model " + e.Model + " not present in capability catalog"
}

// Catalog maps model ids to capability metadata. This is the production source
// of truth for compaction thresholds; it replaces hard-coded agent constants.
//
// Unknown models must be treated as an error by the caller: there is no safe
// static fallback window, because the correct context limit is route-specific.
type Catalog map[string]llm.Capabilities

// CapabilitiesFor resolves model metadata. found is false for models that are
// not in the catalog; the caller should fail visibly instead of assuming
// defaults.
func (cat Catalog) CapabilitiesFor(model string) (llm.Capabilities, bool) {
	caps, ok := cat[model]
	return caps, ok
}

// Model returns a capability entry, registering it in the catalog.
func (cat Catalog) Model(id string, window, maxOutput int64) llm.Capabilities {
	caps := llm.Capabilities{
		Provider: ProviderName, Model: id,
		ContextWindow: window, MaxOutput: maxOutput,
		SupportsTools: true, SupportsMedia: true, SupportsSystem: true, SupportsReasoning: true,
	}
	cat[id] = caps
	return caps
}
