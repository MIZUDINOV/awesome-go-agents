// Package context assembles the model-facing context (system prompt, tools,
// history, tool results) and manages compaction: token metering, deterministic
// pruning, LLM summarization checkpoints, and emergency overflow recovery.
package context

import "unicode/utf8"

// Meter estimates token counts. When the provider returns real Usage numbers
// they should be preferred; Meter.Value is a fallback approximation.
type Meter struct {
	// charsPerToken is the heuristic ratio, ~4 chars/token.
	charsPerToken float64
}

// NewMeter returns a Meter with the default ~4 chars/token heuristic.
func NewMeter() *Meter {
	return &Meter{charsPerToken: 4.0}
}

// Estimate counts tokens for text using the heuristic.
func (m *Meter) Estimate(text string) int64 {
	if m == nil || m.charsPerToken <= 0 {
		m = NewMeter()
	}
	runes := int64(utf8.RuneCountInString(text))
	if runes == 0 {
		return 0
	}
	tokens := runes / int64(m.charsPerToken)
	if runes%int64(m.charsPerToken) != 0 {
		tokens++
	}
	// Structural overhead for turned text blocks.
	return tokens + 1
}

// EstimateBytes counts tokens for raw bytes using the heuristic.
func (m *Meter) EstimateBytes(data []byte) int64 {
	if m == nil {
		m = NewMeter()
	}
	return m.Estimate(string(data))
}

// DefaultCompactionRatios are the DSH-style trigger/target fractions.
const (
	DefaultThresholdRatio = 0.80 // compact at ~80% of context window
	DefaultTargetRatio    = 0.50 // aim to free ~50% of window
)
