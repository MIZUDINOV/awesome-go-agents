package llm

// Usage is the provider-neutral token/cost accounting for a generation.
type Usage struct {
	InputTokens      int64   `json:"input_tokens"`
	OutputTokens     int64   `json:"output_tokens"`
	CachedTokens     int64   `json:"cached_tokens"`
	CacheWriteTokens int64   `json:"cache_write_tokens"`
	CostUSD          *float64 `json:"cost_usd,omitempty"`
}

// TotalTokens returns input + output.
func (u *Usage) TotalTokens() int64 {
	if u == nil {
		return 0
	}
	return u.InputTokens + u.OutputTokens
}
