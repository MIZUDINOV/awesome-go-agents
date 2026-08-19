package openrouter

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

// Usage mirrors the OpenRouter usage envelope.
type Usage struct {
	InputTokens      int64    `json:"input_tokens"`
	OutputTokens     int64    `json:"output_tokens"`
	CachedTokens     int64    `json:"cached_tokens"`
	CacheWriteTokens int64    `json:"cache_write_tokens"`
	CostUSD          *float64 `json:"cost_usd,omitempty"`
}

// ResponseMetadata is the raw OpenRouter response envelope kept on
// llm.Response.Raw for debugging, replay, and recovery.
type ResponseMetadata struct {
	ID                 string          `json:"id,omitempty"`
	Model              string          `json:"model,omitempty"`
	Provider           string          `json:"provider,omitempty"`
	GenerationID       string          `json:"generation_id,omitempty"`
	RequestID          string          `json:"request_id,omitempty"`
	Reasoning          string          `json:"reasoning,omitempty"`
	ReasoningDetails   json.RawMessage `json:"reasoning_details,omitempty"`
	Annotations        json.RawMessage `json:"annotations,omitempty"`
	OpenRouterMetadata json.RawMessage `json:"openrouter_metadata,omitempty"`
	Usage              Usage           `json:"usage"`
	AdapterVersion     string          `json:"adapter_version"`
	StreamStarted      bool            `json:"stream_started"`
	TimeToFirstChunkMS int64           `json:"time_to_first_chunk_ms,omitempty"`
	StreamDurationMS   int64           `json:"stream_duration_ms,omitempty"`
}

// RoutingMetadata is a safe, bounded projection of OpenRouter routing
// diagnostics, excluding messages and tool payloads.
type RoutingMetadata struct {
	Requested        string
	Strategy         string
	Summary          string
	SelectedProvider string
	SelectedModel    string
}

// Error is the provider error surfaced to the core as an llm.Error cause.
type Error struct {
	StatusCode     int
	Code           int
	Type           string
	ProviderCode   string
	Message        string
	RetryAfter     time.Duration
	RequestID      string
	GenerationID   string
	StreamStarted  bool
	RouterMetadata RoutingMetadata
}

func (e *Error) Error() string {
	if e == nil {
		return "openrouter error"
	}
	parts := []string{"openrouter request failed"}
	if e.Type != "" {
		parts = append(parts, "type="+e.Type)
	}
	if e.StatusCode != 0 {
		parts = append(parts, "status="+strconv.Itoa(e.StatusCode))
	}
	if e.Message != "" {
		parts = append(parts, e.Message)
	}
	return stringsJoin(parts, ": ")
}

func stringsJoin(parts []string, sep string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += sep
		}
		out += p
	}
	return out
}

// Temporary reports whether the error is a transient pre-stream failure.
func (e *Error) Temporary() bool {
	if e == nil || e.StreamStarted {
		return false
	}
	switch e.Type {
	case "rate_limit_exceeded", "provider_overloaded", "provider_unavailable", "timeout", "server", "network", "premature_eof":
		return true
	}
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode == http.StatusServiceUnavailable || e.StatusCode >= 500
}


// rawMetadata marshals an optional JSON value, returning nil for emptiness.
func rawMetadata(value any) json.RawMessage {
	if value == nil {
		return nil
	}
	if raw, ok := value.(json.RawMessage); ok {
		return raw
	}
	encoded, _ := json.Marshal(value)
	return encoded
}

func rawJSON(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}
