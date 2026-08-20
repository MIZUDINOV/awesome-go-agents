// Package openrouter implements llm.Provider for the OpenRouter Chat
// Completions API. It is a concrete provider inside agentkit so the library
// ships a working example; the core depends only on llm.Provider.
package openrouter

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"time"
	"unicode/utf8"
)

const (
	ProviderName          = "openrouter"
	DefaultBaseURL        = "https://openrouter.ai/api/v1"
	DefaultRequestTimeout = 10 * time.Minute
	MaxSSEEventBytes      = 2 << 20
	AdapterVersion        = "openrouter-chat-completions/v1"
)

// ProviderPreferences controls OpenRouter provider routing.
type ProviderPreferences struct {
	RequireParameters bool     `json:"require_parameters"`
	DataCollection    string   `json:"data_collection,omitempty"`
	ZDR               bool     `json:"zdr"`
	Order             []string `json:"order,omitempty"`
	AllowFallbacks    *bool    `json:"allow_fallbacks,omitempty"`
}

// ReasoningConfig enables/controls model reasoning.
type ReasoningConfig struct {
	Enabled *bool  `json:"enabled,omitempty"`
	Effort  string `json:"effort,omitempty"`
	Exclude *bool  `json:"exclude,omitempty"`
}

// PDFConfig enables PDF parsing.
type PDFConfig struct {
	Engine string `json:"engine,omitempty"`
}

// GenerateConfig is the provider-specific OpenRouter configuration embedded in
// llm.Request.Config. It surfaces stable OpenRouter features without coupling
// the core to a provider SDK.
type GenerateConfig struct {
	Models               []string                   `json:"models,omitempty"`
	MaxTokens            int64                      `json:"max_tokens,omitempty"`
	ParallelToolCalls    *bool                      `json:"parallel_tool_calls,omitempty"`
	StrictToolNames      []string                   `json:"strict_tool_names,omitempty"`
	Temperature          *float64                   `json:"temperature,omitempty"`
	TopP                 *float64                   `json:"top_p,omitempty"`
	Seed                 *int64                     `json:"seed,omitempty"`
	Stop                 []string                   `json:"stop,omitempty"`
	Provider             *ProviderPreferences       `json:"provider,omitempty"`
	Reasoning            *ReasoningConfig           `json:"reasoning,omitempty"`
	PDF                  *PDFConfig                 `json:"pdf,omitempty"`
	ContextCompression   bool                       `json:"context_compression,omitempty"`
	SessionID            string                     `json:"session_id,omitempty"`
	Metadata             map[string]string          `json:"metadata,omitempty"`
	ServiceTier          string                     `json:"service_tier,omitempty"`
	ExtraBody            map[string]json.RawMessage `json:"extra_body,omitempty"`
	RequestTimeoutMillis int64                      `json:"request_timeout_ms,omitempty"`
}

func (c GenerateConfig) requestTimeout() time.Duration {
	if c.RequestTimeoutMillis <= 0 {
		return DefaultRequestTimeout
	}
	return time.Duration(c.RequestTimeoutMillis) * time.Millisecond
}

func (c GenerateConfig) validate() error {
	if utf8.RuneCountInString(c.SessionID) > 256 {
		return errNew("openrouter session_id exceeds 256 characters")
	}
	if len(c.Metadata) > 16 {
		return errNew("openrouter metadata has more than 16 entries")
	}
	for key, value := range c.Metadata {
		if utf8.RuneCountInString(key) > 64 {
			return fmt.Errorf("openrouter metadata key %q exceeds 64 characters", key)
		}
		if utf8.RuneCountInString(value) > 512 {
			return fmt.Errorf("openrouter metadata value for %q exceeds 512 characters", key)
		}
	}
	if len(c.Stop) > 4 {
		return errNew("openrouter stop has more than 4 sequences")
	}
	if c.Temperature != nil && (!isFinite(*c.Temperature) || *c.Temperature < 0 || *c.Temperature > 2) {
		return errNew("openrouter temperature must be between 0 and 2")
	}
	if c.TopP != nil && (!isFinite(*c.TopP) || *c.TopP < 0 || *c.TopP > 1) {
		return errNew("openrouter top_p must be between 0 and 1")
	}
	return validateExtraBody(c.ExtraBody)
}

func validateExtraBody(extra map[string]json.RawMessage) error {
	reserved := map[string]struct{}{
		"model": {}, "models": {}, "messages": {}, "tools": {}, "tool_choice": {},
		"parallel_tool_calls": {}, "stream": {}, "stream_options": {}, "response_format": {},
		"max_tokens": {}, "max_completion_tokens": {}, "provider": {}, "temperature": {}, "top_p": {}, "seed": {},
		"stop": {}, "reasoning": {}, "session_id": {}, "metadata": {}, "service_tier": {}, "plugins": {},
	}
	for key, value := range extra {
		if _, blocked := reserved[key]; blocked {
			return fmt.Errorf("extra_body cannot override %q", key)
		}
		if !json.Valid(value) {
			return fmt.Errorf("extra_body %q is not valid JSON", key)
		}
	}
	return nil
}

func isFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func errNew(message string) error { return fmt.Errorf("%s", message) }

func parseRetryAfter(value string) time.Duration {
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil {
		return max(time.Until(when), 0)
	}
	return 0
}
