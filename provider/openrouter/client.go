package openrouter

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/wzhooh/agentkit/llm"
)

// Client is the OpenRouter llm.Provider implementation.
type Client struct {
	APIKey      string
	BaseURL     string
	HTTPClient  *http.Client
	Logger      *slog.Logger
	HTTPReferer string
	AppTitle    string

	// CapabilitiesFor overrides model capability resolution (defaults to the
	// built-in static defaults). Used for tests and model catalogs.
	CapabilitiesFor func(model string) llm.Capabilities
}

// New returns an OpenRouter Provider. API key is required.
func New(apiKey string, logger *slog.Logger) *Client {
	c := &Client{APIKey: apiKey, BaseURL: DefaultBaseURL, Logger: logger}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
	if c.HTTPClient == nil {
		c.HTTPClient = defaultHTTPClient()
	}
	if strings.TrimSpace(c.BaseURL) == "" {
		c.BaseURL = DefaultBaseURL
	}
	return c
}

func (c *Client) Name() string { return ProviderName }

// Capabilities resolves model capacity. Static defaults are returned unless a
// caller provides CapabilitiesFor (production wires the model catalog).
func (c *Client) Capabilities(_ context.Context, model string) (llm.Capabilities, error) {
	if c.CapabilitiesFor != nil {
		return c.CapabilitiesFor(model), nil
	}
	return llm.Capabilities{
		Provider: ProviderName, Model: model,
		ContextWindow: 1_000_000, MaxOutput: 256_000,
		SupportsTools: true, SupportsMedia: true, SupportsSystem: true, SupportsReasoning: true,
	}, nil
}

func (c *Client) Generate(ctx context.Context, req *llm.Request, cb llm.StreamCallback) (*llm.Response, error) {
	if strings.TrimSpace(c.APIKey) == "" {
		return nil, &llm.Error{Kind: llm.ErrorKindAuth, Message: "openrouter API key is required"}
	}
	config, err := decodeConfig(req.Config)
	if err != nil {
		return nil, err
	}
	stream := cb != nil
	payload, err := buildRequest(req.Model, req, config, stream)
	if err != nil {
		return nil, &llm.Error{Kind: llm.ErrorKindInvalidRequest, Message: err.Error()}
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, &llm.Error{Kind: llm.ErrorKindInvalidRequest, Message: err.Error()}
	}

	callCtx, cancel := context.WithTimeout(ctx, config.requestTimeout())
	defer cancel()
	httpReq, err := http.NewRequestWithContext(callCtx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return nil, &llm.Error{Kind: llm.ErrorKindNetwork, Message: err.Error()}
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("X-OpenRouter-Metadata", "enabled")
	if c.HTTPReferer != "" {
		httpReq.Header.Set("HTTP-Referer", c.HTTPReferer)
	}
	if c.AppTitle != "" {
		httpReq.Header.Set("X-Title", c.AppTitle)
	}

	started := time.Now()
	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		if cause := context.Cause(callCtx); cause != nil {
			return nil, cause
		}
		return nil, &llm.Error{Kind: llm.ErrorKindNetwork, Message: err.Error()}
	}
	defer resp.Body.Close()
	requestID := resp.Header.Get("x-request-id")
	generationID := resp.Header.Get("X-Generation-Id")
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		providerErr := decodeHTTPError(resp, requestID, generationID)
		c.logError(payload, req.Model, providerErr)
		return nil, wrapProviderError(providerErr)
	}

	acc := newAccumulator(req.Model, requestID, generationID)
	acc.startedAt = started
	if cb == nil {
		if err := decodeNonStreaming(callCtx, resp.Body, acc); err != nil {
			return nil, c.mapDecodeError(err, acc)
		}
	} else if err := consumeSSE(callCtx, resp.Body, acc, cb); err != nil {
		if cause := context.Cause(callCtx); cause != nil {
			return nil, cause
		}
		var providerErr *Error
		if errors.As(err, &providerErr) {
			return nil, wrapProviderError(providerErr)
		}
		if errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, &llm.Error{Kind: llm.ErrorKindPrematureEOF, Message: err.Error(), StreamStarted: acc.streamStarted}
		}
		return nil, err
	}
	result, err := acc.response(callCtx, cb, time.Since(started), req)
	if err != nil {
		return nil, err
	}
	// Emit the terminal done event.
	if cb != nil {
		if err := cb(ctx, llm.StreamEvent{Type: llm.StreamEventDone, Response: result}); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (c *Client) mapDecodeError(err error, acc *accumulator) error {
	var providerErr *Error
	if errors.As(err, &providerErr) {
		return wrapProviderError(providerErr)
	}
	return err
}

func (c *Client) logError(payload map[string]any, model string, providerErr *Error) {
	c.Logger.Warn("openrouter model request rejected",
		"run_id", requestMetadataValue(payload, "run_id"),
		"role", requestMetadataValue(payload, "role"),
		"model", model,
		"status", providerErr.StatusCode,
		"request_id", providerErr.RequestID,
		"generation_id", providerErr.GenerationID,
		"provider_error_type", providerErr.Type,
		"provider_code", providerErr.ProviderCode,
		"router_strategy", providerErr.RouterMetadata.Strategy,
		"router_summary", providerErr.RouterMetadata.Summary,
		"selected_provider", providerErr.RouterMetadata.SelectedProvider,
		"selected_model", providerErr.RouterMetadata.SelectedModel,
	)
}

func requestMetadataValue(payload map[string]any, key string) string {
	metadata, ok := payload["metadata"].(map[string]string)
	if !ok {
		return ""
	}
	return metadata[key]
}

// wrapProviderError converts an OpenRouter Error into the provider-neutral
// llm.Error with correct Kind and Retryable classification.
func wrapProviderError(e *Error) *llm.Error {
	kind := classifyError(e)
	return &llm.Error{
		Kind: kind, Message: e.Message, ProviderCode: e.ProviderCode,
		StatusCode: e.StatusCode, Retryable: e.Temporary(),
		StreamStarted: e.StreamStarted, Cause: e,
	}
}

func classifyError(e *Error) llm.ErrorKind {
	switch e.Type {
	case "rate_limit_exceeded":
		return llm.ErrorKindRateLimit
	case "provider_overloaded", "provider_unavailable", "timeout", "server", "network", "premature_eof":
		return llm.ErrorKindProvider
	case "authentication_error", "invalid_api_key", "permission":
		return llm.ErrorKindAuth
	case "invalid_request_error", "bad_request":
		return llm.ErrorKindInvalidRequest
	}
	if e.StatusCode == http.StatusUnauthorized || e.StatusCode == http.StatusForbidden {
		return llm.ErrorKindAuth
	}
	if e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= 500 {
		return llm.ErrorKindProvider
	}
	return llm.ErrorKindUnknown
}

func defaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	transport.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	transport.TLSHandshakeTimeout = 10 * time.Second
	transport.ResponseHeaderTimeout = 30 * time.Second
	transport.ExpectContinueTimeout = time.Second
	transport.IdleConnTimeout = 90 * time.Second
	return &http.Client{Transport: transport}
}

func decodeConfig(value json.RawMessage) (GenerateConfig, error) {
	if len(value) == 0 {
		return GenerateConfig{}, nil
	}
	var config GenerateConfig
	if err := json.Unmarshal(value, &config); err != nil {
		return GenerateConfig{}, fmt.Errorf("decode OpenRouter config: %w", err)
	}
	if err := config.validate(); err != nil {
		return GenerateConfig{}, err
	}
	return config, nil
}

var _ = fmt.Sprintf
