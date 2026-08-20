package openrouter

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

func TestClientGenerateNonStreamingToolCallsFromWire(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		var payload struct {
			Stream   bool              `json:"stream"`
			Tools    []json.RawMessage `json:"tools"`
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if payload.Stream || len(payload.Tools) != 1 || len(payload.Messages) != 1 {
			t.Errorf("wire request stream=%v tools=%d messages=%d", payload.Stream, len(payload.Tools), len(payload.Messages))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"gen-wire","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.txt\"}"}},{"id":"call-2","type":"function","function":{"name":"grep","arguments":"{\"pattern\":\"TODO\"}"}}]},"finish_reason":"tool_calls"}]}`))
	}))
	defer server.Close()

	client := &Client{APIKey: "test-key", BaseURL: server.URL, HTTPClient: server.Client()}
	response, err := client.Generate(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{*llm.NewUserMessage("inspect")},
		Tools: []*llm.ToolDefinition{{
			Name:        "read",
			Description: "read a file",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if requests != 1 {
		t.Fatalf("wire requests = %d, want 1", requests)
	}
	calls := response.Message.ToolCalls()
	if len(calls) != 2 || calls[0].CallID != "call-1" || calls[1].CallID != "call-2" {
		t.Fatalf("tool calls = %+v", calls)
	}
	if response.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("finish reason = %q, want %q", response.FinishReason, llm.FinishReasonToolCalls)
	}
}

func TestClientGenerateStreamingSSEAssemblesFinalMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if !payload.Stream {
			t.Error("stream flag = false, want true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"gen-stream\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"gen-stream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	client := &Client{APIKey: "test-key", BaseURL: server.URL, HTTPClient: server.Client()}
	var events []llm.StreamEvent
	response, err := client.Generate(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{*llm.NewUserMessage("hello")},
	}, func(_ context.Context, event llm.StreamEvent) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if response == nil || response.Message == nil || response.Message.Text() != "hello world" {
		t.Fatalf("response = %+v", response)
	}
	if len(events) != 3 || events[0].Type != llm.StreamEventText || events[1].Type != llm.StreamEventText || events[2].Type != llm.StreamEventDone {
		t.Fatalf("stream events = %+v", events)
	}
}

func TestDecodeNonStreamingToolCallsPreservesWireOrder(t *testing.T) {
	body := `{"id":"gen-1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"read","arguments":"{\"file_path\":\"a.txt\"}"}},{"id":"call-2","type":"function","function":{"name":"grep","arguments":"{\"pattern\":\"TODO\"}"}}]},"finish_reason":"tool_calls"}]}`
	acc := newAccumulator("m", "req-1", "gen-1")
	if err := decodeNonStreaming(context.Background(), strings.NewReader(body), acc); err != nil {
		t.Fatalf("decodeNonStreaming: %v", err)
	}
	response, err := acc.response(context.Background(), nil, 0)
	if err != nil {
		t.Fatalf("response: %v", err)
	}
	calls := response.Message.ToolCalls()
	if len(calls) != 2 {
		t.Fatalf("tool calls = %d, want 2", len(calls))
	}
	if calls[0].CallID != "call-1" || calls[0].Name != "read" || string(calls[0].Arguments) != `{"file_path":"a.txt"}` {
		t.Fatalf("first tool call = %+v", calls[0])
	}
	if calls[1].CallID != "call-2" || calls[1].Name != "grep" || string(calls[1].Arguments) != `{"pattern":"TODO"}` {
		t.Fatalf("second tool call = %+v", calls[1])
	}
	if response.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("finish reason = %q, want %q", response.FinishReason, llm.FinishReasonToolCalls)
	}
}

func TestStreamingToolCallsPreserveSparseIndexes(t *testing.T) {
	acc := newAccumulator("m", "req-1", "gen-1")
	chunk := chatChunk{Choices: []chatChoice{{Delta: chatDelta{ToolCalls: []chatToolCall{
		{Index: 3, ID: "call-1", Function: chatToolFunction{Name: "read", Arguments: `{"file_path":"a.txt"}`}},
		{Index: 9, ID: "call-2", Function: chatToolFunction{Name: "grep", Arguments: `{"pattern":"TODO"}`}},
	}}}}}
	if err := acc.addChunk(context.Background(), chunk, nil, false); err != nil {
		t.Fatalf("addChunk: %v", err)
	}
	response, err := acc.response(context.Background(), nil, 0)
	if err != nil {
		t.Fatalf("response: %v", err)
	}
	calls := response.Message.ToolCalls()
	if len(calls) != 2 || calls[0].CallID != "call-1" || calls[1].CallID != "call-2" {
		t.Fatalf("tool calls = %+v", calls)
	}
}

func TestBuildRequestMessagesAndTools(t *testing.T) {
	req := &llm.Request{
		Model:  "deepseek/deepseek-v3",
		System: []llm.Message{*llm.NewTextMessage(llm.RoleSystem, "sys")},
		Messages: []llm.Message{
			*llm.NewUserMessage("hello"),
			*llm.NewAssistantMessage("hi", "thinking", []llm.ToolCallRequest{{CallID: "c1", Name: "read", Arguments: []byte(`{"path":"a"}`)}}),
			*llm.NewToolResultMessage(llm.ToolCallResult{CallID: "c1", Name: "read", Output: []byte(`{"ok":true}`)}),
		},
		Tools: []*llm.ToolDefinition{
			{Name: "read", Description: "read", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		MaxTokens: 100,
	}
	config := GenerateConfig{StrictToolNames: []string{"read"}}
	payload, err := buildRequest(req.Model, req, config, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	if payload["model"] != "deepseek/deepseek-v3" {
		t.Errorf("model = %v", payload["model"])
	}
	msgs := payload["messages"].([]chatMessage)
	if len(msgs) != 4 { // system + user + assistant + tool
		t.Fatalf("messages = %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Errorf("msg0 role = %s", msgs[0].Role)
	}
	if msgs[2].Role != "assistant" || len(msgs[2].ToolCalls) != 1 || msgs[2].ToolCalls[0].ID != "c1" {
		t.Errorf("assistant tool call mismatch: %+v", msgs[2])
	}
	if msgs[3].Role != "tool" || msgs[3].ToolCallID != "c1" {
		t.Errorf("tool message mismatch: %+v", msgs[3])
	}
	tools := payload["tools"].([]chatTool)
	if len(tools) != 1 {
		t.Fatalf("tools = %d", len(tools))
	}
	if !tools[0].Function.Strict {
		t.Error("expected strict tool")
	}
}

func TestWrapProviderErrorClassification(t *testing.T) {
	e := &Error{Type: "rate_limit_exceeded", StatusCode: 429, StreamStarted: false}
	llmerr := wrapProviderError(e)
	if llmerr.Kind != llm.ErrorKindRateLimit {
		t.Errorf("kind = %s", llmerr.Kind)
	}
	if !llmerr.Retryable {
		t.Error("expected retryable")
	}
	if llmerr.Code != "rate_limit_exceeded" {
		t.Errorf("code = %q", llmerr.Code)
	}
	// Structured overflow type must classify as context overflow.
	overflow := wrapProviderError(&Error{Type: "context_length_exceeded", StatusCode: 400})
	if overflow.Kind != llm.ErrorKindContextOverflow {
		t.Errorf("overflow kind = %s", overflow.Kind)
	}
	// Message heuristic for invalid_request_error without structured type.
	heuristic := wrapProviderError(&Error{Type: "invalid_request_error", StatusCode: 400, Message: "This model's maximum context length is 128000 tokens. You requested 200000 tokens."})
	if heuristic.Kind != llm.ErrorKindContextOverflow {
		t.Errorf("heuristic overflow kind = %s", heuristic.Kind)
	}
}

func TestCatalogCapabilities(t *testing.T) {
	cat := Catalog{}
	cat.Model("m-small", 65536, 4096)
	resolver, ok := cat.CapabilitiesFor("m-small")
	if !ok {
		t.Fatal("known model not found")
	}
	if resolver.ContextWindow != 65536 || resolver.MaxOutput != 4096 {
		t.Errorf("resolved = %+v", resolver)
	}
	// Unknown model must fail visibly: no unsafe static fallback window.
	if _, ok := cat.CapabilitiesFor("unknown"); ok {
		t.Error("expected unknown model to be missing from catalog")
	}
	client := &Client{}
	if _, err := client.Capabilities(nil, "m-small"); !errors.Is(err, ErrNoCapabilitySource) {
		t.Errorf("expected ErrNoCapabilitySource, got %v", err)
	}
	client.ModelCatalog = cat
	if _, err := client.Capabilities(nil, "unknown"); err == nil {
		t.Error("expected unknown model error")
	}
	caps, err := client.Capabilities(nil, "m-small")
	if err != nil {
		t.Fatalf("capabilities: %v", err)
	}
	if caps.ContextWindow != 65536 {
		t.Errorf("window = %d", caps.ContextWindow)
	}
}

func TestValidateConfig(t *testing.T) {
	config := GenerateConfig{Metadata: map[string]string{"a": "b"}}
	if err := config.validate(); err != nil {
		t.Errorf("valid config rejected: %v", err)
	}
	bad := GenerateConfig{Stop: []string{"1", "2", "3", "4", "5"}}
	if err := bad.validate(); err == nil {
		t.Error("expected stop > 4 to fail")
	}
}
