package openrouter

import (
	"encoding/json"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

func TestBuildRequestMessagesAndTools(t *testing.T) {
	req := &llm.Request{
		Model: "deepseek/deepseek-v3",
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
}

func TestCatalogCapabilities(t *testing.T) {
	cat := Catalog{}
	cat.Model("m-small", 65536, 4096)
	caps, _ := (&Client{}).Capabilities(nil, "m-small")
	_ = caps
	resolver := cat.CapabilitiesFor("m-small")
	if resolver.ContextWindow != 65536 || resolver.MaxOutput != 4096 {
		t.Errorf("resolved = %+v", resolver)
	}
	// Unknown model falls back to defaults.
	unknown := cat.CapabilitiesFor("unknown")
	if unknown.ContextWindow != 1_000_000 {
		t.Errorf("unknown fallback window = %d", unknown.ContextWindow)
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
