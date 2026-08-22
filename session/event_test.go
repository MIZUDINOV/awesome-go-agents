package session

import (
	"encoding/json"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

func TestRequestErrorPayloadWithRetryable(t *testing.T) {
	var payload map[string]any
	if err := json.Unmarshal(RequestErrorPayloadWithRetryable("rate_limit", "try again", false, true), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["retryable"] != true || payload["stream_started"] != false {
		t.Fatalf("unexpected request error payload: %#v", payload)
	}
}

func TestRequestErrorPayloadWithMetadata(t *testing.T) {
	var payload map[string]any
	metadata := &llm.RequestMetadata{Provider: "test", RequestID: "req-1", ProviderResponseID: "resp-1"}
	if err := json.Unmarshal(RequestErrorPayloadWithMetadata("provider", "failed", true, false, metadata), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["provider"] != "test" || payload["request_id"] != "req-1" || payload["provider_response_id"] != "resp-1" || payload["stream_started"] != true {
		t.Fatalf("unexpected request error payload: %#v", payload)
	}
}

func TestRequestHeaderPayloadWithMetadata(t *testing.T) {
	var payload map[string]any
	metadata := RequestHeaderMetadata{PromptVersion: "prompt-1", PromptHash: "prompt-hash", ToolsHash: "tools-hash"}
	data := RequestHeaderPayloadWithSnapshotAndMetadata("model", "provider", []string{"system"}, []string{"tool"}, "config", "request", nil, metadata, json.RawMessage(`{"wire":true}`))
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["prompt_version"] != "prompt-1" || payload["prompt_hash"] != "prompt-hash" || payload["tools_hash"] != "tools-hash" {
		t.Fatalf("unexpected request header payload: %#v", payload)
	}
}
