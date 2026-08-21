package session

import (
	"encoding/json"
	"testing"
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
