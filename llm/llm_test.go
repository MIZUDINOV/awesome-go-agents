package llm

import (
	"errors"
	"testing"
)

func TestIsContextOverflow(t *testing.T) {
	overflow := &Error{Kind: ErrorKindContextOverflow, Message: "context length exceeded"}
	if !IsContextOverflow(overflow) {
		t.Error("expected context overflow true")
	}
	// Heuristic path for wrapped provider text.
	if !IsContextOverflow(errors.New("request failed: prompt is too long")) {
		t.Error("expected heuristic overflow true")
	}
	if IsContextOverflow(errors.New("rate limit")) {
		t.Error("expected false for non-overflow")
	}
}

func TestErrorCode(t *testing.T) {
	err := &Error{Kind: ErrorKindContextOverflow, Code: "context_length_exceeded", Message: "exceeded"}
	if err.Code != "context_length_exceeded" {
		t.Errorf("code = %q", err.Code)
	}
	if !IsContextOverflow(err) {
		t.Error("expected overflow classification via Kind")
	}
}

func TestIsRetryable(t *testing.T) {
	if !IsRetryable(&Error{Kind: ErrorKindNetwork, Retryable: true}) {
		t.Error("expected retryable true")
	}
	// Stream-started failures are never blind-retried.
	if IsRetryable(&Error{Kind: ErrorKindNetwork, Retryable: true, StreamStarted: true}) {
		t.Error("expected stream-started not retryable")
	}
	if IsRetryable(&Error{Kind: ErrorKindAuth, Retryable: false}) {
		t.Error("expected auth not retryable")
	}
}

func TestUsageTotal(t *testing.T) {
	u := &Usage{InputTokens: 10, OutputTokens: 5}
	if u.TotalTokens() != 15 {
		t.Errorf("total = %d", u.TotalTokens())
	}
	if (*Usage)(nil).TotalTokens() != 0 {
		t.Error("nil total should be 0")
	}
}

func TestMessageHelpers(t *testing.T) {
	msg := NewAssistantMessage("hello", "thinking", []ToolCallRequest{{CallID: "c1", Name: "read", Arguments: []byte(`{}`)}})
	if msg.Role != RoleAssistant {
		t.Errorf("role = %s", msg.Role)
	}
	if msg.Text() != "hello" {
		t.Errorf("text = %q", msg.Text())
	}
	if msg.Reasoning() != "thinking" {
		t.Errorf("reasoning = %q", msg.Reasoning())
	}
	calls := msg.ToolCalls()
	if len(calls) != 1 || calls[0].CallID != "c1" {
		t.Errorf("calls = %+v", calls)
	}
	if byID := msg.ToolCallByID("c1"); byID == nil || byID.Name != "read" {
		t.Errorf("ToolCallByID failed")
	}
}
