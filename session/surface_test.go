package session

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

func TestSurfaceProjectsOnlySurfaceEvents(t *testing.T) {
	events := []Event{
		{Seq: 1, Type: EventTurnStart, Timestamp: now(), Data: json.RawMessage(`{}`)},
		{Seq: 2, Type: EventStepStart, Timestamp: now(), Data: json.RawMessage(`{}`)},
		{Seq: 3, Type: EventUserMessage, Timestamp: now(), Data: UserText("сделай лендинг")},
		{Seq: 4, Type: EventAssistantChunk, Timestamp: now(), Data: json.RawMessage(`{"text":"Привет"}`)},
		{Seq: 5, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("Привет", "", nil)},
		{Seq: 6, Type: EventStepEnd, Timestamp: now(), Data: json.RawMessage(`{}`)},
		{Seq: 7, Type: EventTurnEnd, Timestamp: now(), Data: json.RawMessage(`{}`)},
	}
	s := NewSurface(SurfaceSpec{})
	msgs, err := s.DeriveMessages(events)
	if err != nil {
		t.Fatalf("DeriveMessages: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != llm.RoleUser || msgs[0].Text() != "сделай лендинг" {
		t.Errorf("user message wrong: %+v", msgs[0])
	}
	if msgs[1].Role != llm.RoleAssistant || msgs[1].Text() != "Привет" {
		t.Errorf("assistant message wrong: %+v", msgs[1])
	}
}

func TestSurfaceToolCallsAndResults(t *testing.T) {
	call := ToolCall{CallID: "call_1", Name: "read", Arguments: json.RawMessage(`{"file_path":"a.ts"}`)}
	events := []Event{
		{Seq: 1, Type: EventUserMessage, Timestamp: now(), Data: UserText("прочитай a.ts")},
		{Seq: 2, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("", "", []ToolCall{call})},
		{Seq: 3, Type: EventToolCall, Timestamp: now(), Data: json.RawMessage(`{"call_id":"call_1"}`)},
		{Seq: 4, Type: EventToolResult, Timestamp: now(), Data: ToolResultPayload("call_1", "read", json.RawMessage(`{"content":"line 1"}`), false), SourceSeqs: []uint64{2, 3}},
	}
	s := NewSurface(SurfaceSpec{})
	msgs, err := s.DeriveMessages(events)
	if err != nil {
		t.Fatalf("DeriveMessages: %v", err)
	}
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	// assistant message
	assistant := msgs[1]
	calls := assistant.ToolCalls()
	if len(calls) != 1 || calls[0].CallID != "call_1" || calls[0].Name != "read" {
		t.Errorf("assistant tool call mismatch: %+v", calls)
	}
	// tool message
	toolMsg := msgs[2]
	if toolMsg.Role != llm.RoleTool {
		t.Errorf("expected tool role, got %s", toolMsg.Role)
	}
	results := toolMsg.ToolResults()
	if len(results) != 1 || results[0].CallID != "call_1" {
		t.Errorf("tool result mismatch: %+v", results)
	}
}

func TestSurfaceCompactionReplacesOldRegion(t *testing.T) {
	events := make([]Event, 0, 6)
	events = append(events, Event{Seq: 1, Type: EventUserMessage, Timestamp: now(), Data: UserText("old A")})
	events = append(events, Event{Seq: 2, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("old reply", "", nil)})
	events = append(events, Event{Seq: 3, Type: EventUserMessage, Timestamp: now(), Data: UserText("recent B")})
	events = append(events, Event{Seq: 4, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("recent reply", "", nil)})

	// Compact everything through seq 2 into a summary.
	spec := SurfaceSpec{Compacted: &CompactedRegion{ThroughSeq: 2, Summary: "SUMMARY checkpoint"}}
	s := NewSurface(spec)
	msgs, err := s.DeriveMessages(events)
	if err != nil {
		t.Fatalf("DeriveMessages: %v", err)
	}
	// summary + recent B + recent reply
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (summary + recent tail), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != llm.RoleUser || msgs[0].Text() != "SUMMARY checkpoint" {
		t.Errorf("summary message wrong: %+v", msgs[0])
	}
	if msgs[1].Text() != "recent B" {
		t.Errorf("recent tail user wrong: %+v", msgs[1])
	}
}

func TestSessionAppendAndDerive(t *testing.T) {
	store := NewMemoryStore()
	session := NewSession("s1", store)
	ctx := context.Background()

	if _, err := session.Append(ctx, Event{Type: EventUserMessage, Timestamp: now(), Data: UserText("hi")}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := session.Append(ctx, Event{Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("hello", "", nil)}); err != nil {
		t.Fatalf("append: %v", err)
	}
	seq, err := session.Sequence(ctx)
	if err != nil || seq != 2 {
		t.Fatalf("sequence = %d err %v", seq, err)
	}
	msgs, err := session.DeriveMessages(ctx)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
}

func TestEventSurfaceFlags(t *testing.T) {
	cases := []struct {
		typ     EventType
		surface bool
	}{
		{EventTurnStart, false},
		{EventStepEnd, false},
		{EventAssistantChunk, false},
		{EventToolCall, false},
		{EventUsage, false},
		{EventUserMessage, true},
		{EventAssistantMessage, true},
		{EventToolResult, true},
	}
	for _, c := range cases {
		if got := c.typ.Surface(); got != c.surface {
			t.Errorf("%s surface = %v want %v", c.typ, got, c.surface)
		}
	}
}

func now() time.Time { return time.Unix(0, 0) }
