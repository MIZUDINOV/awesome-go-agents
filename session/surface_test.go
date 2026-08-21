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

// TestSessionReplayIdenticalSurfaceAfterRestart mirrors scenario H's core
// property (DB/worker restart): a FRESH Session over the same durable store
// rebuilds a byte-identical model surface without any network or in-flight
// state. This is the runnable (non-DB) demonstration of the gate; the
// Durable host-store replay is covered by the host adapter's integration tests.
func TestSessionReplayIdenticalSurfaceAfterRestart(t *testing.T) {
	store := NewMemoryStore()
	s1 := NewSession("s-restart", store)
	call := ToolCall{CallID: "call_1", Name: "read", Arguments: json.RawMessage(`{"file_path":"a.ts"}`)}

	if _, err := s1.AppendAll(context.Background(), []Event{
		{Type: EventUserMessage, Data: UserText("прочитай a.ts")},
		{Type: EventAssistantMessage, Data: AssistantContent("", "", []ToolCall{call})},
		{Type: EventToolCall, CallID: "call_1", Data: ToolCallPayload("call_1", "read", json.RawMessage(`{"file_path":"a.ts"}`))},
		{Type: EventToolResult, CallID: "call_1", Data: ToolResultPayload("call_1", "read", json.RawMessage(`{"content":"line 1"}`), false)},
	}); err != nil {
		t.Fatalf("append: %v", err)
	}
	before, _, err := s1.Project(context.Background())
	if err != nil {
		t.Fatalf("worker-1 project: %v", err)
	}

	// "Restart": an entirely new Session object re-reads the durable log.
	s2 := NewSession("s-restart", store)
	after, _, err := s2.Project(context.Background())
	if err != nil {
		t.Fatalf("worker-2 project after restart: %v", err)
	}
	if len(before) != len(after) {
		t.Fatalf("surface length changed after restart: %d != %d", len(before), len(after))
	}
	for i := range before {
		if before[i].Role != after[i].Role || before[i].Text() != after[i].Text() {
			t.Errorf("message %d differs after restart:\n  before=%+v\n  after =%+v", i, before[i], after[i])
		}
		bc, ac := before[i].ToolCalls(), after[i].ToolCalls()
		if len(bc) != len(ac) {
			t.Errorf("message %d tool-call count differs: %d != %d", i, len(bc), len(ac))
			continue
		}
		for j := range bc {
			if bc[j].CallID != ac[j].CallID || bc[j].Name != ac[j].Name {
				t.Errorf("call %d differs after restart: %+v != %+v", j, bc[j], ac[j])
			}
		}
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
		{EventCompactionStart, false},
		{EventCompactionSummary, false},
		{EventCompactionSurface, false},
		{EventCompactionEnd, false},
		{EventRequestHeader, false},
		{EventUserMessage, true},
		{EventAssistantMessage, true},
		{EventToolResult, true},
	}
	for _, c := range cases {
		if got := c.typ.Surface(); got != c.surface {
			t.Errorf("%s surface = %v want %v", c.typ, got, c.surface)
		}
		if !c.typ.Known() {
			t.Errorf("%s should be a known type", c.typ)
		}
	}
}

func TestEventValidation(t *testing.T) {
	good := Event{Type: EventUserMessage, Data: json.RawMessage(`{"text":"hi"}`), CallID: ""}
	if err := good.Validate(); err != nil {
		t.Errorf("valid event rejected: %v", err)
	}
	// Unknown mandatory type fails validation.
	bad := Event{Type: "mystery/event", Data: json.RawMessage(`{}`)}
	if err := bad.Validate(); err == nil {
		t.Error("unknown type should fail validation")
	}
	// Unsupported future format version fails.
	future := Event{Type: EventUserMessage, FormatVersion: EventFormatVersion + 1, Data: json.RawMessage(`{"text":"hi"}`)}
	if err := future.Validate(); err == nil {
		t.Error("future format version should fail validation")
	}
	// Tool events require call correlation.
	tool := Event{Type: EventToolCall, Data: json.RawMessage(`{}`)}
	if err := tool.Validate(); err == nil {
		t.Error("tool/call without call_id should fail validation")
	}
}

func TestSurfaceRebuildsFromCompactionEvents(t *testing.T) {
	// A durable compaction is recorded as start/summary/end events; the
	// surface projection must rebuild the replacement purely from the log.
	payload := CompactionSummaryPayload(1, "tx-1", 2, []uint64{1, 2}, "SUMMARY durable", "fp-1")
	events := []Event{
		{Seq: 1, Type: EventUserMessage, Timestamp: now(), Data: UserText("old A")},
		{Seq: 2, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("old reply", "", nil)},
		{Seq: 3, Type: EventCompactionStart, Timestamp: now(), Data: CompactionStartPayload(1, "tx-1", []uint64{1, 2})},
		{Seq: 4, Type: EventCompactionSummary, Timestamp: now(), Data: payload, SourceSeqs: []uint64{1, 2}},
		{Seq: 5, Type: EventCompactionEnd, Timestamp: now(), Data: CompactionEndPayload(1, "tx-1")},
		{Seq: 6, Type: EventUserMessage, Timestamp: now(), Data: UserText("recent B")},
		{Seq: 7, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("recent reply", "", nil)},
	}
	s := NewSurface(SurfaceSpec{})
	msgs, proj, err := s.Project(events)
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	// summary + recent tail
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != llm.RoleUser || msgs[0].Text() != "SUMMARY durable" {
		t.Errorf("summary message wrong: %+v", msgs[0])
	}
	if msgs[1].Text() != "recent B" || msgs[2].Text() != "recent reply" {
		t.Errorf("recent tail wrong: %+v", msgs)
	}
	if proj.Generation != 1 {
		t.Errorf("generation = %d", proj.Generation)
	}
	if len(proj.ShadowedSeqs) != 2 || proj.ShadowedSeqs[0] != 1 || proj.ShadowedSeqs[1] != 2 {
		t.Errorf("shadowed = %v", proj.ShadowedSeqs)
	}
	// Replay identity: projecting the same log twice is identical.
	msgs2, _, err := s.Project(events)
	if err != nil {
		t.Fatalf("replay Project: %v", err)
	}
	if len(msgs2) != len(msgs) {
		t.Fatalf("replay message count differs: %d vs %d", len(msgs2), len(msgs))
	}
	for i := range msgs {
		if msgs[i].Role != msgs2[i].Role || msgs[i].Text() != msgs2[i].Text() {
			t.Errorf("replay diverged at %d: %+v vs %+v", i, msgs[i], msgs2[i])
		}
	}
}

func TestSurfaceRejectsBrokenPairingAcrossCompaction(t *testing.T) {
	// The assistant message with the tool call is shadowed (seq 2) but its
	// tool/result (seq 3) would remain visible: pairing would break.
	events := []Event{
		{Seq: 1, Type: EventUserMessage, Timestamp: now(), Data: UserText("go")},
		{Seq: 2, Type: EventAssistantMessage, Timestamp: now(), Data: AssistantContent("", "", []ToolCall{{CallID: "c1", Name: "read", Arguments: json.RawMessage(`{}`)}})},
		{Seq: 3, Type: EventToolResult, Timestamp: now(), Data: ToolResultPayload("c1", "read", json.RawMessage(`{"ok":true}`), false), SourceSeqs: []uint64{2}},
		{Seq: 4, Type: EventCompactionSummary, Timestamp: now(), Data: CompactionSummaryPayload(1, "tx-1", 2, []uint64{1, 2}, "summary", "fp")},
	}
	s := NewSurface(SurfaceSpec{})
	if _, _, err := s.Project(events); err == nil {
		t.Error("expected pairing violation to be rejected")
	}
}

func TestSurfaceV2CompactionUsesExactSourceSeqs(t *testing.T) {
	format := EventFormatVersion
	events := []Event{
		{Seq: 1, FormatVersion: format, Type: EventUserMessage, Data: UserText("old A")},
		{Seq: 2, FormatVersion: format, Type: EventUserMessage, Data: UserText("keep")},
		{Seq: 3, FormatVersion: format, Type: EventAssistantMessage, Data: AssistantContent("old B", "", nil)},
		{Seq: 4, FormatVersion: format, Type: EventCompactionStart, Data: CompactionStartPayload(1, "tx-exact", []uint64{1, 3})},
		{Seq: 5, FormatVersion: format, Type: EventCompactionSummary, Data: CompactionSummaryPayload(1, "tx-exact", 3, []uint64{1, 3}, "summary", "fp")},
		{Seq: 6, FormatVersion: format, Type: EventCompactionSurface, Data: CompactionSurfacePayload(1, "tx-exact", []uint64{1, 3}, "summary", "fp")},
		{Seq: 7, FormatVersion: format, Type: EventCompactionEnd, Data: CompactionEndPayload(1, "tx-exact")},
	}
	msgs, projection, err := NewSurface(SurfaceSpec{}).Project(events)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 || msgs[0].Text() != "summary" || msgs[1].Text() != "keep" {
		t.Fatalf("exact compaction surface = %+v", msgs)
	}
	if got := projection.ShadowedSeqs; len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Fatalf("shadowed seqs = %v, want [1 3]", got)
	}
}

func TestSurfaceV2InterruptedCompactionDoesNotChangeSurface(t *testing.T) {
	format := EventFormatVersion
	events := []Event{
		{Seq: 1, FormatVersion: format, Type: EventUserMessage, Data: UserText("old")},
		{Seq: 2, FormatVersion: format, Type: EventCompactionStart, Data: CompactionStartPayload(1, "tx-open", []uint64{1})},
		{Seq: 3, FormatVersion: format, Type: EventCompactionSummary, Data: CompactionSummaryPayload(1, "tx-open", 1, []uint64{1}, "must not show", "fp")},
	}
	msgs, projection, err := NewSurface(SurfaceSpec{}).Project(events)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].Text() != "old" || projection.Summary != "" {
		t.Fatalf("interrupted compaction changed surface: messages=%+v projection=%+v", msgs, projection)
	}
}

func TestSessionValidatesBeforeAppend(t *testing.T) {
	store := NewMemoryStore()
	session := NewSession("s1", store)
	ctx := context.Background()
	if _, err := session.Append(ctx, Event{Type: "unknown/type", Data: UserText("x")}); err == nil {
		t.Error("expected unknown type to be rejected before append")
	}
	seq, err := session.Sequence(ctx)
	if err != nil || seq != 0 {
		t.Errorf("nothing should be stored, seq = %d err %v", seq, err)
	}
	if _, err := session.Append(ctx, Event{Type: EventUserMessage, Data: UserText("hi")}); err != nil {
		t.Fatalf("valid append failed: %v", err)
	}
}

func TestSessionDurableCompaction(t *testing.T) {
	store := NewMemoryStore()
	session := NewSession("s1", store)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		if _, err := session.Append(ctx, Event{Type: EventUserMessage, Data: UserText("msg")}); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	if _, err := session.CompactionSummary(ctx, 1, "tx-1", 2, []uint64{1, 2}, "SUMMARY", "fp-1"); err != nil {
		t.Fatalf("compaction: %v", err)
	}
	// Re-hydrate from a fresh session over the same store: identical surface.
	replay := NewSession("s1", store)
	msgs, proj, err := replay.Project(ctx)
	if err != nil {
		t.Fatalf("replay project: %v", err)
	}
	if len(msgs) != 2 { // summary + remaining latest user message
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Text() != "SUMMARY" {
		t.Errorf("summary = %q", msgs[0].Text())
	}
	if msgs[1].Text() != "msg" {
		t.Errorf("tail = %q", msgs[1].Text())
	}
	if proj.Generation != 1 {
		t.Errorf("generation = %d", proj.Generation)
	}
}

func now() time.Time { return time.Unix(0, 0) }
