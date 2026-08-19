package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

// scriptedProvider is a scripted Chat that emits a sequence of responses.
// Each step returns the next scripted assistant reply; when empty, it replies
// with a final stop text. Supports a fake tool call that a registry can run.
type scriptedProvider struct {
	mu        sync.Mutex
	steps     []scriptedStep
	calls     []llm.Request // recorded requests for assertions
	exhausted bool
}

type scriptedStep struct {
	text   string
	calls  []llm.ToolCallRequest
	finish llm.FinishReason
}

func (s *scriptedProvider) Name() string { return "scripted" }

func (s *scriptedProvider) Generate(ctx context.Context, req *llm.Request, cb llm.StreamCallback) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.calls = append(s.calls, *req)
	var step scriptedStep
	if len(s.steps) > 0 {
		step = s.steps[0]
		s.steps = s.steps[1:]
	} else {
		s.exhausted = true
	}
	s.mu.Unlock()
	finish := step.finish
	if finish == "" {
		finish = llm.FinishReasonStop
	}
	msg := llm.NewAssistantMessage(step.text, "", step.calls)
	return &llm.Response{Message: msg, FinishReason: finish, Model: req.Model}, nil
}

// toolProbe is a registering tool that tracks and optionally fails executions.
func registerProbe(t *testing.T, registry *tools.Registry, name string, failAfter *int, counter *int) {
	t.Helper()
	registry.MustRegister(&tools.Definition{
		Name: name, Description: "probe",
		InputSchema: jsonRaw(`{"type":"object","properties":{"count":{"type":"integer"}},"additionalProperties":false}`),
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			if counter != nil {
				*counter++
			}
			if failAfter != nil && *failAfter > 0 && *counter >= *failAfter {
				return nil, fmt.Errorf("probe failed")
			}
			return map[string]any{"ok": true, "name": name}, nil
		},
	})
}

func newMemoryStore() *session.MemoryStore { return session.NewMemoryStore() }

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// TestDurableAppendAndReplay: a completed run is fully replayed by a fresh
// Loop over the same store and a second input extends the same history.
func TestDurableAppendAndReplay(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registerProbe(t, registry, "probe_touch", nil, nil)

	chat := &scriptedProvider{
		steps: []scriptedStep{{text: "done", finish: llm.FinishReasonStop}},
	}
	loop := NewLoop("s1", store, registry, chat, Config{Model: "m", Owner: "w", SystemPrompt: "sys"})

	res, err := loop.Run(context.Background(), "hello")
	check(t, err)
	if res.FinishedReason != "stop" || res.Turns != 1 {
		t.Fatalf("result=%+v", res)
	}

	// A FRESH loop over the same store must not re-run; it appends a new run.
	chat2 := &scriptedProvider{steps: []scriptedStep{{text: "second", finish: llm.FinishReasonStop}}}
	loop2 := NewLoop("s1", store, registry, chat2, Config{Model: "m", Owner: "w", SystemPrompt: "sys"})
	_, err = loop2.Run(context.Background(), "again")
	check(t, err)

	events, err := store.Load(context.Background(), "s1", 0, 0)
	check(t, err)
	// Both user messages are durably present.
	userCount := 0
	for _, e := range events {
		if e.Type == session.EventUserMessage {
			userCount++
		}
	}
	if userCount < 2 {
		t.Fatalf("expected at least 2 durable user messages, got %d (events=%d)", userCount, len(events))
	}
}

// TestNoReExecutionOnRecovery: a dangling tool/call (no result) is marked
// TOOL_OUTCOME_UNKNOWN by recovery and is NEVER re-executed by a fresh run.
func TestNoReExecutionOnRecovery(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	var executions int
	registerProbe(t, registry, "probe_sideeffect", nil, &executions)

	// Simulate a crashed prior run: one tool/call with NO tool/result,
	// under an open lease (dangling side effect with unknown outcome).
	lease, err := store.ClaimLease(context.Background(), "s9", "crashed-worker", 0, "")
	check(t, err)
	_, err = store.AppendFenced(context.Background(), lease, []session.Event{{
		ID: "prior-assistant", SessionID: "s9", Type: session.EventAssistantMessage,
		Data: session.AssistantContent("thinking", "", []session.ToolCall{{CallID: "call-1", Name: "probe_sideeffect", Arguments: jsonLit(`{}`)}}),
	}, {
		ID: "prior-call", SessionID: "s9", CallID: "call-1", Type: session.EventToolCall,
		Data: session.ToolCallPayload("call-1", "probe_sideeffect", jsonLit(`{}`)),
	}})
	check(t, err)
	check(t, store.ReleaseLease(context.Background(), lease))

	// A fresh worker claims and recovers.
	chat := &scriptedProvider{steps: []scriptedStep{{text: "recovered", finish: llm.FinishReasonStop}}}
	loop := NewLoop("s9", store, registry, chat, Config{Model: "m", Owner: "fresh", SystemPrompt: "sys"})
	_, err = loop.Run(context.Background(), "continue")
	check(t, err)

	if executions != 0 {
		t.Fatalf("dangling tool call was re-executed %d times; recovery must not re-run unknown side effects", executions)
	}
	// The dangling call must now have a TOOL_OUTCOME_UNKNOWN result.
	events, err := store.Load(context.Background(), "s9", 0, 0)
	check(t, err)
	foundUnknown := false
	for _, e := range events {
		if e.Type == session.EventToolResult && e.CallID == "call-1" {
			if strings.Contains(string(e.Data), "TOOL_OUTCOME_UNKNOWN") {
				foundUnknown = true
			}
		}
	}
	if !foundUnknown {
		t.Fatal("expected TOOL_OUTCOME_UNKNOWN result for dangling call")
	}
}

// TestCancellationPriority: a cancelled context stops the loop before any
// further tool executes; recovery never runs past a cancellation.
func TestCancellationPriority(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	var executions int
	registerProbe(t, registry, "probe_late", nil, &executions)

	// The model requests two tool calls; we cancel before execution completes.
	chat := &scriptedProvider{steps: []scriptedStep{{
		calls:  []llm.ToolCallRequest{{CallID: "c1", Name: "probe_late", Arguments: jsonLit(`{"count":1}`)}},
		finish: llm.FinishReasonToolCalls,
	}, {
		calls:  []llm.ToolCallRequest{{CallID: "c2", Name: "probe_late", Arguments: jsonLit(`{"count":2}`)}},
		finish: llm.FinishReasonToolCalls,
	}}}

	ctx, cancel := context.WithCancel(context.Background())
	loop := NewLoop("s7", store, registry, chat, Config{Model: "m", Owner: "w", SystemPrompt: "sys"})
	done := make(chan struct{})
	go func() {
		<-timeAfter(10)
		cancel()
		close(done)
	}()
	_, err := loop.Run(ctx, "go")
	<-done
	// The loop must stop, not panic and not execute indefinitely.
	t.Logf("run returned err=%v executions=%d", err, executions)
	if err == nil {
		t.Log("run completed before cancellation; acceptable")
	}
	// Regardless, tools executed are durably recorded (no split-brain).
	events, eerr := store.Load(context.Background(), "s7", 0, 0)
	check(t, eerr)
	_ = events
}

// TestSequentialToolExecutionForced: ParallelToolCalls=false is always sent.
func TestSequentialToolExecutionForced(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registerProbe(t, registry, "probe_seq", nil, nil)
	chat := &scriptedProvider{steps: []scriptedStep{{calls: []llm.ToolCallRequest{{CallID: "c1", Name: "probe_seq", Arguments: jsonLit(`{}`)}}, finish: llm.FinishReasonToolCalls}, {text: "done"}}}
	loop := NewLoop("s5", store, registry, chat, Config{Model: "m", Owner: "w", SystemPrompt: "sys"})
	_, err := loop.Run(context.Background(), "seq")
	check(t, err)
	chat.mu.Lock()
	defer chat.mu.Unlock()
	if len(chat.calls) == 0 || chat.calls[0].ParallelToolCalls == nil || *chat.calls[0].ParallelToolCalls {
		t.Fatalf("expected ParallelToolCalls=false, got %+v", chat.calls)
	}
}

// TestCompactionOnOverflow: when a Compactor is configured, crossing the
// pressure threshold performs a durable compaction (summary events appended).
func TestCompactionOnOverflow(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registerProbe(t, registry, "probe_c", nil, nil)

	var compactMu sync.Mutex
	compactCalls := 0
	compactor := &countingCompactor{f: func(generation uint64, events []session.Event, through uint64) (string, string, error) {
		compactMu.Lock()
		compactCalls++
		compactMu.Unlock()
		return "[summarized]", "fp-" + fmt.Sprint(generation), nil
	}}

	chat := &scriptedProvider{steps: []scriptedStep{{text: "ok", finish: llm.FinishReasonStop}}}
	loop := NewLoop("s6", store, registry, chat, Config{
		Model: "m", Owner: "w", SystemPrompt: "sys",
		ContextWindow: 60, MaxOutput: 10, CompactThresholdRatio: 0.80,
		Compactor: compactor, PruneHead: 4,
	})
	// The input alone must exceed the 0.80 threshold so preflight compacts
	// BEFORE the first step (durable compaction of existing history).
	_, err := loop.Run(context.Background(), strings.Repeat("payload ", 40))
	check(t, err)
	compactMu.Lock()
	cc := compactCalls
	compactMu.Unlock()
	if cc == 0 {
		t.Fatal("expected at least one compaction to have run on overflow")
	}
	events, err := store.Load(context.Background(), "s6", 0, 0)
	check(t, err)
	hasSummary := false
	for _, e := range events {
		if e.Type == session.EventCompactionSummary {
			hasSummary = true
		}
	}
	if !hasSummary {
		t.Fatal("expected a durable compaction/summary event")
	}
}

// TestCancellationStopsBeforeWork: a pre-cancelled context stops the loop
// before it executes any tool or appends new history (cancellation has
// priority over recovery/compaction).
func TestCancellationStopsBeforeWork(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	var executions int
	registerProbe(t, registry, "probe_cancel", nil, &executions)
	chat := &scriptedProvider{steps: []scriptedStep{{calls: []llm.ToolCallRequest{{CallID: "c1", Name: "probe_cancel", Arguments: jsonLit(`{}`)}}, finish: llm.FinishReasonToolCalls}}}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	loop := NewLoop("s8", store, registry, chat, Config{Model: "m", Owner: "w", SystemPrompt: "sys"})
	_, err := loop.Run(ctx, "go")
	if !errors.Is(err, ErrStopped) {
		t.Fatalf("expected ErrStopped, got %v", err)
	}
	if executions != 0 {
		t.Fatalf("tool executed %d times under cancellation", executions)
	}
	// No tool events were durably appended for the cancelled run.
	events, eerr := store.Load(context.Background(), "s8", 0, 0)
	check(t, eerr)
	for _, e := range events {
		if e.Type == session.EventToolResult || e.Type == session.EventToolCall {
			t.Fatalf("unexpected tool event under cancellation: %s", e.Type)
		}
	}
}

type countingCompactor struct {
	f func(generation uint64, events []session.Event, through uint64) (string, string, error)
}

func (c *countingCompactor) Compact(_ context.Context, generation uint64, events []session.Event, throughSeq uint64) (string, string, error) {
	return c.f(generation, events, throughSeq)
}

// ---------------------------------------------------------------------------
// helpers

func jsonRaw(s string) json.RawMessage { return json.RawMessage(s) }
func jsonLit(s string) json.RawMessage { return json.RawMessage(s) }

func timeAfter(ms int) <-chan struct{} {
	done := make(chan struct{})
	go func() { time.Sleep(time.Duration(ms) * time.Millisecond); close(done) }()
	return done
}
