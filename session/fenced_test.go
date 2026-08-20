package session

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestFencedLeaseAndWrite(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	lease, err := store.ClaimLease(ctx, "s1", "worker-a", 30*time.Second, "tenant-1")
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if lease.Token == "" || lease.Fence == 0 {
		t.Fatalf("bad lease: %+v", lease)
	}
	if _, err := store.AppendFenced(ctx, lease, []Event{{Type: EventUserMessage, Data: UserText("hi")}}); err != nil {
		t.Fatalf("fenced append: %v", err)
	}

	// A second worker cannot claim while the lease is held.
	if _, err := store.ClaimLease(ctx, "s1", "worker-b", 30*time.Second, "tenant-1"); !errors.Is(err, ErrLeaseHeld) {
		t.Errorf("expected ErrLeaseHeld, got %v", err)
	}

	// A stale lease (wrong fence) cannot write.
	stale := lease
	stale.Fence = lease.Fence + 100
	if _, err := store.AppendFenced(ctx, stale, []Event{{Type: EventUserMessage, Data: UserText("late")}}); !errors.Is(err, ErrLeaseLost) {
		t.Errorf("expected ErrLeaseLost for stale fence, got %v", err)
	}

	// Release then re-claim: new fence invalidates the old lease's writes.
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatalf("release: %v", err)
	}
	lease2, err := store.ClaimLease(ctx, "s1", "worker-b", 30*time.Second, "tenant-1")
	if err != nil {
		t.Fatalf("re-claim: %v", err)
	}
	if lease2.Fence <= lease.Fence {
		t.Errorf("fence did not increase: %d -> %d", lease.Fence, lease2.Fence)
	}
	if _, err := store.AppendFenced(ctx, lease, []Event{{Type: EventUserMessage, Data: UserText("old worker late")}}); !errors.Is(err, ErrLeaseLost) {
		t.Errorf("old lease must not write after re-claim, got %v", err)
	}
	seq, err := store.AppendFenced(ctx, lease2, []Event{{Type: EventUserMessage, Data: UserText("new worker ok")}})
	if err != nil {
		t.Fatalf("new lease append: %v", err)
	}
	if seq != 2 {
		t.Errorf("seq = %d, want 2 (old worker's late write must be rejected)", seq)
	}
}

func TestFencedIdempotentAppend(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "s1", "w", time.Minute, "tenant-1")
	if err != nil {
		t.Fatal(err)
	}
	batch := []Event{
		{Type: EventToolCall, ID: "run1:call1", CallID: "call1", Data: ToolCallPayload("call1", "read", json.RawMessage(`{}`))},
		{Type: EventToolResult, ID: "run1:call1:res", CallID: "call1", Data: ToolResultPayload("call1", "read", json.RawMessage(`{"ok":true}`), false), SourceSeqs: []uint64{1}},
	}
	first, err := store.AppendFenced(ctx, lease, batch)
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	// Retry the exact same batch: no duplicates, same canonical seqs.
	retry, err := store.AppendFenced(ctx, lease, batch)
	if err != nil {
		t.Fatalf("retry: %v", err)
	}
	if retry != first {
		t.Errorf("retry last seq = %d, want %d", retry, first)
	}
	seq, _ := store.Sequence(ctx, "s1")
	if seq != first {
		t.Errorf("sequence = %d, want %d (no gap, no dupe)", seq, first)
	}
	load, err := store.Load(ctx, "s1", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(load) != 2 {
		t.Errorf("stored events = %d, want 2", len(load))
	}
}

func TestRecoverClosesTurnAndMarksUnknown(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	lease, _ := store.ClaimLease(ctx, "s1", "w", time.Minute, "tenant-1")
	// Open turn + a durable tool/call whose result was never persisted.
	if _, err := store.AppendFenced(ctx, lease, []Event{
		{RunID: "run-1", TurnID: "turn-1", Type: EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "turn-1-step-00", Type: EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "turn-1-step-00", CallID: "call1", Type: EventToolCall, ID: "t:call1", Data: ToolCallPayload("call1", "write", json.RawMessage(`{"path":"a.ts"}`))},
		{RunID: "run-1", TurnID: "turn-1", StepID: "turn-1-step-00", CallID: "call1", Type: EventToolDispatched, Data: json.RawMessage(`{"text":"dispatched"}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "turn-1-step-00", CallID: "call1", Type: EventToolRunning, Data: json.RawMessage(`{"text":"running"}`)},
	}); err != nil {
		t.Fatal(err)
	}
	// Simulate crash: new worker takes over after the lease expires.
	store.ReleaseLease(ctx, lease)
	lease2, _ := store.ClaimLease(ctx, "s1", "w2", time.Minute, "tenant-1")
	report, err := store.Recover(ctx, lease2)
	if err != nil {
		t.Fatalf("recover: %v", err)
	}
	if !report.TurnClosed {
		t.Error("open turn not closed")
	}
	if len(report.DanglingCalls) != 1 || report.DanglingCalls[0] != "call1" {
		t.Errorf("dangling calls = %v", report.DanglingCalls)
	}
	all, _ := store.Load(ctx, "s1", 0, 0)
	foundUnknown := false
	for _, e := range all {
		if e.Type == EventToolResult && e.CallID == "call1" {
			var p struct {
				Code string `json:"code"`
			}
			_ = json.Unmarshal(e.Data, &p)
			if p.Code == "TOOL_OUTCOME_UNKNOWN" {
				foundUnknown = true
			}
		}
	}
	if !foundUnknown {
		t.Error("missing TOOL_OUTCOME_UNKNOWN result for dangling call")
	}
	resultIndex, stepEndIndex, turnEndIndex := -1, -1, -1
	for index, event := range all {
		if event.Type == EventToolResult && event.CallID == "call1" {
			resultIndex = index
		}
		if event.Type == EventStepEnd && event.StepID == "turn-1-step-00" {
			stepEndIndex = index
		}
		if event.Type == EventTurnEnd && event.TurnID == "turn-1" {
			turnEndIndex = index
		}
	}
	if resultIndex < 0 || stepEndIndex < 0 || turnEndIndex < 0 || !(resultIndex < stepEndIndex && stepEndIndex < turnEndIndex) {
		t.Fatalf("recovery lifecycle order result=%d step_end=%d turn_end=%d", resultIndex, stepEndIndex, turnEndIndex)
	}
}

func TestRecoverClosesOpenStep(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	lease, _ := store.ClaimLease(ctx, "s-step", "w", time.Minute, "tenant-1")
	if _, err := store.AppendFenced(ctx, lease, []Event{
		{RunID: "run-1", TurnID: "turn-1", StepID: "turn-1-step-00", Type: EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "turn-1-step-00", Type: EventStepStart, Data: json.RawMessage(`{}`)},
	}); err != nil {
		t.Fatal(err)
	}
	store.ReleaseLease(ctx, lease)
	lease, _ = store.ClaimLease(ctx, "s-step", "w2", time.Minute, "tenant-1")
	report, err := store.Recover(ctx, lease)
	if err != nil {
		t.Fatalf("recover: %v", err)
	}
	if report.StepsClosed != 1 {
		t.Fatalf("steps closed = %d, want 1", report.StepsClosed)
	}
	all, _ := store.Load(ctx, "s-step", 0, 0)
	found := false
	for _, event := range all {
		if event.Type == EventStepEnd && event.StepID == "turn-1-step-00" {
			found = true
		}
	}
	if !found {
		t.Fatal("missing recovered step/end")
	}
}

func TestMemoryStoreConcurrentAppends(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			sid := "s"
			_, _ = store.Append(ctx, sid, []Event{{Type: EventUserMessage, Data: UserText("x")}})
		}(i)
	}
	wg.Wait()
	seq, err := store.Sequence(ctx, "s")
	if err != nil || seq != 20 {
		t.Errorf("seq = %d, want 20 (%v)", seq, err)
	}
	events, _ := store.Load(ctx, "s", 0, 0)
	if len(events) != 20 {
		t.Errorf("events = %d, want 20", len(events))
	}
	// Strict monotonic, no dupes.
	seen := map[uint64]bool{}
	for _, e := range events {
		if seen[e.Seq] {
			t.Errorf("duplicate seq %d", e.Seq)
		}
		seen[e.Seq] = true
	}
}
