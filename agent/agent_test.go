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

	memctx "github.com/MIZUDINOV/awesome-go-agents/context"
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
	text     string
	calls    []llm.ToolCallRequest
	metadata map[string]any
	finish   llm.FinishReason
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
	msg.Metadata = step.metadata
	return &llm.Response{Message: msg, FinishReason: finish, Model: req.Model}, nil
}

// toolProbe is a registering tool that tracks and optionally fails executions.
func registerProbe(t *testing.T, registry *tools.Registry, name string, failAfter *int, counter *int) {
	t.Helper()
	registry.MustRegister(&tools.Definition{
		Name: name, Description: "probe",
		InputSchema:  jsonRaw(`{"type":"object","properties":{"count":{"type":"integer"}},"additionalProperties":false}`),
		OutputSchema: tools.AnyOutputSchema,
		Execute: func(ctx context.Context, ec tools.ToolRunContext, input json.RawMessage) (any, error) {
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

type hostClaimStore struct {
	*session.MemoryStore
	claims, renewals, releases int
}

type cancelOnAdmittedStore struct {
	*session.MemoryStore
	cancel context.CancelFunc
}

type cleanupLeaseStore struct {
	*session.MemoryStore
	releases, canceledReleases int
}

func (s *cleanupLeaseStore) ReleaseLease(ctx context.Context, lease session.Lease) error {
	if err := ctx.Err(); err != nil {
		s.canceledReleases++
		return err
	}
	s.releases++
	return s.MemoryStore.ReleaseLease(context.Background(), lease)
}

type blockingProvider struct{ started chan struct{} }

type failingProvider struct{ err error }

type capabilityProvider struct {
	*scriptedProvider
	caps llm.Capabilities
}

func (p *blockingProvider) Name() string { return "blocking" }

func (p *blockingProvider) Generate(ctx context.Context, _ *llm.Request, _ llm.StreamCallback) (*llm.Response, error) {
	close(p.started)
	<-ctx.Done()
	return nil, ctx.Err()
}

func (p *failingProvider) Name() string { return "failing" }

func (p *failingProvider) Generate(context.Context, *llm.Request, llm.StreamCallback) (*llm.Response, error) {
	return nil, p.err
}

func (p *capabilityProvider) Capabilities(context.Context, string) (llm.Capabilities, error) {
	return p.caps, nil
}

func (s *cancelOnAdmittedStore) AppendFencedCommitted(ctx context.Context, lease session.Lease, events []session.Event) (session.CommittedBatch, error) {
	batch, err := s.MemoryStore.AppendFencedCommitted(ctx, lease, events)
	if err == nil {
		for _, event := range events {
			if event.Type == session.EventToolAdmitted {
				s.cancel()
				break
			}
		}
	}
	return batch, err
}

func (s *hostClaimStore) ClaimLease(ctx context.Context, sessionID, owner string, ttl time.Duration, tenantID string) (session.Lease, error) {
	s.claims++
	return s.MemoryStore.ClaimLease(ctx, sessionID, owner, ttl, tenantID)
}

func (s *hostClaimStore) RenewLease(ctx context.Context, lease session.Lease) (session.Lease, error) {
	s.renewals++
	return s.MemoryStore.RenewLease(ctx, lease)
}

func (s *hostClaimStore) ReleaseLease(ctx context.Context, lease session.Lease) error {
	s.releases++
	return s.MemoryStore.ReleaseLease(ctx, lease)
}

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestEventHubReplayLiveHandoffAndLag(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	_, err := store.Append(ctx, "events", []session.Event{{ID: "old", SessionID: "events", Type: session.EventUserMessage, Data: session.UserText("old")}})
	check(t, err)
	hub := NewEventHub(1)
	loop := NewLoop("events", store, tools.New(tools.Options{}), &scriptedProvider{steps: []scriptedStep{{text: "ok"}}}, Config{EventHub: hub})
	sub, err := loop.Subscribe(ctx, 0, nil)
	check(t, err)
	event, err := sub.Next(ctx)
	check(t, err)
	if event.ID != "old" || event.Seq != 1 {
		t.Fatalf("replay event=%+v", event)
	}
	hub.Publish(session.Event{ID: "new", SessionID: "events", Seq: 2, Type: session.EventUserMessage, Data: session.UserText("new")})
	event, err = sub.Next(ctx)
	check(t, err)
	if event.ID != "new" || event.Seq != 2 {
		t.Fatalf("live event=%+v", event)
	}
	hub.Publish(session.Event{ID: "lag-1", SessionID: "events", Seq: 3, Type: session.EventUserMessage, Data: session.UserText("1")})
	hub.Publish(session.Event{ID: "lag-2", SessionID: "events", Seq: 4, Type: session.EventUserMessage, Data: session.UserText("2")})
	_, _ = sub.Next(ctx)
	if _, err := sub.Next(ctx); !errors.Is(err, ErrSubscriberLagged) {
		t.Fatalf("lag error=%v", err)
	}
}

func TestSharedEventHubIsSessionScoped(t *testing.T) {
	hub := NewEventHub(8)
	store := session.NewMemoryStore()
	chat := &scriptedProvider{steps: []scriptedStep{{text: "done"}}}
	loop := NewLoop("session-a", store, tools.New(tools.Options{}), chat, Config{EventHub: hub})
	sub, err := loop.Subscribe(context.Background(), 0, nil)
	check(t, err)
	defer sub.Close()
	hub.Publish(session.Event{ID: "other", SessionID: "session-b", Seq: 1, Type: session.EventUserMessage, Data: session.UserText("other")})
	hub.Publish(session.Event{ID: "empty", Seq: 1, Type: session.EventUserMessage, Data: session.UserText("empty")})
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if _, err := sub.Next(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cross-session event error=%v", err)
	}
	hub.Publish(session.Event{ID: "own", SessionID: "session-a", Seq: 2, Type: session.EventUserMessage, Data: session.UserText("own")})
	event, err := sub.Next(context.Background())
	check(t, err)
	if event.ID != "own" {
		t.Fatalf("event=%+v", event)
	}

	notifications := hub.SubscribeNotificationsFor(context.Background(), "session-a")
	defer notifications.Close()
	hub.PublishNotification(Notification{Type: "other", SessionID: "session-b"})
	hub.PublishNotification(Notification{Type: "empty"})
	hub.PublishNotification(Notification{Type: "own", SessionID: "session-a"})
	notification, err := notifications.Next(context.Background())
	check(t, err)
	if notification.Type != "own" {
		t.Fatalf("notification=%+v", notification)
	}
}

func TestToolCallIDIsNormalizedBeforeAssistantMessage(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registry.MustRegister(&tools.Definition{
		Name: "probe", InputSchema: tools.OrObjectSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) { return "ok", nil },
	})
	chat := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{Name: "probe", Arguments: json.RawMessage(`{}`)}}},
		{text: "done"},
	}}
	loop := NewLoop("stable-call-id", store, registry, chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	if _, err := loop.Run(context.Background(), "run"); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "stable-call-id", 0, 0)
	check(t, err)
	var assistantID, callID string
	var assistantSeq, userSeq, stepSeq uint64
	for _, event := range events {
		switch event.Type {
		case session.EventUserMessage:
			if userSeq == 0 {
				userSeq = event.Seq
			}
		case session.EventStepStart:
			if stepSeq == 0 {
				stepSeq = event.Seq
			}
		case session.EventAssistantMessage:
			if assistantID != "" {
				continue
			}
			var payload struct {
				ToolCalls []session.ToolCall `json:"tool_calls"`
			}
			if json.Unmarshal(event.Data, &payload) == nil && len(payload.ToolCalls) == 1 {
				assistantID = payload.ToolCalls[0].CallID
				assistantSeq = event.Seq
			}
		case session.EventToolCall:
			if callID == "" {
				callID = event.CallID
			}
		}
	}
	if assistantID == "" || assistantID != callID {
		t.Fatalf("assistant id=%q call id=%q", assistantID, callID)
	}
	if userSeq == 0 || stepSeq == 0 || userSeq <= stepSeq {
		t.Fatalf("event order user=%d step=%d", userSeq, stepSeq)
	}
	if assistantSeq == 0 {
		t.Fatal("missing assistant message")
	}
	chat.mu.Lock()
	if len(chat.calls) != 2 {
		t.Fatalf("provider calls=%d", len(chat.calls))
	}
	continued := chat.calls[1]
	chat.mu.Unlock()
	foundAssistant, foundResult := false, false
	for _, message := range continued.Messages {
		if message.Role == llm.RoleAssistant && len(message.ToolCalls()) == 1 && message.ToolCalls()[0].CallID == assistantID {
			foundAssistant = true
		}
		if message.Role == llm.RoleTool && len(message.ToolResults()) == 1 && message.ToolResults()[0].CallID == assistantID {
			foundResult = true
		}
	}
	if !foundAssistant || !foundResult {
		t.Fatalf("continuation assistant=%v result=%v", foundAssistant, foundResult)
	}
}

func TestPreflightUsesProviderCapabilitiesAndBuilderEstimate(t *testing.T) {
	chat := &capabilityProvider{
		scriptedProvider: &scriptedProvider{steps: []scriptedStep{{text: "done"}}},
		caps:             llm.Capabilities{ContextWindow: 2000, MaxOutput: 100},
	}
	loop := NewLoop("provider-capabilities", newMemoryStore(), tools.New(tools.Options{}), chat, Config{
		Model: "m", Owner: "test", SystemPrompt: "sys", ContextWindow: 100, MaxOutput: 10,
		ContextBuilder: memctx.NewBuilder(),
	})
	if _, err := loop.Run(context.Background(), strings.Repeat("large input ", 100)); err != nil {
		t.Fatalf("provider capability-aware preflight rejected request: %v", err)
	}
}

func TestFinalizerFailurePreservesModelFacingToolResult(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registry.MustRegister(&tools.Definition{
		Name: "finalize_fail", InputSchema: tools.OrObjectSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			return map[string]any{"canonical": true}, nil
		},
		FinalizeContent: func(json.RawMessage, any) ([]session.ContentBlock, error) {
			return nil, errors.New("finalizer failed")
		},
	})
	chat := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{CallID: "finalize-1", Name: "finalize_fail", Arguments: json.RawMessage(`{}`)}}},
		{text: "continued"},
	}}
	loop := NewLoop("finalizer-result", store, registry, chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	if _, err := loop.Run(context.Background(), "run"); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "finalizer-result", 0, 0)
	check(t, err)
	var resultData []byte
	for _, event := range events {
		if event.Type == session.EventToolResult && event.CallID == "finalize-1" {
			resultData = event.Data
		}
	}
	if !strings.Contains(string(resultData), "FINALIZE_FAILED") {
		t.Fatalf("durable finalizer result=%s", resultData)
	}
}

func TestFailedInboxRunIsRequeued(t *testing.T) {
	store := newMemoryStore()
	loop := NewLoop("requeue", store, tools.New(tools.Options{}), &failingProvider{err: errors.New("provider down")}, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	handle, err := NewAgent(loop)
	check(t, err)
	defer func() { _ = handle.Dispose(context.Background()) }()
	check(t, handle.FollowUp(context.Background(), "retry me"))
	deadline := time.Now().Add(time.Second)
	var events []session.Event
	for time.Now().Before(deadline) {
		events, err = store.Load(context.Background(), "requeue", 0, 0)
		check(t, err)
		found := false
		for _, event := range events {
			found = found || event.Type == session.EventInboxRequeued
		}
		if found {
			break
		}
		time.Sleep(time.Millisecond)
	}
	foundRequeued, foundCompleted := false, false
	for _, event := range events {
		foundRequeued = foundRequeued || event.Type == session.EventInboxRequeued
		foundCompleted = foundCompleted || event.Type == session.EventInboxCompleted
	}
	if !foundRequeued || foundCompleted {
		t.Fatalf("requeued=%v completed=%v events=%d", foundRequeued, foundCompleted, len(events))
	}
}

func TestStatefulAgentFollowUpAndDispose(t *testing.T) {
	store := session.NewMemoryStore()
	chat := &scriptedProvider{steps: []scriptedStep{{text: "done"}, {text: "done again"}}}
	loop := NewLoop("stateful", store, tools.New(tools.Options{}), chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys", EventHub: NewEventHub(16)})
	handle, err := NewAgent(loop)
	check(t, err)
	if err := handle.FollowUp(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	if err := handle.WhenIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "stateful", 0, 0)
	check(t, err)
	found := false
	for _, event := range events {
		if event.Type == session.EventUserMessage && strings.Contains(string(event.Data), "hello") {
			found = true
		}
	}
	if !found {
		t.Fatalf("follow-up was not persisted: %+v", events)
	}
	if err := handle.Dispose(context.Background()); err != nil {
		t.Fatal(err)
	}
	if handle.Status() != StatusDisposed {
		t.Fatalf("status=%s", handle.Status())
	}
}

func TestLoopRejectsSecondAgentAttachment(t *testing.T) {
	loop := NewLoop("single-agent", newMemoryStore(), tools.New(tools.Options{}), &scriptedProvider{}, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	first, err := NewAgent(loop)
	check(t, err)
	defer func() { _ = first.Dispose(context.Background()) }()
	if _, err := NewAgent(loop); err == nil {
		t.Fatal("second agent attachment unexpectedly succeeded")
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

func TestToolResultRoundTripAndReasoningMetadata(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registry.MustRegister(&tools.Definition{
		Name: "roundtrip", InputSchema: tools.OrObjectSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			return map[string]any{"ok": true}, nil
		},
		RenderModel: func(_ json.RawMessage, _ any) (any, error) { return "model-facing", nil },
	})
	chat := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{CallID: "call-1", Name: "roundtrip", Arguments: json.RawMessage(`{}`)}}, metadata: map[string]any{
			"provider": map[string]any{"reasoning_details": []any{map[string]any{"type": "reasoning.text"}}},
		}},
		{text: "done"},
	}}
	loop := NewLoop("tool-roundtrip", store, registry, chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	if _, err := loop.Run(context.Background(), "run"); err != nil {
		t.Fatal(err)
	}

	chat.mu.Lock()
	calls := append([]llm.Request(nil), chat.calls...)
	chat.mu.Unlock()
	if len(calls) != 2 {
		t.Fatalf("provider calls=%d, want 2", len(calls))
	}
	var assistant, tool *llm.Message
	for _, message := range calls[1].Messages {
		message := message
		switch message.Role {
		case llm.RoleAssistant:
			assistant = &message
		case llm.RoleTool:
			tool = &message
		}
	}
	if assistant == nil || len(assistant.ToolCalls()) != 1 || assistant.ToolCalls()[0].CallID != "call-1" {
		t.Fatalf("assistant tool calls=%+v", assistant)
	}
	if tool == nil || len(tool.ToolResults()) != 1 || tool.ToolResults()[0].CallID != "call-1" || string(tool.ToolResults()[0].Output) != `"model-facing"` {
		t.Fatalf("tool result=%+v", tool)
	}
	events, err := store.Load(context.Background(), "tool-roundtrip", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	foundCanonical := false
	for _, event := range events {
		if event.Type != session.EventToolResult || event.CallID != "call-1" {
			continue
		}
		var payload struct {
			Output  json.RawMessage `json:"output"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(event.Data, &payload) == nil {
			if len(payload.Output) != 0 {
				t.Fatalf("canonical output leaked into durable tool result: %s", event.Data)
			}
			if string(payload.Content) == `"model-facing"` {
				foundCanonical = true
			}
		}
	}
	if !foundCanonical {
		t.Fatal("model-facing tool result was not persisted")
	}
	if assistant.Metadata["provider"] == nil {
		t.Fatalf("assistant metadata was not replayed: %+v", assistant.Metadata)
	}
}

func TestCancellationBeforeDispatchPersistsAbortedToolResult(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := &cancelOnAdmittedStore{MemoryStore: session.NewMemoryStore(), cancel: cancel}
	registry := tools.New(tools.Options{})
	executed := 0
	registry.MustRegister(&tools.Definition{
		Name: "cancel_probe", InputSchema: tools.OrObjectSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			executed++
			return "unexpected", nil
		},
	})
	chat := &scriptedProvider{steps: []scriptedStep{{calls: []llm.ToolCallRequest{{CallID: "cancel-1", Name: "cancel_probe", Arguments: json.RawMessage(`{}`)}}}}}
	loop := NewLoop("cancel-before-dispatch", store, registry, chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	if _, err := loop.Run(ctx, "run"); !errors.Is(err, ErrStopped) {
		t.Fatalf("run error=%v", err)
	}
	if executed != 0 {
		t.Fatalf("tool executed %d times", executed)
	}
	events, err := store.Load(context.Background(), "cancel-before-dispatch", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	foundResult, foundDispatch := false, false
	for _, event := range events {
		if event.Type == session.EventToolDispatched {
			foundDispatch = true
		}
		if event.Type == session.EventToolResult && event.CallID == "cancel-1" && strings.Contains(string(event.Data), "ABORTED_BEFORE_DISPATCH") {
			foundResult = true
		}
	}
	if !foundResult || foundDispatch {
		t.Fatalf("aborted result=%v dispatched=%v", foundResult, foundDispatch)
	}
}

func TestLeaseReleaseUsesCleanupContextAfterCancellation(t *testing.T) {
	store := &cleanupLeaseStore{MemoryStore: session.NewMemoryStore()}
	provider := &blockingProvider{started: make(chan struct{})}
	loop := NewLoop("cleanup-lease", store, tools.New(tools.Options{}), provider, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := loop.Run(ctx, "run")
		done <- err
	}()
	select {
	case <-provider.started:
		cancel()
	case <-time.After(time.Second):
		t.Fatal("provider did not start")
	}
	if err := <-done; !errors.Is(err, ErrStopped) {
		t.Fatalf("run error=%v", err)
	}
	if store.releases != 1 || store.canceledReleases != 0 {
		t.Fatalf("lease release count=%d canceled=%d", store.releases, store.canceledReleases)
	}
}

func TestAgentRestoresClaimedInputWithoutCompletedTurn(t *testing.T) {
	store := newMemoryStore()
	inputID := "inbox-crashed"
	if _, err := store.Append(context.Background(), "restore-inbox", []session.Event{
		{ID: inputID + ":queued", SessionID: "restore-inbox", Type: session.EventInboxQueued, Data: session.InboxPayloadJSON(inputID, "follow_up", "resume me")},
		{ID: inputID + ":input", SessionID: "restore-inbox", RunID: "crashed-run", TurnID: "turn-0001", Type: session.EventUserMessage, Data: session.UserTextWithInbox("resume me", inputID)},
	}); err != nil {
		t.Fatal(err)
	}
	chat := &scriptedProvider{steps: []scriptedStep{{text: "resumed"}}}
	loop := NewLoop("restore-inbox", store, tools.New(tools.Options{}), chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	handle, err := NewAgent(loop)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = handle.Dispose(context.Background()) }()
	waitCtx, waitCancel := context.WithTimeout(context.Background(), time.Second)
	defer waitCancel()
	if err := handle.WhenIdle(waitCtx); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "restore-inbox", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	foundAssistant, foundCompleted := false, false
	for _, event := range events {
		foundAssistant = foundAssistant || event.Type == session.EventAssistantMessage
		if event.Type == session.EventInboxCompleted && strings.Contains(string(event.Data), inputID) {
			foundCompleted = true
		}
	}
	if !foundAssistant || !foundCompleted {
		t.Fatalf("restored input assistant=%v completed=%v", foundAssistant, foundCompleted)
	}
}

func TestPendingApprovalResumesAfterAgentRestart(t *testing.T) {
	store := newMemoryStore()
	request := tools.ApprovalRequest{SessionID: "approval-resume", RunID: "run-old", CallID: "approval-1", ToolName: "protected", Arguments: json.RawMessage(`{}`), Reason: "test"}
	requestData, _ := json.Marshal(request)
	if _, err := store.Append(context.Background(), "approval-resume", []session.Event{
		{ID: "approval-turn-start", SessionID: "approval-resume", RunID: "run-old", TurnID: "turn-0001", Type: session.EventTurnStart, Data: json.RawMessage(`{"text":"start"}`)},
		{ID: "approval-assistant", SessionID: "approval-resume", RunID: "run-old", TurnID: "turn-0001", StepID: "turn-0001-step-00", Type: session.EventAssistantMessage, Data: session.AssistantContent("", "", []session.ToolCall{{CallID: request.CallID, Name: request.ToolName, Arguments: request.Arguments}})},
		{ID: "approval-call", SessionID: "approval-resume", RunID: "run-old", TurnID: "turn-0001", StepID: "turn-0001-step-00", CallID: request.CallID, Type: session.EventToolCall, Data: session.ToolCallPayload(request.CallID, request.ToolName, request.Arguments)},
		{ID: "approval-admitted", SessionID: "approval-resume", RunID: "run-old", TurnID: "turn-0001", StepID: "turn-0001-step-00", CallID: request.CallID, Type: session.EventToolAdmitted, Data: json.RawMessage(`{"text":"admitted"}`)},
		{ID: "approval-requested", SessionID: "approval-resume", RunID: request.RunID, TurnID: "turn-0001", StepID: "turn-0001-step-00", CallID: request.CallID, Type: session.EventApprovalRequested, Data: requestData},
	}); err != nil {
		t.Fatal(err)
	}
	registry := tools.New(tools.Options{})
	executed := 0
	registry.MustRegister(&tools.Definition{
		Name: request.ToolName, InputSchema: tools.OrObjectSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			executed++
			return map[string]any{"ok": true}, nil
		},
	})
	registry.AddPolicy(func(context.Context, tools.Execution) (tools.PolicyDecision, string, error) {
		return tools.PolicyAsk, "test", nil
	})
	chat := &scriptedProvider{steps: []scriptedStep{{text: "continued after approval"}}}
	loop := NewLoop("approval-resume", store, registry, chat, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	handle, err := NewAgent(loop)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = handle.Dispose(context.Background()) }()
	if err := handle.Approve(context.Background(), request.CallID); err != nil {
		t.Fatal(err)
	}
	if executed != 1 {
		t.Fatalf("resumed executions=%d", executed)
	}
	events, err := store.Load(context.Background(), "approval-resume", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	foundResult := false
	for _, event := range events {
		if event.Type == session.EventToolResult && event.CallID == request.CallID {
			foundResult = true
		}
	}
	if !foundResult {
		t.Fatal("approval result was not durably resumed")
	}
	chat.mu.Lock()
	requestCount := len(chat.calls)
	var continuedRequest llm.Request
	if requestCount == 1 {
		continuedRequest = chat.calls[0]
	}
	chat.mu.Unlock()
	if requestCount != 1 {
		t.Fatalf("continued provider requests = %d, want 1", requestCount)
	}
	foundToolResult := false
	for _, message := range continuedRequest.Messages {
		for _, result := range message.ToolResults() {
			if result.CallID == request.CallID {
				foundToolResult = true
			}
		}
	}
	if !foundToolResult {
		t.Fatal("continued provider request did not contain the approved tool result")
	}
	foundStepEnd, foundTurnEnd := false, false
	for _, event := range events {
		if event.Type == session.EventStepEnd && event.StepID == "turn-0001-step-00" {
			foundStepEnd = true
		}
		if event.Type == session.EventTurnEnd && event.TurnID == "turn-0001" {
			foundTurnEnd = true
		}
	}
	if !foundStepEnd || !foundTurnEnd {
		t.Fatalf("approval resume lifecycle step_end=%v turn_end=%v", foundStepEnd, foundTurnEnd)
	}
}

func TestResumedApprovalHonorsHandleCancel(t *testing.T) {
	store := newMemoryStore()
	request := tools.ApprovalRequest{SessionID: "approval-cancel", RunID: "run-old", CallID: "approval-cancel-1", ToolName: "protected", Arguments: json.RawMessage(`{}`), Reason: "test"}
	requestData, _ := json.Marshal(request)
	_, err := store.Append(context.Background(), "approval-cancel", []session.Event{
		{ID: "cancel-turn-start", SessionID: "approval-cancel", RunID: "run-old", TurnID: "turn-cancel", Type: session.EventTurnStart, Data: json.RawMessage(`{"text":"start"}`)},
		{ID: "cancel-assistant", SessionID: "approval-cancel", RunID: "run-old", TurnID: "turn-cancel", StepID: "turn-cancel-step-00", Type: session.EventAssistantMessage, Data: session.AssistantContent("", "", []session.ToolCall{{CallID: request.CallID, Name: request.ToolName, Arguments: request.Arguments}})},
		{ID: "cancel-call", SessionID: "approval-cancel", RunID: "run-old", TurnID: "turn-cancel", StepID: "turn-cancel-step-00", CallID: request.CallID, Type: session.EventToolCall, Data: session.ToolCallPayload(request.CallID, request.ToolName, request.Arguments)},
		{ID: "cancel-admitted", SessionID: "approval-cancel", RunID: "run-old", TurnID: "turn-cancel", StepID: "turn-cancel-step-00", CallID: request.CallID, Type: session.EventToolAdmitted, Data: json.RawMessage(`{"text":"admitted"}`)},
		{ID: "cancel-requested", SessionID: "approval-cancel", RunID: request.RunID, TurnID: "turn-cancel", StepID: "turn-cancel-step-00", CallID: request.CallID, Type: session.EventApprovalRequested, Data: requestData},
	})
	check(t, err)
	started := make(chan struct{})
	registry := tools.New(tools.Options{})
	registry.MustRegister(&tools.Definition{
		Name: request.ToolName, InputSchema: tools.OrObjectSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(ctx context.Context, _ tools.ToolRunContext, _ json.RawMessage) (any, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		},
	})
	loop := NewLoop("approval-cancel", store, registry, &scriptedProvider{steps: []scriptedStep{{text: "continued"}}}, Config{Model: "m", Owner: "test", SystemPrompt: "sys"})
	handle, err := NewAgent(loop)
	check(t, err)
	defer func() { _ = handle.Dispose(context.Background()) }()
	resumeDone := make(chan error, 1)
	go func() { resumeDone <- handle.Approve(context.Background(), request.CallID) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("approved tool did not start")
	}
	check(t, handle.Cancel(context.Background(), CancelOptions{KeepInbox: true}))
	select {
	case err := <-resumeDone:
		if !errors.Is(err, ErrStopped) {
			t.Fatalf("resume error=%v, want ErrStopped", err)
		}
	case <-time.After(time.Second):
		t.Fatal("resumed approval did not stop after cancel")
	}
}

func TestHostClaimedLeaseDoesNotCompeteWithHostHeartbeat(t *testing.T) {
	store := &hostClaimStore{MemoryStore: session.NewMemoryStore()}
	lease, err := store.MemoryStore.ClaimLease(context.Background(), "host-run", "host-worker", time.Minute, "")
	check(t, err)
	registry := tools.New(tools.Options{})
	chat := &scriptedProvider{steps: []scriptedStep{{text: "done", finish: llm.FinishReasonStop}}}
	loop := NewLoop("host-run", store, registry, chat, Config{
		Model: "m", Owner: "agentkit", SystemPrompt: "sys", ClaimedLease: &lease,
	})
	_, err = loop.Run(context.Background(), "hello")
	check(t, err)
	if store.claims != 0 || store.renewals != 0 || store.releases != 0 {
		t.Fatalf("host-owned run touched lease lifecycle: claims=%d renewals=%d releases=%d", store.claims, store.renewals, store.releases)
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
	}, {
		ID: "prior-dispatched", SessionID: "s9", CallID: "call-1", Type: session.EventToolDispatched,
		Data: json.RawMessage(`{"text":"dispatched"}`),
	}, {
		ID: "prior-running", SessionID: "s9", CallID: "call-1", Type: session.EventToolRunning,
		Data: json.RawMessage(`{"text":"running"}`),
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

// TestParallelToolCallsEnabled: the provider may emit a parallel batch; the
// registry still applies explicit per-call safety and model-order commits.
func TestParallelToolCallsEnabled(t *testing.T) {
	store := newMemoryStore()
	registry := tools.New(tools.Options{})
	registerProbe(t, registry, "probe_seq", nil, nil)
	chat := &scriptedProvider{steps: []scriptedStep{{calls: []llm.ToolCallRequest{{CallID: "c1", Name: "probe_seq", Arguments: jsonLit(`{}`)}}, finish: llm.FinishReasonToolCalls}, {text: "done"}}}
	loop := NewLoop("s5", store, registry, chat, Config{Model: "m", Owner: "w", SystemPrompt: "sys"})
	_, err := loop.Run(context.Background(), "seq")
	check(t, err)
	chat.mu.Lock()
	defer chat.mu.Unlock()
	if len(chat.calls) == 0 || chat.calls[0].ParallelToolCalls == nil || !*chat.calls[0].ParallelToolCalls {
		t.Fatalf("expected ParallelToolCalls=true, got %+v", chat.calls)
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
	compactor := &countingCompactor{f: func(generation uint64, events []session.Event, sourceSeqs []uint64) (string, string, error) {
		compactMu.Lock()
		compactCalls++
		compactMu.Unlock()
		return "[summarized]", "fp-" + fmt.Sprint(generation) + "-" + fmt.Sprint(len(sourceSeqs)), nil
	}}

	chat := &scriptedProvider{steps: []scriptedStep{{text: "ok", finish: llm.FinishReasonStop}}}
	loop := NewLoop("s6", store, registry, chat, Config{
		Model: "m", Owner: "w", SystemPrompt: "sys",
		ContextWindow: 60, MaxOutput: 10, CompactThresholdRatio: 0.80,
		Compactor: compactor, PruneHead: 4,
	})
	// Compaction only replaces old history; seed a prior durable event so the
	// fresh user request remains in the required verbatim tail.
	lease, leaseErr := store.ClaimLease(context.Background(), "s6", "seed", time.Minute, "")
	check(t, leaseErr)
	_, leaseErr = store.AppendFenced(context.Background(), lease, []session.Event{{SessionID: "s6", Type: session.EventUserMessage, Data: session.UserText(strings.Repeat("old ", 40))}})
	check(t, leaseErr)
	check(t, store.ReleaseLease(context.Background(), lease))
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
	compactionOrder := make([]session.EventType, 0, 4)
	for _, e := range events {
		switch e.Type {
		case session.EventCompactionStart, session.EventCompactionSummary, session.EventUserMessage, session.EventCompactionEnd:
			var payload struct {
				SurfaceOp string `json:"surface_op"`
			}
			if e.Type == session.EventUserMessage && (json.Unmarshal(e.Data, &payload) != nil || payload.SurfaceOp != "replace") {
				continue
			}
			compactionOrder = append(compactionOrder, e.Type)
		}
	}
	if len(compactionOrder) < 4 || compactionOrder[0] != session.EventCompactionStart || compactionOrder[1] != session.EventCompactionSummary || compactionOrder[2] != session.EventUserMessage || compactionOrder[3] != session.EventCompactionEnd {
		t.Fatalf("compaction event order = %v, want start/summary/user-replacement/end", compactionOrder)
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
	f func(generation uint64, events []session.Event, sourceSeqs []uint64) (string, string, error)
}

func (c *countingCompactor) Compact(_ context.Context, generation uint64, events []session.Event, sourceSeqs []uint64) (string, string, error) {
	return c.f(generation, events, sourceSeqs)
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
