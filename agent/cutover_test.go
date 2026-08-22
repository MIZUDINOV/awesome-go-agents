package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

func TestSubmitPersistsHostInputID(t *testing.T) {
	store := session.NewMemoryStore()
	chat := &scriptedProvider{steps: []scriptedStep{{text: "ok", finish: llm.FinishReasonStop}}}
	loop := NewLoop("submit-session", store, tools.New(tools.Options{}), chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	handle, err := NewAgent(loop)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = handle.Close(context.Background()) }()
	if _, err := handle.Submit(context.Background(), Input{ID: "input-42", Type: session.EventUserMessage, Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "submit-session", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, event := range events {
		if event.ID == "input-42:input" {
			found = true
		}
	}
	if !found {
		t.Fatal("host input id was not preserved in the durable event")
	}
}

func TestRunPolicyStopsBeforeNextModelStep(t *testing.T) {
	store := session.NewMemoryStore()
	chat := &scriptedProvider{steps: []scriptedStep{{text: "never", finish: llm.FinishReasonStop}}}
	loop := NewLoop("policy-session", store, tools.New(tools.Options{}), chat, Config{
		Model: "m", Owner: "worker", SystemPrompt: "sys",
		Policy: runPolicyFunc(func(context.Context, StepSnapshot) (StepDecision, error) { return StepStop, nil }),
	})
	_, err := loop.Run(context.Background(), "hello")
	if !errors.Is(err, ErrPolicyStopped) {
		t.Fatalf("Run() error = %v, want ErrPolicyStopped", err)
	}
	if len(chat.calls) != 0 {
		t.Fatalf("policy stop called provider %d times", len(chat.calls))
	}
}

func TestDeferredToolCanResumeThroughAgentAPI(t *testing.T) {
	store := session.NewMemoryStore()
	registry := tools.New(tools.Options{})
	registry.MustRegister(&tools.Definition{
		Name: "deferred", Description: "deferred", InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: tools.AnyOutputSchema,
		ResolveContinuation: func(_ json.RawMessage, _ any) (tools.ToolContinuation, string, string, error) {
			return tools.ToolDeferred, "worker-result-1", "workspace", nil
		},
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			return map[string]any{"accepted": true}, nil
		},
	})
	chat := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{CallID: "call-1", Name: "deferred", Arguments: json.RawMessage(`{}`)}}, finish: llm.FinishReasonToolCalls},
		{text: "resumed", finish: llm.FinishReasonStop},
	}}
	loop := NewLoop("deferred-session", store, registry, chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if _, err := loop.Run(context.Background(), "start"); !errors.Is(err, ErrToolDeferred) {
		t.Fatalf("Run() error = %v, want ErrToolDeferred", err)
	}
	if err := loop.ResumeTool(context.Background(), ToolResume{CallID: "call-1", ResumeKey: "worker-result-1", Result: &tools.Result{Canonical: map[string]any{"ok": true}}}); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "deferred-session", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	var deferred, resumeStarted, result bool
	for _, event := range events {
		deferred = deferred || event.Type == session.EventToolDeferred
		resumeStarted = resumeStarted || event.Type == session.EventToolResumeStarted && event.CallID == "call-1"
		result = result || event.Type == session.EventToolResult && event.CallID == "call-1"
	}
	if !deferred || !resumeStarted || !result {
		t.Fatalf("deferred=%v resume_started=%v result=%v, want durable resume barrier", deferred, resumeStarted, result)
	}
}

func TestDeferredToolResumesOriginalCallWithoutNewTurn(t *testing.T) {
	store := session.NewMemoryStore()
	registry := tools.New(tools.Options{})
	executions := 0
	registry.MustRegister(&tools.Definition{
		Name: "lazy", Description: "lazy", InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			executions++
			return map[string]any{"execution": executions}, nil
		},
		ResolveContinuation: func(_ json.RawMessage, _ any) (tools.ToolContinuation, string, string, error) {
			if executions == 1 {
				return tools.ToolDeferred, "workspace:call-1", "workspace", nil
			}
			return tools.ToolContinue, "", "", nil
		},
	})
	chat := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{CallID: "call-1", Name: "lazy", Arguments: json.RawMessage(`{}`)}}, finish: llm.FinishReasonToolCalls},
		{text: "done", finish: llm.FinishReasonStop},
	}}
	loop := NewLoop("deferred-resume-session", store, registry, chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if _, err := loop.Run(context.Background(), "start"); !errors.Is(err, ErrToolDeferred) {
		t.Fatalf("Run() error = %v, want ErrToolDeferred", err)
	}
	if err := loop.ResumeDeferredTools(context.Background(), []DeferredToolResume{{CallID: "call-1", ResumeKey: "workspace:call-1"}}); err != nil {
		t.Fatal(err)
	}
	if executions != 2 {
		t.Fatalf("tool executions = %d, want 2", executions)
	}
	events, err := store.Load(context.Background(), "deferred-resume-session", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	turnStarts, resumeStarts, results, resolvedStepEnds := 0, 0, 0, 0
	resumeIndex, resultIndex, resolvedStepEndIndex := -1, -1, -1
	resolvedStepID := ""
	for index, event := range events {
		switch event.Type {
		case session.EventTurnStart:
			turnStarts++
		case session.EventToolResumeStarted:
			resumeStarts++
			resumeIndex = index
		case session.EventToolResult:
			if event.CallID == "call-1" {
				results++
				resultIndex = index
				resolvedStepID = event.StepID
			}
		case session.EventStepEnd:
			if resolvedStepID != "" && event.StepID == resolvedStepID {
				resolvedStepEnds++
				resolvedStepEndIndex = index
			}
		}
	}
	if turnStarts != 1 || resumeStarts != 1 || results != 1 || resumeIndex < 0 || resultIndex <= resumeIndex {
		t.Fatalf("resume lifecycle turns=%d resume_started=%d results=%d indexes=%d/%d", turnStarts, resumeStarts, results, resumeIndex, resultIndex)
	}
	if resolvedStepEnds != 1 || resolvedStepEndIndex <= resultIndex {
		t.Fatalf("deferred step lifecycle step_ends=%d result=%d step_end=%d", resolvedStepEnds, resultIndex, resolvedStepEndIndex)
	}
}

func TestResumeStartedRecoveryDoesNotReexecuteTool(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "resume-crash-session", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "side_effect", json.RawMessage(`{}`))},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: json.RawMessage(`{"name":"side_effect","resume_key":"workspace:call-1"}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolResumeStarted, Data: session.ToolResumeStartedPayload("call-1", "side_effect", "workspace:call-1")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	executions := 0
	registry := tools.New(tools.Options{})
	registry.MustRegister(&tools.Definition{
		Name: "side_effect", Description: "side effect", InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			executions++
			return map[string]any{"ok": true}, nil
		},
	})
	loop := NewLoop("resume-crash-session", store, registry, &scriptedProvider{}, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if err := loop.ResumeDeferredTools(ctx, []DeferredToolResume{{CallID: "call-1", ResumeKey: "workspace:call-1"}}); err != nil {
		t.Fatal(err)
	}
	if executions != 0 {
		t.Fatalf("resumed tool executed %d times after crash barrier", executions)
	}
	all, _ := store.Load(ctx, "resume-crash-session", 0, 0)
	unknown := false
	for _, event := range all {
		if event.Type != session.EventToolResult || event.CallID != "call-1" {
			continue
		}
		var payload struct {
			Code string `json:"code"`
		}
		_ = json.Unmarshal(event.Data, &payload)
		unknown = unknown || payload.Code == "TOOL_OUTCOME_UNKNOWN"
	}
	if !unknown {
		t.Fatal("missing unknown outcome after resume barrier crash")
	}
}

type runPolicyFunc func(context.Context, StepSnapshot) (StepDecision, error)

func (f runPolicyFunc) BeforeStep(ctx context.Context, snapshot StepSnapshot) (StepDecision, error) {
	return f(ctx, snapshot)
}
