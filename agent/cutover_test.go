package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

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
		FinalizeContent: func(result *tools.Result) error {
			result.Continuation = tools.ToolDeferred
			result.ResumeKey = "worker-result-1"
			result.WaitingReason = "workspace"
			return nil
		},
		Execute: func(context.Context, tools.ExecContext, json.RawMessage) (any, error) {
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
	var deferred, result bool
	for _, event := range events {
		deferred = deferred || event.Type == session.EventToolDeferred
		result = result || event.Type == session.EventToolResult && event.CallID == "call-1"
	}
	if !deferred || !result {
		t.Fatalf("deferred=%v result=%v, want both durable events", deferred, result)
	}
}

type runPolicyFunc func(context.Context, StepSnapshot) (StepDecision, error)

func (f runPolicyFunc) BeforeStep(ctx context.Context, snapshot StepSnapshot) (StepDecision, error) {
	return f(ctx, snapshot)
}
