package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

func TestDeferredQuestionsCanResumeAcrossRoundsBeforePlan(t *testing.T) {
	store := session.NewMemoryStore()
	registry := tools.New(tools.Options{})
	questionSchema := json.RawMessage(`{"type":"object","properties":{"round":{"type":"integer"}},"additionalProperties":false}`)
	registry.MustRegister(&tools.Definition{
		Name: "ask_questions", Description: "ask questions", InputSchema: questionSchema, OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			return map[string]any{"questions": []map[string]any{{"id": "q", "prompt": "Choose", "type": "single", "required": true}}}, nil
		},
		ResolveContinuation: func(args json.RawMessage, _ any) (tools.ToolContinuation, string, string, error) {
			var input struct {
				Round int `json:"round"`
			}
			if err := json.Unmarshal(args, &input); err != nil {
				return tools.ToolContinue, "", "", err
			}
			return tools.ToolDeferred, "questions:" + strconv.Itoa(input.Round), "questions", nil
		},
	})
	registry.MustRegister(&tools.Definition{
		Name: "propose_plan", Description: "propose plan", InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			return map[string]any{"plan": "ready"}, nil
		},
		ResolveContinuation: func(json.RawMessage, any) (tools.ToolContinuation, string, string, error) {
			return tools.ToolConclude, "", "", nil
		},
	})
	provider := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{CallID: "question-1", Name: "ask_questions", Arguments: json.RawMessage(`{"round":1}`)}}, finish: llm.FinishReasonToolCalls},
		{calls: []llm.ToolCallRequest{{CallID: "question-2", Name: "ask_questions", Arguments: json.RawMessage(`{"round":2}`)}}, finish: llm.FinishReasonToolCalls},
		{calls: []llm.ToolCallRequest{{CallID: "plan-1", Name: "propose_plan", Arguments: json.RawMessage(`{}`)}}, finish: llm.FinishReasonToolCalls},
	}}
	loop := NewLoop("questions-plan-flow", store, registry, provider, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	ctx := context.Background()
	if _, err := loop.Run(ctx, "discover the project"); !errors.Is(err, ErrToolDeferred) {
		t.Fatalf("first round error = %v, want deferred", err)
	}
	if err := loop.ResumeMaterializedTool(ctx, ToolResume{
		CallID: "question-1", ResumeKey: "questions:1",
		Result: &tools.Result{Name: "ask_questions", Canonical: map[string]any{"answers": []map[string]any{{"id": "q", "value": "one"}}}},
	}); !errors.Is(err, ErrToolDeferred) {
		t.Fatalf("second round error = %v, want deferred", err)
	}
	if err := loop.ResumeMaterializedTool(ctx, ToolResume{
		CallID: "question-2", ResumeKey: "questions:2",
		Result: &tools.Result{Name: "ask_questions", Canonical: map[string]any{"answers": []map[string]any{{"id": "q", "value": "two"}}}},
	}); err != nil {
		t.Fatalf("plan round error = %v", err)
	}

	events, err := store.Load(ctx, "questions-plan-flow", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	materializedResults := 0
	planResults := 0
	deferredCalls := map[string]bool{}
	for _, event := range events {
		switch event.Type {
		case session.EventToolDeferred:
			deferredCalls[event.CallID] = true
		case session.EventToolResult:
			var payload struct {
				Name         string `json:"name"`
				Materialized bool   `json:"materialized"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				t.Fatal(err)
			}
			if payload.Name == "ask_questions" {
				if !payload.Materialized {
					t.Fatalf("question result %s is missing materialized marker", event.CallID)
				}
				materializedResults++
			}
			if payload.Name == "propose_plan" {
				planResults++
				if payload.Materialized {
					t.Fatal("plan result must not be marked materialized")
				}
			}
		}
	}
	if len(provider.calls) != 3 || materializedResults != 2 || planResults != 1 || !deferredCalls["question-1"] || !deferredCalls["question-2"] {
		t.Fatalf("provider calls=%d materialized_results=%d plan_results=%d deferred=%v", len(provider.calls), materializedResults, planResults, deferredCalls)
	}
}
