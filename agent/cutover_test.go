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
	var deferred, resumeStarted, result, deferredContent bool
	for _, event := range events {
		deferred = deferred || event.Type == session.EventToolDeferred
		resumeStarted = resumeStarted || event.Type == session.EventToolResumeStarted && event.CallID == "call-1"
		result = result || event.Type == session.EventToolResult && event.CallID == "call-1"
		if event.Type == session.EventToolDeferred && event.CallID == "call-1" {
			var payload struct {
				Content json.RawMessage `json:"content"`
			}
			deferredContent = json.Unmarshal(event.Data, &payload) == nil && len(payload.Content) > 0
		}
	}
	if !deferred || !deferredContent || !resumeStarted || !result {
		t.Fatalf("deferred=%v content=%v resume_started=%v result=%v, want durable resume barrier", deferred, deferredContent, resumeStarted, result)
	}
}

func TestMaterializedDeferredResumeCanFinishAfterResumeBarrier(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "materialized-resume-session", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "ask_questions", json.RawMessage(`{}`))},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("ask_questions", "questions:call-1", "questions", json.RawMessage(`{"questions":[]}`), map[string]any{"status": "waiting"})},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolResumeStarted, Data: session.ToolResumeStartedPayloadWithMode("call-1", "ask_questions", "questions:call-1", true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	loop := NewLoop("materialized-resume-session", store, tools.New(tools.Options{}), &scriptedProvider{steps: []scriptedStep{{text: "done", finish: llm.FinishReasonStop}}}, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if err := loop.ResumeMaterializedTool(ctx, ToolResume{
		CallID: "call-1", ResumeKey: "questions:call-1",
		Result: &tools.Result{CallID: "call-1", Name: "different_tool", Canonical: map[string]any{"answers": []any{}}},
	}); err == nil {
		t.Fatal("materialized resume accepted a different tool name")
	}
	if err := loop.ResumeMaterializedTool(ctx, ToolResume{
		CallID: "call-1", ResumeKey: "questions:call-1",
		Result: &tools.Result{CallID: "call-1", Name: "ask_questions", Canonical: map[string]any{"answers": []any{}}},
	}); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(ctx, "materialized-resume-session", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	resumeStarted, results, unknown := 0, 0, false
	for _, event := range events {
		if event.Type == session.EventToolResumeStarted && event.CallID == "call-1" {
			resumeStarted++
		}
		if event.Type != session.EventToolResult || event.CallID != "call-1" {
			continue
		}
		results++
		var payload struct {
			Code string `json:"code"`
		}
		_ = json.Unmarshal(event.Data, &payload)
		unknown = unknown || payload.Code == "TOOL_OUTCOME_UNKNOWN"
	}
	if resumeStarted != 1 || results != 1 || unknown {
		t.Fatalf("resume_started=%d results=%d unknown=%v", resumeStarted, results, unknown)
	}
}

func TestMaterializedDeferredResumeContinuesAfterResultCrash(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "materialized-result-crash", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "ask_questions", json.RawMessage(`{}`))},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("ask_questions", "questions:call-1", "questions", json.RawMessage(`{"questions":[]}`), map[string]any{"status": "waiting"})},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolResumeStarted, Data: session.ToolResumeStartedPayloadWithMode("call-1", "ask_questions", "questions:call-1", true)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolResult, Data: session.ToolResultStructuredPayloadWithContent("call-1", "ask_questions", json.RawMessage(`{"answers":[]}`), nil, "", false, nil, nil, false)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	chat := &scriptedProvider{steps: []scriptedStep{{text: "continued", finish: llm.FinishReasonStop}}}
	loop := NewLoop("materialized-result-crash", store, tools.New(tools.Options{}), chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if err := loop.ResumeMaterializedTool(ctx, ToolResume{
		CallID: "call-1", ResumeKey: "questions:call-1",
		Result: &tools.Result{CallID: "call-1", Name: "ask_questions", Canonical: map[string]any{"answers": []any{}}},
	}); err != nil {
		t.Fatal(err)
	}
	if len(chat.calls) != 1 {
		t.Fatalf("provider calls = %d, want one continuation call", len(chat.calls))
	}
	events, err := store.Load(ctx, "materialized-result-crash", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	resumeStarted, results, stepEnds := 0, 0, 0
	for _, event := range events {
		switch event.Type {
		case session.EventToolResumeStarted:
			if event.CallID == "call-1" {
				resumeStarted++
			}
		case session.EventToolResult:
			if event.CallID == "call-1" {
				results++
			}
		case session.EventStepEnd:
			if event.StepID == "step-1" {
				stepEnds++
			}
		}
	}
	if resumeStarted != 1 || results != 1 || stepEnds != 1 {
		t.Fatalf("resume_started=%d results=%d original_step_ends=%d", resumeStarted, results, stepEnds)
	}
}

func TestToolResumeRetainsLegacyUnkeyedFieldShape(t *testing.T) {
	request := ToolResume{"call-1", "resume-1", nil, nil}
	if request.CallID != "call-1" || request.ResumeKey != "resume-1" {
		t.Fatalf("legacy ToolResume fields changed: %+v", request)
	}
}

func TestSingleDeferredResumeKeepsBatchOpenUntilEveryCallResolves(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "single-resume-batch", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	calls := []session.ToolCall{
		{CallID: "call-1", Name: "question_one", Arguments: json.RawMessage(`{}`)},
		{CallID: "call-2", Name: "question_two", Arguments: json.RawMessage(`{}`)},
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventAssistantMessage, Data: session.AssistantContent("", "", calls)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "question_one", calls[0].Arguments)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("question_one", "questions:call-1", "questions", nil, nil)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-2", Type: session.EventToolCall, ID: "call-2", Data: session.ToolCallPayload("call-2", "question_two", calls[1].Arguments)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-2", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("question_two", "questions:call-2", "questions", nil, nil)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	chat := &scriptedProvider{steps: []scriptedStep{{text: "continued"}}}
	loop := NewLoop("single-resume-batch", store, tools.New(tools.Options{}), chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if err := loop.ResumeTool(ctx, ToolResume{CallID: "call-1", ResumeKey: "questions:call-1", Result: &tools.Result{Canonical: map[string]any{"answer": "one"}}}); !errors.Is(err, ErrToolDeferred) {
		t.Fatalf("first resume error = %v, want deferred", err)
	}
	firstEvents, err := store.Load(ctx, "single-resume-batch", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range firstEvents {
		if event.Type == session.EventStepEnd && event.StepID == "step-1" {
			t.Fatal("first single resume closed the step before the second result")
		}
	}
	if err := loop.ResumeTool(ctx, ToolResume{CallID: "call-2", ResumeKey: "questions:call-2", Result: &tools.Result{Canonical: map[string]any{"answer": "two"}}}); err != nil {
		t.Fatal(err)
	}
	if len(chat.calls) != 1 {
		t.Fatalf("provider calls = %d, want one continuation", len(chat.calls))
	}
}

func TestMaterializedDeferredFailureIsPersistedAsError(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "materialized-failure", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "workspace", json.RawMessage(`{}`))},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("workspace", "workspace:call-1", "workspace", nil, nil)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	loop := NewLoop("materialized-failure", store, tools.New(tools.Options{}), &scriptedProvider{steps: []scriptedStep{{text: "recovered", finish: llm.FinishReasonStop}}}, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if err := loop.ResumeMaterializedTool(ctx, ToolResume{
		CallID: "call-1", ResumeKey: "workspace:call-1",
		Result: &tools.Result{
			CallID: "call-1", Name: "workspace", Code: "workspace_failed",
			ModelFacing: map[string]any{"ok": false},
			Failure:     &tools.Failure{Code: "workspace_failed", Message: "workspace failed"},
		},
		Err: errors.New("workspace failed"),
	}); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(ctx, "materialized-failure", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range events {
		if event.Type != session.EventToolResult || event.CallID != "call-1" {
			continue
		}
		var payload struct {
			IsError bool   `json:"is_error"`
			Code    string `json:"code"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			t.Fatal(err)
		}
		if !payload.IsError || payload.Code != "workspace_failed" {
			t.Fatalf("materialized failure payload = %+v", payload)
		}
		return
	}
	t.Fatal("materialized failure result was not persisted")
}

func TestMaterializedDeferredBatchResolvesEveryCallBeforeContinuing(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "materialized-failure-batch", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	calls := []session.ToolCall{
		{CallID: "call-1", Name: "write", Arguments: json.RawMessage(`{"path":"a"}`)},
		{CallID: "call-2", Name: "bash", Arguments: json.RawMessage(`{"command":"false"}`)},
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventAssistantMessage, Data: session.AssistantContent("", "", calls)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "write", calls[0].Arguments)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("write", "workspace:call-1", "workspace_activation", nil, nil)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-2", Type: session.EventToolCall, ID: "call-2", Data: session.ToolCallPayload("call-2", "bash", calls[1].Arguments)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-2", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("bash", "workspace:call-2", "workspace_activation", nil, nil)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	chat := &scriptedProvider{steps: []scriptedStep{{text: "recovered", finish: llm.FinishReasonStop}}}
	loop := NewLoop("materialized-failure-batch", store, tools.New(tools.Options{}), chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	failure := func(callID, name string) ToolResume {
		return ToolResume{
			CallID: callID, ResumeKey: "workspace:" + callID,
			Result: &tools.Result{CallID: callID, Name: name, Code: "workspace_activation_failed", ModelFacing: map[string]any{"ok": false}, Failure: &tools.Failure{Code: "workspace_activation_failed", Message: "workspace failed"}},
			Err:    errors.New("workspace failed"),
		}
	}
	if err := loop.ResumeMaterializedTools(ctx, []ToolResume{failure("call-2", "bash"), failure("call-1", "write")}); err != nil {
		t.Fatal(err)
	}
	if len(chat.calls) != 1 {
		t.Fatalf("provider calls = %d, want one continuation call", len(chat.calls))
	}
	providerResults := map[string]bool{}
	providerOrder := make([]string, 0, 2)
	for _, message := range chat.calls[0].Messages {
		if message.Role != llm.RoleTool {
			continue
		}
		for _, result := range message.ToolResults() {
			providerResults[result.CallID] = result.IsError
			providerOrder = append(providerOrder, result.CallID)
		}
	}
	if !providerResults["call-1"] || !providerResults["call-2"] {
		t.Fatalf("provider tool results = %#v", providerResults)
	}
	if len(providerOrder) != 2 || providerOrder[0] != "call-1" || providerOrder[1] != "call-2" {
		t.Fatalf("provider tool result order = %v, want [call-1 call-2]", providerOrder)
	}
	events, err := store.Load(ctx, "materialized-failure-batch", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	results := map[string]bool{}
	stepEnds := 0
	for _, event := range events {
		if event.Type == session.EventStepEnd && event.StepID == "step-1" {
			stepEnds++
		}
		if event.Type != session.EventToolResult {
			continue
		}
		var payload struct {
			IsError bool   `json:"is_error"`
			Code    string `json:"code"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			t.Fatal(err)
		}
		if event.CallID == "call-1" || event.CallID == "call-2" {
			results[event.CallID] = payload.IsError && payload.Code == "workspace_activation_failed"
		}
	}
	if !results["call-1"] || !results["call-2"] || stepEnds != 1 {
		t.Fatalf("results=%v original_step_ends=%d", results, stepEnds)
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

func TestDeferredToolResumeContinuesAfterResultCrash(t *testing.T) {
	store := session.NewMemoryStore()
	ctx := context.Background()
	lease, err := store.ClaimLease(ctx, "deferred-result-crash", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.AppendFenced(ctx, lease, []session.Event{
		{RunID: "run-1", TurnID: "turn-1", Type: session.EventTurnStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", Type: session.EventStepStart, Data: json.RawMessage(`{}`)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolCall, ID: "call-1", Data: session.ToolCallPayload("call-1", "side_effect", json.RawMessage(`{}`))},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolDeferred, Data: session.ToolDeferredPayload("side_effect", "workspace:call-1", "workspace", nil, nil)},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolResumeStarted, Data: session.ToolResumeStartedPayload("call-1", "side_effect", "workspace:call-1")},
		{RunID: "run-1", TurnID: "turn-1", StepID: "step-1", CallID: "call-1", Type: session.EventToolResult, Data: session.ToolResultStructuredPayloadWithContent("call-1", "side_effect", json.RawMessage(`{"ok":true}`), nil, "", false, nil, nil, false)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(ctx, lease); err != nil {
		t.Fatal(err)
	}
	chat := &scriptedProvider{steps: []scriptedStep{{text: "continued", finish: llm.FinishReasonStop}}}
	registry := tools.New(tools.Options{})
	executions := 0
	registry.MustRegister(&tools.Definition{
		Name: "side_effect", Description: "side effect", InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: tools.AnyOutputSchema,
		Execute: func(context.Context, tools.ToolRunContext, json.RawMessage) (any, error) {
			executions++
			return map[string]any{"ok": true}, nil
		},
	})
	loop := NewLoop("deferred-result-crash", store, registry, chat, Config{Model: "m", Owner: "worker", SystemPrompt: "sys"})
	if err := loop.ResumeDeferredTools(ctx, []DeferredToolResume{{CallID: "call-1", ResumeKey: "workspace:call-1"}}); err != nil {
		t.Fatal(err)
	}
	if executions != 0 || len(chat.calls) != 1 {
		t.Fatalf("tool executions=%d provider calls=%d, want no retry and one continuation", executions, len(chat.calls))
	}
	events, err := store.Load(ctx, "deferred-result-crash", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	resumeStarted, results, stepEnds := 0, 0, 0
	for _, event := range events {
		switch event.Type {
		case session.EventToolResumeStarted:
			if event.CallID == "call-1" {
				resumeStarted++
			}
		case session.EventToolResult:
			if event.CallID == "call-1" {
				results++
			}
		case session.EventStepEnd:
			if event.StepID == "step-1" {
				stepEnds++
			}
		}
	}
	if resumeStarted != 1 || results != 1 || stepEnds != 1 {
		t.Fatalf("resume_started=%d results=%d original_step_ends=%d", resumeStarted, results, stepEnds)
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
