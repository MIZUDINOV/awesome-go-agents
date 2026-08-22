package tools

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func TestRegistryExecutePipeline(t *testing.T) {
	registry := New(Options{})
	inputSchema, err := RawSchema(map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}})
	if err != nil {
		t.Fatal(err)
	}
	executed := false
	err = registry.Register(&Definition{
		Name: "read", Description: "read a file",
		InputSchema: inputSchema, OutputSchema: AnyOutputSchema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) {
			executed = true
			return map[string]any{"content": "data"}, nil
		},
		RenderModel: func(_ json.RawMessage, canonical any) (any, error) {
			return "The file was read.", nil
		},
		PresentUI: func(_ json.RawMessage, canonical any) (map[string]any, error) {
			return map[string]any{"path": "a.ts"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := registry.Run(context.Background(), ExecContext{}, "read", "call_1", []byte(`{"path":"a.ts"}`))
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !executed {
		t.Error("executor not called")
	}
	if result.ModelFacing != "The file was read." {
		t.Errorf("model facing = %v", result.ModelFacing)
	}
	if result.UI["path"] != "a.ts" {
		t.Errorf("ui = %v", result.UI)
	}
}

func TestRegistryArgsAwareProjectionAndContentOnlyPostPolicy(t *testing.T) {
	registry := New(Options{})
	if err := registry.Register(&Definition{
		Name: "project", InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`), OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			return map[string]any{"ok": true}, nil
		},
		RenderModel: func(args json.RawMessage, _ any) (any, error) {
			var input struct {
				Path string `json:"path"`
			}
			if err := json.Unmarshal(args, &input); err != nil {
				return nil, err
			}
			return "rendered:" + input.Path, nil
		},
		PresentUI: func(args json.RawMessage, _ any) (map[string]any, error) {
			var input struct {
				Path string `json:"path"`
			}
			if err := json.Unmarshal(args, &input); err != nil {
				return nil, err
			}
			return map[string]any{"path": input.Path}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	registry.AddPostPolicy(func(context.Context, Execution, *Result) (PolicyDecision, any, string, error) {
		return PolicyAllow, ReplacePostPolicyContent("policy-content"), "", nil
	})
	result, err := registry.Run(context.Background(), ExecContext{}, "project", "call-1", []byte(`{"path":"a.ts"}`))
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelFacing != "policy-content" || result.Canonical.(map[string]any)["ok"] != true || result.UI["path"] != "a.ts" || result.Content[0].Text != "policy-content" {
		t.Fatalf("projection replacement changed the wrong fields: %+v", result)
	}
}

func TestResultFreezeDeepCopiesTypedNestedValues(t *testing.T) {
	type child struct {
		Values []string
	}
	type output struct {
		Labels map[string]string
		Child  *child
	}
	original := output{Labels: map[string]string{"status": "ok"}, Child: &child{Values: []string{"one"}}}
	result := (&Result{Canonical: original, ModelFacing: map[string]any{"nested": []any{map[string]string{"value": "ok"}}}}).Freeze()
	copyValue := result.Canonical.(output)
	copyValue.Labels["status"] = "changed"
	copyValue.Child.Values[0] = "changed"
	model := result.ModelFacing.(map[string]any)
	model["nested"].([]any)[0].(map[string]string)["value"] = "changed"
	if original.Labels["status"] != "ok" || original.Child.Values[0] != "one" {
		t.Fatalf("typed canonical value was aliased: %+v", original)
	}
	if result.ModelFacing.(map[string]any)["nested"].([]any)[0].(map[string]string)["value"] != "changed" {
		t.Fatal("frozen model value was not independently mutable")
	}
}

func TestRunBatchCommitsModelOrderAcrossParallelCalls(t *testing.T) {
	registry := New(Options{MaxParallel: 2})
	started := make(chan string, 2)
	release := make(chan struct{})
	if err := registry.Register(&Definition{Name: "read", Description: "read", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, ConcurrencySafe: true,
		Execute: func(_ context.Context, _ ExecContext, input json.RawMessage) (any, error) {
			started <- string(input)
			<-release
			return string(input), nil
		}}); err != nil {
		t.Fatal(err)
	}
	completed := make(chan []Outcome, 1)
	go func() {
		completed <- registry.RunBatch(context.Background(), ExecContext{}, []Call{{Name: "read", CallID: "first", Input: []byte(`{"n":1}`)}, {Name: "read", CallID: "second", Input: []byte(`{"n":2}`)}})
	}()
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("parallel calls did not both start")
		}
	}
	close(release)
	results := <-completed
	if results[0].Call.CallID != "first" || results[1].Call.CallID != "second" || results[0].Err != nil || results[1].Err != nil {
		t.Fatalf("outcomes = %#v", results)
	}
}

func TestZeroMaxParallelMeansUnlimited(t *testing.T) {
	registry := New(Options{MaxParallel: 0})
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	registry.MustRegister(&Definition{
		Name: "parallel", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, ConcurrencySafe: true,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			started <- struct{}{}
			<-release
			return "ok", nil
		},
	})
	done := make(chan []Outcome, 1)
	go func() {
		done <- registry.RunBatch(context.Background(), ExecContext{}, []Call{{Name: "parallel", CallID: "one", Input: []byte(`{}`)}, {Name: "parallel", CallID: "two", Input: []byte(`{}`)}})
	}()
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("unlimited parallel calls did not overlap")
		}
	}
	close(release)
	if outcomes := <-done; len(outcomes) != 2 || outcomes[0].Err != nil || outcomes[1].Err != nil {
		t.Fatalf("outcomes=%#v", outcomes)
	}
}

func TestRegistryModelToolsStripsRuntime(t *testing.T) {
	registry := New(Options{})
	err := registry.Register(&Definition{
		Name: "greet", Description: "greet",
		InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return "hi", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := registry.ModelTools()
	if len(tools) != 1 {
		t.Fatalf("tools = %d", len(tools))
	}
	if tools[0].Description != "greet" {
		t.Errorf("description = %q", tools[0].Description)
	}
	// ModelTools must not include runtime fields.
	if tools[0].Strict {
		t.Error("strict should default false")
	}
}

func TestRegistryNotFoundAndTimeout(t *testing.T) {
	registry := New(Options{})
	if _, err := registry.Run(context.Background(), ExecContext{}, "nope", "", nil); !errors.Is(err, ErrToolNotFound) {
		t.Errorf("expected ErrToolNotFound, got %v", err)
	}
}

func TestRegistryRequiresOutputSchema(t *testing.T) {
	registry := New(Options{})
	if err := registry.Register(&Definition{
		Name: "missing_output", InputSchema: OrObjectSchema, OutputSchema: json.RawMessage(" \n null "),
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return nil, nil },
	}); !errors.Is(err, ErrInvalidArguments) {
		t.Fatalf("missing output schema error=%v", err)
	}
	if err := registry.Register(&Definition{
		Name: "missing_input", InputSchema: json.RawMessage(" \n null "), OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return nil, nil },
	}); !errors.Is(err, ErrInvalidArguments) {
		t.Fatalf("missing input schema error=%v", err)
	}
}

func TestRegistryTimeoutCoversPipelineAndFinalizer(t *testing.T) {
	registry := New(Options{DefaultTimeout: 10 * time.Millisecond})
	registry.AddPreExecute(func(ctx context.Context, _ string, _ json.RawMessage) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if err := registry.Register(&Definition{
		Name: "slow_pipeline", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "ok", nil },
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Run(context.Background(), ExecContext{}, "slow_pipeline", "c1", []byte(`{}`)); !errors.Is(err, ErrToolTimeout) {
		t.Fatalf("pre-hook timeout=%v", err)
	}

	finalizing := New(Options{DefaultTimeout: 10 * time.Millisecond})
	if err := finalizing.Register(&Definition{
		Name: "slow_finalizer", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute:         func(context.Context, ExecContext, json.RawMessage) (any, error) { return "ok", nil },
		FinalizeContent: func(*Result) error { time.Sleep(25 * time.Millisecond); return nil },
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := finalizing.Run(context.Background(), ExecContext{}, "slow_finalizer", "c2", []byte(`{}`)); !errors.Is(err, ErrToolTimeout) {
		t.Fatalf("finalizer timeout=%v", err)
	}
}

func TestValidateInput(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	if err := ValidateInput(schema, []byte(`{"a":1}`)); err != nil {
		t.Errorf("valid input rejected: %v", err)
	}
	// not an object with object schema
	if err := ValidateInput(schema, []byte(`[1,2]`)); err == nil {
		t.Errorf("expected object requirement to reject array")
	}
	if err := ValidateInput(schema, []byte(`not json`)); err == nil {
		t.Errorf("expected invalid json rejection")
	}
}

func TestValidateInputRejectsUnsupportedConstraints(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string","maxLength":3}}}`)
	if err := ValidateSchema(schema); !errors.Is(err, ErrUnsupportedSchema) {
		t.Fatalf("constraint schema error = %v, want ErrUnsupportedSchema", err)
	}
}

func TestValidateSchemaMatchesDSHSubset(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"mode":{"type":"string","enum":["safe","fast"],"const":"safe"}},"required":["mode"],"additionalProperties":false}`)
	if err := ValidateInput(schema, []byte(`{"mode":"safe"}`)); err != nil {
		t.Fatalf("valid const input rejected: %v", err)
	}
	if err := ValidateInput(schema, []byte(`{"mode":"fast"}`)); err == nil {
		t.Fatal("const mismatch accepted")
	}
	invalid := []json.RawMessage{
		json.RawMessage(`{"oneOf":[{"type":"string"}]}`),
		json.RawMessage(`{"type":"string","oneOf":[{"type":"string"},{"type":"number"}]}`),
		json.RawMessage(`{"properties":{"name":{"type":"string"}}}`),
		json.RawMessage(`{"type":"string","minimum":1}`),
		json.RawMessage(`{"const":"safe"}`),
	}
	for _, candidate := range invalid {
		if err := ValidateSchema(candidate); err == nil {
			t.Fatalf("unsupported schema accepted: %s", candidate)
		}
	}
}

func TestFromStruct(t *testing.T) {
	type args struct {
		Path string `json:"path"`
	}
	schema, err := FromStruct[args]()
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(schema) {
		t.Error("schema not valid JSON")
	}
}

func TestRegistryRejectsInvalidInputBeforeExec(t *testing.T) {
	registry := New(Options{})
	schema, _ := RawSchema(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
		"required": []any{"path"},
	})
	executed := false
	err := registry.Register(&Definition{
		Name: "edit", Description: "edit",
		InputSchema: schema, OutputSchema: AnyOutputSchema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) {
			executed = true
			return "ok", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Missing required field must not reach the executor.
	if _, err := registry.Run(context.Background(), ExecContext{}, "edit", "c1", []byte(`{}`)); !errors.Is(err, ErrInvalidArguments) {
		t.Errorf("expected ErrInvalidArguments, got %v", err)
	}
	if executed {
		t.Error("executor ran despite invalid arguments")
	}
	// Valid input reaches the executor.
	if _, err := registry.Run(context.Background(), ExecContext{}, "edit", "c2", []byte(`{"path":"a.ts"}`)); err != nil {
		t.Errorf("valid run failed: %v", err)
	}
	if !executed {
		t.Error("executor did not run for valid input")
	}
}

func TestRegistryValidatesOutputSchema(t *testing.T) {
	registry := New(Options{})
	schema, _ := RawSchema(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"ok": map[string]any{"type": "boolean"},
		},
		"required": []any{"ok"},
	})
	err := registry.Register(&Definition{
		Name: "probe", Description: "probe",
		InputSchema:  OrObjectSchema,
		OutputSchema: schema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) {
			return map[string]any{"ok": "not-a-bool"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Run(context.Background(), ExecContext{}, "probe", "c1", []byte(`{}`)); !errors.Is(err, ErrInvalidOutput) {
		t.Errorf("expected ErrInvalidOutput, got %v", err)
	}
}

func TestModelToolsDeterministicOrder(t *testing.T) {
	registry := New(Options{})
	for _, name := range []string{"zeta", "alpha", "mike"} {
		def := &Definition{Name: name, Description: name, InputSchema: OrObjectSchema,
			OutputSchema: AnyOutputSchema,
			Execute:      func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return nil, nil }}
		if err := registry.Register(def); err != nil {
			t.Fatal(err)
		}
	}
	first := registry.ModelTools()
	names := make([]string, len(first))
	for i, td := range first {
		names[i] = td.Name
	}
	if names[0] != "alpha" || names[1] != "mike" || names[2] != "zeta" {
		t.Errorf("unexpected order: %v", names)
	}
	// Repeat calls are identical.
	second := registry.ModelTools()
	for i := range first {
		if first[i].Name != second[i].Name {
			t.Errorf("non-deterministic at %d", i)
		}
	}
	if got := registry.Names(); got[0] != "alpha" || got[2] != "zeta" {
		t.Errorf("Names not sorted: %v", got)
	}
}

func TestRegistryValidatesOneOfSchema(t *testing.T) {
	registry := New(Options{})
	def := &Definition{Name: "bad", Description: "bad",
		InputSchema:  json.RawMessage(`{"oneOf":[{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false},{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}]}`),
		OutputSchema: AnyOutputSchema,
		Execute:      func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return nil, nil }}
	if err := registry.Register(def); err != nil {
		t.Fatalf("register oneOf: %v", err)
	}
	if _, err := registry.Run(context.Background(), ExecContext{}, "bad", "one", []byte(`{"path":"a"}`)); err != nil {
		t.Fatalf("valid oneOf: %v", err)
	}
	if _, err := registry.Run(context.Background(), ExecContext{}, "bad", "two", []byte(`{"path":"a","id":1}`)); !errors.Is(err, ErrInvalidArguments) {
		t.Fatalf("ambiguous oneOf error = %v", err)
	}
}

func TestRegistryRejectsInvalidName(t *testing.T) {
	registry := New(Options{})
	def := &Definition{Name: "Bad Name!", Description: "x", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return nil, nil }}
	if err := registry.Register(def); err == nil {
		t.Error("expected invalid tool name to be rejected")
	}
}

type testApproval struct{ approved bool }

func (a testApproval) Approve(context.Context, ApprovalRequest) (bool, error) { return a.approved, nil }

func TestRegistryPolicyGuardApprovalAndFinalizer(t *testing.T) {
	registry := New(Options{Approval: testApproval{approved: true}})
	executed := false
	if err := registry.Register(&Definition{
		Name: "protected", Description: "protected", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			executed = true
			return map[string]any{"ok": true}, nil
		},
		FinalizeContent: func(result *Result) error {
			result.Name = "tampered"
			result.CallID = "tampered"
			result.Kind = OutcomeFailure
			result.Canonical = map[string]any{"ok": false}
			result.Code = "tampered"
			result.Meta = map[string]any{"tampered": true}
			result.Continuation = ToolDeferred
			result.ResumeKey = "tampered-resume"
			result.WaitingReason = "tampered-wait"
			result.ConcludesTurn = true
			result.ModelFacing = "finalized"
			return nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	registry.AddPolicy(func(context.Context, Execution) (PolicyDecision, string, error) {
		return PolicyAsk, "needs approval", nil
	})
	registry.AddGuard(func(Execution) (string, error) { return "", nil })
	result, err := registry.Run(context.Background(), ExecContext{}, "protected", "c1", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	canonical, _ := result.Canonical.(map[string]any)
	if !executed || result.Name != "protected" || result.CallID != "c1" || result.Kind != OutcomeSuccess || result.Code != "" || canonical["ok"] != true || result.ModelFacing != "finalized" || result.Meta != nil || result.Continuation != ToolContinue || result.ResumeKey != "" || result.WaitingReason != "" || result.ConcludesTurn {
		t.Fatalf("executed=%v result=%+v", executed, result)
	}

	denied := New(Options{})
	if err := denied.Register(&Definition{Name: "protected", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return nil, nil }}); err != nil {
		t.Fatal(err)
	}
	denied.AddPolicy(func(context.Context, Execution) (PolicyDecision, string, error) {
		return PolicyAsk, "approval unavailable", nil
	})
	if _, err := denied.Run(context.Background(), ExecContext{}, "protected", "c2", []byte(`{}`)); !errors.Is(err, ErrPolicyDenied) {
		t.Fatalf("fail-closed approval error=%v", err)
	}
}

func TestPostExecuteReceivesFailureOutcome(t *testing.T) {
	registry := New(Options{})
	executionErr := errors.New("execution failed")
	seen := false
	registry.AddPostExecute(func(_ context.Context, _ Execution, result *Result) error {
		seen = result.Kind == OutcomeFailure
		result.ModelFacing = "partial"
		return nil
	})
	registry.MustRegister(&Definition{
		Name: "fails", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return nil, executionErr },
	})
	result, err := registry.Run(context.Background(), ExecContext{}, "fails", "failure-hook", []byte(`{}`))
	if !errors.Is(err, executionErr) || !seen || result == nil || result.ModelFacing != "partial" {
		t.Fatalf("err=%v seen=%v result=%+v", err, seen, result)
	}
}

func TestBeforeExecuteShortCircuitPreservesDefinition(t *testing.T) {
	registry := New(Options{})
	executed, dispatched, finalized, observed := false, false, false, false
	definition := &Definition{
		Name: "workspace_read", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			executed = true
			return map[string]any{"ok": true}, nil
		},
		FinalizeContent: func(*Result) error { finalized = true; return nil },
	}
	inputSchema := append(json.RawMessage(nil), definition.InputSchema...)
	outputSchema := append(json.RawMessage(nil), definition.OutputSchema...)
	if err := registry.Register(definition); err != nil {
		t.Fatal(err)
	}
	registry.AddBeforeExecute(func(_ context.Context, ec ExecContext, execution Execution) (BeforeExecuteDecision, *Result, error) {
		if ec.CallID != "call-1" || execution.CallID != "call-1" {
			t.Fatalf("call identity not propagated: ec=%+v execution=%+v", ec, execution)
		}
		return BeforeExecuteShortCircuit, NewDeferredResult(
			execution.Name, execution.CallID, "Workspace is activating", map[string]any{"status": "waiting"}, "resume-1", "workspace_activation",
		), nil
	})
	registry.AddObserver(func(_ context.Context, result *Result) { observed = result.Frozen })
	result, err := registry.Run(context.Background(), ExecContext{OnDispatch: func(context.Context, string, string) error { dispatched = true; return nil }}, "workspace_read", "call-1", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if executed || dispatched || finalized || !observed {
		t.Fatalf("short-circuit side effects executed=%v dispatched=%v finalized=%v observed=%v", executed, dispatched, finalized, observed)
	}
	if result.Name != "workspace_read" || result.CallID != "call-1" || result.Continuation != ToolDeferred || result.ResumeKey != "resume-1" || result.Canonical != nil || !result.Frozen {
		t.Fatalf("unexpected deferred result: %+v", result)
	}
	registered, ok := registry.Get("workspace_read")
	if !ok || string(registered.InputSchema) != string(inputSchema) || string(registered.OutputSchema) != string(outputSchema) {
		t.Fatalf("definition schemas changed: %+v", registered)
	}
}

func TestBeforeExecuteContinueRunsDefinitionAndScopeCopiesHook(t *testing.T) {
	root := New(Options{})
	before := 0
	root.AddBeforeExecute(func(_ context.Context, _ ExecContext, _ Execution) (BeforeExecuteDecision, *Result, error) {
		before++
		return BeforeExecuteContinue, nil, nil
	})
	root.MustRegister(&Definition{
		Name: "root_tool", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "root", nil },
	})
	scope := root.NewScope()
	if err := scope.Register(&Definition{
		Name: "local_tool", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "local", nil },
	}); err != nil {
		t.Fatal(err)
	}
	result, err := scope.Run(context.Background(), ExecContext{}, "local_tool", "call-2", []byte(`{}`))
	if err != nil || result.Canonical != "local" || before != 1 {
		t.Fatalf("scope before hook result=%+v err=%v before=%d", result, err, before)
	}
}

func TestCancellationBeforeDispatchIsClassified(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	registry := New(Options{})
	registry.AddPreExecute(func(context.Context, string, json.RawMessage) error {
		cancel()
		return ctx.Err()
	})
	registry.MustRegister(&Definition{
		Name: "cancel", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "unexpected", nil },
	})
	if _, err := registry.Run(ctx, ExecContext{}, "cancel", "cancel-before-dispatch", []byte(`{}`)); !errors.Is(err, ErrAbortedBeforeDispatch) {
		t.Fatalf("error=%v, want ErrAbortedBeforeDispatch", err)
	}
}

func TestDefineToolGeneratesTypedSchemas(t *testing.T) {
	type input struct {
		Name string `json:"name" jsonschema:"required"`
	}
	type output struct {
		Count int `json:"count" jsonschema:"required"`
	}
	definition := DefineTool[input, output](DefineToolOptions[input, output]{
		Name: "typed", Execute: func(context.Context, ExecContext, input) (output, error) {
			return output{Count: 1}, nil
		},
	})
	if err := ValidateInput(definition.InputSchema, []byte(`{"name":"ok"}`)); err != nil {
		t.Fatalf("generated input schema rejected valid input: %v", err)
	}
	if err := ValidateInput(definition.OutputSchema, []byte(`{"count":1}`)); err != nil {
		t.Fatalf("generated output schema rejected valid output: %v", err)
	}
	if err := ValidateInput(definition.OutputSchema, []byte(`{"count":"wrong"}`)); err == nil {
		t.Fatal("generated output schema accepted invalid output")
	}
}

func TestDefineToolRejectsSchemaDrift(t *testing.T) {
	type input struct {
		Name string `json:"name" jsonschema:"required"`
	}
	definition := DefineTool[input, string](DefineToolOptions[input, string]{
		Name:        "typed_drift",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"other":{"type":"string"}}}`),
		Execute:     func(context.Context, ExecContext, input) (string, error) { return "ok", nil },
	})
	if err := (New(Options{})).Register(definition); !errors.Is(err, ErrInvalidArguments) {
		t.Fatalf("Register error=%v, want typed schema mismatch", err)
	}
}

func TestDefineToolRejectsOutputSchemaDrift(t *testing.T) {
	type input struct {
		Name string `json:"name" jsonschema:"required"`
	}
	type output struct {
		Count int `json:"count" jsonschema:"required"`
	}
	definition := DefineTool[input, output](DefineToolOptions[input, output]{
		Name:         "typed_output_drift",
		OutputSchema: json.RawMessage(`{"type":"object","properties":{"other":{"type":"string"}}}`),
		Execute:      func(context.Context, ExecContext, input) (output, error) { return output{Count: 1}, nil },
	})
	if err := (New(Options{})).Register(definition); !errors.Is(err, ErrInvalidArguments) {
		t.Fatalf("Register error=%v, want typed output schema mismatch", err)
	}
}

func TestFinalizerDoesNotReplaceExecutionError(t *testing.T) {
	registry := New(Options{})
	executionErr := errors.New("execution failed")
	if err := registry.Register(&Definition{
		Name: "fails", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			return nil, executionErr
		},
		FinalizeContent: func(result *Result) error {
			result.ModelFacing = "partial"
			return errors.New("finalizer failed")
		},
	}); err != nil {
		t.Fatal(err)
	}
	result, err := registry.Run(context.Background(), ExecContext{}, "fails", "c1", []byte(`{}`))
	if !errors.Is(err, executionErr) {
		t.Fatalf("execution error=%v", err)
	}
	if result == nil || result.Code != "TOOL_FAILED" || result.Kind != OutcomeFailure || result.ModelFacing != "partial" {
		t.Fatalf("finalizer replaced execution failure: %+v", result)
	}
}

func TestStructuredFailurePreservesHostCodeAndMetadata(t *testing.T) {
	registry := New(Options{})
	if err := registry.Register(&Definition{
		Name: "structured_failure", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			return nil, NewFailureError("workspace_unavailable", "workspace is not ready", map[string]any{"retryable": true, "retry_hint": "wait"})
		},
	}); err != nil {
		t.Fatal(err)
	}
	result, err := registry.Run(context.Background(), ExecContext{}, "structured_failure", "c1", []byte(`{}`))
	if err == nil || result == nil || result.Failure == nil {
		t.Fatalf("structured failure result=%+v err=%v", result, err)
	}
	if result.Code != "workspace_unavailable" || result.Failure.Code != "workspace_unavailable" || result.Failure.Message != "workspace is not ready" {
		t.Fatalf("structured failure=%+v", result.Failure)
	}
	if result.Failure.Meta["retryable"] != true || result.Failure.Meta["retry_hint"] != "wait" {
		t.Fatalf("structured failure metadata=%+v", result.Failure.Meta)
	}
}

func TestScopedRegistryShadowsAndRestricts(t *testing.T) {
	root := New(Options{})
	if err := root.Register(&Definition{Name: "read", Description: "root", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "root", nil }}); err != nil {
		t.Fatal(err)
	}
	scope := root.NewScope()
	if err := scope.Register(&Definition{Name: "read", Description: "local", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "local", nil }}); err != nil {
		t.Fatal(err)
	}
	if err := scope.Restrict([]string{"read"}); err != nil {
		t.Fatal(err)
	}
	tools := scope.ModelTools()
	if len(tools) != 1 || tools[0].Description != "local" {
		t.Fatalf("scoped tools=%+v", tools)
	}
	result, err := scope.Run(context.Background(), ExecContext{}, "read", "c1", []byte(`{}`))
	if err != nil || result.Canonical != "local" {
		t.Fatalf("scoped result=%+v err=%v", result, err)
	}
	if _, err := scope.Run(context.Background(), ExecContext{}, "write", "c2", []byte(`{}`)); !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("restricted tool error=%v", err)
	}
}

func TestScopedRegistryKeepsInheritedCatalogLiveAndLocalToolsUnrestricted(t *testing.T) {
	root := New(Options{})
	register := func(name string) {
		if err := root.Register(&Definition{Name: name, InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return name, nil }}); err != nil {
			t.Fatal(err)
		}
	}
	register("read")
	scope := root.NewScope()
	scope.Deny("read")
	if err := scope.Register(&Definition{Name: "local", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "local", nil }}); err != nil {
		t.Fatal(err)
	}
	register("late")
	tools := scope.ModelTools()
	names := make(map[string]bool, len(tools))
	for _, tool := range tools {
		names[tool.Name] = true
	}
	if names["read"] || !names["late"] || !names["local"] {
		t.Fatalf("scope visibility = %v, want late and local but not denied read", names)
	}
	if _, err := scope.Run(context.Background(), ExecContext{}, "local", "local-1", []byte(`{}`)); err != nil {
		t.Fatalf("local tool was incorrectly filtered: %v", err)
	}
}

type testCodeRuntime struct{}

func (testCodeRuntime) ExecuteCode(context.Context, ExecContext, string, string) (any, error) {
	return "ok", nil
}

func TestScopedRegistryKeepsCodeModeTransportVisible(t *testing.T) {
	root := New(Options{Presentation: PresentationBoth, CodeRuntime: testCodeRuntime{}})
	scope := root.NewScope()
	scope.Deny("run_code")

	found := false
	for _, tool := range scope.ModelTools() {
		found = found || tool.Name == "run_code"
	}
	if !found {
		t.Fatal("scope restriction hid the reserved run_code transport")
	}
	if _, err := scope.Run(context.Background(), ExecContext{}, "run_code", "code-1", []byte(`{"code":"return 1"}`)); err != nil {
		t.Fatalf("reserved run_code transport was denied: %v", err)
	}
}

func TestScopedRegistryRunsRootHooksForLocalTools(t *testing.T) {
	root := New(Options{})
	pre, post := 0, 0
	root.AddPreExecute(func(context.Context, string, json.RawMessage) error { pre++; return nil })
	root.AddPostExecute(func(context.Context, Execution, *Result) error { post++; return nil })
	scope := root.NewScope()
	if err := scope.Register(&Definition{Name: "local", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "ok", nil }}); err != nil {
		t.Fatal(err)
	}
	if _, err := scope.Run(context.Background(), ExecContext{}, "local", "c1", []byte(`{}`)); err != nil {
		t.Fatal(err)
	}
	if pre != 1 || post != 1 {
		t.Fatalf("root hooks pre=%d post=%d", pre, post)
	}
}

func TestReversibleToolRegistration(t *testing.T) {
	registry := New(Options{})
	handle, err := registry.RegisterTool(&Definition{
		Name: "temporary", InputSchema: OrObjectSchema, OutputSchema: AnyOutputSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "ok", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := handle.Unregister(); err != nil {
		t.Fatal(err)
	}
	if err := handle.Unregister(); err != nil {
		t.Fatalf("idempotent unregister=%v", err)
	}
	if _, err := registry.Run(context.Background(), ExecContext{}, "temporary", "c1", []byte(`{}`)); !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("removed tool error=%v", err)
	}
}
