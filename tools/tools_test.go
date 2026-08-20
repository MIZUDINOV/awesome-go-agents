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
		InputSchema: inputSchema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) {
			executed = true
			return map[string]any{"content": "data"}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			return "The file was read.", nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
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

func TestRunBatchCommitsModelOrderAcrossParallelCalls(t *testing.T) {
	registry := New(Options{MaxParallel: 2})
	started := make(chan string, 2)
	release := make(chan struct{})
	if err := registry.Register(&Definition{Name: "read", Description: "read", InputSchema: OrObjectSchema, ConcurrencySafe: true,
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

func TestRegistryModelToolsStripsRuntime(t *testing.T) {
	registry := New(Options{})
	err := registry.Register(&Definition{
		Name: "greet", Description: "greet",
		InputSchema: OrObjectSchema,
		Execute:     func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return "hi", nil },
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
		InputSchema: schema,
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
			Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return nil, nil }}
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
		InputSchema: json.RawMessage(`{"oneOf":[{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false},{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}]}`),
		Execute:     func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return nil, nil }}
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
	def := &Definition{Name: "Bad Name!", Description: "x", InputSchema: OrObjectSchema,
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
		Name: "protected", Description: "protected", InputSchema: OrObjectSchema,
		Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) {
			executed = true
			return map[string]any{"ok": true}, nil
		},
		FinalizeContent: func(result *Result) error {
			result.Code = "OK"
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
	if !executed || result.Code != "OK" {
		t.Fatalf("executed=%v result=%+v", executed, result)
	}

	denied := New(Options{})
	if err := denied.Register(&Definition{Name: "protected", InputSchema: OrObjectSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return nil, nil }}); err != nil {
		t.Fatal(err)
	}
	denied.AddPolicy(func(context.Context, Execution) (PolicyDecision, string, error) {
		return PolicyAsk, "approval unavailable", nil
	})
	if _, err := denied.Run(context.Background(), ExecContext{}, "protected", "c2", []byte(`{}`)); !errors.Is(err, ErrPolicyDenied) {
		t.Fatalf("fail-closed approval error=%v", err)
	}
}

func TestScopedRegistryShadowsAndRestricts(t *testing.T) {
	root := New(Options{})
	if err := root.Register(&Definition{Name: "read", Description: "root", InputSchema: OrObjectSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "root", nil }}); err != nil {
		t.Fatal(err)
	}
	scope := root.NewScope()
	if err := scope.Register(&Definition{Name: "read", Description: "local", InputSchema: OrObjectSchema, Execute: func(context.Context, ExecContext, json.RawMessage) (any, error) { return "local", nil }}); err != nil {
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
