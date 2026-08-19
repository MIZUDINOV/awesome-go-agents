package tools

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
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

func TestRegistryRejectsUnsupportedSchema(t *testing.T) {
	registry := New(Options{})
	def := &Definition{Name: "bad", Description: "bad",
		InputSchema: json.RawMessage(`{"type":"object","oneOf":[{"type":"string"}]}`),
		Execute:     func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) { return nil, nil }}
	if err := registry.Register(def); err == nil {
		t.Error("expected unsupported schema to be rejected at registration")
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
