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
