package tools

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/invopop/jsonschema"
)

// FromStruct generates a JSON Schema for a Go type T using its jsonschema
// struct tags. The returned bytes are a JSON Schema object suitable for
// ToolDefinition.InputSchema.
func FromStruct[T any]() (json.RawMessage, error) {
	reflector := &jsonschema.Reflector{DoNotReference: true}
	schema := reflector.Reflect(new(T))
	encoded, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	return normalizeGeneratedSchema(encoded)
}

func normalizeGeneratedSchema(document json.RawMessage) (json.RawMessage, error) {
	var node any
	if err := json.Unmarshal(document, &node); err != nil {
		return nil, err
	}
	normalized, err := normalizeGeneratedNode(node)
	if err != nil {
		return nil, err
	}
	return json.Marshal(normalized)
}

func normalizeGeneratedNode(node any) (any, error) {
	obj, ok := node.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("generated schema node must be an object")
	}
	out := make(map[string]any, len(obj))
	for key, value := range obj {
		switch key {
		case "$schema", "$id", "$comment", "format",
			"minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
			"minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties":
			// The enforced wire subset has no format or scalar constraint
			// vocabulary. Go decoding remains the typed runtime boundary.
		case "type", "required", "additionalProperties", "enum", "const", "description", "title", "default", "examples":
			out[key] = value
		case "properties":
			properties, ok := value.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("generated properties must be an object")
			}
			normalized := make(map[string]any, len(properties))
			for name, property := range properties {
				child, err := normalizeGeneratedNode(property)
				if err != nil {
					return nil, err
				}
				normalized[name] = child
			}
			out[key] = normalized
		case "items":
			child, err := normalizeGeneratedNode(value)
			if err != nil {
				return nil, err
			}
			out[key] = child
		case "oneOf":
			branches, ok := value.([]any)
			if !ok {
				return nil, fmt.Errorf("generated oneOf must be an array")
			}
			normalized := make([]any, len(branches))
			for index, branch := range branches {
				child, err := normalizeGeneratedNode(branch)
				if err != nil {
					return nil, err
				}
				normalized[index] = child
			}
			out[key] = normalized
		case "anyOf":
			branches, ok := value.([]any)
			if !ok || len(branches) < 2 {
				return nil, fmt.Errorf("generated anyOf cannot be represented as oneOf")
			}
			normalized := make([]any, len(branches))
			for index, branch := range branches {
				child, err := normalizeGeneratedNode(branch)
				if err != nil {
					return nil, err
				}
				normalized[index] = child
			}
			out["oneOf"] = normalized
		default:
			return nil, fmt.Errorf("generated schema keyword %q is outside the AgentKit subset", key)
		}
	}
	return out, nil
}

// RawSchema wraps a literal JSON Schema document.
func RawSchema(document any) (json.RawMessage, error) {
	return json.Marshal(document)
}

// OrObjectSchema returns a JSON Schema that accepts any object.
var OrObjectSchema = json.RawMessage(`{"type":"object"}`)

// AnyOutputSchema is the explicit lossless fallback for tools whose runtime
// output is intentionally unconstrained.
var AnyOutputSchema = json.RawMessage(`{}`)

func schemaAbsent(schema []byte) bool {
	trimmed := bytes.TrimSpace(schema)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

// MergeSchemas returns the first non-empty schema. Most tools provide a single
// input schema; kept for API stability.
func MergeSchemas(schemas ...json.RawMessage) json.RawMessage {
	for _, s := range schemas {
		if !schemaAbsent(s) {
			return s
		}
	}
	return OrObjectSchema
}
