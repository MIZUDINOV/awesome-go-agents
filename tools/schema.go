package tools

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
)

// FromStruct generates a JSON Schema for a Go type T using its jsonschema
// struct tags. The returned bytes are a JSON Schema object suitable for
// ToolDefinition.InputSchema.
func FromStruct[T any]() (json.RawMessage, error) {
	reflector := &jsonschema.Reflector{DoNotReference: true}
	schema := reflector.Reflect(new(T))
	return json.Marshal(schema)
}

// RawSchema wraps a literal JSON Schema document.
func RawSchema(document any) (json.RawMessage, error) {
	return json.Marshal(document)
}

// OrObjectSchema returns a JSON Schema that accepts any object.
var OrObjectSchema = json.RawMessage(`{"type":"object"}`)

// MergeSchemas returns the first non-empty schema. Most tools provide a single
// input schema; kept for API stability.
func MergeSchemas(schemas ...json.RawMessage) json.RawMessage {
	for _, s := range schemas {
		if len(s) > 0 && string(s) != "null" {
			return s
		}
	}
	return OrObjectSchema
}
