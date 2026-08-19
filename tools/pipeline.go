package tools

import (
	"encoding/json"
	"fmt"
)

// ValidateInput validates JSON arguments against a JSON Schema object using a
// minimal, dependency-free validator. It checks structural type conformance so
// the most common argument mistakes surface early. Tools with strict needs may
// plug a full JSON Schema validator into their own Execute instead.
func ValidateInput(schema json.RawMessage, input []byte) error {
	if len(schema) == 0 || string(schema) == "null" {
		return nil
	}
	var schemaType any
	if err := json.Unmarshal(schema, &schemaType); err != nil {
		return fmt.Errorf("%w: input schema is invalid JSON: %v", ErrInvalidArguments, err)
	}
	if !json.Valid(input) {
		return fmt.Errorf("%w: arguments are not valid JSON", ErrInvalidArguments)
	}
	var inputValue any
	if err := json.Unmarshal(input, &inputValue); err != nil {
		return fmt.Errorf("%w: decode arguments: %v", ErrInvalidArguments, err)
	}
	// Enforce top-level "object" requirement when the schema declares it and
	// the arguments are not an object.
	if doc, ok := schemaType.(map[string]any); ok {
		if declared, ok := doc["type"].(string); ok && declared == "object" {
			if _, ok := inputValue.(map[string]any); !ok {
				return fmt.Errorf("%w: expected an object argument", ErrInvalidArguments)
			}
		}
	}
	return nil
}
