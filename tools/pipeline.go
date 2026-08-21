package tools

import (
	"encoding/json"
	"fmt"
	"math"
)

// Supported schema subset this package validates. Constructs outside the
// subset are rejected at registration time by ValidateSchema (H-TOOLS-008), so
// a tool can never fail at runtime over a schema keyword it silently ignored.
//
//	type          string                (object/array/string/number/integer/boolean/null)
//	properties    map[string]Schema
//	required      []string
//	items         Schema
//	enum          []value
//	const         scalar value
//	additionalProperties bool           (false = closed object)
//
// The metadata keys are the only non-structural extension. oneOf is exact-one
// and cannot be combined with another structural schema keyword.
var supportedSchemaKeywords = map[string]bool{
	"type": true, "properties": true, "required": true,
	"items": true, "enum": true, "const": true, "additionalProperties": true,
	"oneOf": true, "title": true, "description": true, "default": true, "examples": true,
}

// ErrUnsupportedSchema identifies a schema construct outside the validated
// subset. It is returned by ValidateSchema at registration so the failure is
// explicit and early rather than silently ignored at runtime.
var ErrUnsupportedSchema = fmt.Errorf("unsupported JSON Schema construct")

// ErrSchemaInvalid identifies a malformed JSON Schema document.
var ErrSchemaInvalid = fmt.Errorf("invalid JSON Schema")

// ValidateSchema checks a JSON Schema document for the supported subset. It
// returns an error for malformed schemas (ErrSchemaInvalid) or constructs
// outside the subset (ErrUnsupportedSchema). Callers run this once at
// registration time.
func ValidateSchema(schema []byte) error {
	if schemaAbsent(schema) {
		return nil
	}
	var node any
	if err := json.Unmarshal(schema, &node); err != nil {
		return fmt.Errorf("%w: schema is not valid JSON: %v", ErrSchemaInvalid, err)
	}
	return validateSchemaNode(node)
}

func validateSchemaNode(node any) error {
	obj, ok := node.(map[string]any)
	if !ok {
		return fmt.Errorf("%w: schema must be an object", ErrSchemaInvalid)
	}
	for key := range obj {
		if !supportedSchemaKeywords[key] {
			return fmt.Errorf("%w: keyword %q is not supported by the agentkit schema subset", ErrUnsupportedSchema, key)
		}
	}
	typeName, hasType := "", false
	if raw, ok := obj["type"]; ok {
		isStr := false
		typeName, isStr = raw.(string)
		if !isStr {
			return fmt.Errorf("%w: type must be a string", ErrSchemaInvalid)
		}
		hasType = true
		switch typeName {
		case "object", "array", "string", "number", "integer", "boolean", "null":
		default:
			return fmt.Errorf("%w: unsupported type %q", ErrUnsupportedSchema, typeName)
		}
	}
	hasOneOf := false
	if raw, ok := obj["oneOf"]; ok {
		hasOneOf = true
		branches, valid := raw.([]any)
		if !valid || len(branches) < 2 {
			return fmt.Errorf("%w: oneOf must contain at least two schemas", ErrSchemaInvalid)
		}
		for _, branch := range branches {
			if err := validateSchemaNode(branch); err != nil {
				return err
			}
		}
	}
	if hasType && hasOneOf {
		return fmt.Errorf("%w: type and oneOf cannot be combined", ErrUnsupportedSchema)
	}
	if hasOneOf {
		for _, key := range []string{"properties", "required", "additionalProperties", "items", "enum", "const"} {
			if _, ok := obj[key]; ok {
				return fmt.Errorf("%w: oneOf cannot be combined with %s", ErrUnsupportedSchema, key)
			}
		}
	}
	if !hasType && !hasOneOf {
		for _, key := range []string{"properties", "required", "additionalProperties", "items", "enum", "const"} {
			if _, ok := obj[key]; ok {
				return fmt.Errorf("%w: %s requires type or oneOf", ErrUnsupportedSchema, key)
			}
		}
		return nil
	}
	for _, key := range []string{"title", "description"} {
		if raw, ok := obj[key]; ok {
			if _, valid := raw.(string); !valid {
				return fmt.Errorf("%w: %s must be a string", ErrSchemaInvalid, key)
			}
		}
	}
	if raw, ok := obj["additionalProperties"]; ok {
		if typeName != "object" {
			return fmt.Errorf("%w: additionalProperties requires type object", ErrSchemaInvalid)
		}
		if _, isBool := raw.(bool); !isBool {
			return fmt.Errorf("%w: additionalProperties must be a boolean", ErrSchemaInvalid)
		}
	}
	if raw, ok := obj["required"]; ok {
		if typeName != "object" {
			return fmt.Errorf("%w: required requires type object", ErrSchemaInvalid)
		}
		list, ok := raw.([]any)
		if !ok {
			return fmt.Errorf("%w: required must be an array of strings", ErrSchemaInvalid)
		}
		seen := make(map[string]bool, len(list))
		for _, item := range list {
			name, isStr := item.(string)
			if !isStr || seen[name] {
				return fmt.Errorf("%w: required entries must be strings", ErrSchemaInvalid)
			}
			seen[name] = true
		}
		if _, hasProperties := obj["properties"]; !hasProperties {
			return fmt.Errorf("%w: required names must be declared in properties", ErrSchemaInvalid)
		}
	}
	if raw, ok := obj["properties"]; ok {
		if typeName != "object" {
			return fmt.Errorf("%w: properties requires type object", ErrSchemaInvalid)
		}
		props, ok := raw.(map[string]any)
		if !ok {
			return fmt.Errorf("%w: properties must be an object", ErrSchemaInvalid)
		}
		for _, sub := range props {
			if err := validateSchemaNode(sub); err != nil {
				return err
			}
		}
		if required, ok := obj["required"].([]any); ok {
			for _, item := range required {
				name := item.(string)
				if _, declared := props[name]; !declared {
					return fmt.Errorf("%w: required property %q is not declared", ErrSchemaInvalid, name)
				}
			}
		}
	}
	if raw, ok := obj["items"]; ok {
		if typeName != "array" {
			return fmt.Errorf("%w: items requires type array", ErrSchemaInvalid)
		}
		if err := validateSchemaNode(raw); err != nil {
			return err
		}
	}
	for _, key := range []string{"enum", "const"} {
		if raw, ok := obj[key]; ok {
			if key == "enum" {
				values, valid := raw.([]any)
				if !valid || len(values) == 0 {
					return fmt.Errorf("%w: enum must be a non-empty array", ErrSchemaInvalid)
				}
				for _, value := range values {
					if !isScalar(value) || (hasType && !typeMatches(typeName, value)) {
						return fmt.Errorf("%w: enum values must be scalar and match type", ErrSchemaInvalid)
					}
				}
			} else if !isScalar(raw) || (hasType && !typeMatches(typeName, raw)) {
				return fmt.Errorf("%w: const must be scalar and match type", ErrSchemaInvalid)
			}
		}
	}
	if enum, ok := obj["enum"].([]any); ok {
		if constant, ok := obj["const"]; ok && !enumContains(enum, constant) {
			return fmt.Errorf("%w: const must be present in enum", ErrSchemaInvalid)
		}
	}
	return nil
}

// ValidateInput validates JSON arguments against a supported JSON Schema
// object, returning ErrInvalidArguments on any structural violation.
func ValidateInput(schema json.RawMessage, input []byte) error {
	if schemaAbsent(schema) {
		return nil
	}
	if err := ValidateSchema(schema); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidArguments, err)
	}
	if !json.Valid(input) {
		return fmt.Errorf("%w: arguments are not valid JSON", ErrInvalidArguments)
	}
	var schemaNode any
	if err := json.Unmarshal(schema, &schemaNode); err != nil {
		return fmt.Errorf("%w: input schema is invalid JSON: %v", ErrInvalidArguments, err)
	}
	var inputValue any
	if err := json.Unmarshal(input, &inputValue); err != nil {
		return fmt.Errorf("%w: decode arguments: %v", ErrInvalidArguments, err)
	}
	if schemaObj, ok := schemaNode.(map[string]any); ok {
		if err := validateNode(schemaObj, inputValue, "$"); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalidArguments, err)
		}
	}
	return nil
}

func validateNode(schema map[string]any, value any, path string) error {
	if raw, ok := schema["oneOf"].([]any); ok {
		matches := 0
		for _, branch := range raw {
			if candidate, ok := branch.(map[string]any); ok && validateNode(candidate, value, path) == nil {
				matches++
			}
		}
		if matches != 1 {
			return fmt.Errorf("%s: value must match exactly one oneOf branch (matched %d)", path, matches)
		}
	}
	if types, ok := schema["type"].(string); ok {
		if !typeMatches(types, value) {
			return fmt.Errorf("%s: expected type %q, got %q", path, types, jsonTypeOf(value))
		}
	}
	if enum, ok := schema["enum"].([]any); ok {
		if !enumContains(enum, value) {
			return fmt.Errorf("%s: value not in enum", path)
		}
	}
	if constant, ok := schema["const"]; ok && !jsonDeepEqual(constant, value) {
		return fmt.Errorf("%s: value does not equal const", path)
	}
	if obj, ok := value.(map[string]any); ok {
		if required, ok := schema["required"].([]any); ok {
			for _, name := range required {
				if name, ok := name.(string); ok {
					if _, present := obj[name]; !present {
						return fmt.Errorf("%s: missing required property %q", path, name)
					}
				}
			}
		}
		props, hasProperties := schema["properties"].(map[string]any)
		if hasProperties {
			for name, propSchema := range props {
				if v, present := obj[name]; present {
					if sub, ok := propSchema.(map[string]any); ok {
						if err := validateNode(sub, v, path+"."+name); err != nil {
							return err
						}
					}
				}
			}
			for name := range obj {
				if _, known := props[name]; !known {
					if closed, ok := schema["additionalProperties"].(bool); ok && closed {
						return fmt.Errorf("%s: unexpected property %q (closed object)", path, name)
					}
				}
			}
		} else if closed, ok := schema["additionalProperties"].(bool); ok && closed && len(obj) > 0 {
			return fmt.Errorf("%s: unexpected property in closed object", path)
		}
	}
	if arr, ok := value.([]any); ok {
		if items, ok := schema["items"].(map[string]any); ok {
			for i, v := range arr {
				if err := validateNode(items, v, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func typeMatches(declared string, value any) bool {
	switch declared {
	case "object":
		return jsonTypeOf(value) == "object"
	case "array":
		return jsonTypeOf(value) == "array"
	case "string":
		return jsonTypeOf(value) == "string"
	case "number":
		return jsonTypeOf(value) == "number"
	case "integer":
		if f, ok := value.(float64); ok {
			return !math.IsNaN(f) && !math.IsInf(f, 0) && f == math.Trunc(f)
		}
		return jsonTypeOf(value) == "integer"
	case "boolean":
		return jsonTypeOf(value) == "boolean"
	case "null":
		return value == nil
	}
	return true
}

func isScalar(value any) bool {
	switch value.(type) {
	case nil, bool, string, float64:
		return true
	default:
		return false
	}
}

func jsonTypeOf(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case string:
		return "string"
	case float64:
		return "number"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return "number"
	}
}

func enumContains(enum []any, value any) bool {
	for _, e := range enum {
		if jsonDeepEqual(e, value) {
			return true
		}
	}
	return false
}

func jsonDeepEqual(a, b any) bool {
	ab, errA := json.Marshal(a)
	bb, errB := json.Marshal(b)
	if errA != nil || errB != nil {
		return false
	}
	return string(ab) == string(bb)
}
