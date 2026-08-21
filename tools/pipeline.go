package tools

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"unicode/utf8"
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
//	additionalProperties bool           (false = closed object)
//	string/number/array/object constraints (min/max and pattern)
//
// Metadata keys ($schema/$id/$comment) are accepted and ignored. The explicit
// DeepSeek-compatible subset supports exact-one oneOf.
var supportedSchemaKeywords = map[string]bool{
	"type": true, "properties": true, "required": true,
	"items": true, "enum": true, "additionalProperties": true,
	"oneOf":   true,
	"$schema": true, "$id": true, "$comment": true,
	"title": true, "description": true, "default": true, "examples": true,
	"format":    true,
	"minLength": true, "maxLength": true, "pattern": true,
	"minimum": true, "maximum": true, "exclusiveMinimum": true, "exclusiveMaximum": true, "multipleOf": true,
	"minItems": true, "maxItems": true, "uniqueItems": true,
	"minProperties": true, "maxProperties": true,
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
	if node == nil {
		return nil
	}
	obj, ok := node.(map[string]any)
	if !ok {
		return fmt.Errorf("%w: schema must be an object", ErrSchemaInvalid)
	}
	for key := range obj {
		if !supportedSchemaKeywords[key] {
			return fmt.Errorf("%w: keyword %q is not supported by the agentkit schema subset", ErrUnsupportedSchema, key)
		}
	}
	if raw, ok := obj["type"]; ok {
		typeName, isStr := raw.(string)
		if !isStr {
			return fmt.Errorf("%w: type must be a string", ErrSchemaInvalid)
		}
		switch typeName {
		case "object", "array", "string", "number", "integer", "boolean", "null":
		default:
			return fmt.Errorf("%w: unsupported type %q", ErrUnsupportedSchema, typeName)
		}
	}
	for _, key := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"} {
		if raw, ok := obj[key]; ok {
			number, valid := raw.(float64)
			if !valid || math.IsNaN(number) || math.IsInf(number, 0) {
				return fmt.Errorf("%w: %s must be a finite number", ErrSchemaInvalid, key)
			}
			if key == "multipleOf" && number <= 0 {
				return fmt.Errorf("%w: multipleOf must be greater than zero", ErrSchemaInvalid)
			}
		}
	}
	for _, key := range []string{"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"} {
		if raw, ok := obj[key]; ok {
			number, valid := raw.(float64)
			if !valid || number < 0 || number != math.Trunc(number) {
				return fmt.Errorf("%w: %s must be a non-negative integer", ErrSchemaInvalid, key)
			}
		}
	}
	if raw, ok := obj["pattern"]; ok {
		pattern, valid := raw.(string)
		if !valid {
			return fmt.Errorf("%w: pattern must be a string", ErrSchemaInvalid)
		}
		if _, err := regexp.Compile(pattern); err != nil {
			return fmt.Errorf("%w: pattern is invalid: %v", ErrSchemaInvalid, err)
		}
	}
	if raw, ok := obj["uniqueItems"]; ok {
		if _, valid := raw.(bool); !valid {
			return fmt.Errorf("%w: uniqueItems must be a boolean", ErrSchemaInvalid)
		}
	}
	if raw, ok := obj["additionalProperties"]; ok {
		if _, isBool := raw.(bool); !isBool {
			return fmt.Errorf("%w: additionalProperties must be a boolean", ErrUnsupportedSchema)
		}
	}
	if raw, ok := obj["required"]; ok {
		list, ok := raw.([]any)
		if !ok {
			return fmt.Errorf("%w: required must be an array of strings", ErrSchemaInvalid)
		}
		for _, item := range list {
			if _, isStr := item.(string); !isStr {
				return fmt.Errorf("%w: required entries must be strings", ErrSchemaInvalid)
			}
		}
	}
	if raw, ok := obj["properties"]; ok {
		props, ok := raw.(map[string]any)
		if !ok {
			return fmt.Errorf("%w: properties must be an object", ErrSchemaInvalid)
		}
		for _, sub := range props {
			if err := validateSchemaNode(sub); err != nil {
				return err
			}
		}
	}
	if raw, ok := obj["items"]; ok {
		if err := validateSchemaNode(raw); err != nil {
			return err
		}
	}
	if raw, ok := obj["oneOf"]; ok {
		branches, ok := raw.([]any)
		if !ok || len(branches) == 0 {
			return fmt.Errorf("%w: oneOf must be a non-empty array", ErrSchemaInvalid)
		}
		for _, branch := range branches {
			if err := validateSchemaNode(branch); err != nil {
				return err
			}
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
	if stringValue, ok := value.(string); ok {
		length := utf8.RuneCountInString(stringValue)
		if minimum, ok := schemaInteger(schema, "minLength"); ok && length < minimum {
			return fmt.Errorf("%s: string is shorter than minLength", path)
		}
		if maximum, ok := schemaInteger(schema, "maxLength"); ok && length > maximum {
			return fmt.Errorf("%s: string is longer than maxLength", path)
		}
		if pattern, ok := schema["pattern"].(string); ok {
			matched, err := regexp.MatchString(pattern, stringValue)
			if err == nil && !matched {
				return fmt.Errorf("%s: string does not match pattern", path)
			}
		}
	}
	if number, ok := value.(float64); ok {
		if minimum, ok := schemaNumber(schema, "minimum"); ok && number < minimum {
			return fmt.Errorf("%s: number is less than minimum", path)
		}
		if maximum, ok := schemaNumber(schema, "maximum"); ok && number > maximum {
			return fmt.Errorf("%s: number is greater than maximum", path)
		}
		if minimum, ok := schemaNumber(schema, "exclusiveMinimum"); ok && number <= minimum {
			return fmt.Errorf("%s: number is not greater than exclusiveMinimum", path)
		}
		if maximum, ok := schemaNumber(schema, "exclusiveMaximum"); ok && number >= maximum {
			return fmt.Errorf("%s: number is not less than exclusiveMaximum", path)
		}
		if multiple, ok := schemaNumber(schema, "multipleOf"); ok && math.Abs(number/multiple-math.Round(number/multiple)) > 1e-9 {
			return fmt.Errorf("%s: number is not a multiple of multipleOf", path)
		}
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
		if minimum, ok := schemaInteger(schema, "minProperties"); ok && len(obj) < minimum {
			return fmt.Errorf("%s: object has fewer properties than minProperties", path)
		}
		if maximum, ok := schemaInteger(schema, "maxProperties"); ok && len(obj) > maximum {
			return fmt.Errorf("%s: object has more properties than maxProperties", path)
		}
	}
	if arr, ok := value.([]any); ok {
		if minimum, ok := schemaInteger(schema, "minItems"); ok && len(arr) < minimum {
			return fmt.Errorf("%s: array has fewer items than minItems", path)
		}
		if maximum, ok := schemaInteger(schema, "maxItems"); ok && len(arr) > maximum {
			return fmt.Errorf("%s: array has more items than maxItems", path)
		}
		if unique, ok := schema["uniqueItems"].(bool); ok && unique {
			for i := range arr {
				for j := i + 1; j < len(arr); j++ {
					if jsonDeepEqual(arr[i], arr[j]) {
						return fmt.Errorf("%s: array items must be unique", path)
					}
				}
			}
		}
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

func schemaNumber(schema map[string]any, key string) (float64, bool) {
	value, ok := schema[key].(float64)
	return value, ok
}

func schemaInteger(schema map[string]any, key string) (int, bool) {
	value, ok := schemaNumber(schema, key)
	if !ok || value < 0 || value != math.Trunc(value) || value > float64(int(^uint(0)>>1)) {
		return 0, false
	}
	return int(value), true
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
			return f == float64(int64(f))
		}
		return jsonTypeOf(value) == "integer"
	case "boolean":
		return jsonTypeOf(value) == "boolean"
	case "null":
		return value == nil
	}
	return true
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
