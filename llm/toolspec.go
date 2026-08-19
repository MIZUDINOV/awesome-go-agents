package llm

import "encoding/json"

// ToolDefinition is the model-facing surface of a tool: exactly what the model
// needs to decide whether and how to call it. Runtime concerns (executor,
// renderers, concurrency, timeout) live in agentkit/tools, never here.
type ToolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// InputSchema is the JSON Schema (object type) for the tool arguments.
	InputSchema json.RawMessage `json:"input_schema"`
	// Strict requests strict JSON Schema adherence when the provider supports
	// it (OpenRouter passes it through as tool.function.strict).
	Strict bool `json:"strict,omitempty"`
}

// MarshalJSON ensures the raw schema is embedded verbatim.
func (t *ToolDefinition) MarshalJSON() ([]byte, error) {
	type alias ToolDefinition
	return json.Marshal((*alias)(t))
}
