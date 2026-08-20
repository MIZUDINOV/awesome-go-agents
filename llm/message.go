// Package llm defines the provider-agnostic conversation and generation
// contract. Providers (OpenRouter, OpenAI-compatible endpoints, ...) implement
// Provider and translate between these types and their wire format. The rest
// of agentkit never imports a provider SDK.
package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Role is the provider-neutral conversation role. Values deliberately mirror
// the OpenRouter/OpenAI chat vocabulary because that is the de-facto canonical
// wire format for most providers.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// PartType enumerates the content kinds a Part can carry. A message may
// contain multiple parts of different types (text + reasoning, text + tools).
type PartType string

const (
	PartText       PartType = "text"
	PartMedia      PartType = "media"
	PartToolCall   PartType = "tool_call"
	PartToolResult PartType = "tool_result"
	PartReasoning  PartType = "reasoning"
)

// MediaContent carries an inline or remote image/document reference.
type MediaContent struct {
	MediaType string `json:"media_type"`
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
}

// ToolCallRequest is a model-requested tool invocation. Arguments carry the
// exact JSON object the model produced (raw, so ordering and precision are
// preserved).
type ToolCallRequest struct {
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolCallResult is the tool execution outcome bound to a specific request
// through CallID.
type ToolCallResult struct {
	CallID string          `json:"call_id"`
	Name   string          `json:"name"`
	Output json.RawMessage `json:"output"`

	// IsError marks a semantic tool failure. It is still a valid tool result;
	// providers render it as a normal tool message with an error envelope.
	IsError bool `json:"is_error,omitempty"`
}

// Part is one unit of message content.
type Part struct {
	Type       PartType         `json:"type"`
	Text       string           `json:"text,omitempty"`
	Media      *MediaContent    `json:"media,omitempty"`
	ToolCall   *ToolCallRequest `json:"tool_call,omitempty"`
	ToolResult *ToolCallResult  `json:"tool_result,omitempty"`
	Reasoning  string           `json:"reasoning,omitempty"`

	// Custom carries provider-specific part enrichment (e.g. OpenRouter
	// file parts, annotations) as opaque JSON.
	Custom json.RawMessage `json:"custom,omitempty"`
}

// Message is a single conversation turn. Metadata is provider-neutral and
// persists reasoning details, annotations, or routing diagnostics without
// polluting the visible content.
type Message struct {
	Role     Role           `json:"role"`
	Parts    []Part         `json:"parts,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewTextMessage builds a simple text message for the given role.
func NewTextMessage(role Role, text string) *Message {
	return &Message{Role: role, Parts: []Part{{Type: PartText, Text: text}}}
}

// NewUserMessage is a convenience constructor for RoleUser.
func NewUserMessage(text string) *Message { return NewTextMessage(RoleUser, text) }

// NewAssistantMessage builds the canonical assistant reply used by
// surface projection: ordered text/reasoning parts followed by tool calls.
func NewAssistantMessage(text, reasoning string, calls []ToolCallRequest) *Message {
	parts := make([]Part, 0, 2+len(calls))
	if reasoning != "" {
		parts = append(parts, Part{Type: PartReasoning, Reasoning: reasoning})
	}
	if text != "" {
		parts = append(parts, Part{Type: PartText, Text: text})
	}
	for _, call := range calls {
		call := call
		parts = append(parts, Part{Type: PartToolCall, ToolCall: &call})
	}
	return &Message{Role: RoleAssistant, Parts: parts}
}

// NewToolResultMessage binds one or more tool outputs to RoleTool.
func NewToolResultMessage(results ...ToolCallResult) *Message {
	parts := make([]Part, 0, len(results))
	for _, result := range results {
		result := result
		parts = append(parts, Part{Type: PartToolResult, ToolResult: &result})
	}
	return &Message{Role: RoleTool, Parts: parts}
}

// Text concatenates all text parts in order.
func (m *Message) Text() string {
	if m == nil {
		return ""
	}
	var builder strings.Builder
	for _, part := range m.Parts {
		if part.Type == PartText {
			builder.WriteString(part.Text)
		}
	}
	return builder.String()
}

// Reasoning concatenates all reasoning parts in order.
func (m *Message) Reasoning() string {
	if m == nil {
		return ""
	}
	var builder strings.Builder
	for _, part := range m.Parts {
		if part.Type == PartReasoning {
			builder.WriteString(part.Reasoning)
		}
	}
	return builder.String()
}

// ToolCalls returns every tool call part in order.
func (m *Message) ToolCalls() []ToolCallRequest {
	if m == nil {
		return nil
	}
	var calls []ToolCallRequest
	for _, part := range m.Parts {
		if part.ToolCall != nil {
			calls = append(calls, *part.ToolCall)
		}
	}
	return calls
}

// ToolResults returns every tool result part in order.
func (m *Message) ToolResults() []ToolCallResult {
	if m == nil {
		return nil
	}
	var results []ToolCallResult
	for _, part := range m.Parts {
		if part.ToolResult != nil {
			results = append(results, *part.ToolResult)
		}
	}
	return results
}

// SystemInstruction returns the concatenated system text when the message is a
// system message, or an error otherwise.
func (m *Message) SystemInstruction() (string, error) {
	if m == nil || m.Role != RoleSystem {
		return "", fmt.Errorf("system instruction requires system role")
	}
	return m.Text(), nil
}

// Clone returns a deep copy of the message.
func (m *Message) Clone() *Message {
	if m == nil {
		return nil
	}
	clone := &Message{Role: m.Role}
	if m.Parts != nil {
		clone.Parts = make([]Part, len(m.Parts))
		for i, part := range m.Parts {
			clone.Parts[i] = part
			clone.Parts[i].Custom = append(json.RawMessage(nil), part.Custom...)
			if part.Media != nil {
				media := *part.Media
				media.Data = append([]byte(nil), part.Media.Data...)
				clone.Parts[i].Media = &media
			}
			if part.ToolCall != nil {
				call := *part.ToolCall
				call.Arguments = append(json.RawMessage(nil), part.ToolCall.Arguments...)
				clone.Parts[i].ToolCall = &call
			}
			if part.ToolResult != nil {
				result := *part.ToolResult
				result.Output = append(json.RawMessage(nil), part.ToolResult.Output...)
				clone.Parts[i].ToolResult = &result
			}
		}
	}
	if m.Metadata != nil {
		clone.Metadata = make(map[string]any, len(m.Metadata))
		for key, value := range m.Metadata {
			clone.Metadata[key] = cloneMetadataValue(value)
		}
	}
	return clone
}

func cloneMetadataValue(value any) any {
	switch typed := value.(type) {
	case json.RawMessage:
		return append(json.RawMessage(nil), typed...)
	case []byte:
		return append([]byte(nil), typed...)
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = cloneMetadataValue(item)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = cloneMetadataValue(item)
		}
		return out
	default:
		return value
	}
}

// ToolCallByID returns the matching tool call or nil.
func (m *Message) ToolCallByID(callID string) *ToolCallRequest {
	for _, part := range m.Parts {
		if part.ToolCall != nil && part.ToolCall.CallID == callID {
			return part.ToolCall
		}
	}
	return nil
}
