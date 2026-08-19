// Package session implements the append-only event log that is the durable
// source of truth for an agent conversation. Model context is a projection
// (surface) of these events, never a mutable array the loop edits in place.
package session

import (
	"encoding/json"
	"fmt"
	"time"
)

// EventType identifies the kind of a session event.
type EventType string

const (
	// Request lifecycle (diagnostic, never part of model surface).
	EventRequestHeader EventType = "request/header"

	// Turn lifecycle.
	EventTurnStart EventType = "turn/start"
	EventTurnEnd   EventType = "turn/end"

	// Step lifecycle (one model call + its tools).
	EventStepStart EventType = "step/start"
	EventStepEnd   EventType = "step/end"

	// Content events.
	EventUserMessage      EventType = "user/message"
	EventAssistantChunk   EventType = "assistant/chunk"
	EventAssistantMessage EventType = "assistant/message"
	EventToolCall         EventType = "tool/call"
	EventToolResult       EventType = "tool/result"

	// Diagnostics.
	EventContextSnapshot EventType = "context/snapshot"
	EventUsage           EventType = "usage"
	EventCompactionStart EventType = "compaction/start"
	EventCompactionEnd   EventType = "compaction/end"
)

// Surface reports whether an event type participates in the model-facing
// projection. Chunks are flagged non-surface: they accumulate into the
// following assistant/message. Tool calls live inside assistant/message parts
// and are also non-surface.
func (t EventType) Surface() bool {
	switch t {
	case EventUserMessage, EventAssistantMessage, EventToolResult, EventContextSnapshot:
		return true
	default:
		return false
	}
}

// Event is one durable entry in the session log.
type Event struct {
	Seq       uint64          `json:"seq"`
	Type      EventType       `json:"type"`
	Timestamp time.Time       `json:"timestamp"`
	Data      json.RawMessage `json:"data"`
	// SourceSeqs links a derived event to the events that produced it
	// (e.g. tool/result → the tool/call it answers). Aids crash recovery.
	SourceSeqs []uint64 `json:"source_seqs,omitempty"`

	// Surface is a runtime-only hint (not persisted); the authoritative value
	// derives from Type.Surface().
	Surface bool `json:"-"`
}

// UserText builds the data payload for user/message.
func UserText(text string) json.RawMessage {
	return mustJSON(map[string]any{"text": text})
}

// AssistantContent builds the payload for assistant/message: ordered text and
// reasoning plus the tool calls the message carries.
func AssistantContent(text, reasoning string, calls []ToolCall) json.RawMessage {
	callPayload := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		callPayload = append(callPayload, map[string]any{
			"call_id":   call.CallID,
			"name":      call.Name,
			"arguments": call.Arguments,
		})
	}
	data := map[string]any{"tool_calls": callPayload}
	if text != "" {
		data["text"] = text
	}
	if reasoning != "" {
		data["reasoning"] = reasoning
	}
	return mustJSON(data)
}

// ToolCall is the durable representation of a model tool request.
type ToolCall struct {
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResultPayload builds the payload for tool/result.
func ToolResultPayload(callID, name string, output json.RawMessage, isError bool) json.RawMessage {
	return mustJSON(map[string]any{
		"call_id":  callID,
		"name":     name,
		"output":   output,
		"is_error": isError,
	})
}

func mustJSON(v any) json.RawMessage {
	encoded, err := json.Marshal(v)
	if err != nil {
		// Data payloads are constructed from marshallable primitives.
		panic(fmt.Sprintf("session: marshal payload: %v", err))
	}
	return encoded
}
