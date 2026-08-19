package session

import (
	"encoding/json"
	"fmt"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// CompactedRegion describes a slice of events that has been summarised. The
// projection replaces that region with a single synthetic user message carrying
// the summary, mirroring DeepSeek Harness' compacted-region + recent-tail model.
type CompactedRegion struct {
	// ThroughSeq is the seq of the last event covered by the summary.
	ThroughSeq uint64
	// Summary is the compaction checkpoint text.
	Summary string
}

// SurfaceSpec controls projection, most importantly how an already compacted
// region is folded into the history.
type SurfaceSpec struct {
	// Compacted, when non-nil, replaces all events with seq <= ThroughSeq with
	// a single synthetic summary message.
	Compacted *CompactedRegion
	// SystemSnapshot, when set, is prepended as a role=system message carrying
	// the current assembled system prompt/sections.
	SystemSnapshot string
}

// Surface projects a run of events into provider-neutral LLM messages. It is
// the durable-history → model-context boundary: diagnostic events (turn/step
// lifecycles, chunks, usage, tool/call) never reach the model.
type Surface struct {
	spec SurfaceSpec
}

// NewSurface returns a Surface with the given spec.
func NewSurface(spec SurfaceSpec) *Surface { return &Surface{spec: spec} }

// DeriveMessages projects events (in ascending seq order) into []*llm.Message.
func (s *Surface) DeriveMessages(events []Event) ([]*llm.Message, error) {
	var messages []*llm.Message
	if s.spec.SystemSnapshot != "" {
		messages = append(messages, llm.NewTextMessage(llm.RoleSystem, s.spec.SystemSnapshot))
	}

	compacted := s.spec.Compacted
	for _, event := range events {
		if compacted != nil && event.Seq <= compacted.ThroughSeq {
			continue
		}
		if !event.Type.Surface() {
			continue
		}
		switch event.Type {
		case EventUserMessage:
			var payload struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("session: decode user message: %w", err)
			}
			messages = append(messages, llm.NewUserMessage(payload.Text))

		case EventAssistantMessage:
			var payload struct {
				Text      string     `json:"text"`
				Reasoning string     `json:"reasoning"`
				ToolCalls []ToolCall `json:"tool_calls"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("session: decode assistant message: %w", err)
			}
			calls := make([]llm.ToolCallRequest, 0, len(payload.ToolCalls))
			for _, call := range payload.ToolCalls {
				calls = append(calls, llm.ToolCallRequest{
					CallID:    call.CallID,
					Name:      call.Name,
					Arguments: append(json.RawMessage(nil), call.Arguments...),
				})
			}
			messages = append(messages, llm.NewAssistantMessage(payload.Text, payload.Reasoning, calls))

		case EventToolResult:
			var payload struct {
				CallID  string          `json:"call_id"`
				Name    string          `json:"name"`
				Output  json.RawMessage `json:"output"`
				IsError bool            `json:"is_error"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("session: decode tool result: %w", err)
			}
			messages = append(messages, llm.NewToolResultMessage(llm.ToolCallResult{
				CallID:  payload.CallID,
				Name:    payload.Name,
				Output:  append(json.RawMessage(nil), payload.Output...),
				IsError: payload.IsError,
			}))

		case EventContextSnapshot:
			var payload struct {
				Snapshot string `json:"snapshot"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, fmt.Errorf("session: decode context snapshot: %w", err)
			}
			messages = append(messages, llm.NewTextMessage(llm.RoleSystem, payload.Snapshot))
		}
	}

	if compacted != nil && compacted.Summary != "" {
		messages = append([]*llm.Message{llm.NewUserMessage(compacted.Summary)}, messages...)
	}
	return messages, nil
}
