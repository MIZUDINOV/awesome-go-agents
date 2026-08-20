package session

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// InterruptedDraftEvents deterministically assembles assistant/chunk records
// that have no assistant/message companion. Stores call this during fenced
// recovery so a provider crash never makes already streamed content vanish.
// The returned events are immutable inputs; the store assigns sequence and
// timestamp values when it commits them.
func InterruptedDraftEvents(events []Event, sessionID string) []Event {
	type draft struct {
		sessionID, runID, turnID, stepID string
		text, reasoning                  string
		calls                            map[string]ToolCall
		order                            []string
		media                            []MediaBlock
		sources                          []uint64
	}
	drafts := map[string]*draft{}
	completed := map[string]bool{}
	for _, event := range events {
		key := draftKey(event.RunID, event.TurnID, event.StepID)
		if event.Type == EventAssistantMessage {
			completed[key] = true
			continue
		}
		if event.Type != EventAssistantChunk {
			continue
		}
		var payload struct {
			Kind              string          `json:"kind"`
			Content           string          `json:"content"`
			CallID            string          `json:"call_id"`
			Name              string          `json:"name"`
			Arguments         json.RawMessage `json:"arguments"`
			ArgumentsFragment string          `json:"arguments_fragment"`
			Media             *struct {
				MediaType string `json:"media_type"`
				URL       string `json:"url,omitempty"`
				Data      []byte `json:"data,omitempty"`
			} `json:"media"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			continue
		}
		current := drafts[key]
		if current == nil {
			current = &draft{sessionID: sessionID, runID: event.RunID, turnID: event.TurnID, stepID: event.StepID, calls: map[string]ToolCall{}}
			drafts[key] = current
		}
		current.sources = append(current.sources, event.Seq)
		switch payload.Kind {
		case "text":
			current.text += payload.Content
		case "reasoning":
			current.reasoning += payload.Content
		case "tool_call":
			if payload.CallID == "" {
				continue
			}
			if _, exists := current.calls[payload.CallID]; !exists {
				current.order = append(current.order, payload.CallID)
			}
			arguments := append(json.RawMessage(nil), payload.Arguments...)
			if len(arguments) == 0 && payload.ArgumentsFragment != "" {
				arguments = json.RawMessage(payload.ArgumentsFragment)
				if previous := current.calls[payload.CallID].Arguments; len(previous) > 0 && !json.Valid(arguments) {
					arguments = append(append(json.RawMessage(nil), previous...), arguments...)
				}
			}
			if !json.Valid(arguments) {
				arguments = json.RawMessage(`{}`)
			}
			current.calls[payload.CallID] = ToolCall{CallID: payload.CallID, Name: payload.Name, Arguments: arguments}
		case "media":
			if payload.Media != nil {
				current.media = append(current.media, MediaBlock{MediaType: payload.Media.MediaType, URL: payload.Media.URL, Data: base64.StdEncoding.EncodeToString(payload.Media.Data)})
			}
		}
	}
	keys := make([]string, 0, len(drafts))
	for key := range drafts {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		left, right := drafts[keys[i]], drafts[keys[j]]
		var leftSeq, rightSeq uint64
		if len(left.sources) > 0 {
			leftSeq = left.sources[0]
		}
		if len(right.sources) > 0 {
			rightSeq = right.sources[0]
		}
		if leftSeq == rightSeq {
			return keys[i] < keys[j]
		}
		return leftSeq < rightSeq
	})
	result := make([]Event, 0, len(drafts))
	for _, key := range keys {
		current := drafts[key]
		if completed[key] || (current.text == "" && current.reasoning == "" && len(current.calls) == 0 && len(current.media) == 0) {
			continue
		}
		calls := make([]ToolCall, 0, len(current.order))
		for _, callID := range current.order {
			calls = append(calls, current.calls[callID])
		}
		digest := sha256.Sum256([]byte(key))
		result = append(result, Event{ID: "recover:draft:" + hex.EncodeToString(digest[:]), SessionID: current.sessionID, RunID: current.runID, TurnID: current.turnID, StepID: current.stepID, Type: EventAssistantMessage, SourceSeqs: append([]uint64(nil), current.sources...), Data: AssistantContentWithMedia(current.text, current.reasoning, calls, current.media, true)})
	}
	return result
}

// InterruptedStepEndEvents closes every step/start that has no matching
// step/end. Recovery emits one deterministic terminal event per step so a
// crash cannot leave an open step lifecycle behind.
func InterruptedStepEndEvents(events []Event, sessionID string) []Event {
	type openStep struct {
		event Event
		seq   uint64
		key   string
	}
	open := make(map[string]openStep)
	for _, event := range events {
		if event.StepID == "" || event.TurnID == "" {
			continue
		}
		key := draftKey(event.RunID, event.TurnID, event.StepID)
		switch event.Type {
		case EventStepStart:
			open[key] = openStep{event: event, seq: event.Seq, key: key}
		case EventStepEnd:
			delete(open, key)
		}
	}
	steps := make([]openStep, 0, len(open))
	for _, step := range open {
		steps = append(steps, step)
	}
	sort.Slice(steps, func(i, j int) bool {
		if steps[i].seq == steps[j].seq {
			return steps[i].key < steps[j].key
		}
		return steps[i].seq < steps[j].seq
	})
	out := make([]Event, 0, len(steps))
	for _, step := range steps {
		digest := sha256.Sum256([]byte(step.key))
		out = append(out, Event{
			ID:        "recover:step-end:" + hex.EncodeToString(digest[:]),
			SessionID: firstNonEmptySessionID(step.event.SessionID, sessionID),
			RunID:     step.event.RunID,
			TurnID:    step.event.TurnID,
			StepID:    step.event.StepID,
			Type:      EventStepEnd,
			Data:      mustJSON(map[string]any{"reason": "interrupted"}),
		})
	}
	return out
}

func firstNonEmptySessionID(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

func draftKey(runID, turnID, stepID string) string {
	return fmt.Sprintf("%s/%s/%s", runID, turnID, stepID)
}
