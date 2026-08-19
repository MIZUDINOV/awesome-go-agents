// Package session implements the append-only event log that is the durable
// source of truth for an agent conversation. Model context is a projection
// (surface) of these events, never a mutable array the loop edits in place.
//
// Durability model: events are append-only and immutable once committed. The
// event log IS the truth; the surface is a deterministic projection rebuilt
// from the log (replay produces an identical surface). Events carry a format
// version so schema evolution happens through controlled migrations, and
// unknown mandatory event types fail the replay instead of being silently
// dropped.
package session

import (
	"encoding/json"
	"fmt"
	"time"
)

// EventFormatVersion is the current event envelope format version. A reader
// may replay events with format version <= this constant; a higher version is
// an explicit migration error, never a silent decode.
const EventFormatVersion = 1

// EventType identifies the kind of a session event.
type EventType string

const (
	// Request lifecycle (diagnostic, never part of model surface).
	EventRequestHeader  EventType = "request/header"
	EventRequestContext EventType = "request/context"

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

	// Compaction lifecycle. The summary event carries the durable checkpoint:
	// the exact shadowed seq list, the summary text, and a fingerprint. Raw
	// history is never deleted (H-COMPACT-001); compaction only replaces the
	// model-visible projection.
	EventCompactionStart   EventType = "compaction/start"
	EventCompactionSummary EventType = "compaction/summary"
	EventCompactionEnd     EventType = "compaction/end"
)

// knownTypes is the authoritative vocabulary. Replay rejects mandatory events
// of an unknown type (H-SESSION-007) instead of silently skipping them.
var knownTypes = map[EventType]bool{
	EventRequestHeader: true, EventRequestContext: true,
	EventTurnStart: true, EventTurnEnd: true,
	EventStepStart: true, EventStepEnd: true,
	EventUserMessage: true, EventAssistantChunk: true, EventAssistantMessage: true,
	EventToolCall: true, EventToolResult: true,
	EventContextSnapshot: true, EventUsage: true,
	EventCompactionStart: true, EventCompactionSummary: true, EventCompactionEnd: true,
}

// Known reports whether t is part of the event vocabulary.
func (t EventType) Known() bool { return knownTypes[t] }

// Surface reports whether an event type participates in the model-facing
// projection. Chunks are flagged non-surface: they accumulate into the
// following assistant/message. Tool calls live inside assistant/message parts
// and are also non-surface. Compaction bookkeeping events are non-surface;
// the summary is folded in as a synthetic message by the projector.
func (t EventType) Surface() bool {
	switch t {
	case EventUserMessage, EventAssistantMessage, EventToolResult, EventContextSnapshot:
		return true
	default:
		return false
	}
}

// Event is one durable entry in the session log. Fields are additive across
// format versions: a reader of version N can always read version <= N, and a
// controlled migration upgrades older rows in place (never destructively).
type Event struct {
	// Seq is the strictly monotonic per-session sequence. Assigned by the
	// store on append; immutable afterwards.
	Seq uint64 `json:"seq"`
	// Type is the stable event type. Unknown mandatory types fail replay.
	Type EventType `json:"type"`
	// FormatVersion is the envelope format version at write time.
	FormatVersion int `json:"format_version"`
	// ID is the client-supplied stable event identity used for idempotent
	// retries of the same logical append. Empty disables retry dedup for that
	// event (older data).
	ID        string          `json:"id,omitempty"`
	Timestamp time.Time       `json:"timestamp"`
	Data      json.RawMessage `json:"data"`
	// SourceSeqs links a derived event to the events that produced it
	// (e.g. tool/result → the tool/call it answers). Aids crash recovery.
	SourceSeqs []uint64 `json:"source_seqs,omitempty"`

	// Correlation metadata binds the event to its session/run/turn/step/call.
	SessionID string `json:"session_id,omitempty"`
	RunID     string `json:"run_id,omitempty"`
	TurnID    string `json:"turn_id,omitempty"`
	StepID    string `json:"step_id,omitempty"`
	CallID    string `json:"call_id,omitempty"`

	// Surface is a runtime-only hint (not persisted); the authoritative value
	// derives from Type.Surface().
	Surface bool `json:"-"`
}

// Clone returns a deep copy of the event, isolating mutable fields (Data,
// SourceSeqs) from the caller (H-SESSION-004).
func (e Event) Clone() Event {
	out := e
	out.Data = append(json.RawMessage(nil), e.Data...)
	if e.SourceSeqs != nil {
		out.SourceSeqs = append([]uint64(nil), e.SourceSeqs...)
	}
	return out
}

// Validate checks the mandatory invariants of an event before it may be
// appended: known type, supported format version, and a valid JSON payload.
// Correlation metadata is type-required where the vocabulary demands it
// (tool events must carry the originating call id).
func (e Event) Validate() error {
	if err := ValidateType(e.Type, e.FormatVersion); err != nil {
		return err
	}
	if len(e.Data) == 0 || !json.Valid(e.Data) {
		return fmt.Errorf("session: event %q data is not valid JSON", e.Type)
	}
	if e.Type == EventToolCall || e.Type == EventToolResult {
		if e.CallID == "" {
			return fmt.Errorf("session: event %q requires call_id correlation", e.Type)
		}
	}
	return nil
}

// ValidateType validates the type/vocabulary and format version of an event or
// a proposed event. Unknown mandatory types and unsupported future versions
// are explicit errors, never silent skips.
func ValidateType(t EventType, formatVersion int) error {
	if !t.Known() {
		return fmt.Errorf("session: unknown event type %q (replay refuses to skip mandatory events)", t)
	}
	if formatVersion == 0 {
		formatVersion = EventFormatVersion
	}
	if formatVersion > EventFormatVersion {
		return fmt.Errorf("session: event type %q carries unsupported format version %d (reader supports <= %d): migration required", t, formatVersion, EventFormatVersion)
	}
	return nil
}

// Normalize fills defaults (format version, timestamp) without mutating Data
// or SourceSeqs references.
func (e *Event) Normalize() {
	if e.FormatVersion == 0 {
		e.FormatVersion = EventFormatVersion
	}
	if e.Timestamp.IsZero() {
		e.Timestamp = time.Now().UTC()
	}
}

// NormalizedFormatVersion returns the effective format version (current
// version when unset) without mutating the event.
func (e Event) NormalizedFormatVersion() int {
	if e.FormatVersion == 0 {
		return EventFormatVersion
	}
	return e.FormatVersion
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

// ToolCallPayload builds the payload for tool/call. It must be persisted
// BEFORE any side effect of the tool executes (H-RUNTIME-008 barrier).
func ToolCallPayload(callID, name string, arguments json.RawMessage) json.RawMessage {
	return mustJSON(map[string]any{
		"call_id":   callID,
		"name":      name,
		"arguments": arguments,
	})
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

// RequestHeaderPayload records the canonical request header: exactly what was
// sent to the provider for a step after adapter defaults were resolved. It is
// durable so the exact request can be reconstructed on replay (H-REQUEST-001).
func RequestHeaderPayload(model, provider string, system []string, tools []string, configHash, requestHash string) json.RawMessage {
	return mustJSON(map[string]any{
		"model": model, "provider": provider,
		"system": system, "tools": tools,
		"config_hash": configHash, "request_hash": requestHash,
	})
}

// RequestContextPayload records the route-specific context capacity that
// governed pressure measurement for a step (H-REQUEST-006).
func RequestContextPayload(model string, contextWindow, maxOutput int64) json.RawMessage {
	return mustJSON(map[string]any{
		"model": model, "context_window": contextWindow, "max_output": maxOutput,
	})
}

// CompactionStartPayload opens a compaction transaction. generation increases
// monotonically per session; transaction_id correlates start/summary/end.
func CompactionStartPayload(generation uint64, transactionID string, sourceSeqs []uint64) json.RawMessage {
	return mustJSON(map[string]any{
		"generation": generation, "transaction_id": transactionID, "source_seqs": sourceSeqs,
	})
}

// CompactionSummaryPayload is the durable checkpoint. ShadowedSeqs is the
// EXACT list of event seqs replaced by Summary, and Fingerprint is a digest of
// the source region used for drift detection.
func CompactionSummaryPayload(generation uint64, transactionID string, throughSeq uint64, shadowedSeqs []uint64, summary, fingerprint string) json.RawMessage {
	return mustJSON(map[string]any{
		"generation": generation, "transaction_id": transactionID,
		"through_seq": throughSeq, "shadowed_seqs": shadowedSeqs,
		"summary": summary, "fingerprint": fingerprint,
	})
}

// CompactionEndPayload seals a compaction transaction.
func CompactionEndPayload(generation uint64, transactionID string) json.RawMessage {
	return mustJSON(map[string]any{
		"generation": generation, "transaction_id": transactionID,
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
