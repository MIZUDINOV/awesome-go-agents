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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// EventFormatVersion is the current event envelope format version. A reader
// may replay events with format version <= this constant; a higher version is
// an explicit migration error, never a silent decode.
const EventFormatVersion = 2

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
	EventUserMessage       EventType = "user/message"
	EventSteeringMessage   EventType = "steering/message"
	EventInjectedContext   EventType = "context/injected"
	EventInboxQueued       EventType = "inbox/queued"
	EventInboxClaimed      EventType = "inbox/claimed"
	EventInboxRequeued     EventType = "inbox/requeued"
	EventInboxCompleted    EventType = "inbox/completed"
	EventInboxDiscarded    EventType = "inbox/discarded"
	EventApprovalRequested EventType = "approval/requested"
	EventApprovalResolved  EventType = "approval/resolved"
	EventAssistantChunk    EventType = "assistant/chunk"
	EventAssistantMessage  EventType = "assistant/message"
	EventToolCall          EventType = "tool/call"
	EventToolAdmitted      EventType = "tool/admitted"
	EventToolDispatched    EventType = "tool/dispatched"
	EventToolRunning       EventType = "tool/running"
	EventToolDeferred      EventType = "tool/deferred"
	EventToolResumeStarted EventType = "tool/resume_started"
	EventToolResult        EventType = "tool/result"
	EventRequestError      EventType = "request/error"

	// Diagnostics.
	EventContextSnapshot EventType = "context/snapshot"
	EventUsage           EventType = "usage"

	// Compaction lifecycle. The summary event carries the durable checkpoint:
	// the exact shadowed seq list, the summary text, and a fingerprint. Raw
	// history is never deleted (H-COMPACT-001); compaction only replaces the
	// model-visible projection.
	EventCompactionStart   EventType = "compaction/start"
	EventCompactionSummary EventType = "compaction/summary"
	EventCompactionSurface EventType = "compaction/surface"
	EventCompactionEnd     EventType = "compaction/end"
)

// knownTypes is the authoritative vocabulary. Replay rejects mandatory events
// of an unknown type (H-SESSION-007) instead of silently skipping them.
var knownTypes = map[EventType]bool{
	EventRequestHeader: true, EventRequestContext: true,
	EventTurnStart: true, EventTurnEnd: true,
	EventStepStart: true, EventStepEnd: true,
	EventUserMessage: true, EventAssistantChunk: true, EventAssistantMessage: true,
	EventSteeringMessage: true, EventInjectedContext: true,
	EventInboxQueued: true, EventInboxClaimed: true, EventInboxRequeued: true, EventInboxCompleted: true, EventInboxDiscarded: true,
	EventApprovalRequested: true, EventApprovalResolved: true,
	EventToolCall: true, EventToolResult: true,
	EventToolAdmitted: true, EventToolDispatched: true, EventToolRunning: true, EventToolDeferred: true, EventToolResumeStarted: true,
	EventRequestError:    true,
	EventContextSnapshot: true, EventUsage: true,
	EventCompactionStart: true, EventCompactionSummary: true, EventCompactionSurface: true, EventCompactionEnd: true,
}

// Known reports whether t is part of the event vocabulary.
func (t EventType) Known() bool { return knownTypes[t] || t.Extension() }

// Extension reports whether the type is a vendor event (`vendor/name`). Core
// vocabulary remains closed: malformed/unknown core names are rejected.
func (t EventType) Extension() bool {
	parts := strings.Split(string(t), "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" || parts[0] == "unknown" {
		return false
	}
	switch parts[0] {
	case "request", "turn", "step", "user", "steering", "context", "inbox", "approval", "assistant", "tool", "compaction", "usage":
		return false
	default:
		return true
	}
}

// Surface reports whether an event type participates in the model-facing
// projection. Chunks are flagged non-surface: they accumulate into the
// following assistant/message. Tool calls live inside assistant/message parts
// and are also non-surface. Compaction bookkeeping events are non-surface;
// the summary is folded in as a synthetic message by the projector.
func (t EventType) Surface() bool {
	switch t {
	case EventUserMessage, EventSteeringMessage, EventInjectedContext, EventAssistantMessage, EventToolResult, EventContextSnapshot:
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

// ExtensionEnvelope is the typed envelope for vendor events. The event type
// carries the vendor/name routing key while Payload remains opaque to core
// replay and is preserved byte-for-byte.
type ExtensionEnvelope struct {
	Namespace string          `json:"namespace"`
	Name      string          `json:"name"`
	Payload   json.RawMessage `json:"payload"`
}

func NewExtensionEvent(namespace, name string, payload any) (Event, error) {
	if namespace == "" || name == "" || namespace == "unknown" || strings.Contains(namespace, "/") || strings.Contains(name, "/") {
		return Event{}, fmt.Errorf("session: invalid extension namespace/name")
	}
	if !EventType(namespace + "/" + name).Extension() {
		return Event{}, fmt.Errorf("session: extension namespace %q is reserved", namespace)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return Event{}, err
	}
	data, err := json.Marshal(ExtensionEnvelope{Namespace: namespace, Name: name, Payload: raw})
	if err != nil {
		return Event{}, err
	}
	return Event{Type: EventType(namespace + "/" + name), Data: data}, nil
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
	if e.Type.Extension() {
		var envelope ExtensionEnvelope
		if err := json.Unmarshal(e.Data, &envelope); err != nil || envelope.Namespace == "" || envelope.Name == "" || len(envelope.Payload) == 0 || !json.Valid(envelope.Payload) {
			return fmt.Errorf("session: extension event %q requires a valid opaque envelope", e.Type)
		}
		if envelope.Namespace+"/"+envelope.Name != string(e.Type) {
			return fmt.Errorf("session: extension envelope route does not match event type %q", e.Type)
		}
	}
	if e.Type == EventToolCall || e.Type == EventToolAdmitted || e.Type == EventToolDispatched || e.Type == EventToolRunning || e.Type == EventToolDeferred || e.Type == EventToolResumeStarted || e.Type == EventToolResult || e.Type == EventApprovalRequested || e.Type == EventApprovalResolved {
		if e.CallID == "" {
			return fmt.Errorf("session: event %q requires call_id correlation", e.Type)
		}
	}
	if e.Type == EventAssistantMessage || e.Type == EventToolResult {
		var payload struct {
			Blocks json.RawMessage `json:"blocks"`
		}
		if err := json.Unmarshal(e.Data, &payload); err != nil {
			return fmt.Errorf("session: event %q payload is not an object: %w", e.Type, err)
		}
		if len(payload.Blocks) > 0 {
			if _, err := UnmarshalBlocks(payload.Blocks); err != nil {
				return fmt.Errorf("session: event %q contains invalid blocks: %w", e.Type, err)
			}
		}
	}
	return nil
}

// ValidateType validates the type/vocabulary and format version of an event or
// a proposed event. Unknown mandatory types and unsupported future versions
// are explicit errors, never silent skips.
func ValidateType(t EventType, formatVersion int) error {
	if !t.Known() && !t.Extension() {
		return fmt.Errorf("session: unknown event type %q (replay refuses to skip mandatory events)", t)
	}
	if formatVersion == 0 {
		formatVersion = EventFormatVersion
	}
	if formatVersion < 0 {
		return fmt.Errorf("session: event type %q carries invalid format version %d", t, formatVersion)
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

func UserTextWithInbox(text, inboxID string) json.RawMessage {
	if inboxID == "" {
		return UserText(text)
	}
	return mustJSON(map[string]any{"text": text, "inbox_id": inboxID})
}

type InboxPayload struct {
	ItemID string `json:"item_id"`
	Kind   string `json:"kind"`
	Text   string `json:"text,omitempty"`
}

func InboxPayloadJSON(itemID, kind, text string) json.RawMessage {
	return mustJSON(InboxPayload{ItemID: itemID, Kind: kind, Text: text})
}

type ApprovalResolvedPayload struct {
	CallID   string `json:"call_id"`
	Approved bool   `json:"approved"`
}

func ApprovalResolvedJSON(callID string, approved bool) json.RawMessage {
	return mustJSON(ApprovalResolvedPayload{CallID: callID, Approved: approved})
}

// AssistantContent builds the payload for assistant/message: ordered text and
// reasoning plus the tool calls the message carries.
func AssistantContent(text, reasoning string, calls []ToolCall) json.RawMessage {
	return assistantContent(text, reasoning, calls, nil, false)
}

func AssistantDraftContent(text, reasoning string, calls []ToolCall, interrupted bool) json.RawMessage {
	return assistantContent(text, reasoning, calls, nil, interrupted)
}

// AssistantContentWithMedia is the block-aware form used by streaming
// assembly. Media is kept in the durable assistant message instead of being
// visible only as an ephemeral chunk.
func AssistantContentWithMedia(text, reasoning string, calls []ToolCall, media []MediaBlock, interrupted bool) json.RawMessage {
	return assistantContent(text, reasoning, calls, media, interrupted)
}

// AssistantContentFromParts preserves the provider's ordered block stream in
// the durable assistant/message event. It is the preferred finalizer when a
// provider returned a complete message alongside streamed chunks.
func AssistantContentFromParts(parts []llm.Part, interrupted bool) json.RawMessage {
	return AssistantContentFromPartsWithMetadata(parts, nil, interrupted)
}

// AssistantContentFromPartsWithMetadata preserves provider metadata needed by
// a later continuation, such as reasoning details or annotations.
func AssistantContentFromPartsWithMetadata(parts []llm.Part, metadata map[string]any, interrupted bool) json.RawMessage {
	blocks := make([]ContentBlock, 0, len(parts))
	var text, reasoning string
	var calls []ToolCall
	for _, part := range parts {
		switch part.Type {
		case llm.PartText:
			text += part.Text
			blocks = append(blocks, TextBlock(part.Text))
		case llm.PartReasoning:
			reasoning += part.Reasoning
			blocks = append(blocks, ReasoningBlock(part.Reasoning))
		case llm.PartMedia:
			if part.Media != nil {
				blocks = append(blocks, MediaContentBlock(MediaBlock{MediaType: part.Media.MediaType, URL: part.Media.URL, Data: base64.StdEncoding.EncodeToString(part.Media.Data)}))
			}
		case llm.PartToolCall:
			if part.ToolCall != nil {
				arguments := append(json.RawMessage(nil), part.ToolCall.Arguments...)
				if !json.Valid(arguments) {
					arguments = json.RawMessage(`{}`)
				}
				call := ToolCall{CallID: part.ToolCall.CallID, Name: part.ToolCall.Name, Arguments: arguments}
				calls = append(calls, call)
				if call.CallID != "" && call.Name != "" {
					blocks = append(blocks, ContentBlock{Kind: BlockToolCall, ToolCall: &ToolCallBlock{CallID: call.CallID, Name: call.Name, Arguments: arguments}})
				}
			}
		}
	}
	data := map[string]any{"tool_calls": calls, "blocks": blocks}
	if text != "" {
		data["text"] = text
	}
	if reasoning != "" {
		data["reasoning"] = reasoning
	}
	if metadata != nil {
		data["metadata"] = metadata
	}
	if interrupted {
		data["interrupted"] = true
	}
	return mustJSON(data)
}

func assistantContent(text, reasoning string, calls []ToolCall, media []MediaBlock, interrupted bool) json.RawMessage {
	callPayload := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		arguments := append(json.RawMessage(nil), call.Arguments...)
		if !json.Valid(arguments) {
			arguments = json.RawMessage(`{}`)
		}
		callPayload = append(callPayload, map[string]any{
			"call_id":   call.CallID,
			"name":      call.Name,
			"arguments": arguments,
		})
	}
	blocks := make([]ContentBlock, 0, 2+len(calls))
	if reasoning != "" {
		blocks = append(blocks, ReasoningBlock(reasoning))
	}
	if text != "" {
		blocks = append(blocks, TextBlock(text))
	}
	for _, call := range calls {
		if call.CallID == "" || call.Name == "" {
			continue
		}
		args := append(json.RawMessage(nil), call.Arguments...)
		if !json.Valid(args) {
			args = json.RawMessage(`{}`)
		}
		blocks = append(blocks, ContentBlock{Kind: BlockToolCall, ToolCall: &ToolCallBlock{CallID: call.CallID, Name: call.Name, Arguments: args}})
	}
	for _, media := range media {
		copyMedia := media
		blocks = append(blocks, MediaContentBlock(copyMedia))
	}
	data := map[string]any{"tool_calls": callPayload, "blocks": blocks}
	if text != "" {
		data["text"] = text
	}
	if reasoning != "" {
		data["reasoning"] = reasoning
	}
	if interrupted {
		data["interrupted"] = true
	}
	return mustJSON(data)
}

// AssistantChunkPayload is a durable streaming chunk. Chunks are diagnostic
// history and are assembled into the final assistant/message projection.
type AssistantChunk struct {
	Kind              string            `json:"kind"`
	Content           string            `json:"content,omitempty"`
	CallID            string            `json:"call_id,omitempty"`
	Name              string            `json:"name,omitempty"`
	Arguments         json.RawMessage   `json:"arguments,omitempty"`
	ArgumentsFragment string            `json:"arguments_fragment,omitempty"`
	Media             *llm.MediaContent `json:"media,omitempty"`
}

func AssistantChunkPayload(kind, content, callID string) json.RawMessage {
	return mustJSON(AssistantChunk{Kind: kind, Content: content, CallID: callID})
}

// AssistantToolCallChunkPayload records an assembled provider tool-call delta.
func AssistantToolCallChunkPayload(delta *llm.ToolCallDelta) json.RawMessage {
	if delta == nil {
		return nil
	}
	chunk := AssistantChunk{Kind: "tool_call", CallID: delta.CallID, Name: delta.Name}
	if json.Valid(delta.Arguments) {
		chunk.Arguments = append(json.RawMessage(nil), delta.Arguments...)
	} else if len(delta.Arguments) > 0 {
		// Provider deltas are allowed to be partial JSON. Keep the exact
		// fragment in a string field so durable chunk encoding never panics;
		// recovery can concatenate fragments before final validation.
		chunk.ArgumentsFragment = string(delta.Arguments)
	}
	return mustJSON(chunk)
}

func AssistantMediaChunkPayload(media *llm.MediaContent) json.RawMessage {
	if media == nil {
		return nil
	}
	return mustJSON(AssistantChunk{Kind: "media", Media: media})
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
	if !json.Valid(arguments) {
		arguments = json.RawMessage(`{}`)
	}
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
		"content":  output,
		"is_error": isError,
	})
}

func ToolResultStructuredPayload(callID, name string, content json.RawMessage, meta map[string]any, code string, isError bool) json.RawMessage {
	return ToolResultStructuredPayloadWithBlocksAndContexts(callID, name, content, meta, code, isError, nil, nil)
}

func ToolResultStructuredPayloadWithBlocks(callID, name string, content json.RawMessage, meta map[string]any, code string, isError bool, blocks []ContentBlock) json.RawMessage {
	return ToolResultStructuredPayloadWithBlocksAndContexts(callID, name, content, meta, code, isError, blocks, nil)
}

func ToolResultStructuredPayloadWithBlocksAndContexts(callID, name string, content json.RawMessage, meta map[string]any, code string, isError bool, blocks []ContentBlock, contexts []llm.Message) json.RawMessage {
	return ToolResultStructuredPayloadWithOptions(callID, name, content, meta, code, isError, blocks, contexts, false)
}

func ToolResultStructuredPayloadWithOptions(callID, name string, content json.RawMessage, meta map[string]any, code string, isError bool, blocks []ContentBlock, contexts []llm.Message, concludesTurn bool) json.RawMessage {
	return ToolResultStructuredPayloadWithOutput(callID, name, content, content, meta, code, isError, blocks, contexts, concludesTurn)
}

// ToolResultStructuredPayloadWithOutput keeps the canonical result separate
// from the model-facing content. The two may differ after rendering.
func ToolResultStructuredPayloadWithOutput(callID, name string, output, content json.RawMessage, meta map[string]any, code string, isError bool, blocks []ContentBlock, contexts []llm.Message, concludesTurn bool) json.RawMessage {
	payload := map[string]any{"call_id": callID, "name": name, "output": output, "content": content, "is_error": isError}
	if meta != nil {
		payload["meta"] = meta
	}
	if code != "" {
		payload["code"] = code
	}
	if len(blocks) > 0 {
		if _, err := MarshalBlocks(blocks); err == nil {
			payload["blocks"] = append([]ContentBlock(nil), blocks...)
		}
	}
	if len(contexts) > 0 {
		payload["additional_contexts"] = contexts
	}
	if concludesTurn {
		payload["concludes_turn"] = true
	}
	return mustJSON(payload)
}

// ToolResumeStartedPayload is the durable barrier immediately before a
// deferred tool is executed for the first time after its prerequisite is
// ready. Recovery treats this as potentially side-effecting and never retries
// it blindly.
func ToolResumeStartedPayload(callID, name, resumeKey string) json.RawMessage {
	return mustJSON(map[string]any{
		"call_id": callID, "name": name, "resume_key": resumeKey,
	})
}

// RequestHeaderPayload records the request header and an optional provider
// snapshot. Adapters that expose their resolved wire payload populate the
// snapshot; otherwise it remains the provider-neutral request. It is durable
// so the request context used by a step can be reconstructed on replay.
func RequestHeaderPayload(model, provider string, system []string, tools []string, configHash, requestHash string, capabilities ...llm.Capabilities) json.RawMessage {
	return requestHeaderPayload(model, provider, system, tools, configHash, requestHash, capabilities, nil)
}

func RequestHeaderPayloadWithSnapshot(model, provider string, system []string, tools []string, configHash, requestHash string, capabilities []llm.Capabilities, requestSnapshot json.RawMessage) json.RawMessage {
	return requestHeaderPayload(model, provider, system, tools, configHash, requestHash, capabilities, requestSnapshot)
}

func requestHeaderPayload(model, provider string, system []string, tools []string, configHash, requestHash string, capabilities []llm.Capabilities, requestSnapshot json.RawMessage) json.RawMessage {
	payload := map[string]any{
		"model": model, "provider": provider,
		"system": system, "tools": tools,
		"config_hash": configHash, "request_hash": requestHash,
	}
	if len(capabilities) > 0 {
		payload["capabilities"] = capabilities[0]
	}
	if len(requestSnapshot) > 0 {
		payload["request_snapshot"] = requestSnapshot
	}
	return mustJSON(payload)
}

// RequestContextPayload records the route-specific context capacity that
// governed pressure measurement for a step (H-REQUEST-006).
func RequestContextPayload(model string, contextWindow, maxOutput int64) json.RawMessage {
	return mustJSON(map[string]any{
		"model": model, "context_window": contextWindow, "max_output": maxOutput,
	})
}

func RequestContextPayloadWithCapabilities(model string, contextWindow, maxOutput int64, capabilities *llm.Capabilities) json.RawMessage {
	payload := map[string]any{
		"model": model, "context_window": contextWindow, "max_output": maxOutput,
	}
	if capabilities != nil {
		payload["capabilities"] = *capabilities
	}
	return mustJSON(payload)
}

// RequestErrorPayload records a provider failure after the request header was
// committed. It makes a failed step observable and replayable without
// pretending that a model response was produced.
func RequestErrorPayload(code, message string, streamStarted bool) json.RawMessage {
	return mustJSON(map[string]any{"code": code, "message": message, "stream_started": streamStarted})
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
		"through_seq": throughSeq, "shadowed_seqs": shadowedSeqs, "source_seqs": shadowedSeqs,
		"summary": summary, "fingerprint": fingerprint,
	})
}

// CompactionSurfacePayload is the explicit model-surface replacement. It is
// separate from the summary so replay can ignore an interrupted transaction:
// only a complete start/summary/surface/end sequence changes the projection.
func CompactionSurfacePayload(generation uint64, transactionID string, sourceSeqs []uint64, summary, fingerprint string) json.RawMessage {
	return mustJSON(map[string]any{
		"generation": generation, "transaction_id": transactionID,
		"source_seqs": sourceSeqs, "summary": summary, "fingerprint": fingerprint,
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
