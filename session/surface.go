package session

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// CompactedRegion describes a slice of events that has been summarised. The
// projection replaces that region with a single synthetic user message carrying
// the summary.
//
// ShadowedSeqs, when present, is the EXACT list of shadowed sequences
// (H-SURFACE-005); ThroughSeq is the inclusive prefix convenience kept for
// compatibility. The durable path records compaction via
// compaction/start|summary|end events; CompactedRegion remains the in-memory
// test/override form.
type CompactedRegion struct {
	// ThroughSeq is the seq of the last event covered by the summary.
	ThroughSeq uint64
	// ShadowedSeqs is the exact shadowed sequence list (preferred over
	// ThroughSeq when non-empty).
	ShadowedSeqs []uint64
	// Summary is the compaction checkpoint text.
	Summary string
	// Generation increases each time the region is replaced (H-SURFACE-010).
	Generation uint64
	// Fingerprint digests the source region for drift detection.
	Fingerprint string
}

// SurfaceSpec controls projection, most importantly how an already compacted
// region is folded into the history.
type SurfaceSpec struct {
	// Compacted, when non-nil, replaces the covered events with a single
	// synthetic summary message. Durable compaction events take precedence
	// over this override when both are present.
	Compacted *CompactedRegion
	// SystemSnapshot, when set, is prepended as a role=system message carrying
	// the current assembled system prompt/sections.
	SystemSnapshot string
}

// Projection is the immutable outcome of projecting a log into the model
// surface. Returning it as a value snapshot guarantees the caller never sees a
// mutated copy (H-SURFACE-007).
type Projection struct {
	// Generation increments whenever the shadowed region changes.
	Generation uint64
	// ShadowedSeqs is the exact current shadowed list (sorted).
	ShadowedSeqs []uint64
	// Summary is the effective checkpoint text ("" when nothing is shadowed).
	Summary string
	// Fingerprint digests the shadowed source region.
	Fingerprint string
}

// Surface projects a run of events into provider-neutral LLM messages. It is
// the durable-history → model-context boundary: diagnostic events (turn/step
// lifecycles, chunks, usage, tool/call, compaction bookkeeping) never reach
// the model. Projection is deterministic: replaying the same log produces the
// identical surface (H-SURFACE-009).
type Surface struct {
	spec SurfaceSpec
}

// NewSurface returns a Surface with the given spec.
func NewSurface(spec SurfaceSpec) *Surface { return &Surface{spec: spec} }

// DeriveMessages projects events (in ascending seq order) into []*llm.Message.
// It never mutates the input and returns a fresh snapshot.
func (s *Surface) DeriveMessages(events []Event) ([]*llm.Message, error) {
	messages, _, err := s.Project(events)
	return messages, err
}

// Project derives the model messages and the projection metadata (generation,
// exact shadowed seqs, summary, fingerprint). The returned Projection is
// owned by the caller and never aliases internal state.
func (s *Surface) Project(events []Event) ([]*llm.Message, *Projection, error) {
	shadowed := make(map[uint64]bool)
	if spec := s.spec.Compacted; spec != nil {
		if len(spec.ShadowedSeqs) > 0 {
			for _, seq := range spec.ShadowedSeqs {
				shadowed[seq] = true
			}
		} else if spec.ThroughSeq > 0 {
			for seq := uint64(1); seq <= spec.ThroughSeq; seq++ {
				shadowed[seq] = true
			}
		}
	}

	proj := &Projection{
		Generation:   regionGeneration(s.spec.Compacted),
		Summary:      regionSummary(s.spec.Compacted),
		Fingerprint:  regionFingerprint(s.spec.Compacted),
		ShadowedSeqs: sortedShadowed(shadowed),
	}

	// First pass: fold all durable compaction/summary events (in order) into
	// the projection. Later checkpoints advance the generation and replace the
	// summary text; shadowed sets accumulate.
	for _, event := range events {
		if event.Type == EventCompactionSummary {
			payload, err := decodeCompactionSummary(event.Data)
			if err != nil {
				return nil, nil, fmt.Errorf("session: decode compaction/summary at seq %d: %w", event.Seq, err)
			}
			if payload.Generation > 0 && payload.Generation < proj.Generation {
				return nil, nil, fmt.Errorf("session: compaction generation regressed at seq %d (%d < %d)", event.Seq, payload.Generation, proj.Generation)
			}
			for _, seq := range payload.ShadowedSeqs {
				shadowed[seq] = true
			}
			proj.Generation = payload.Generation
			proj.Summary = payload.Summary
			if payload.Fingerprint != "" {
				proj.Fingerprint = payload.Fingerprint
			}
		}
	}
	proj.ShadowedSeqs = sortedShadowed(shadowed)

	// Pairing integrity (H-SURFACE-008): a shadowed assistant message must not
	// orphan its tool result into the visible tail.
	if err := validatePairing(events, shadowed); err != nil {
		return nil, nil, err
	}

	// Second pass: build the messages from non-shadowed surface events.
	var messages []*llm.Message
	for _, event := range events {
		if shadowed[event.Seq] {
			continue
		}
		if !event.Type.Surface() {
			continue
		}
		msg, err := projectSurfaceEvent(event)
		if err != nil {
			return nil, nil, err
		}
		messages = append(messages, msg)
	}

	if s.spec.SystemSnapshot != "" {
		messages = append([]*llm.Message{llm.NewTextMessage(llm.RoleSystem, s.spec.SystemSnapshot)}, messages...)
	}
	if proj.Summary != "" {
		messages = append([]*llm.Message{llm.NewUserMessage(proj.Summary)}, messages...)
	}
	return messages, proj, nil
}

// ReplaceSurfaceFold applies an in-memory compaction replacement to a fresh
// Projection. Kept for compatibility with callers that pass CompactedRegion.
// Never mutates prior events.

func projectSurfaceEvent(event Event) (*llm.Message, error) {
	switch event.Type {
	case EventUserMessage:
		var payload struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, fmt.Errorf("session: decode user message: %w", err)
		}
		return llm.NewUserMessage(payload.Text), nil

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
		return llm.NewAssistantMessage(payload.Text, payload.Reasoning, calls), nil

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
		return llm.NewToolResultMessage(llm.ToolCallResult{
			CallID:  payload.CallID,
			Name:    payload.Name,
			Output:  append(json.RawMessage(nil), payload.Output...),
			IsError: payload.IsError,
		}), nil

	case EventContextSnapshot:
		var payload struct {
			Snapshot string `json:"snapshot"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, fmt.Errorf("session: decode context snapshot: %w", err)
		}
		return llm.NewTextMessage(llm.RoleSystem, payload.Snapshot), nil
	}
	return nil, nil
}

type compactionSummaryPayload struct {
	Generation    uint64   `json:"generation"`
	TransactionID string   `json:"transaction_id"`
	ThroughSeq    uint64   `json:"through_seq"`
	ShadowedSeqs  []uint64 `json:"shadowed_seqs"`
	Summary       string   `json:"summary"`
	Fingerprint   string   `json:"fingerprint"`
}

func decodeCompactionSummary(data json.RawMessage) (compactionSummaryPayload, error) {
	var payload compactionSummaryPayload
	if err := json.Unmarshal(data, &payload); err != nil {
		return payload, fmt.Errorf("decode payload: %w", err)
	}
	return payload, nil
}

// validatePairing ensures no visible tool/result is orphaned by shadowing:
// when a result survives but its originating assistant message (or tool/call
// record) was shadowed, the pairing is broken and the log/projection is
// inconsistent (H-SURFACE-008).
func validatePairing(events []Event, shadowed map[uint64]bool) error {
	for _, event := range events {
		if event.Type != EventToolResult || shadowed[event.Seq] {
			continue
		}
		for _, source := range event.SourceSeqs {
			if shadowed[source] {
				return fmt.Errorf("session: compaction shadowed seq %d while its tool/result seq %d remains visible: call/result pairing would break", source, event.Seq)
			}
		}
	}
	return nil
}

func regionGeneration(region *CompactedRegion) uint64 {
	if region == nil {
		return 0
	}
	return region.Generation
}

func regionSummary(region *CompactedRegion) string {
	if region == nil {
		return ""
	}
	return region.Summary
}

func regionFingerprint(region *CompactedRegion) string {
	if region == nil {
		return ""
	}
	return region.Fingerprint
}

func sortedShadowed(set map[uint64]bool) []uint64 {
	out := make([]uint64, 0, len(set))
	for seq := range set {
		out = append(out, seq)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
