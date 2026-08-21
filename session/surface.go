package session

import (
	"encoding/base64"
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
	legacySummary := proj.Summary != ""

	// First pass: fold durable compaction transactions into the projection.
	// Format v1 summaries are kept readable as legacy replacements. Format v2
	// requires the explicit surface event and a sealed end event, so an
	// interrupted transaction cannot change the model surface during replay.
	type compactionTransaction struct {
		generation    uint64
		transactionID string
		sourceSeqs    []uint64
		summary       string
		fingerprint   string
		start         bool
		summarySeen   bool
		surfaceSeen   bool
		end           bool
	}
	type surfaceReplacement struct {
		sourceSeqs []uint64
		summary    string
	}
	transactions := make(map[string]*compactionTransaction)
	transactionOrder := make([]string, 0)
	replacements := make([]surfaceReplacement, 0)
	getTransaction := func(transactionID string) *compactionTransaction {
		tx := transactions[transactionID]
		if tx == nil {
			tx = &compactionTransaction{transactionID: transactionID}
			transactions[transactionID] = tx
			transactionOrder = append(transactionOrder, transactionID)
		}
		return tx
	}
	apply := func(generation uint64, sourceSeqs []uint64, summary, fingerprint string, seq uint64) error {
		if generation > 0 && generation < proj.Generation {
			return fmt.Errorf("session: compaction generation regressed at seq %d (%d < %d)", seq, generation, proj.Generation)
		}
		for _, sourceSeq := range sourceSeqs {
			if sourceSeq > 0 {
				shadowed[sourceSeq] = true
			}
		}
		if generation > 0 {
			proj.Generation = generation
		}
		proj.Summary = summary
		if fingerprint != "" {
			proj.Fingerprint = fingerprint
		}
		return nil
	}
	for _, event := range events {
		switch event.Type {
		case EventCompactionStart:
			if event.FormatVersion < 2 {
				continue
			}
			var payload struct {
				Generation    uint64   `json:"generation"`
				TransactionID string   `json:"transaction_id"`
				SourceSeqs    []uint64 `json:"source_seqs"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, nil, fmt.Errorf("session: decode compaction/start at seq %d: %w", event.Seq, err)
			}
			if payload.TransactionID == "" {
				return nil, nil, fmt.Errorf("session: compaction/start at seq %d has no transaction id", event.Seq)
			}
			if err := validateCompactionSourceSeqs(payload.SourceSeqs); err != nil {
				return nil, nil, fmt.Errorf("session: compaction/start at seq %d: %w", event.Seq, err)
			}
			tx := getTransaction(payload.TransactionID)
			if tx.start || tx.summarySeen || tx.surfaceSeen || tx.end {
				return nil, nil, fmt.Errorf("session: compaction transaction %q has invalid start order", payload.TransactionID)
			}
			tx.generation, tx.sourceSeqs, tx.start = payload.Generation, append([]uint64(nil), payload.SourceSeqs...), true
		case EventCompactionSummary:
			payload, err := decodeCompactionSummary(event.Data)
			if err != nil {
				return nil, nil, fmt.Errorf("session: decode compaction/summary at seq %d: %w", event.Seq, err)
			}
			// A zero format version is how old hand-built events and legacy rows
			// without an envelope version arrive. They must retain v1 read
			// semantics even though Normalize now writes v2.
			if event.FormatVersion == 0 || event.FormatVersion < 2 {
				seqs := payload.ShadowedSeqs
				if len(seqs) == 0 {
					seqs = payload.SourceSeqs
				}
				if err := apply(payload.Generation, seqs, payload.Summary, payload.Fingerprint, event.Seq); err != nil {
					return nil, nil, err
				}
				legacySummary = legacySummary || payload.Summary != ""
				continue
			}
			if payload.TransactionID == "" {
				return nil, nil, fmt.Errorf("session: compaction/summary at seq %d has no transaction id", event.Seq)
			}
			seqs := payload.SourceSeqs
			if len(seqs) == 0 {
				seqs = payload.ShadowedSeqs
			}
			if err := validateCompactionSourceSeqs(seqs); err != nil {
				return nil, nil, fmt.Errorf("session: compaction/summary at seq %d: %w", event.Seq, err)
			}
			tx := getTransaction(payload.TransactionID)
			if !tx.start || tx.summarySeen || tx.surfaceSeen || tx.end {
				return nil, nil, fmt.Errorf("session: compaction transaction %q has invalid summary order", payload.TransactionID)
			}
			if tx.generation != payload.Generation || !sameSeqs(tx.sourceSeqs, seqs) {
				return nil, nil, fmt.Errorf("session: compaction transaction %q summary does not match start", payload.TransactionID)
			}
			tx.summary, tx.fingerprint, tx.summarySeen = payload.Summary, payload.Fingerprint, true
		case EventCompactionSurface:
			if event.FormatVersion < 2 {
				continue
			}
			var payload struct {
				Generation    uint64   `json:"generation"`
				TransactionID string   `json:"transaction_id"`
				SourceSeqs    []uint64 `json:"source_seqs"`
				Summary       string   `json:"summary"`
				Fingerprint   string   `json:"fingerprint"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, nil, fmt.Errorf("session: decode compaction/surface at seq %d: %w", event.Seq, err)
			}
			if payload.TransactionID == "" {
				return nil, nil, fmt.Errorf("session: compaction/surface at seq %d has no transaction id", event.Seq)
			}
			if err := validateCompactionSourceSeqs(payload.SourceSeqs); err != nil {
				return nil, nil, fmt.Errorf("session: compaction/surface at seq %d: %w", event.Seq, err)
			}
			tx := getTransaction(payload.TransactionID)
			if !tx.start || !tx.summarySeen || tx.surfaceSeen || tx.end {
				return nil, nil, fmt.Errorf("session: compaction transaction %q has invalid surface order", payload.TransactionID)
			}
			if tx.generation != payload.Generation || !sameSeqs(tx.sourceSeqs, payload.SourceSeqs) || tx.summary != payload.Summary || tx.fingerprint != payload.Fingerprint {
				return nil, nil, fmt.Errorf("session: compaction transaction %q surface does not match summary", payload.TransactionID)
			}
			tx.surfaceSeen = true
		case EventCompactionEnd:
			if event.FormatVersion < 2 {
				continue
			}
			var payload struct {
				Generation    uint64 `json:"generation"`
				TransactionID string `json:"transaction_id"`
			}
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				return nil, nil, fmt.Errorf("session: decode compaction/end at seq %d: %w", event.Seq, err)
			}
			if payload.TransactionID == "" {
				return nil, nil, fmt.Errorf("session: compaction/end at seq %d has no transaction id", event.Seq)
			}
			tx := transactions[payload.TransactionID]
			if tx == nil || !tx.start || !tx.summarySeen || !tx.surfaceSeen || tx.end {
				return nil, nil, fmt.Errorf("session: compaction transaction %q has invalid end order", payload.TransactionID)
			}
			if tx.generation != payload.Generation {
				return nil, nil, fmt.Errorf("session: compaction transaction %q end does not match start", payload.TransactionID)
			}
			tx.end = true
		}
	}
	for _, transactionID := range transactionOrder {
		tx := transactions[transactionID]
		if !tx.start || !tx.summarySeen || !tx.surfaceSeen || !tx.end {
			continue
		}
		if err := apply(tx.generation, tx.sourceSeqs, tx.summary, tx.fingerprint, 0); err != nil {
			return nil, nil, err
		}
		replacements = append(replacements, surfaceReplacement{sourceSeqs: append([]uint64(nil), tx.sourceSeqs...), summary: tx.summary})
	}
	proj.ShadowedSeqs = sortedShadowed(shadowed)

	// Pairing integrity (H-SURFACE-008): a shadowed assistant message must not
	// orphan its tool result into the visible tail.
	if err := validatePairing(events, shadowed); err != nil {
		return nil, nil, err
	}

	// Second pass: build surface nodes from non-shadowed events. A durable
	// replacement is inserted where its first source surface event lived;
	// source seqs are not treated as a numeric range.
	type surfaceNode struct {
		seq      uint64
		messages []*llm.Message
	}
	nodes := make([]surfaceNode, 0)
	for _, event := range events {
		if !event.Type.Surface() {
			continue
		}
		extra, err := projectSurfaceEvents(event)
		if err != nil {
			return nil, nil, err
		}
		nodes = append(nodes, surfaceNode{seq: event.Seq, messages: extra})
	}
	for _, replacement := range replacements {
		selected := make(map[uint64]bool, len(replacement.sourceSeqs))
		for _, seq := range replacement.sourceSeqs {
			selected[seq] = true
		}
		insertAt := -1
		for index, node := range nodes {
			if selected[node.seq] {
				insertAt = index
				break
			}
		}
		if insertAt < 0 {
			return nil, nil, fmt.Errorf("session: compaction replacement has no visible source surface")
		}
		folded := make([]surfaceNode, 0, len(nodes)-len(selected)+1)
		inserted := false
		for _, node := range nodes {
			if selected[node.seq] {
				if !inserted {
					folded = append(folded, surfaceNode{messages: []*llm.Message{llm.NewUserMessage(replacement.summary)}})
					inserted = true
				}
				continue
			}
			folded = append(folded, node)
		}
		nodes = folded
	}
	var messages []*llm.Message
	for _, node := range nodes {
		if node.seq != 0 && shadowed[node.seq] {
			continue
		}
		messages = append(messages, node.messages...)
	}

	if s.spec.SystemSnapshot != "" {
		messages = append([]*llm.Message{llm.NewTextMessage(llm.RoleSystem, s.spec.SystemSnapshot)}, messages...)
	}
	if legacySummary && len(replacements) == 0 && proj.Summary != "" {
		messages = append([]*llm.Message{llm.NewUserMessage(proj.Summary)}, messages...)
	}
	return messages, proj, nil
}

// ReplaceSurfaceFold applies an in-memory compaction replacement to a fresh
// Projection. Kept for compatibility with callers that pass CompactedRegion.
// Never mutates prior events.

func projectSurfaceEvent(event Event) (*llm.Message, error) {
	switch event.Type {
	case EventUserMessage, EventSteeringMessage, EventInjectedContext:
		var payload struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, fmt.Errorf("session: decode user message: %w", err)
		}
		return llm.NewUserMessage(payload.Text), nil

	case EventAssistantMessage:
		var payload struct {
			Text      string         `json:"text"`
			Reasoning string         `json:"reasoning"`
			ToolCalls []ToolCall     `json:"tool_calls"`
			Blocks    []ContentBlock `json:"blocks"`
			Metadata  map[string]any `json:"metadata"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, fmt.Errorf("session: decode assistant message: %w", err)
		}
		if len(payload.Blocks) > 0 {
			parts := make([]llm.Part, 0, len(payload.Blocks))
			for _, block := range payload.Blocks {
				switch block.Kind {
				case BlockText:
					parts = append(parts, llm.Part{Type: llm.PartText, Text: block.Text})
				case BlockReasoning:
					parts = append(parts, llm.Part{Type: llm.PartReasoning, Reasoning: block.Text})
				case BlockMedia:
					if block.Media != nil {
						data, _ := base64.StdEncoding.DecodeString(block.Media.Data)
						parts = append(parts, llm.Part{Type: llm.PartMedia, Media: &llm.MediaContent{MediaType: block.Media.MediaType, URL: block.Media.URL, Data: data}})
					}
				case BlockToolCall:
					if block.ToolCall != nil {
						parts = append(parts, llm.Part{Type: llm.PartToolCall, ToolCall: &llm.ToolCallRequest{CallID: block.ToolCall.CallID, Name: block.ToolCall.Name, Arguments: append(json.RawMessage(nil), block.ToolCall.Arguments...)}})
					}
				case BlockToolResult:
					if block.ToolResult != nil {
						parts = append(parts, llm.Part{Type: llm.PartToolResult, ToolResult: &llm.ToolCallResult{CallID: block.ToolResult.CallID, Name: block.ToolResult.Name, Output: append(json.RawMessage(nil), block.ToolResult.Content...), IsError: block.ToolResult.Error != ""}})
					}
				case BlockExtension:
					custom, _ := json.Marshal(block)
					parts = append(parts, llm.Part{Type: llm.PartText, Custom: custom})
				}
			}
			return &llm.Message{Role: llm.RoleAssistant, Parts: parts, Metadata: payload.Metadata}, nil
		}
		calls := make([]llm.ToolCallRequest, 0, len(payload.ToolCalls))
		for _, call := range payload.ToolCalls {
			calls = append(calls, llm.ToolCallRequest{
				CallID:    call.CallID,
				Name:      call.Name,
				Arguments: append(json.RawMessage(nil), call.Arguments...),
			})
		}
		message := llm.NewAssistantMessage(payload.Text, payload.Reasoning, calls)
		message.Metadata = payload.Metadata
		return message, nil

	case EventToolResult:
		var payload struct {
			CallID  string          `json:"call_id"`
			Name    string          `json:"name"`
			Output  json.RawMessage `json:"output"`
			Content json.RawMessage `json:"content"`
			IsError bool            `json:"is_error"`
			Blocks  []ContentBlock  `json:"blocks"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, fmt.Errorf("session: decode tool result: %w", err)
		}
		modelOutput := payload.Content
		if len(modelOutput) == 0 {
			modelOutput = payload.Output
		}
		if len(modelOutput) == 0 {
			modelOutput = json.RawMessage(`null`)
		}
		return llm.NewToolResultMessage(llm.ToolCallResult{
			CallID:  payload.CallID,
			Name:    payload.Name,
			Output:  append(json.RawMessage(nil), modelOutput...),
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

func projectSurfaceEvents(event Event) ([]*llm.Message, error) {
	message, err := projectSurfaceEvent(event)
	if err != nil {
		return nil, err
	}
	if message == nil {
		return nil, nil
	}
	out := []*llm.Message{message}
	if event.Type == EventToolResult {
		var payload struct {
			AdditionalContexts []llm.Message `json:"additional_contexts"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, fmt.Errorf("session: decode tool additional contexts: %w", err)
		}
		for i := range payload.AdditionalContexts {
			clone := payload.AdditionalContexts[i].Clone()
			if clone != nil {
				out = append(out, clone)
			}
		}
	}
	return out, nil
}

type compactionSummaryPayload struct {
	Generation    uint64   `json:"generation"`
	TransactionID string   `json:"transaction_id"`
	ThroughSeq    uint64   `json:"through_seq"`
	ShadowedSeqs  []uint64 `json:"shadowed_seqs"`
	SourceSeqs    []uint64 `json:"source_seqs"`
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

func validateCompactionSourceSeqs(seqs []uint64) error {
	if len(seqs) == 0 {
		return fmt.Errorf("source_seqs must not be empty")
	}
	seen := make(map[uint64]bool, len(seqs))
	for _, seq := range seqs {
		if seq == 0 || seen[seq] {
			return fmt.Errorf("source_seqs must contain unique positive sequences")
		}
		seen[seq] = true
	}
	return nil
}

func sameSeqs(left, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

// validatePairing ensures no visible tool/result is orphaned by shadowing:
// when a result survives but its originating assistant message (or tool/call
// record) was shadowed, the pairing is broken and the log/projection is
// inconsistent (H-SURFACE-008).
func validatePairing(events []Event, shadowed map[uint64]bool) error {
	assistantByCall := make(map[string]uint64)
	resultByCall := make(map[string]uint64)
	for _, event := range events {
		if event.Type == EventToolResult && event.CallID != "" {
			resultByCall[event.CallID] = event.Seq
		}
	}
	for _, event := range events {
		if event.Type != EventAssistantMessage {
			continue
		}
		var payload struct {
			ToolCalls []ToolCall `json:"tool_calls"`
		}
		if json.Unmarshal(event.Data, &payload) != nil {
			continue
		}
		for _, call := range payload.ToolCalls {
			if call.CallID != "" {
				assistantByCall[call.CallID] = event.Seq
				if result, ok := resultByCall[call.CallID]; ok && !shadowed[event.Seq] && shadowed[result] {
					return fmt.Errorf("session: compaction shadowed seq %d while assistant tool-call seq %d remains visible: call/result pairing would break", result, event.Seq)
				}
			}
		}
	}
	for _, event := range events {
		if event.Type != EventToolResult || shadowed[event.Seq] {
			continue
		}
		sources := event.SourceSeqs
		if len(sources) == 0 && event.CallID != "" {
			if source, ok := assistantByCall[event.CallID]; ok {
				sources = []uint64{source}
			}
		}
		for _, source := range sources {
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
