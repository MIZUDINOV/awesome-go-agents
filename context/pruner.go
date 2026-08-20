package context

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// PruneCaps bound how a single tool result is truncated deterministically
// before summarization (the cheaper layer in front of LLM compaction).
type PruneCaps struct {
	// ThresholdChars disables pruning below this size.
	ThresholdChars int
	// HeadChars / TailChars keep the head and tail of an oversized result.
	HeadChars int
	TailChars int
}

// DefaultPruneCaps mirrors the DSH base bundle (8KiB threshold, 4KiB head,
// 1KiB tail).
func DefaultPruneCaps() PruneCaps {
	return PruneCaps{ThresholdChars: 8192, HeadChars: 4096, TailChars: 1024}
}

// PruneResult truncates a single tool result string, inserting a mid-marker.
// Truncation is Unicode-safe: caps are measured in Unicode code points, so a
// multi-byte rune is never split (H-PRUNE-004). Returns the possibly-shortened
// string without mutating the input.
func (c PruneCaps) PruneResult(text string) string {
	runes := []rune(text)
	if len(runes) <= c.ThresholdChars {
		return text
	}
	headCount := clamp(c.HeadChars, 0, len(runes))
	tailCount := clamp(c.TailChars, 0, len(runes))
	if headCount+tailCount >= len(runes) {
		// Nothing meaningful can be removed; keep the original intact rather
		// than returning something longer than the source.
		return text
	}
	head := string(runes[:headCount])
	tail := string(runes[len(runes)-tailCount:])
	const marker = "\n... pruned ...\n"
	return head + marker + tail
}

// PruneMessages rewrites oversized tool results across a message slice,
// returning how many results were truncated.
func (c PruneCaps) PruneMessages(messages []*llm.Message) int {
	pruned := 0
	for _, msg := range messages {
		if msg == nil || msg.Role != llm.RoleTool {
			continue
		}
		for i := range msg.Parts {
			part := &msg.Parts[i]
			if part.ToolResult == nil || part.ToolResult.IsError {
				continue
			}
			original := string(part.ToolResult.Output)
			shortened := c.PruneResult(original)
			if shortened != original {
				part.ToolResult.Output = []byte(shortened)
				pruned++
			}
		}
	}
	return pruned
}

// SpillRef is a pointer to a complete artifact kept out of the model context
// (e.g. an object-store key) plus a bounded inline preview.
type SpillRef struct {
	Artifact string
	Preview  string
	Size     int64
	SHA256   string
}

// Spill caps inline output and returns a SpillRef when the result exceeds the
// maxInlineBytes. Both caps are measured in Unicode code points so multi-byte
// runes are never split; the preview is taken from the tail.
func Spill(maxInlineBytes int, previewTail int, full string, artifact string) (string, *SpillRef) {
	runes := []rune(full)
	if len(runes) <= maxInlineBytes {
		return full, nil
	}
	preview := full
	if previewTail > 0 && len(runes) > previewTail {
		preview = string(runes[len(runes)-previewTail:])
	}
	digest := sha256.Sum256([]byte(full))
	return preview, &SpillRef{Artifact: artifact, Preview: preview, Size: int64(len([]byte(full))), SHA256: hex.EncodeToString(digest[:])}
}

// String is a compact deterministic representation suitable for a model
// hint or durable metadata field.
func (r SpillRef) String() string {
	return r.Artifact + " (" + strconv.FormatInt(r.Size, 10) + " bytes, sha256=" + r.SHA256 + ")"
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
