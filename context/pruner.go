package context

import (
	"strings"

	"github.com/wzhooh/agentkit/llm"
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

// PruneResult truncates a single tool result string in place, inserting a
// mid-marker. Returns the possibly-shortened string.
func (c PruneCaps) PruneResult(text string) string {
	if len(text) <= c.ThresholdChars {
		return text
	}
	head := text[:min(c.HeadChars, len(text))]
	tail := text[len(text)-min(c.TailChars, len(text)):]
	return head + "\n... pruned ...\n" + tail
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
}

// Spill caps inline output and returns a SpillRef when the result exceeds the
// maxInlineBytes.
func Spill(maxInlineBytes int, previewTail int, full string, artifact string) (string, *SpillRef) {
	if len(full) <= maxInlineBytes {
		return full, nil
	}
	preview := full
	if previewTail > 0 && len(preview) > previewTail {
		preview = preview[len(preview)-previewTail:]
	}
	return preview, &SpillRef{Artifact: artifact, Preview: preview}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var _ = strings.TrimSpace
