package session

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// BlockKind is the provider-neutral content vocabulary shared by durable
// assistant messages, streamed chunks and tool results.
type BlockKind string

const (
	BlockText       BlockKind = "text"
	BlockReasoning  BlockKind = "reasoning"
	BlockMedia      BlockKind = "media"
	BlockToolCall   BlockKind = "tool_call"
	BlockToolResult BlockKind = "tool_result"
	BlockExtension  BlockKind = "extension"
)

type MediaBlock struct {
	MediaType string `json:"media_type"`
	URL       string `json:"url,omitempty"`
	Data      string `json:"data,omitempty"`
}

type ToolCallBlock struct {
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type ToolResultBlock struct {
	CallID  string          `json:"call_id"`
	Name    string          `json:"name,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
	Code    string          `json:"code,omitempty"`
	Error   string          `json:"error,omitempty"`
}

// ContentBlock is a tagged union. Only fields matching Kind are meaningful;
// MarshalJSON rejects malformed unions so an invalid block never reaches a
// provider or a durable event.
type ContentBlock struct {
	Kind       BlockKind
	Text       string
	Media      *MediaBlock
	ToolCall   *ToolCallBlock
	ToolResult *ToolResultBlock
	Namespace  string
	Name       string
	Payload    json.RawMessage
}

func (b ContentBlock) Validate() error {
	switch b.Kind {
	case BlockText, BlockReasoning:
		if b.Media != nil || b.ToolCall != nil || b.ToolResult != nil || b.Namespace != "" || b.Name != "" || len(b.Payload) > 0 {
			return fmt.Errorf("session: %s block has incompatible fields", b.Kind)
		}
	case BlockMedia:
		if b.Media == nil || b.Media.MediaType == "" || b.Text != "" || b.ToolCall != nil || b.ToolResult != nil || b.Namespace != "" || b.Name != "" || len(b.Payload) > 0 {
			return fmt.Errorf("session: media block requires media_type")
		}
		if b.Media.Data != "" {
			if _, err := base64.StdEncoding.DecodeString(b.Media.Data); err != nil {
				return fmt.Errorf("session: media block data is not base64: %w", err)
			}
		}
	case BlockToolCall:
		if b.ToolCall == nil || b.ToolCall.CallID == "" || b.ToolCall.Name == "" || !json.Valid(b.ToolCall.Arguments) || b.Text != "" || b.Media != nil || b.ToolResult != nil || b.Namespace != "" || b.Name != "" || len(b.Payload) > 0 {
			return fmt.Errorf("session: malformed tool_call block")
		}
	case BlockToolResult:
		if b.ToolResult == nil || b.ToolResult.CallID == "" || b.Text != "" || b.Media != nil || b.ToolCall != nil || b.Namespace != "" || b.Name != "" || len(b.Payload) > 0 {
			return fmt.Errorf("session: malformed tool_result block")
		}
		if len(b.ToolResult.Content) > 0 && !json.Valid(b.ToolResult.Content) {
			return fmt.Errorf("session: tool_result content is not valid JSON")
		}
	case BlockExtension:
		if b.Namespace == "" || b.Name == "" || b.Namespace == "unknown" || strings.Contains(b.Namespace, "/") || strings.Contains(b.Name, "/") || len(b.Payload) == 0 || !json.Valid(b.Payload) || b.Text != "" || b.Media != nil || b.ToolCall != nil || b.ToolResult != nil {
			return fmt.Errorf("session: malformed extension block")
		}
		switch b.Namespace {
		case "request", "turn", "step", "user", "steering", "context", "inbox", "approval", "assistant", "tool", "compaction", "usage":
			return fmt.Errorf("session: extension block namespace %q is reserved", b.Namespace)
		}
	default:
		return fmt.Errorf("session: unknown content block kind %q", b.Kind)
	}
	return nil
}

func (b ContentBlock) MarshalJSON() ([]byte, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	m := map[string]any{"kind": b.Kind}
	switch b.Kind {
	case BlockText, BlockReasoning:
		m["content"] = b.Text
	case BlockMedia:
		m["media"] = b.Media
	case BlockToolCall:
		m["tool_call"] = b.ToolCall
	case BlockToolResult:
		m["tool_result"] = b.ToolResult
	case BlockExtension:
		m["namespace"], m["name"], m["payload"] = b.Namespace, b.Name, b.Payload
	}
	return json.Marshal(m)
}

func (b *ContentBlock) UnmarshalJSON(data []byte) error {
	var raw struct {
		Kind       BlockKind        `json:"kind"`
		Content    string           `json:"content"`
		Media      *MediaBlock      `json:"media"`
		ToolCall   *ToolCallBlock   `json:"tool_call"`
		ToolResult *ToolResultBlock `json:"tool_result"`
		Namespace  string           `json:"namespace"`
		Name       string           `json:"name"`
		Payload    json.RawMessage  `json:"payload"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*b = ContentBlock{Kind: raw.Kind, Text: raw.Content, Media: raw.Media, ToolCall: raw.ToolCall, ToolResult: raw.ToolResult, Namespace: raw.Namespace, Name: raw.Name, Payload: append(json.RawMessage(nil), raw.Payload...)}
	return b.Validate()
}

func TextBlock(text string) ContentBlock      { return ContentBlock{Kind: BlockText, Text: text} }
func ReasoningBlock(text string) ContentBlock { return ContentBlock{Kind: BlockReasoning, Text: text} }
func MediaContentBlock(media MediaBlock) ContentBlock {
	return ContentBlock{Kind: BlockMedia, Media: &media}
}

func MarshalBlocks(blocks []ContentBlock) (json.RawMessage, error) {
	if blocks == nil {
		blocks = []ContentBlock{}
	}
	for _, block := range blocks {
		if err := block.Validate(); err != nil {
			return nil, err
		}
	}
	encoded, err := json.Marshal(blocks)
	return encoded, err
}

func UnmarshalBlocks(data []byte) ([]ContentBlock, error) {
	if !json.Valid(data) {
		return nil, fmt.Errorf("session: invalid blocks JSON")
	}
	if trimmed := bytes.TrimSpace(data); len(trimmed) == 0 || trimmed[0] != '[' {
		return nil, fmt.Errorf("session: blocks must be a JSON array")
	}
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err != nil {
		return nil, err
	}
	return blocks, nil
}

// EqualBlocks is useful for replay/golden tests and deliberately compares the
// canonical JSON representation, not pointer identity.
func EqualBlocks(a, b []ContentBlock) bool {
	left, errLeft := MarshalBlocks(a)
	right, errRight := MarshalBlocks(b)
	return errLeft == nil && errRight == nil && bytes.Equal(left, right)
}
