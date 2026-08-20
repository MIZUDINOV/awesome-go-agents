package context

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Section is one ordered block of provider-facing context.
type Section struct {
	Title   string
	Content string
}

// BuildInput is the immutable input set for one provider request. Category
// order is fixed by Builder: system, instructions, tool guidance, runtime and
// workspace, followed by derived history supplied in Messages.
type BuildInput struct {
	Model         string
	System        []Section
	Instructions  []Section
	ToolGuidance  []Section
	Runtime       []Section
	Workspace     []Section
	Tools         []*llm.ToolDefinition
	Messages      []*llm.Message
	Config        json.RawMessage
	MaxTokens     int64
	Stream        bool
	Capabilities  *llm.Capabilities
	ParallelTools bool
}

// Snapshot contains the exact request and hashes persisted in a request
// header. The request and all mutable nested values are defensive copies.
type Snapshot struct {
	Request       llm.Request
	System        []Section
	TokenEstimate int64
	PromptHash    string
	ToolsHash     string
	RequestHash   string
}

type Builder struct{}

func NewBuilder() *Builder { return &Builder{} }

func (b *Builder) Build(input BuildInput) (Snapshot, error) {
	sections := make([]Section, 0, len(input.System)+len(input.Instructions)+len(input.ToolGuidance)+len(input.Runtime)+len(input.Workspace))
	sections = append(sections, cloneSections(input.System)...)
	sections = append(sections, cloneSections(input.Instructions)...)
	sections = append(sections, cloneSections(input.ToolGuidance)...)
	sections = append(sections, cloneSections(input.Runtime)...)
	sections = append(sections, cloneSections(input.Workspace)...)
	var prompt strings.Builder
	for _, section := range sections {
		content := strings.TrimSpace(section.Content)
		if content == "" {
			continue
		}
		if section.Title != "" {
			prompt.WriteString("## ")
			prompt.WriteString(section.Title)
			prompt.WriteByte('\n')
		}
		prompt.WriteString(content)
		prompt.WriteString("\n\n")
	}
	system := strings.TrimSpace(prompt.String())
	request := llm.Request{Model: input.Model, System: []llm.Message{{Role: llm.RoleSystem, Parts: []llm.Part{{Type: llm.PartText, Text: system}}}}, Messages: cloneMessages(input.Messages), Tools: cloneTools(input.Tools), MaxTokens: input.MaxTokens, Config: append(json.RawMessage(nil), input.Config...), Stream: input.Stream, Capabilities: cloneCapabilities(input.Capabilities)}
	if input.ParallelTools {
		v := true
		request.ParallelToolCalls = &v
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return Snapshot{}, fmt.Errorf("context: encode request snapshot: %w", err)
	}
	promptHash := fmt.Sprintf("%x", sha256.Sum256([]byte(system)))
	toolsJSON, _ := json.Marshal(request.Tools)
	toolsHash := fmt.Sprintf("%x", sha256.Sum256(toolsJSON))
	return Snapshot{Request: request, System: sections, TokenEstimate: estimateRequest(request), PromptHash: promptHash, ToolsHash: toolsHash, RequestHash: fmt.Sprintf("%x", sha256.Sum256(encoded))}, nil
}

func cloneSections(in []Section) []Section {
	out := make([]Section, len(in))
	copy(out, in)
	return out
}
func cloneCapabilities(in *llm.Capabilities) *llm.Capabilities {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}

func cloneTools(in []*llm.ToolDefinition) []*llm.ToolDefinition {
	out := make([]*llm.ToolDefinition, len(in))
	for i, tool := range in {
		if tool != nil {
			clone := *tool
			clone.InputSchema = append(json.RawMessage(nil), tool.InputSchema...)
			out[i] = &clone
		}
	}
	return out
}

func cloneMessages(in []*llm.Message) []llm.Message {
	out := make([]llm.Message, 0, len(in))
	for _, message := range in {
		if message != nil {
			if clone := message.Clone(); clone != nil {
				out = append(out, *clone)
			}
		}
	}
	return out
}

func estimateRequest(request llm.Request) int64 {
	data, _ := json.Marshal(request)
	return int64(len(data) / 4)
}
