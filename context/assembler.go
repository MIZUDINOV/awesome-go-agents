package context

import (
	"fmt"
	"strings"

	"github.com/wzhooh/agentkit/llm"
)

// Section is one ordered block of the system prompt.
type Section struct {
	// Title is a human/marker visible heading (optional).
	Title string
	// Content is the section body text.
	Content string
}

// Assembly is the fully-assembled context: the system prompt text, the
// model-facing tools, and the history to pass to the provider.
type Assembly struct {
	// System is the assembled system prompt (sections joined).
	System string
	// Tools are the model-facing tool schemas.
	Tools []*llm.ToolDefinition
	// Messages is the conversation history (from session projection).
	Messages []*llm.Message
	// Sections retains the raw parts for telemetry/debugging.
	Sections []Section
}

// Assembler builds the system prompt and final request from ordered sections.
type Assembler struct {
	// DefaultPersona is prepended before all sections when non-empty.
	DefaultPersona string
	// SectionProvider appends dynamic sections each assembly.
	SectionProvider func() []Section
}

// NewAssembler returns an Assembler with a provider that yields no dynamic
// sections.
func NewAssembler() *Assembler {
	return &Assembler{}
}

// WithDefaultPersona sets the leading identity line.
func (a *Assembler) WithDefaultPersona(persona string) *Assembler {
	a.DefaultPersona = persona
	return a
}

// WithSections sets a dynamic section provider.
func (a *Assembler) WithSections(provider func() []Section) *Assembler {
	a.SectionProvider = provider
	return a
}

// Assemble renders the system prompt and packages tools + history.
func (a *Assembler) Assemble(tools []*llm.ToolDefinition, messages []*llm.Message) (*Assembly, error) {
	var sections []Section
	if a.DefaultPersona != "" {
		sections = append(sections, Section{Title: "identity", Content: a.DefaultPersona})
	}
	if a.SectionProvider != nil {
		sections = append(sections, a.SectionProvider()...)
	}
	system := a.render(sections)
	return &Assembly{
		System:   system,
		Tools:    tools,
		Messages: messages,
		Sections: sections,
	}, nil
}

func (a *Assembler) render(sections []Section) string {
	var builder strings.Builder
	for _, section := range sections {
		content := strings.TrimSpace(section.Content)
		if content == "" {
			continue
		}
		if section.Title != "" {
			builder.WriteString("## ")
			builder.WriteString(section.Title)
			builder.WriteString("\n")
		}
		builder.WriteString(content)
		builder.WriteString("\n\n")
	}
	return strings.TrimSpace(builder.String())
}

// RenderSection is a helper for callers building Section slices.
func RenderSection(title, content string) Section { return Section{Title: title, Content: content} }

// ensure fmt is used (deferred helpers may use it).
var _ = fmt.Sprintf
