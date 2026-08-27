// Package skilltool exposes the model-facing loader for AgentKit skills.
package skilltool

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/skill"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

type Input struct {
	Name     string `json:"name" jsonschema:"required,description=Exact kebab-case skill name from available_skills"`
	Resource string `json:"resource,omitempty" jsonschema:"description=Optional manifest name or relative resource path from an already loaded skill"`
}

type Output struct {
	Name        string           `json:"name" jsonschema:"required"`
	Resource    string           `json:"resource,omitempty"`
	Provider    string           `json:"provider,omitempty"`
	Source      string           `json:"source,omitempty"`
	Version     string           `json:"version,omitempty"`
	ContentHash string           `json:"content_hash,omitempty"`
	Content     string           `json:"content" jsonschema:"required"`
	Media       []skill.Resource `json:"-"`
	Activated   bool             `json:"-"`
}

// RegisterSkillTool installs a reversible model-facing skill loader.
func RegisterSkillTool(registry *tools.Registry, runtime *skill.Runtime) (*tools.Registration, error) {
	if registry == nil || runtime == nil {
		return nil, fmt.Errorf("skilltool: registry and runtime are required")
	}
	definition := tools.DefineTool(tools.DefineToolOptions[Input, Output]{
		Name:        "skill",
		Version:     "agentkit-skills-v2",
		Description: "Load one available skill by exact name, or lazily read one referenced resource from an already loaded skill.",
		ConcurrencySafe: func(Input) bool {
			return true
		},
		Execute: func(ctx context.Context, runContext tools.ToolRunContext, input Input) (Output, error) {
			if !skill.IsName(input.Name) {
				return Output{}, tools.NewFailureError("SKILL_INVALID_NAME", "skill name must be exact kebab-case", map[string]any{"name": input.Name})
			}
			if input.Resource != "" {
				return loadResource(ctx, runtime, runContext, input)
			}
			loaded, err := runtime.GetModel(ctx, input.Name)
			if err != nil {
				return Output{}, classifyError(input.Name, err)
			}
			output := Output{
				Name: loaded.Name, Provider: loaded.Provider, Source: loaded.Source, Version: loaded.Version,
				ContentHash: loaded.ContentHash, Content: Render(loaded), Activated: true,
			}
			return output, nil
		},
		RenderModel: func(output Output) (any, error) { return output.Content, nil },
		RenderContent: func(output Output) ([]session.ContentBlock, error) {
			blocks := []session.ContentBlock{session.TextBlock(output.Content)}
			for _, resource := range output.Media {
				blocks = append(blocks, session.MediaContentBlock(session.MediaBlock{MediaType: resource.MediaType, Data: base64.StdEncoding.EncodeToString(resource.Data)}))
			}
			return blocks, nil
		},
		PresentCall: func(input Input) (map[string]any, error) {
			result := map[string]any{"kind": "skill", "name": input.Name}
			if input.Resource != "" {
				result["resource"] = input.Resource
			}
			return result, nil
		},
		PresentResult: func(output Output) (map[string]any, error) {
			result := map[string]any{"kind": "skill", "name": output.Name, "content_hash": output.ContentHash}
			if output.Resource != "" {
				result["resource"] = output.Resource
			} else {
				result["version"] = output.Version
			}
			return result, nil
		},
	})
	return registry.RegisterTool(definition)
}

func loadResource(ctx context.Context, runtime *skill.Runtime, runContext tools.ToolRunContext, input Input) (Output, error) {
	resource, err := runtime.ResolveLoadedResource(ctx, input.Name, input.Resource)
	if err != nil {
		return Output{}, classifyResourceError(input.Name, input.Resource, err)
	}
	output := Output{Name: input.Name, Resource: input.Resource, ContentHash: resource.SHA256}
	if strings.HasPrefix(resource.MediaType, "text/") || resource.MediaType == "application/json" || resource.MediaType == "application/yaml" {
		output.Content = RenderResource(input.Name, input.Resource, resource.MediaType, string(resource.Data))
		return output, nil
	}
	if !supportsMedia(runContext.Vars) || resource.MediaType == "" || len(resource.Data) == 0 {
		return Output{}, tools.NewFailureError("SKILL_RESOURCE_MEDIA_UNSUPPORTED", "skill resource cannot be projected by this model", map[string]any{"name": input.Name, "resource": input.Resource})
	}
	output.Content = RenderResource(input.Name, input.Resource, resource.MediaType, "Media resource attached to this tool result.")
	output.Media = []skill.Resource{resource}
	return output, nil
}

func supportsMedia(values map[string]any) bool {
	value, _ := values["supports_media"].(bool)
	return value
}

// Render returns the canonical safe wrapper shared by model and explicit user
// invocation. Provider locators and local resource paths are intentionally absent.
func Render(definition skill.Definition) string {
	resources := make([]string, 0, len(definition.ResourceManifest))
	for _, resource := range definition.ResourceManifest {
		line := "- `" + escape(resource.Name) + "`"
		if resource.Description != "" {
			line += ": " + escape(resource.Description)
		}
		if resource.MediaType != "" {
			line += " (" + escape(resource.MediaType) + ")"
		}
		resources = append(resources, line)
	}
	if definition.ResourceBase != nil && definition.ResourceBase.Kind == skill.ResourceOpaque && definition.ResourceBase.Description != "" {
		resources = append(resources, escape(definition.ResourceBase.Description))
	}
	if len(resources) == 0 {
		resources = append(resources, "No named resources are declared for this skill.")
	}
	resources = append(resources, "Load only a referenced resource when needed by calling `skill` again with this skill name and `resource` set to its listed name or relative path.")
	return strings.Join([]string{
		`<skill_content name="` + escapeAttribute(definition.Name) + `">`,
		"<skill_resources>", strings.Join(resources, "\n"), "</skill_resources>",
		"",
		"<skill_instructions>",
		definition.Content,
		"</skill_instructions>",
		"</skill_content>",
	}, "\n")
}

// RenderResource safely wraps one lazily resolved skill resource.
func RenderResource(name, reference, mediaType, content string) string {
	return strings.Join([]string{
		`<skill_resource skill="` + escapeAttribute(name) + `" reference="` + escapeAttribute(reference) + `" media_type="` + escapeAttribute(mediaType) + `">`,
		content,
		"</skill_resource>",
	}, "\n")
}

func classifyError(name string, err error) error {
	code, message := "SKILL_LOAD_FAILED", "skill could not be loaded"
	switch {
	case errors.Is(err, skill.ErrSkillNotFound):
		code, message = "SKILL_NOT_FOUND", "skill is not available in this run"
	case errors.Is(err, skill.ErrPolicyDenied):
		code, message = "SKILL_POLICY_DENIED", "skill is not model-invocable"
	case errors.Is(err, skill.ErrPinnedMismatch), errors.Is(err, skill.ErrProviderDisposed):
		code, message = "SKILL_PIN_MISMATCH", "pinned skill content is unavailable"
	case errors.Is(err, skill.ErrAlreadyLoaded):
		code, message = "SKILL_ALREADY_LOADED", "skill was already injected by explicit user invocation"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		code, message = "ABORTED", "skill loading was cancelled"
	}
	return tools.NewFailureError(code, message, map[string]any{"name": name})
}

func classifyResourceError(name, resource string, err error) error {
	code, message := "SKILL_RESOURCE_LOAD_FAILED", "skill resource could not be loaded"
	switch {
	case errors.Is(err, skill.ErrSkillNotLoaded):
		code, message = "SKILL_NOT_LOADED", "load the skill before requesting one of its resources"
	case errors.Is(err, skill.ErrInvalidResource), errors.Is(err, skill.ErrSkillNotFound):
		code, message = "SKILL_RESOURCE_NOT_FOUND", "skill resource is not available"
	case errors.Is(err, skill.ErrPinnedMismatch), errors.Is(err, skill.ErrProviderDisposed):
		code, message = "SKILL_PIN_MISMATCH", "pinned skill resource is unavailable"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		code, message = "ABORTED", "skill resource loading was cancelled"
	}
	return tools.NewFailureError(code, message, map[string]any{"name": name, "resource": resource})
}

func escape(value string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", "`", "&#96;").Replace(value)
}

func escapeAttribute(value string) string {
	return strings.NewReplacer("&", "&amp;", `"`, "&quot;", "<", "&lt;", ">", "&gt;").Replace(value)
}
