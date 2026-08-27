package skill

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

type ParseOptions struct {
	Provider     string
	Source       string
	ResourceBase *ResourceBase
}

type frontmatter struct {
	Name                   string         `yaml:"name"`
	Description            string         `yaml:"description"`
	WhenToUse              string         `yaml:"whenToUse"`
	License                string         `yaml:"license"`
	Compatibility          string         `yaml:"compatibility"`
	Metadata               map[string]any `yaml:"metadata"`
	AllowedTools           any            `yaml:"allowed-tools"`
	DisableModelInvocation *bool          `yaml:"disable-model-invocation"`
	UserInvocable          *bool          `yaml:"user-invocable"`
	Resources              []ResourceRef  `yaml:"resources"`
}

func Parse(data []byte, options ParseOptions) (Definition, error) {
	normalized := bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	header, body, err := splitFrontmatter(data)
	if err != nil {
		return Definition{}, err
	}
	var metadata frontmatter
	decoder := yaml.NewDecoder(bytes.NewReader(header))
	if err := decoder.Decode(&metadata); err != nil {
		return Definition{}, fmt.Errorf("%w: invalid yaml frontmatter: %v", ErrInvalidSkill, err)
	}
	var rawFields map[string]any
	if err := yaml.Unmarshal(header, &rawFields); err != nil {
		return Definition{}, fmt.Errorf("%w: invalid yaml frontmatter: %v", ErrInvalidSkill, err)
	}
	for legacy, canonical := range map[string]string{
		"disableModelInvocation": "disable-model-invocation",
		"userInvocable":          "user-invocable",
	} {
		if _, exists := rawFields[legacy]; exists {
			return Definition{}, fmt.Errorf("%w: use %q instead of legacy %q", ErrInvalidSkill, canonical, legacy)
		}
	}
	metadata.Name = strings.TrimSpace(metadata.Name)
	metadata.Description = strings.TrimSpace(metadata.Description)
	body = []byte(strings.TrimSpace(string(body)))
	metadata.Compatibility = strings.TrimSpace(metadata.Compatibility)
	if !IsName(metadata.Name) || metadata.Description == "" || len(body) == 0 {
		return Definition{}, fmt.Errorf("%w: frontmatter requires a valid name and description plus a non-empty body", ErrInvalidSkill)
	}
	if utf8.RuneCountInString(metadata.Description) > MaxSkillDescriptionLength {
		return Definition{}, fmt.Errorf("%w: description exceeds %d characters", ErrInvalidSkill, MaxSkillDescriptionLength)
	}
	if utf8.RuneCountInString(metadata.Compatibility) > MaxSkillCompatibilityLength {
		return Definition{}, fmt.Errorf("%w: compatibility exceeds %d characters", ErrInvalidSkill, MaxSkillCompatibilityLength)
	}

	modelInvocable := true
	if metadata.DisableModelInvocation != nil {
		modelInvocable = !*metadata.DisableModelInvocation
	}
	userInvocable := true
	if metadata.UserInvocable != nil {
		userInvocable = *metadata.UserInvocable
	}

	// Pin the complete normalized SKILL.md contract. Description, invocation
	// policy and resource declarations are as operationally relevant as body.
	digest := sha256.Sum256(normalized)
	contentHash := hex.EncodeToString(digest[:])
	version := contentHash
	if value, ok := metadata.Metadata["version"].(string); ok && strings.TrimSpace(value) != "" {
		version = strings.TrimSpace(value)
	}
	parsedMetadata := cloneMap(metadata.Metadata)
	if metadata.License != "" {
		parsedMetadata["license"] = metadata.License
	}
	if metadata.Compatibility != "" {
		parsedMetadata["compatibility"] = metadata.Compatibility
	}
	if metadata.AllowedTools != nil {
		// Advisory only. AgentKit's tool policy remains the capability boundary.
		parsedMetadata["allowed-tools"] = metadata.AllowedTools
	}
	definition := Definition{
		Summary: Summary{
			Name:         metadata.Name,
			Description:  metadata.Description,
			WhenToUse:    strings.TrimSpace(metadata.WhenToUse),
			Policy:       InvocationPolicy{Model: modelInvocable, User: userInvocable},
			Provider:     options.Provider,
			Source:       options.Source,
			ResourceBase: cloneResourceBase(options.ResourceBase),
		},
		Version:          version,
		ContentHash:      contentHash,
		Content:          string(body),
		ResourceManifest: append([]ResourceRef{}, metadata.Resources...),
		Metadata:         parsedMetadata,
	}
	if err := validateDefinition(definition); err != nil {
		return Definition{}, err
	}
	return definition, nil
}

func splitFrontmatter(data []byte) ([]byte, []byte, error) {
	normalized := bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	if !bytes.HasPrefix(normalized, []byte("---\n")) {
		return nil, nil, fmt.Errorf("%w: missing yaml frontmatter", ErrInvalidSkill)
	}
	rest := normalized[len("---\n"):]
	end := bytes.Index(rest, []byte("\n---\n"))
	if end < 0 {
		return nil, nil, fmt.Errorf("%w: unterminated yaml frontmatter", ErrInvalidSkill)
	}
	return rest[:end], rest[end+len("\n---\n"):], nil
}

func cloneResourceBase(base *ResourceBase) *ResourceBase {
	if base == nil {
		return nil
	}
	out := *base
	return &out
}
