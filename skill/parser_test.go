package skill

import (
	"errors"
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		input     string
		wantModel bool
		wantUser  bool
		wantError error
	}{
		{
			name: "defaults both invocation modes",
			input: `---
name: web-design-taste
description: Design direction
metadata:
  version: "1"
---
Use a deliberate visual hierarchy.`,
			wantModel: true,
			wantUser:  true,
		},
		{
			name: "independent invocation controls",
			input: `---
name: user-only
description: User-only instructions
disable-model-invocation: true
user-invocable: true
---
Follow the requested workflow.`,
			wantModel: false,
			wantUser:  true,
		},
		{
			name:      "missing frontmatter",
			input:     "plain markdown",
			wantError: ErrInvalidSkill,
		},
		{
			name: "invalid name",
			input: `---
name: Bad_Name
description: Invalid
---
Body`,
			wantError: ErrInvalidSkill,
		},
		{
			name: "rejects legacy invocation spelling",
			input: `---
name: legacy-policy
description: Legacy policy
disableModelInvocation: true
---
Body`,
			wantError: ErrInvalidSkill,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			definition, err := Parse([]byte(test.input), ParseOptions{Provider: "test", Source: "bundled"})
			if test.wantError != nil {
				if !errors.Is(err, test.wantError) {
					t.Fatalf("Parse() error = %v, want %v", err, test.wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("Parse() error = %v", err)
			}
			if definition.Policy.Model != test.wantModel || definition.Policy.User != test.wantUser {
				t.Fatalf("policy = %+v, want model=%t user=%t", definition.Policy, test.wantModel, test.wantUser)
			}
			if !isSHA256(definition.ContentHash) {
				t.Fatalf("content hash = %q", definition.ContentHash)
			}
		})
	}
}

func TestParseAgentSkillsFieldLimits(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name  string
		field string
	}{
		{name: "name", field: "a" + strings.Repeat("-a", MaxSkillNameLength/2)},
		{name: "description", field: strings.Repeat("d", MaxSkillDescriptionLength+1)},
		{name: "compatibility", field: strings.Repeat("c", MaxSkillCompatibilityLength+1)},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			name, description, compatibility := "valid-name", "description", ""
			switch test.name {
			case "name":
				name = test.field
			case "description":
				description = test.field
			case "compatibility":
				compatibility = test.field
			}
			raw := "---\nname: " + name + "\ndescription: " + description + "\ncompatibility: " + compatibility + "\n---\nBody"
			if _, err := Parse([]byte(raw), ParseOptions{Provider: "test", Source: "test"}); !errors.Is(err, ErrInvalidSkill) {
				t.Fatalf("Parse() error = %v, want ErrInvalidSkill", err)
			}
		})
	}
}

func TestParseRejectsDuplicateAndUnpinnedURLResources(t *testing.T) {
	t.Parallel()
	for _, resources := range []string{
		"  - name: hero\n    url: https://cdn.example/hero.png",
		"  - name: hero\n  - name: hero",
	} {
		raw := "---\nname: resource-skill\ndescription: Resources\nresources:\n" + resources + "\n---\nBody"
		if _, err := Parse([]byte(raw), ParseOptions{Provider: "test", Source: "test"}); !errors.Is(err, ErrInvalidResource) {
			t.Fatalf("Parse() error = %v, want ErrInvalidResource", err)
		}
	}
}

func TestParseContentHashPinsFrontmatterAndNormalizesLineEndings(t *testing.T) {
	t.Parallel()
	base := "---\nname: pinned\ndescription: First description\n---\nBody\n"
	windows := strings.ReplaceAll(base, "\n", "\r\n")
	first, err := Parse([]byte(base), ParseOptions{Provider: "test", Source: "test"})
	if err != nil {
		t.Fatal(err)
	}
	same, err := Parse([]byte(windows), ParseOptions{Provider: "test", Source: "test"})
	if err != nil {
		t.Fatal(err)
	}
	changed, err := Parse([]byte(strings.Replace(base, "First description", "Second description", 1)), ParseOptions{Provider: "test", Source: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if first.ContentHash != same.ContentHash {
		t.Fatalf("line ending normalization changed hash: %s != %s", first.ContentHash, same.ContentHash)
	}
	if first.ContentHash == changed.ContentHash {
		t.Fatal("frontmatter change did not change content hash")
	}
}
