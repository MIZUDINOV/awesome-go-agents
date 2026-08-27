package skill

import (
	"strings"
	"testing"
	"testing/fstest"
)

func TestEmbeddedProviderListsRelativeResourcesWithoutHostLocator(t *testing.T) {
	t.Parallel()
	provider, err := NewEmbeddedProvider("embedded", fstest.MapFS{
		"builtin/design/SKILL.md":             {Data: []byte("---\nname: design\ndescription: Design guidance\n---\nRead the relevant rule.")},
		"builtin/design/rules/composition.md": {Data: []byte("Composition guidance")},
	}, EmbeddedRoot{Path: "builtin", Source: "built-in"})
	if err != nil {
		t.Fatal(err)
	}
	observation, err := provider.List(t.Context(), ListRequest{})
	if err != nil || len(observation.Candidates) != 1 {
		t.Fatalf("observation = %+v, err=%v", observation, err)
	}
	definition, err := provider.Get(t.Context(), observation.Candidates[0], Lookup{})
	if err != nil {
		t.Fatal(err)
	}
	description := definition.ResourceBase.Description
	if !strings.Contains(description, "rules/composition.md") || strings.Contains(description, "builtin/design") {
		t.Fatalf("resource description = %q", description)
	}
}

func TestEmbeddedProviderPinsBundledResourceContent(t *testing.T) {
	t.Parallel()
	source := fstest.MapFS{
		"builtin/design/SKILL.md":     {Data: []byte("---\nname: design\ndescription: Design guidance\n---\nRead the rule.")},
		"builtin/design/rules/one.md": {Data: []byte("first")},
	}
	provider, err := NewEmbeddedProvider("embedded", source, EmbeddedRoot{Path: "builtin", Source: "built-in"})
	if err != nil {
		t.Fatal(err)
	}
	before, err := provider.List(t.Context(), ListRequest{})
	if err != nil || len(before.Candidates) != 1 {
		t.Fatalf("before = %+v, err=%v", before, err)
	}
	source["builtin/design/rules/one.md"] = &fstest.MapFile{Data: []byte("second")}
	after, err := provider.Get(t.Context(), before.Candidates[0], Lookup{})
	if err != nil {
		t.Fatal(err)
	}
	if after.ContentHash == before.Candidates[0].ContentHash {
		t.Fatal("bundled resource change did not change pinned content hash")
	}
}
