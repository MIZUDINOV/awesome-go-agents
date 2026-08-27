package skill

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestRuntimeCatalogAndPolicies(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	provider := mustMemoryProvider(t, "provider", 100,
		mustDefinition(t, "model", "  model   description  ", "provider", true, false),
		mustDefinition(t, "user", "user description", "provider", false, true),
	)
	mustRegisterProvider(t, registry, ProviderOptions{Name: "provider", Provider: provider})
	runtime, err := NewRuntime(registry, RuntimeOptions{Pinned: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Initialize(t.Context()); err != nil {
		t.Fatal(err)
	}
	catalog := runtime.Catalog()
	if len(catalog.Skills) != 1 || catalog.Skills[0].Name != "model" || catalog.Skills[0].Description != "model description" {
		t.Fatalf("catalog = %+v", catalog)
	}
	if _, err := runtime.GetModel(t.Context(), "user"); !errors.Is(err, ErrPolicyDenied) {
		t.Fatalf("GetModel(user) error = %v", err)
	}
	if _, err := runtime.GetUser(t.Context(), "user"); err != nil {
		t.Fatalf("GetUser(user) error = %v", err)
	}
	rendered := runtime.RenderCatalog(false)
	if !strings.Contains(rendered, "<available_skills>") || strings.Contains(rendered, "user description") {
		t.Fatalf("rendered catalog = %q", rendered)
	}
	if !strings.Contains(rendered, "Load all applicable skills") {
		t.Fatalf("catalog omits multi-skill activation guidance: %q", rendered)
	}
}

type resourcePathResolverFunc func(context.Context, Definition, string) (Resource, error)

func (function resourcePathResolverFunc) ResolvePath(ctx context.Context, definition Definition, reference string) (Resource, error) {
	return function(ctx, definition, reference)
}

func TestRuntimeResolvesPathOnlyAfterActivation(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	provider := mustMemoryProvider(t, "provider", 100, mustDefinition(t, "model", "model description", "provider", true, false))
	mustRegisterProvider(t, registry, ProviderOptions{Name: "provider", Provider: provider})
	resolved := 0
	runtime, err := NewRuntime(registry, RuntimeOptions{ResourcePaths: resourcePathResolverFunc(func(_ context.Context, _ Definition, reference string) (Resource, error) {
		resolved++
		return Resource{Name: reference, MediaType: "text/markdown", Data: []byte("details")}, nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Initialize(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.ResolveLoadedResource(t.Context(), "model", "rules/details.md"); !errors.Is(err, ErrSkillNotLoaded) {
		t.Fatalf("resource before activation error = %v", err)
	}
	if _, err := runtime.GetModel(t.Context(), "model"); err != nil {
		t.Fatal(err)
	}
	runtime.MarkLoaded("model")
	resource, err := runtime.ResolveLoadedResource(t.Context(), "model", "rules/details.md")
	if err != nil || string(resource.Data) != "details" || resolved != 1 {
		t.Fatalf("resolved resource = %+v, err=%v, calls=%d", resource, err, resolved)
	}
	for _, invalid := range []string{"", " rules/details.md", "rules/details.md\x00"} {
		if _, err := runtime.ResolveLoadedResource(t.Context(), "model", invalid); !errors.Is(err, ErrInvalidResource) {
			t.Errorf("invalid reference %q error = %v", invalid, err)
		}
	}
}

func TestRuntimePinnedSnapshotDoesNotRefresh(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	provider := mustMemoryProvider(t, "provider", 100, mustDefinition(t, "one", "one", "provider", true, true))
	registration := mustRegisterProvider(t, registry, ProviderOptions{Name: "provider", Provider: provider})
	runtime, err := NewRuntime(registry, RuntimeOptions{Pinned: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Initialize(t.Context()); err != nil {
		t.Fatal(err)
	}
	provider.Delete("one")
	registration.Invalidate("deleted")
	changed, err := runtime.Refresh(t.Context())
	if err != nil || changed {
		t.Fatalf("Refresh() = %t, %v", changed, err)
	}
	if len(runtime.Catalog().Skills) != 1 {
		t.Fatalf("pinned catalog changed: %+v", runtime.Catalog())
	}
}
