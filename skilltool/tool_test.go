package skilltool

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/skill"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

type resourceResolverFunc func(context.Context, skill.Definition, skill.ResourceRef) (skill.Resource, error)

func (function resourceResolverFunc) Resolve(ctx context.Context, definition skill.Definition, reference skill.ResourceRef) (skill.Resource, error) {
	return function(ctx, definition, reference)
}

func TestSkillToolRendersPinnedDefinitionWithoutLocalPaths(t *testing.T) {
	body := "Use a restrained visual hierarchy."
	digest := sha256.Sum256([]byte(body))
	hash := hex.EncodeToString(digest[:])
	definition := skill.Definition{
		Summary: skill.Summary{Name: "web-design-taste", Description: "Design direction", Policy: skill.InvocationPolicy{Model: true, User: true}, Provider: skill.RuntimeProviderName, Source: "embedded-test", ResourceBase: &skill.ResourceBase{Kind: skill.ResourceDirectory, Path: `C:\secret\skills`}},
		Version: "v1", ContentHash: hash, Content: body,
		ResourceManifest: []skill.ResourceRef{{Name: "hero", MediaType: "image/webp", Description: "Editorial reference"}},
	}
	registry := skill.NewRegistry(0)
	if _, err := registry.Register("", definition, 100); err != nil {
		t.Fatal(err)
	}
	runtime, err := skill.NewRuntime(registry, skill.RuntimeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Initialize(context.Background()); err != nil {
		t.Fatal(err)
	}
	toolRegistry := tools.New(tools.Options{MaxParallel: 2})
	registration, err := RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	result, err := toolRegistry.Run(context.Background(), tools.ToolRunContext{}, "skill", "call-1", json.RawMessage(`{"name":"web-design-taste"}`))
	if err != nil {
		t.Fatal(err)
	}
	text, ok := result.ModelFacing.(string)
	if !ok || !strings.Contains(text, `<skill_content name="web-design-taste">`) || !strings.Contains(text, "<skill_resources>") {
		t.Fatalf("unexpected model result: %#v", result.ModelFacing)
	}
	if strings.Contains(text, `C:\secret`) {
		t.Fatalf("local path leaked: %s", text)
	}
}

func TestSkillToolRechecksPolicyAndNoDoubleLoad(t *testing.T) {
	body := "Only for direct user invocation."
	digest := sha256.Sum256([]byte(body))
	definition := skill.Definition{
		Summary: skill.Summary{Name: "user-only", Description: "User only", Policy: skill.InvocationPolicy{User: true}, Provider: skill.RuntimeProviderName, Source: "embedded-test"},
		Version: "v1", ContentHash: hex.EncodeToString(digest[:]), Content: body,
	}
	registry := skill.NewRegistry(0)
	if _, err := registry.Register("", definition, 100); err != nil {
		t.Fatal(err)
	}
	runtime, _ := skill.NewRuntime(registry, skill.RuntimeOptions{})
	if err := runtime.Initialize(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.GetModel(context.Background(), "user-only"); !strings.Contains(err.Error(), "invocation denied") {
		t.Fatalf("expected policy denial, got %v", err)
	}
	runtime.MarkUserLoaded("user-only")
	if _, err := runtime.GetModel(context.Background(), "user-only"); !strings.Contains(err.Error(), "already loaded") {
		t.Fatalf("expected duplicate denial, got %v", err)
	}
}

func TestSkillToolResolvesDeclaredMediaOnlyOnExplicitResourceRequest(t *testing.T) {
	body := "Compare against the declared reference."
	digest := sha256.Sum256([]byte(body))
	definition := skill.Definition{
		Summary: skill.Summary{Name: "visual-reference", Description: "Visual reference", Policy: skill.InvocationPolicy{Model: true}, Provider: skill.RuntimeProviderName, Source: "backend-test"},
		Version: "v1", ContentHash: hex.EncodeToString(digest[:]), Content: body,
		ResourceManifest: []skill.ResourceRef{{Name: "reference-image", SHA256: strings.Repeat("a", 64), MediaType: "image/png"}},
	}
	registry := skill.NewRegistry(0)
	if _, err := registry.Register("", definition, 100); err != nil {
		t.Fatal(err)
	}
	resolved := 0
	runtime, err := skill.NewRuntime(registry, skill.RuntimeOptions{Resources: resourceResolverFunc(func(context.Context, skill.Definition, skill.ResourceRef) (skill.Resource, error) {
		resolved++
		return skill.Resource{Name: "reference-image", MediaType: "image/png", Data: []byte{1, 2, 3}, SHA256: strings.Repeat("a", 64)}, nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Initialize(context.Background()); err != nil {
		t.Fatal(err)
	}
	toolRegistry := tools.New(tools.Options{})
	if _, err := RegisterSkillTool(toolRegistry, runtime); err != nil {
		t.Fatal(err)
	}
	activation, err := toolRegistry.Run(context.Background(), tools.ToolRunContext{Vars: map[string]any{"supports_media": true}}, "skill", "activation", json.RawMessage(`{"name":"visual-reference"}`))
	if err != nil {
		t.Fatal(err)
	}
	if resolved != 0 || len(activation.Content) != 1 {
		t.Fatalf("activation eagerly resolved media: resolved=%d blocks=%d", resolved, len(activation.Content))
	}
	runtime.MarkLoaded("visual-reference")
	withMedia, err := toolRegistry.Run(context.Background(), tools.ToolRunContext{Vars: map[string]any{"supports_media": true}}, "skill", "with", json.RawMessage(`{"name":"visual-reference","resource":"reference-image"}`))
	if err != nil {
		t.Fatal(err)
	}
	if resolved != 1 || len(withMedia.Content) != 2 || withMedia.Content[1].Kind != session.BlockMedia {
		t.Fatalf("unexpected media result: resolved=%d content=%+v", resolved, withMedia.Content)
	}
}
