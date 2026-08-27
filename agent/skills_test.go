package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/skill"
	"github.com/MIZUDINOV/awesome-go-agents/skilltool"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

func TestSkillsCatalogIsDurableStableAndPresentInRequestMetadata(t *testing.T) {
	runtime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	toolRegistry := tools.New(tools.Options{})
	registration, err := skilltool.RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	store := newMemoryStore()
	provider := &scriptedProvider{steps: []scriptedStep{{text: "first"}, {text: "second"}}}
	loop := NewLoop("skills-catalog", store, toolRegistry, provider, Config{Model: "m", Owner: "test", SystemPrompt: "sys", Skills: runtime})
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "design a page"); err != nil {
		t.Fatal(err)
	}
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "continue"); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "skills-catalog", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	catalogs := 0
	requestHeaders := 0
	for _, event := range events {
		switch event.Type {
		case session.EventSkillCatalog:
			catalogs++
		case session.EventRequestHeader:
			requestHeaders++
			var payload map[string]any
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"skill_catalog_hash", "skill_snapshot_hash", "skill_schema_version"} {
				if payload[key] == "" || payload[key] == nil {
					t.Fatalf("request header missing %s: %v", key, payload)
				}
			}
		}
	}
	if catalogs != 1 {
		t.Fatalf("catalog replacements = %d, want 1", catalogs)
	}
	if requestHeaders != 2 {
		t.Fatalf("request headers = %d, want 2", requestHeaders)
	}
	if len(provider.calls) == 0 || !requestContains(provider.calls[0], "<available_skills>") {
		encoded, _ := json.Marshal(provider.calls)
		t.Fatalf("model request did not contain the skill catalog: %s", encoded)
	}
}

func TestExplicitSkillInvocationIsInjectedWithoutRawCommand(t *testing.T) {
	runtime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	toolRegistry := tools.New(tools.Options{})
	registration, err := skilltool.RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	store := newMemoryStore()
	provider := &scriptedProvider{steps: []scriptedStep{{text: "done"}}}
	loop := NewLoop("skills-user", store, toolRegistry, provider, Config{Model: "m", Owner: "test", SystemPrompt: "sys", Skills: runtime})
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "/skill web-design-taste"); err != nil {
		t.Fatal(err)
	}
	events, _ := store.Load(context.Background(), "skills-user", 0, 0)
	invocations, rawCommands := 0, 0
	for _, event := range events {
		if event.Type == session.EventSkillInvocation {
			invocations++
		}
		if event.Type == session.EventUserMessage && strings.Contains(string(event.Data), "/skill") {
			rawCommands++
		}
	}
	if invocations != 1 || rawCommands != 0 {
		t.Fatalf("invocations=%d raw_commands=%d", invocations, rawCommands)
	}
	if len(provider.calls) == 0 || !requestContains(provider.calls[0], `<skill_content name="web-design-taste">`) || !requestContains(provider.calls[0], "restrained visual hierarchy") || requestContains(provider.calls[0], "/skill web-design-taste") {
		encoded, _ := json.Marshal(provider.calls)
		t.Fatalf("explicit skill was not safely injected: %s", encoded)
	}
	if _, err := runtime.GetModel(context.Background(), "web-design-taste"); err == nil || !strings.Contains(err.Error(), "already loaded") {
		t.Fatalf("expected no-double-load guard, got %v", err)
	}
}

func TestSteeringSkillInvocationUsesSameExplicitPath(t *testing.T) {
	runtime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	store := newMemoryStore()
	loop := NewLoop("skills-steering", store, tools.New(tools.Options{}), &scriptedProvider{}, Config{Skills: runtime})
	lease, err := store.ClaimLease(context.Background(), "skills-steering", "test", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	em := &emitter{sessionID: "skills-steering", runID: "run"}
	if err := loop.appendStepInputs(context.Background(), lease, em, "turn", 1, []StepInput{{ID: "steer-1", Type: session.EventSteeringMessage, Text: "/skill web-design-taste"}}); err != nil {
		t.Fatal(err)
	}
	events, err := store.Load(context.Background(), "skills-steering", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, event := range events {
		if event.Type == session.EventSkillInvocation {
			found = true
		}
	}
	if !found {
		t.Fatalf("steering skill command was not converted: %+v", events)
	}
}

func TestInvalidSkillCommandRemainsModelVisibleInput(t *testing.T) {
	runtime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	toolRegistry := tools.New(tools.Options{})
	registration, err := skilltool.RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	store := newMemoryStore()
	provider := &scriptedProvider{steps: []scriptedStep{{text: "not available"}}}
	loop := NewLoop("skills-invalid", store, toolRegistry, provider, Config{Model: "m", Owner: "test", Skills: runtime})
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "/skill missing-skill"); err != nil {
		t.Fatalf("unknown user command terminated the run: %v", err)
	}
	if len(provider.calls) != 1 || !requestContains(provider.calls[0], "/skill missing-skill") {
		t.Fatalf("unknown command was not left model-visible: %+v", provider.calls)
	}
}

func TestExplicitSkillInvocationNoDoubleLoadSurvivesRuntimeResume(t *testing.T) {
	firstRuntime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	firstTools := tools.New(tools.Options{})
	firstRegistration, err := skilltool.RegisterSkillTool(firstTools, firstRuntime)
	if err != nil {
		t.Fatal(err)
	}
	store := newMemoryStore()
	firstProvider := &scriptedProvider{steps: []scriptedStep{{text: "done"}}}
	firstLoop := NewLoop("skills-resume-user", store, firstTools, firstProvider, Config{Model: "m", Owner: "first", SystemPrompt: "sys", Skills: firstRuntime})
	if _, err := firstLoop.RunInput(context.Background(), session.EventUserMessage, "/skill web-design-taste"); err != nil {
		t.Fatal(err)
	}
	firstRegistration.Unregister()

	resumedRuntime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	resumedTools := tools.New(tools.Options{})
	resumedRegistration, err := skilltool.RegisterSkillTool(resumedTools, resumedRuntime)
	if err != nil {
		t.Fatal(err)
	}
	defer resumedRegistration.Unregister()
	resumedProvider := &scriptedProvider{steps: []scriptedStep{{
		calls:  []llm.ToolCallRequest{{CallID: "duplicate-skill", Name: "skill", Arguments: json.RawMessage(`{"name":"web-design-taste"}`)}},
		finish: llm.FinishReasonToolCalls,
	}, {text: "done"}}}
	resumedLoop := NewLoop("skills-resume-user", store, resumedTools, resumedProvider, Config{Model: "m", Owner: "second", SystemPrompt: "sys", Skills: resumedRuntime})
	if _, err := resumedLoop.RunInput(context.Background(), session.EventUserMessage, "continue"); err != nil {
		t.Fatal(err)
	}
	events, _ := store.Load(context.Background(), "skills-resume-user", 0, 0)
	invocations, modelLoads := 0, 0
	for _, event := range events {
		if event.Type == session.EventSkillInvocation {
			invocations++
		}
		if event.Type == session.EventSkillLoaded {
			modelLoads++
		}
	}
	if invocations != 1 || modelLoads != 0 {
		t.Fatalf("invocations=%d model_loads=%d", invocations, modelLoads)
	}
}

func TestModelSkillLoadWritesDurableMetadata(t *testing.T) {
	runtime := testSkillRuntime(t, skill.InvocationPolicy{Model: true, User: true})
	toolRegistry := tools.New(tools.Options{})
	registration, err := skilltool.RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	store := newMemoryStore()
	provider := &scriptedProvider{steps: []scriptedStep{
		{calls: []llm.ToolCallRequest{{CallID: "skill-call", Name: "skill", Arguments: json.RawMessage(`{"name":"web-design-taste"}`)}}, finish: llm.FinishReasonToolCalls},
		{text: "done"},
	}}
	loop := NewLoop("skills-model-load", store, toolRegistry, provider, Config{Model: "m", Owner: "test", SystemPrompt: "sys", Skills: runtime})
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "design a page"); err != nil {
		t.Fatal(err)
	}
	events, _ := store.Load(context.Background(), "skills-model-load", 0, 0)
	loaded := 0
	for _, event := range events {
		if event.Type != session.EventSkillLoaded {
			continue
		}
		loaded++
		var payload session.SkillInvocationEventPayload
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			t.Fatal(err)
		}
		if payload.Name != "web-design-taste" || payload.Origin != "model" || payload.ContentHash == "" {
			t.Fatalf("unexpected load metadata: %+v", payload)
		}
	}
	if loaded != 1 {
		t.Fatalf("loaded metadata events = %d, want 1", loaded)
	}
}

func TestSkillsCatalogWritesEmptyReplacementAfterRemoval(t *testing.T) {
	body := "Design guidance."
	digest := sha256.Sum256([]byte(body))
	definition := skill.Definition{Summary: skill.Summary{Name: "design-taste", Description: "Design guidance", Policy: skill.InvocationPolicy{Model: true}, Provider: skill.RuntimeProviderName, Source: "backend-test"}, Version: "v1", ContentHash: hex.EncodeToString(digest[:]), Content: body}
	skillRegistry := skill.NewRegistry(0)
	dispose, err := skillRegistry.Register("", definition, 100)
	if err != nil {
		t.Fatal(err)
	}
	runtime, _ := skill.NewRuntime(skillRegistry, skill.RuntimeOptions{})
	if err := runtime.Initialize(context.Background()); err != nil {
		t.Fatal(err)
	}
	toolRegistry := tools.New(tools.Options{})
	registration, err := skilltool.RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	store := newMemoryStore()
	provider := &scriptedProvider{steps: []scriptedStep{{text: "first"}, {text: "second"}}}
	loop := NewLoop("skills-empty", store, toolRegistry, provider, Config{Model: "m", Owner: "test", SystemPrompt: "sys", Skills: runtime})
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "first"); err != nil {
		t.Fatal(err)
	}
	dispose()
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "second"); err != nil {
		t.Fatal(err)
	}
	events, _ := store.Load(context.Background(), "skills-empty", 0, 0)
	var catalogs []session.SkillCatalogEventPayload
	for _, event := range events {
		if event.Type == session.EventSkillCatalog {
			var payload session.SkillCatalogEventPayload
			if err := json.Unmarshal(event.Data, &payload); err != nil {
				t.Fatal(err)
			}
			catalogs = append(catalogs, payload)
		}
	}
	if len(catalogs) != 2 || len(catalogs[1].Skills) != 0 || !catalogs[1].Update {
		t.Fatalf("unexpected replacements: %+v", catalogs)
	}
	if !requestContains(provider.calls[1], "No skills are currently available") || requestContains(provider.calls[1], "`design-taste`") {
		t.Fatal("empty replacement did not tombstone the prior catalog")
	}
}

func TestSkillsCatalogIsReestablishedAfterCompaction(t *testing.T) {
	runtime := testSkillRuntime(t, skill.InvocationPolicy{Model: true})
	toolRegistry := tools.New(tools.Options{})
	registration, err := skilltool.RegisterSkillTool(toolRegistry, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer registration.Unregister()
	store := newMemoryStore()
	lease, err := store.ClaimLease(context.Background(), "skills-compacted", "seed", 0, "")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := runtime.Snapshot()
	catalog := runtime.Catalog()
	initial := session.SkillCatalogEventPayload{SchemaVersion: CurrentSkillSchemaVersion, Complete: true, CatalogHash: catalog.Hash, SnapshotHash: snapshot.SnapshotHash, SnapshotID: snapshot.SnapshotHash, Text: runtime.RenderCatalog(false), Skills: []session.SkillSummaryRef{{Name: "web-design-taste", Description: "Tasteful web composition"}}}
	if _, err := store.AppendFenced(context.Background(), lease, []session.Event{{ID: "catalog", SessionID: "skills-compacted", Type: session.EventSkillCatalog, Data: session.SkillCatalogPayload(initial)}}); err != nil {
		t.Fatal(err)
	}
	tx, fingerprint := "compact-skills", "compaction-fingerprint"
	compaction := []session.Event{
		{ID: "start", SessionID: "skills-compacted", Type: session.EventCompactionStart, Data: session.CompactionStartPayload(1, tx, []uint64{1})},
		{ID: "summary", SessionID: "skills-compacted", Type: session.EventCompactionSummary, Data: session.CompactionSummaryPayload(1, tx, 1, []uint64{1}, "Earlier context.", fingerprint), SourceSeqs: []uint64{1}},
		{ID: "surface", SessionID: "skills-compacted", Type: session.EventUserMessage, Data: session.CompactionSurfaceReplacementPayload(1, tx, []uint64{1}, "Earlier context.", []session.ContentBlock{session.TextBlock("Earlier context.")}, fingerprint), SourceSeqs: []uint64{1}},
		{ID: "end", SessionID: "skills-compacted", Type: session.EventCompactionEnd, Data: session.CompactionEndPayload(1, tx)},
	}
	if _, err := store.AppendFenced(context.Background(), lease, compaction); err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	provider := &scriptedProvider{steps: []scriptedStep{{text: "done"}}}
	loop := NewLoop("skills-compacted", store, toolRegistry, provider, Config{Model: "m", Owner: "test", SystemPrompt: "sys", Skills: runtime})
	if _, err := loop.RunInput(context.Background(), session.EventUserMessage, "continue"); err != nil {
		t.Fatal(err)
	}
	events, _ := store.Load(context.Background(), "skills-compacted", 0, 0)
	catalogEvents := 0
	for _, event := range events {
		if event.Type == session.EventSkillCatalog {
			catalogEvents++
		}
	}
	if catalogEvents != 2 || !requestContains(provider.calls[0], "`web-design-taste`") {
		t.Fatalf("catalog was not restored after compaction: events=%d", catalogEvents)
	}
}

func TestCompactionPreservesSkillBodiesAndCompleteToolGroup(t *testing.T) {
	invocation := session.SkillInvocationEventPayload{SchemaVersion: CurrentSkillSchemaVersion, Name: "web-design-taste", Provider: "runtime", Source: "test", Version: "v1", ContentHash: strings.Repeat("a", sha256.Size*2), Origin: "user", Text: "<skill_content>instructions</skill_content>"}
	events := []session.Event{
		{Seq: 1, Type: session.EventUserMessage, Data: session.UserText("before")},
		{Seq: 2, Type: session.EventSkillInvocation, Data: session.SkillInvocationPayload(invocation)},
		{Seq: 3, Type: session.EventUserMessage, Data: session.UserText("work")},
		{Seq: 4, Type: session.EventAssistantMessage, Data: session.AssistantContent("", "", []session.ToolCall{{CallID: "skill-call", Name: "skill", Arguments: json.RawMessage(`{"name":"web-design-taste"}`)}, {CallID: "read-call", Name: "read", Arguments: json.RawMessage(`{"path":"x"}`)}})},
		{Seq: 5, Type: session.EventToolResult, CallID: "skill-call", SourceSeqs: []uint64{4}, Data: session.ToolResultPayload("skill-call", "skill", json.RawMessage(`"<skill_content name=\"web-design-taste\">instructions</skill_content>"`), false)},
		{Seq: 6, Type: session.EventToolResult, CallID: "read-call", SourceSeqs: []uint64{4}, Data: session.ToolResultPayload("read-call", "read", json.RawMessage(`{"ok":true}`), false)},
		{Seq: 7, Type: session.EventUserMessage, Data: session.UserText("continue")},
		{Seq: 8, Type: session.EventAssistantMessage, Data: session.AssistantContent("latest", "", nil)},
	}
	shadowed := shadowedSeqsRetainingTail(events, 1, nil)
	shadowedSet := make(map[uint64]bool, len(shadowed))
	for _, seq := range shadowed {
		shadowedSet[seq] = true
	}
	for _, protected := range []uint64{2, 4, 5, 6} {
		if shadowedSet[protected] {
			t.Errorf("skill context seq %d was compacted: %v", protected, shadowed)
		}
	}
	if !shadowedSet[1] || !shadowedSet[3] || !shadowedSet[7] {
		t.Fatalf("ordinary old context was not compacted: %v", shadowed)
	}
}

func testSkillRuntime(t *testing.T, policy skill.InvocationPolicy) *skill.Runtime {
	t.Helper()
	body := "Use a restrained visual hierarchy and verify responsive composition."
	digest := sha256.Sum256([]byte(body))
	definition := skill.Definition{
		Summary: skill.Summary{Name: "web-design-taste", Description: "Tasteful web composition", Policy: policy, Provider: skill.RuntimeProviderName, Source: "backend-test"},
		Version: "v1", ContentHash: hex.EncodeToString(digest[:]), Content: body,
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
	return runtime
}

func requestContains(request llm.Request, needle string) bool {
	for _, message := range request.Messages {
		if strings.Contains(message.Text(), needle) {
			return true
		}
	}
	return false
}
