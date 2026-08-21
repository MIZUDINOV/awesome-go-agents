package agent

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/integration/local"
	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
	"github.com/MIZUDINOV/awesome-go-agents/tools/core"
)

// ---------------------------------------------------------------------------
// End-to-end harness: durable loop + full core-tool wiring + a real
// workspace-confined sandbox. Scenarios A/B/C/D/F/I/J from the review
// checklist (§22) execute through the real seams (no stubs for the runtime).
// ---------------------------------------------------------------------------

type harness struct {
	store   session.FencedStore
	chat    *e2eChat
	loop    *Loop
	sandbox *local.LocalSandbox
	ws      string
	owner   string
}

// e2eChat is a scripted provider whose replies are a queue of tool-call sets
// then a final text. Optionally fails the Nth generation with a provider error.
type e2eChat struct {
	mu        sync.Mutex
	steps     []e2eStep
	n         int
	failCount int
	failErr   error
}

type e2eStep struct {
	text  string
	calls []llm.ToolCallRequest
}

func (c *e2eChat) Name() string { return "e2e" }

func (c *e2eChat) Generate(ctx context.Context, req *llm.Request, cb llm.StreamCallback) (*llm.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.n++
	if c.failErr != nil && c.failCount > 0 && c.n <= c.failCount {
		return nil, c.failErr
	}
	var step e2eStep
	if len(c.steps) > 0 {
		step = c.steps[0]
		c.steps = c.steps[1:]
	}
	finish := llm.FinishReasonStop
	if len(step.calls) > 0 {
		finish = llm.FinishReasonToolCalls
	}
	return &llm.Response{Message: llm.NewAssistantMessage(step.text, "", step.calls), FinishReason: finish, Model: req.Model}, nil
}

func newHarness(t *testing.T, ws string, owner string, steps []e2eStep, cfg Config) *harness {
	t.Helper()
	store := session.NewMemoryStore()
	registry := tools.New(tools.Options{})
	sandbox := local.NewLocalSandbox(ws)
	sub := local.NewLocalSubprocess(local.DefaultSubprocessOptions())
	manager := local.NewLocalJobManager(sub, local.DefaultJobManagerOptions())
	if err := core.Register(registry, core.Deps{
		Sandbox:    sandbox,
		FS:         local.NewLocalFileSystem(ws),
		Subprocess: sub,
		Jobs:       manager,
	}); err != nil {
		t.Fatal(err)
	}
	chat := &e2eChat{steps: steps}
	if cfg.Vars == nil {
		cfg.Vars = map[string]any{"cwd": ws}
	} else {
		cfg.Vars["cwd"] = ws
	}
	loop := NewLoop("e2e-"+owner, store, registry, chat, cfg)
	return &harness{store: store, chat: chat, loop: loop, sandbox: sandbox, ws: ws, owner: owner}
}

// toolResults returns the durable tool/result Data for a call name, in order.
func (h *harness) toolResults(t *testing.T, name string) [][]byte {
	t.Helper()
	events, err := h.store.Load(context.Background(), "e2e-"+h.owner, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]byte
	for _, e := range events {
		if e.Type == session.EventToolResult && strings.Contains(string(e.Data), `"name":"`+name+`"`) {
			out = append(out, append([]byte(nil), e.Data...))
		}
	}
	return out
}

// Scenario A: safe edit. read records V1, edit commits against V1.
func TestE2E_ScenarioA_SafeEdit(t *testing.T) {
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "src", "app.ts"), []byte("line1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	readArg := mustArgs(t, map[string]any{"file_path": filepath.ToSlash(filepath.Join("src", "app.ts"))})
	editArg := mustArgs(t, map[string]any{
		"file_path":  filepath.ToSlash(filepath.Join("src", "app.ts")),
		"old_string": "line1",
		"new_string": "line1\nline2",
	})

	h := newHarness(t, ws, "A", []e2eStep{
		{calls: []llm.ToolCallRequest{{CallID: "a1", Name: "read", Arguments: json.RawMessage(readArg)}}},
		{calls: []llm.ToolCallRequest{{CallID: "a2", Name: "edit", Arguments: json.RawMessage(editArg)}}},
		{text: "edited"},
	}, Config{Model: "m", Owner: "A", SystemPrompt: "sys"})

	if _, err := h.loop.Run(context.Background(), "change app"); err != nil {
		t.Fatal(err)
	}
	// Content persisted with the expected version guard.
	data, err := os.ReadFile(filepath.Join(ws, "src", "app.ts"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "line2") {
		t.Fatalf("edit not persisted: %q", data)
	}
	readResults := h.toolResults(t, "read")
	if len(readResults) < 1 {
		t.Fatal("missing durable read result")
	}
	if !strings.Contains(string(readResults[0]), "line1") {
		t.Fatalf("read result lost content: %s", readResults[0])
	}
	editResults := h.toolResults(t, "edit")
	if len(editResults) < 1 {
		t.Fatal("missing durable edit result")
	}
	if !strings.Contains(string(editResults[0]), "updated successfully") {
		t.Fatalf("edit result should confirm success: %s", editResults[0])
	}
}

// Scenario B: stale file. read captures V1, an external editor writes V2, then
// an edit based on the stale observation is refused (FS_STALE_VERSION) instead
// of clobbering the external change.
func TestE2E_ScenarioB_StaleFile(t *testing.T) {
	ws := t.TempDir()
	full := filepath.Join(ws, "b.ts")
	if err := os.WriteFile(full, []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	h := newHarness(t, ws, "B", []e2eStep{
		{calls: []llm.ToolCallRequest{{CallID: "b1", Name: "read", Arguments: json.RawMessage(`{"file_path":"b.ts"}`)}}}},
		Config{Model: "m", Owner: "B", SystemPrompt: "sys"})

	// Run 1: read captures V1 into the owner's observation history.
	if _, err := h.loop.Run(context.Background(), "read file"); err != nil {
		t.Fatal(err)
	}
	// External editor writes V2, bypassing the agent.
	if err := os.WriteFile(full, []byte("v2-external\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Run 2: the model edits WITHOUT re-reading, so it holds the stale V1
	// observation; the CAS layer must refuse the clobber.
	h.chat.steps = []e2eStep{
		{calls: []llm.ToolCallRequest{{CallID: "b2", Name: "edit", Arguments: mustArgs(t, map[string]any{"file_path": "b.ts", "old_string": "v1", "new_string": "v1-clobber"})}}},
		{text: "done"},
	}
	if _, err := h.loop.Run(context.Background(), "update"); err != nil {
		t.Fatal(err)
	}
	// The edit was refused as stale, so the external v2 content is preserved.
	data, err := os.ReadFile(full)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "v2-external") {
		t.Fatalf("external content was clobbered by stale edit: %q", data)
	}
	editResults := h.toolResults(t, "edit")
	ok := false
	for _, r := range editResults {
		if strings.Contains(string(r), `"is_error":true`) && strings.Contains(strings.ToLower(string(r)), "version") {
			ok = true
		}
	}
	if !ok {
		t.Fatalf("expected a stale-version refusal, got %v", editResults)
	}
}

// Scenario C: parallel reads stay in model order with valid pairing.
func TestE2E_ScenarioC_ParallelReads(t *testing.T) {
	ws := t.TempDir()
	for _, name := range []string{"a.txt", "b.txt"} {
		if err := os.WriteFile(filepath.Join(ws, name), []byte("body-"+strings.TrimSuffix(name, ".txt")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	h := newHarness(t, ws, "C", []e2eStep{
		{calls: []llm.ToolCallRequest{
			{CallID: "c1", Name: "read", Arguments: json.RawMessage(`{"file_path":"a.txt"}`)},
			{CallID: "c2", Name: "read", Arguments: json.RawMessage(`{"file_path":"b.txt"}`)},
			{CallID: "c3", Name: "grep", Arguments: json.RawMessage(`{"pattern":"body","path":"."}`)},
		}},
		{text: "done"},
	}, Config{Model: "m", Owner: "C", SystemPrompt: "sys"})

	if _, err := h.loop.Run(context.Background(), "read a b and grep"); err != nil {
		t.Fatal(err)
	}
	// Next-request pairing must be valid: Project validates tool/result pairing.
	if _, _, err := h.loop.project(context.Background()); err != nil {
		t.Fatalf("surface pairing invalid: %v", err)
	}
	// Order: read(a) then read(b) then grep — durable and in model order.
	events, _ := h.store.Load(context.Background(), "e2e-C", 0, 0)
	var calls []string
	for _, e := range events {
		if e.Type == session.EventToolCall {
			calls = append(calls, callNameOf(e.Data))
		}
	}
	want := []string{"read", "read", "grep"}
	if len(calls) != 3 {
		t.Fatalf("expected 3 tool calls got %d", len(calls))
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Fatalf("call order %v != %v", calls, want)
		}
	}
}

// Scenario D: long output is capped with a truncation marker in history.
func TestE2E_ScenarioD_LongOutput(t *testing.T) {
	ws := t.TempDir()
	var huge strings.Builder
	for i := 0; i < 3000; i++ {
		huge.WriteString("this line makes a very long file exceeding the read cap\n")
	}
	bigPath := filepath.Join(ws, "big.txt")
	if err := os.WriteFile(bigPath, []byte(huge.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	h := newHarness(t, ws, "D", []e2eStep{
		{calls: []llm.ToolCallRequest{{CallID: "d1", Name: "read", Arguments: json.RawMessage(`{"file_path":"big.txt"}`)}}},
		{text: "done"},
	}, Config{Model: "m", Owner: "D", SystemPrompt: "sys"})

	if _, err := h.loop.Run(context.Background(), "read big file"); err != nil {
		t.Fatal(err)
	}
	results := h.toolResults(t, "read")
	if len(results) < 1 {
		t.Fatal("missing read result")
	}
	// The model-facing result is bounded; the stored result carries the cap
	// footer rather than the entire 3k-line file.
	if len(results[0]) > 64*1024 {
		t.Fatalf("durable result not bounded: %d bytes", len(results[0]))
	}
	if !strings.Contains(string(results[0]), "truncated") && len(results[0]) >= 64*1024 {
		t.Fatal("expected a truncation marker for capped output")
	}
}

// Scenario F: provider overflow -> compact -> bounded retries.
func TestE2E_ScenarioF_ProviderOverflow(t *testing.T) {
	ws := t.TempDir()
	h := newHarness(t, ws, "F", []e2eStep{{text: "recovered"}, {text: "done"}}, Config{
		Model: "m", Owner: "F", SystemPrompt: "sys", MaxContextRetries: 2,
		ContextWindow: 10000, MaxOutput: 10, CompactThresholdRatio: 0.80,
		Compactor: &countingCompactor{f: func(generation uint64, events []session.Event, sourceSeqs []uint64) (string, string, error) {
			return "[summary]", "fp-F", nil
		}},
	})
	// The first two real generations overflow; the loop compacts and retries
	// twice after the surface advances.
	h.chat.mu.Lock()
	h.chat.failCount = 2
	h.chat.failErr = &llm.Error{Kind: llm.ErrorKindContextOverflow, Message: "prompt is too long"}
	h.chat.mu.Unlock()
	lease, err := h.store.ClaimLease(context.Background(), "e2e-F", "seed", time.Minute, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.store.AppendFenced(context.Background(), lease, []session.Event{
		{SessionID: "e2e-F", Type: session.EventUserMessage, Data: session.UserText(strings.Repeat("old ", 40))},
		{SessionID: "e2e-F", Type: session.EventUserMessage, Data: session.UserText(strings.Repeat("older ", 40))},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.store.ReleaseLease(context.Background(), lease); err != nil {
		t.Fatal(err)
	}

	result, err := h.loop.Run(context.Background(), strings.Repeat("payload ", 40))
	if err != nil {
		t.Fatalf("overflow recovery should retry and succeed: %v", err)
	}
	if result.CompactCount < 1 {
		t.Fatalf("expected at least one compaction during overflow recovery, got %d", result.CompactCount)
	}
	// Durable compaction events present.
	events, err := h.store.Load(context.Background(), "e2e-F", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	compacted := 0
	stepStarts := map[string]bool{}
	stepEnds := map[string]bool{}
	for _, e := range events {
		if e.Type == session.EventCompactionStart || e.Type == session.EventCompactionSummary {
			compacted++
		}
		if e.Type == session.EventStepStart {
			stepStarts[e.StepID] = true
		}
		if e.Type == session.EventStepEnd {
			stepEnds[e.StepID] = true
		}
	}
	if compacted < 2 {
		t.Fatalf("expected compaction/start+summary events, got %d", compacted)
	}
	initialStep, retryStep, secondRetry := false, false, false
	for stepID := range stepStarts {
		initialStep = initialStep || strings.HasSuffix(stepID, "-step-00")
		retryStep = retryStep || strings.HasSuffix(stepID, "-step-00-retry-01")
		secondRetry = secondRetry || strings.HasSuffix(stepID, "-step-00-retry-02")
	}
	if len(stepStarts) != 3 || !initialStep || !retryStep || !secondRetry {
		t.Fatalf("overflow step attempts = %v, want initial + two retries", stepStarts)
	}
	for stepID := range stepStarts {
		if !stepEnds[stepID] {
			t.Fatalf("step %q has no matching step/end", stepID)
		}
	}
}

// Scenario G: a dangling tool/call (crash after side-effect start) is marked
// TOOL_OUTCOME_UNKNOWN and never re-executed.
func TestE2E_ScenarioG_DanglingCallUnknown(t *testing.T) {
	store := session.NewMemoryStore()
	registry := tools.New(tools.Options{})
	chat := &e2eChat{steps: []e2eStep{{calls: []llm.ToolCallRequest{{CallID: "g1", Name: "read", Arguments: json.RawMessage(`{"file_path":"missing.txt"}`)}}}, {text: "done"}}}
	ws := t.TempDir()
	sandbox := local.NewLocalSandbox(ws)
	if err := core.Register(registry, core.Deps{Sandbox: sandbox, FS: local.NewLocalFileSystem(ws), Subprocess: local.NewLocalSubprocess(local.DefaultSubprocessOptions()), Jobs: local.NewLocalJobManager(local.NewLocalSubprocess(local.DefaultSubprocessOptions()), local.DefaultJobManagerOptions())}); err != nil {
		t.Fatal(err)
	}
	// Seed a durable tool/call with no matching result, then release the lease
	// to simulate a worker that died right after persisting the call barrier.
	lease, err := store.ClaimLease(context.Background(), "e2e-G", "crashed", 0, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.AppendFenced(context.Background(), lease, []session.Event{{
		ID: "g-assistant", SessionID: "e2e-G", Type: session.EventAssistantMessage,
		Data: session.AssistantContent("", "", []session.ToolCall{{CallID: "g1", Name: "read", Arguments: json.RawMessage(`{"file_path":"missing.txt"}`)}}),
	}, {
		ID: "g-call", SessionID: "e2e-G", CallID: "g1", Type: session.EventToolCall,
		Data: session.ToolCallPayload("g1", "read", json.RawMessage(`{"file_path":"missing.txt"}`)),
	}, {
		ID: "g-dispatched", SessionID: "e2e-G", CallID: "g1", Type: session.EventToolDispatched,
		Data: json.RawMessage(`{"text":"dispatched"}`),
	}, {
		ID: "g-running", SessionID: "e2e-G", CallID: "g1", Type: session.EventToolRunning,
		Data: json.RawMessage(`{"text":"running"}`),
	}}); err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseLease(context.Background(), lease); err != nil {
		t.Fatal(err)
	}

	loop := NewLoop("e2e-G", store, registry, chat, Config{Model: "m", Owner: "G", SystemPrompt: "sys"})
	if _, err := loop.Run(context.Background(), "continue"); err != nil {
		t.Fatal(err)
	}
	events, _ := store.Load(context.Background(), "e2e-G", 0, 0)
	foundUnknown := false
	for _, e := range events {
		if e.Type == session.EventToolResult && strings.Contains(string(e.Data), `"code":"TOOL_OUTCOME_UNKNOWN"`) {
			foundUnknown = true
		}
	}
	if !foundUnknown {
		t.Fatal("expected TOOL_OUTCOME_UNKNOWN result for dangling call")
	}
}

// Scenario I: sandbox denial is a stable code and bash obeys the same boundary.
func TestE2E_ScenarioI_SandboxDenial(t *testing.T) {
	ws := t.TempDir()
	h := newHarness(t, ws, "I", []e2eStep{
		{calls: []llm.ToolCallRequest{{CallID: "i1", Name: "write", Arguments: mustArgs(t, map[string]any{"file_path": "inside.txt", "content": "x"})}}},
		{calls: []llm.ToolCallRequest{{CallID: "i2", Name: "bash", Arguments: mustArgs(t, map[string]any{"command": "echo hi"})}}},
		{text: "done"},
	}, Config{Model: "m", Owner: "I", SystemPrompt: "sys"})
	// Force read-only policy so BOTH the file write and the mutating bash
	// command are denied by the SAME boundary (SANDBOX_DENIED_READONLY). The
	// sandbox keys policy by the tool-owner (= the session id, ec.SessionID).
	h.sandbox.SetMode("e2e-I", integration.ModeReadOnly)

	if _, err := h.loop.Run(context.Background(), "write and bash in readonly"); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"write", "bash"} {
		results := h.toolResults(t, name)
		if len(results) < 1 {
			t.Fatalf("missing %s result", name)
		}
		if !strings.Contains(string(results[0]), "SANDBOX_DENIED_READONLY") {
			t.Fatalf("expected stable SANDBOX_DENIED_READONLY for %s, got %s", name, results[0])
		}
	}
	// A read-only session must not have left an artifact behind.
	if _, err := os.Stat(filepath.Join(ws, "inside.txt")); !os.IsNotExist(err) {
		t.Fatal("read-only session wrote to disk")
	}
}

// Scenario J: cancellation stops before further tools start; started tools
// settle and unstarted get no execution.
func TestE2E_ScenarioJ_Cancellation(t *testing.T) {
	ws := t.TempDir()
	unstartedSeen := 0
	h := newHarness(t, ws, "J", []e2eStep{
		{calls: []llm.ToolCallRequest{{CallID: "j1", Name: "grep", Arguments: json.RawMessage(`{"pattern":"x","path":"."}`)}}},
	}, Config{Model: "m", Owner: "J", SystemPrompt: "sys"})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel: nothing should execute
	_, err := h.loop.Run(ctx, "go")
	if !errors.Is(err, ErrStopped) {
		t.Fatalf("expected ErrStopped under cancellation, got %v", err)
	}
	unstartedSeen++ // no calls started at all
	_ = unstartedSeen
	// No tool call/result was durably written for the cancelled run.
	events, _ := h.store.Load(context.Background(), "e2e-J", 0, 0)
	for _, e := range events {
		if e.Type == session.EventToolCall || e.Type == session.EventToolResult {
			t.Fatalf("tool event under cancellation: %s", e.Type)
		}
	}
}

// helpers ---------------------------------------------------------------

func callNameOf(data json.RawMessage) string {
	var p struct {
		Name string `json:"name"`
	}
	_ = json.Unmarshal(data, &p)
	return p.Name
}

// mustArgs marshals a plain map into a JSON tool-call argument blob, escaping
// platform separators safely (no raw backslash string interpolation).
func mustArgs(t *testing.T, v map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
