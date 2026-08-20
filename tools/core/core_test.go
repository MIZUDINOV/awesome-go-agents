package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/integration/local"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

// newTestEnv wires a registry with the core tools against a local sandbox.
func newTestEnv(t *testing.T) (*tools.Registry, *local.LocalSandbox, string, map[string]any) {
	t.Helper()
	root := t.TempDir()
	sandbox := local.NewLocalSandbox(root)
	fs := local.NewLocalFileSystem(root)
	sub := local.NewLocalSubprocess(local.DefaultSubprocessOptions())
	manager := local.NewLocalJobManager(sub, local.DefaultJobManagerOptions())
	registry := tools.New(tools.Options{})
	if err := Register(registry, Deps{Sandbox: sandbox, FS: fs, Subprocess: sub, Jobs: manager}); err != nil {
		t.Fatalf("register core: %v", err)
	}
	vars := map[string]any{"cwd": root}
	return registry, sandbox, root, vars
}

func ctx() context.Context { return context.Background() }

func TestTypedCoreInputSchemasIncludeArgumentDescriptions(t *testing.T) {
	schemas := []struct {
		name   string
		schema json.RawMessage
	}{
		{"read", typedSchema[readInput]()}, {"read_image", typedSchema[readImageInput]()},
		{"write", typedSchema[writeInput]()}, {"edit", typedSchema[editInput]()},
		{"glob", typedSchema[globInput]()}, {"grep", typedSchema[grepInput]()},
		{"bash", typedSchema[bashInput]()}, {"job_start", typedSchema[jobStartInput]()},
		{"job_list", typedSchema[jobListInput]()}, {"job_output", typedSchema[jobOutputInput]()},
		{"job_kill", typedSchema[jobKillInput]()}, {"web_search", typedSchema[webSearchInput]()},
		{"web_fetch", typedSchema[webFetchInput]()}, {"lsp_diagnostics", typedSchema[lspDiagnosticsInput]()},
		{"terminal", typedSchema[terminalInput]()},
	}
	for _, item := range schemas {
		var schema struct {
			Properties map[string]struct {
				Description string `json:"description"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(item.schema, &schema); err != nil {
			t.Fatalf("%s: %v", item.name, err)
		}
		for field, property := range schema.Properties {
			if property.Description == "" {
				t.Errorf("%s.%s description is missing from the typed input schema", item.name, field)
			}
		}
	}
}

// call runs a tool with JSON-encoded args and returns the canonical result.
func call(t *testing.T, registry *tools.Registry, name string, vars map[string]any, args map[string]any) any {
	t.Helper()
	input, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	ec := tools.ExecContext{SessionID: "session-1", Vars: vars}
	out, err := registry.Run(ctx(), ec, name, "call-x", input)
	if err != nil {
		t.Fatalf("%s(%v): %v", name, args, err)
	}
	return out.Canonical
}

func errContains(t *testing.T, registry *tools.Registry, name string, vars map[string]any, args map[string]any, wantSub string) {
	t.Helper()
	input, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	ec := tools.ExecContext{SessionID: "session-1", Vars: vars}
	_, err = registry.Run(ctx(), ec, name, "call-x", input)
	if err == nil {
		t.Fatalf("%s(%v): expected error containing %q, got success", name, args, wantSub)
	}
	if !strings.Contains(err.Error(), wantSub) {
		t.Fatalf("%s(%v): error=%q, want substring %q", name, args, err, wantSub)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// readThenEdit is the happy path: read registers observation, then edit works.
func TestCoreReadThenEdit(t *testing.T) {
	registry, _, root, vars := newTestEnv(t)
	if err := os.MkdirAll(filepath.Join(root, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	call(t, registry, "write", vars, map[string]any{"file_path": "src/a.ts", "content": "hello world\nsecond\n"})
	call(t, registry, "read", vars, map[string]any{"file_path": "src/a.ts"})
	call(t, registry, "edit", vars, map[string]any{"file_path": "src/a.ts", "old_string": "hello", "new_string": "HELLO"})
	got, _ := os.ReadFile(filepath.Join(root, "src", "a.ts"))
	if !strings.Contains(string(got), "HELLO") {
		t.Errorf("edit did not apply: %q", string(got))
	}
}

// H-EDIT-004: editing without a prior read is refused (ErrNotObserved).
func TestCoreEditWithoutReadRefused(t *testing.T) {
	registry, _, root, vars := newTestEnv(t)
	writeFile(t, filepath.Join(root, "x.txt"), "content")
	errContains(t, registry, "edit", vars, map[string]any{"file_path": "x.txt", "old_string": "content", "new_string": "CONTENT"}, "read the file first")
}

// Scenario B (stale): concurrent external mutation invalidates the read version.
func TestCoreEditStaleVersion(t *testing.T) {
	registry, _, root, vars := newTestEnv(t)
	writeFile(t, filepath.Join(root, "x.txt"), "version1")
	call(t, registry, "read", vars, map[string]any{"file_path": "x.txt"})
	// External mutation invalidates the observed version.
	writeFile(t, filepath.Join(root, "x.txt"), "version2-changed")
	errContains(t, registry, "edit", vars, map[string]any{"file_path": "x.txt", "old_string": "version1", "new_string": "V1"}, "version changed since it was read")
}

// Scenario I (sandbox denial): a read-only session gets a stable denial code
// at the SAME boundary for files and shell.
func TestCoreSandboxDenialStableCode(t *testing.T) {
	registry, sandbox, root, vars := newTestEnv(t)
	sandbox.SetMode("session-1", integration.ModeReadOnly)
	writeFile(t, filepath.Join(root, "x.txt"), "data")
	call(t, registry, "read", vars, map[string]any{"file_path": "x.txt"})
	errContains(t, registry, "write", vars, map[string]any{"file_path": "y.txt", "content": "nope"}, "SANDBOX_DENIED_READONLY")
	errContains(t, registry, "bash", vars, map[string]any{"command": "echo hi", "workdir": root}, "SANDBOX_DENIED_READONLY")
}

// bash runs with bounded output and structured exit codes.
func TestCoreBashRuns(t *testing.T) {
	registry, sandbox, root, vars := newTestEnv(t)
	sandbox.SetMode("session-1", integration.ModeDangerFullAccess)
	m := call(t, registry, "bash", vars, map[string]any{"command": "echo bash-ok", "workdir": root}).(bashOutput)
	if m.ExitCode != 0 || !strings.Contains(m.Output, "bash-ok") {
		t.Errorf("bash result = %+v", m)
	}
	nc := call(t, registry, "bash", vars, map[string]any{"command": "exit 7", "workdir": root}).(bashOutput)
	if nc.ExitCode != 7 {
		t.Errorf("expected exit 7, got %+v", nc)
	}
}

// glob/grep are deterministic and bounded.
func TestCoreGlobGrep(t *testing.T) {
	registry, _, root, vars := newTestEnv(t)
	if err := os.MkdirAll(filepath.Join(root, "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, "pkg", "a.go"), "func A(){} // marker\n")
	writeFile(t, filepath.Join(root, "pkg", "b.go"), "func B(){} // marker\n")
	writeFile(t, filepath.Join(root, "pkg", "c.txt"), "no match")
	g := call(t, registry, "glob", vars, map[string]any{"pattern": "*.go", "path": "pkg"}).(globOutput)
	if len(g.Matches) != 2 {
		t.Errorf("glob matches = %#v", g.Matches)
	}
	gr := call(t, registry, "grep", vars, map[string]any{"pattern": "marker", "path": "pkg"}).(grepOutput)
	if n := len(gr.Matches); n != 2 {
		t.Errorf("grep matches length = %d", n)
	}
	// Traversal patterns resolve inside the root (empty result, no escape).
	out := call(t, registry, "glob", vars, map[string]any{"pattern": "../*"}).(globOutput)
	if len(out.Matches) != 0 {
		t.Errorf("escaping glob returned matches: %#v", out.Matches)
	}
}

// job lifecycle, owner scoped.
func TestCoreJobs(t *testing.T) {
	registry, sandbox, root, vars := newTestEnv(t)
	sandbox.SetMode("session-1", integration.ModeDangerFullAccess)
	start := call(t, registry, "job_start", vars, map[string]any{"command": "echo j1", "workdir": root}).(jobStartOutput)
	id := start.JobID
	if id == "" {
		t.Fatal("no job id")
	}
	found := false
	for i := 0; i < 300; i++ {
		o := call(t, registry, "job_output", vars, map[string]any{"job_id": id}).(jobOutputOutput)
		if status := o.Status; status == "completed" || status == "failed" {
			if !strings.Contains(o.Output, "j1") {
				t.Errorf("job output = %v", o)
			}
			found = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !found {
		t.Error("job never reached a terminal state")
	}
}
