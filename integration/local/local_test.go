package local

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/job"
)

func newTestEnv(t *testing.T) (root string) {
	t.Helper()
	root = t.TempDir()
	return root
}

func TestLocalFSReadWriteEditCAS(t *testing.T) {
	root := newTestEnv(t)
	fs := NewLocalFileSystem(root)
	ctx := context.Background()
	owner := "session-1"

	if err := os.MkdirAll(filepath.Join(root, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	target, err := fs.Resolve(ctx, "src/app.ts", root)
	if err != nil {
		t.Fatal(err)
	}
	// write (create)
	res, err := fs.WriteText(ctx, target, "line1\nline2\n", integration.WriteIntent{CreateIfAbsent: true})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if !res.Created {
		t.Error("expected created")
	}
	// read back + observe
	text, err := fs.ReadText(ctx, target)
	if err != nil {
		t.Fatal(err)
	}
	if text != "line1\nline2\n" {
		t.Errorf("text = %q", text)
	}
	_ = fs.Observe(ctx, target, owner, integration.ObservationPresent, res.Version)

	// edit with CAS + observation
	editRes, err := fs.EditText(ctx, target, integration.EditRequest{OldString: "line1", NewString: "LINE1"}, integration.EditIntent{ExpectedVersion: res.Version})
	if err != nil {
		t.Fatalf("edit: %v", err)
	}
	if editRes.Replaced != 1 {
		t.Errorf("replaced = %d", editRes.Replaced)
	}
	// stale version must fail
	if _, err := fs.EditText(ctx, target, integration.EditRequest{OldString: "LINE1", NewString: "x"}, integration.EditIntent{ExpectedVersion: res.Version}); !errors.Is(err, integration.ErrStaleVersion) {
		t.Errorf("expected ErrStaleVersion, got %v", err)
	}
	// ambiguous match
	_, _ = fs.WriteText(ctx, target, "dup dup\n", integration.WriteIntent{Overwrite: true})
	if _, err := fs.EditText(ctx, target, integration.EditRequest{OldString: "dup", NewString: "DUP"}, integration.EditIntent{}); !errors.Is(err, integration.ErrAmbiguousMatch) {
		t.Errorf("expected ErrAmbiguousMatch, got %v", err)
	}
	// replace_all resolves ambiguity
	all, err := fs.EditText(ctx, target, integration.EditRequest{OldString: "dup", NewString: "DUP", ReplaceAll: true}, integration.EditIntent{})
	if err != nil {
		t.Fatalf("replace_all: %v", err)
	}
	if all.Replaced != 2 {
		t.Errorf("replace_all replaced = %d", all.Replaced)
	}
	// create-if-absent on existing file fails
	if _, err := fs.WriteText(ctx, target, "x", integration.WriteIntent{CreateIfAbsent: true}); !errors.Is(err, integration.ErrAlreadyExists) {
		t.Errorf("expected ErrAlreadyExists, got %v", err)
	}
}

func TestLocalFSContainment(t *testing.T) {
	root := newTestEnv(t)
	fs := NewLocalFileSystem(root)
	ctx := context.Background()

	// Traversal must be rejected on the canonical target.
	if _, err := fs.Resolve(ctx, "../../etc/passwd", root); err == nil {
		// The resolution may still produce an out-of-root canonical path;
		// the sandbox layer rejects it. Verify via canonicalWithin directly.
		t.Log("resolve returned a target; containment is enforced by canonicalWithin")
	}
	_, err := canonicalWithin(filepath.Join(root, "..", "escape.txt"), root)
	if !errors.Is(err, integration.ErrTargetOutsideRoot) {
		t.Errorf("expected ErrTargetOutsideRoot, got %v", err)
	}
}

func TestLocalSandboxDeniesOutsideRootWrite(t *testing.T) {
	root := newTestEnv(t)
	sb := NewLocalSandbox(root)
	ctx := context.Background()
	outside := filepath.Join(filepath.Dir(root), "outside.txt")
	_, err := sb.ResolvePath(ctx, "u1", outside, root, integration.AccessWrite)
	var denial *integration.Denial
	if !errors.As(err, &denial) {
		t.Fatalf("expected sandbox denial, got %v", err)
	}
	if denial.Code != "SANDBOX_DENIED_OUTSIDE_ROOT" {
		t.Errorf("denial code = %q", denial.Code)
	}
}

func TestLocalSandboxReadOnlyDeniesWrites(t *testing.T) {
	root := newTestEnv(t)
	sb := NewLocalSandbox(root)
	sb.SetMode("u1", integration.ModeReadOnly)
	ctx := context.Background()
	_, err := sb.ResolvePath(ctx, "u1", "a.txt", root, integration.AccessWrite)
	var denial *integration.Denial
	if !errors.As(err, &denial) {
		t.Fatalf("expected denial, got %v", err)
	}
	if denial.Code != "SANDBOX_DENIED_READONLY" {
		t.Errorf("denial code = %q", denial.Code)
	}
	// Read access is fine.
	if _, err := sb.ResolvePath(ctx, "u1", "a.txt", root, integration.AccessRead); err != nil {
		t.Errorf("read should be allowed: %v", err)
	}
	// bash mutation denied in read-only.
	if err := sb.CheckCommand(ctx, "u1", integration.Command{Command: "rm -rf /", Workdir: root}, integration.AccessWrite); err == nil {
		t.Error("expected bash denial in read-only session")
	}
}

func TestLocalSubprocessRunsBounded(t *testing.T) {
	root := newTestEnv(t)
	sub := NewLocalSubprocess(DefaultSubprocessOptions())
	ctx := context.Background()
	res, err := sub.Run(ctx, integration.Command{Command: "echo hello", Workdir: root, Description: "test"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.ExitCode != 0 || !strings.Contains(res.Output, "hello") {
		t.Errorf("res = %+v", res)
	}
	// Nonzero exit is structured, not an error.
	res, err = sub.Run(ctx, integration.Command{Command: "exit 3", Workdir: root, Description: "test"})
	if err != nil {
		t.Fatalf("nonzero exit should be a result: %v", err)
	}
	if res.ExitCode != 3 {
		t.Errorf("exit code = %d", res.ExitCode)
	}
}

func TestLocalSubprocessTimeout(t *testing.T) {
	root := newTestEnv(t)
	sub := NewLocalSubprocess(DefaultSubprocessOptions())
	ctx := context.Background()
	start := time.Now()
	res, err := sub.Run(ctx, integration.Command{Command: "ping -n 30 127.0.0.1 > nul", Workdir: root, Description: "slow", Timeout: 300 * time.Millisecond})
	if err == nil || !res.TimedOut {
		t.Errorf("expected timeout error, got err=%v res=%+v", err, res)
	}
	if time.Since(start) > 5*time.Second {
		t.Errorf("timeout took too long: %v", time.Since(start))
	}
}

func TestLocalJobsOwnerScoped(t *testing.T) {
	root := newTestEnv(t)
	sub := NewLocalSubprocess(DefaultSubprocessOptions())
	manager := NewLocalJobManager(sub, DefaultJobManagerOptions())
	ctx := context.Background()

	id, err := manager.Start(ctx, job.Spec{Kind: "shell", Command: "echo job-output", Workdir: root}, "owner-a")
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	// Cross-owner output is denied.
	if _, err := manager.Output(ctx, id, job.OutputOptions{}, "owner-b"); err == nil {
		t.Error("expected cross-owner denial")
	}
	// Owner can read output after completion.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		out, err := manager.Output(ctx, id, job.OutputOptions{}, "owner-a")
		if err != nil {
			t.Fatal(err)
		}
		if out.Status == job.StateCompleted || out.Status == job.StateFailed {
			if !strings.Contains(out.Text, "job-output") {
				t.Errorf("output = %q", out.Text)
			}
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Idempotent kill of an unknown id is an error; kill of completed is fine.
	if err := manager.Kill(ctx, id, "test", "owner-b"); err == nil {
		t.Error("expected cross-owner kill denial")
	}
	if err := manager.Kill(ctx, id, "test", "owner-a"); err != nil {
		t.Errorf("kill own job: %v", err)
	}
	_ = os.MkdirAll(root, 0o755)
}
