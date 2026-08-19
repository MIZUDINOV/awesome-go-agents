package e2b

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/job"
)

// fakeTransport is an in-memory Transport used to assert contract parity with
// the local adapters: containment, CAS, structured denials, structured command
// exits, and owner-scoped jobs.
type fakeTransport struct {
	mu     sync.Mutex
	files  map[string]string
	root   string
	denies map[string]*integration.Denial
	jobs   map[string]job.Output
	next   int
}

func newFakeTransport(root string) *fakeTransport {
	return &fakeTransport{
		files:  map[string]string{},
		root:   root,
		denies: map[string]*integration.Denial{},
		jobs:   map[string]job.Output{},
	}
}

func (f *fakeTransport) Resolve(_ context.Context, rawPath, cwd string) (string, bool, error) {
	joined := rawPath
	if !strings.HasPrefix(rawPath, f.root) {
		joined = f.root + "/" + strings.TrimPrefix(rawPath, "/")
	}
	contained := strings.HasPrefix(joined, f.root) && !strings.Contains(strings.TrimPrefix(joined, f.root), "..")
	return joined, contained, nil
}

func (f *fakeTransport) Stat(_ context.Context, path string) (integration.FileInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	content, ok := f.files[path]
	if !ok {
		return integration.FileInfo{Exists: false}, nil
	}
	return integration.FileInfo{Exists: true, Size: int64(len(content)), Version: sha(content)}, nil
}

func (f *fakeTransport) Read(_ context.Context, path string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	content, ok := f.files[path]
	if !ok {
		return nil, integration.ErrNotObserved // ~ not found
	}
	return []byte(content), nil
}

func (f *fakeTransport) Write(_ context.Context, path string, content []byte, expectedVersion string, createIfAbsent, overwrite bool) (string, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	existing, exists := f.files[path]
	if createIfAbsent && exists {
		return "", false, integration.ErrAlreadyExists
	}
	if exists && expectedVersion != "" && sha(existing) != expectedVersion {
		return "", false, integration.ErrStaleVersion
	}
	if exists && !overwrite {
		return "", false, errors.New("refusing overwrite")
	}
	f.files[path] = string(content)
	return sha(string(content)), !exists, nil
}

func (f *fakeTransport) Run(_ context.Context, command, workdir string, timeoutSec int) (integration.Result, error) {
	cmd := strings.TrimSpace(command)
	switch {
	case cmd == "echo hello":
		return integration.Result{ExitCode: 0, Output: "hello\n"}, nil
	case cmd == "exit 7":
		return integration.Result{ExitCode: 7, Output: ""}, nil
	case cmd == "sleep-forever":
		return integration.Result{ExitCode: -1, Output: "", TimedOut: true}, context.DeadlineExceeded
	}
	return integration.Result{ExitCode: 0, Output: ""}, nil
}

func (f *fakeTransport) Start(_ context.Context, command, workdir, owner string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.next++
	id := "job-" + itoa(f.next)
	f.jobs[id] = job.Output{Status: job.StateCompleted, Text: command}
	return id, nil
}

func (f *fakeTransport) Output(_ context.Context, id, owner string, tail, wait bool) (job.Output, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out, ok := f.jobs[id]
	if !ok {
		return job.Output{}, errors.New("job not found")
	}
	return out, nil
}

func (f *fakeTransport) Kill(_ context.Context, id, reason, owner string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.jobs[id]; !ok {
		return errors.New("job not found")
	}
	return nil
}

func (f *fakeTransport) Deny(_ context.Context, owner, resource string, access integration.Access) *integration.Denial {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.denies[resource]
}

func sha(s string) string {
	var h uint64 = 1469598103934665603
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	const digits = "0123456789abcdef"
	out := make([]byte, 16)
	for i := range out {
		out[i] = digits[(h>>(uint(i%16)*4))&0xf]
	}
	return string(out)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b []byte
	for v > 0 {
		b = append([]byte{byte('0' + v%10)}, b...)
		v /= 10
	}
	return string(b)
}

// Parity: read-before-edit, CAS stale, containment, denials, structured exit,
// and jobs all behave like the local adapters through the same seams.
func TestRemoteParityWithLocal(t *testing.T) {
	tp := newFakeTransport("/root")
	client := New(tp)
	fs := client.FS()
	sb := client.Sandbox()
	ctx := context.Background()
	const owner = "session-1"

	// containment: escaping path denied at the sandbox.
	if _, err := sb.ResolvePath(ctx, owner, "../etc/passwd", "/root", integration.AccessRead); err == nil {
		t.Error("expected containment denial for ../ escape")
	}

	// create-if-absent write registers observation; read before edit.
	target, err := sb.ResolvePath(ctx, owner, "a.txt", "/root", integration.AccessWrite)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fs.WriteText(ctx, target, "hello", integration.WriteIntent{CreateIfAbsent: true}); err != nil {
		t.Fatalf("write: %v", err)
	}
	_ = fs.(integration.ObservationRecorder).Observe(ctx, target, owner, integration.ObservationPresent, sha("hello"))

	// write again without createIfAbsent must not overwrite blindly (no intent).
	if _, err := fs.WriteText(ctx, target, "x", integration.WriteIntent{}); err == nil {
		t.Error("expected overwrite refusal without Overwrite intent")
	}
	// CAS write with stale expected version fails.
	if _, err := fs.WriteText(ctx, target, "y", integration.WriteIntent{Overwrite: true, ExpectedVersion: "wrong"}); !errors.Is(err, integration.ErrStaleVersion) {
		t.Errorf("expected ErrStaleVersion, got %v", err)
	}

	// edit enforces ambiguous/nonexistent semantics client-side.
	if _, err := fs.EditText(ctx, target, integration.EditRequest{OldString: "missing", NewString: "z"}, integration.EditIntent{}); err == nil {
		t.Error("expected edit no-match error")
	}

	// structured command exit.
	sub := client.Subprocess()
	res, err := sub.Run(ctx, integration.Command{Command: "exit 7", Workdir: "/root"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.ExitCode != 7 {
		t.Errorf("exit code = %d", res.ExitCode)
	}
	// cancellation maps to TimedOut.
	res, err = sub.Run(ctx, integration.Command{Command: "sleep-forever", Workdir: "/root"})
	if err == nil || !res.TimedOut {
		t.Errorf("expected timeout, got err=%v res=%+v", err, res)
	}

	// jobs owner-scoped via Transport.
	jobs := client.Jobs()
	id, err := jobs.Start(ctx, job.Spec{Command: "echo j", Workdir: "/root"}, owner)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := jobs.Output(ctx, id, job.OutputOptions{}, owner); err != nil || out.Status != job.StateCompleted {
		t.Errorf("job output = %+v err=%v", out, err)
	}
}

// Denial surfaces as a stable SANDBOX_DENIED code through the sandbox seam.
func TestRemoteSandboxDenial(t *testing.T) {
	tp := newFakeTransport("/root")
	tp.denies["path:write"] = integration.SandboxDenied("SANDBOX_DENIED_READONLY", "write", "/root/a.txt", "read-only")
	client := New(tp)
	_, err := client.Sandbox().ResolvePath(context.Background(), "u1", "a.txt", "/root", integration.AccessWrite)
	var denial *integration.Denial
	if !errors.As(err, &denial) || denial.Code != "SANDBOX_DENIED_READONLY" {
		t.Fatalf("expected SANDBOX_DENIED_READONLY, got %v", err)
	}
}
