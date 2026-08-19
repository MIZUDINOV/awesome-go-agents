package local

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
)

// SubprocessOptions bounds command execution.
type SubprocessOptions struct {
	// MaxOutputBytes caps captured stdout/stderr (default 8 MiB). Overflow is
	// truncated with a marker instead of unbounded memory growth.
	MaxOutputBytes int64
	// MaxOutputTail keeps the tail of truncated output.
	MaxOutputTail int64
	// MinEnv is the minimal environment passed to commands. nil uses a small
	// safe default (PATH, TMP, HOME-less) with NO ambient host secrets.
	MinEnv []string
	// GracePeriod before force-killing the process tree on cancel.
	GracePeriod time.Duration
}

// DefaultSubprocessOptions returns the safe defaults.
func DefaultSubprocessOptions() SubprocessOptions {
	return SubprocessOptions{
		MaxOutputBytes: 8 << 20,
		MaxOutputTail:  4 << 10,
		GracePeriod:    5 * time.Second,
	}
}

// LocalSubprocess runs commands with process-tree ownership, cancellation,
// bounded output and a fresh (non-ambient) environment.
type LocalSubprocess struct {
	opts SubprocessOptions
}

// NewLocalSubprocess returns a LocalSubprocess.
func NewLocalSubprocess(opts SubprocessOptions) *LocalSubprocess {
	if opts.MaxOutputBytes <= 0 {
		opts.MaxOutputBytes = 8 << 20
	}
	if opts.MaxOutputTail <= 0 {
		opts.MaxOutputTail = 4 << 10
	}
	if opts.GracePeriod <= 0 {
		opts.GracePeriod = 5 * time.Second
	}
	return &LocalSubprocess{opts: opts}
}

// Run executes a command (shell-free where possible) and returns bounded
// combined output. workdir is required; the process tree belongs to the
// handle and is terminated completely on cancellation (H-PROC-004).
func (s *LocalSubprocess) Run(ctx context.Context, cmd integration.Command) (integration.Result, error) {
	if strings.TrimSpace(cmd.Workdir) == "" {
		return integration.Result{}, fmt.Errorf("subprocess: workdir is required")
	}
	if strings.TrimSpace(cmd.Command) == "" {
		return integration.Result{}, fmt.Errorf("subprocess: empty command")
	}
	runCtx := ctx
	cancel := func() {}
	if cmd.Timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, cmd.Timeout)
	}
	defer cancel()

	command := exec.Command(shellExe(), shellArgs(cmd.Command)...)
	command.Dir = cmd.Workdir
	command.Env = s.commandEnv()
	attachProcessGroup(command)

	stdout := boundedBuffer{max: int(s.opts.MaxOutputBytes)}
	stderr := boundedBuffer{max: int(s.opts.MaxOutputBytes)}
	command.Stdout = &stdout
	command.Stderr = &stderr

	if err := command.Start(); err != nil {
		return integration.Result{}, fmt.Errorf("subprocess: start: %w", err)
	}
	// Best effort: bind the tree to a job object for whole-tree termination.
	// Failure is non-fatal (some environments deny OpenProcess); the direct
	// child remains killable via Go's own handle.
	_ = bindJob(command)
	done := make(chan struct{})
	var waitErr error
	go func() {
		waitErr = command.Wait()
		releaseJob(command)
		close(done)
	}()

	// Watch for cancellation/timeout and terminate the WHOLE process tree
	// (H-PROC-004). We do not rely on exec.CommandContext killing only the
	// direct child, which would orphan grandchildren (e.g. a shell + its
	// child).
	killDone := make(chan struct{})
	go func() {
		defer close(killDone)
		select {
		case <-runCtx.Done():
			killTreeGraceful(command, done, s.opts.GracePeriod)
		case <-done:
		}
	}()

	select {
	case <-runCtx.Done():
		<-killDone
		<-done // reap
		timedOut := errors.Is(runCtx.Err(), context.DeadlineExceeded)
		return integration.Result{
			ExitCode: -1,
			Output:   combinedOutput(stdout, stderr, s.opts),
			TimedOut: timedOut,
		}, fmt.Errorf("subprocess: %s: %w", cmd.Description, runCtx.Err())
	case <-done:
		<-killDone
		if waitErr != nil {
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				return integration.Result{
					ExitCode: exitErr.ExitCode(),
					Output:   combinedOutput(stdout, stderr, s.opts),
					TimedOut: runCtx.Err() == context.DeadlineExceeded,
				}, nil
			}
			return integration.Result{}, fmt.Errorf("subprocess: wait: %w", waitErr)
		}
		return integration.Result{
			ExitCode: 0,
			Output:   combinedOutput(stdout, stderr, s.opts),
		}, nil
	}
}

// killTreeGraceful terminates the process tree; if the process has not exited
// within the grace period it is force-killed. The direct child is always
// killed through Go's own handle (guaranteed), and tree containment is a
// best-effort job/group kill on top.
func killTreeGraceful(command *exec.Cmd, done <-chan struct{}, grace time.Duration) {
	terminateProcessTree(command)
	directChildKill(command)
	select {
	case <-done:
	case <-time.After(grace):
		killProcessTree(command)
		directChildKill(command)
		<-done
	}
}

// commandEnv builds a minimal, deterministic environment without ambient host
// secrets (H-ANTI-019). Callers can extend it explicitly.
func (s *LocalSubprocess) commandEnv() []string {
	if s.opts.MinEnv != nil {
		return append([]string(nil), s.opts.MinEnv...)
	}
	return []string{
		"PATH=" + os.Getenv("PATH"),
		"TMPDIR=" + os.TempDir(),
		"TEMP=" + os.TempDir(),
		"TMP=" + os.TempDir(),
		"SYSTEMROOT=" + os.Getenv("SYSTEMROOT"),
		"SystemRoot=" + os.Getenv("SystemRoot"),
		"COMSPEC=" + os.Getenv("COMSPEC"),
	}
}

type boundedBuffer struct {
	max  int
	data []byte
	over bool
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	original := len(p)
	room := b.max - len(b.data)
	if room <= 0 {
		b.over = true
		return original, nil
	}
	if len(p) > room {
		b.data = append(b.data, p[:room]...)
		b.over = true
		return original, nil
	}
	b.data = append(b.data, p...)
	return original, nil
}

func combinedOutput(stdout, stderr boundedBuffer, opts SubprocessOptions) string {
	parts := make([]string, 0, 2)
	for _, buf := range []boundedBuffer{stdout, stderr} {
		if len(buf.data) > 0 {
			parts = append(parts, string(buf.data))
		}
	}
	out := strings.Join(parts, "\n")
	if stdout.over || stderr.over {
		runes := []rune(out)
		if int64(len(runes)) > opts.MaxOutputTail {
			out = string(runes[len(runes)-int(opts.MaxOutputTail):])
		}
		out += "\n... [output truncated: exceeded " + fmt.Sprint(opts.MaxOutputBytes) + " bytes] ...\n"
	}
	return out
}

var _ integration.Subprocess = (*LocalSubprocess)(nil)
