// Package integration defines provider-neutral capability seams used by
// agent tools (filesystem, subprocess, web). The agent core depends only on
// these interfaces; concrete bound environments (local host, E2B sandbox,
// remote execution) implement them outside the core.
package integration

import (
	"context"
	"errors"
	"time"
)

// Target is a resolved, sandbox-safe path. Implementations perform path
// resolution against their own root/cwd and prevent escapes.
type Target struct {
	Path string
}

// FileInfo describes a resolved target without exposing host-specific types.
type FileInfo struct {
	Exists  bool
	IsDir   bool
	Size    int64
	Version string // content version for CAS (hash or mtime-based)
}

// WriteIntent controls write semantics.
type WriteIntent struct {
	// ExpectedVersion, when non-empty, enables optimistic concurrency: the
	// write succeeds only if the current version matches (CAS guard).
	ExpectedVersion string
	// CreateIfAbsent fails the write if the file already exists.
	CreateIfAbsent bool
	// Overwrite guards full-content replaces; conservative default.
	Overwrite bool
}

// WriteResult reports the outcome of a write.
type WriteResult struct {
	Version string
	// Created is true when a new file was created.
	Created bool
}

// EditRequest is a literal, targeted replacement (old_string must match).
type EditRequest struct {
	OldString string
	NewString string
	// ReplaceAll replaces every occurrence instead of requiring uniqueness.
	ReplaceAll bool
}

// EditIntent mirrors WriteIntent for CAS on edits.
type EditIntent struct {
	ExpectedVersion string
}

// EditResult reports the edit outcome.
type EditResult struct {
	Version   string
	Replaced  int
	OldString string
	NewString string
}

// FileSystem is the provider-neutral filesystem seam. Local and sandbox (E2B)
// implementations satisfy it; tools never know which is active.
type FileSystem interface {
	// Resolve binds a raw path against cwd and returns a sandbox-safe Target.
	Resolve(ctx context.Context, rawPath, cwd string) (Target, error)
	// Stat returns metadata for a resolved target.
	Stat(ctx context.Context, target Target) (FileInfo, error)
	// ReadText returns the file content.
	ReadText(ctx context.Context, target Target) (string, error)
	// WriteText creates or replaces a file with CAS support.
	WriteText(ctx context.Context, target Target, content string, intent WriteIntent) (WriteResult, error)
	// EditText performs a targeted replacement with CAS support.
	EditText(ctx context.Context, target Target, edit EditRequest, intent EditIntent) (EditResult, error)
}

// Sentinel errors returned by FileSystem implementations.
var (
	ErrNotObserved = errors.New("file was not read before edit (read the file, then retry)")
	ErrStaleVersion = errors.New("file version changed since it was read (re-read, then retry)")
	ErrAlreadyExists = errors.New("file already exists (createIfAbsent)")
	ErrAmbiguousMatch = errors.New("old_string matched more than once (use replace_all or a more specific match)")
	ErrTargetOutsideRoot = errors.New("resolved target escapes the sandbox root")
)

// Subprocess is the provider-neutral command execution seam.
type Subprocess interface {
	// Run executes a command and returns combined output. workdir is required
	// (commands are stateless shell processes; do not rely on cd).
	Run(ctx context.Context, cmd Command) (Result, error)
}

// Command is a provider-neutral command request.
type Command struct {
	// Command is the full command line (e.g. "pnpm test").
	Command string
	// Description is human-readable, for telemetry/approvals.
	Description string
	// Workdir is the working directory (required; commands do not share state).
	Workdir string
	// Timeout bounds execution; zero uses the provider default.
	Timeout time.Duration
}

// Result is a completed command outcome.
type Result struct {
	ExitCode int
	Output   string
	TimedOut bool
	// JobID is set when the command ran in the background.
	JobID string
}
