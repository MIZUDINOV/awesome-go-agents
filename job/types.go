// Package job provides a generic background-job abstraction shared by shell
// commands, and (in future) other long-running work. Jobs are identified by a
// durable ID and expose list/output/kill semantics.
package job

import (
	"context"
	"time"
)

// ID is a durable job identifier.
type ID string

// State is the lifecycle state of a background job.
type State string

const (
	StateRunning   State = "running"
	StateCompleted State = "completed"
	StateFailed    State = "failed"
	StateKilled    State = "killed"
)

// Descriptor describes a job for listing.
type Descriptor struct {
	ID         ID
	Kind       string // e.g. "shell"
	Status     State
	StartedAt  time.Time
	FinishedAt *time.Time
	ExitCode   int
}

// OutputOptions controls job_output.
type OutputOptions struct {
	// Wait blocks until the job reaches a terminal state.
	Wait bool
	// Timeout bounds the wait (0 = provider default).
	Timeout time.Duration
	// Tail returns only output since the previous read.
	Tail bool
}

// Manager is the provider-neutral background-job seam.
type Manager interface {
	// Start launches a job and returns its ID.
	Start(ctx context.Context, spec Spec) (ID, error)
	// List returns known jobs (running and finished).
	List(ctx context.Context) ([]Descriptor, error)
	// Output reads job output (respecting OutputOptions).
	Output(ctx context.Context, id ID, opts OutputOptions) (Output, error)
	// Kill requests cancellation of a job.
	Kill(ctx context.Context, id ID, reason string) error
}

// ScopedManager is the owner-scoped job seam used by agent tools. Every
// operation is scoped by owner (session/agent id): cross-owner access is
// denied, kills are idempotent, and finished jobs respect retention
// (H-JOB-001..006).
type ScopedManager interface {
	Start(ctx context.Context, spec Spec, owner string) (ID, error)
	List(ctx context.Context, owner string) ([]Descriptor, error)
	Output(ctx context.Context, id ID, opts OutputOptions, owner string) (Output, error)
	Kill(ctx context.Context, id ID, reason string, owner string) error
}

// Spec describes what a job runs. For shell jobs, Command/Workdir are used.
type Spec struct {
	Kind    string
	Command string
	Workdir string
	// Extra carries kind-specific options.
	Extra map[string]any
}

// Output is buffered job output.
type Output struct {
	// Text is the accumulated output (or tail when Tail was set).
	Text string
	// Status reflects the job state at read time.
	Status State
}

// OutputSince returns the textual suffix after a marker; used by Tail reads
// when the provider does not track read offsets natively.
func (o Output) OutputSince(marker string) string {
	if marker == "" {
		return o.Text
	}
	for i := 0; i+len(marker) <= len(o.Text); i++ {
		if o.Text[i:i+len(marker)] == marker {
			return o.Text[i+len(marker):]
		}
	}
	return o.Text
}
