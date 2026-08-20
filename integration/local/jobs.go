package local

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/job"
)

// JobManagerOptions bounds background jobs.
type JobManagerOptions struct {
	// MaxOutputBytes caps buffered job output.
	MaxOutputBytes int64
	// Retention limits how long finished jobs stay listed.
	Retention time.Duration
}

// DefaultJobManagerOptions returns safe defaults.
func DefaultJobManagerOptions() JobManagerOptions {
	return JobManagerOptions{MaxOutputBytes: 8 << 20, Retention: 24 * time.Hour}
}

// LocalJobManager runs background commands with owner scoping, bounded
// output, idempotent kill and retention (H-JOB-*).
type LocalJobManager struct {
	subproc *LocalSubprocess
	opts    JobManagerOptions

	mu    sync.Mutex
	jobs  map[job.ID]*localJob
	order []job.ID
	next  uint64
}

type localJob struct {
	id        job.ID
	owner     string
	kind      string
	status    job.State
	startedAt time.Time
	finished  *time.Time
	exitCode  int
	output    boundedBuffer
	cancel    context.CancelFunc
	done      chan struct{}
	lastRead  int // byte offset of last read for Tail
}

// NewLocalJobManager returns a job manager.
func NewLocalJobManager(subproc *LocalSubprocess, opts JobManagerOptions) *LocalJobManager {
	if opts.MaxOutputBytes <= 0 {
		opts.MaxOutputBytes = 8 << 20
	}
	if opts.Retention <= 0 {
		opts.Retention = 24 * time.Hour
	}
	return &LocalJobManager{subproc: subproc, opts: opts, jobs: make(map[job.ID]*localJob)}
}

// Start launches a background job. owner scopes visibility.
func (m *LocalJobManager) Start(ctx context.Context, spec job.Spec, owner string) (job.ID, error) {
	if strings.TrimSpace(spec.Workdir) == "" {
		return "", fmt.Errorf("job: workdir is required")
	}
	m.mu.Lock()
	m.next++
	id := job.ID(fmt.Sprintf("job-%d", m.next))
	innerCtx, cancel := context.WithCancel(context.Background())
	j := &localJob{
		id: id, owner: owner, kind: spec.Kind, status: job.StateRunning,
		startedAt: time.Now(), output: boundedBuffer{max: int(m.opts.MaxOutputBytes)},
		cancel: cancel, done: make(chan struct{}),
	}
	m.jobs[id] = j
	m.order = append(m.order, id)
	m.mu.Unlock()

	go func() {
		defer close(j.done)
		res, err := m.subproc.Run(innerCtx, integration.Command{
			Command: spec.Command, Workdir: spec.Workdir, Description: spec.Kind + " background job",
		})
		m.mu.Lock()
		defer m.mu.Unlock()
		j.output.Write([]byte(res.Output))
		if err != nil && !errors.Is(err, context.Canceled) {
			j.status = job.StateFailed
		} else if j.status != job.StateKilled {
			j.status = job.StateCompleted
		}
		if res.ExitCode != 0 && j.status != job.StateKilled {
			j.status = job.StateFailed
		}
		j.exitCode = res.ExitCode
		now := time.Now()
		j.finished = &now
	}()
	return id, nil
}

// List returns owner-scoped job descriptors (oldest first).
func (m *LocalJobManager) List(_ context.Context, owner string) ([]job.Descriptor, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.purgeLocked()
	out := make([]job.Descriptor, 0, len(m.order))
	for _, id := range m.order {
		j := m.jobs[id]
		if j == nil || (owner != "" && j.owner != owner) {
			continue
		}
		d := job.Descriptor{
			ID: id, Kind: j.kind, Status: j.status,
			StartedAt: j.startedAt, FinishedAt: j.finished, ExitCode: j.exitCode,
		}
		out = append(out, d)
	}
	return out, nil
}

// Output reads job output; Tail returns only bytes since the previous read.
// When Wait is set, it blocks until the job reaches a terminal state or the
// caller context/option timeout expires.
func (m *LocalJobManager) Output(ctx context.Context, id job.ID, opts job.OutputOptions, owner string) (job.Output, error) {
	m.mu.Lock()
	j, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return job.Output{}, fmt.Errorf("job %s: not found", id)
	}
	if owner != "" && j.owner != owner {
		m.mu.Unlock()
		return job.Output{}, fmt.Errorf("job %s: access denied (cross-owner)", id)
	}
	done := j.done
	running := j.status == job.StateRunning
	m.mu.Unlock()
	if opts.Wait && running {
		waitCtx := ctx
		cancel := func() {}
		if opts.Timeout > 0 {
			waitCtx, cancel = context.WithTimeout(ctx, opts.Timeout)
		}
		defer cancel()
		select {
		case <-done:
		case <-waitCtx.Done():
			return job.Output{}, waitCtx.Err()
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok = m.jobs[id]
	if !ok {
		return job.Output{}, fmt.Errorf("job %s: not found", id)
	}
	if owner != "" && j.owner != owner {
		return job.Output{}, fmt.Errorf("job %s: access denied (cross-owner)", id)
	}
	text := string(j.output.data)
	if opts.Tail {
		if j.lastRead < len(j.output.data) {
			text = string(j.output.data[j.lastRead:])
			j.lastRead = len(j.output.data)
		} else {
			text = ""
		}
	}
	if j.output.over {
		text += "\n... [output truncated] ...\n"
	}
	return job.Output{Text: text, Status: j.status}, nil
}

// Kill requests cancellation; idempotent and owner-scoped.
func (m *LocalJobManager) Kill(_ context.Context, id job.ID, reason string, owner string) error {
	m.mu.Lock()
	j, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("job %s: not found", id)
	}
	if owner != "" && j.owner != owner {
		m.mu.Unlock()
		return fmt.Errorf("job %s: access denied (cross-owner)", id)
	}
	if j.status == job.StateRunning {
		j.status = job.StateKilled
		j.cancel()
	}
	m.mu.Unlock()
	return nil
}

func (m *LocalJobManager) purgeLocked() {
	cutoff := time.Now().Add(-m.opts.Retention)
	kept := m.order[:0]
	for _, id := range m.order {
		j := m.jobs[id]
		if j != nil && j.finished != nil && j.finished.Before(cutoff) {
			delete(m.jobs, id)
			continue
		}
		kept = append(kept, id)
	}
	m.order = kept
}
