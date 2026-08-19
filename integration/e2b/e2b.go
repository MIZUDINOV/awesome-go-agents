// Package e2b provides provider-neutral execution-world adapters for a remote
// (E2B-like) sandbox. agentkit never imports an E2B SDK: the host
// (wzhooh-back) supplies a concrete Transport bound to its real remote
// client, and tools talk only to the integration seams. Contract parity with
// the local adapters (containment, observation, CAS, structured denials,
// process-tree termination, bounded output) is tested against a fake
// Transport, so a tool behaves identically across execution worlds.
package e2b

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/job"
)

// Transport is the host-supplied remote execution boundary. A concrete binding
// (e.g. wzhooh-back/internal/e2b) implements it; the remote owns
// canonicalization, containment, permissions, process-tree termination and
// bounded output. Parameters mirror the integration contracts so the remote
// can enforce the SAME policy as the local Sandbox (H-THEORY: Execution World
// is an explicit seam, not a provider).
type Transport interface {
	// Resolve canonicalizes rawPath relative to cwd inside the remote root and
	// reports whether the resolved path is contained.
	Resolve(ctx context.Context, rawPath, cwd string) (resolved string, contained bool, err error)
	// Stat returns info for a resolved path.
	Stat(ctx context.Context, path string) (integration.FileInfo, error)
	// Read returns file bytes.
	Read(ctx context.Context, path string) ([]byte, error)
	// Write stores content with CAS/create-if-absent semantics enforced
	// remotely and returns the new version.
	Write(ctx context.Context, path string, content []byte, expectedVersion string, createIfAbsent, overwrite bool) (version string, created bool, err error)
	// Run executes command in workdir with a timeout (0 = none). The remote
	// owns process-tree termination and output bounding.
	Run(ctx context.Context, command, workdir string, timeoutSec int) (integration.Result, error)
	// Start/Output/Kill manage background jobs remotely (owner-scoped).
	Start(ctx context.Context, command, workdir, owner string) (string, error)
	Output(ctx context.Context, id, owner string, tail, wait bool) (job.Output, error)
	Kill(ctx context.Context, id, reason, owner string) error
	// Deny returns a structured denial if access to resource is not permitted
	// for owner; nil means allowed. Central policy lives here (mirrors
	// LocalSandbox).
	Deny(ctx context.Context, owner, resource string, access integration.Access) *integration.Denial
}

// Client adapts a Transport to the execution-world seams.
type Client struct {
	transport Transport
	obs       *observationRegistry
}

// New returns a Client wrapping transport. A nil observation registry is
// created internally (client-side, per-owner), so read-before-edit is enforced
// identically to the local adapters.
func New(transport Transport) *Client {
	return &Client{transport: transport, obs: newObservationRegistry()}
}

// Sandbox returns the remote sandbox seam.
func (c *Client) Sandbox() integration.Sandbox { return &remoteSandbox{transport: c.transport} }

// FS returns the remote filesystem seam (with observation recorder).
func (c *Client) FS() integration.FileSystem { return &remoteFS{transport: c.transport, obs: c.obs} }

// Subprocess returns the remote subprocess seam.
func (c *Client) Subprocess() integration.Subprocess {
	return &remoteSubprocess{transport: c.transport}
}

// Jobs returns the remote owner-scoped job seam.
func (c *Client) Jobs() job.ScopedManager { return &remoteJobs{transport: c.transport} }

// ---------------------------------------------------------------------------
// Sandbox

type remoteSandbox struct{ transport Transport }

func (s *remoteSandbox) Root() string { return "<remote>" }
func (s *remoteSandbox) Mode(owner string) integration.Mode {
	return integration.ModeWorkspaceWrite
}

func (s *remoteSandbox) ResolvePath(ctx context.Context, owner, rawPath, cwd string, access integration.Access) (integration.Target, error) {
	if denial := s.transport.Deny(ctx, owner, "path:"+string(access), access); denial != nil {
		return integration.Target{}, denial
	}
	resolved, contained, err := s.transport.Resolve(ctx, rawPath, cwd)
	if err != nil {
		return integration.Target{}, err
	}
	if !contained {
		return integration.Target{}, integration.SandboxDenied("SANDBOX_DENIED_OUTSIDE_ROOT", "path", rawPath,
			"the target escapes the remote workspace root")
	}
	return integration.Target{Path: resolved}, nil
}

func (s *remoteSandbox) CheckCommand(ctx context.Context, owner string, cmd integration.Command, access integration.Access) error {
	if denial := s.transport.Deny(ctx, owner, "command:"+string(access), access); denial != nil {
		return denial
	}
	if strings.TrimSpace(cmd.Workdir) == "" {
		return integration.SandboxDenied("SANDBOX_DENIED_WORKDIR", "bash", cmd.Workdir, "a working directory is required")
	}
	return nil
}

func (s *remoteSandbox) AllowTool(ctx context.Context, owner, toolName string, mutation bool) error {
	access := "tool:read"
	if mutation {
		access = "tool:write"
	}
	if denial := s.transport.Deny(ctx, owner, access+":"+toolName, integration.AccessRead); denial != nil {
		return denial
	}
	if mutation {
		if denial := s.transport.Deny(ctx, owner, access+":"+toolName, integration.AccessWrite); denial != nil {
			return denial
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// FileSystem + observation

type remoteFS struct {
	transport Transport
	obs       *observationRegistry
}

func (f *remoteFS) Resolve(ctx context.Context, rawPath, cwd string) (integration.Target, error) {
	resolved, contained, err := f.transport.Resolve(ctx, rawPath, cwd)
	if err != nil {
		return integration.Target{}, err
	}
	if !contained {
		return integration.Target{}, integration.ErrTargetOutsideRoot
	}
	return integration.Target{Path: resolved}, nil
}

func (f *remoteFS) Stat(ctx context.Context, target integration.Target) (integration.FileInfo, error) {
	return f.transport.Stat(ctx, target.Path)
}

func (f *remoteFS) ReadText(ctx context.Context, target integration.Target) (string, error) {
	data, err := f.transport.Read(ctx, target.Path)
	if err != nil {
		return "", err
	}
	if !utf8Valid(data) {
		return "", fmt.Errorf("read %s: not valid UTF-8 text", target.Path)
	}
	return string(data), nil
}

func (f *remoteFS) WriteText(ctx context.Context, target integration.Target, content string, intent integration.WriteIntent) (integration.WriteResult, error) {
	version, created, err := f.transport.Write(ctx, target.Path, []byte(content),
		intent.ExpectedVersion, intent.CreateIfAbsent, intent.Overwrite)
	if err != nil {
		return integration.WriteResult{}, err
	}
	return integration.WriteResult{Version: version, Created: created}, nil
}

func (f *remoteFS) EditText(ctx context.Context, target integration.Target, edit integration.EditRequest, intent integration.EditIntent) (integration.EditResult, error) {
	// Edit is expressed as a read-modify-write with CAS; the transport Write
	// enforces expectedVersion remotely. The literal-replacement semantics are
	// validated client-side before the CAS write for parity with LocalFileSystem.
	text, err := f.transport.Read(ctx, target.Path)
	if err != nil {
		return integration.EditResult{}, err
	}
	body := string(text)
	count := strings.Count(body, edit.OldString)
	if count == 0 {
		return integration.EditResult{}, fmt.Errorf("edit %s: old_string not found (no match)", target.Path)
	}
	if count > 1 && !edit.ReplaceAll {
		return integration.EditResult{}, fmt.Errorf("%w: %s (%d matches; use replace_all)", integration.ErrAmbiguousMatch, target.Path, count)
	}
	var next string
	if edit.ReplaceAll {
		next = strings.ReplaceAll(body, edit.OldString, edit.NewString)
	} else {
		next = strings.Replace(body, edit.OldString, edit.NewString, 1)
	}
	version, _, err := f.transport.Write(ctx, target.Path, []byte(next), intent.ExpectedVersion, false, true)
	if err != nil {
		return integration.EditResult{}, err
	}
	return integration.EditResult{Version: version, Replaced: count, OldString: edit.OldString, NewString: edit.NewString}, nil
}

func (f *remoteFS) Observe(ctx context.Context, target integration.Target, owner string, state integration.Observation, version string) error {
	f.obs.record(owner, target.Path, state, version)
	return nil
}

func (f *remoteFS) Observed(ctx context.Context, target integration.Target, owner string) (integration.Observation, string, bool) {
	return f.obs.lookup(owner, target.Path)
}

// observationRegistry records per-owner target observations (client-side).
type observationRegistry struct {
	mu sync.Mutex
	m  map[string]obsRec
}

type obsRec struct {
	state   integration.Observation
	version string
}

func newObservationRegistry() *observationRegistry {
	return &observationRegistry{m: make(map[string]obsRec)}
}

func (r *observationRegistry) record(owner, path string, state integration.Observation, version string) {
	if owner == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[owner+"\x00"+path] = obsRec{state: state, version: version}
}

func (r *observationRegistry) lookup(owner, path string) (integration.Observation, string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	rec, ok := r.m[owner+"\x00"+path]
	if !ok {
		return integration.ObservationUnseen, "", false
	}
	return rec.state, rec.version, true
}

// ---------------------------------------------------------------------------
// Subprocess

type remoteSubprocess struct{ transport Transport }

func (s *remoteSubprocess) Run(ctx context.Context, cmd integration.Command) (integration.Result, error) {
	if strings.TrimSpace(cmd.Workdir) == "" {
		return integration.Result{}, errors.New("subprocess: workdir is required")
	}
	timeoutSec := 0
	if cmd.Timeout > 0 {
		timeoutSec = int(cmd.Timeout.Seconds())
	}
	return s.transport.Run(ctx, cmd.Command, cmd.Workdir, timeoutSec)
}

// ---------------------------------------------------------------------------
// Jobs

type remoteJobs struct{ transport Transport }

func (j *remoteJobs) Start(ctx context.Context, spec job.Spec, owner string) (job.ID, error) {
	id, err := j.transport.Start(ctx, spec.Command, spec.Workdir, owner)
	if err != nil {
		return "", err
	}
	return job.ID(id), nil
}

func (j *remoteJobs) List(ctx context.Context, owner string) ([]job.Descriptor, error) {
	// Remote listing is optional; best effort via empty list.
	return nil, nil
}

func (j *remoteJobs) Output(ctx context.Context, id job.ID, opts job.OutputOptions, owner string) (job.Output, error) {
	return j.transport.Output(ctx, string(id), owner, opts.Tail, opts.Wait)
}

func (j *remoteJobs) Kill(ctx context.Context, id job.ID, reason string, owner string) error {
	return j.transport.Kill(ctx, string(id), reason, owner)
}

func utf8Valid(data []byte) bool {
	return strings.ToValidUTF8(string(data), "\uFFFD") == string(data)
}
