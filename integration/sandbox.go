package integration

import (
	"context"
	"errors"
	"fmt"
)

// Access is the operation kind used for path/command admission.
type Access string

const (
	// AccessRead permits reads (and metadata probes).
	AccessRead Access = "read"
	// AccessWrite permits mutations.
	AccessWrite Access = "write"
)

// Mode is the sandbox containment mode applied to an owner's session
// (H-FS-SBX-001..004).
type Mode string

const (
	// ModeReadOnly forbids all mutations regardless of tool prompting.
	ModeReadOnly Mode = "read-only"
	// ModeWorkspaceWrite confines mutations to workspace/temp roots.
	ModeWorkspaceWrite Mode = "workspace-write"
	// ModeDangerFullAccess is an explicit, scoped escalation.
	ModeDangerFullAccess Mode = "danger-full-access"
)

// Denial is a structured sandbox denial. It carries a stable machine code
// (SANDBOX_DENIED-*), a concrete reason, and NEVER suggests a bypass
// (H-FS-SBX-006). Tools surface it verbatim to the model.
type Denial struct {
	// Code is the stable machine-readable denial code.
	Code string
	// Reason is the concrete explanation of the boundary.
	Reason string
	// Resource identifies the denied path/command/workdir.
	Resource string
	// Tool is the tool that was denied.
	Tool string
}

func (d *Denial) Error() string {
	return fmt.Sprintf("%s: %s (%s)", d.Code, d.Reason, d.Resource)
}

// IsDenial reports whether err is a structured sandbox denial.
func IsDenial(err error) bool {
	var denial *Denial
	return errors.As(err, &denial)
}

// Sandbox is the provider-neutral execution-world boundary. It canonicalizes
// paths against a root (symlink-safe, no traversal), admits commands and
// tools, and returns structured Denial errors on violation. Filesystem and
// subprocess providers use the SAME sandbox boundary so a denied write cannot
// be bypassed through bash (H-FS-SBX-004).
type Sandbox interface {
	// Root returns the canonical containment root.
	Root() string
	// Mode returns the containment mode for the owner.
	Mode(owner string) Mode
	// ResolvePath canonicalizes rawPath against cwd and enforces containment
	// for the requested access. Returns ErrTargetOutsideRoot or *Denial.
	ResolvePath(ctx context.Context, owner, rawPath, cwd string, access Access) (Target, error)
	// CheckCommand admits a command: workdir containment plus command policy.
	// Returns *Denial on violation.
	CheckCommand(ctx context.Context, owner string, cmd Command, access Access) error
	// AllowTool admits a tool execution for the owner.
	AllowTool(ctx context.Context, owner, toolName string, mutation bool) error
}

// SandboxDenied is a convenience constructor for a denial.
func SandboxDenied(code, tool, resource, reason string) *Denial {
	return &Denial{Code: code, Tool: tool, Resource: resource, Reason: reason}
}
