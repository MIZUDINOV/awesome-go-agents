package local

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
)

// LocalSandbox confines an owner's session to a root with a per-owner mode.
// The SAME boundary backs filesystem and command admission, so a denied write
// cannot be bypassed through bash (H-FS-SBX-004).
type LocalSandbox struct {
	root string
	fs   *LocalFileSystem
	mu   sync.RWMutex
	// modes maps owner -> containment mode (default workspace-write).
	modes map[string]integration.Mode
	// allowedTools maps owner -> tool allow-list ("" = default allow all).
	allowedTools map[string]map[string]bool
}

// NewLocalSandbox returns a sandbox confined to root.
func NewLocalSandbox(root string) *LocalSandbox {
	return &LocalSandbox{
		root: filepath.Clean(root),
		fs:   NewLocalFileSystem(root),
		modes: map[string]integration.Mode{
			"": integration.ModeWorkspaceWrite,
		},
		allowedTools: map[string]map[string]bool{},
	}
}

// Root returns the canonical containment root.
func (s *LocalSandbox) Root() string { return s.root }

// SetMode sets the containment mode for an owner.
func (s *LocalSandbox) SetMode(owner string, mode integration.Mode) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modes[owner] = mode
}

// Mode returns the containment mode for the owner.
func (s *LocalSandbox) Mode(owner string) integration.Mode {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if mode, ok := s.modes[owner]; ok {
		return mode
	}
	return s.modes[""]
}

// GrantTool permits a tool for an owner ("" = all owners). The allow-list is
// enforced by AllowTool at execution time (H-TOOLS-010).
func (s *LocalSandbox) GrantTool(owner, toolName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.allowedTools[owner] == nil {
		s.allowedTools[owner] = map[string]bool{}
	}
	s.allowedTools[owner][toolName] = true
}

// RevokeTool denies a tool for an owner.
func (s *LocalSandbox) RevokeTool(owner, toolName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.allowedTools[owner] == nil {
		s.allowedTools[owner] = map[string]bool{}
	}
	s.allowedTools[owner][toolName] = false
}

// ResolvePath canonicalizes and contains a path, then applies the mode/access
// policy. Containment violations and policy refusals surface as a structured
// *integration.Denial (stable SANDBOX_DENIED-* code).
func (s *LocalSandbox) ResolvePath(ctx context.Context, owner, rawPath, cwd string, access integration.Access) (integration.Target, error) {
	target, err := s.fs.Resolve(ctx, rawPath, cwd)
	if err != nil {
		if errors.Is(err, integration.ErrTargetOutsideRoot) {
			return integration.Target{}, integration.SandboxDenied("SANDBOX_DENIED_OUTSIDE_ROOT", "file", rawPath,
				"the target escapes the workspace root")
		}
		return integration.Target{}, err
	}
	if access == integration.AccessWrite && s.Mode(owner) == integration.ModeReadOnly {
		return integration.Target{}, integration.SandboxDenied("SANDBOX_DENIED_READONLY", "file", target.Path,
			"the session is read-only; write operations are blocked by policy")
	}
	if access == integration.AccessWrite && s.Mode(owner) == integration.ModeWorkspaceWrite {
		if !s.withinRoot(target.Path) {
			return integration.Target{}, integration.SandboxDenied("SANDBOX_DENIED_OUTSIDE_ROOT", "file", target.Path,
				"the target escapes the workspace root")
		}
	}
	return target, nil
}

// CheckCommand admits a command only when the execution world provides a real
// command boundary. LocalSubprocess runs an unrestricted host shell, so
// workspace-write fails closed instead of pretending that a checked workdir
// contains paths used later by the shell. Explicit danger-full-access is the
// opt-in for trusted local development; production should use an OS-enforced
// sandbox.
func (s *LocalSandbox) CheckCommand(_ context.Context, owner string, cmd integration.Command, access integration.Access) error {
	if strings.TrimSpace(cmd.Workdir) == "" {
		return integration.SandboxDenied("SANDBOX_DENIED_WORKDIR", "bash", cmd.Workdir, "a working directory is required")
	}
	mode := s.Mode(owner)
	if access == integration.AccessWrite && mode == integration.ModeReadOnly {
		return integration.SandboxDenied("SANDBOX_DENIED_READONLY", "bash", cmd.Command,
			"the session is read-only; mutating commands are blocked by policy")
	}
	if mode != integration.ModeDangerFullAccess {
		return integration.SandboxDenied("SANDBOX_DENIED_COMMAND_CONTAINMENT", "bash", cmd.Command,
			"the local execution environment cannot enforce command containment; use a real sandbox or explicit danger-full-access")
	}
	return nil
}

// AllowTool admits a tool execution (H-TOOLS-010: same visibility for schema
// lookup and execution).
func (s *LocalSandbox) AllowTool(_ context.Context, owner, toolName string, mutation bool) error {
	s.mu.RLock()
	global, hasGlobal := s.allowedTools[""]
	ownerSet, hasOwner := s.allowedTools[owner]
	globalAllowed, globalKnown := global[toolName]
	ownerAllowed, ownerKnown := ownerSet[toolName]
	s.mu.RUnlock()
	if hasGlobal && (!globalKnown || !globalAllowed) {
		return integration.SandboxDenied("SANDBOX_DENIED_TOOL", toolName, owner, "the tool is not permitted by the global session allow-list")
	}
	if hasOwner && (!ownerKnown || !ownerAllowed) {
		return integration.SandboxDenied("SANDBOX_DENIED_TOOL", toolName, owner, "the tool is not permitted for this session")
	}
	if mutation && s.Mode(owner) == integration.ModeReadOnly && isMutatingTool(toolName) {
		return integration.SandboxDenied("SANDBOX_DENIED_READONLY", toolName, owner, "the session is read-only")
	}
	return nil
}

func (s *LocalSandbox) withinRoot(path string) bool {
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(s.root, abs)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func isMutatingTool(toolName string) bool {
	switch toolName {
	case "write", "edit", "bash":
		return true
	}
	return false
}

var _ integration.Sandbox = (*LocalSandbox)(nil)
