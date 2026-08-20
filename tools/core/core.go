// Package core provides the provider-neutral built-in tools registered against
// execution-world seams. Each tool's public contract lives in typed.go.
package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/job"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

// Deps wires the execution-world seams into the core catalog.
type Deps struct {
	Sandbox    integration.Sandbox
	FS         integration.FileSystem
	Subprocess integration.Subprocess
	Jobs       job.ScopedManager
	Artifacts  integration.ArtifactStore
	Web        integration.Web
	LSP        integration.LSP
	Terminal   integration.Terminal
}

// ReadDefaults bounds the read tool.
type ReadDefaults struct {
	MaxLines         int
	MaxLineChars     int
	MaxSelectedBytes int
	MaxFileBytes     int64
}

// DefaultReadDefaults returns the reference bounds.
func DefaultReadDefaults() ReadDefaults {
	return ReadDefaults{MaxLines: 2000, MaxLineChars: 2000, MaxSelectedBytes: 24 << 10, MaxFileBytes: 10 << 20}
}

// Register registers every typed core tool into registry.
func Register(registry *tools.Registry, deps Deps) error {
	defs := []*tools.Definition{
		readToolTyped(deps), readImageToolTyped(deps), writeToolTyped(deps),
		editToolTyped(deps), globToolTyped(deps), grepToolTyped(deps),
		bashToolTyped(deps), jobStartToolTyped(deps), jobListToolTyped(deps),
		jobOutputToolTyped(deps), jobKillToolTyped(deps), webSearchToolTyped(deps),
		webFetchToolTyped(deps), lspDiagnosticsToolTyped(deps), terminalToolTyped(deps),
	}
	for _, def := range defs {
		if err := registry.Register(def); err != nil {
			return fmt.Errorf("core: register %s: %w", def.Name, err)
		}
	}
	return nil
}

func cwdOf(ec tools.ExecContext) string {
	if value, ok := ec.Vars["cwd"].(string); ok {
		return value
	}
	return "."
}

func ownerOf(ec tools.ExecContext) string { return ec.SessionID }

func requireSandbox(deps Deps) (integration.Sandbox, error) {
	if deps.Sandbox == nil {
		return nil, fmt.Errorf("core: sandbox admission port is required")
	}
	return deps.Sandbox, nil
}

// requireObservation fails closed when read-before-edit cannot be enforced.
func requireObservation(fs integration.FileSystem) (integration.ObservationRecorder, error) {
	recorder, ok := fs.(integration.ObservationRecorder)
	if !ok {
		return nil, fmt.Errorf("%w: execution environment does not track observations", integration.ErrNotObserved)
	}
	return recorder, nil
}

func truncateRunes(value string, max int) string {
	runes := []rune(value)
	if len(runes) <= max {
		return value
	}
	return string(runes[:max]) + "…"
}

const maxToolOutputBytes = 24 << 10

func boundOutput(value string) (string, bool) {
	if len(value) <= maxToolOutputBytes {
		return value, false
	}
	return strings.ToValidUTF8(value[:maxToolOutputBytes], "�"), true
}

func spillOutput(ctx context.Context, ec tools.ExecContext, store integration.ArtifactStore, name string, data []byte, contentType string) (*integration.ArtifactRef, error) {
	if store == nil || len(data) <= maxToolOutputBytes {
		return nil, nil
	}
	ref, err := store.Put(ctx, ownerOf(ec), name, append([]byte(nil), data...), contentType)
	if err != nil {
		return nil, fmt.Errorf("core: spill output: %w", err)
	}
	if ref.ID == "" {
		return nil, fmt.Errorf("core: spill output: artifact store returned an empty reference")
	}
	ref.Size = int64(len(data))
	digest := sha256.Sum256(data)
	ref.SHA256 = hex.EncodeToString(digest[:])
	return &ref, nil
}
