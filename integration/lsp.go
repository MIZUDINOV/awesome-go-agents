package integration

import "context"

// LSP is the provider-neutral language-server seam. Concrete servers remain
// host-owned; the core only defines bounded, structured diagnostics.
type LSP interface {
	Diagnostics(ctx context.Context, path string) ([]Diagnostic, error)
}

type Diagnostic struct {
	Path     string `json:"path"`
	Line     int    `json:"line"`
	Column   int    `json:"column"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}
