package integration

import "context"

// Terminal is the host-owned persistent terminal seam. The session key keeps
// shell state in the execution world; the core never starts a process itself.
type Terminal interface {
	Execute(ctx context.Context, sessionID, command, workdir string) (TerminalResult, error)
}

type TerminalResult struct {
	Output    string `json:"output"`
	ExitCode  int    `json:"exit_code"`
	Session   string `json:"session"`
	Truncated bool   `json:"truncated"`
}
