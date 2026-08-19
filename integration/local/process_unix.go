//go:build !windows

package local

import (
	"os/exec"
	"syscall"
)

func shellExe() string { return "/bin/sh" }
func shellArgs(command string) []string {
	return []string{"-c", command}
}

// attachProcessGroup puts the child in its own process group (setsid) so the
// whole tree can be signalled on cancellation.
func attachProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}

func terminateProcessTree(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
}

func killProcessTree(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
}

// On Unix the process group is established at Start via Setsid; no extra bind
// or release step is needed.
func bindJob(cmd *exec.Cmd) error { return nil }
func releaseJob(cmd *exec.Cmd)    {}

// directChildKill kills the exact process through Go's own handle.
func directChildKill(cmd *exec.Cmd) {
	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
