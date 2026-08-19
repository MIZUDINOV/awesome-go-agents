//go:build windows

package local

import (
	"fmt"
	"os/exec"
	"sync"
	"syscall"
	"unsafe"
)

// Windows process-tree termination strategy, in order of preference:
//
//  1. Job Object (best effort): children are assigned to a job shortly after
//     start; TerminateJobObject kills the whole tree (the reliable H-PROC-004
//     semantics). Binding may be refused by the environment (restricted
//     tokens can deny OpenProcess), in which case we degrade gracefully.
//  2. Direct-child kill via Go's own process handle (always available and
//     instant — TerminateProcess on the handle CreateProcess returned).
//  3. taskkill /T /F as a final extra attempt when no job was bound.
//
// The direct-child kill is the guaranteed step; tree containment is best
// effort where the OS denies OpenProcess.

var jobs struct {
	sync.Mutex
	m map[*exec.Cmd]syscall.Handle
}

const (
	jobObjectExtendedLimitInfo   = 9
	jobObjectLimitKillOnJobClose = 0x2000
	processSetQuota              = 0x0100
	processTerminate             = 0x0001
)

type jobObjectBasicLimitInformation struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type jobObjectExtendedLimitInformation struct {
	BasicLimitInformation jobObjectBasicLimitInformation
	IoInfo                [6]uint64
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	procCreateJobObjectW         = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJobObject  = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJobObject = kernel32.NewProc("AssignProcessToJobObject")
	procTerminateJobObject       = kernel32.NewProc("TerminateJobObject")
	procOpenProcess              = kernel32.NewProc("OpenProcess")
)

func shellExe() string { return "cmd.exe" }
func shellArgs(command string) []string {
	return []string{"/C", command}
}

// attachProcessGroup has nothing to prepare on Windows; job binding happens in
// bindJob right after Start.
func attachProcessGroup(cmd *exec.Cmd) {}

// bindJob attempts to assign the child to a Job Object for tree termination.
// Failures are non-fatal: the command still runs and the direct child stays
// killable through Go's process handle.
func bindJob(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	procHandle, _, err := procOpenProcess.Call(processSetQuota|processTerminate, 0, uintptr(cmd.Process.Pid))
	if procHandle == 0 {
		return fmt.Errorf("OpenProcess: %w", err)
	}
	defer syscall.CloseHandle(syscall.Handle(procHandle))

	jobHandle, _, err := procCreateJobObjectW.Call(0, 0)
	if jobHandle == 0 {
		return fmt.Errorf("CreateJobObject: %w", err)
	}
	info := jobObjectExtendedLimitInformation{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	r, _, err := procSetInformationJobObject.Call(jobHandle, jobObjectExtendedLimitInfo,
		uintptr(unsafe.Pointer(&info)), unsafe.Sizeof(info))
	if r == 0 {
		_ = syscall.CloseHandle(syscall.Handle(jobHandle))
		return fmt.Errorf("SetInformationJobObject: %w", err)
	}
	r, _, err = procAssignProcessToJobObject.Call(jobHandle, procHandle)
	if r == 0 {
		// Already in another job (e.g. parent is job-contained): continue
		// without tree kill.
		_ = syscall.CloseHandle(syscall.Handle(jobHandle))
		return fmt.Errorf("AssignProcessToJobObject: %w", err)
	}
	jobs.Lock()
	if jobs.m == nil {
		jobs.m = make(map[*exec.Cmd]syscall.Handle)
	}
	jobs.m[cmd] = syscall.Handle(jobHandle)
	jobs.Unlock()
	return nil
}

// terminateProcessTree kills the job (whole tree) if bound, else falls back to
// taskkill /T /F.
func terminateProcessTree(cmd *exec.Cmd) {
	h := takeJob(cmd)
	if h != 0 {
		_, _, _ = procTerminateJobObject.Call(uintptr(h), 1)
		releaseJob(cmd)
		return
	}
	if cmd.Process == nil {
		return
	}
	_ = exec.Command("taskkill", "/PID", fmt.Sprint(cmd.Process.Pid), "/T", "/F").Run()
}

// killProcessTree is the force step; on Windows TerminateJobObject / taskkill
// are already definitive.
func killProcessTree(cmd *exec.Cmd) {
	terminateProcessTree(cmd)
}

// directChildKill terminates the exact process through Go's own handle.
func directChildKill(cmd *exec.Cmd) {
	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}

func takeJob(cmd *exec.Cmd) syscall.Handle {
	jobs.Lock()
	h := jobs.m[cmd]
	jobs.Unlock()
	return h
}

// releaseJob closes the job handle after the process exits.
func releaseJob(cmd *exec.Cmd) {
	jobs.Lock()
	h := jobs.m[cmd]
	delete(jobs.m, cmd)
	jobs.Unlock()
	if h != 0 {
		_ = syscall.CloseHandle(h)
	}
}
