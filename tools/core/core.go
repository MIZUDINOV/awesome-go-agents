// Package core provides the provider-neutral built-in tools
// (glob/grep/read/write/edit/bash/job_*) registered against execution-world
// seams (Sandbox, FileSystem, Subprocess, jobs). Tools depend only on the
// integration interfaces, never on a concrete environment (local host, E2B),
// so the same catalog behaves identically across execution worlds.
package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

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
}

// ReadDefaults bound the read tool (mirrors the review checklist §11.3).
type ReadDefaults struct {
	MaxLines         int
	MaxLineChars     int
	MaxSelectedBytes int
	MaxFileBytes     int64
}

// DefaultReadDefaults returns the reference bounds.
func DefaultReadDefaults() ReadDefaults {
	return ReadDefaults{MaxLines: 2000, MaxLineChars: 2000, MaxSelectedBytes: 51200, MaxFileBytes: 10 << 20}
}

// Register registers the core catalog into registry against deps. Every
// execution passes through the registry pipeline (validation, permission
// admission via the sandbox, timeout, output validation).
func Register(registry *tools.Registry, deps Deps) error {
	defs := []*tools.Definition{
		readTool(deps),
		writeTool(deps),
		editTool(deps),
		globTool(deps),
		grepTool(deps),
		bashTool(deps),
		jobStartTool(deps),
		jobOutputTool(deps),
		jobKillTool(deps),
	}
	for _, def := range defs {
		if err := registry.Register(def); err != nil {
			return fmt.Errorf("core: register %s: %w", def.Name, err)
		}
	}
	return nil
}

func cwdOf(ec tools.ExecContext) string {
	if v, ok := ec.Vars["cwd"].(string); ok {
		return v
	}
	return "."
}

func ownerOf(ec tools.ExecContext) string { return ec.SessionID }

// requireObservation returns the observation recorder or fails closed
// (read-before-edit cannot be enforced without one).
func requireObservation(fs integration.FileSystem) (integration.ObservationRecorder, error) {
	rec, ok := fs.(integration.ObservationRecorder)
	if !ok {
		return nil, fmt.Errorf("%w: execution environment does not track observations", integration.ErrNotObserved)
	}
	return rec, nil
}

// ---------------------------------------------------------------------------
// read

func readTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "read", Description: "Read a UTF-8 text file and return a bounded, line-numbered window. Use read (not cat) before editing; offsets are 1-based.",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"file_path":{"type":"string"},
				"offset":{"type":"integer"},
				"limit":{"type":"integer"}
			},
			"required":["file_path"],
			"additionalProperties":false
		}`),
		ConcurrencySafe: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				FilePath string `json:"file_path"`
				Offset   int    `json:"offset"`
				Limit    int    `json:"limit"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			defaults := DefaultReadDefaults()
			if args.Offset < 1 {
				args.Offset = 1
			}
			if args.Limit <= 0 {
				args.Limit = defaults.MaxLines
			}
			target, err := deps.Sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return nil, err
			}
			info, err := deps.FS.Stat(ctx, target)
			if err != nil {
				return nil, err
			}
			if !info.Exists {
				if rec, ok := deps.FS.(integration.ObservationRecorder); ok {
					_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationAbsent, "")
				}
				return nil, fmt.Errorf("read %s: file does not exist", args.FilePath)
			}
			if info.IsDir {
				return nil, fmt.Errorf("read %s: target is a directory", args.FilePath)
			}
			if info.Size > defaults.MaxFileBytes {
				return nil, fmt.Errorf("read %s: file exceeds %d bytes; use grep/bash with bounded output instead", args.FilePath, defaults.MaxFileBytes)
			}
			text, err := deps.FS.ReadText(ctx, target)
			if err != nil {
				return nil, err
			}
			if rec, ok := deps.FS.(integration.ObservationRecorder); ok {
				_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationPresent, info.Version)
			}
			lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
			start := args.Offset - 1
			if start > len(lines) {
				start = len(lines)
			}
			end := start + args.Limit
			if end > len(lines) {
				end = len(lines)
			}
			var rendered strings.Builder
			for i := start; i < end; i++ {
				line := truncateRunes(lines[i], defaults.MaxLineChars)
				fmt.Fprintf(&rendered, "%6d|%s\n", i+1, line)
			}
			eof := end >= len(lines)
			capReached := rendered.Len() >= defaults.MaxSelectedBytes
			footer := " (end of file)"
			switch {
			case eof:
				footer = " (end of file)"
			case capReached:
				footer = " (output truncated to size cap; use offset/limit to page)"
			default:
				footer = fmt.Sprintf(" (paged; continue with offset=%d)", end+1)
			}
			out := rendered.String()
			if len(out) > defaults.MaxSelectedBytes {
				out = out[:defaults.MaxSelectedBytes] + "\n" + footer
			} else if !strings.HasSuffix(out, footer) {
				out = strings.TrimSuffix(out, "\n") + "\n" + footer
			}
			canonical := map[string]any{
				"path": args.FilePath, "version": info.Version,
				"offset": args.Offset, "lines_returned": end - start,
				"eof": eof, "footer": footer,
			}
			return &readResult{Canonical: canonical, Model: out, UI: canonical}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			r, ok := canonical.(*readResult)
			if !ok {
				return nil, fmt.Errorf("read: bad canonical")
			}
			return r.Model, nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			r, ok := canonical.(*readResult)
			if !ok {
				return nil, fmt.Errorf("read: bad canonical")
			}
			return r.UI, nil
		},
	}
}

type readResult struct {
	Canonical map[string]any
	Model     string
	UI        map[string]any
}

// ---------------------------------------------------------------------------
// write

func writeTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "write", Description: "Create a new UTF-8 file or fully replace an existing one. Overwriting a file you have not read is refused; use read first.",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"file_path":{"type":"string"},
				"content":{"type":"string"},
				"create_if_absent":{"type":"boolean"}
			},
			"required":["file_path","content"],
			"additionalProperties":false
		}`),
		MutatesWorkspace: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				FilePath       string `json:"file_path"`
				Content        string `json:"content"`
				CreateIfAbsent bool   `json:"create_if_absent"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			target, err := deps.Sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessWrite)
			if err != nil {
				return nil, err
			}
			info, err := deps.FS.Stat(ctx, target)
			if err != nil {
				return nil, err
			}
			intent := integration.WriteIntent{CreateIfAbsent: args.CreateIfAbsent}
			if info.Exists {
				rec, err := requireObservation(deps.FS)
				if err != nil {
					return nil, err
				}
				state, version, observed := rec.Observed(ctx, target, ownerOf(ec))
				if !observed || state == integration.ObservationUnseen {
					return nil, fmt.Errorf("%w: %s (read the file first)", integration.ErrNotObserved, args.FilePath)
				}
				if args.CreateIfAbsent {
					return nil, fmt.Errorf("%w: %s", integration.ErrAlreadyExists, args.FilePath)
				}
				intent.Overwrite = true
				intent.ExpectedVersion = version
			} else {
				intent.CreateIfAbsent = true
			}
			result, err := deps.FS.WriteText(ctx, target, args.Content, intent)
			if err != nil {
				return nil, err
			}
			if rec, ok := deps.FS.(integration.ObservationRecorder); ok {
				_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationPresent, result.Version)
			}
			verb := "updated"
			if result.Created {
				verb = "created"
			}
			model := fmt.Sprintf("The file %s was %s successfully.", args.FilePath, verb)
			ui := map[string]any{"path": args.FilePath, "version": result.Version, "created": result.Created}
			return map[string]any{"path": args.FilePath, "version": result.Version, "created": result.Created, "model": model, "ui": ui}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("write: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("write: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

// ---------------------------------------------------------------------------
// edit

func editTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "edit", Description: "Apply a targeted literal replacement to a file you have read. old_string must match exactly once unless replace_all is true.",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"file_path":{"type":"string"},
				"old_string":{"type":"string"},
				"new_string":{"type":"string"},
				"replace_all":{"type":"boolean"}
			},
			"required":["file_path","old_string","new_string"],
			"additionalProperties":false
		}`),
		MutatesWorkspace: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				FilePath   string `json:"file_path"`
				OldString  string `json:"old_string"`
				NewString  string `json:"new_string"`
				ReplaceAll bool   `json:"replace_all"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			if strings.TrimSpace(args.OldString) == "" {
				return nil, fmt.Errorf("%w: old_string must not be empty", tools.ErrInvalidArguments)
			}
			if args.OldString == args.NewString {
				return nil, fmt.Errorf("%w: old_string equals new_string (no-op edit)", tools.ErrInvalidArguments)
			}
			target, err := deps.Sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessWrite)
			if err != nil {
				return nil, err
			}
			rec, err := requireObservation(deps.FS)
			if err != nil {
				return nil, err
			}
			state, version, observed := rec.Observed(ctx, target, ownerOf(ec))
			if !observed || state == integration.ObservationUnseen {
				return nil, fmt.Errorf("%w: %s (read the file first)", integration.ErrNotObserved, args.FilePath)
			}
			if state == integration.ObservationAbsent {
				return nil, fmt.Errorf("edit %s: file does not exist", args.FilePath)
			}
			result, err := deps.FS.EditText(ctx, target, integration.EditRequest{
				OldString: args.OldString, NewString: args.NewString, ReplaceAll: args.ReplaceAll,
			}, integration.EditIntent{ExpectedVersion: version})
			if err != nil {
				return nil, err
			}
			_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationPresent, result.Version)
			model := fmt.Sprintf("The file %s was updated successfully (%d replacement(s)).", args.FilePath, result.Replaced)
			return map[string]any{"path": args.FilePath, "version": result.Version, "replaced": result.Replaced, "model": model, "ui": map[string]any{"path": args.FilePath, "version": result.Version, "replaced": result.Replaced}}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("edit: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("edit: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

// ---------------------------------------------------------------------------
// glob

func globTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "glob", Description: "List files matching a path pattern under a directory (deterministic, bounded).",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"pattern":{"type":"string"},
				"path":{"type":"string"},
				"max_results":{"type":"integer"}
			},
			"required":["pattern"],
			"additionalProperties":false
		}`),
		ConcurrencySafe: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				Pattern    string `json:"pattern"`
				Path       string `json:"path"`
				MaxResults int    `json:"max_results"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			if args.Pattern == "" {
				return nil, fmt.Errorf("%w: pattern is required", tools.ErrInvalidArguments)
			}
			dir := args.Path
			if dir == "" {
				dir = "."
			}
			searchDir, err := deps.Sandbox.ResolvePath(ctx, ownerOf(ec), dir, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return nil, err
			}
			searcher, ok := deps.FS.(integration.SearchFileSystem)
			if !ok {
				return nil, fmt.Errorf("glob: execution environment does not support directory listing")
			}
			matches, err := searcher.Glob(ctx, searchDir.Path, args.Pattern, args.MaxResults)
			if err != nil {
				return nil, err
			}
			model := "No matches."
			if len(matches) > 0 {
				model = strings.Join(matches, "\n")
			}
			return map[string]any{"pattern": args.Pattern, "matches": matches, "model": model, "ui": map[string]any{"matches": matches}}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("glob: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("glob: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

// ---------------------------------------------------------------------------
// grep

func grepTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "grep", Description: "Search file contents for a regex pattern (bounded). Use read on matched files for surrounding context.",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"pattern":{"type":"string"},
				"path":{"type":"string"},
				"max_matches":{"type":"integer"}
			},
			"required":["pattern"],
			"additionalProperties":false
		}`),
		ConcurrencySafe: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				Pattern    string `json:"pattern"`
				Path       string `json:"path"`
				MaxMatches int    `json:"max_matches"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			dir := args.Path
			if dir == "" {
				dir = "."
			}
			searchDir, err := deps.Sandbox.ResolvePath(ctx, ownerOf(ec), dir, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return nil, err
			}
			searcher, ok := deps.FS.(integration.SearchFileSystem)
			if !ok {
				return nil, fmt.Errorf("grep: execution environment does not support content search")
			}
			matches, err := searcher.Grep(ctx, searchDir.Path, args.Pattern, args.MaxMatches, 4<<20)
			if err != nil {
				return nil, err
			}
			var model strings.Builder
			for _, m := range matches {
				fmt.Fprintf(&model, "%s:%d:%s\n", m.Path, m.Line, truncateRunes(m.Text, 300))
			}
			if model.Len() == 0 {
				model.WriteString("No matches.")
			} else {
				model.WriteString("\n(Use read to inspect a matched file for surrounding context.)")
			}
			type match struct {
				Path string `json:"path"`
				Line int    `json:"line"`
				Text string `json:"text"`
			}
			list := make([]match, 0, len(matches))
			for _, m := range matches {
				list = append(list, match{Path: m.Path, Line: m.Line, Text: truncateRunes(m.Text, 300)})
			}
			return map[string]any{"matches": list, "model": model.String(), "ui": map[string]any{"matches": list}}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("grep: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("grep: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

// ---------------------------------------------------------------------------
// bash

func bashTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "bash", Description: "Run a command line in a fresh shell with a sandbox-checked working directory. Output is bounded; timeouts kill the whole process tree.",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"command":{"type":"string"},
				"workdir":{"type":"string"},
				"timeout_seconds":{"type":"integer"}
			},
			"required":["command"],
			"additionalProperties":false
		}`),
		MutatesWorkspace: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				Command        string `json:"command"`
				Workdir        string `json:"workdir"`
				TimeoutSeconds int    `json:"timeout_seconds"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			workdir := args.Workdir
			if workdir == "" {
				workdir = cwdOf(ec)
			}
			cmd := integration.Command{
				Command: args.Command, Description: "bash: " + truncateRunes(args.Command, 120),
				Workdir: workdir,
			}
			if args.TimeoutSeconds > 0 {
				cmd.Timeout = time.Duration(args.TimeoutSeconds) * time.Second
			}
			if err := deps.Sandbox.CheckCommand(ctx, ownerOf(ec), cmd, integration.AccessWrite); err != nil {
				return nil, err
			}
			if deps.Subprocess == nil {
				return nil, fmt.Errorf("bash: execution environment does not support subprocesses")
			}
			result, err := deps.Subprocess.Run(ctx, cmd)
			if err != nil {
				// Cancellation/timeout carries a typed result; surface it.
				var denial *integration.Denial
				if errors.As(err, &denial) {
					return nil, err
				}
				return map[string]any{
					"exit_code": -1, "output": result.Output, "timed_out": result.TimedOut,
					"job_id": result.JobID, "error": err.Error(),
					"model": "Command failed: " + err.Error() + "\n" + result.Output,
					"ui":    map[string]any{"exit_code": -1, "timed_out": result.TimedOut},
				}, nil
			}
			model := fmt.Sprintf("exit code: %d\n%s", result.ExitCode, result.Output)
			return map[string]any{
				"exit_code": result.ExitCode, "output": result.Output,
				"timed_out": result.TimedOut, "job_id": result.JobID,
				"model": model, "ui": map[string]any{"exit_code": result.ExitCode, "timed_out": result.TimedOut},
			}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("bash: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("bash: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

// ---------------------------------------------------------------------------
// jobs

func jobStartTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "job_start", Description: "Start a long-running command as a background job; returns a durable job id.",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"command":{"type":"string"},
				"workdir":{"type":"string"}
			},
			"required":["command"],
			"additionalProperties":false
		}`),
		MutatesWorkspace: true,
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				Command string `json:"command"`
				Workdir string `json:"workdir"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			workdir := args.Workdir
			if workdir == "" {
				workdir = cwdOf(ec)
			}
			if deps.Jobs == nil {
				return nil, fmt.Errorf("job_start: execution environment does not support background jobs")
			}
			id, err := deps.Jobs.Start(ctx, job.Spec{Kind: "shell", Command: args.Command, Workdir: workdir}, ownerOf(ec))
			if err != nil {
				return nil, err
			}
			return map[string]any{"job_id": string(id), "model": "Job started: " + string(id), "ui": map[string]any{"job_id": string(id)}}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("job_start: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("job_start: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

func jobOutputTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "job_output", Description: "Read a background job's output (optionally the delta since the last read).",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"job_id":{"type":"string"},
				"wait":{"type":"boolean"},
				"tail":{"type":"boolean"}
			},
			"required":["job_id"],
			"additionalProperties":false
		}`),
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				JobID string `json:"job_id"`
				Wait  bool   `json:"wait"`
				Tail  bool   `json:"tail"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			if deps.Jobs == nil {
				return nil, fmt.Errorf("job_output: execution environment does not support background jobs")
			}
			out, err := deps.Jobs.Output(ctx, job.ID(args.JobID), job.OutputOptions{Wait: args.Wait, Tail: args.Tail}, ownerOf(ec))
			if err != nil {
				return nil, err
			}
			return map[string]any{"job_id": args.JobID, "status": string(out.Status), "output": out.Text, "model": "status: " + string(out.Status) + "\n" + out.Text, "ui": map[string]any{"status": string(out.Status)}}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("job_output: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("job_output: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

func jobKillTool(deps Deps) *tools.Definition {
	return &tools.Definition{
		Name: "job_kill", Description: "Cancel a background job (idempotent; only the owning session may kill it).",
		Version: "1",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{
				"job_id":{"type":"string"}
			},
			"required":["job_id"],
			"additionalProperties":false
		}`),
		Execute: func(ctx context.Context, ec tools.ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				JobID string `json:"job_id"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return nil, fmt.Errorf("%w: %v", tools.ErrInvalidArguments, err)
			}
			if deps.Jobs == nil {
				return nil, fmt.Errorf("job_kill: execution environment does not support background jobs")
			}
			if err := deps.Jobs.Kill(ctx, job.ID(args.JobID), "model request", ownerOf(ec)); err != nil {
				return nil, err
			}
			return map[string]any{"job_id": args.JobID, "model": "Job " + args.JobID + " killed.", "ui": map[string]any{"job_id": args.JobID}}, nil
		},
		RenderModel: func(canonical any) (any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("job_kill: bad canonical")
			}
			return m["model"], nil
		},
		PresentUI: func(canonical any) (map[string]any, error) {
			m, ok := canonical.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("job_kill: bad canonical")
			}
			return m["ui"].(map[string]any), nil
		},
	}
}

func truncateRunes(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max]) + "…"
}
