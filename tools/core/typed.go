package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"path/filepath"
	"strings"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/job"
	"github.com/MIZUDINOV/awesome-go-agents/session"
	"github.com/MIZUDINOV/awesome-go-agents/tools"
)

func typedSchema[T any]() json.RawMessage {
	schema, err := tools.FromStruct[T]()
	if err != nil {
		panic(fmt.Sprintf("core: build schema for %T: %v", *new(T), err))
	}
	return schema
}

type readInput struct {
	FilePath string `json:"file_path" jsonschema:"required,description=Sandbox-relative UTF-8 file path to read."`
	Offset   int    `json:"offset,omitempty" jsonschema:"description=1-based line offset for paging."`
	Limit    int    `json:"limit,omitempty" jsonschema:"description=Maximum number of lines to return."`
}

type readOutput struct {
	Path          string                   `json:"path"`
	Version       string                   `json:"version"`
	Offset        int                      `json:"offset"`
	LinesReturned int                      `json:"lines_returned"`
	Total         int                      `json:"total"`
	Truncated     bool                     `json:"truncated"`
	EOF           bool                     `json:"eof"`
	Footer        string                   `json:"footer"`
	FullSize      int                      `json:"full_size,omitempty"`
	Artifact      *integration.ArtifactRef `json:"artifact,omitempty"`
	Model         string                   `json:"-"`
}

func readToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[readInput, readOutput](tools.DefineToolOptions[readInput, readOutput]{
		Name: "read", Description: "Read a UTF-8 text file and return a bounded, line-numbered window. Use read (not cat) before editing; offsets are 1-based.", Version: "1",
		InputSchema: typedSchema[readInput](), OutputSchema: typedSchema[readOutput](),
		ConcurrencySafe: func(readInput) bool { return true },
		Execute: func(ctx context.Context, ec tools.ExecContext, args readInput) (readOutput, error) {
			defaults := DefaultReadDefaults()
			if args.Offset < 1 {
				args.Offset = 1
			}
			if args.Limit <= 0 {
				args.Limit = defaults.MaxLines
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return readOutput{}, err
			}
			target, err := sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return readOutput{}, err
			}
			info, err := deps.FS.Stat(ctx, target)
			if err != nil {
				return readOutput{}, err
			}
			if !info.Exists {
				if rec, ok := deps.FS.(integration.ObservationRecorder); ok {
					_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationAbsent, "")
				}
				return readOutput{}, fmt.Errorf("read %s: file does not exist", args.FilePath)
			}
			if info.IsDir {
				return readOutput{}, fmt.Errorf("read %s: target is a directory", args.FilePath)
			}
			if info.Size > defaults.MaxFileBytes {
				return readOutput{}, fmt.Errorf("read %s: file exceeds %d bytes; use grep/bash with bounded output instead", args.FilePath, defaults.MaxFileBytes)
			}
			text, err := deps.FS.ReadText(ctx, target)
			if err != nil {
				return readOutput{}, err
			}
			if rec, ok := deps.FS.(integration.ObservationRecorder); ok {
				_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationPresent, info.Version)
			}
			artifact, err := spillOutput(ctx, ec, deps.Artifacts, args.FilePath, []byte(text), "text/plain")
			if err != nil {
				return readOutput{}, err
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
			returned := 0
			capReached := false
			for i := start; i < end; i++ {
				line := fmt.Sprintf("%6d|%s\n", i+1, truncateRunes(lines[i], defaults.MaxLineChars))
				if rendered.Len()+len(line) > defaults.MaxSelectedBytes {
					capReached = true
					break
				}
				rendered.WriteString(line)
				returned++
			}
			eof := !capReached && end >= len(lines)
			footer := " (end of file)"
			switch {
			case capReached:
				footer = " (output truncated to size cap; use offset/limit to page)"
			case !eof:
				footer = fmt.Sprintf(" (paged; continue with offset=%d)", end+1)
			}
			out := strings.TrimSuffix(rendered.String(), "\n") + "\n" + footer
			fullSize := 0
			if artifact != nil {
				fullSize = len(text)
			}
			return readOutput{Path: args.FilePath, Version: info.Version, Offset: args.Offset, LinesReturned: returned, Total: len(lines), Truncated: !eof || capReached, EOF: eof, Footer: footer, FullSize: fullSize, Artifact: artifact, Model: out}, nil
		},
		RenderModel: func(value readOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value readOutput) (map[string]any, error) {
			ui := map[string]any{"path": value.Path, "version": value.Version, "offset": value.Offset, "lines_returned": value.LinesReturned, "total": value.Total, "truncated": value.Truncated, "eof": value.EOF, "footer": value.Footer}
			if value.FullSize > 0 {
				ui["full_size"] = value.FullSize
			}
			if value.Artifact != nil {
				ui["artifact"] = *value.Artifact
			}
			return ui, nil
		},
	})
}

type readImageInput struct {
	FilePath string `json:"file_path" jsonschema:"required,description=Sandbox-relative image file path."`
}

type readImageOutput struct {
	Path       string `json:"path"`
	MediaType  string `json:"media_type"`
	DataBase64 string `json:"data_base64"`
	Size       int    `json:"size"`
	Version    string `json:"version"`
}

func readImageToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[readImageInput, readImageOutput](tools.DefineToolOptions[readImageInput, readImageOutput]{
		Name: "read_image", Description: "Read a sandbox-contained image and return typed media content.", Version: "1",
		InputSchema: typedSchema[readImageInput](), OutputSchema: typedSchema[readImageOutput](),
		ConcurrencySafe: func(readImageInput) bool { return true },
		Execute: func(ctx context.Context, ec tools.ExecContext, args readImageInput) (readImageOutput, error) {
			binaryFS, ok := deps.FS.(integration.BinaryFileSystem)
			if !ok {
				return readImageOutput{}, fmt.Errorf("read_image: execution environment does not support binary reads")
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return readImageOutput{}, err
			}
			target, err := sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return readImageOutput{}, err
			}
			info, err := deps.FS.Stat(ctx, target)
			if err != nil {
				return readImageOutput{}, err
			}
			if !info.Exists || info.IsDir {
				return readImageOutput{}, fmt.Errorf("read_image: target is not a file")
			}
			if info.Size > 10<<20 {
				return readImageOutput{}, fmt.Errorf("read_image: image exceeds 10 MiB")
			}
			data, err := binaryFS.ReadBytes(ctx, target)
			if err != nil {
				return readImageOutput{}, err
			}
			mediaType := mime.TypeByExtension(filepath.Ext(args.FilePath))
			if mediaType == "" {
				mediaType = "application/octet-stream"
			}
			return readImageOutput{Path: args.FilePath, MediaType: mediaType, DataBase64: base64.StdEncoding.EncodeToString(data), Size: len(data), Version: info.Version}, nil
		},
		RenderModel: func(value readImageOutput) (any, error) { return value, nil },
		PresentUI: func(value readImageOutput) (map[string]any, error) {
			return map[string]any{"path": value.Path, "media_type": value.MediaType, "size": value.Size, "version": value.Version}, nil
		},
		FinalizeContent: func(result *tools.Result) error {
			value, ok := result.Canonical.(readImageOutput)
			if !ok || value.MediaType == "" || value.DataBase64 == "" {
				return tools.ErrInvalidOutput
			}
			if _, err := base64.StdEncoding.DecodeString(value.DataBase64); err != nil {
				return tools.ErrInvalidOutput
			}
			result.Content = []session.ContentBlock{session.MediaContentBlock(session.MediaBlock{MediaType: value.MediaType, Data: value.DataBase64})}
			return nil
		},
	})
}

type writeInput struct {
	FilePath       string `json:"file_path" jsonschema:"required,description=Sandbox-relative file path."`
	Content        string `json:"content" jsonschema:"required,description=Complete UTF-8 file content."`
	CreateIfAbsent bool   `json:"create_if_absent,omitempty" jsonschema:"description=Fail if the target already exists."`
}

type writeOutput struct {
	Path    string `json:"path"`
	Version string `json:"version"`
	Created bool   `json:"created"`
	Model   string `json:"-"`
}

func writeToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[writeInput, writeOutput](tools.DefineToolOptions[writeInput, writeOutput]{
		Name: "write", Description: "Create a new UTF-8 file or fully replace an existing one. Overwriting a file you have not read is refused; use read first.", Version: "1", MutatesWorkspace: true,
		InputSchema: typedSchema[writeInput](), OutputSchema: typedSchema[writeOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args writeInput) (writeOutput, error) {
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return writeOutput{}, err
			}
			target, err := sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessWrite)
			if err != nil {
				return writeOutput{}, err
			}
			info, err := deps.FS.Stat(ctx, target)
			if err != nil {
				return writeOutput{}, err
			}
			intent := integration.WriteIntent{CreateIfAbsent: args.CreateIfAbsent}
			if info.Exists {
				rec, err := requireObservation(deps.FS)
				if err != nil {
					return writeOutput{}, err
				}
				state, version, observed := rec.Observed(ctx, target, ownerOf(ec))
				if !observed || state == integration.ObservationUnseen {
					return writeOutput{}, fmt.Errorf("%w: %s (read the file first)", integration.ErrNotObserved, args.FilePath)
				}
				if args.CreateIfAbsent {
					return writeOutput{}, fmt.Errorf("%w: %s", integration.ErrAlreadyExists, args.FilePath)
				}
				intent.Overwrite = true
				intent.ExpectedVersion = version
			} else {
				intent.CreateIfAbsent = true
			}
			result, err := deps.FS.WriteText(ctx, target, args.Content, intent)
			if err != nil {
				return writeOutput{}, err
			}
			if rec, ok := deps.FS.(integration.ObservationRecorder); ok {
				_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationPresent, result.Version)
			}
			verb := "updated"
			if result.Created {
				verb = "created"
			}
			return writeOutput{Path: args.FilePath, Version: result.Version, Created: result.Created, Model: fmt.Sprintf("The file %s was %s successfully.", args.FilePath, verb)}, nil
		},
		RenderModel: func(value writeOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value writeOutput) (map[string]any, error) {
			return map[string]any{"path": value.Path, "version": value.Version, "created": value.Created}, nil
		},
	})
}

type editInput struct {
	FilePath   string `json:"file_path" jsonschema:"required,description=Sandbox-relative file path."`
	OldString  string `json:"old_string" jsonschema:"required,description=Literal text that must match in the file."`
	NewString  string `json:"new_string" jsonschema:"required,description=Replacement literal text."`
	ReplaceAll bool   `json:"replace_all,omitempty" jsonschema:"description=Replace every match instead of requiring one match."`
}

type editOutput struct {
	Path     string `json:"path"`
	Version  string `json:"version"`
	Replaced int    `json:"replaced"`
	Model    string `json:"-"`
}

func editToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[editInput, editOutput](tools.DefineToolOptions[editInput, editOutput]{
		Name: "edit", Description: "Apply a targeted literal replacement to a file you have read. old_string must match exactly once unless replace_all is true.", Version: "1", MutatesWorkspace: true,
		InputSchema: typedSchema[editInput](), OutputSchema: typedSchema[editOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args editInput) (editOutput, error) {
			if strings.TrimSpace(args.OldString) == "" {
				return editOutput{}, fmt.Errorf("%w: old_string must not be empty", tools.ErrInvalidArguments)
			}
			if args.OldString == args.NewString {
				return editOutput{}, fmt.Errorf("%w: old_string equals new_string (no-op edit)", tools.ErrInvalidArguments)
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return editOutput{}, err
			}
			target, err := sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessWrite)
			if err != nil {
				return editOutput{}, err
			}
			rec, err := requireObservation(deps.FS)
			if err != nil {
				return editOutput{}, err
			}
			state, version, observed := rec.Observed(ctx, target, ownerOf(ec))
			if !observed || state == integration.ObservationUnseen {
				return editOutput{}, fmt.Errorf("%w: %s (read the file first)", integration.ErrNotObserved, args.FilePath)
			}
			if state == integration.ObservationAbsent {
				return editOutput{}, fmt.Errorf("edit %s: file does not exist", args.FilePath)
			}
			result, err := deps.FS.EditText(ctx, target, integration.EditRequest{OldString: args.OldString, NewString: args.NewString, ReplaceAll: args.ReplaceAll}, integration.EditIntent{ExpectedVersion: version})
			if err != nil {
				return editOutput{}, err
			}
			_ = rec.Observe(ctx, target, ownerOf(ec), integration.ObservationPresent, result.Version)
			return editOutput{Path: args.FilePath, Version: result.Version, Replaced: result.Replaced, Model: fmt.Sprintf("The file %s was updated successfully (%d replacement(s)).", args.FilePath, result.Replaced)}, nil
		},
		RenderModel: func(value editOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value editOutput) (map[string]any, error) {
			return map[string]any{"path": value.Path, "version": value.Version, "replaced": value.Replaced}, nil
		},
	})
}

type globInput struct {
	Pattern    string `json:"pattern" jsonschema:"required,description=Glob pattern to match."`
	Path       string `json:"path,omitempty" jsonschema:"description=Sandbox-relative directory to search."`
	MaxResults int    `json:"max_results,omitempty" jsonschema:"description=Maximum number of matches."`
}

type globOutput struct {
	Pattern       string   `json:"pattern"`
	Matches       []string `json:"matches"`
	Total         int      `json:"total"`
	Truncated     bool     `json:"truncated"`
	TotalComplete bool     `json:"total_complete"`
	Model         string   `json:"-"`
}

func globToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[globInput, globOutput](tools.DefineToolOptions[globInput, globOutput]{
		Name: "glob", Description: "List files matching a path pattern under a directory (deterministic, bounded).", Version: "1",
		InputSchema: typedSchema[globInput](), OutputSchema: typedSchema[globOutput](), ConcurrencySafe: func(globInput) bool { return true },
		Execute: func(ctx context.Context, ec tools.ExecContext, args globInput) (globOutput, error) {
			if args.Pattern == "" {
				return globOutput{}, fmt.Errorf("%w: pattern is required", tools.ErrInvalidArguments)
			}
			dir := args.Path
			if dir == "" {
				dir = "."
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return globOutput{}, err
			}
			searchDir, err := sandbox.ResolvePath(ctx, ownerOf(ec), dir, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return globOutput{}, err
			}
			searcher, ok := deps.FS.(integration.SearchFileSystem)
			if !ok {
				return globOutput{}, fmt.Errorf("glob: execution environment does not support directory listing")
			}
			max := args.MaxResults
			if max <= 0 {
				max = 100
			}
			matches, err := searcher.Glob(ctx, searchDir.Path, args.Pattern, max)
			if err != nil {
				return globOutput{}, err
			}
			if matches == nil {
				matches = []string{}
			}
			truncated := len(matches) >= max
			model := "No matches."
			if len(matches) > 0 {
				model = strings.Join(matches, "\n")
			}
			return globOutput{Pattern: args.Pattern, Matches: matches, Total: len(matches), Truncated: truncated, TotalComplete: !truncated, Model: model}, nil
		},
		RenderModel: func(value globOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value globOutput) (map[string]any, error) {
			return map[string]any{"matches": value.Matches, "total": value.Total, "truncated": value.Truncated}, nil
		},
	})
}

type grepInput struct {
	Pattern    string `json:"pattern" jsonschema:"required,description=Regular expression to search for."`
	Path       string `json:"path,omitempty" jsonschema:"description=Sandbox-relative directory to search."`
	MaxMatches int    `json:"max_matches,omitempty" jsonschema:"description=Maximum number of matching lines."`
}

type grepMatch struct {
	Path string `json:"path"`
	Line int    `json:"line"`
	Text string `json:"text"`
}

type grepOutput struct {
	Matches       []grepMatch `json:"matches"`
	Total         int         `json:"total"`
	Truncated     bool        `json:"truncated"`
	TotalComplete bool        `json:"total_complete"`
	Model         string      `json:"-"`
}

func grepToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[grepInput, grepOutput](tools.DefineToolOptions[grepInput, grepOutput]{
		Name: "grep", Description: "Search file contents for a regex pattern (bounded). Use read on matched files for surrounding context.", Version: "1",
		InputSchema: typedSchema[grepInput](), OutputSchema: typedSchema[grepOutput](), ConcurrencySafe: func(grepInput) bool { return true },
		Execute: func(ctx context.Context, ec tools.ExecContext, args grepInput) (grepOutput, error) {
			dir := args.Path
			if dir == "" {
				dir = "."
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return grepOutput{}, err
			}
			searchDir, err := sandbox.ResolvePath(ctx, ownerOf(ec), dir, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return grepOutput{}, err
			}
			searcher, ok := deps.FS.(integration.SearchFileSystem)
			if !ok {
				return grepOutput{}, fmt.Errorf("grep: execution environment does not support content search")
			}
			max := args.MaxMatches
			if max <= 0 {
				max = 200
			}
			matches, err := searcher.Grep(ctx, searchDir.Path, args.Pattern, max, 4<<20)
			if err != nil {
				return grepOutput{}, err
			}
			list := make([]grepMatch, 0, len(matches))
			var model strings.Builder
			for _, match := range matches {
				text := truncateRunes(match.Text, 300)
				list = append(list, grepMatch{Path: match.Path, Line: match.Line, Text: text})
				fmt.Fprintf(&model, "%s:%d:%s\n", match.Path, match.Line, text)
			}
			if model.Len() == 0 {
				model.WriteString("No matches.")
			} else {
				model.WriteString("\n(Use read to inspect a matched file for surrounding context.)")
			}
			truncated := len(matches) >= max
			return grepOutput{Matches: list, Total: len(list), Truncated: truncated, TotalComplete: !truncated, Model: model.String()}, nil
		},
		RenderModel: func(value grepOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value grepOutput) (map[string]any, error) {
			return map[string]any{"matches": value.Matches, "total": value.Total, "truncated": value.Truncated}, nil
		},
	})
}

type bashInput struct {
	Command        string `json:"command" jsonschema:"required,description=Command line to execute."`
	Workdir        string `json:"workdir,omitempty" jsonschema:"description=Sandbox-relative working directory."`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty" jsonschema:"description=Execution timeout in seconds."`
}

type bashOutput struct {
	ExitCode  int                      `json:"exit_code"`
	Output    string                   `json:"output"`
	Total     int                      `json:"total"`
	Truncated bool                     `json:"truncated"`
	TimedOut  bool                     `json:"timed_out"`
	JobID     string                   `json:"job_id"`
	Error     string                   `json:"error,omitempty"`
	Artifact  *integration.ArtifactRef `json:"artifact,omitempty"`
	Model     string                   `json:"-"`
}

func bashToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[bashInput, bashOutput](tools.DefineToolOptions[bashInput, bashOutput]{
		Name: "bash", Description: "Run a command line in a fresh shell with a sandbox-checked working directory. Output is bounded; timeouts kill the whole process tree.", Version: "1", MutatesWorkspace: true,
		InputSchema: typedSchema[bashInput](), OutputSchema: typedSchema[bashOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args bashInput) (bashOutput, error) {
			workdir := args.Workdir
			if workdir == "" {
				workdir = cwdOf(ec)
			}
			cmd := integration.Command{Command: args.Command, Description: "bash: " + truncateRunes(args.Command, 120), Workdir: workdir}
			if args.TimeoutSeconds > 0 {
				cmd.Timeout = time.Duration(args.TimeoutSeconds) * time.Second
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return bashOutput{}, err
			}
			if err := sandbox.CheckCommand(ctx, ownerOf(ec), cmd, integration.AccessWrite); err != nil {
				return bashOutput{}, err
			}
			if deps.Subprocess == nil {
				return bashOutput{}, fmt.Errorf("bash: execution environment does not support subprocesses")
			}
			result, runErr := deps.Subprocess.Run(ctx, cmd)
			bounded, truncated := boundOutput(result.Output)
			artifact, err := spillOutput(ctx, ec, deps.Artifacts, "bash-output", []byte(result.Output), "text/plain")
			if err != nil {
				return bashOutput{}, err
			}
			if runErr != nil {
				var denial *integration.Denial
				if errors.As(runErr, &denial) {
					return bashOutput{}, runErr
				}
				return bashOutput{ExitCode: -1, Output: bounded, Total: len(result.Output), Truncated: truncated, TimedOut: result.TimedOut, JobID: result.JobID, Error: runErr.Error(), Artifact: artifact, Model: "Command failed: " + runErr.Error() + "\n" + bounded}, nil
			}
			return bashOutput{ExitCode: result.ExitCode, Output: bounded, Total: len(result.Output), Truncated: truncated, TimedOut: result.TimedOut, JobID: result.JobID, Artifact: artifact, Model: fmt.Sprintf("exit code: %d\n%s", result.ExitCode, bounded)}, nil
		},
		RenderModel: func(value bashOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value bashOutput) (map[string]any, error) {
			return map[string]any{"exit_code": value.ExitCode, "timed_out": value.TimedOut}, nil
		},
	})
}

type jobStartInput struct {
	Command string `json:"command" jsonschema:"required,description=Command line to run in the background."`
	Workdir string `json:"workdir,omitempty" jsonschema:"description=Sandbox-relative working directory."`
}

type jobStartOutput struct {
	JobID string `json:"job_id"`
	Model string `json:"-"`
}

func jobStartToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[jobStartInput, jobStartOutput](tools.DefineToolOptions[jobStartInput, jobStartOutput]{
		Name: "job_start", Description: "Start a long-running command as a background job; returns a durable job id.", Version: "1", MutatesWorkspace: true,
		InputSchema: typedSchema[jobStartInput](), OutputSchema: typedSchema[jobStartOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args jobStartInput) (jobStartOutput, error) {
			workdir := args.Workdir
			if workdir == "" {
				workdir = cwdOf(ec)
			}
			if deps.Jobs == nil {
				return jobStartOutput{}, fmt.Errorf("job_start: execution environment does not support background jobs")
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return jobStartOutput{}, err
			}
			if err := sandbox.CheckCommand(ctx, ownerOf(ec), integration.Command{Command: args.Command, Workdir: workdir, Description: "job_start: " + truncateRunes(args.Command, 120)}, integration.AccessWrite); err != nil {
				return jobStartOutput{}, err
			}
			id, err := deps.Jobs.Start(ctx, job.Spec{Kind: "shell", Command: args.Command, Workdir: workdir}, ownerOf(ec))
			if err != nil {
				return jobStartOutput{}, err
			}
			return jobStartOutput{JobID: string(id), Model: "Job started: " + string(id)}, nil
		},
		RenderModel: func(value jobStartOutput) (any, error) { return value.Model, nil },
		PresentUI:   func(value jobStartOutput) (map[string]any, error) { return map[string]any{"job_id": value.JobID}, nil },
	})
}

type jobListInput struct{}

type jobListOutput struct {
	Jobs  []job.Descriptor `json:"jobs"`
	Total int              `json:"total"`
}

func jobListToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[jobListInput, jobListOutput](tools.DefineToolOptions[jobListInput, jobListOutput]{
		Name: "job_list", Description: "List background jobs owned by this session.", Version: "1", InputSchema: typedSchema[jobListInput](), OutputSchema: typedSchema[jobListOutput](), ConcurrencySafe: func(jobListInput) bool { return true },
		Execute: func(ctx context.Context, ec tools.ExecContext, _ jobListInput) (jobListOutput, error) {
			if deps.Jobs == nil {
				return jobListOutput{}, fmt.Errorf("job_list: execution environment does not support background jobs")
			}
			jobs, err := deps.Jobs.List(ctx, ownerOf(ec))
			if err != nil {
				return jobListOutput{}, err
			}
			if jobs == nil {
				jobs = []job.Descriptor{}
			}
			return jobListOutput{Jobs: jobs, Total: len(jobs)}, nil
		},
	})
}

type jobOutputInput struct {
	JobID          string `json:"job_id" jsonschema:"required,description=Owned background job identifier."`
	Wait           bool   `json:"wait,omitempty" jsonschema:"description=Wait until the job reaches a terminal state."`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty" jsonschema:"description=Maximum seconds to wait when wait is true."`
	Tail           bool   `json:"tail,omitempty" jsonschema:"description=Return output since the previous read when supported."`
}

type jobOutputOutput struct {
	JobID     string                   `json:"job_id"`
	Status    string                   `json:"status"`
	Output    string                   `json:"output"`
	Total     int                      `json:"total"`
	Truncated bool                     `json:"truncated"`
	Artifact  *integration.ArtifactRef `json:"artifact,omitempty"`
	Model     string                   `json:"-"`
}

func jobOutputToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[jobOutputInput, jobOutputOutput](tools.DefineToolOptions[jobOutputInput, jobOutputOutput]{
		Name: "job_output", Description: "Read a background job's output (optionally the delta since the last read).", Version: "1", InputSchema: typedSchema[jobOutputInput](), OutputSchema: typedSchema[jobOutputOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args jobOutputInput) (jobOutputOutput, error) {
			if deps.Jobs == nil {
				return jobOutputOutput{}, fmt.Errorf("job_output: execution environment does not support background jobs")
			}
			if args.TimeoutSeconds < 0 {
				return jobOutputOutput{}, fmt.Errorf("%w: timeout_seconds must not be negative", tools.ErrInvalidArguments)
			}
			out, err := deps.Jobs.Output(ctx, job.ID(args.JobID), job.OutputOptions{Wait: args.Wait, Timeout: time.Duration(args.TimeoutSeconds) * time.Second, Tail: args.Tail}, ownerOf(ec))
			if err != nil {
				return jobOutputOutput{}, err
			}
			bounded, truncated := boundOutput(out.Text)
			artifact, err := spillOutput(ctx, ec, deps.Artifacts, string(args.JobID), []byte(out.Text), "text/plain")
			if err != nil {
				return jobOutputOutput{}, err
			}
			status := string(out.Status)
			return jobOutputOutput{JobID: args.JobID, Status: status, Output: bounded, Total: len(out.Text), Truncated: truncated, Artifact: artifact, Model: "status: " + status + "\n" + bounded}, nil
		},
		RenderModel: func(value jobOutputOutput) (any, error) { return value.Model, nil },
		PresentUI: func(value jobOutputOutput) (map[string]any, error) {
			return map[string]any{"status": value.Status, "truncated": value.Truncated}, nil
		},
	})
}

type jobKillInput struct {
	JobID string `json:"job_id" jsonschema:"required,description=Owned background job identifier to cancel."`
}

type jobKillOutput struct {
	JobID string `json:"job_id"`
	Model string `json:"-"`
}

func jobKillToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[jobKillInput, jobKillOutput](tools.DefineToolOptions[jobKillInput, jobKillOutput]{
		Name: "job_kill", Description: "Cancel a background job (idempotent; only the owning session may kill it).", Version: "1", InputSchema: typedSchema[jobKillInput](), OutputSchema: typedSchema[jobKillOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args jobKillInput) (jobKillOutput, error) {
			if deps.Jobs == nil {
				return jobKillOutput{}, fmt.Errorf("job_kill: execution environment does not support background jobs")
			}
			if err := deps.Jobs.Kill(ctx, job.ID(args.JobID), "model request", ownerOf(ec)); err != nil {
				return jobKillOutput{}, err
			}
			return jobKillOutput{JobID: args.JobID, Model: "Job " + args.JobID + " killed."}, nil
		},
		RenderModel: func(value jobKillOutput) (any, error) { return value.Model, nil },
		PresentUI:   func(value jobKillOutput) (map[string]any, error) { return map[string]any{"job_id": value.JobID}, nil },
	})
}

type webSearchInput struct {
	Query      string `json:"query" jsonschema:"required,description=Search query."`
	MaxResults int    `json:"max_results,omitempty" jsonschema:"description=Maximum number of search results."`
}

type webSearchOutput struct {
	Query         string                     `json:"query"`
	Results       []integration.SearchResult `json:"results"`
	Total         int                        `json:"total"`
	Truncated     bool                       `json:"truncated"`
	TotalComplete bool                       `json:"total_complete"`
}

func webSearchToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[webSearchInput, webSearchOutput](tools.DefineToolOptions[webSearchInput, webSearchOutput]{
		Name: "web_search", Description: "Search the web through the host-provided bounded web port.", Version: "1", InputSchema: typedSchema[webSearchInput](), OutputSchema: typedSchema[webSearchOutput](), ConcurrencySafe: func(webSearchInput) bool { return true },
		Execute: func(ctx context.Context, _ tools.ExecContext, args webSearchInput) (webSearchOutput, error) {
			if deps.Web == nil {
				return webSearchOutput{}, fmt.Errorf("web_search: web port is not configured")
			}
			if strings.TrimSpace(args.Query) == "" {
				return webSearchOutput{}, fmt.Errorf("%w: query is required", tools.ErrInvalidArguments)
			}
			max := args.MaxResults
			if max <= 0 {
				max = 8
			}
			results, err := deps.Web.Search(ctx, args.Query, integration.SearchOptions{MaxResults: max})
			if err != nil {
				return webSearchOutput{}, err
			}
			if results == nil {
				results = []integration.SearchResult{}
			}
			truncated := len(results) >= max
			return webSearchOutput{Query: args.Query, Results: results, Total: len(results), Truncated: truncated, TotalComplete: !truncated}, nil
		},
		RenderModel: func(value webSearchOutput) (any, error) { return value, nil },
		PresentUI: func(value webSearchOutput) (map[string]any, error) {
			return map[string]any{"query": value.Query, "results": value.Results, "total": value.Total, "truncated": value.Truncated, "total_complete": value.TotalComplete}, nil
		},
	})
}

type webFetchInput struct {
	URL      string `json:"url" jsonschema:"required,description=Absolute URL to fetch."`
	MaxBytes int    `json:"max_bytes,omitempty" jsonschema:"description=Maximum number of response bytes."`
}

type webFetchOutput struct {
	URL           string `json:"url"`
	Title         string `json:"title"`
	Content       string `json:"content"`
	Total         int    `json:"total"`
	Truncated     bool   `json:"truncated"`
	TotalComplete bool   `json:"total_complete"`
}

func webFetchToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[webFetchInput, webFetchOutput](tools.DefineToolOptions[webFetchInput, webFetchOutput]{
		Name: "web_fetch", Description: "Fetch a bounded web document through the host-provided web port.", Version: "1", InputSchema: typedSchema[webFetchInput](), OutputSchema: typedSchema[webFetchOutput](), ConcurrencySafe: func(webFetchInput) bool { return true },
		Execute: func(ctx context.Context, _ tools.ExecContext, args webFetchInput) (webFetchOutput, error) {
			if deps.Web == nil {
				return webFetchOutput{}, fmt.Errorf("web_fetch: web port is not configured")
			}
			if strings.TrimSpace(args.URL) == "" {
				return webFetchOutput{}, fmt.Errorf("%w: url is required", tools.ErrInvalidArguments)
			}
			max := args.MaxBytes
			if max <= 0 {
				max = 200000
			}
			doc, err := deps.Web.Fetch(ctx, args.URL, integration.FetchOptions{MaxBytes: max})
			if err != nil {
				return webFetchOutput{}, err
			}
			if doc == nil {
				return webFetchOutput{}, fmt.Errorf("web_fetch: empty document")
			}
			content := doc.Content
			total := len([]byte(content))
			complete := true
			if content == "" && doc.Reader != nil {
				body, err := io.ReadAll(io.LimitReader(doc.Reader, int64(max)+1))
				if err != nil {
					return webFetchOutput{}, err
				}
				content = string(body)
				total = len(body)
				complete = len(body) <= max
			}
			truncated := total > max
			if truncated {
				content = strings.ToValidUTF8(string([]byte(content)[:max]), "�")
			}
			return webFetchOutput{URL: doc.URL, Title: doc.Title, Content: content, Total: total, Truncated: truncated, TotalComplete: complete && !truncated}, nil
		},
		RenderModel: func(value webFetchOutput) (any, error) { return value, nil },
		PresentUI: func(value webFetchOutput) (map[string]any, error) {
			return map[string]any{"url": value.URL, "title": value.Title, "content": value.Content, "total": value.Total, "truncated": value.Truncated, "total_complete": value.TotalComplete}, nil
		},
	})
}

type lspDiagnosticsInput struct {
	FilePath string `json:"file_path" jsonschema:"required,description=Sandbox-relative file path for diagnostics."`
}

type lspDiagnosticsOutput struct {
	FilePath    string                   `json:"file_path"`
	Diagnostics []integration.Diagnostic `json:"diagnostics"`
	Total       int                      `json:"total"`
	Truncated   bool                     `json:"truncated"`
}

func lspDiagnosticsToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[lspDiagnosticsInput, lspDiagnosticsOutput](tools.DefineToolOptions[lspDiagnosticsInput, lspDiagnosticsOutput]{
		Name: "lsp_diagnostics", Description: "Read bounded diagnostics from the host-provided language server.", Version: "1", InputSchema: typedSchema[lspDiagnosticsInput](), OutputSchema: typedSchema[lspDiagnosticsOutput](), ConcurrencySafe: func(lspDiagnosticsInput) bool { return true },
		Execute: func(ctx context.Context, ec tools.ExecContext, args lspDiagnosticsInput) (lspDiagnosticsOutput, error) {
			if deps.LSP == nil {
				return lspDiagnosticsOutput{}, fmt.Errorf("lsp_diagnostics: LSP port is not configured")
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return lspDiagnosticsOutput{}, err
			}
			target, err := sandbox.ResolvePath(ctx, ownerOf(ec), args.FilePath, cwdOf(ec), integration.AccessRead)
			if err != nil {
				return lspDiagnosticsOutput{}, err
			}
			diagnostics, err := deps.LSP.Diagnostics(ctx, target.Path)
			if err != nil {
				return lspDiagnosticsOutput{}, err
			}
			if diagnostics == nil {
				diagnostics = []integration.Diagnostic{}
			}
			return lspDiagnosticsOutput{FilePath: args.FilePath, Diagnostics: diagnostics, Total: len(diagnostics)}, nil
		},
		RenderModel: func(value lspDiagnosticsOutput) (any, error) { return value, nil },
		PresentUI: func(value lspDiagnosticsOutput) (map[string]any, error) {
			return map[string]any{"file_path": value.FilePath, "diagnostics": value.Diagnostics, "total": value.Total, "truncated": value.Truncated}, nil
		},
	})
}

type terminalInput struct {
	Command string `json:"command" jsonschema:"required,description=Command line to execute in the persistent terminal."`
	Workdir string `json:"workdir,omitempty" jsonschema:"description=Sandbox-relative working directory."`
}

type terminalOutput struct {
	Command   string                   `json:"command"`
	Workdir   string                   `json:"workdir"`
	Output    string                   `json:"output"`
	Total     int                      `json:"total"`
	ExitCode  int                      `json:"exit_code"`
	Session   string                   `json:"session"`
	Truncated bool                     `json:"truncated"`
	Artifact  *integration.ArtifactRef `json:"artifact,omitempty"`
}

func terminalToolTyped(deps Deps) *tools.Definition {
	return tools.DefineTool[terminalInput, terminalOutput](tools.DefineToolOptions[terminalInput, terminalOutput]{
		Name: "terminal", Description: "Execute a command in the host-owned persistent terminal session.", Version: "1", MutatesWorkspace: true, InputSchema: typedSchema[terminalInput](), OutputSchema: typedSchema[terminalOutput](),
		Execute: func(ctx context.Context, ec tools.ExecContext, args terminalInput) (terminalOutput, error) {
			if deps.Terminal == nil {
				return terminalOutput{}, fmt.Errorf("terminal: terminal port is not configured")
			}
			if strings.TrimSpace(args.Command) == "" {
				return terminalOutput{}, fmt.Errorf("%w: command is required", tools.ErrInvalidArguments)
			}
			if args.Workdir == "" {
				args.Workdir = cwdOf(ec)
			}
			sandbox, err := requireSandbox(deps)
			if err != nil {
				return terminalOutput{}, err
			}
			if err := sandbox.CheckCommand(ctx, ownerOf(ec), integration.Command{Command: args.Command, Workdir: args.Workdir, Description: "terminal: " + truncateRunes(args.Command, 120)}, integration.AccessWrite); err != nil {
				return terminalOutput{}, err
			}
			result, err := deps.Terminal.Execute(ctx, ownerOf(ec), args.Command, args.Workdir)
			if err != nil {
				return terminalOutput{}, err
			}
			bounded, outputTruncated := boundOutput(result.Output)
			artifact, err := spillOutput(ctx, ec, deps.Artifacts, args.Command, []byte(result.Output), "text/plain")
			if err != nil {
				return terminalOutput{}, err
			}
			return terminalOutput{Command: args.Command, Workdir: args.Workdir, Output: bounded, Total: len(result.Output), ExitCode: result.ExitCode, Session: result.Session, Truncated: result.Truncated || outputTruncated, Artifact: artifact}, nil
		},
		RenderModel: func(value terminalOutput) (any, error) { return value, nil },
		PresentUI:   func(value terminalOutput) (map[string]any, error) { return valueMap(value), nil },
	})
}

func valueMap(value terminalOutput) map[string]any {
	return map[string]any{"command": value.Command, "workdir": value.Workdir, "output": value.Output, "total": value.Total, "exit_code": value.ExitCode, "session": value.Session, "truncated": value.Truncated}
}
