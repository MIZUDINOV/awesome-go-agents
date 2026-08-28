// Package local provides concrete local-host implementations of the
// provider-neutral seams (FileSystem, Subprocess, JobManager, Sandbox). They
// enforce the same containment, observation and CAS contracts that remote
// adapters enforce, so tools behave identically across execution worlds.
package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
)

// Observation state constants are re-exported from the integration contract.
type Observation = integration.Observation

// LocalFileSystem is a root-confined FileSystem implementation with per-owner
// observation state and content-version CAS.
type LocalFileSystem struct {
	// Root is the canonical containment root. All resolution happens against
	// it; escapes are rejected on the canonical path, not the raw string
	// (H-FS-010).
	Root string

	mu   sync.Mutex
	obs  map[obsKey]obsEntry
	lock sync.Mutex // per-target mutation serialization
}

type obsKey struct {
	owner string
	path  string
}

type obsEntry struct {
	state   Observation
	version integration.FileVersion
}

// NewLocalFileSystem returns a filesystem confined to root.
func NewLocalFileSystem(root string) *LocalFileSystem {
	return &LocalFileSystem{Root: filepath.Clean(root), obs: make(map[obsKey]obsEntry)}
}

// Version computes the content version (SHA-256 hex) of data.
func Version(data []byte) integration.FileVersion {
	sum := sha256.Sum256(data)
	return integration.FileVersion(hex.EncodeToString(sum[:]))
}

// Resolve canonicalizes rawPath against cwd and confines it to Root
// (H-FS-002/010). Symlinks are resolved through the deepest existing ancestor
// so a symlink escape is rejected on the canonical target. Absolute raw paths
// are checked directly against the root instead of being joined onto cwd.
func (f *LocalFileSystem) Resolve(_ context.Context, rawPath, cwd string) (integration.Target, error) {
	if strings.TrimSpace(rawPath) == "" {
		return integration.Target{}, fmt.Errorf("resolve: empty path")
	}
	raw := filepath.FromSlash(rawPath)
	var joined string
	if filepath.IsAbs(raw) {
		joined = raw
	} else {
		base := f.Root
		if strings.TrimSpace(cwd) != "" {
			base = cwd
		}
		joined = filepath.Join(base, raw)
	}
	canonical, err := canonicalWithin(joined, f.Root)
	if err != nil {
		return integration.Target{}, err
	}
	displayPath := filepath.ToSlash(relToRoot(canonical, f.Root))
	if integration.IsProtectedWorkspacePath(displayPath) {
		return integration.Target{}, integration.SandboxDenied("SANDBOX_DENIED_PROTECTED_PATH", "file", displayPath,
			"protected workspace metadata and credentials are not available to model-facing tools")
	}
	return integration.Target{Key: canonical, Path: canonical, DisplayPath: displayPath}, nil
}

func relToRoot(target, root string) string {
	rel, err := filepath.Rel(root, target)
	if err != nil || rel == "." {
		return "."
	}
	return rel
}

// canonicalWithin resolves path to a clean absolute path and verifies it stays
// within root (symlink-safe through the existing ancestor chain).
func canonicalWithin(path, root string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve absolute path: %w", err)
	}
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve root: %w", err)
	}
	// Resolve symlinks of the deepest existing ancestor.
	resolved := resolveDeepestExisting(abs)
	rel, err := filepath.Rel(rootAbs, resolved)
	if err != nil {
		return "", fmt.Errorf("%w: %s", integration.ErrTargetOutsideRoot, abs)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("%w: %s", integration.ErrTargetOutsideRoot, abs)
	}
	return resolved, nil
}

func resolveDeepestExisting(path string) string {
	current := filepath.Clean(path)
	for {
		evaluated, err := filepath.EvalSymlinks(current)
		if err == nil {
			// Resolve the remainder (not-yet-existing tail) onto the evaluated
			// ancestor, then re-clean.
			rel, relErr := filepath.Rel(current, path)
			if relErr != nil || rel == "." {
				return evaluated
			}
			return filepath.Join(evaluated, rel)
		}
		parent := filepath.Dir(current)
		if parent == current {
			return path
		}
		current = parent
	}
}

func (f *LocalFileSystem) Stat(_ context.Context, target integration.Target) (integration.FileInfo, error) {
	info, err := os.Stat(target.Path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return integration.FileInfo{Exists: false}, nil
		}
		return integration.FileInfo{}, classifyLocalFileError("stat", target.Path, err)
	}
	version, err := fileVersion(target.Path, info)
	if err != nil {
		return integration.FileInfo{}, classifyLocalFileError("stat version", target.Path, err)
	}
	return integration.FileInfo{
		Exists:  true,
		IsDir:   info.IsDir(),
		Size:    info.Size(),
		Version: version,
		Kind:    fileKind(info.Mode()),
	}, nil
}

func fileKind(mode fs.FileMode) integration.FileKind {
	switch {
	case mode.IsRegular():
		return integration.FileKindFile
	case mode.IsDir():
		return integration.FileKindDirectory
	case mode&fs.ModeSymlink != 0:
		return integration.FileKindSymlink
	default:
		return integration.FileKindOther
	}
}

// fileVersion derives a content version from the file (hash) or, for
// directories, a size+mtime digest.
func fileVersion(path string, info os.FileInfo) (integration.FileVersion, error) {
	if info.IsDir() {
		sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d:%d", path, info.Size(), info.ModTime().UnixNano())))
		return integration.FileVersion(hex.EncodeToString(sum[:])), nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return Version(data), nil
}

func (f *LocalFileSystem) ReadText(ctx context.Context, target integration.Target) (string, error) {
	read, err := f.ReadTextVersioned(ctx, target, 0)
	return read.Content, err
}

func (f *LocalFileSystem) ReadTextVersioned(ctx context.Context, target integration.Target, maxBytes int64) (integration.TextRead, error) {
	if err := ctx.Err(); err != nil {
		return integration.TextRead{}, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", err)
	}
	info, err := os.Stat(target.Path)
	if err != nil {
		return integration.TextRead{}, classifyLocalFileError("read", target.Path, err)
	}
	if !info.Mode().IsRegular() {
		return integration.TextRead{}, integration.NewFSError(integration.FSNotRegularFile, false, "select a regular text file", nil)
	}
	if maxBytes > 0 && info.Size() > maxBytes {
		return integration.TextRead{}, integration.NewFSError(integration.FSTooLarge, false, "use a bounded search or artifact for large files", nil)
	}
	data, err := os.ReadFile(target.Path)
	if err != nil {
		return integration.TextRead{}, classifyLocalFileError("read", target.Path, err)
	}
	if maxBytes > 0 && int64(len(data)) > maxBytes {
		return integration.TextRead{}, integration.NewFSError(integration.FSTooLarge, false, "use a bounded search or artifact for large files", nil)
	}
	if !utf8Valid(data) {
		return integration.TextRead{}, integration.NewFSError(integration.FSNotText, false, "select a UTF-8 text file", nil)
	}
	return integration.TextRead{Content: string(data), Version: Version(data), Size: int64(len(data))}, nil
}

func (f *LocalFileSystem) ReadBytes(ctx context.Context, target integration.Target) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", err)
	}
	data, err := os.ReadFile(target.Path)
	if err != nil {
		return nil, classifyLocalFileError("read bytes", target.Path, err)
	}
	return data, nil
}

func (f *LocalFileSystem) WriteText(ctx context.Context, target integration.Target, content string, intent integration.WriteIntent) (integration.WriteResult, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	kind, expectedVersion, err := intent.Normalized()
	if err != nil {
		return integration.WriteResult{}, err
	}
	if err := ensureParentInsideRoot(target.Path, f.Root); err != nil {
		return integration.WriteResult{}, err
	}

	info, err := os.Stat(target.Path)
	exists := err == nil
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return integration.WriteResult{}, classifyLocalFileError("write", target.Path, err)
	}
	if exists && info.IsDir() {
		return integration.WriteResult{}, integration.NewFSError(integration.FSNotRegularFile, false, "select a regular file path", nil)
	}

	if kind == integration.WriteIntentCreateIfAbsent && exists {
		return integration.WriteResult{}, integration.NewFSError(integration.FSAlreadyExists, true, "read the existing file before replacing it", integration.ErrAlreadyExists)
	}
	if kind == integration.WriteIntentReplaceIfVersion {
		if !exists {
			return integration.WriteResult{}, staleLocalFileError(target.Path)
		}
		current, versionErr := fileVersion(target.Path, info)
		if versionErr != nil {
			return integration.WriteResult{}, classifyLocalFileError("write version", target.Path, versionErr)
		}
		if current != expectedVersion {
			return integration.WriteResult{}, staleLocalFileError(target.Path)
		}
	}
	if err := ctx.Err(); err != nil {
		return integration.WriteResult{}, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", err)
	}

	if err := atomicWrite(target.Path, []byte(content), kind == integration.WriteIntentCreateIfAbsent); err != nil {
		if errors.Is(err, integration.ErrAlreadyExists) {
			return integration.WriteResult{}, integration.NewFSError(integration.FSAlreadyExists, true, "read the existing file before replacing it", err)
		}
		return integration.WriteResult{}, classifyLocalFileError("write commit", target.Path, err)
	}
	return integration.WriteResult{Version: Version([]byte(content)), Created: !exists}, nil
}

func (f *LocalFileSystem) EditText(ctx context.Context, target integration.Target, edit integration.EditRequest, intent integration.EditIntent) (integration.EditResult, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if edit.OldString == "" {
		return integration.EditResult{}, fmt.Errorf("edit %s: old_string must not be empty", target.Path)
	}
	if edit.OldString == edit.NewString {
		return integration.EditResult{}, fmt.Errorf("edit %s: old_string equals new_string (no-op edit)", target.Path)
	}
	info, err := os.Stat(target.Path)
	if err != nil {
		return integration.EditResult{}, classifyLocalFileError("edit", target.Path, err)
	}
	if intent.ExpectedVersion != "" {
		current, versionErr := fileVersion(target.Path, info)
		if versionErr != nil {
			return integration.EditResult{}, classifyLocalFileError("edit version", target.Path, versionErr)
		}
		if current != intent.ExpectedVersion {
			return integration.EditResult{}, staleLocalFileError(target.Path)
		}
	}
	data, err := os.ReadFile(target.Path)
	if err != nil {
		return integration.EditResult{}, classifyLocalFileError("edit read", target.Path, err)
	}
	if !utf8Valid(data) {
		return integration.EditResult{}, integration.NewFSError(integration.FSNotText, false, "select a UTF-8 text file", nil)
	}
	text := string(data)
	normalized, crlf := normalizeLineEndings(text)
	oldString, _ := normalizeLineEndings(edit.OldString)
	newString, _ := normalizeLineEndings(edit.NewString)
	count := strings.Count(normalized, oldString)
	if count == 0 {
		return integration.EditResult{}, integration.NewFSError(integration.FSEditNotFound, false, "re-read the file and provide an exact literal", nil)
	}
	if count > 1 && !edit.ReplaceAll {
		return integration.EditResult{}, integration.NewFSError(integration.FSAmbiguousEdit, false, "use replace_all or a more specific literal", integration.ErrAmbiguousMatch)
	}
	replaced := count
	newText := normalized
	if edit.ReplaceAll {
		newText = strings.ReplaceAll(normalized, oldString, newString)
	} else {
		newText = strings.Replace(normalized, oldString, newString, 1)
	}
	if crlf {
		newText = strings.ReplaceAll(newText, "\n", "\r\n")
	}
	if err := ctx.Err(); err != nil {
		return integration.EditResult{}, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", err)
	}
	if err := atomicWrite(target.Path, []byte(newText), false); err != nil {
		return integration.EditResult{}, classifyLocalFileError("edit commit", target.Path, err)
	}
	return integration.EditResult{
		Version: Version([]byte(newText)), Replaced: replaced,
		OldString: edit.OldString, NewString: edit.NewString,
	}, nil
}

func (f *LocalFileSystem) Delete(ctx context.Context, target integration.Target, expectedVersion integration.FileVersion) (integration.DeleteResult, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	if expectedVersion == "" {
		return integration.DeleteResult{}, integration.NewFSError(integration.FSNotObserved, false, "read the file before deleting it", integration.ErrNotObserved)
	}
	info, err := os.Stat(target.Path)
	if err != nil {
		return integration.DeleteResult{}, classifyLocalFileError("delete", target.Path, err)
	}
	if !info.Mode().IsRegular() {
		return integration.DeleteResult{}, integration.NewFSError(integration.FSNotRegularFile, false, "select a regular file", nil)
	}
	current, versionErr := fileVersion(target.Path, info)
	if versionErr != nil {
		return integration.DeleteResult{}, classifyLocalFileError("delete version", target.Path, versionErr)
	}
	if current != expectedVersion {
		return integration.DeleteResult{}, staleLocalFileError(target.Path)
	}
	if err := ctx.Err(); err != nil {
		return integration.DeleteResult{}, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", err)
	}
	if err := os.Remove(target.Path); err != nil {
		return integration.DeleteResult{}, classifyLocalFileError("delete commit", target.Path, err)
	}
	return integration.DeleteResult{Deleted: true}, nil
}

func classifyLocalFileError(operation, path string, err error) error {
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return integration.NewFSError(integration.FSNotFound, false, "read the current workspace state", err)
	case errors.Is(err, fs.ErrPermission):
		return integration.NewFSError(integration.FSPermissionDenied, false, "use an accessible workspace file", err)
	default:
		return integration.NewFSError(integration.FSIO, true, "retry the filesystem operation", fmt.Errorf("%s %s: %w", operation, path, err))
	}
}

func staleLocalFileError(path string) error {
	return integration.NewFSError(integration.FSStaleVersion, true, "re-read the file, then retry", fmt.Errorf("%w: %s", integration.ErrStaleVersion, path))
}

// Glob lists files under dir matching pattern (sorted, bounded).
func (f *LocalFileSystem) Glob(ctx context.Context, dir, pattern string, maxResults int) ([]string, error) {
	root, err := canonicalWithin(dir, f.Root)
	if err != nil {
		return nil, err
	}
	if maxResults <= 0 {
		maxResults = 100
	}
	if _, err := filepath.Match(filepath.FromSlash(pattern), ""); err != nil {
		return nil, integration.NewFSError(integration.FSInvalidPath, false, "provide a valid glob pattern", err)
	}
	var matches []string
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip hidden dirs and vendor/node_modules by default depth rule.
			name := d.Name()
			if name != root && (strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" || name == ".git") {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if integration.IsProtectedWorkspacePath(filepath.ToSlash(rel)) {
			return nil
		}
		matched, err := filepath.Match(filepath.FromSlash(pattern), rel)
		if err != nil {
			return fmt.Errorf("glob pattern %q: %w", pattern, err)
		}
		if matched {
			matches = append(matches, filepath.ToSlash(rel))
			if len(matches) >= maxResults {
				return fs.ErrClosed // stop walking
			}
		}
		return nil
	})
	if errors.Is(walkErr, fs.ErrClosed) {
		walkErr = nil
	}
	if walkErr != nil {
		if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
			return nil, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", walkErr)
		}
		return nil, classifyLocalFileError("glob", root, walkErr)
	}
	sort.Strings(matches)
	return matches, nil
}

// Grep scans regular text files under dir for the literal/regex pattern.
// maxMatches bounds results; lines carry 1-based numbers.
func (f *LocalFileSystem) Grep(ctx context.Context, dir, pattern string, maxMatches int, maxBytesPerFile int64) ([]integration.GrepMatch, error) {
	root, err := canonicalWithin(dir, f.Root)
	if err != nil {
		return nil, err
	}
	if maxMatches <= 0 {
		maxMatches = 200
	}
	if maxBytesPerFile <= 0 {
		maxBytesPerFile = 1 << 20
	}
	var out []integration.GrepMatch
	re, err := compilePattern(pattern)
	if err != nil {
		return nil, integration.NewFSError(integration.FSInvalidPath, false, "provide a valid regular expression", err)
	}
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if name != root && (strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" || name == ".git") {
				return filepath.SkipDir
			}
			return nil
		}
		if len(out) >= maxMatches {
			return fs.ErrClosed
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if integration.IsProtectedWorkspacePath(filepath.ToSlash(rel)) {
			return nil
		}
		if info.Size() > maxBytesPerFile {
			return nil // skip oversized binaries
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if !utf8Valid(data) {
			return nil
		}
		lines := strings.Split(string(data), "\n")
		for i, line := range lines {
			if re.MatchString(line) {
				out = append(out, integration.GrepMatch{Path: filepath.ToSlash(rel), Line: i + 1, Text: boundLine(line)})
				if len(out) >= maxMatches {
					return fs.ErrClosed
				}
			}
		}
		return nil
	})
	if errors.Is(walkErr, fs.ErrClosed) {
		walkErr = nil
	}
	if walkErr != nil {
		if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
			return nil, integration.NewFSError(integration.FSAborted, true, "retry after cancellation", walkErr)
		}
		return nil, classifyLocalFileError("grep", root, walkErr)
	}
	return out, nil
}

func boundLine(line string) string {
	runes := []rune(line)
	if len(runes) > 300 {
		return string(runes[:300]) + "…"
	}
	return line
}

// Observation types are re-exported for convenience.
const (
	ObservationUnseen  = integration.ObservationUnseen
	ObservationAbsent  = integration.ObservationAbsent
	ObservationPresent = integration.ObservationPresent
)

// record registers an observation for the owner's session (H-FS-OBS-001..006).
func (f *LocalFileSystem) record(target integration.Target, owner string, state integration.Observation, version integration.FileVersion) {
	if owner == "" {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.obs[obsKey{owner: owner, path: target.Path}] = obsEntry{state: state, version: version}
}

// Observe marks a target observed (read path). State is scoped per owner and
// never leaks between users (H-FS-OBS-007).
func (f *LocalFileSystem) Observe(_ context.Context, target integration.Target, owner string, state integration.Observation, version integration.FileVersion) error {
	f.record(target, owner, state, version)
	return nil
}

// Observed returns the recorded state/version for owner+target.
func (f *LocalFileSystem) Observed(_ context.Context, target integration.Target, owner string) (integration.Observation, integration.FileVersion, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	entry, ok := f.obs[obsKey{owner: owner, path: target.Path}]
	if !ok {
		return integration.ObservationUnseen, "", false
	}
	return entry.state, entry.version, true
}

// ObservedVersion returns the recorded version for owner+target and whether
// the target was observed.
func (f *LocalFileSystem) ObservedVersion(owner string, target integration.Target) (integration.FileVersion, bool) {
	_, version, ok := f.Observed(context.Background(), target, owner)
	return version, ok
}

// ObservedState returns the recorded observation state.
func (f *LocalFileSystem) ObservedState(owner string, target integration.Target) (integration.Observation, integration.FileVersion, bool) {
	return f.Observed(context.Background(), target, owner)
}

func ensureParentInsideRoot(target, root string) error {
	parent := filepath.Dir(target)
	if _, err := canonicalWithin(parent, root); err != nil {
		return integration.NewFSError(integration.FSSandboxDenied, false, "use a workspace-contained path", err)
	}
	if err := os.MkdirAll(parent, 0o750); err != nil {
		return integration.NewFSError(integration.FSParentCreateFailed, false, "use a writable workspace directory", err)
	}
	if _, err := canonicalWithin(parent, root); err != nil {
		return integration.NewFSError(integration.FSSandboxDenied, false, "the parent resolved outside the workspace", err)
	}
	return nil
}

func atomicWrite(path string, data []byte, createOnly bool) error {
	dir := filepath.Dir(path)
	stagingDir, err := os.MkdirTemp(dir, ".agentkit-staging-*")
	if err != nil {
		return fmt.Errorf("write %s: create staging directory: %w", path, err)
	}
	if err := os.Chmod(stagingDir, 0o700); err != nil {
		_ = os.RemoveAll(stagingDir)
		return fmt.Errorf("write %s: secure staging directory: %w", path, err)
	}
	defer func() {
		_ = os.RemoveAll(stagingDir)
	}()
	tmpName := filepath.Join(stagingDir, "content")
	tmp, err := os.OpenFile(tmpName, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("write %s: create staged file: %w", path, err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write %s: %w", path, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("write %s: sync: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write %s: close temp: %w", path, err)
	}
	if info, err := os.Stat(path); err == nil {
		if err := os.Chmod(tmpName, info.Mode().Perm()); err != nil {
			return fmt.Errorf("write %s: preserve permissions: %w", path, err)
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("write %s: inspect permissions: %w", path, err)
	}
	if createOnly {
		if err := os.Link(tmpName, path); err != nil {
			if errors.Is(err, fs.ErrExist) {
				return fmt.Errorf("%w: %s", integration.ErrAlreadyExists, path)
			}
			return fmt.Errorf("write %s: publish create: %w", path, err)
		}
		return syncDirectory(dir)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("write %s: commit: %w", path, err)
	}
	return syncDirectory(dir)
}

func syncDirectory(path string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open parent for sync: %w", err)
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return fmt.Errorf("sync parent directory: %w", err)
	}
	return nil
}

func utf8Valid(data []byte) bool {
	return strings.ToValidUTF8(string(data), "\uFFFD") == string(data)
}

func normalizeLineEndings(value string) (string, bool) {
	crlf := strings.Count(value, "\r\n") > strings.Count(strings.ReplaceAll(value, "\r\n", ""), "\n")
	return strings.ReplaceAll(value, "\r\n", "\n"), crlf
}

// ensure interface satisfaction
var _ integration.FileSystem = (*LocalFileSystem)(nil)
