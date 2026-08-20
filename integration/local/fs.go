// Package local provides concrete local-host implementations of the
// provider-neutral seams (FileSystem, Subprocess, JobManager, Sandbox). They
// enforce the same containment, observation and CAS contracts that the E2B
// adapter enforces, so tools behave identically across execution worlds.
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
	version string
}

// NewLocalFileSystem returns a filesystem confined to root.
func NewLocalFileSystem(root string) *LocalFileSystem {
	return &LocalFileSystem{Root: filepath.Clean(root), obs: make(map[obsKey]obsEntry)}
}

// Version computes the content version (SHA-256 hex) of data.
func Version(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
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
	return integration.Target{Path: canonical}, nil
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
		return integration.FileInfo{}, fmt.Errorf("stat %s: %w", target.Path, err)
	}
	return integration.FileInfo{
		Exists:  true,
		IsDir:   info.IsDir(),
		Size:    info.Size(),
		Version: fileVersion(target.Path, info),
	}, nil
}

// fileVersion derives a content version from the file (hash) or, for
// directories, a size+mtime digest.
func fileVersion(path string, info os.FileInfo) string {
	if info.IsDir() {
		sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d:%d", path, info.Size(), info.ModTime().UnixNano())))
		return hex.EncodeToString(sum[:])
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return Version(data)
}

func (f *LocalFileSystem) ReadText(ctx context.Context, target integration.Target) (string, error) {
	data, err := os.ReadFile(target.Path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("read %s: file does not exist", target.Path)
		}
		return "", fmt.Errorf("read %s: %w", target.Path, err)
	}
	if !utf8Valid(data) {
		return "", fmt.Errorf("read %s: not valid UTF-8 text", target.Path)
	}
	return string(data), nil
}

func (f *LocalFileSystem) ReadBytes(_ context.Context, target integration.Target) ([]byte, error) {
	return os.ReadFile(target.Path)
}

func (f *LocalFileSystem) WriteText(_ context.Context, target integration.Target, content string, intent integration.WriteIntent) (integration.WriteResult, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

	info, err := os.Stat(target.Path)
	exists := err == nil
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return integration.WriteResult{}, fmt.Errorf("write %s: %w", target.Path, err)
	}
	if exists && info.IsDir() {
		return integration.WriteResult{}, fmt.Errorf("write %s: target is a directory", target.Path)
	}

	if intent.CreateIfAbsent && exists {
		return integration.WriteResult{}, fmt.Errorf("%w: %s", integration.ErrAlreadyExists, target.Path)
	}
	if exists && intent.ExpectedVersion != "" {
		current := fileVersion(target.Path, info)
		if current != intent.ExpectedVersion {
			return integration.WriteResult{}, fmt.Errorf("%w: %s (expected %s)", integration.ErrStaleVersion, target.Path, intent.ExpectedVersion)
		}
	}
	if exists && !intent.Overwrite {
		return integration.WriteResult{}, fmt.Errorf("write %s: refusing to overwrite without overwrite intent", target.Path)
	}

	if err := atomicWrite(target.Path, []byte(content)); err != nil {
		return integration.WriteResult{}, err
	}
	return integration.WriteResult{Version: Version([]byte(content)), Created: !exists}, nil
}

func (f *LocalFileSystem) EditText(_ context.Context, target integration.Target, edit integration.EditRequest, intent integration.EditIntent) (integration.EditResult, error) {
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
		if errors.Is(err, fs.ErrNotExist) {
			return integration.EditResult{}, fmt.Errorf("edit %s: file does not exist", target.Path)
		}
		return integration.EditResult{}, fmt.Errorf("edit %s: %w", target.Path, err)
	}
	if intent.ExpectedVersion != "" {
		current := fileVersion(target.Path, info)
		if current != intent.ExpectedVersion {
			return integration.EditResult{}, fmt.Errorf("%w: %s", integration.ErrStaleVersion, target.Path)
		}
	}
	data, err := os.ReadFile(target.Path)
	if err != nil {
		return integration.EditResult{}, fmt.Errorf("edit %s: %w", target.Path, err)
	}
	text := string(data)
	count := strings.Count(text, edit.OldString)
	if count == 0 {
		return integration.EditResult{}, fmt.Errorf("edit %s: old_string not found (no match)", target.Path)
	}
	if count > 1 && !edit.ReplaceAll {
		return integration.EditResult{}, fmt.Errorf("%w: %s (%d matches; use replace_all or a more specific old_string)", integration.ErrAmbiguousMatch, target.Path, count)
	}
	replaced := count
	newText := text
	if edit.ReplaceAll {
		newText = strings.ReplaceAll(text, edit.OldString, edit.NewString)
	} else {
		newText = strings.Replace(text, edit.OldString, edit.NewString, 1)
	}
	if err := atomicWrite(target.Path, []byte(newText)); err != nil {
		return integration.EditResult{}, err
	}
	return integration.EditResult{
		Version: Version([]byte(newText)), Replaced: replaced,
		OldString: edit.OldString, NewString: edit.NewString,
	}, nil
}

// Glob lists files under dir matching pattern (sorted, bounded).
func (f *LocalFileSystem) Glob(_ context.Context, dir, pattern string, maxResults int) ([]string, error) {
	root, err := canonicalWithin(dir, f.Root)
	if err != nil {
		return nil, err
	}
	if maxResults <= 0 {
		maxResults = 100
	}
	var matches []string
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
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
		return nil, walkErr
	}
	sort.Strings(matches)
	return matches, nil
}

// Grep scans regular text files under dir for the literal/regex pattern.
// maxMatches bounds results; lines carry 1-based numbers.
func (f *LocalFileSystem) Grep(_ context.Context, dir, pattern string, maxMatches int, maxBytesPerFile int64) ([]integration.GrepMatch, error) {
	root, err := canonicalWithin(dir, f.Root)
	if err != nil {
		return nil, err
	}
	if maxMatches <= 0 {
		maxMatches = 200
	}
	var out []integration.GrepMatch
	re, err := compilePattern(pattern)
	if err != nil {
		return nil, err
	}
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
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
		if err != nil || info.IsDir() {
			return nil
		}
		if info.Size() > maxBytesPerFile {
			return nil // skip oversized binaries
		}
		data, err := os.ReadFile(path)
		if err != nil || !utf8Valid(data) {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
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
		return nil, walkErr
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
func (f *LocalFileSystem) record(target integration.Target, owner string, state integration.Observation, version string) {
	if owner == "" {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.obs[obsKey{owner: owner, path: target.Path}] = obsEntry{state: state, version: version}
}

// Observe marks a target observed (read path). State is scoped per owner and
// never leaks between users (H-FS-OBS-007).
func (f *LocalFileSystem) Observe(_ context.Context, target integration.Target, owner string, state integration.Observation, version string) error {
	f.record(target, owner, state, version)
	return nil
}

// Observed returns the recorded state/version for owner+target.
func (f *LocalFileSystem) Observed(_ context.Context, target integration.Target, owner string) (integration.Observation, string, bool) {
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
func (f *LocalFileSystem) ObservedVersion(owner string, target integration.Target) (string, bool) {
	_, version, ok := f.Observed(context.Background(), target, owner)
	return version, ok
}

// ObservedState returns the recorded observation state.
func (f *LocalFileSystem) ObservedState(owner string, target integration.Target) (integration.Observation, string, bool) {
	return f.Observed(context.Background(), target, owner)
}

func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".agentkit-write-*")
	if err != nil {
		return fmt.Errorf("write %s: create temp: %w", path, err)
	}
	tmpName := tmp.Name()
	defer func() {
		_ = os.Remove(tmpName)
	}()
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
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("write %s: commit: %w", path, err)
	}
	return nil
}

func utf8Valid(data []byte) bool {
	return strings.ToValidUTF8(string(data), "\uFFFD") == string(data)
}

// ensure interface satisfaction
var _ integration.FileSystem = (*LocalFileSystem)(nil)
