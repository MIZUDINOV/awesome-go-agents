package skill

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type FilesystemRoot struct {
	Path       string
	Rank       int
	Source     string
	SkipSystem bool
}

type FilesystemOptions struct {
	Name           string
	Roots          []FilesystemRoot
	FollowSymlinks bool
}

type FilesystemProvider struct {
	name           string
	roots          []FilesystemRoot
	followSymlinks bool
}

func (*FilesystemProvider) SupportsPinnedLookup() bool { return false }

func NewFilesystemProvider(options FilesystemOptions) (*FilesystemProvider, error) {
	if options.Name == "" || len(options.Roots) == 0 {
		return nil, fmt.Errorf("%w: filesystem provider", ErrInvalidSkill)
	}
	roots := append([]FilesystemRoot{}, options.Roots...)
	for index := range roots {
		absolute, err := filepath.Abs(roots[index].Path)
		if err != nil {
			return nil, fmt.Errorf("skill: resolving root %q: %w", roots[index].Path, err)
		}
		roots[index].Path = filepath.Clean(absolute)
		if roots[index].Rank == 0 {
			roots[index].Rank = BundledRank
		}
		if roots[index].Source == "" {
			roots[index].Source = "custom"
		}
	}
	return &FilesystemProvider{
		name:           options.Name,
		roots:          roots,
		followSymlinks: options.FollowSymlinks,
	}, nil
}

func (p *FilesystemProvider) List(ctx context.Context, _ ListRequest) (Observation, error) {
	candidates := []Candidate{}
	for _, root := range p.roots {
		if err := ctx.Err(); err != nil {
			return Observation{}, err
		}
		entries, err := os.ReadDir(root.Path)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return Observation{Candidates: candidates, Complete: false}, nil
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
		for _, entry := range entries {
			if root.SkipSystem && entry.Name() == ".system" {
				continue
			}
			locator, resourceDir, ok := filesystemLocator(root.Path, entry)
			if !ok || (!p.followSymlinks && isSymlink(locator)) {
				continue
			}
			definition, err := p.parse(root, locator, resourceDir)
			if err != nil {
				continue
			}
			candidates = append(candidates, candidateFromDefinition(definition, root.Rank, locator))
		}
	}
	return Observation{Candidates: candidates, Complete: true}, nil
}

func (p *FilesystemProvider) Get(
	ctx context.Context,
	candidate Candidate,
	_ Lookup,
) (Definition, error) {
	if err := ctx.Err(); err != nil {
		return Definition{}, err
	}
	for _, root := range p.roots {
		if candidate.Source != root.Source || !containedFilePath(root.Path, candidate.Locator) {
			continue
		}
		resourceDir := filepath.Dir(candidate.Locator)
		if filepath.Base(candidate.Locator) != "SKILL.md" {
			resourceDir = root.Path
		}
		return p.parse(root, candidate.Locator, resourceDir)
	}
	return Definition{}, ErrSkillNotFound
}

// Watch blocks until ctx is cancelled and invalidates when direct skill
// entries change. Hosts opt in by starting it under their own lifecycle.
func (p *FilesystemProvider) Watch(
	ctx context.Context,
	interval time.Duration,
	invalidate func(string),
) error {
	if interval <= 0 {
		return fmt.Errorf("%w: watcher interval", ErrInvalidSkill)
	}
	previous, err := p.fingerprint(ctx)
	if err != nil {
		return err
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			current, err := p.fingerprint(ctx)
			if err != nil {
				if invalidate != nil {
					invalidate("filesystem watcher incomplete")
				}
				continue
			}
			if current != previous {
				previous = current
				if invalidate != nil {
					invalidate("filesystem catalog changed")
				}
			}
		}
	}
}

func (p *FilesystemProvider) parse(
	root FilesystemRoot,
	locator string,
	resourceDir string,
) (Definition, error) {
	if !containedFilePath(root.Path, locator) {
		return Definition{}, ErrInvalidResource
	}
	data, err := os.ReadFile(locator)
	if err != nil {
		return Definition{}, fmt.Errorf("skill: reading definition %q: %w", locator, err)
	}
	return Parse(data, ParseOptions{
		Provider: p.name,
		Source:   root.Source,
		ResourceBase: &ResourceBase{
			Kind: ResourceDirectory,
			Path: resourceDir,
		},
	})
}

func (p *FilesystemProvider) fingerprint(ctx context.Context) (string, error) {
	observation, err := p.List(ctx, ListRequest{})
	if err != nil {
		return "", err
	}
	if !observation.Complete {
		return "", ErrIncompleteCatalog
	}
	parts := make([]string, 0, len(observation.Candidates))
	for _, candidate := range observation.Candidates {
		parts = append(parts, candidate.Name+":"+candidate.ContentHash)
	}
	return hashBytes([]byte(strings.Join(parts, "\n"))), nil
}

func filesystemLocator(root string, entry os.DirEntry) (string, string, bool) {
	if entry.IsDir() {
		directory := filepath.Join(root, entry.Name())
		return filepath.Join(directory, "SKILL.md"), directory, true
	}
	if strings.HasSuffix(entry.Name(), ".md") {
		return filepath.Join(root, entry.Name()), root, true
	}
	return "", "", false
}

func containedFilePath(root, target string) bool {
	relative, err := filepath.Rel(root, target)
	if err != nil || relative == ".." || filepath.IsAbs(relative) {
		return false
	}
	return !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func isSymlink(path string) bool {
	info, err := os.Lstat(path)
	return err == nil && info.Mode()&os.ModeSymlink != 0
}
