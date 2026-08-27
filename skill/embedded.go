package skill

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
)

type EmbeddedRoot struct {
	Path   string
	Rank   int
	Source string
}

type EmbeddedProvider struct {
	name  string
	fs    fs.FS
	roots []EmbeddedRoot
}

func (*EmbeddedProvider) SupportsPinnedLookup() bool { return true }

func NewEmbeddedProvider(name string, sourceFS fs.FS, roots ...EmbeddedRoot) (*EmbeddedProvider, error) {
	if name == "" || sourceFS == nil || len(roots) == 0 {
		return nil, fmt.Errorf("%w: embedded provider", ErrInvalidSkill)
	}
	cloned := append([]EmbeddedRoot{}, roots...)
	for index := range cloned {
		cloned[index].Path = strings.Trim(cloned[index].Path, "/")
		if cloned[index].Rank == 0 {
			cloned[index].Rank = BundledRank
		}
		if cloned[index].Source == "" {
			cloned[index].Source = "bundled"
		}
	}
	return &EmbeddedProvider{name: name, fs: sourceFS, roots: cloned}, nil
}

func (p *EmbeddedProvider) List(ctx context.Context, _ ListRequest) (Observation, error) {
	candidates := []Candidate{}
	for _, root := range p.roots {
		if err := ctx.Err(); err != nil {
			return Observation{}, err
		}
		entries, err := fs.ReadDir(p.fs, root.Path)
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			return Observation{}, fmt.Errorf("skill: reading embedded root %q: %w", root.Path, err)
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
		for _, entry := range entries {
			locator, resourceDir, ok := embeddedLocator(root.Path, entry)
			if !ok {
				continue
			}
			definition, err := p.parse(locator, root.Source, resourceDir)
			if err != nil {
				continue
			}
			candidates = append(candidates, candidateFromDefinition(definition, root.Rank, locator))
		}
	}
	return Observation{Candidates: candidates, Complete: true}, nil
}

func (p *EmbeddedProvider) Get(
	ctx context.Context,
	candidate Candidate,
	_ Lookup,
) (Definition, error) {
	if err := ctx.Err(); err != nil {
		return Definition{}, err
	}
	for _, root := range p.roots {
		if candidate.Source != root.Source || !containedSlashPath(root.Path, candidate.Locator) {
			continue
		}
		resourceDir := path.Dir(candidate.Locator)
		if strings.HasSuffix(candidate.Locator, ".md") && path.Base(candidate.Locator) != "SKILL.md" {
			resourceDir = root.Path
		}
		return p.parse(candidate.Locator, root.Source, resourceDir)
	}
	return Definition{}, ErrSkillNotFound
}

func (p *EmbeddedProvider) parse(locator, source, resourceDir string) (Definition, error) {
	data, err := fs.ReadFile(p.fs, locator)
	if err != nil {
		return Definition{}, fmt.Errorf("skill: reading embedded definition %q: %w", locator, err)
	}
	definition, err := Parse(data, ParseOptions{
		Provider: p.name,
		Source:   source,
		ResourceBase: &ResourceBase{
			Kind:        ResourceOpaque,
			Description: p.resourceDescription(resourceDir),
		},
	})
	if err != nil {
		return Definition{}, err
	}
	if path.Base(locator) != "SKILL.md" {
		definition.ResourceBase.Description = "No bundled resources are available for this skill."
		return definition, nil
	}
	bundleHash, err := p.bundleContentHash(resourceDir, definition.ContentHash)
	if err != nil {
		return Definition{}, err
	}
	if definition.Version == definition.ContentHash {
		definition.Version = bundleHash
	}
	definition.ContentHash = bundleHash
	return definition, nil
}

func (p *EmbeddedProvider) bundleContentHash(resourceDir, skillHash string) (string, error) {
	paths := make([]string, 0)
	if err := fs.WalkDir(p.fs, resourceDir, func(resourcePath string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && path.Base(resourcePath) != "SKILL.md" {
			paths = append(paths, resourcePath)
		}
		return nil
	}); err != nil {
		return "", fmt.Errorf("skill: hashing embedded resources %q: %w", resourceDir, err)
	}
	sort.Strings(paths)
	digest := sha256.New()
	_, _ = digest.Write([]byte(skillHash))
	for _, resourcePath := range paths {
		data, err := fs.ReadFile(p.fs, resourcePath)
		if err != nil {
			return "", fmt.Errorf("skill: hashing embedded resource %q: %w", resourcePath, err)
		}
		_, _ = digest.Write([]byte{0})
		_, _ = digest.Write([]byte(strings.TrimPrefix(resourcePath, strings.TrimSuffix(resourceDir, "/")+"/")))
		resourceHash := sha256.Sum256(data)
		_, _ = digest.Write(resourceHash[:])
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func (p *EmbeddedProvider) resourceDescription(resourceDir string) string {
	const maxEntries = 64
	resources := make([]string, 0)
	_ = fs.WalkDir(p.fs, resourceDir, func(resourcePath string, entry fs.DirEntry, err error) error {
		if err != nil || len(resources) >= maxEntries {
			return nil
		}
		if entry.IsDir() || path.Base(resourcePath) == "SKILL.md" {
			return nil
		}
		relative := strings.TrimPrefix(resourcePath, strings.TrimSuffix(resourceDir, "/")+"/")
		if relative != "" && relative != resourcePath {
			resources = append(resources, relative)
		}
		return nil
	})
	if len(resources) == 0 {
		return "No bundled resources are available for this skill."
	}
	sort.Strings(resources)
	return "Bundled resources available by relative path: " + strings.Join(resources, ", ")
}

func embeddedLocator(root string, entry fs.DirEntry) (string, string, bool) {
	if entry.IsDir() {
		return path.Join(root, entry.Name(), "SKILL.md"), path.Join(root, entry.Name()), true
	}
	if strings.HasSuffix(entry.Name(), ".md") {
		return path.Join(root, entry.Name()), root, true
	}
	return "", "", false
}

func containedSlashPath(root, target string) bool {
	root = strings.TrimSuffix(path.Clean(root), "/")
	target = path.Clean(target)
	return target == root || strings.HasPrefix(target, root+"/")
}
