package skill

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

type RuntimeOptions struct {
	Request                     ListRequest
	Pinned                      bool
	CatalogDescriptionMaxLength int
	Snapshot                    *Snapshot
	Resources                   ResourceResolver
	ResourcePaths               ResourcePathResolver
}

type Runtime struct {
	mu            sync.RWMutex
	registry      *Registry
	request       ListRequest
	pinned        bool
	maxLength     int
	snapshot      Snapshot
	hasSnapshot   bool
	resources     ResourceResolver
	resourcePaths ResourcePathResolver
	loaded        map[string]struct{}
}

func NewRuntime(registry *Registry, options RuntimeOptions) (*Runtime, error) {
	if registry == nil {
		return nil, fmt.Errorf("%w: nil registry", ErrInvalidSkill)
	}
	maxLength := options.CatalogDescriptionMaxLength
	if maxLength == 0 {
		maxLength = DefaultCatalogDescriptionMaxLength
	}
	if maxLength < 3 {
		return nil, fmt.Errorf("%w: catalog description max length", ErrInvalidSkill)
	}
	runtime := &Runtime{
		registry:      registry,
		request:       cloneListRequest(options.Request),
		pinned:        options.Pinned,
		maxLength:     maxLength,
		resources:     options.Resources,
		resourcePaths: options.ResourcePaths,
		loaded:        make(map[string]struct{}),
	}
	if options.Snapshot != nil {
		runtime.snapshot = cloneSnapshot(*options.Snapshot)
		runtime.hasSnapshot = true
	}
	return runtime, nil
}

func (r *Runtime) Initialize(ctx context.Context) error {
	_, err := r.Refresh(ctx)
	return err
}

func (r *Runtime) Refresh(ctx context.Context) (bool, error) {
	r.mu.RLock()
	if r.pinned && r.hasSnapshot {
		r.mu.RUnlock()
		return false, nil
	}
	r.mu.RUnlock()

	snapshot, err := r.registry.Snapshot(ctx, r.request)
	if err != nil {
		return false, err
	}
	if !snapshot.Complete {
		r.mu.RLock()
		hasSnapshot := r.hasSnapshot
		r.mu.RUnlock()
		if hasSnapshot {
			return false, nil
		}
		return false, ErrIncompleteCatalog
	}
	r.mu.Lock()
	changed := !r.hasSnapshot || r.snapshot.SnapshotHash != snapshot.SnapshotHash
	r.snapshot = cloneSnapshot(snapshot)
	r.hasSnapshot = true
	r.mu.Unlock()
	return changed, nil
}

func (r *Runtime) Snapshot() (Snapshot, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if !r.hasSnapshot {
		return Snapshot{}, false
	}
	return cloneSnapshot(r.snapshot), true
}

func (r *Runtime) Catalog() Catalog {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if !r.hasSnapshot {
		return Catalog{Skills: []Summary{}, Complete: false}
	}
	summaries := make([]Summary, 0, len(r.snapshot.Skills))
	for _, pinned := range r.snapshot.Skills {
		if !pinned.Policy.Model {
			continue
		}
		summary := cloneSummary(pinned.Summary)
		summary.Description = normalizeDescription(summary.Description, r.maxLength)
		// The model catalog is invocation-neutral metadata only. Provider,
		// source and resource locations remain runtime-owned.
		summary.Provider = ""
		summary.Source = ""
		summary.ResourceBase = nil
		summaries = append(summaries, summary)
	}
	sort.Slice(summaries, func(i, j int) bool { return summaries[i].Name < summaries[j].Name })
	return Catalog{Skills: summaries, Complete: r.snapshot.Complete, Hash: catalogDigest(summaries)}
}

func (r *Runtime) GetModel(ctx context.Context, name string) (Definition, error) {
	r.mu.RLock()
	_, alreadyLoaded := r.loaded[name]
	r.mu.RUnlock()
	if alreadyLoaded {
		return Definition{}, ErrAlreadyLoaded
	}
	return r.get(ctx, name, func(policy InvocationPolicy) bool { return policy.Model })
}

func (r *Runtime) GetUser(ctx context.Context, name string) (Definition, error) {
	r.mu.RLock()
	_, alreadyLoaded := r.loaded[name]
	r.mu.RUnlock()
	if alreadyLoaded {
		return Definition{}, ErrAlreadyLoaded
	}
	return r.get(ctx, name, func(policy InvocationPolicy) bool { return policy.User })
}

// MarkUserLoaded prevents a model tool call from loading an explicitly
// injected user skill a second time in the same run.
func (r *Runtime) MarkUserLoaded(name string) {
	r.mu.Lock()
	r.loaded[name] = struct{}{}
	r.mu.Unlock()
}

// MarkLoaded restores a successful model activation from durable history.
func (r *Runtime) MarkLoaded(name string) {
	if !IsName(name) {
		return
	}
	r.mu.Lock()
	r.loaded[name] = struct{}{}
	r.mu.Unlock()
}

func (r *Runtime) ResolveResource(
	ctx context.Context,
	definition Definition,
	resource ResourceRef,
) (Resource, error) {
	if r.resources == nil {
		return Resource{}, ErrInvalidResource
	}
	return r.resources.Resolve(ctx, definition, resource)
}

// ResolveLoadedResource resolves exactly one resource from an activated skill.
// Manifest names use the configured ResourceResolver; other references are
// delegated to the provider-owned relative-path resolver.
func (r *Runtime) ResolveLoadedResource(ctx context.Context, name, reference string) (Resource, error) {
	if !IsName(name) || !validResourceReference(reference) {
		return Resource{}, ErrInvalidResource
	}
	r.mu.RLock()
	_, loaded := r.loaded[name]
	if !loaded || !r.hasSnapshot {
		r.mu.RUnlock()
		return Resource{}, ErrSkillNotLoaded
	}
	var matched *PinnedSkill
	for index := range r.snapshot.Skills {
		if r.snapshot.Skills[index].Name == name {
			entry := r.snapshot.Skills[index]
			matched = &entry
			break
		}
	}
	request := cloneListRequest(r.request)
	resources, resourcePaths := r.resources, r.resourcePaths
	r.mu.RUnlock()
	if matched == nil {
		return Resource{}, ErrSkillNotFound
	}
	definition, err := r.registry.LoadPinned(ctx, request, *matched)
	if err != nil {
		return Resource{}, err
	}
	for _, declared := range definition.ResourceManifest {
		if declared.Name == reference {
			if resources == nil {
				return Resource{}, ErrInvalidResource
			}
			return resources.Resolve(ctx, definition, declared)
		}
	}
	if resourcePaths == nil {
		return Resource{}, ErrInvalidResource
	}
	return resourcePaths.ResolvePath(ctx, definition, reference)
}

func validResourceReference(reference string) bool {
	return reference == strings.TrimSpace(reference) && reference != "" && len(reference) <= 512 && !strings.ContainsRune(reference, '\x00')
}

func (r *Runtime) get(
	ctx context.Context,
	name string,
	allowed func(InvocationPolicy) bool,
) (Definition, error) {
	if !IsName(name) {
		return Definition{}, fmt.Errorf("%w: name %q", ErrInvalidSkill, name)
	}
	r.mu.RLock()
	if !r.hasSnapshot {
		r.mu.RUnlock()
		return Definition{}, ErrIncompleteCatalog
	}
	var matched *PinnedSkill
	for index := range r.snapshot.Skills {
		if r.snapshot.Skills[index].Name == name {
			entry := r.snapshot.Skills[index]
			matched = &entry
			break
		}
	}
	request := cloneListRequest(r.request)
	r.mu.RUnlock()
	if matched == nil {
		return Definition{}, ErrSkillNotFound
	}
	if !allowed(matched.Policy) {
		return Definition{}, ErrPolicyDenied
	}
	definition, err := r.registry.LoadPinned(ctx, request, *matched)
	if err != nil {
		return Definition{}, err
	}
	if !allowed(definition.Policy) {
		return Definition{}, ErrPolicyDenied
	}
	return definition, nil
}

func (r *Runtime) RenderCatalog(update bool) string {
	catalog := r.Catalog()
	lines := make([]string, 0, len(catalog.Skills))
	for _, summary := range catalog.Skills {
		lines = append(lines, fmt.Sprintf("- `%s`: %s", summary.Name, escapeCatalogText(summary.Description)))
	}
	intro := "A skill is a reusable set of task-specific instructions. The following skills are available in this session:"
	instructions := "If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer skill instructions until loaded."
	if update {
		intro = "The available skill catalog changed. This complete catalog replaces every earlier available-skills list in this session:"
		if len(lines) == 0 {
			instructions = "No skills are currently available through the `skill` tool. Do not use names from earlier skill catalogs."
		}
	}
	return strings.Join([]string{
		"<system-reminder>",
		intro,
		"",
		"<available_skills>",
		strings.Join(lines, "\n"),
		"</available_skills>",
		"",
		instructions,
		"A user may also invoke a skill directly with `/skill <name>`; follow the injected <skill_content> and do not call the `skill` tool again for that skill.",
		"</system-reminder>",
	}, "\n")
}

func catalogDigest(summaries []Summary) string {
	entries := make([][2]string, 0, len(summaries))
	for _, summary := range summaries {
		entries = append(entries, [2]string{summary.Name, summary.Description})
	}
	data, _ := json.Marshal(entries)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func escapeCatalogText(value string) string {
	replacer := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")
	return replacer.Replace(value)
}

func cloneSnapshot(snapshot Snapshot) Snapshot {
	out := snapshot
	out.Skills = make([]PinnedSkill, len(snapshot.Skills))
	for index, entry := range snapshot.Skills {
		out.Skills[index] = PinnedSkill{Candidate: cloneCandidate(entry.Candidate), Scope: entry.Scope}
	}
	return out
}
