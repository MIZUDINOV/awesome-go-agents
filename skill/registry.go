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
	"sync/atomic"
	"time"
)

const maxCollectAttempts = 2

type providerBinding struct {
	name     string
	scope    string
	provider Provider
	order    uint64
	ctx      context.Context
	cancel   context.CancelCauseFunc
	active   atomic.Bool
}

type layer struct {
	providers map[string]*providerBinding
	runtime   map[string]runtimeEntry
}

type runtimeEntry struct {
	definition Definition
	rank       int
}

type cacheEntry struct {
	revision uint64
	entries  map[string]indexedCandidate
}

type indexedCandidate struct {
	candidate     Candidate
	binding       *providerBinding
	providerOrder uint64
	localOrder    int
	scope         string
}

type collectResult struct {
	entries  map[string]indexedCandidate
	complete bool
}

type Registry struct {
	mu              sync.RWMutex
	layers          map[string]*layer
	cache           map[string]cacheEntry
	cacheOrder      []string
	maxCacheEntries int
	revision        uint64
	nextOrder       uint64
	listeners       map[uint64]func(Invalidation)
	nextListener    uint64
}

func NewRegistry(maxCacheEntries int) *Registry {
	if maxCacheEntries <= 0 {
		maxCacheEntries = DefaultCollectCacheMaxEntries
	}
	return &Registry{
		layers:          map[string]*layer{"": newLayer()},
		cache:           make(map[string]cacheEntry),
		maxCacheEntries: maxCacheEntries,
		listeners:       make(map[uint64]func(Invalidation)),
	}
}

func newLayer() *layer {
	return &layer{
		providers: make(map[string]*providerBinding),
		runtime:   make(map[string]runtimeEntry),
	}
}

type Registration struct {
	registry *Registry
	binding  *providerBinding
	once     sync.Once
}

func (r *Registration) Invalidate(reason string) {
	if r == nil || r.registry == nil || r.binding == nil || !r.binding.active.Load() {
		return
	}
	r.registry.invalidate(Invalidation{
		Provider: r.binding.name,
		Scope:    r.binding.scope,
		Reason:   reason,
	})
}

func (r *Registration) Close() error {
	if r == nil {
		return nil
	}
	r.once.Do(func() {
		r.registry.unregister(r.binding)
	})
	return nil
}

func (r *Registry) RegisterProvider(options ProviderOptions) (*Registration, error) {
	if r == nil || options.Provider == nil || options.Name == "" || strings.Contains(options.Name, "/") {
		return nil, fmt.Errorf("%w: provider registration", ErrInvalidSkill)
	}
	if options.Name == RuntimeProviderName {
		return nil, fmt.Errorf("%w: provider name %q is reserved", ErrInvalidSkill, options.Name)
	}

	providerCtx, cancel := context.WithCancelCause(context.Background())
	r.mu.Lock()
	current := r.layers[options.Scope]
	if current == nil {
		current = newLayer()
		r.layers[options.Scope] = current
	}
	if _, exists := current.providers[options.Name]; exists {
		cancel(ErrDuplicateProvider)
		r.mu.Unlock()
		return nil, fmt.Errorf("%w: %s", ErrDuplicateProvider, options.Name)
	}
	r.nextOrder++
	binding := &providerBinding{
		name:     options.Name,
		scope:    options.Scope,
		provider: options.Provider,
		order:    r.nextOrder,
		ctx:      providerCtx,
		cancel:   cancel,
	}
	binding.active.Store(true)
	current.providers[options.Name] = binding
	r.invalidateLocked()
	listeners := r.listenersLocked()
	r.mu.Unlock()
	notifyListeners(listeners, Invalidation{Provider: options.Name, Scope: options.Scope, Reason: "provider registered"})
	return &Registration{registry: r, binding: binding}, nil
}

func (r *Registry) Register(scope string, definition Definition, rank int) (func(), error) {
	if err := validateDefinition(definition); err != nil {
		return nil, err
	}
	if rank == 0 {
		rank = RuntimeRank
	}
	definition = cloneDefinition(definition)
	r.mu.Lock()
	current := r.layers[scope]
	if current == nil {
		current = newLayer()
		r.layers[scope] = current
	}
	if _, exists := current.runtime[definition.Name]; exists {
		r.mu.Unlock()
		return func() {}, nil
	}
	current.runtime[definition.Name] = runtimeEntry{definition: definition, rank: rank}
	r.invalidateLocked()
	listeners := r.listenersLocked()
	r.mu.Unlock()
	notifyListeners(listeners, Invalidation{Provider: RuntimeProviderName, Scope: scope, Reason: "runtime skill registered"})

	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			if current := r.layers[scope]; current != nil {
				delete(current.runtime, definition.Name)
			}
			r.invalidateLocked()
			listeners := r.listenersLocked()
			r.mu.Unlock()
			notifyListeners(listeners, Invalidation{Provider: RuntimeProviderName, Scope: scope, Reason: "runtime skill disposed"})
		})
	}, nil
}

func (r *Registry) Subscribe(listener func(Invalidation)) func() {
	if r == nil || listener == nil {
		return func() {}
	}
	r.mu.Lock()
	r.nextListener++
	id := r.nextListener
	r.listeners[id] = listener
	r.mu.Unlock()
	return func() {
		r.mu.Lock()
		delete(r.listeners, id)
		r.mu.Unlock()
	}
}

func (r *Registry) Invalidate(ctx context.Context, invalidation Invalidation) error {
	if r == nil {
		return fmt.Errorf("%w: nil registry", ErrInvalidSkill)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	r.invalidate(invalidation)
	return nil
}

func (r *Registry) List(ctx context.Context, request ListRequest) (Catalog, error) {
	snapshot, err := r.Snapshot(ctx, request)
	if err != nil {
		return Catalog{}, err
	}
	summaries := make([]Summary, 0, len(snapshot.Skills))
	for _, entry := range snapshot.Skills {
		summaries = append(summaries, cloneSummary(entry.Summary))
	}
	return Catalog{
		Skills:   summaries,
		Complete: snapshot.Complete,
		Hash:     snapshot.CatalogHash,
	}, nil
}

func (r *Registry) Snapshot(ctx context.Context, request ListRequest) (Snapshot, error) {
	collected, err := r.collect(ctx, request)
	if err != nil {
		return Snapshot{}, err
	}
	names := make([]string, 0, len(collected.entries))
	for name := range collected.entries {
		names = append(names, name)
	}
	sort.Strings(names)
	entries := make([]PinnedSkill, 0, len(names))
	for _, name := range names {
		entry := collected.entries[name]
		entries = append(entries, PinnedSkill{
			Candidate: cloneCandidate(entry.candidate),
			Scope:     entry.scope,
		})
	}
	catalogHash, snapshotHash, err := snapshotHashes(entries)
	if err != nil {
		return Snapshot{}, fmt.Errorf("skill: hashing snapshot: %w", err)
	}
	return Snapshot{
		SchemaVersion: "agentkit-skills-v2",
		Skills:        entries,
		Complete:      collected.complete,
		CatalogHash:   catalogHash,
		SnapshotHash:  snapshotHash,
		CreatedAt:     time.Now().UTC(),
	}, nil
}

func (r *Registry) Get(ctx context.Context, lookup Lookup) (Definition, error) {
	if !IsName(lookup.Name) {
		return Definition{}, fmt.Errorf("%w: name %q", ErrInvalidSkill, lookup.Name)
	}
	collected, err := r.collect(ctx, lookup.ListRequest)
	if err != nil {
		return Definition{}, err
	}
	entry, exists := collected.entries[lookup.Name]
	if !exists {
		return Definition{}, ErrSkillNotFound
	}
	if lookup.Provider != "" && lookup.Provider != entry.candidate.Provider {
		return Definition{}, ErrSkillNotFound
	}
	return r.load(ctx, entry, lookup)
}

func (r *Registry) LoadPinned(ctx context.Context, request ListRequest, pinned PinnedSkill) (Definition, error) {
	r.mu.RLock()
	current := r.layers[pinned.Scope]
	var binding *providerBinding
	if current != nil {
		binding = current.providers[pinned.Provider]
	}
	r.mu.RUnlock()
	if pinned.Provider == RuntimeProviderName {
		if current == nil {
			return Definition{}, ErrSkillNotFound
		}
		r.mu.RLock()
		entry, exists := current.runtime[pinned.Name]
		r.mu.RUnlock()
		if !exists {
			return Definition{}, ErrSkillNotFound
		}
		return verifyPinned(entry.definition, pinned.Candidate)
	}
	if binding == nil || !binding.active.Load() {
		return Definition{}, ErrProviderDisposed
	}
	pinnedProvider, supportsPinned := binding.provider.(PinnedProvider)
	if !supportsPinned || !pinnedProvider.SupportsPinnedLookup() {
		return Definition{}, ErrUnsupportedMutable
	}
	entry := indexedCandidate{
		candidate: pinned.Candidate,
		binding:   binding,
		scope:     pinned.Scope,
	}
	lookup := Lookup{
		ListRequest: request,
		Name:        pinned.Name,
		Provider:    pinned.Provider,
		Scope:       pinned.Scope,
		Locator:     pinned.Locator,
		Version:     pinned.Version,
		ContentHash: pinned.ContentHash,
	}
	return r.load(ctx, entry, lookup)
}

func (r *Registry) collect(ctx context.Context, request ListRequest) (collectResult, error) {
	if err := ctx.Err(); err != nil {
		return collectResult{}, err
	}
	for attempt := 0; attempt < maxCollectAttempts; attempt++ {
		r.mu.RLock()
		revision := r.revision
		key := cacheKey(request, revision)
		cached, ok := r.cache[key]
		r.mu.RUnlock()
		if ok {
			if err := ctx.Err(); err != nil {
				return collectResult{}, err
			}
			return collectResult{entries: cloneIndexedMap(cached.entries), complete: true}, nil
		}

		result, err := r.collectFresh(ctx, request)
		if err != nil {
			return collectResult{}, err
		}
		if err := ctx.Err(); err != nil {
			return collectResult{}, err
		}
		r.mu.Lock()
		if revision != r.revision {
			r.mu.Unlock()
			if attempt+1 < maxCollectAttempts {
				continue
			}
			result.complete = false
			return result, nil
		}
		if result.complete {
			r.cache[key] = cacheEntry{revision: revision, entries: cloneIndexedMap(result.entries)}
			r.cacheOrder = append(r.cacheOrder, key)
			for len(r.cacheOrder) > r.maxCacheEntries {
				oldest := r.cacheOrder[0]
				r.cacheOrder = r.cacheOrder[1:]
				delete(r.cache, oldest)
			}
		}
		r.mu.Unlock()
		return result, nil
	}
	return collectResult{}, ErrIncompleteCatalog
}

func (r *Registry) collectFresh(ctx context.Context, request ListRequest) (collectResult, error) {
	scopes := make([]string, 0, len(request.Scopes)+1)
	scopes = append(scopes, "")
	scopes = append(scopes, request.Scopes...)
	merged := make(map[string]indexedCandidate)
	complete := true
	for _, scope := range scopes {
		entries, layerComplete, err := r.collectLayer(ctx, scope, request)
		if err != nil {
			return collectResult{}, err
		}
		if !layerComplete {
			complete = false
		}
		for _, entry := range entries {
			merged[entry.candidate.Name] = entry
		}
	}
	return collectResult{entries: merged, complete: complete}, nil
}

func (r *Registry) collectLayer(
	ctx context.Context,
	scope string,
	request ListRequest,
) ([]indexedCandidate, bool, error) {
	r.mu.RLock()
	current := r.layers[scope]
	if current == nil {
		r.mu.RUnlock()
		return []indexedCandidate{}, true, nil
	}
	runtime := make([]runtimeEntry, 0, len(current.runtime))
	for _, entry := range current.runtime {
		runtime = append(runtime, runtimeEntry{definition: cloneDefinition(entry.definition), rank: entry.rank})
	}
	bindings := make([]*providerBinding, 0, len(current.providers))
	for _, binding := range current.providers {
		bindings = append(bindings, binding)
	}
	r.mu.RUnlock()

	sort.Slice(bindings, func(i, j int) bool { return bindings[i].order < bindings[j].order })
	sort.Slice(runtime, func(i, j int) bool { return runtime[i].definition.Name < runtime[j].definition.Name })
	entries := make([]indexedCandidate, 0, len(runtime))
	for index, entry := range runtime {
		definition := entry.definition
		entries = append(entries, indexedCandidate{
			candidate:     candidateFromDefinition(definition, entry.rank, definition.Name),
			providerOrder: 0,
			localOrder:    index,
			scope:         scope,
		})
	}

	complete := true
	for _, binding := range bindings {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		if !binding.active.Load() {
			complete = false
			continue
		}
		callCtx, cancel := context.WithCancelCause(ctx)
		stop := context.AfterFunc(binding.ctx, func() { cancel(ErrProviderDisposed) })
		observation, err := awaitProvider(callCtx, func() (Observation, error) {
			return binding.provider.List(callCtx, cloneListRequest(request))
		})
		stop()
		cancel(nil)
		if err != nil {
			if ctx.Err() != nil {
				return nil, false, ctx.Err()
			}
			complete = false
			continue
		}
		if !observation.Complete {
			complete = false
		}
		for index, candidate := range observation.Candidates {
			if candidate.Provider == "" {
				candidate.Provider = binding.name
			}
			if candidate.Provider != binding.name {
				return nil, false, fmt.Errorf("%w: provider %q returned candidate owned by %q", ErrInvalidSkill, binding.name, candidate.Provider)
			}
			if err := validateCandidate(candidate); err != nil {
				return nil, false, fmt.Errorf("provider %q: %w", binding.name, err)
			}
			entries = append(entries, indexedCandidate{
				candidate:     cloneCandidate(candidate),
				binding:       binding,
				providerOrder: binding.order,
				localOrder:    index,
				scope:         scope,
			})
		}
	}

	sort.SliceStable(entries, func(i, j int) bool {
		left := entries[i]
		right := entries[j]
		if left.candidate.Rank != right.candidate.Rank {
			return left.candidate.Rank < right.candidate.Rank
		}
		if left.providerOrder != right.providerOrder {
			return left.providerOrder < right.providerOrder
		}
		return left.localOrder < right.localOrder
	})
	seen := make(map[string]struct{})
	winners := make([]indexedCandidate, 0, len(entries))
	for _, entry := range entries {
		if _, exists := seen[entry.candidate.Name]; exists {
			continue
		}
		seen[entry.candidate.Name] = struct{}{}
		winners = append(winners, entry)
	}
	return winners, complete, nil
}

func (r *Registry) load(ctx context.Context, entry indexedCandidate, lookup Lookup) (Definition, error) {
	if entry.binding == nil {
		r.mu.RLock()
		definition, exists := r.layers[entry.scope].runtime[entry.candidate.Name]
		r.mu.RUnlock()
		if !exists {
			return Definition{}, ErrSkillNotFound
		}
		return verifyPinned(definition.definition, entry.candidate)
	}
	if !entry.binding.active.Load() {
		return Definition{}, ErrProviderDisposed
	}
	callCtx, cancel := context.WithCancelCause(ctx)
	stop := context.AfterFunc(entry.binding.ctx, func() { cancel(ErrProviderDisposed) })
	definition, err := awaitProvider(callCtx, func() (Definition, error) {
		return entry.binding.provider.Get(callCtx, cloneCandidate(entry.candidate), lookup)
	})
	stop()
	cancel(nil)
	if err != nil {
		return Definition{}, err
	}
	if err := validateDefinition(definition); err != nil {
		return Definition{}, err
	}
	return verifyPinned(definition, entry.candidate)
}

type providerResult[T any] struct {
	value T
	err   error
}

// awaitProvider makes context cancellation an AgentKit guarantee even when a
// third-party provider fails to observe the context it receives. The buffered
// result lets a late provider completion exit without retaining the caller.
func awaitProvider[T any](ctx context.Context, call func() (T, error)) (T, error) {
	result := make(chan providerResult[T], 1)
	go func() {
		value, err := call()
		result <- providerResult[T]{value: value, err: err}
	}()
	select {
	case <-ctx.Done():
		var zero T
		return zero, context.Cause(ctx)
	case completed := <-result:
		return completed.value, completed.err
	}
}

func verifyPinned(definition Definition, candidate Candidate) (Definition, error) {
	if definition.Name != candidate.Name || definition.Provider != candidate.Provider {
		return Definition{}, ErrPinnedMismatch
	}
	if candidate.Version != "" && definition.Version != candidate.Version {
		return Definition{}, ErrPinnedMismatch
	}
	if candidate.ContentHash != "" && definition.ContentHash != candidate.ContentHash {
		return Definition{}, ErrPinnedMismatch
	}
	return cloneDefinition(definition), nil
}

func (r *Registry) unregister(binding *providerBinding) {
	if r == nil || binding == nil || !binding.active.Swap(false) {
		return
	}
	r.mu.Lock()
	if current := r.layers[binding.scope]; current != nil && current.providers[binding.name] == binding {
		delete(current.providers, binding.name)
	}
	r.invalidateLocked()
	listeners := r.listenersLocked()
	r.mu.Unlock()
	binding.cancel(ErrProviderDisposed)
	notifyListeners(listeners, Invalidation{Provider: binding.name, Scope: binding.scope, Reason: "provider disposed"})
}

func (r *Registry) invalidate(invalidation Invalidation) {
	r.mu.Lock()
	r.invalidateLocked()
	listeners := r.listenersLocked()
	r.mu.Unlock()
	notifyListeners(listeners, invalidation)
}

func (r *Registry) invalidateLocked() {
	r.revision++
	r.cache = make(map[string]cacheEntry)
	r.cacheOrder = []string{}
}

func (r *Registry) listenersLocked() []func(Invalidation) {
	listeners := make([]func(Invalidation), 0, len(r.listeners))
	for _, listener := range r.listeners {
		listeners = append(listeners, listener)
	}
	return listeners
}

func notifyListeners(listeners []func(Invalidation), invalidation Invalidation) {
	for _, listener := range listeners {
		func() {
			defer func() { _ = recover() }()
			listener(invalidation)
		}()
	}
}

func validateCandidate(candidate Candidate) error {
	if !IsName(candidate.Name) || strings.TrimSpace(candidate.Description) == "" || candidate.Provider == "" || candidate.Source == "" {
		return fmt.Errorf("%w: malformed candidate", ErrInvalidSkill)
	}
	if candidate.Locator == "" || candidate.Version == "" || !isSHA256(candidate.ContentHash) {
		return fmt.Errorf("%w: candidate %q is not immutable", ErrUnsupportedMutable, candidate.Name)
	}
	return validateResourceBase(candidate.ResourceBase)
}

func validateDefinition(definition Definition) error {
	candidate := candidateFromDefinition(definition, 1, definition.Name)
	if err := validateCandidate(candidate); err != nil {
		return err
	}
	if strings.TrimSpace(definition.Content) == "" {
		return fmt.Errorf("%w: empty content", ErrInvalidSkill)
	}
	seenResources := make(map[string]struct{}, len(definition.ResourceManifest))
	for _, resource := range definition.ResourceManifest {
		if !IsName(resource.Name) {
			return fmt.Errorf("%w: invalid resource name %q", ErrInvalidResource, resource.Name)
		}
		if _, duplicate := seenResources[resource.Name]; duplicate {
			return fmt.Errorf("%w: duplicate resource name %q", ErrInvalidResource, resource.Name)
		}
		seenResources[resource.Name] = struct{}{}
		if resource.URL != "" && !isSHA256(resource.SHA256) {
			return fmt.Errorf("%w: URL resource %q is not hash-pinned", ErrInvalidResource, resource.Name)
		}
	}
	return nil
}

func validateResourceBase(base *ResourceBase) error {
	if base == nil {
		return nil
	}
	switch base.Kind {
	case ResourceDirectory:
		if base.Path == "" {
			return ErrInvalidResource
		}
	case ResourceURL:
		if base.URL == "" {
			return ErrInvalidResource
		}
	case ResourceOpaque:
		if base.Description == "" {
			return ErrInvalidResource
		}
	default:
		return ErrInvalidResource
	}
	return nil
}

func candidateFromDefinition(definition Definition, rank int, locator string) Candidate {
	return Candidate{
		Summary:     cloneSummary(definition.Summary),
		Rank:        rank,
		Locator:     locator,
		Version:     definition.Version,
		ContentHash: definition.ContentHash,
		Metadata:    cloneMap(definition.Metadata),
	}
}

func snapshotHashes(entries []PinnedSkill) (string, string, error) {
	catalogEntries := make([][2]string, 0, len(entries))
	for _, entry := range entries {
		if entry.Policy.Model {
			catalogEntries = append(catalogEntries, [2]string{entry.Name, normalizeDescription(entry.Description, DefaultCatalogDescriptionMaxLength)})
		}
	}
	catalogJSON, err := json.Marshal(catalogEntries)
	if err != nil {
		return "", "", err
	}
	snapshotJSON, err := json.Marshal(entries)
	if err != nil {
		return "", "", err
	}
	return hashBytes(catalogJSON), hashBytes(snapshotJSON), nil
}

func cacheKey(request ListRequest, revision uint64) string {
	data, _ := json.Marshal(struct {
		CWD      string
		Scopes   []string
		Revision uint64
	}{CWD: request.CWD, Scopes: request.Scopes, Revision: revision})
	return string(data)
}

func hashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func isSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func normalizeDescription(value string, maxLength int) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) <= maxLength {
		return value
	}
	return value[:maxLength-3] + "..."
}

func cloneListRequest(request ListRequest) ListRequest {
	return ListRequest{CWD: request.CWD, Scopes: append([]string{}, request.Scopes...)}
}

func cloneSummary(summary Summary) Summary {
	out := summary
	if summary.ResourceBase != nil {
		base := *summary.ResourceBase
		out.ResourceBase = &base
	}
	return out
}

func cloneCandidate(candidate Candidate) Candidate {
	out := candidate
	out.Summary = cloneSummary(candidate.Summary)
	out.Metadata = cloneMap(candidate.Metadata)
	return out
}

func cloneDefinition(definition Definition) Definition {
	out := definition
	out.Summary = cloneSummary(definition.Summary)
	out.ResourceManifest = append([]ResourceRef{}, definition.ResourceManifest...)
	out.Metadata = cloneMap(definition.Metadata)
	return out
}

func cloneMap(input map[string]any) map[string]any {
	if input == nil {
		return make(map[string]any)
	}
	data, err := json.Marshal(input)
	if err != nil {
		return make(map[string]any)
	}
	output := make(map[string]any)
	if err := json.Unmarshal(data, &output); err != nil {
		return make(map[string]any)
	}
	return output
}

func cloneIndexedMap(input map[string]indexedCandidate) map[string]indexedCandidate {
	output := make(map[string]indexedCandidate, len(input))
	for name, entry := range input {
		entry.candidate = cloneCandidate(entry.candidate)
		output[name] = entry
	}
	return output
}
