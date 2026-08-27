package skill

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

func TestRegistryPrecedence(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	builtIn := mustMemoryProvider(t, "built-in", 600,
		mustDefinition(t, "shared", "built in", "built-in", true, true),
		mustDefinition(t, "only-built-in", "baseline", "built-in", true, true),
	)
	tenant := mustMemoryProvider(t, "tenant", 400,
		mustDefinition(t, "shared", "tenant", "tenant", true, true),
	)
	project := mustMemoryProvider(t, "project", 900,
		mustDefinition(t, "shared", "project", "project", true, true),
	)
	mustRegisterProvider(t, registry, ProviderOptions{Name: "built-in", Provider: builtIn})
	mustRegisterProvider(t, registry, ProviderOptions{Name: "tenant", Scope: "tenant:t1", Provider: tenant})
	mustRegisterProvider(t, registry, ProviderOptions{Name: "project", Scope: "project:p1", Provider: project})

	snapshot, err := registry.Snapshot(t.Context(), ListRequest{Scopes: []string{"tenant:t1", "project:p1"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Skills) != 2 {
		t.Fatalf("skills = %+v", snapshot.Skills)
	}
	if snapshot.Skills[1].Name != "shared" || snapshot.Skills[1].Description != "project" {
		t.Fatalf("nearest scope did not win: %+v", snapshot.Skills)
	}
}

func TestRegistryRankProviderAndLocalOrder(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	first := mustMemoryProvider(t, "first", 200, mustDefinition(t, "shared", "first", "first", true, true))
	second := mustMemoryProvider(t, "second", 100, mustDefinition(t, "shared", "second", "second", true, true))
	mustRegisterProvider(t, registry, ProviderOptions{Name: "first", Provider: first})
	mustRegisterProvider(t, registry, ProviderOptions{Name: "second", Provider: second})

	snapshot, err := registry.Snapshot(t.Context(), ListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Skills) != 1 || snapshot.Skills[0].Provider != "second" {
		t.Fatalf("lower rank did not win: %+v", snapshot.Skills)
	}
}

func TestRegistryIncompleteObservationIsNotCached(t *testing.T) {
	t.Parallel()
	provider := &countingProvider{definition: mustDefinition(t, "one", "one", "counting", true, true)}
	registry := NewRegistry(8)
	mustRegisterProvider(t, registry, ProviderOptions{Name: "counting", Provider: provider})

	first, err := registry.Snapshot(t.Context(), ListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	second, err := registry.Snapshot(t.Context(), ListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if first.Complete || second.Complete || provider.lists.Load() != 2 {
		t.Fatalf("incomplete observations must not cache: first=%+v second=%+v lists=%d", first, second, provider.lists.Load())
	}
}

func TestRegistrationInvalidationAndDisposal(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	provider := mustMemoryProvider(t, "provider", 100, mustDefinition(t, "one", "one", "provider", true, true))
	var changes atomic.Int64
	unsubscribe := registry.Subscribe(func(Invalidation) { changes.Add(1) })
	t.Cleanup(unsubscribe)
	registration := mustRegisterProvider(t, registry, ProviderOptions{Name: "provider", Provider: provider})
	registration.Invalidate("changed")
	if err := registration.Close(); err != nil {
		t.Fatal(err)
	}
	registration.Invalidate("stale")
	if changes.Load() != 3 {
		t.Fatalf("change count = %d, want 3", changes.Load())
	}
	_, err := registry.Get(t.Context(), Lookup{Name: "one"})
	if !errors.Is(err, ErrSkillNotFound) {
		t.Fatalf("Get() error = %v, want not found", err)
	}
}

func TestRegistryPinnedMismatch(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	provider := mustMemoryProvider(t, "provider", 100, mustDefinition(t, "one", "one", "provider", true, true))
	mustRegisterProvider(t, registry, ProviderOptions{Name: "provider", Provider: provider})
	snapshot, err := registry.Snapshot(t.Context(), ListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	updated, err := Parse([]byte("---\nname: one\ndescription: changed\n---\nChanged instructions"), ParseOptions{
		Provider: "provider",
		Source:   "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := provider.Set(updated); err != nil {
		t.Fatal(err)
	}
	_, err = registry.LoadPinned(t.Context(), ListRequest{}, snapshot.Skills[0])
	if !errors.Is(err, ErrPinnedMismatch) {
		t.Fatalf("LoadPinned() error = %v, want pinned mismatch", err)
	}
}

func TestRegistryCancellationStopsHostileProviderLoad(t *testing.T) {
	t.Parallel()
	definition := mustDefinition(t, "hostile", "hostile", "hostile", true, true)
	provider := &hostileProvider{definition: definition, blockGet: make(chan struct{}), getStarted: make(chan struct{})}
	registry := NewRegistry(8)
	mustRegisterProvider(t, registry, ProviderOptions{Name: "hostile", Provider: provider})
	ctx, cancel := context.WithCancel(t.Context())
	loaded := make(chan error, 1)
	go func() {
		_, err := registry.Get(ctx, Lookup{Name: "hostile"})
		loaded <- err
	}()
	<-provider.getStarted
	cancel()
	if err := <-loaded; !errors.Is(err, context.Canceled) {
		t.Fatalf("Get() error = %v, want context.Canceled", err)
	}
	close(provider.blockGet)
}

func TestRegistryConcurrentLifecycle(t *testing.T) {
	t.Parallel()
	registry := NewRegistry(8)
	var group sync.WaitGroup
	for index := 0; index < 16; index++ {
		index := index
		group.Add(1)
		go func() {
			defer group.Done()
			name := fmt.Sprintf("provider-%d", index)
			provider := mustMemoryProvider(t, name, 100, mustDefinition(t, fmt.Sprintf("skill-%d", index), name, name, true, true))
			registration, err := registry.RegisterProvider(ProviderOptions{Name: name, Provider: provider})
			if err != nil {
				t.Errorf("RegisterProvider() error = %v", err)
				return
			}
			_, _ = registry.Snapshot(t.Context(), ListRequest{})
			registration.Invalidate("test")
			_ = registration.Close()
		}()
	}
	group.Wait()
}

type countingProvider struct {
	lists      atomic.Int64
	definition Definition
}

type hostileProvider struct {
	definition Definition
	blockGet   chan struct{}
	getOnce    sync.Once
	getStarted chan struct{}
}

func (p *hostileProvider) List(context.Context, ListRequest) (Observation, error) {
	return Observation{Candidates: []Candidate{candidateFromDefinition(p.definition, 100, p.definition.Name)}, Complete: true}, nil
}

func (p *hostileProvider) Get(context.Context, Candidate, Lookup) (Definition, error) {
	p.getOnce.Do(func() { close(p.getStarted) })
	<-p.blockGet
	return cloneDefinition(p.definition), nil
}

func (p *countingProvider) List(context.Context, ListRequest) (Observation, error) {
	p.lists.Add(1)
	return Observation{
		Candidates: []Candidate{candidateFromDefinition(p.definition, 100, p.definition.Name)},
		Complete:   false,
	}, nil
}

func (p *countingProvider) Get(context.Context, Candidate, Lookup) (Definition, error) {
	return cloneDefinition(p.definition), nil
}

func mustDefinition(t *testing.T, name, description, provider string, model, user bool) Definition {
	t.Helper()
	definition, err := Parse([]byte("---\nname: "+name+"\ndescription: "+description+"\n---\nInstructions for "+name), ParseOptions{
		Provider: provider,
		Source:   "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	definition.Policy = InvocationPolicy{Model: model, User: user}
	return definition
}

func mustMemoryProvider(t *testing.T, name string, rank int, definitions ...Definition) *MemoryProvider {
	t.Helper()
	provider, err := NewMemoryProvider(name, rank, definitions...)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func mustRegisterProvider(t *testing.T, registry *Registry, options ProviderOptions) *Registration {
	t.Helper()
	registration, err := registry.RegisterProvider(options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = registration.Close() })
	return registration
}
