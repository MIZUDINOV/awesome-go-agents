package skill

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

type MemoryProvider struct {
	mu          sync.RWMutex
	name        string
	rank        int
	definitions map[string]Definition
	revision    uint64
}

func (*MemoryProvider) SupportsPinnedLookup() bool { return true }

func NewMemoryProvider(name string, rank int, definitions ...Definition) (*MemoryProvider, error) {
	if name == "" {
		return nil, fmt.Errorf("%w: memory provider name", ErrInvalidSkill)
	}
	provider := &MemoryProvider{
		name:        name,
		rank:        rank,
		definitions: make(map[string]Definition),
	}
	if rank == 0 {
		provider.rank = RuntimeRank
	}
	for _, definition := range definitions {
		if err := provider.Set(definition); err != nil {
			return nil, err
		}
	}
	return provider, nil
}

func (p *MemoryProvider) Set(definition Definition) error {
	if err := validateDefinition(definition); err != nil {
		return err
	}
	if definition.Provider != p.name {
		return fmt.Errorf("%w: definition provider %q does not match %q", ErrInvalidSkill, definition.Provider, p.name)
	}
	p.mu.Lock()
	p.definitions[definition.Name] = cloneDefinition(definition)
	p.revision++
	p.mu.Unlock()
	return nil
}

func (p *MemoryProvider) Delete(name string) {
	p.mu.Lock()
	delete(p.definitions, name)
	p.revision++
	p.mu.Unlock()
}

func (p *MemoryProvider) List(ctx context.Context, _ ListRequest) (Observation, error) {
	if err := ctx.Err(); err != nil {
		return Observation{}, err
	}
	p.mu.RLock()
	names := make([]string, 0, len(p.definitions))
	for name := range p.definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	candidates := make([]Candidate, 0, len(names))
	for _, name := range names {
		candidates = append(candidates, candidateFromDefinition(p.definitions[name], p.rank, name))
	}
	revision := fmt.Sprintf("%d", p.revision)
	p.mu.RUnlock()
	return Observation{Candidates: candidates, Complete: true, Revision: revision}, nil
}

func (p *MemoryProvider) Get(
	ctx context.Context,
	candidate Candidate,
	_ Lookup,
) (Definition, error) {
	if err := ctx.Err(); err != nil {
		return Definition{}, err
	}
	p.mu.RLock()
	definition, exists := p.definitions[candidate.Name]
	p.mu.RUnlock()
	if !exists {
		return Definition{}, ErrSkillNotFound
	}
	return cloneDefinition(definition), nil
}
