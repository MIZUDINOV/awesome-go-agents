package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Runtime is the loop-facing tool seam. Registry and scoped catalogs both
// implement it, so an Agent can receive an isolated visible tool view.
type Runtime interface {
	ModelTools() []*llm.ToolDefinition
	RunBatch(context.Context, ExecContext, []Call) []Outcome
}

// Catalog is the consumer-facing scoped authoring seam.
type Catalog interface {
	Runtime
	Register(*Definition) error
	Restrict([]string) error
	Deny(...string)
	SetApprovalService(ApprovalService)
}

// Scope is an agent-local catalog. Local definitions shadow root definitions;
// visibility restrictions affect both model schemas and execution resolution.
type Scope struct {
	root *Registry
	// rootSnapshot freezes the root catalog and its execution middleware at
	// scope creation. An agent therefore cannot observe a later global
	// registration or policy mutation from another agent.
	rootSnapshot *Registry
	local        *Registry
	approval     ApprovalService
	parallel     int
	mu           sync.RWMutex
	allow        map[string]struct{}
	deny         map[string]struct{}
}

// NewScope creates an isolated catalog over a root registry.
func (r *Registry) NewScope() *Scope {
	r.mu.RLock()
	opts := Options{DefaultTimeout: r.defaultTimeout, MaxParallel: r.maxParallel, Sandbox: r.sandbox, Approval: r.approval, Presentation: r.presentation, CodeRuntime: r.codeRuntime, CodeLanguage: r.codeLanguage, CodeSDKRenderer: r.codeSDKRenderer}
	rootSnapshot := New(opts)
	for name, definition := range r.definitions {
		rootSnapshot.definitions[name] = cloneDefinition(definition)
	}
	rootSnapshot.preExecute = append([]Hook(nil), r.preExecute...)
	rootSnapshot.postExecute = append([]Hook(nil), r.postExecute...)
	rootSnapshot.policies = append([]Policy(nil), r.policies...)
	rootSnapshot.guards = append([]Guard(nil), r.guards...)
	rootSnapshot.postPolicies = append([]PostPolicy(nil), r.postPolicies...)
	rootSnapshot.observers = append([]Observer(nil), r.observers...)
	local := New(opts)
	local.policies = append([]Policy(nil), r.policies...)
	local.guards = append([]Guard(nil), r.guards...)
	local.postPolicies = append([]PostPolicy(nil), r.postPolicies...)
	local.observers = append([]Observer(nil), r.observers...)
	parallel := r.maxParallel
	r.mu.RUnlock()
	return &Scope{root: r, rootSnapshot: rootSnapshot, local: local, approval: opts.Approval, parallel: parallel}
}

func (s *Scope) Register(def *Definition) error { return s.local.Register(def) }

func (s *Scope) SetApprovalService(service ApprovalService) {
	s.mu.Lock()
	s.approval = service
	s.mu.Unlock()
}

// Restrict installs an allow mask. Empty names are rejected and later calls
// replace the current mask explicitly.
func (s *Scope) Restrict(names []string) error {
	if len(names) == 0 {
		return ErrInvalidArguments
	}
	allow := make(map[string]struct{}, len(names))
	for _, name := range names {
		if name == "" {
			return ErrInvalidArguments
		}
		allow[name] = struct{}{}
	}
	s.mu.Lock()
	s.allow = allow
	s.mu.Unlock()
	return nil
}

func (s *Scope) Deny(names ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.deny == nil {
		s.deny = make(map[string]struct{})
	}
	for _, name := range names {
		s.deny[name] = struct{}{}
	}
}

func (s *Scope) visible(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, denied := s.deny[name]; denied {
		return false
	}
	if len(s.allow) > 0 {
		_, ok := s.allow[name]
		return ok
	}
	return true
}

func (s *Scope) localDefinition(name string) (*Definition, bool) { return s.local.Get(name) }

func (s *Scope) ModelTools() []*llm.ToolDefinition {
	root := s.rootSnapshot.ModelTools()
	local := s.local.ModelTools()
	visible := make(map[string]*llm.ToolDefinition, len(root)+len(local))
	localNames := make(map[string]struct{}, len(local))
	for _, tool := range local {
		localNames[tool.Name] = struct{}{}
	}
	for _, tool := range root {
		if _, ok := localNames[tool.Name]; ok || !s.visible(tool.Name) {
			continue
		}
		visible[tool.Name] = tool
	}
	for _, tool := range local {
		if s.visible(tool.Name) {
			visible[tool.Name] = tool
		}
	}
	out := make([]*llm.ToolDefinition, 0, len(visible))
	for _, tool := range visible {
		out = append(out, tool)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (s *Scope) Run(ctx context.Context, ec ExecContext, name, callID string, input json.RawMessage) (*Result, error) {
	if !s.visible(name) {
		return nil, ErrToolNotFound
	}
	if _, ok := s.localDefinition(name); ok {
		ec.Runtime = s
		return s.local.run(ctx, ec, name, callID, input, s.currentApproval())
	}
	ec.Runtime = s
	return s.rootSnapshot.run(ctx, ec, name, callID, input, s.currentApproval())
}

func (s *Scope) currentApproval() ApprovalService {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.approval
}

func (s *Scope) RunBatch(ctx context.Context, ec ExecContext, calls []Call) []Outcome {
	out := make([]Outcome, len(calls))
	for i, call := range calls {
		out[i].Call = call
	}
	// Keep the same barrier semantics as the root registry. A scoped call is
	// parallel-safe only when its currently visible definition explicitly opts
	// in; an exclusive call fences both neighbouring parallel batches.
	for start := 0; start < len(calls); {
		if !s.executionMode(calls[start].Name, calls[start].Input) {
			out[start].Result, out[start].Err = s.Run(ctx, ec, calls[start].Name, calls[start].CallID, calls[start].Input)
			start++
			continue
		}
		end := start + 1
		for end < len(calls) && s.executionMode(calls[end].Name, calls[end].Input) {
			end++
		}
		runBatchPool(s.maxParallel(), end-start, func(offset int) {
			index := start + offset
			out[index].Result, out[index].Err = s.Run(ctx, ec, calls[index].Name, calls[index].CallID, calls[index].Input)
		})
		start = end
	}
	return out
}

func (s *Scope) executionMode(name string, input json.RawMessage) bool {
	if !s.visible(name) {
		return false
	}
	if _, ok := s.localDefinition(name); ok {
		return s.local.ExecutionMode(name, input)
	}
	return s.rootSnapshot.ExecutionMode(name, input)
}

func (s *Scope) maxParallel() int {
	return s.parallel
}

// CodeGuidance renders guidance from the scoped visible catalog, never from
// definitions hidden by the root scope restrictions.
func (s *Scope) CodeGuidance() (string, error) {
	renderer := s.rootSnapshot.codeSDKRenderer
	presentation := s.rootSnapshot.presentation
	codeRuntime := s.rootSnapshot.codeRuntime
	if presentation == PresentationNative {
		return "", nil
	}
	if codeRuntime == nil {
		return "", ErrCodeModeUnavailable
	}
	if renderer == nil {
		return "", fmt.Errorf("%w: language-specific SDK renderer is required", ErrCodeModeUnavailable)
	}
	native := make([]*llm.ToolDefinition, 0)
	for _, tool := range s.ModelTools() {
		if tool.Name != "run_code" {
			native = append(native, tool)
		}
	}
	return renderer(native)
}
