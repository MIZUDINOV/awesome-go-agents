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
	RunBatch(context.Context, ToolRunContext, []Call) []Outcome
}

// Catalog is the consumer-facing scoped authoring seam.
type Catalog interface {
	Runtime
	Register(*Definition) error
	RegisterTool(*Definition) (*Registration, error)
	Unregister(string) error
	Restrict([]string) error
	Deny(...string)
	SetApprovalService(ApprovalService)
}

// Scope is an agent-local catalog. Local definitions shadow root definitions;
// visibility restrictions apply to inherited tools while local registrations
// remain available to their owning scope.
type Scope struct {
	// root stays live for inherited definitions and middleware. Local
	// definitions are isolated and shadow root names only for this scope.
	root        *Registry
	local       *Registry
	approval    ApprovalService
	approvalSet bool
	parallel    int
	mu          sync.RWMutex
	allow       map[string]struct{}
	deny        map[string]struct{}
}

// NewScope creates an isolated catalog over a root registry.
func (r *Registry) NewScope() *Scope {
	snapshot := r.snapshot()
	return &Scope{
		root: r, local: New(Options{
			DefaultTimeout:  snapshot.defaultTimeout,
			MaxParallel:     snapshot.maxParallel,
			Sandbox:         snapshot.sandbox,
			Approval:        snapshot.approval,
			Presentation:    snapshot.presentation,
			CodeRuntime:     snapshot.codeRuntime,
			CodeLanguage:    snapshot.codeLanguage,
			CodeSDKRenderer: snapshot.codeSDKRenderer,
		}),
		parallel: snapshot.maxParallel,
	}
}

func (s *Scope) Register(def *Definition) error {
	if err := s.local.Register(def); err != nil {
		return err
	}
	return nil
}

func (s *Scope) RegisterTool(def *Definition) (*Registration, error) {
	name := ""
	if def != nil {
		name = def.Name
	}
	if err := s.Register(def); err != nil {
		return nil, err
	}
	return newRegistration(func() error { return s.unregisterLocal(name) }), nil
}

func (s *Scope) unregisterLocal(name string) error {
	if err := s.local.Unregister(name); err != nil {
		return err
	}
	return nil
}

func (s *Scope) Unregister(name string) error { return s.unregisterLocal(name) }

func (s *Scope) SetApprovalService(service ApprovalService) {
	s.mu.Lock()
	s.approval = service
	s.approvalSet = true
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

// RestrictHandle installs an allow mask and restores the previous mask when
// the returned handle is disposed.
func (s *Scope) RestrictHandle(names []string) (*Registration, error) {
	s.mu.RLock()
	previous := make(map[string]struct{}, len(s.allow))
	for name := range s.allow {
		previous[name] = struct{}{}
	}
	s.mu.RUnlock()
	if err := s.Restrict(names); err != nil {
		return nil, err
	}
	return newRegistration(func() error {
		s.mu.Lock()
		s.allow = previous
		s.mu.Unlock()
		return nil
	}), nil
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

// DenyHandle adds a deny mask and restores the previous mask on disposal.
func (s *Scope) DenyHandle(names ...string) *Registration {
	s.mu.RLock()
	previous := make(map[string]struct{}, len(s.deny))
	for name := range s.deny {
		previous[name] = struct{}{}
	}
	s.mu.RUnlock()
	s.Deny(names...)
	return newRegistration(func() error {
		s.mu.Lock()
		s.deny = previous
		s.mu.Unlock()
		return nil
	})
}

func (s *Scope) visibleInherited(name string) bool {
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

func (s *Scope) localNames() map[string]struct{} {
	s.local.mu.RLock()
	defer s.local.mu.RUnlock()
	names := make(map[string]struct{}, len(s.local.definitions))
	for name := range s.local.definitions {
		names[name] = struct{}{}
	}
	return names
}

func (s *Scope) executionRegistry() *Registry {
	registry := s.root.snapshot()
	s.local.mu.RLock()
	for name, definition := range s.local.definitions {
		registry.definitions[name] = cloneDefinition(definition)
	}
	s.local.mu.RUnlock()
	if approval, overridden := s.currentApproval(); overridden {
		registry.SetApprovalService(approval)
	}
	return registry
}

func (s *Scope) ModelTools() []*llm.ToolDefinition {
	combined := s.executionRegistry()
	localNames := s.localNames()
	visible := make(map[string]*llm.ToolDefinition)
	for _, tool := range combined.ModelTools() {
		if _, local := localNames[tool.Name]; local || isReservedCodeModeTool(tool.Name) || s.visibleInherited(tool.Name) {
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

// GuidanceSections returns guidance only for the currently visible catalog.
func (s *Scope) GuidanceSections() []PromptSection {
	combined := s.executionRegistry()
	localNames := s.localNames()
	visible := make(map[string]struct{})
	for _, tool := range s.ModelTools() {
		visible[tool.Name] = struct{}{}
	}
	sections := make([]PromptSection, 0)
	for name := range visible {
		definition, ok := combined.Get(name)
		if !ok {
			continue
		}
		if _, local := localNames[name]; !local && !isReservedCodeModeTool(name) && !s.visibleInherited(name) {
			continue
		}
		sections = append(sections, definition.Guidance...)
	}
	sort.SliceStable(sections, func(i, j int) bool {
		if sections[i].Title == sections[j].Title {
			return sections[i].Content < sections[j].Content
		}
		return sections[i].Title < sections[j].Title
	})
	return sections
}

func (s *Scope) Run(ctx context.Context, ec ToolRunContext, name, callID string, input json.RawMessage) (*Result, error) {
	if _, ok := s.localDefinition(name); ok {
		ec.Runtime = s
		return s.executionRegistry().run(ctx, ec, name, callID, input, s.currentApprovalValue())
	}
	if !isReservedCodeModeTool(name) && !s.visibleInherited(name) {
		return nil, ErrToolNotFound
	}
	ec.Runtime = s
	return s.root.run(ctx, ec, name, callID, input, s.currentApprovalValue())
}

func (s *Scope) currentApproval() (ApprovalService, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.approval, s.approvalSet
}

func (s *Scope) currentApprovalValue() ApprovalService {
	if approval, overridden := s.currentApproval(); overridden {
		return approval
	}
	return nil
}

func (s *Scope) RunBatch(ctx context.Context, ec ToolRunContext, calls []Call) []Outcome {
	return runRollingBatch(ctx, s.maxParallel(), calls,
		func(call Call) bool { return s.executionMode(call.Name, call.Input) },
		func(call Call) (*Result, error) { return s.Run(ctx, ec, call.Name, call.CallID, call.Input) })
}

func (s *Scope) executionMode(name string, input json.RawMessage) bool {
	if _, ok := s.localDefinition(name); ok {
		return s.executionRegistry().ExecutionMode(name, input)
	}
	if !isReservedCodeModeTool(name) && !s.visibleInherited(name) {
		return false
	}
	return s.root.ExecutionMode(name, input)
}

// Code Mode is a transport boundary, not an inherited native tool. Scope
// restrictions must not disable the reserved run_code bridge.
func isReservedCodeModeTool(name string) bool { return name == "run_code" }

func (s *Scope) maxParallel() int {
	return s.parallel
}

// CodeGuidance renders guidance from the scoped visible catalog, never from
// definitions hidden by the root scope restrictions.
func (s *Scope) CodeGuidance() (string, error) {
	root := s.root.snapshot()
	renderer := root.codeSDKRenderer
	presentation := root.presentation
	codeRuntime := root.codeRuntime
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
