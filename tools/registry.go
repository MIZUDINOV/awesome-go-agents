package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"sync"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// toolNamePattern restricts tool names to stable, lower-snake identifiers
// (H-TOOLS-005).
var toolNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// Hook is a pipeline extension point invoked before/after execution. Hooks are
// snapshotted at dispatch time, so a hook slice can be mutated concurrently
// without racing the running pipeline.
type Hook func(ctx context.Context, name string, input json.RawMessage) error

// Options configures a Registry.
type Options struct {
	// DefaultTimeout is applied when a tool has no explicit Timeout.
	DefaultTimeout time.Duration
	// MaxParallel bounds concurrent tool executions (0 = no limit).
	MaxParallel int
}

// Registry holds tool definitions and the execution pipeline.
type Registry struct {
	mu             sync.RWMutex
	definitions    map[string]*Definition
	preExecute     []Hook
	postExecute    []Hook
	defaultTimeout time.Duration
	sem            chan struct{}
}

// New returns an empty Registry.
func New(opts Options) *Registry {
	if opts.DefaultTimeout <= 0 {
		opts.DefaultTimeout = 5 * time.Minute
	}
	r := &Registry{
		definitions:    make(map[string]*Definition),
		defaultTimeout: opts.DefaultTimeout,
	}
	if opts.MaxParallel > 0 {
		r.sem = make(chan struct{}, opts.MaxParallel)
	}
	return r
}

// Register adds a tool. It returns an error (rather than panicking) on a
// missing/invalid definition, an unsupported or malformed schema, an invalid
// name, or a name conflict.
func (r *Registry) Register(def *Definition) error {
	if def == nil || def.Name == "" || def.Execute == nil {
		return ErrInvalidArguments
	}
	if !toolNamePattern.MatchString(def.Name) {
		return fmt.Errorf("%w: tool name %q invalid (must match %s)", ErrInvalidArguments, def.Name, toolNamePattern.String())
	}
	if err := ValidateSchema(def.InputSchema); err != nil {
		return fmt.Errorf("%w: tool %q input schema: %v", ErrInvalidArguments, def.Name, err)
	}
	if err := ValidateSchema(def.OutputSchema); err != nil {
		return fmt.Errorf("%w: tool %q output schema: %v", ErrInvalidArguments, def.Name, err)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.definitions[def.Name]; exists {
		return ErrToolAlreadyExists
	}
	r.definitions[def.Name] = def
	return nil
}

// MustRegister registers or panics on conflict (for init-time registration).
func (r *Registry) MustRegister(def *Definition) {
	if err := r.Register(def); err != nil {
		panic(err)
	}
}

// AddPreExecute / AddPostExecute append pipeline hooks. Safe to call while
// runs are in flight: runs operate on a snapshot taken at dispatch time.
func (r *Registry) AddPreExecute(hook Hook) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.preExecute = append(r.preExecute, hook)
}
func (r *Registry) AddPostExecute(hook Hook) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.postExecute = append(r.postExecute, hook)
}

func (r *Registry) snapshotHooks() (pre, post []Hook) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	pre = append([]Hook(nil), r.preExecute...)
	post = append([]Hook(nil), r.postExecute...)
	return pre, post
}

// Get returns a registered tool.
func (r *Registry) Get(name string) (*Definition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	def, ok := r.definitions[name]
	return def, ok
}

// Names returns all registered tool names in deterministic (sorted) order.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.definitions))
	for name := range r.definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// ModelTools returns the provider-neutral, model-facing tool schema slice in
// deterministic (sorted) order. This is exactly what is sent to the LLM;
// runtime fields are stripped and the ordering no longer depends on Go map
// iteration (H-ANTI-016).
func (r *Registry) ModelTools() []*llm.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.definitions))
	for name := range r.definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	tools := make([]*llm.ToolDefinition, 0, len(names))
	for _, name := range names {
		def := r.definitions[name]
		tools = append(tools, &llm.ToolDefinition{
			Name:        def.Name,
			Description: def.Description,
			InputSchema: def.InputSchema,
		})
	}
	return tools
}

// Run executes a tool through the pipeline, returning the three-part Result.
//
// Pipeline order (mirrors the review checklist §9.1):
//
//	resolve visible tool -> freeze arguments -> input validation -> pre-execute
//	hooks -> execute wrappers (timeout/cancellation) -> tool body -> output
//	validation -> post-execute hooks -> render canonical/model/UI.
//
// The executor is never invoked for invalid arguments (H-RUNTIME-001) or with
// a mutating Input schema.
func (r *Registry) Run(ctx context.Context, ec ExecContext, name, callID string, input []byte) (*Result, error) {
	r.mu.RLock()
	def, ok := r.definitions[name]
	r.mu.RUnlock()
	if !ok {
		return nil, ErrToolNotFound
	}
	preHooks, postHooks := r.snapshotHooks()

	// Freeze arguments: never hand callers' mutable backing array to the
	// executor or the model.
	args := make([]byte, len(input))
	copy(args, input)

	if r.sem != nil {
		select {
		case r.sem <- struct{}{}:
			defer func() { <-r.sem }()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	timeout := def.Timeout
	if timeout <= 0 {
		timeout = r.defaultTimeout
	}

	// Input validation before any hook or side effect (H-RUNTIME-001).
	if err := ValidateInput(def.InputSchema, args); err != nil {
		return nil, err
	}

	for _, hook := range preHooks {
		if err := hook(ctx, name, args); err != nil {
			return nil, err
		}
	}

	execCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		execCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	canonical, err := def.Execute(execCtx, ec, args)
	if err != nil {
		if execCtx.Err() == context.DeadlineExceeded && ctx.Err() == nil {
			return nil, ErrToolTimeout
		}
		return nil, err
	}
	if execCtx.Err() == context.DeadlineExceeded && ctx.Err() == nil {
		return nil, ErrToolTimeout
	}

	// Output validation before rendering/persistence (H-TOOLS-007).
	if len(def.OutputSchema) > 0 && string(def.OutputSchema) != "null" {
		encoded, err := json.Marshal(canonical)
		if err != nil {
			return nil, fmt.Errorf("%w: canonical output is not JSON-serializable: %v", ErrInvalidOutput, err)
		}
		if err := ValidateInput(def.OutputSchema, encoded); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidOutput, err)
		}
	}

	for _, hook := range postHooks {
		if err := hook(ctx, name, args); err != nil {
			return nil, err
		}
	}

	result := &Result{Name: name, CallID: callID, Canonical: canonical}
	if def.RenderModel != nil {
		modelFacing, err := def.RenderModel(canonical)
		if err != nil {
			return nil, err
		}
		result.ModelFacing = modelFacing
	}
	if def.PresentUI != nil {
		ui, err := def.PresentUI(canonical)
		if err != nil {
			return nil, err
		}
		result.UI = ui
	}
	return result, nil
}
