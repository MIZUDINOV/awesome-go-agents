package tools

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Hook is a pipeline extension point invoked before/after execution.
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
	mu            sync.RWMutex
	definitions   map[string]*Definition
	preExecute    []Hook
	postExecute   []Hook
	defaultTimeout time.Duration
	sem           chan struct{}
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

// Register adds a tool. Panics-free: returns ErrToolAlreadyExists on conflict.
func (r *Registry) Register(def *Definition) error {
	if def == nil || def.Name == "" || def.Execute == nil {
		return ErrInvalidArguments
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

// AddPreExecute / AddPostExecute append pipeline hooks.
func (r *Registry) AddPreExecute(hook Hook)  { r.preExecute = append(r.preExecute, hook) }
func (r *Registry) AddPostExecute(hook Hook) { r.postExecute = append(r.postExecute, hook) }

// Get returns a registered tool.
func (r *Registry) Get(name string) (*Definition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	def, ok := r.definitions[name]
	return def, ok
}

// Names returns all registered tool names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.definitions))
	for name := range r.definitions {
		names = append(names, name)
	}
	return names
}

// ModelTools returns the provider-neutral, model-facing tool schema slice.
// This is exactly what is sent to the LLM; runtime fields are stripped.
func (r *Registry) ModelTools() []*llm.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tools := make([]*llm.ToolDefinition, 0, len(r.definitions))
	for _, def := range r.definitions {
		tools = append(tools, &llm.ToolDefinition{
			Name:        def.Name,
			Description: def.Description,
			InputSchema: def.InputSchema,
		})
	}
	return tools
}

// Run executes a tool through the pipeline, returning the three-part Result.
func (r *Registry) Run(ctx context.Context, ec ExecContext, name, callID string, input []byte) (*Result, error) {
	r.mu.RLock()
	def, ok := r.definitions[name]
	r.mu.RUnlock()
	if !ok {
		return nil, ErrToolNotFound
	}
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

	for _, hook := range r.preExecute {
		if err := hook(ctx, name, input); err != nil {
			return nil, err
		}
	}

	execCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		execCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	canonical, err := def.Execute(execCtx, ec, input)
	if err != nil {
		return nil, err
	}
	if execCtx.Err() == context.DeadlineExceeded {
		return nil, ErrToolTimeout
	}

	for _, hook := range r.postExecute {
		if err := hook(ctx, name, input); err != nil {
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
