package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"sync"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
	"github.com/MIZUDINOV/awesome-go-agents/llm"
	"github.com/MIZUDINOV/awesome-go-agents/session"
)

var ErrCodeModeUnavailable = fmt.Errorf("code mode requires an injected code runtime")

// toolNamePattern restricts tool names to stable, lower-snake identifiers
// (H-TOOLS-005).
var toolNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// Hook is a pipeline extension point invoked before/after execution. Hooks are
// snapshotted at dispatch time, so a hook slice can be mutated concurrently
// without racing the running pipeline.
type Hook func(ctx context.Context, name string, input json.RawMessage) error
type Observer func(context.Context, *Result)

// Options configures a Registry.
type Options struct {
	// DefaultTimeout is applied when a tool has no explicit Timeout.
	DefaultTimeout time.Duration
	// MaxParallel bounds concurrent tool executions (0 = no limit).
	MaxParallel int
	// Sandbox is the central tool-admission boundary.
	Sandbox         integration.Sandbox
	Approval        ApprovalService
	Presentation    PresentationMode
	CodeRuntime     CodeRuntime
	CodeLanguage    string
	CodeSDKRenderer func([]*llm.ToolDefinition) (string, error)
}

// Registry holds tool definitions and the execution pipeline.
type Registry struct {
	mu              sync.RWMutex
	definitions     map[string]*Definition
	preExecute      []Hook
	postExecute     []Hook
	defaultTimeout  time.Duration
	sem             chan struct{}
	sandbox         integration.Sandbox
	approval        ApprovalService
	maxParallel     int
	policies        []Policy
	guards          []Guard
	postPolicies    []PostPolicy
	observers       []Observer
	presentation    PresentationMode
	codeRuntime     CodeRuntime
	codeLanguage    string
	codeSDKRenderer func([]*llm.ToolDefinition) (string, error)
}

// New returns an empty Registry.
func New(opts Options) *Registry {
	if opts.DefaultTimeout <= 0 {
		opts.DefaultTimeout = 5 * time.Minute
	}
	if opts.MaxParallel <= 0 {
		opts.MaxParallel = 10
	}
	r := &Registry{
		definitions:     make(map[string]*Definition),
		defaultTimeout:  opts.DefaultTimeout,
		sandbox:         opts.Sandbox,
		approval:        opts.Approval,
		maxParallel:     opts.MaxParallel,
		presentation:    opts.Presentation,
		codeRuntime:     opts.CodeRuntime,
		codeLanguage:    opts.CodeLanguage,
		codeSDKRenderer: opts.CodeSDKRenderer,
	}
	if r.presentation == "" {
		r.presentation = PresentationNative
	}
	if r.codeLanguage == "" {
		r.codeLanguage = "go"
	}
	r.sem = make(chan struct{}, opts.MaxParallel)
	return r
}

// AddPolicy appends a pre-execution policy. Calls snapshot policies at
// dispatch, so registration does not race an in-flight tool.
func (r *Registry) AddPolicy(policy Policy) {
	if policy == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policies = append(r.policies, policy)
}

// AddGuard appends a monotonic guard evaluated after policy decisions.
func (r *Registry) AddGuard(guard Guard) {
	if guard == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.guards = append(r.guards, guard)
}

func (r *Registry) AddPostPolicy(policy PostPolicy) {
	if policy == nil {
		return
	}
	r.mu.Lock()
	r.postPolicies = append(r.postPolicies, policy)
	r.mu.Unlock()
}

func (r *Registry) AddObserver(observer Observer) {
	if observer == nil {
		return
	}
	r.mu.Lock()
	r.observers = append(r.observers, observer)
	r.mu.Unlock()
}

// SetApprovalService replaces the process-local approval seam for future
// executions. In-flight calls retain the service selected at their decision.
func (r *Registry) SetApprovalService(service ApprovalService) {
	r.mu.Lock()
	r.approval = service
	r.mu.Unlock()
}

func (r *Registry) snapshotPolicies() ([]Policy, []Guard, []PostPolicy) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]Policy(nil), r.policies...), append([]Guard(nil), r.guards...), append([]PostPolicy(nil), r.postPolicies...)
}

func (r *Registry) snapshotObservers() []Observer {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]Observer(nil), r.observers...)
}

// Call is one model-ordered invocation prepared by the loop after its
// tool/call barrier is durable.
type Call struct {
	Name   string
	CallID string
	Input  json.RawMessage
}

// Outcome preserves the model's call order even when bodies overlapped.
type Outcome struct {
	Call   Call
	Result *Result
	Err    error
}

// ExecutionMode reports whether a validated call explicitly opts in to
// parallel dispatch. Unknown, invalid, or classifier-failing calls are always
// exclusive.
func (r *Registry) ExecutionMode(name string, input json.RawMessage) bool {
	if name == "run_code" {
		return false
	}
	r.mu.RLock()
	def, ok := r.definitions[name]
	r.mu.RUnlock()
	if !ok || ValidateInput(def.InputSchema, input) != nil {
		return false
	}
	return def.IsConcurrencySafe(input)
}

// RunBatch executes consecutive parallel-safe calls in bounded pools and
// treats every exclusive call as a barrier. It returns results in model order;
// callers must commit that order to the durable surface.
func (r *Registry) RunBatch(ctx context.Context, ec ExecContext, calls []Call) []Outcome {
	outcomes := make([]Outcome, len(calls))
	for i, call := range calls {
		outcomes[i].Call = call
	}
	for start := 0; start < len(calls); {
		if !r.ExecutionMode(calls[start].Name, calls[start].Input) {
			outcomes[start].Result, outcomes[start].Err = r.Run(ctx, ec, calls[start].Name, calls[start].CallID, calls[start].Input)
			start++
			continue
		}
		end := start + 1
		for end < len(calls) && r.ExecutionMode(calls[end].Name, calls[end].Input) {
			end++
		}
		runBatchPool(r.maxParallel, end-start, func(offset int) {
			index := start + offset
			outcomes[index].Result, outcomes[index].Err = r.Run(ctx, ec, calls[index].Name, calls[index].CallID, calls[index].Input)
		})
		start = end
	}
	return outcomes
}

func runBatchPool(limit, size int, run func(int)) {
	if size == 0 {
		return
	}
	if limit > size {
		limit = size
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	for worker := 0; worker < limit; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				run(index)
			}
		}()
	}
	for index := 0; index < size; index++ {
		jobs <- index
	}
	close(jobs)
	wg.Wait()
}

// Register adds a tool. It returns an error (rather than panicking) on a
// missing/invalid definition, an unsupported or malformed schema, an invalid
// name, or a name conflict.
func (r *Registry) Register(def *Definition) error {
	if def == nil || def.Name == "" || def.Execute == nil {
		return ErrInvalidArguments
	}
	if def.Name == "run_code" {
		return fmt.Errorf("%w: run_code is reserved for Code Mode", ErrInvalidArguments)
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
	r.definitions[def.Name] = cloneDefinition(def)
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
	if !ok {
		return nil, false
	}
	return cloneDefinition(def), true
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
	if r.presentation == PresentationCode {
		if r.codeRuntime == nil {
			return nil
		}
		return []*llm.ToolDefinition{runCodeToolDefinition()}
	}
	tools := r.nativeModelToolsLocked()
	if r.presentation == PresentationBoth && r.codeRuntime != nil {
		tools = append(tools, runCodeToolDefinition())
	}
	return tools
}

func (r *Registry) nativeModelToolsLocked() []*llm.ToolDefinition {
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
			InputSchema: append(json.RawMessage(nil), def.InputSchema...),
		})
	}
	return tools
}

func (r *Registry) runCodeDefinition() (*Definition, bool) {
	r.mu.RLock()
	runtime, language := r.codeRuntime, r.codeLanguage
	r.mu.RUnlock()
	if runtime == nil {
		return nil, false
	}
	return &Definition{
		Name: "run_code", Description: "Execute generated code through the scoped tool SDK.",
		InputSchema: runCodeToolDefinition().InputSchema,
		Execute: func(ctx context.Context, ec ExecContext, input json.RawMessage) (any, error) {
			var args struct {
				Code     string `json:"code"`
				Language string `json:"language,omitempty"`
			}
			if err := json.Unmarshal(input, &args); err != nil || args.Code == "" {
				return nil, ErrInvalidArguments
			}
			if args.Language == "" {
				args.Language = language
			}
			return runtime.ExecuteCode(ctx, ec, args.Code, args.Language)
		},
		ConcurrencySafe:  false,
		MutatesWorkspace: true,
	}, true
}

func runCodeToolDefinition() *llm.ToolDefinition {
	return &llm.ToolDefinition{Name: "run_code", Description: "Execute generated code through the scoped tool SDK.", InputSchema: json.RawMessage(`{"type":"object","properties":{"code":{"type":"string"},"language":{"type":"string"}},"required":["code"],"additionalProperties":false}`)}
}

// CodeGuidance returns deterministic SDK guidance for a Code Mode request.
func (r *Registry) CodeGuidance() (string, error) {
	r.mu.RLock()
	presentation, codeRuntime, renderer := r.presentation, r.codeRuntime, r.codeSDKRenderer
	r.mu.RUnlock()
	if presentation == PresentationNative {
		return "", nil
	}
	if codeRuntime == nil {
		return "", ErrCodeModeUnavailable
	}
	if renderer != nil {
		r.mu.RLock()
		tools := r.nativeModelToolsLocked()
		r.mu.RUnlock()
		return renderer(tools)
	}
	return "", fmt.Errorf("%w: language-specific SDK renderer is required", ErrCodeModeUnavailable)
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
func (r *Registry) Run(ctx context.Context, ec ExecContext, name, callID string, input []byte) (result *Result, retErr error) {
	return r.run(ctx, ec, name, callID, input, nil)
}

func (r *Registry) run(ctx context.Context, ec ExecContext, name, callID string, input []byte, approvalOverride ApprovalService) (result *Result, retErr error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrAbortedBeforeDispatch, err)
	}
	r.mu.RLock()
	presentation, defaultApproval := r.presentation, r.approval
	r.mu.RUnlock()
	r.mu.RLock()
	def, ok := r.definitions[name]
	r.mu.RUnlock()
	if name == "run_code" && presentation != PresentationNative {
		def, ok = r.runCodeDefinition()
		if !ok {
			return nil, ErrCodeModeUnavailable
		}
	}
	if presentation == PresentationCode && name != "run_code" {
		return nil, ErrToolNotFound
	}
	if !ok {
		return nil, ErrToolNotFound
	}
	if ec.Runtime == nil {
		ec.Runtime = r
	}
	observers := r.snapshotObservers()
	defer func() {
		if retErr != nil && result != nil && result.Kind != OutcomeFailure {
			code := toolErrorCode(retErr)
			result.Kind = OutcomeFailure
			result.Canonical = nil
			result.ModelFacing = nil
			result.UI = nil
			result.Meta = nil
			result.Content = nil
			result.AdditionalContexts = nil
			result.ConcludesTurn = false
			result.Code = code
			result.Failure = &Failure{Code: code, Message: failureMessage(retErr)}
		}
		if result == nil && retErr != nil {
			code := toolErrorCode(retErr)
			result = &Result{Name: name, CallID: callID, Kind: OutcomeFailure, Code: code, Failure: &Failure{Code: code, Message: failureMessage(retErr)}}
		}
		if result == nil {
			return
		}
		frozen := result.Freeze()
		for _, observer := range observers {
			func() {
				defer func() { _ = recover() }()
				observer(ctx, frozen)
			}()
		}
		result = frozen
	}()
	defer func() {
		if def.FinalizeContent == nil {
			return
		}
		if retErr != nil && result != nil && result.Kind != OutcomeFailure {
			code := toolErrorCode(retErr)
			result.Kind = OutcomeFailure
			result.Canonical = nil
			result.ModelFacing = nil
			result.UI = nil
			result.Meta = nil
			result.Content = nil
			result.AdditionalContexts = nil
			result.ConcludesTurn = false
			result.Code = code
			result.Failure = &Failure{Code: code, Message: failureMessage(retErr)}
		}
		if result == nil {
			code := toolErrorCode(retErr)
			result = &Result{Name: name, CallID: callID, Kind: OutcomeFailure, Code: code, Failure: &Failure{Code: code, Message: failureMessage(retErr)}}
		}
		var finalizeErr error
		func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					finalizeErr = fmt.Errorf("finalizer panic: %v", recovered)
				}
			}()
			finalizeErr = def.FinalizeContent(result)
		}()
		if finalizeErr != nil {
			if retErr == nil {
				retErr = fmt.Errorf("finalize tool content: %w", finalizeErr)
			}
			result.Kind = OutcomeFailure
			result.Canonical = nil
			result.ModelFacing = nil
			result.UI = nil
			result.Meta = nil
			result.Content = nil
			result.AdditionalContexts = nil
			result.ConcludesTurn = false
			result.Code = "FINALIZE_FAILED"
			result.Failure = &Failure{Code: result.Code, Message: failureMessage(finalizeErr)}
		}
		if retErr != nil && result.Kind != OutcomeFailure {
			code := toolErrorCode(retErr)
			result.Kind = OutcomeFailure
			result.Canonical = nil
			result.ModelFacing = nil
			result.UI = nil
			result.Meta = nil
			result.Content = nil
			result.AdditionalContexts = nil
			result.ConcludesTurn = false
			result.Code = code
			result.Failure = &Failure{Code: code, Message: failureMessage(retErr)}
		}
		result = result.Freeze()
	}()
	defer func() {
		if recovered := recover(); recovered != nil {
			retErr = fmt.Errorf("tool execution panic: %v", recovered)
			if result == nil {
				result = &Result{Name: name, CallID: callID, Kind: OutcomeFailure, Code: "TOOL_PANIC", Failure: &Failure{Code: "TOOL_PANIC", Message: failureMessage(retErr)}}
			} else {
				result.Kind = OutcomeFailure
				result.Canonical = nil
				result.ModelFacing = nil
				result.UI = nil
				result.Meta = nil
				result.Content = nil
				result.AdditionalContexts = nil
				result.ConcludesTurn = false
				result.Code = "TOOL_PANIC"
				result.Failure = &Failure{Code: result.Code, Message: failureMessage(retErr)}
			}
		}
	}()
	preHooks, postHooks := r.snapshotHooks()
	policies, guards, postPolicies := r.snapshotPolicies()

	// Freeze arguments: never hand callers' mutable backing array to the
	// executor or the model.
	args := make([]byte, len(input))
	copy(args, input)

	if r.sem != nil {
		select {
		case r.sem <- struct{}{}:
			defer func() { <-r.sem }()
		case <-ctx.Done():
			return nil, fmt.Errorf("%w: %v", ErrAbortedBeforeDispatch, ctx.Err())
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
	execution := Execution{SessionID: ec.SessionID, RunID: ec.RunID, CallID: callID, Name: name, Arguments: append(json.RawMessage(nil), args...), Mutates: def.MutatesWorkspace}
	for _, policy := range policies {
		decision, reason, err := policy(ctx, execution)
		if err != nil {
			return nil, fmt.Errorf("tool policy: %w", err)
		}
		switch decision {
		case PolicyAllow:
		case PolicyDeny:
			return nil, fmt.Errorf("%w: %s", ErrPolicyDenied, reason)
		case PolicyAsk:
			approval := defaultApproval
			if approvalOverride != nil {
				approval = approvalOverride
			}
			if approval == nil {
				return nil, fmt.Errorf("%w: approval service unavailable", ErrPolicyDenied)
			}
			bindApprovalLease(approval, ec)
			approved, err := approval.Approve(ctx, ApprovalRequest{SessionID: ec.SessionID, RunID: ec.RunID, CallID: callID, ToolName: name, Arguments: append(json.RawMessage(nil), args...), Reason: reason})
			if err != nil {
				return nil, fmt.Errorf("tool approval: %w", err)
			}
			if !approved {
				return nil, ErrApprovalDenied
			}
		default:
			return nil, fmt.Errorf("%w: unknown policy decision", ErrPolicyDenied)
		}
	}
	for _, guard := range guards {
		reason, err := guard(execution)
		if err != nil {
			return nil, fmt.Errorf("tool guard: %w", err)
		}
		if reason != "" {
			return nil, fmt.Errorf("%w: %s", ErrPolicyDenied, reason)
		}
	}
	sandbox := ec.Sandbox
	if sandbox == nil {
		sandbox = r.sandbox
	}
	if sandbox != nil {
		if err := sandbox.AllowTool(ctx, ec.SessionID, name, def.MutatesWorkspace); err != nil {
			return nil, err
		}
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
	if err := validateCanonicalOutput(def, canonical); err != nil {
		return nil, err
	}

	for _, hook := range postHooks {
		if err := hook(ctx, name, args); err != nil {
			return nil, err
		}
	}

	result = &Result{Name: name, CallID: callID, Kind: OutcomeSuccess, Canonical: canonical, ModelFacing: canonical}
	if def.RenderModel != nil {
		modelFacing, err := def.RenderModel(canonical)
		if err != nil {
			return nil, err
		}
		result.ModelFacing = modelFacing
	}
	result.Content = renderedContent(result.ModelFacing)
	if def.PresentUI != nil {
		ui, err := def.PresentUI(canonical)
		if err != nil {
			return nil, err
		}
		result.UI = ui
		result.Meta = ui
	}
	for _, policy := range postPolicies {
		decision, replacement, reason, err := policy(ctx, execution, result)
		if err != nil {
			return nil, fmt.Errorf("tool post-policy: %w", err)
		}
		switch decision {
		case PolicyAllow:
		case PolicyDeny:
			return nil, fmt.Errorf("%w: %s", ErrPolicyDenied, reason)
		case PolicyAsk:
			approval := defaultApproval
			if approvalOverride != nil {
				approval = approvalOverride
			}
			if approval == nil {
				return nil, fmt.Errorf("%w: post-policy approval unavailable", ErrPolicyDenied)
			}
			bindApprovalLease(approval, ec)
			approved, approveErr := approval.Approve(ctx, ApprovalRequest{SessionID: ec.SessionID, RunID: ec.RunID, CallID: callID, ToolName: name, Arguments: append(json.RawMessage(nil), args...), Reason: reason})
			if approveErr != nil {
				return nil, fmt.Errorf("tool post-policy approval: %w", approveErr)
			}
			if !approved {
				return nil, ErrApprovalDenied
			}
		case PolicyDecision(0):
			return nil, fmt.Errorf("%w: unknown post-policy decision", ErrPolicyDenied)
		default:
			return nil, fmt.Errorf("%w: unknown post-policy decision", ErrPolicyDenied)
		}
		if replacement != nil {
			result.Canonical = replacement
			if err := validateCanonicalOutput(def, replacement); err != nil {
				return nil, err
			}
			result.ModelFacing = replacement
			if def.RenderModel != nil {
				result.ModelFacing, err = def.RenderModel(replacement)
				if err != nil {
					return nil, err
				}
			}
			if def.PresentUI != nil {
				result.UI, err = def.PresentUI(replacement)
				if err != nil {
					return nil, err
				}
			} else {
				result.UI = nil
			}
			result.Meta = result.UI
			result.Content = renderedContent(result.ModelFacing)
		}
	}
	return result, nil
}

func renderedContent(modelFacing any) []session.ContentBlock {
	text, ok := modelFacing.(string)
	if !ok || text == "" {
		return nil
	}
	return []session.ContentBlock{session.TextBlock(text)}
}

func bindApprovalLease(approval ApprovalService, ec ExecContext) {
	if ec.Lease == nil {
		return
	}
	if setter, ok := approval.(interface{ SetLease(session.Lease) }); ok {
		setter.SetLease(*ec.Lease)
	}
}

func validateCanonicalOutput(def *Definition, canonical any) error {
	if len(def.OutputSchema) == 0 || string(def.OutputSchema) == "null" {
		return nil
	}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return fmt.Errorf("%w: canonical output is not JSON-serializable: %v", ErrInvalidOutput, err)
	}
	if err := ValidateInput(def.OutputSchema, encoded); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidOutput, err)
	}
	return nil
}

func cloneDefinition(def *Definition) *Definition {
	if def == nil {
		return nil
	}
	clone := *def
	clone.InputSchema = canonicalSchema(def.InputSchema)
	clone.OutputSchema = canonicalSchema(def.OutputSchema)
	if len(clone.InputSchema) == 0 {
		clone.InputSchema = json.RawMessage(`{"type":"object","additionalProperties":false}`)
	}
	if len(clone.OutputSchema) == 0 {
		clone.OutputSchema = json.RawMessage(`{}`)
	}
	return &clone
}

func canonicalSchema(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return append(json.RawMessage(nil), raw...)
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return append(json.RawMessage(nil), raw...)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return append(json.RawMessage(nil), raw...)
	}
	return encoded
}
