package tools

import (
	"context"
	"encoding/json"
	"errors"
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

// Hook is a pre-execution pipeline extension point. Hooks are snapshotted at
// dispatch time, so a hook slice can be mutated concurrently without racing a
// running pipeline.
type Hook func(ctx context.Context, name string, input json.RawMessage) error

// PostHook observes and may enrich the completed execution outcome.
type PostHook func(context.Context, Execution, *Result) error
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
	mu                 sync.RWMutex
	definitions        map[string]*Definition
	preExecute         []Hook
	beforeExecute      []BeforeExecuteHook
	postExecute        []PostHook
	defaultTimeout     time.Duration
	sem                chan struct{}
	sandbox            integration.Sandbox
	approval           ApprovalService
	maxParallel        int
	policies           []Policy
	policyIDs          []uint64
	guards             []Guard
	guardIDs           []uint64
	nextRegistrationID uint64
	postPolicies       []PostPolicy
	observers          []Observer
	presentation       PresentationMode
	codeRuntime        CodeRuntime
	codeLanguage       string
	codeSDKRenderer    func([]*llm.ToolDefinition) (string, error)
}

// New returns an empty Registry.
func New(opts Options) *Registry {
	if opts.DefaultTimeout <= 0 {
		opts.DefaultTimeout = 5 * time.Minute
	}
	if opts.MaxParallel < 0 {
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
	if opts.MaxParallel > 0 {
		r.sem = make(chan struct{}, opts.MaxParallel)
	}
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
	r.nextRegistrationID++
	r.policyIDs = append(r.policyIDs, r.nextRegistrationID)
}

// AddPolicyHandle appends a policy and returns an idempotent disposer.
func (r *Registry) AddPolicyHandle(policy Policy) *Registration {
	if policy == nil {
		return newRegistration(func() error { return nil })
	}
	r.mu.Lock()
	r.nextRegistrationID++
	id := r.nextRegistrationID
	r.policies = append(r.policies, policy)
	r.policyIDs = append(r.policyIDs, id)
	r.mu.Unlock()
	return newRegistration(func() error {
		r.mu.Lock()
		defer r.mu.Unlock()
		for i, candidateID := range r.policyIDs {
			if candidateID == id {
				r.policies = append(r.policies[:i], r.policies[i+1:]...)
				r.policyIDs = append(r.policyIDs[:i], r.policyIDs[i+1:]...)
				return nil
			}
		}
		return nil
	})
}

// AddGuard appends a monotonic guard evaluated after policy decisions.
func (r *Registry) AddGuard(guard Guard) {
	if guard == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.guards = append(r.guards, guard)
	r.nextRegistrationID++
	r.guardIDs = append(r.guardIDs, r.nextRegistrationID)
}

// AddGuardHandle appends a guard and returns an idempotent disposer.
func (r *Registry) AddGuardHandle(guard Guard) *Registration {
	if guard == nil {
		return newRegistration(func() error { return nil })
	}
	r.mu.Lock()
	r.nextRegistrationID++
	id := r.nextRegistrationID
	r.guards = append(r.guards, guard)
	r.guardIDs = append(r.guardIDs, id)
	r.mu.Unlock()
	return newRegistration(func() error {
		r.mu.Lock()
		defer r.mu.Unlock()
		for i, candidateID := range r.guardIDs {
			if candidateID == id {
				r.guards = append(r.guards[:i], r.guards[i+1:]...)
				r.guardIDs = append(r.guardIDs[:i], r.guardIDs[i+1:]...)
				return nil
			}
		}
		return nil
	})
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

// Registration is a reversible tool registration. Unregister is idempotent
// on the handle, while new executions observe the removal immediately.
type Registration struct {
	once       sync.Once
	unregister func() error
	err        error
}

func newRegistration(unregister func() error) *Registration {
	return &Registration{unregister: unregister}
}

func (r *Registration) Unregister() error {
	if r == nil {
		return ErrInvalidArguments
	}
	r.once.Do(func() { r.err = r.unregister() })
	return r.err
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

// RunBatch uses a rolling bounded pool. Exclusive calls are barriers: all
// already-started safe calls drain before the exclusive call starts. Calls
// that are not started after cancellation are never dispatched.
func (r *Registry) RunBatch(ctx context.Context, ec ToolRunContext, calls []Call) []Outcome {
	return runRollingBatch(ctx, r.maxParallel, calls,
		func(call Call) bool { return r.ExecutionMode(call.Name, call.Input) },
		func(call Call) (*Result, error) { return r.Run(ctx, ec, call.Name, call.CallID, call.Input) })
}

func runRollingBatch(ctx context.Context, limit int, calls []Call, executionMode func(Call) bool, run func(Call) (*Result, error)) []Outcome {
	outcomes := make([]Outcome, len(calls))
	for index, call := range calls {
		outcomes[index].Call = call
	}
	if len(calls) == 0 {
		return outcomes
	}
	type completed struct {
		index  int
		result *Result
		err    error
	}
	done := make(chan completed, len(calls))
	active, next := 0, 0
	terminal := false
	if limit <= 0 || limit > len(calls) {
		limit = len(calls)
	}
	launch := func(index int) {
		active++
		go func() {
			result, err := run(calls[index])
			done <- completed{index: index, result: result, err: err}
		}()
	}
	drainOne := func() completed {
		item := <-done
		outcomes[item.index].Result = item.result
		outcomes[item.index].Err = item.err
		active--
		return item
	}
	markUnstarted := func() {
		if ctx.Err() == nil {
			return
		}
		for next < len(calls) {
			outcomes[next].Err = fmt.Errorf("%w: %v", ErrAbortedBeforeDispatch, ctx.Err())
			next++
		}
	}
	markTerminal := func() {
		for next < len(calls) {
			outcomes[next].Err = fmt.Errorf("%w: a previous tool concluded the turn", ErrAbortedBeforeDispatch)
			next++
		}
	}
	isTerminal := func(result *Result) bool {
		return result != nil && (result.Continuation == ToolConclude || result.Continuation == ToolDeferred || result.ConcludesTurn)
	}
	for next < len(calls) || active > 0 {
		if ctx.Err() != nil {
			markUnstarted()
			for active > 0 {
				drainOne()
			}
			break
		}
		if terminal {
			markTerminal()
			for active > 0 {
				drainOne()
			}
			break
		}
		for next < len(calls) && active < limit && executionMode(calls[next]) {
			launch(next)
			next++
		}
		if next < len(calls) && !executionMode(calls[next]) {
			for active > 0 {
				item := drainOne()
				terminal = terminal || isTerminal(item.result)
			}
			if ctx.Err() != nil {
				markUnstarted()
				break
			}
			if terminal {
				continue
			}
			outcomes[next].Result, outcomes[next].Err = run(calls[next])
			terminal = isTerminal(outcomes[next].Result)
			next++
			continue
		}
		if active > 0 {
			item := drainOne()
			terminal = terminal || isTerminal(item.result)
		}
	}
	return outcomes
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
	if schemaAbsent(def.InputSchema) {
		return fmt.Errorf("%w: tool %q input schema is required", ErrInvalidArguments, def.Name)
	}
	if err := ValidateSchema(def.InputSchema); err != nil {
		return fmt.Errorf("%w: tool %q input schema: %v", ErrInvalidArguments, def.Name, err)
	}
	if err := ValidateObjectRoot(def.InputSchema); err != nil {
		return fmt.Errorf("%w: tool %q input schema: %v", ErrInvalidArguments, def.Name, err)
	}
	if len(def.typedInputSchema) > 0 && !schemasEquivalent(def.InputSchema, def.typedInputSchema) {
		return fmt.Errorf("%w: tool %q input schema does not match its typed input contract", ErrInvalidArguments, def.Name)
	}
	if schemaAbsent(def.OutputSchema) {
		return fmt.Errorf("%w: tool %q output schema is required", ErrInvalidArguments, def.Name)
	}
	if err := ValidateSchema(def.OutputSchema); err != nil {
		return fmt.Errorf("%w: tool %q output schema: %v", ErrInvalidArguments, def.Name, err)
	}
	if !schemasEquivalent(def.OutputSchema, AnyOutputSchema) {
		if err := ValidateObjectRoot(def.OutputSchema); err != nil {
			return fmt.Errorf("%w: tool %q output schema: %v", ErrInvalidArguments, def.Name, err)
		}
	}
	if len(def.typedOutputSchema) > 0 && !schemasEquivalent(def.OutputSchema, def.typedOutputSchema) {
		return fmt.Errorf("%w: tool %q output schema does not match its typed output contract", ErrInvalidArguments, def.Name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.definitions[def.Name]; exists {
		return ErrToolAlreadyExists
	}
	r.definitions[def.Name] = cloneDefinition(def)
	return nil
}

func (r *Registry) RegisterTool(def *Definition) (*Registration, error) {
	name := ""
	if def != nil {
		name = def.Name
	}
	if err := r.Register(def); err != nil {
		return nil, err
	}
	return newRegistration(func() error { return r.Unregister(name) }), nil
}

func (r *Registry) Unregister(name string) error {
	if name == "" {
		return ErrInvalidArguments
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.definitions[name]; !ok {
		return ErrToolNotFound
	}
	delete(r.definitions, name)
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
	if hook == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.preExecute = append(r.preExecute, hook)
}

// AddBeforeExecute appends an explicit pre-dispatch decision hook. Hooks are
// snapshotted per execution and may return a deferred result without invoking
// the definition body.
func (r *Registry) AddBeforeExecute(hook BeforeExecuteHook) {
	if hook == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.beforeExecute = append(r.beforeExecute, hook)
}

func (r *Registry) AddPostExecute(hook PostHook) {
	if hook == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.postExecute = append(r.postExecute, hook)
}

func (r *Registry) snapshotHooks() (pre []Hook, before []BeforeExecuteHook, post []PostHook) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	pre = append([]Hook(nil), r.preExecute...)
	before = append([]BeforeExecuteHook(nil), r.beforeExecute...)
	post = append([]PostHook(nil), r.postExecute...)
	return pre, before, post
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

// GuidanceSections returns tool-owned prompt sections in deterministic tool
// order. Descriptions remain in ModelTools; guidance never enters provider
// tool schemas.
func (r *Registry) GuidanceSections() []PromptSection {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.definitions))
	for name := range r.definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	sections := make([]PromptSection, 0)
	for _, name := range names {
		for _, section := range r.definitions[name].Guidance {
			if section.Title == "" || section.Content == "" {
				continue
			}
			sections = append(sections, section)
		}
	}
	return sections
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
		InputSchema:  runCodeToolDefinition().InputSchema,
		OutputSchema: AnyOutputSchema,
		Execute: func(ctx context.Context, ec ToolRunContext, input json.RawMessage) (any, error) {
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
func (r *Registry) Run(ctx context.Context, ec ToolRunContext, name, callID string, input []byte) (result *Result, retErr error) {
	return r.run(ctx, ec, name, callID, input, nil)
}

func (r *Registry) run(ctx context.Context, ec ToolRunContext, name, callID string, input []byte, approvalOverride ApprovalService) (result *Result, retErr error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrAbortedBeforeDispatch, err)
	}
	ec.CallID = callID
	if ec.RootCallID == "" {
		ec.RootCallID = callID
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
	parentCtx := ctx
	shortCircuited := false
	timeout := def.Timeout
	if timeout <= 0 {
		timeout = r.defaultTimeout
	}
	pipelineCtx, pipelineCancel := context.WithTimeout(parentCtx, timeout)
	ctx = pipelineCtx
	ec.Signal = pipelineCtx.Done()
	ec.control = &toolRunControl{continuation: ToolContinue}
	defer pipelineCancel()
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
			result.Continuation = ToolContinue
			result.ResumeKey = ""
			result.WaitingReason = ""
			result.ConcludesTurn = false
			result.Code = code
			result.Failure = failureFor(retErr)
		}
		if result == nil && retErr != nil {
			code := toolErrorCode(retErr)
			result = &Result{Name: name, CallID: callID, Kind: OutcomeFailure, Code: code, Failure: failureFor(retErr)}
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
	var args []byte
	defer func() {
		if shortCircuited || def.FinalizeContent == nil || result == nil || result.Kind != OutcomeSuccess {
			return
		}
		var blocks []session.ContentBlock
		var finalizeErr error
		func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					finalizeErr = fmt.Errorf("finalizer panic: %v", recovered)
				}
			}()
			blocks, finalizeErr = def.FinalizeContent(args, result.Canonical)
		}()
		if parentCtx.Err() == nil && errors.Is(pipelineCtx.Err(), context.DeadlineExceeded) {
			retErr = ErrToolTimeout
		}
		if finalizeErr != nil {
			retErr = fmt.Errorf("finalize tool content: %w", finalizeErr)
			result.Kind = OutcomeFailure
			result.Canonical = nil
			result.ModelFacing = nil
			result.UI = nil
			result.Meta = nil
			result.Content = nil
			result.Continuation = ToolContinue
			result.ResumeKey = ""
			result.WaitingReason = ""
			result.ConcludesTurn = false
			result.Code = "FINALIZE_FAILED"
			result.Failure = &Failure{Code: result.Code, Message: failureMessage(finalizeErr)}
		} else {
			result.Content = blocks
		}
		result = result.Freeze()
	}()
	defer func() {
		if parentCtx.Err() == nil && errors.Is(pipelineCtx.Err(), context.DeadlineExceeded) {
			retErr = ErrToolTimeout
		}
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
				result.Continuation = ToolContinue
				result.ResumeKey = ""
				result.WaitingReason = ""
				result.ConcludesTurn = false
				result.Code = "TOOL_PANIC"
				result.Failure = failureFor(retErr)
			}
		}
	}()
	preHooks, beforeHooks, postHooks := r.snapshotHooks()
	policies, guards, postPolicies := r.snapshotPolicies()
	dispatchStarted := false
	defer func() {
		if !dispatchStarted && retErr != nil && (errors.Is(retErr, context.Canceled) || errors.Is(retErr, context.DeadlineExceeded)) {
			retErr = fmt.Errorf("%w: %v", ErrAbortedBeforeDispatch, retErr)
		}
	}()

	// Freeze arguments: never hand callers' mutable backing array to the
	// executor or the model.
	args = make([]byte, len(input))
	copy(args, input)

	if r.sem != nil {
		select {
		case r.sem <- struct{}{}:
			defer func() { <-r.sem }()
		case <-ctx.Done():
			return nil, fmt.Errorf("%w: %v", ErrAbortedBeforeDispatch, ctx.Err())
		}
	}

	// Input validation before any hook or side effect (H-RUNTIME-001).
	if err := ValidateInput(def.InputSchema, args); err != nil {
		return nil, err
	}
	execution := Execution{SessionID: ec.SessionID, RunID: ec.RunID, TurnID: ec.TurnID, StepID: ec.StepID, RootCallID: ec.RootCallID, ParentCallID: ec.ParentCallID, CallID: callID, Name: name, Arguments: append(json.RawMessage(nil), args...), Mutates: def.MutatesWorkspace}
	var callPresentation map[string]any
	var err error
	if def.PresentCall != nil {
		callPresentation, err = def.PresentCall(append(json.RawMessage(nil), args...))
		if err != nil {
			return nil, fmt.Errorf("present tool call: %w", err)
		}
	}
	failAfterExecution := func(execErr error) (*Result, error) {
		code := toolErrorCode(execErr)
		result = &Result{Name: name, CallID: callID, CallPresentation: callPresentation, Kind: OutcomeFailure, Code: code, Failure: failureFor(execErr)}
		for _, hook := range postHooks {
			if hookErr := hook(ctx, execution, result); hookErr != nil {
				return result, fmt.Errorf("tool post-execute: %w", hookErr)
			}
		}
		return result, execErr
	}
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
			approved, err := approval.Approve(ctx, ApprovalRequest{SessionID: ec.SessionID, RunID: ec.RunID, TurnID: ec.TurnID, StepID: ec.StepID, CallID: callID, ToolName: name, Arguments: append(json.RawMessage(nil), args...), Reason: reason})
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
	for _, hook := range beforeHooks {
		decision, shortResult, err := hook(ctx, ec, execution)
		if err != nil {
			return nil, fmt.Errorf("tool before-execute: %w", err)
		}
		switch decision {
		case BeforeExecuteContinue:
			if shortResult != nil {
				return nil, ErrInvalidBeforeExecuteResult
			}
		case BeforeExecuteShortCircuit:
			if err := validateBeforeExecuteResult(shortResult, name, callID); err != nil {
				return nil, err
			}
			shortCircuited = true
			result = shortResult
			return result, nil
		default:
			return nil, ErrInvalidBeforeExecuteResult
		}
	}
	dispatchStarted = true
	if ec.OnDispatch != nil {
		if err := ec.OnDispatch(ctx, name, callID); err != nil {
			return nil, err
		}
	}

	canonical, err := def.Execute(ctx, ec, args)
	if err != nil {
		if pipelineCtx.Err() == context.DeadlineExceeded && parentCtx.Err() == nil {
			return failAfterExecution(ErrToolTimeout)
		}
		return failAfterExecution(err)
	}
	if pipelineCtx.Err() == context.DeadlineExceeded && parentCtx.Err() == nil {
		return failAfterExecution(ErrToolTimeout)
	}

	// Output validation before rendering/persistence (H-TOOLS-007).
	if err := validateCanonicalOutput(def, canonical); err != nil {
		return failAfterExecution(err)
	}

	result = &Result{Name: name, CallID: callID, CallPresentation: callPresentation, Kind: OutcomeSuccess, Canonical: canonical, ModelFacing: canonical, Continuation: ToolContinue}
	if def.RenderModel != nil {
		modelFacing, err := renderModel(def, args, canonical)
		if err != nil {
			return failAfterExecution(err)
		}
		result.ModelFacing = modelFacing
	}
	result.Content = renderedContent(result.ModelFacing)
	if def.RenderContent != nil {
		result.Content, err = def.RenderContent(append(json.RawMessage(nil), args...), canonical)
		if err != nil {
			return failAfterExecution(err)
		}
	}
	presentResult := def.PresentResult
	if presentResult != nil {
		ui, err := presentResult(append(json.RawMessage(nil), args...), canonical)
		if err != nil {
			return failAfterExecution(err)
		}
		result.UI = ui
		result.Meta = ui
	}
	for _, hook := range postHooks {
		if err := hook(ctx, execution, result); err != nil {
			return result, fmt.Errorf("tool post-execute: %w", err)
		}
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
			approved, approveErr := approval.Approve(ctx, ApprovalRequest{SessionID: ec.SessionID, RunID: ec.RunID, TurnID: ec.TurnID, StepID: ec.StepID, CallID: callID, ToolName: name, Arguments: append(json.RawMessage(nil), args...), Reason: reason})
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
			contentOnly := false
			if typed, ok := replacement.(PostPolicyReplacement); ok {
				contentOnly, replacement = typed.ContentOnly, typed.Value
			}
			if contentOnly {
				if err := applyContentReplacement(result, replacement); err != nil {
					return nil, err
				}
				continue
			}
			result.Canonical = replacement
			if err := validateCanonicalOutput(def, replacement); err != nil {
				return nil, err
			}
			result.ModelFacing = replacement
			if def.RenderModel != nil {
				result.ModelFacing, err = renderModel(def, args, replacement)
				if err != nil {
					return nil, err
				}
			}
			presentResult := def.PresentResult
			if presentResult != nil {
				result.UI, err = presentResult(append(json.RawMessage(nil), args...), replacement)
				if err != nil {
					return nil, err
				}
			} else {
				result.UI = nil
			}
			result.Meta = result.UI
			result.Content = renderedContent(result.ModelFacing)
			if def.RenderContent != nil {
				result.Content, err = def.RenderContent(append(json.RawMessage(nil), args...), replacement)
				if err != nil {
					return nil, err
				}
			}
		}
	}
	if def.ResolveContinuation != nil {
		continuation, resumeKey, waitingReason, resolveErr := def.ResolveContinuation(append(json.RawMessage(nil), args...), result.Canonical)
		if resolveErr != nil {
			return nil, fmt.Errorf("resolve tool continuation: %w", resolveErr)
		}
		result.Continuation = continuation
		result.ResumeKey = resumeKey
		result.WaitingReason = waitingReason
		result.ConcludesTurn = continuation == ToolConclude
	}
	if continuation, resumeKey, waitingReason, concludes := ec.directive(); concludes || continuation != ToolContinue {
		result.Continuation = continuation
		result.ResumeKey = resumeKey
		result.WaitingReason = waitingReason
		result.ConcludesTurn = concludes || continuation == ToolConclude
	}
	if pipelineCtx.Err() == context.DeadlineExceeded && parentCtx.Err() == nil {
		return nil, ErrToolTimeout
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

// snapshot returns a point-in-time runtime copy. Scopes use it only for a
// single local execution; inherited executions stay on the live root.
func (r *Registry) snapshot() *Registry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	copy := New(Options{
		DefaultTimeout:  r.defaultTimeout,
		MaxParallel:     r.maxParallel,
		Sandbox:         r.sandbox,
		Approval:        r.approval,
		Presentation:    r.presentation,
		CodeRuntime:     r.codeRuntime,
		CodeLanguage:    r.codeLanguage,
		CodeSDKRenderer: r.codeSDKRenderer,
	})
	for name, definition := range r.definitions {
		copy.definitions[name] = cloneDefinition(definition)
	}
	copy.preExecute = append([]Hook(nil), r.preExecute...)
	copy.beforeExecute = append([]BeforeExecuteHook(nil), r.beforeExecute...)
	copy.postExecute = append([]PostHook(nil), r.postExecute...)
	copy.policies = append([]Policy(nil), r.policies...)
	copy.policyIDs = append([]uint64(nil), r.policyIDs...)
	copy.guards = append([]Guard(nil), r.guards...)
	copy.guardIDs = append([]uint64(nil), r.guardIDs...)
	copy.nextRegistrationID = r.nextRegistrationID
	copy.postPolicies = append([]PostPolicy(nil), r.postPolicies...)
	copy.observers = append([]Observer(nil), r.observers...)
	return copy
}

func renderModel(def *Definition, args json.RawMessage, canonical any) (any, error) {
	if def.RenderModel != nil {
		return def.RenderModel(append(json.RawMessage(nil), args...), canonical)
	}
	return canonical, nil
}

func applyContentReplacement(result *Result, content any) error {
	result.ModelFacing = content
	if blocks, ok := content.([]session.ContentBlock); ok {
		encoded, err := session.MarshalBlocks(blocks)
		if err != nil {
			return fmt.Errorf("post-policy content: %w", err)
		}
		decoded, err := session.UnmarshalBlocks(encoded)
		if err != nil {
			return fmt.Errorf("post-policy content: %w", err)
		}
		result.Content = decoded
		return nil
	}
	result.Content = renderedContent(content)
	return nil
}

func restoreFinalizerState(result, before *Result) {
	if result == nil || before == nil {
		return
	}
	modelFacing, content := result.ModelFacing, result.Content
	*result = *before
	result.ModelFacing = modelFacing
	result.Content = content
}

func bindApprovalLease(approval ApprovalService, ec ToolRunContext) {
	if ec.Lease == nil {
		return
	}
	if setter, ok := approval.(interface{ SetLease(session.Lease) }); ok {
		setter.SetLease(*ec.Lease)
	}
}

func validateCanonicalOutput(def *Definition, canonical any) error {
	if schemaAbsent(def.OutputSchema) {
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
	clone.Guidance = append([]PromptSection(nil), def.Guidance...)
	clone.InputSchema = canonicalSchema(def.InputSchema)
	clone.OutputSchema = canonicalSchema(def.OutputSchema)
	clone.typedInputSchema = canonicalSchema(def.typedInputSchema)
	clone.typedOutputSchema = canonicalSchema(def.typedOutputSchema)
	if len(clone.InputSchema) == 0 {
		clone.InputSchema = json.RawMessage(`{"type":"object","additionalProperties":false}`)
	}
	if len(clone.OutputSchema) == 0 {
		clone.OutputSchema = json.RawMessage(`{}`)
	}
	return &clone
}

func schemasEquivalent(left, right json.RawMessage) bool {
	return string(canonicalSchema(left)) == string(canonicalSchema(right))
}

func canonicalSchema(raw json.RawMessage) json.RawMessage {
	if schemaAbsent(raw) {
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
