package tools

import (
	"context"
	"errors"

	"github.com/MIZUDINOV/awesome-go-agents/integration"
)

type OutcomeKind string

const (
	OutcomeSuccess OutcomeKind = "success"
	OutcomeFailure OutcomeKind = "failure"
)

// ToolContinuation controls what the AgentKit loop does after a tool result.
// Deferred is used when the execution world completes the call asynchronously.
type ToolContinuation string

const (
	ToolContinue ToolContinuation = "continue"
	ToolConclude ToolContinuation = "conclude"
	ToolDeferred ToolContinuation = "deferred"
)

// ToolExecutionResult is the explicit public name for Result.
type ToolExecutionResult = Result

// NewDeferredResult creates an explicit host-owned continuation without
// pretending that the registered tool body produced its canonical output.
// The registry validates the identity and lifecycle fields before returning it.
func NewDeferredResult(name, callID string, modelFacing any, ui map[string]any, resumeKey, waitingReason string) *Result {
	return &Result{
		Name:          name,
		CallID:        callID,
		Kind:          OutcomeSuccess,
		ModelFacing:   modelFacing,
		UI:            ui,
		Meta:          ui,
		Content:       renderedContent(modelFacing),
		Continuation:  ToolDeferred,
		ResumeKey:     resumeKey,
		WaitingReason: waitingReason,
		ConcludesTurn: false,
	}
}

// Sentinel errors returned by the registry / pipeline.
var (
	ErrToolNotFound      = errors.New("tool not found")
	ErrToolAlreadyExists = errors.New("tool already registered")
	ErrToolDisabled      = errors.New("tool disabled")
	ErrInvalidArguments  = errors.New("invalid tool arguments")
	ErrInvalidOutput     = errors.New("invalid tool output")
	ErrToolTimeout       = errors.New("tool execution timed out")
	// ErrAbortedBeforeDispatch means cancellation won before the executor body
	// started. It is a durable scheduler outcome, not an unknown side effect.
	ErrAbortedBeforeDispatch = errors.New("tool aborted before dispatch")
)

// Failure is the safe, structured portion of a failed execution. Canonical
// output is deliberately absent; callers may persist Code and Message while
// keeping provider/user payloads out of routine logs.
type Failure struct {
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Meta    map[string]any `json:"meta,omitempty"`
}

// FailureError lets a host preserve a safe, structured tool failure while
// keeping the executor free to return an ordinary error. The registry copies
// the metadata into the immutable Result.Failure value.
type FailureError struct {
	Code    string
	Message string
	Meta    map[string]any
}

func (e *FailureError) Error() string {
	if e == nil || e.Message == "" {
		return "tool failed"
	}
	return e.Message
}

// NewFailureError creates a host-facing structured failure error. Metadata is
// copied so the caller cannot mutate a result after returning it.
func NewFailureError(code, message string, meta map[string]any) error {
	copyMeta := make(map[string]any, len(meta))
	for key, value := range meta {
		copyMeta[key] = value
	}
	return &FailureError{Code: code, Message: message, Meta: copyMeta}
}

func failureFor(err error) *Failure {
	if structured := structuredFailure(err); structured != nil {
		meta := make(map[string]any, len(structured.Meta))
		for key, value := range structured.Meta {
			meta[key] = value
		}
		code := structured.Code
		if code == "" {
			code = toolErrorCode(err)
		}
		message := structured.Message
		if message == "" {
			message = failureMessage(err)
		}
		return &Failure{Code: code, Message: message, Meta: meta}
	}
	code := toolErrorCode(err)
	return &Failure{Code: code, Message: failureMessage(err)}
}

func structuredFailure(err error) *FailureError {
	if err == nil {
		return nil
	}
	var structured *FailureError
	if errors.As(err, &structured) {
		return structured
	}
	return nil
}

func toolErrorCode(err error) string {
	if err == nil {
		return ""
	}
	if structured := structuredFailure(err); structured != nil && structured.Code != "" {
		return structured.Code
	}
	var denial *integration.Denial
	if errors.As(err, &denial) && denial.Code != "" {
		return denial.Code
	}
	switch {
	case errors.Is(err, integration.ErrStaleVersion):
		return "FS_STALE_VERSION"
	case errors.Is(err, integration.ErrNotObserved):
		return "FS_NOT_OBSERVED"
	case errors.Is(err, integration.ErrAlreadyExists):
		return "FS_ALREADY_EXISTS"
	case errors.Is(err, integration.ErrAmbiguousMatch):
		return "FS_AMBIGUOUS_MATCH"
	case errors.Is(err, integration.ErrTargetOutsideRoot):
		return "SANDBOX_DENIED_PATH"
	case errors.Is(err, ErrInvalidArguments):
		return "INVALID_ARGS"
	case errors.Is(err, ErrInvalidOutput):
		return "INVALID_OUTPUT"
	case errors.Is(err, ErrToolTimeout):
		return "TIMEOUT"
	case errors.Is(err, ErrAbortedBeforeDispatch):
		return "ABORTED_BEFORE_DISPATCH"
	case errors.Is(err, ErrPolicyDenied):
		return "POLICY_DENIED"
	case errors.Is(err, ErrApprovalDenied):
		return "APPROVAL_DENIED"
	case errors.Is(err, ErrToolNotFound):
		return "UNKNOWN_TOOL"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return "ABORTED"
	default:
		return "TOOL_FAILED"
	}
}

func failureMessage(err error) string {
	if err == nil {
		return ""
	}
	var denial *integration.Denial
	if errors.As(err, &denial) && denial.Reason != "" {
		return denial.Reason
	}
	switch toolErrorCode(err) {
	case "FS_STALE_VERSION":
		return "The file changed after it was read. Read it again before editing."
	case "FS_NOT_OBSERVED":
		return "Read the file before editing it."
	case "FS_ALREADY_EXISTS":
		return "The target file already exists."
	case "FS_AMBIGUOUS_MATCH":
		return "The edit matched multiple locations; narrow the old text or use replace_all."
	case "SANDBOX_DENIED_PATH":
		return "The requested path is outside the sandbox."
	case "INVALID_ARGS":
		return "The tool arguments are invalid; correct them and retry."
	case "INVALID_OUTPUT":
		return "The tool returned an invalid result; retry with a narrower request."
	case "TIMEOUT":
		return "The tool timed out; narrow the request or retry."
	case "ABORTED_BEFORE_DISPATCH":
		return "The tool was cancelled before it started."
	case "ABORTED":
		return "The tool was cancelled before it completed."
	case "POLICY_DENIED":
		return "The tool call was denied by runtime policy."
	case "APPROVAL_DENIED":
		return "The tool call was not approved."
	case "UNKNOWN_TOOL":
		return "The requested tool is unavailable in this session."
	default:
		return "The tool failed; retry or choose another tool."
	}
}
