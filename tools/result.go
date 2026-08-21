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

func toolErrorCode(err error) string {
	if err == nil {
		return ""
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
