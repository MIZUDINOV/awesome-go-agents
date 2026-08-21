package tools

import (
	"context"
	"encoding/json"
	"errors"
)

// PolicyDecision is the pre-execution authority outcome.
type PolicyDecision int

const (
	PolicyAllow PolicyDecision = iota + 1
	PolicyDeny
	PolicyAsk
)

var (
	ErrPolicyDenied   = errors.New("tool policy denied execution")
	ErrApprovalDenied = errors.New("tool approval denied execution")
)

// Execution is the immutable view supplied to guards and policy.
type Execution struct {
	SessionID string          `json:"session_id"`
	RunID     string          `json:"run_id"`
	TurnID    string          `json:"turn_id"`
	StepID    string          `json:"step_id"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Mutates   bool            `json:"mutates"`
}

// Policy may allow, deny or ask for an approval. Ask fails closed when no
// ApprovalService is configured.
type Policy func(context.Context, Execution) (PolicyDecision, string, error)

// Guard is monotonic: once it denies a call, later stages cannot re-enable it.
type Guard func(Execution) (string, error)

// PostPolicy runs after model/UI rendering. A non-nil replacement becomes the
// new canonical value and is validated/rendered again before finalization.
type PostPolicy func(context.Context, Execution, *Result) (PolicyDecision, any, string, error)

// PostPolicyReplacement makes the replacement mode explicit while preserving
// the legacy any return type of PostPolicy. A content replacement keeps the
// canonical value and UI metadata; a value replacement is revalidated and
// rendered from the new canonical value.
type PostPolicyReplacement struct {
	ContentOnly bool
	Value       any
}

func ReplacePostPolicyValue(value any) PostPolicyReplacement {
	return PostPolicyReplacement{Value: value}
}

func ReplacePostPolicyContent(content any) PostPolicyReplacement {
	return PostPolicyReplacement{ContentOnly: true, Value: content}
}

// ApprovalRequest is provider-neutral. The host may render it in any UI.
type ApprovalRequest struct {
	SessionID string          `json:"session_id"`
	RunID     string          `json:"run_id"`
	TurnID    string          `json:"turn_id"`
	StepID    string          `json:"step_id"`
	CallID    string          `json:"call_id"`
	ToolName  string          `json:"tool_name"`
	Arguments json.RawMessage `json:"arguments"`
	Reason    string          `json:"reason"`
}

type ApprovalService interface {
	Approve(context.Context, ApprovalRequest) (bool, error)
}
