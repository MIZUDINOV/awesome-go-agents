package tools

import (
	"context"
	"errors"
)

// BeforeExecuteDecision controls whether the registered definition body starts.
// The zero value is invalid so a forgotten decision fails closed.
type BeforeExecuteDecision uint8

const (
	BeforeExecuteContinue BeforeExecuteDecision = iota + 1
	BeforeExecuteShortCircuit
)

// BeforeExecuteHook runs after admission and before OnDispatch or the
// definition executor. It is intentionally explicit: a non-nil Result is not
// implicitly treated as a short-circuit.
type BeforeExecuteHook func(context.Context, ToolRunContext, Execution) (BeforeExecuteDecision, *Result, error)

var ErrInvalidBeforeExecuteResult = errors.New("invalid before-execute result")

func validateBeforeExecuteResult(result *Result, name, callID string) error {
	if result == nil || result.Name != name || result.CallID != callID {
		return ErrInvalidBeforeExecuteResult
	}
	if result.Kind != OutcomeSuccess || result.Continuation != ToolDeferred || result.ResumeKey == "" {
		return ErrInvalidBeforeExecuteResult
	}
	if result.Canonical != nil || result.Failure != nil || result.ConcludesTurn {
		return ErrInvalidBeforeExecuteResult
	}
	return nil
}
