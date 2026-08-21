package agent

import (
	"context"
	"errors"
	"time"
)

// StepDecision is the provider-neutral decision returned by a host run policy.
// The policy decides whether the loop may make another model request; the
// loop remains responsible for persisting the decision and ending the turn.
type StepDecision uint8

const (
	StepContinue StepDecision = iota
	StepStop
	StepWait
)

// StepSnapshot is a read-only view of the current durable run budget.
type StepSnapshot struct {
	SessionID     string
	RunID         string
	TurnID        string
	Model         string
	StepIndex     int
	ToolCallCount int
	TotalTokens   int64
	Elapsed       time.Duration
}

// RunPolicy is an optional host-owned policy seam. Implementations may enforce
// soft budgets, cost limits, or product-specific continuation rules without
// owning the AgentKit orchestration loop.
type RunPolicy interface {
	BeforeStep(context.Context, StepSnapshot) (StepDecision, error)
}

var (
	ErrPolicyStopped = errors.New("agent: run policy stopped the turn")
	ErrPolicyWaiting = errors.New("agent: run policy paused the turn")
)
