package llm

import (
	"errors"
	"strings"
)

// ErrorKind classifies provider failures so the agent loop can decide between
// retry, overflow recovery, or terminal failure.
type ErrorKind string

const (
	ErrorKindUnknown         ErrorKind = "unknown"
	ErrorKindNetwork         ErrorKind = "network"
	ErrorKindRateLimit       ErrorKind = "rate_limit"
	ErrorKindProvider        ErrorKind = "provider"
	ErrorKindAuth            ErrorKind = "auth"
	ErrorKindInvalidRequest  ErrorKind = "invalid_request"
	ErrorKindContextOverflow ErrorKind = "context_overflow"
	ErrorKindPrematureEOF    ErrorKind = "premature_eof"
)

// Error is the provider-neutral error returned by a Provider. Providers SHOULD
// populate Kind, Code and Retryable so the loop does not need provider-specific
// knowledge. StreamStarted distinguishes a failure partway through a stream
// (not safe to blind-retry a side-effecting turn).
type Error struct {
	Kind ErrorKind
	// Code is a stable, provider-specific machine-readable error code from the
	// provider's own taxonomy. It is the
	// canonical identifier for error mapping and does not change between
	// releases; leave empty when the provider has no distinct code.
	Code          string
	Message       string
	ProviderCode  string
	StatusCode    int
	Retryable     bool
	StreamStarted bool
	// Cause preserves the provider's typed error before wrapping.
	Cause error
}

func (e *Error) Error() string {
	var builder strings.Builder
	builder.WriteString("agentkit llm error: kind=")
	builder.WriteString(string(e.Kind))
	if e.Code != "" {
		builder.WriteString(" code=")
		builder.WriteString(e.Code)
	}
	if e.StatusCode != 0 {
		builder.WriteString(" status=")
		builder.WriteString(itoa(e.StatusCode))
	}
	if e.Message != "" {
		builder.WriteString(" ")
		builder.WriteString(e.Message)
	}
	return builder.String()
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var digits []byte
	for v > 0 {
		digits = append([]byte{byte('0' + v%10)}, digits...)
		v /= 10
	}
	return string(digits)
}

// Unwrap supports errors.As/Is against the Cause.
func (e *Error) Unwrap() error { return e.Cause }

// IsContextOverflow reports whether err indicates the model context window was
// exceeded. Used to trigger emergency overflow recovery.
func IsContextOverflow(err error) bool {
	var llmErr *Error
	if errors.As(err, &llmErr) && llmErr.Kind == ErrorKindContextOverflow {
		return true
	}
	message := strings.ToLower(errString(err))
	return (strings.Contains(message, "context") && (strings.Contains(message, "length") || strings.Contains(message, "token") || strings.Contains(message, "window"))) || strings.Contains(message, "prompt is too long")
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// IsRetryable reports whether an error is a transient, pre-stream failure that
// is safe to retry without risking duplicate side effects.
func IsRetryable(err error) bool {
	var llmErr *Error
	if errors.As(err, &llmErr) {
		return llmErr.Retryable && !llmErr.StreamStarted
	}
	// Context overflow is not a retryable transient failure.
	return false
}

// RetryableError is a convenience constructor for a retryable pre-stream error.
func RetryableError(kind ErrorKind, message string) *Error {
	return &Error{Kind: kind, Message: message, Retryable: true}
}
