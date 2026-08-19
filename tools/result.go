package tools

import "errors"

// Sentinel errors returned by the registry / pipeline.
var (
	ErrToolNotFound      = errors.New("tool not found")
	ErrToolAlreadyExists = errors.New("tool already registered")
	ErrToolDisabled      = errors.New("tool disabled")
	ErrInvalidArguments  = errors.New("invalid tool arguments")
	ErrInvalidOutput     = errors.New("invalid tool output")
	ErrToolTimeout       = errors.New("tool execution timed out")
)
