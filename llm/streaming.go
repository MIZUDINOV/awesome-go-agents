package llm

import (
	"context"
	"errors"
)

// StreamEventType classifies each streaming event.
type StreamEventType string

const (
	// StreamEventText carries a text delta.
	StreamEventText StreamEventType = "text"
	// StreamEventReasoning carries a reasoning delta.
	StreamEventReasoning StreamEventType = "reasoning"
	// StreamEventToolCall carries one accumulated tool call definition.
	StreamEventToolCall StreamEventType = "tool_call"
	StreamEventMedia    StreamEventType = "media"
	// StreamEventDone is the terminal event; it carries the assembled Response.
	StreamEventDone StreamEventType = "done"
)

// ToolCallDelta yields a completed tool call (arguments already assembled).
type ToolCallDelta struct {
	CallID    string
	Name      string
	Arguments []byte // raw JSON object
}

// StreamEvent is a single unit emitted during streaming.
type StreamEvent struct {
	Type      StreamEventType
	Text      string
	Reasoning string
	Media     *MediaContent
	ToolCall  *ToolCallDelta
	// Response is set only on StreamEventDone.
	Response *Response
	// Err is set on a terminal failure before Done.
	Err error
}

// StreamCallback is the push-style callback a Provider invokes per event.
type StreamCallback func(ctx context.Context, event StreamEvent) error

// CollectStream adapts a provider's push callback into a receive-only channel.
// The channel yields StreamEvents in order and closes after the final event.
// A terminal provider failure is delivered as an event with Err set; the
// channel then closes without a Done event.
func CollectStream(ctx context.Context, p Provider, req *Request) (<-chan StreamEvent, error) {
	if p == nil {
		return nil, errors.New("llm: provider is required")
	}
	out := make(chan StreamEvent)
	go func() {
		defer close(out)
		doneSent := false
		send := func(event StreamEvent) bool {
			select {
			case out <- event:
				return true
			case <-ctx.Done():
				return false
			}
		}
		response, err := p.Generate(ctx, req, func(_ context.Context, event StreamEvent) error {
			if event.Type == StreamEventDone {
				doneSent = true
			}
			if !send(event) {
				return ctx.Err()
			}
			return nil
		})
		if err != nil {
			_ = send(StreamEvent{Err: err})
			return
		}
		if response != nil && !doneSent {
			_ = send(StreamEvent{Type: StreamEventDone, Response: response})
		}
	}()
	return out, nil
}
