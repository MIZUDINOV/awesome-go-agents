package llm

import (
	"context"
	"errors"
	"testing"
)

type collectingErrorProvider struct {
	err error
}

func (p collectingErrorProvider) Name() string { return "collecting-error" }

func (p collectingErrorProvider) Generate(ctx context.Context, _ *Request, cb StreamCallback) (*Response, error) {
	if cb != nil {
		if err := cb(ctx, StreamEvent{Type: StreamEventText, Text: "partial"}); err != nil {
			return nil, err
		}
	}
	return nil, p.err
}

func (p collectingErrorProvider) Capabilities(context.Context, string) (Capabilities, error) {
	return Capabilities{}, nil
}

func TestCollectStreamDeliversProviderError(t *testing.T) {
	want := errors.New("provider failed")
	stream, err := CollectStream(context.Background(), collectingErrorProvider{err: want}, &Request{Model: "m"})
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}
	var gotPartial, gotError bool
	for event := range stream {
		if event.Type == StreamEventText && event.Text == "partial" {
			gotPartial = true
		}
		if event.Err != nil {
			gotError = errors.Is(event.Err, want)
		}
	}
	if !gotPartial || !gotError {
		t.Fatalf("partial=%v providerError=%v", gotPartial, gotError)
	}
}

func TestCollectStreamRejectsNilProvider(t *testing.T) {
	if _, err := CollectStream(context.Background(), nil, &Request{}); err == nil {
		t.Fatal("nil provider was accepted")
	}
}
