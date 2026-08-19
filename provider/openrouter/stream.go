package openrouter

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

type toolAccumulator struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

type accumulator struct {
	model              string
	id                 string
	provider           string
	requestID          string
	generationID       string
	finishReason       string
	text               strings.Builder
	reasoning          strings.Builder
	reasoningSummary   strings.Builder
	reasoningDetails   json.RawMessage
	annotations        json.RawMessage
	openRouterMetadata json.RawMessage
	usage              Usage
	tools              map[int]*toolAccumulator
	streamStarted      bool
	startedAt          time.Time
	firstSemanticAt    time.Time
	done               bool
}

func newAccumulator(model, requestID, generationID string) *accumulator {
	return &accumulator{model: model, requestID: requestID, generationID: generationID, tools: make(map[int]*toolAccumulator), startedAt: time.Now()}
}

func consumeSSE(ctx context.Context, body io.Reader, acc *accumulator, cb llm.StreamCallback) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64<<10), MaxSSEEventBytes+1)
	var data strings.Builder
	dispatch := func() error {
		if data.Len() == 0 {
			return nil
		}
		value := strings.TrimSuffix(data.String(), "\n")
		data.Reset()
		if value == "[DONE]" {
			acc.done = true
			return nil
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(value), &chunk); err != nil {
			return fmt.Errorf("decode OpenRouter SSE event: %w", err)
		}
		return acc.addChunk(ctx, chunk, cb)
	}
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return context.Cause(ctx)
		}
		line := scanner.Text()
		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
			if acc.done {
				return nil
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			value := strings.TrimPrefix(line, "data:")
			value = strings.TrimPrefix(value, " ")
			if data.Len()+len(value)+1 > MaxSSEEventBytes {
				return errors.New("OpenRouter SSE event exceeds 2 MiB")
			}
			data.WriteString(value)
			data.WriteByte('\n')
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read OpenRouter SSE stream: %w", err)
	}
	if err := dispatch(); err != nil {
		return err
	}
	if !acc.done {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (a *accumulator) addChunk(ctx context.Context, chunk chatChunk, cb llm.StreamCallback) error {
	if chunk.ID != "" {
		a.id = chunk.ID
	}
	if chunk.Model != "" {
		a.model = chunk.Model
	}
	if chunk.Provider != "" {
		a.provider = chunk.Provider
	}
	if len(chunk.OpenRouterMetadata) > 0 {
		a.openRouterMetadata = chunk.OpenRouterMetadata
	}
	if chunk.Usage != nil {
		a.usage = Usage{InputTokens: chunk.Usage.PromptTokens, OutputTokens: chunk.Usage.CompletionTokens, CachedTokens: chunk.Usage.PromptTokensDetails.CachedTokens, CacheWriteTokens: chunk.Usage.PromptTokensDetails.CacheWriteTokens, CostUSD: chunk.Usage.Cost}
	}
	if chunk.Error != nil {
		return a.providerError(chunk.Error)
	}
	for _, choice := range chunk.Choices {
		if choice.Error != nil {
			return a.providerError(choice.Error)
		}
		delta := choice.Delta
		if delta.Content == nil && choice.Message.Content != nil {
			delta = choice.Message
		}
		if delta.Content != nil && *delta.Content != "" {
			a.markStreamStarted()
			a.text.WriteString(*delta.Content)
			if cb != nil {
				if err := cb(ctx, llm.StreamEvent{Type: llm.StreamEventText, Text: *delta.Content}); err != nil {
					return err
				}
			}
		}
		// Raw reasoning is private; never emitted to public drafts, but it is a
		// semantic provider chunk for retry/cost classification.
		if delta.Reasoning != nil && *delta.Reasoning != "" {
			a.markStreamStarted()
			a.reasoning.WriteString(*delta.Reasoning)
		}
		if len(delta.ReasoningDetails) > 0 {
			a.reasoningDetails = joinRawArray(a.reasoningDetails, delta.ReasoningDetails)
			for _, summary := range reasoningSummaries(delta.ReasoningDetails) {
				a.markStreamStarted()
				a.reasoningSummary.WriteString(summary)
				if cb != nil {
					if err := cb(ctx, llm.StreamEvent{Type: llm.StreamEventReasoning, Reasoning: summary}); err != nil {
						return err
					}
				}
			}
		}
		if len(delta.Annotations) > 0 {
			a.annotations = joinRawArray(a.annotations, delta.Annotations)
		}
		for _, call := range delta.ToolCalls {
			state := a.tools[call.Index]
			if state == nil {
				state = &toolAccumulator{}
				a.tools[call.Index] = state
			}
			if call.ID != "" {
				state.ID = call.ID
			}
			if call.Function.Name != "" {
				state.Name = call.Function.Name
			}
			if call.Function.Arguments != "" {
				a.markStreamStarted()
				state.Arguments.WriteString(call.Function.Arguments)
			}
		}
		if choice.FinishReason != "" {
			a.finishReason = choice.FinishReason
		}
	}
	return nil
}

func (a *accumulator) markStreamStarted() {
	if !a.streamStarted {
		a.streamStarted = true
		a.firstSemanticAt = time.Now()
	}
}

func (a *accumulator) providerError(envelope *errorEnvelope) error {
	return &Error{Code: errorCode(envelope.Code), Type: envelope.Metadata.ErrorType, ProviderCode: envelope.Metadata.ProviderCode, Message: envelope.Message, RequestID: a.requestID, GenerationID: a.generationID, StreamStarted: a.streamStarted, RouterMetadata: routingMetadata(a.openRouterMetadata)}
}

// response assembles the final llm.Response from accumulated state and emits
// completed tool-call events.
func (a *accumulator) response(ctx context.Context, cb llm.StreamCallback, duration time.Duration, req *llm.Request) (*llm.Response, error) {
	parts := make([]llm.Part, 0, 2+len(a.tools))
	if a.text.Len() > 0 {
		parts = append(parts, llm.Part{Type: llm.PartText, Text: a.text.String()})
	}
	if a.reasoningSummary.Len() > 0 {
		parts = append(parts, llm.Part{Type: llm.PartReasoning, Reasoning: a.reasoningSummary.String()})
	}
	ordered := make([]*toolAccumulator, len(a.tools))
	for index, tool := range a.tools {
		ordered[index] = tool
	}
	for index := 0; index < len(ordered); index++ {
		tool := ordered[index]
		if tool == nil {
			continue
		}
		arguments := strings.TrimSpace(tool.Arguments.String())
		if arguments == "" {
			arguments = "{}"
		}
		if !json.Valid([]byte(arguments)) {
			return nil, fmt.Errorf("decode tool arguments for %s: invalid JSON", tool.Name)
		}
		if tool.ID == "" || tool.Name == "" {
			return nil, errors.New("OpenRouter returned an incomplete tool call")
		}
		call := llm.ToolCallRequest{CallID: tool.ID, Name: tool.Name, Arguments: json.RawMessage(arguments)}
		if cb != nil {
			if err := cb(ctx, llm.StreamEvent{Type: llm.StreamEventToolCall, ToolCall: &llm.ToolCallDelta{CallID: tool.ID, Name: tool.Name, Arguments: []byte(arguments)}}); err != nil {
				return nil, err
			}
		}
		parts = append(parts, llm.Part{Type: llm.PartToolCall, ToolCall: &call})
	}

	message := &llm.Message{Role: llm.RoleAssistant, Parts: parts}
	if a.reasoningSummary.Len() > 0 || a.reasoning.Len() > 0 {
		message.Metadata = map[string]any{"openrouter": map[string]any{"reasoning": a.reasoning.String(), "reasoning_details": a.reasoningDetails, "annotations": a.annotations}}
	}

	metadata := ResponseMetadata{ID: a.id, Model: a.model, Provider: a.provider, GenerationID: firstNonEmpty(a.generationID, a.id), RequestID: a.requestID, Reasoning: a.reasoning.String(), ReasoningDetails: a.reasoningDetails, Annotations: a.annotations, OpenRouterMetadata: a.openRouterMetadata, Usage: a.usage, AdapterVersion: AdapterVersion, StreamStarted: a.streamStarted, StreamDurationMS: duration.Milliseconds()}
	if !a.firstSemanticAt.IsZero() {
		metadata.TimeToFirstChunkMS = max(int64(0), a.firstSemanticAt.Sub(a.startedAt).Milliseconds())
	}

	var cost *float64
	if a.usage.CostUSD != nil {
		cost = a.usage.CostUSD
	}
	return &llm.Response{
		Message:            message,
		FinishReason:       mapFinishReason(a.finishReason),
		Usage:              &llm.Usage{InputTokens: a.usage.InputTokens, OutputTokens: a.usage.OutputTokens, CachedTokens: a.usage.CachedTokens, CacheWriteTokens: a.usage.CacheWriteTokens, CostUSD: cost},
		Model:              a.model,
		Provider:           a.provider,
		ProviderResponseID: metadata.GenerationID,
		RequestID:          a.requestID,
		Latency:            duration,
		Raw:                rawJSON(metadata),
	}, nil
}

func mapFinishReason(reason string) llm.FinishReason {
	switch reason {
	case "stop", "tool_calls":
		return llm.FinishReasonStop
	case "length":
		return llm.FinishReasonLength
	case "content_filter":
		return llm.FinishReasonContentFilter
	default:
		return llm.FinishReasonOther
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func joinRawArray(current json.RawMessage, additions json.RawMessage) json.RawMessage {
	if len(additions) == 0 || string(additions) == "null" {
		return current
	}
	var next []json.RawMessage
	if json.Unmarshal(additions, &next) != nil {
		return current
	}
	var all []json.RawMessage
	_ = json.Unmarshal(current, &all)
	all = append(all, next...)
	encoded, _ := json.Marshal(all)
	return encoded
}

func reasoningSummaries(raw json.RawMessage) []string {
	var entries []map[string]any
	if json.Unmarshal(raw, &entries) != nil {
		return nil
	}
	var summaries []string
	for _, entry := range entries {
		kind, _ := entry["type"].(string)
		if kind != "reasoning.summary" {
			continue
		}
		if text, ok := entry["summary"].(string); ok && strings.TrimSpace(text) != "" {
			summaries = append(summaries, text)
		}
		if text, ok := entry["text"].(string); ok && strings.TrimSpace(text) != "" {
			summaries = append(summaries, text)
		}
	}
	return summaries
}

func routingMetadata(raw json.RawMessage) RoutingMetadata {
	var document struct {
		Requested        string `json:"requested"`
		Strategy         string `json:"strategy"`
		Summary          string `json:"summary"`
		SelectedProvider string `json:"selected_provider"`
		SelectedModel    string `json:"selected_model"`
	}
	_ = json.Unmarshal(raw, &document)
	return RoutingMetadata{Requested: document.Requested, Strategy: document.Strategy, Summary: document.Summary, SelectedProvider: document.SelectedProvider, SelectedModel: document.SelectedModel}
}

func errorCode(raw json.RawMessage) int {
	var number int
	if json.Unmarshal(raw, &number) == nil {
		return number
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		fmt.Sscanf(text, "%d", &number)
	}
	return number
}

func decodeHTTPError(resp *http.Response, requestID, generationID string) *Error {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, MaxSSEEventBytes+1))
	var envelope struct {
		Error              *errorEnvelope  `json:"error"`
		OpenRouterMetadata json.RawMessage `json:"openrouter_metadata"`
	}
	_ = json.Unmarshal(data, &envelope)
	result := &Error{StatusCode: resp.StatusCode, RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After")), RequestID: requestID, GenerationID: generationID, RouterMetadata: routingMetadata(envelope.OpenRouterMetadata)}
	if envelope.Error != nil {
		result.Code = errorCode(envelope.Error.Code)
		result.Message = envelope.Error.Message
		result.Type = envelope.Error.Metadata.ErrorType
		result.ProviderCode = envelope.Error.Metadata.ProviderCode
	}
	if result.Message == "" {
		result.Message = http.StatusText(resp.StatusCode)
	}
	return result
}

func decodeNonStreaming(ctx context.Context, body io.Reader, acc *accumulator) error {
	decoder := json.NewDecoder(io.LimitReader(body, MaxSSEEventBytes+1))
	var chunk chatChunk
	if err := decoder.Decode(&chunk); err != nil {
		return fmt.Errorf("decode OpenRouter response: %w", err)
	}
	return acc.addChunk(ctx, chunk, nil)
}
