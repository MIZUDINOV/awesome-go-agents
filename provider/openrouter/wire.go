package openrouter

import (
	"encoding/json"
	"fmt"

	"github.com/MIZUDINOV/awesome-go-agents/llm"
)

// Wire types for the OpenRouter chat.completions payload.

type chatMessage struct {
	Role             string          `json:"role"`
	Content          any             `json:"content,omitempty"`
	ToolCalls        []chatToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	Reasoning        string          `json:"reasoning,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
}

type chatContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
	File     *chatFile     `json:"file,omitempty"`
}

type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type chatFile struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
}

type chatToolCall struct {
	Index    int              `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function chatToolFunction `json:"function"`
}

type chatToolFunction struct {
	Name        string `json:"name,omitempty"`
	Arguments   string `json:"arguments,omitempty"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      bool   `json:"strict,omitempty"`
}

type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

type chatUsage struct {
	PromptTokens        int64    `json:"prompt_tokens"`
	CompletionTokens    int64    `json:"completion_tokens"`
	TotalTokens         int64    `json:"total_tokens"`
	Cost                *float64 `json:"cost,omitempty"`
	PromptTokensDetails struct {
		CachedTokens     int64 `json:"cached_tokens"`
		CacheWriteTokens int64 `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
}

type errorEnvelope struct {
	Code     json.RawMessage `json:"code,omitempty"`
	Message  string          `json:"message,omitempty"`
	Metadata struct {
		ErrorType    string `json:"error_type,omitempty"`
		ProviderCode string `json:"provider_code,omitempty"`
	} `json:"metadata,omitempty"`
}

type chatChoice struct {
	Index              int            `json:"index"`
	Delta              chatDelta      `json:"delta"`
	Message            chatDelta      `json:"message"`
	FinishReason       string         `json:"finish_reason,omitempty"`
	NativeFinishReason string         `json:"native_finish_reason,omitempty"`
	Error              *errorEnvelope `json:"error,omitempty"`
}

type chatDelta struct {
	Role             string          `json:"role,omitempty"`
	Content          *string         `json:"content,omitempty"`
	ToolCalls        []chatToolCall  `json:"tool_calls,omitempty"`
	Reasoning        *string         `json:"reasoning,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
	Annotations      json.RawMessage `json:"annotations,omitempty"`
}

type chatChunk struct {
	ID                 string          `json:"id,omitempty"`
	Model              string          `json:"model,omitempty"`
	Provider           string          `json:"provider,omitempty"`
	Choices            []chatChoice    `json:"choices,omitempty"`
	Usage              *chatUsage      `json:"usage,omitempty"`
	Error              *errorEnvelope  `json:"error,omitempty"`
	OpenRouterMetadata json.RawMessage `json:"openrouter_metadata,omitempty"`
}

// buildRequest converts an llm.Request into the OpenRouter JSON body.
func buildRequest(modelName string, req *llm.Request, config GenerateConfig, stream bool) (map[string]any, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	messages, err := encodeMessages(req)
	if err != nil {
		return nil, err
	}
	request := map[string]any{
		"model":    modelName,
		"messages": messages,
		"stream":   stream,
	}
	if stream {
		request["stream_options"] = map[string]any{"include_usage": true}
	}
	if len(config.Models) > 0 {
		request["models"] = config.Models
	}
	maxTokens := config.MaxTokens
	if maxTokens == 0 && req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		request["max_completion_tokens"] = maxTokens
	}
	if config.Provider != nil {
		request["provider"] = config.Provider
	}
	if config.Temperature != nil {
		request["temperature"] = *config.Temperature
	}
	if config.TopP != nil {
		request["top_p"] = *config.TopP
	}
	if config.Seed != nil {
		request["seed"] = *config.Seed
	}
	if len(config.Stop) > 0 {
		request["stop"] = config.Stop
	}
	if config.Reasoning != nil {
		request["reasoning"] = config.Reasoning
	}
	if config.SessionID != "" {
		request["session_id"] = config.SessionID
	}
	if len(config.Metadata) > 0 {
		request["metadata"] = config.Metadata
	}
	if config.ServiceTier != "" {
		request["service_tier"] = config.ServiceTier
	}
	plugins := make([]any, 0, 2)
	if config.PDF != nil {
		engine := config.PDF.Engine
		if engine == "" {
			engine = "cloudflare-ai"
		}
		plugins = append(plugins, map[string]any{"id": "file-parser", "pdf": map[string]any{"engine": engine}})
	}
	if config.ContextCompression {
		plugins = append(plugins, map[string]any{"id": "context-compression", "enabled": true})
	}
	if len(plugins) > 0 {
		request["plugins"] = plugins
	}

	tools := make([]chatTool, 0, len(req.Tools))
	hasTools := false
	for _, tool := range req.Tools {
		if tool == nil {
			continue
		}
		hasTools = true
		tools = append(tools, chatTool{Type: "function", Function: chatToolFunction{
			Name: tool.Name, Description: tool.Description,
			Parameters: tool.InputSchema, Strict: strictTool(config.StrictToolNames, tool.Name) || tool.Strict,
		}})
	}
	if hasTools {
		request["tools"] = tools
		choice := string(req.ToolChoice)
		if choice == "" {
			choice = "auto"
		}
		request["tool_choice"] = choice
		// The durable scheduler forces sequential tool execution
		// (H-SCHED-001). llm.Request.ParallelToolCalls (the neutral signal)
		// wins over the opaque config.
		parallel := config.ParallelToolCalls
		if req.ParallelToolCalls != nil {
			parallel = req.ParallelToolCalls
		}
		if parallel != nil {
			request["parallel_tool_calls"] = *parallel
		}
	}
	if len(req.StructuredOutputSchema) > 0 {
		request["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{
			"name": "wzhooh_response", "strict": req.StructuredStrict, "schema": req.StructuredOutputSchema,
		}}
	}
	for key, value := range config.ExtraBody {
		request[key] = value
	}
	return request, nil
}

func strictTool(names []string, name string) bool {
	for _, candidate := range names {
		if candidate == name {
			return true
		}
	}
	return false
}

// encodeMessages converts llm.Messages into the OpenRouter wire shape.
func encodeMessages(req *llm.Request) ([]chatMessage, error) {
	messages := make([]chatMessage, 0, len(req.System)+len(req.Messages))
	for _, sm := range req.System {
		messages = append(messages, chatMessage{Role: "system", Content: sm.Text()})
	}
	for i := range req.Messages {
		message := &req.Messages[i]
		if message.Role == llm.RoleTool {
			for _, part := range message.Parts {
				if part.ToolResult == nil {
					continue
				}
				encoded, err := json.Marshal(part.ToolResult.Output)
				if err != nil {
					return nil, fmt.Errorf("encode tool response %s: %w", part.ToolResult.CallID, err)
				}
				s := string(encoded)
				messages = append(messages, chatMessage{Role: "tool", ToolCallID: part.ToolResult.CallID, Content: s})
			}
			continue
		}
		result := chatMessage{Role: roleName(message.Role)}
		content := make([]chatContentPart, 0, len(message.Parts))
		for _, part := range message.Parts {
			switch part.Type {
			case llm.PartText:
				content = append(content, chatContentPart{Type: "text", Text: part.Text})
			case llm.PartMedia:
				content = append(content, chatContentPart{Type: "image_url", ImageURL: &chatImageURL{URL: part.Media.URL, Detail: "high"}})
			case llm.PartToolCall:
				arguments := part.ToolCall.Arguments
				result.ToolCalls = append(result.ToolCalls, chatToolCall{ID: part.ToolCall.CallID, Type: "function", Function: chatToolFunction{Name: part.ToolCall.Name, Arguments: string(arguments)}})
			case llm.PartReasoning:
				// reasoning is carried in message metadata, not as content
			case llm.PartToolResult:
				// handled above for tool role; ignore nested results elsewhere
			}
			if part.Media != nil && part.Media.URL != "" && part.Type != llm.PartMedia {
				content = append(content, chatContentPart{Type: "image_url", ImageURL: &chatImageURL{URL: part.Media.URL}})
			}
		}
		if len(content) == 1 && content[0].Type == "text" {
			result.Content = content[0].Text
		} else if len(content) > 0 {
			result.Content = content
		} else if len(result.ToolCalls) > 0 {
			// OpenRouter continuation contract distinguishes absent from null.
			result.Content = json.RawMessage("null")
		}
		if private, ok := message.Metadata["openrouter"].(map[string]any); ok {
			result.ReasoningDetails = rawMetadata(private["reasoning_details"])
			if len(result.ReasoningDetails) == 0 || string(result.ReasoningDetails) == "null" || string(result.ReasoningDetails) == "[]" {
				result.Reasoning, _ = private["reasoning"].(string)
			}
		}
		messages = append(messages, result)
	}
	return messages, nil
}

func roleName(role llm.Role) string {
	switch role {
	case llm.RoleSystem:
		return "system"
	case llm.RoleAssistant:
		return "assistant"
	case llm.RoleTool:
		return "tool"
	default:
		return "user"
	}
}
