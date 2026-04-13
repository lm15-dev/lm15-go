package provider

import (
	"encoding/json"
	"fmt"
	"strings"

	lm15 "github.com/lm15-dev/lm15-go"
)

// AnthropicAdapter implements the Messages API.
type AnthropicAdapter struct {
	lm15.BaseAdapter
	APIKey     string
	BaseURL    string
	APIVersion string
}

// NewAnthropic creates an Anthropic adapter.
func NewAnthropic(apiKey string, transport lm15.Transport) *AnthropicAdapter {
	return &AnthropicAdapter{
		BaseAdapter: lm15.BaseAdapter{Provider: "anthropic", Tport: transport},
		APIKey:      apiKey,
		BaseURL:     "https://api.anthropic.com/v1",
		APIVersion:  "2023-06-01",
	}
}

func (a *AnthropicAdapter) Manifest() lm15.ProviderManifest {
	return lm15.ProviderManifest{
		Provider: "anthropic",
		Supports: lm15.EndpointSupport{Complete: true, Stream: true, Files: true, Batches: true},
		EnvKeys:  []string{"ANTHROPIC_API_KEY"},
	}
}

func (a *AnthropicAdapter) headers() map[string]string {
	return map[string]string{
		"x-api-key":         a.APIKey,
		"anthropic-version": a.APIVersion,
		"content-type":      "application/json",
	}
}

func isContextMsg(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "prompt is too long") || strings.Contains(m, "too many tokens") ||
		strings.Contains(m, "context window") || strings.Contains(m, "context length")
}

var anthropicErrorMap = map[string]string{
	"authentication_error":  "auth",
	"permission_error":      "auth",
	"billing_error":         "billing",
	"rate_limit_error":      "rate_limit",
	"request_too_large":     "invalid_request",
	"not_found_error":       "invalid_request",
	"invalid_request_error": "invalid_request",
	"api_error":             "server",
	"overloaded_error":      "server",
	"timeout_error":         "timeout",
}

func (a *AnthropicAdapter) NormalizeError(status int, body string) error {
	var data map[string]any
	if err := json.Unmarshal([]byte(body), &data); err == nil {
		errObj, _ := data["error"].(map[string]any)
		if errObj != nil {
			msg := toString(errObj["message"])
			errType := toString(errObj["type"])

			if errType == "invalid_request_error" && isContextMsg(msg) {
				return &lm15.ContextLengthError{lm15.InvalidRequestError{lm15.ProviderError{lm15.ULMError{msg}}}}
			}
			if code, ok := anthropicErrorMap[errType]; ok {
				return lm15.ErrorForCode(code, msg)
			}
			if errType != "" && !strings.Contains(msg, errType) {
				msg = fmt.Sprintf("%s (%s)", msg, errType)
			}
			return lm15.MapHTTPError(status, msg)
		}
	}
	return lm15.MapHTTPError(status, truncate(body, 200))
}

func (a *AnthropicAdapter) partPayload(p lm15.Part) map[string]any {
	switch p.Type {
	case lm15.PartText:
		return map[string]any{"type": "text", "text": p.Text}
	case lm15.PartImage:
		if p.Source != nil {
			return map[string]any{"type": "image", "source": dsToAnthropicSource(*p.Source)}
		}
	case lm15.PartDocument:
		if p.Source != nil {
			return map[string]any{"type": "document", "source": dsToAnthropicSource(*p.Source)}
		}
	case lm15.PartToolCall:
		return map[string]any{
			"type": "tool_use", "id": p.ID,
			"name": p.Name, "input": p.Input,
		}
	case lm15.PartToolResult:
		out := map[string]any{"type": "tool_result", "tool_use_id": p.ID}
		text := partsToText(p.Content)
		if text != "" {
			out["content"] = text
		}
		if p.IsError != nil && *p.IsError {
			out["is_error"] = true
		}
		return out
	}
	return map[string]any{"type": "text", "text": p.Text}
}

func dsToAnthropicSource(ds lm15.DataSource) map[string]any {
	switch ds.Type {
	case "url":
		return map[string]any{"type": "url", "url": ds.URL}
	case "file":
		return map[string]any{"type": "file", "file_id": ds.FileID}
	default:
		return map[string]any{"type": "base64", "media_type": ds.MediaType, "data": ds.Data}
	}
}

func (a *AnthropicAdapter) buildPayload(request lm15.LMRequest, stream bool) map[string]any {
	var messages []map[string]any
	for _, m := range request.Messages {
		var content []map[string]any
		for _, p := range m.Parts {
			content = append(content, a.partPayload(p))
		}
		role := string(m.Role)
		if role == "tool" {
			role = "user"
		}
		messages = append(messages, map[string]any{"role": role, "content": content})
	}

	maxTokens := 1024
	if request.Config.MaxTokens != nil {
		maxTokens = *request.Config.MaxTokens
	}

	payload := map[string]any{
		"model": request.Model, "messages": messages,
		"stream": stream, "max_tokens": maxTokens,
	}

	if request.System != "" {
		payload["system"] = request.System
	}
	if request.Config.Temperature != nil {
		payload["temperature"] = *request.Config.Temperature
	}
	if len(request.Tools) > 0 {
		var tools []map[string]any
		for _, t := range request.Tools {
			if t.Type != "function" {
				continue
			}
			params := t.Parameters
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			tools = append(tools, map[string]any{"name": t.Name, "description": t.Description, "input_schema": params})
		}
		if len(tools) > 0 {
			payload["tools"] = tools
		}
	}
	if request.Config.Reasoning != nil {
		if enabled, ok := request.Config.Reasoning["enabled"].(bool); ok && enabled {
			budget := 1024
			if b, ok := request.Config.Reasoning["budget"]; ok {
				budget = toInt(b)
			}
			payload["thinking"] = map[string]any{"type": "enabled", "budget_tokens": budget}
		}
	}
	if request.Config.Provider != nil {
		for k, v := range request.Config.Provider {
			if k == "prompt_caching" {
				continue
			}
			payload[k] = v
		}
	}
	return payload
}

func (a *AnthropicAdapter) BuildRequest(request lm15.LMRequest, stream bool) lm15.HTTPRequest {
	body, _ := json.Marshal(a.buildPayload(request, stream))
	timeout := 60_000
	if stream {
		timeout = 120_000
	}
	return lm15.HTTPRequest{
		Method: "POST", URL: a.BaseURL + "/messages",
		Headers: a.headers(), Body: body, Timeout: durationMs(timeout),
	}
}

func (a *AnthropicAdapter) ParseResponse(request lm15.LMRequest, response lm15.HTTPResponse) (lm15.LMResponse, error) {
	var data map[string]any
	if err := json.Unmarshal(response.Body, &data); err != nil {
		return lm15.LMResponse{}, err
	}

	var parts []lm15.Part
	for _, block := range toSlice(data["content"]) {
		b, _ := block.(map[string]any)
		if b == nil {
			continue
		}
		switch b["type"] {
		case "text":
			parts = append(parts, lm15.TextPart(toString(b["text"])))
		case "tool_use":
			input, _ := b["input"].(map[string]any)
			if input == nil {
				input = map[string]any{}
			}
			parts = append(parts, lm15.ToolCallPart(toString(b["id"]), toString(b["name"]), input))
		case "thinking":
			parts = append(parts, lm15.ThinkingPart(toString(b["thinking"])))
		case "redacted_thinking":
			p := lm15.ThinkingPart("[redacted]")
			t := true
			p.Redacted = &t
			parts = append(parts, p)
		}
	}

	if len(parts) == 0 {
		parts = append(parts, lm15.TextPart(""))
	}

	finish := lm15.FinishStop
	for _, p := range parts {
		if p.Type == lm15.PartToolCall {
			finish = lm15.FinishToolCall
			break
		}
	}

	u, _ := data["usage"].(map[string]any)
	usage := lm15.Usage{
		InputTokens:  toInt(u["input_tokens"]),
		OutputTokens: toInt(u["output_tokens"]),
		TotalTokens:  toInt(u["input_tokens"]) + toInt(u["output_tokens"]),
	}
	if v, ok := u["cache_read_input_tokens"]; ok {
		i := toInt(v)
		usage.CacheReadTokens = &i
	}
	if v, ok := u["cache_creation_input_tokens"]; ok {
		i := toInt(v)
		usage.CacheWriteTokens = &i
	}

	return lm15.LMResponse{
		ID: toString(data["id"]), Model: orDefault(toString(data["model"]), request.Model),
		Message: lm15.Message{Role: lm15.RoleAssistant, Parts: parts},
		FinishReason: finish, Usage: usage, Provider: data,
	}, nil
}

func (a *AnthropicAdapter) ParseStreamEvent(request lm15.LMRequest, raw lm15.SSEEvent) (*lm15.StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	var p map[string]any
	if err := json.Unmarshal([]byte(raw.Data), &p); err != nil {
		return nil, nil
	}

	et := toString(p["type"])

	switch et {
	case "message_start":
		msg, _ := p["message"].(map[string]any)
		return &lm15.StreamEvent{Type: "start", ID: toString(msg["id"]), Model: toString(msg["model"])}, nil

	case "content_block_start":
		block, _ := p["content_block"].(map[string]any)
		if block != nil && block["type"] == "tool_use" {
			idx := toInt(p["index"])
			return &lm15.StreamEvent{
				Type: "delta", PartIndex: &idx,
				DeltaRaw: map[string]any{
					"type": "tool_call", "id": block["id"],
					"name": block["name"], "input": "",
				},
			}, nil
		}
		idx := toInt(p["index"])
		pt := toString(block["type"])
		return &lm15.StreamEvent{Type: "part_start", PartIndex: &idx, PartType: pt}, nil

	case "content_block_delta":
		delta, _ := p["delta"].(map[string]any)
		idx := toInt(p["index"])
		switch toString(delta["type"]) {
		case "text_delta":
			return &lm15.StreamEvent{Type: "delta", PartIndex: &idx, Delta: &lm15.PartDelta{Type: "text", Text: toString(delta["text"])}}, nil
		case "input_json_delta":
			return &lm15.StreamEvent{Type: "delta", PartIndex: &idx, DeltaRaw: map[string]any{"type": "tool_call", "input": toString(delta["partial_json"])}}, nil
		case "thinking_delta":
			return &lm15.StreamEvent{Type: "delta", PartIndex: &idx, Delta: &lm15.PartDelta{Type: "thinking", Text: toString(delta["thinking"])}}, nil
		}

	case "content_block_stop":
		idx := toInt(p["index"])
		return &lm15.StreamEvent{Type: "part_end", PartIndex: &idx}, nil

	case "message_stop":
		return &lm15.StreamEvent{Type: "end", FinishReason: lm15.FinishStop}, nil

	case "error":
		e, _ := p["error"].(map[string]any)
		code := toString(e["type"])
		msg := toString(e["message"])
		canonCode := "provider"
		if c, ok := anthropicErrorMap[code]; ok {
			canonCode = c
		}
		return &lm15.StreamEvent{Type: "error", Error: &lm15.ErrorInfo{Code: canonCode, Message: msg, ProviderCode: code}}, nil
	}

	return nil, nil
}

func (a *AnthropicAdapter) FileUpload(request lm15.FileUploadRequest) (lm15.FileUploadResponse, error) {
	req := lm15.HTTPRequest{
		Method: "POST", URL: a.BaseURL + "/files",
		Headers: map[string]string{
			"x-api-key": a.APIKey, "anthropic-version": a.APIVersion,
			"content-type": request.MediaType, "x-filename": request.Filename,
		},
		Body: request.BytesData, Timeout: durationMs(120_000),
	}
	resp, err := a.Tport.Request(req)
	if err != nil {
		return lm15.FileUploadResponse{}, err
	}
	if resp.Status >= 400 {
		return lm15.FileUploadResponse{}, a.NormalizeError(resp.Status, resp.Text())
	}
	var data map[string]any
	json.Unmarshal(resp.Body, &data)
	fileID := toString(data["id"])
	if fileID == "" {
		if f, ok := data["file"].(map[string]any); ok {
			fileID = toString(f["id"])
		}
	}
	return lm15.FileUploadResponse{ID: fileID, Provider: data}, nil
}
