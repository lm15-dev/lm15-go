package provider

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	lm15 "github.com/lm15-dev/lm15-go"
)

// OpenAIAdapter implements the Responses API.
type OpenAIAdapter struct {
	lm15.BaseAdapter
	APIKey  string
	BaseURL string
}

// NewOpenAI creates an OpenAI adapter.
func NewOpenAI(apiKey string, transport lm15.Transport) *OpenAIAdapter {
	return &OpenAIAdapter{
		BaseAdapter: lm15.BaseAdapter{Provider: "openai", Tport: transport},
		APIKey:      apiKey,
		BaseURL:     "https://api.openai.com/v1",
	}
}

func (a *OpenAIAdapter) Manifest() lm15.ProviderManifest {
	return lm15.ProviderManifest{
		Provider: "openai",
		Supports: lm15.EndpointSupport{
			Complete: true, Stream: true, Live: true,
			Embeddings: true, Files: true, Batches: true,
			Images: true, Audio: true,
		},
		EnvKeys: []string{"OPENAI_API_KEY"},
	}
}

func (a *OpenAIAdapter) headers() map[string]string {
	return map[string]string{
		"Authorization": "Bearer " + a.APIKey,
		"Content-Type":  "application/json",
	}
}

func (a *OpenAIAdapter) payload(request lm15.LMRequest, stream bool) map[string]any {
	messages := make([]map[string]any, len(request.Messages))
	for i, m := range request.Messages {
		messages[i] = messageToOpenAIInput(m)
	}

	p := map[string]any{
		"model":  request.Model,
		"input":  messages,
		"stream": stream,
	}

	if request.System != "" {
		p["instructions"] = request.System
	}
	if request.Config.MaxTokens != nil {
		p["max_output_tokens"] = *request.Config.MaxTokens
	}
	if request.Config.Temperature != nil {
		p["temperature"] = *request.Config.Temperature
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
			tools = append(tools, map[string]any{
				"type":        "function",
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			})
		}
		if len(tools) > 0 {
			p["tools"] = tools
		}
	}
	if request.Config.Provider != nil {
		for k, v := range request.Config.Provider {
			if k == "prompt_caching" {
				continue
			}
			p[k] = v
		}
	}
	return p
}

func (a *OpenAIAdapter) BuildRequest(request lm15.LMRequest, stream bool) lm15.HTTPRequest {
	body, _ := json.Marshal(a.payload(request, stream))
	timeout := 60_000
	if stream {
		timeout = 120_000
	}
	return lm15.HTTPRequest{
		Method:  "POST",
		URL:     a.BaseURL + "/responses",
		Headers: a.headers(),
		Body:    body,
		Timeout: durationMs(timeout),
	}
}

func (a *OpenAIAdapter) NormalizeError(status int, body string) error {
	var data map[string]any
	if err := json.Unmarshal([]byte(body), &data); err == nil {
		errObj, _ := data["error"].(map[string]any)
		if errObj != nil {
			msg, _ := errObj["message"].(string)
			code, _ := errObj["code"].(string)
			errType, _ := errObj["type"].(string)

			switch {
			case code == "context_length_exceeded":
				return &lm15.ContextLengthError{lm15.InvalidRequestError{lm15.ProviderError{lm15.ULMError{msg}}}}
			case code == "insufficient_quota" || errType == "insufficient_quota":
				return &lm15.BillingError{lm15.ProviderError{lm15.ULMError{msg}}}
			case code == "invalid_api_key" || errType == "authentication_error":
				return &lm15.AuthError{lm15.ProviderError{lm15.ULMError{msg}}}
			case code == "rate_limit_exceeded" || errType == "rate_limit_error":
				return &lm15.RateLimitError{lm15.ProviderError{lm15.ULMError{msg}}}
			}
			if code != "" && !strings.Contains(msg, code) {
				msg = fmt.Sprintf("%s (%s)", msg, code)
			}
			return lm15.MapHTTPError(status, msg)
		}
	}
	return lm15.MapHTTPError(status, truncate(body, 200))
}

func (a *OpenAIAdapter) ParseResponse(request lm15.LMRequest, response lm15.HTTPResponse) (lm15.LMResponse, error) {
	var data map[string]any
	if err := json.Unmarshal(response.Body, &data); err != nil {
		return lm15.LMResponse{}, err
	}

	// In-band error
	if errObj, ok := data["error"].(map[string]any); ok {
		msg, _ := errObj["message"].(string)
		return lm15.LMResponse{}, &lm15.ServerError{lm15.ProviderError{lm15.ULMError{msg}}}
	}

	var parts []lm15.Part
	for _, item := range toSlice(data["output"]) {
		m, _ := item.(map[string]any)
		if m == nil {
			continue
		}
		if m["type"] == "message" {
			for _, c := range toSlice(m["content"]) {
				cm, _ := c.(map[string]any)
				if cm == nil {
					continue
				}
				switch cm["type"] {
				case "output_text", "text":
					parts = append(parts, lm15.TextPart(toString(cm["text"])))
				case "refusal":
					parts = append(parts, lm15.RefusalPart(toString(cm["refusal"])))
				}
			}
		} else if m["type"] == "function_call" {
			var args map[string]any
			if s, ok := m["arguments"].(string); ok && s != "" {
				json.Unmarshal([]byte(s), &args)
			}
			if args == nil {
				args = map[string]any{}
			}
			parts = append(parts, lm15.ToolCallPart(toString(m["call_id"]), toString(m["name"]), args))
		}
	}

	if len(parts) == 0 {
		parts = append(parts, lm15.TextPart(toString(data["output_text"])))
	}

	finish := lm15.FinishStop
	for _, p := range parts {
		if p.Type == lm15.PartToolCall {
			finish = lm15.FinishToolCall
			break
		}
	}

	usage := parseOpenAIUsage(data)

	return lm15.LMResponse{
		ID:           toString(data["id"]),
		Model:        orDefault(toString(data["model"]), request.Model),
		Message:      lm15.Message{Role: lm15.RoleAssistant, Parts: parts},
		FinishReason: finish,
		Usage:        usage,
		Provider:     data,
	}, nil
}

func (a *OpenAIAdapter) ParseStreamEvent(request lm15.LMRequest, raw lm15.SSEEvent) (*lm15.StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	if raw.Data == "[DONE]" {
		return &lm15.StreamEvent{Type: "end", FinishReason: lm15.FinishStop}, nil
	}

	var p map[string]any
	if err := json.Unmarshal([]byte(raw.Data), &p); err != nil {
		return nil, nil
	}

	et := toString(p["type"])

	switch et {
	case "response.created":
		resp, _ := p["response"].(map[string]any)
		return &lm15.StreamEvent{Type: "start", ID: toString(resp["id"]), Model: request.Model}, nil

	case "response.output_text.delta", "response.refusal.delta":
		return &lm15.StreamEvent{
			Type:      "delta",
			PartIndex: intPtr(0),
			Delta:     &lm15.PartDelta{Type: "text", Text: toString(p["delta"])},
		}, nil

	case "response.output_audio.delta":
		return &lm15.StreamEvent{
			Type:      "delta",
			PartIndex: intPtr(0),
			Delta:     &lm15.PartDelta{Type: "audio", Data: toString(p["delta"])},
		}, nil

	case "response.output_item.added":
		item, _ := p["item"].(map[string]any)
		if item != nil && item["type"] == "function_call" {
			idx := toInt(p["output_index"])
			return &lm15.StreamEvent{
				Type:      "delta",
				PartIndex: &idx,
				DeltaRaw: map[string]any{
					"type": "tool_call", "id": item["call_id"],
					"name": item["name"], "input": toString(item["arguments"]),
				},
			}, nil
		}

	case "response.function_call_arguments.delta":
		idx := toInt(p["output_index"])
		return &lm15.StreamEvent{
			Type:      "delta",
			PartIndex: &idx,
			DeltaRaw: map[string]any{
				"type": "tool_call", "id": p["call_id"],
				"name": p["name"], "input": toString(p["delta"]),
			},
		}, nil

	case "response.completed":
		resp, _ := p["response"].(map[string]any)
		usage := parseOpenAIUsage(resp)
		finish := lm15.FinishStop
		for _, item := range toSlice(resp["output"]) {
			m, _ := item.(map[string]any)
			if m != nil && m["type"] == "function_call" {
				finish = lm15.FinishToolCall
				break
			}
		}
		return &lm15.StreamEvent{Type: "end", FinishReason: finish, Usage: &usage}, nil

	case "response.error", "error":
		errObj, _ := p["error"].(map[string]any)
		code := "provider"
		msg := ""
		if errObj != nil {
			code = orDefault(toString(errObj["code"]), toString(errObj["type"]))
			msg = toString(errObj["message"])
		}
		return &lm15.StreamEvent{Type: "error", Error: &lm15.ErrorInfo{Code: code, Message: msg}}, nil
	}

	return nil, nil
}

func (a *OpenAIAdapter) Embeddings(request lm15.EmbeddingRequest) (lm15.EmbeddingResponse, error) {
	payload := map[string]any{"model": request.Model, "input": request.Inputs}
	body, _ := json.Marshal(payload)
	req := lm15.HTTPRequest{
		Method: "POST", URL: a.BaseURL + "/embeddings",
		Headers: a.headers(), Body: body, Timeout: durationMs(60_000),
	}
	resp, err := a.Tport.Request(req)
	if err != nil {
		return lm15.EmbeddingResponse{}, err
	}
	if resp.Status >= 400 {
		return lm15.EmbeddingResponse{}, a.NormalizeError(resp.Status, resp.Text())
	}
	var data map[string]any
	json.Unmarshal(resp.Body, &data)
	var vectors [][]float64
	for _, item := range toSlice(data["data"]) {
		m, _ := item.(map[string]any)
		if m == nil {
			continue
		}
		vec := toFloat64Slice(m["embedding"])
		vectors = append(vectors, vec)
	}
	return lm15.EmbeddingResponse{Model: request.Model, Vectors: vectors, Provider: data}, nil
}

func (a *OpenAIAdapter) FileUpload(request lm15.FileUploadRequest) (lm15.FileUploadResponse, error) {
	boundary := fmt.Sprintf("lm15-%d", time.Now().UnixNano())
	var body []byte
	body = append(body, []byte(fmt.Sprintf("--%s\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n", boundary))...)
	body = append(body, []byte(fmt.Sprintf("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n", boundary, request.Filename, request.MediaType))...)
	body = append(body, request.BytesData...)
	body = append(body, []byte(fmt.Sprintf("\r\n--%s--\r\n", boundary))...)

	req := lm15.HTTPRequest{
		Method: "POST", URL: a.BaseURL + "/files",
		Headers: map[string]string{
			"Authorization": "Bearer " + a.APIKey,
			"Content-Type":  "multipart/form-data; boundary=" + boundary,
		},
		Body: body, Timeout: durationMs(120_000),
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
	return lm15.FileUploadResponse{ID: toString(data["id"]), Provider: data}, nil
}

func (a *OpenAIAdapter) ImageGenerate(request lm15.ImageGenerationRequest) (lm15.ImageGenerationResponse, error) {
	payload := map[string]any{"model": request.Model, "prompt": request.Prompt}
	if request.Size != "" {
		payload["size"] = request.Size
	}
	if request.Provider != nil {
		for k, v := range request.Provider {
			payload[k] = v
		}
	}
	body, _ := json.Marshal(payload)
	req := lm15.HTTPRequest{
		Method: "POST", URL: a.BaseURL + "/images/generations",
		Headers: a.headers(), Body: body, Timeout: durationMs(120_000),
	}
	resp, err := a.Tport.Request(req)
	if err != nil {
		return lm15.ImageGenerationResponse{}, err
	}
	if resp.Status >= 400 {
		return lm15.ImageGenerationResponse{}, a.NormalizeError(resp.Status, resp.Text())
	}
	var data map[string]any
	json.Unmarshal(resp.Body, &data)
	var images []lm15.DataSource
	for _, d := range toSlice(data["data"]) {
		m, _ := d.(map[string]any)
		if m == nil {
			continue
		}
		if b64, ok := m["b64_json"].(string); ok && b64 != "" {
			images = append(images, lm15.DataSource{Type: "base64", MediaType: "image/png", Data: b64})
		} else if url, ok := m["url"].(string); ok && url != "" {
			images = append(images, lm15.DataSource{Type: "url", URL: url, MediaType: "image/png"})
		}
	}
	return lm15.ImageGenerationResponse{Images: images, Provider: data}, nil
}

func (a *OpenAIAdapter) AudioGenerate(request lm15.AudioGenerationRequest) (lm15.AudioGenerationResponse, error) {
	voice := request.Voice
	if voice == "" {
		voice = "alloy"
	}
	format := request.Format
	if format == "" {
		format = "wav"
	}
	payload := map[string]any{"model": request.Model, "input": request.Prompt, "voice": voice, "format": format}
	if request.Provider != nil {
		for k, v := range request.Provider {
			payload[k] = v
		}
	}
	body, _ := json.Marshal(payload)
	req := lm15.HTTPRequest{
		Method: "POST", URL: a.BaseURL + "/audio/speech",
		Headers: a.headers(), Body: body, Timeout: durationMs(120_000),
	}
	resp, err := a.Tport.Request(req)
	if err != nil {
		return lm15.AudioGenerationResponse{}, err
	}
	if resp.Status >= 400 {
		return lm15.AudioGenerationResponse{}, a.NormalizeError(resp.Status, resp.Text())
	}
	ctype := "audio/wav"
	if ct, ok := resp.Headers["Content-Type"]; ok {
		ctype = ct
	}
	b64 := base64.StdEncoding.EncodeToString(resp.Body)
	return lm15.AudioGenerationResponse{Audio: lm15.DataSource{Type: "base64", MediaType: ctype, Data: b64}, Provider: map[string]any{"content_type": ctype}}, nil
}

// ── Helpers ────────────────────────────────────────────────────────

func parseOpenAIUsage(data map[string]any) lm15.Usage {
	if data == nil {
		return lm15.Usage{}
	}
	u, _ := data["usage"].(map[string]any)
	if u == nil {
		return lm15.Usage{}
	}
	uIn, _ := u["input_tokens_details"].(map[string]any)
	uOut, _ := u["output_tokens_details"].(map[string]any)
	usage := lm15.Usage{
		InputTokens:  toInt(u["input_tokens"]),
		OutputTokens: toInt(u["output_tokens"]),
		TotalTokens:  toInt(u["total_tokens"]),
	}
	if uOut != nil {
		if v, ok := uOut["reasoning_tokens"]; ok {
			i := toInt(v)
			usage.ReasoningTokens = &i
		}
	}
	if uIn != nil {
		if v, ok := uIn["cached_tokens"]; ok {
			i := toInt(v)
			usage.CacheReadTokens = &i
		}
	}
	return usage
}

func messageToOpenAIInput(m lm15.Message) map[string]any {
	var content []map[string]any
	for _, p := range m.Parts {
		content = append(content, partToOpenAIInput(p))
	}
	return map[string]any{"role": string(m.Role), "content": content}
}

func partToOpenAIInput(p lm15.Part) map[string]any {
	switch p.Type {
	case lm15.PartText:
		return map[string]any{"type": "input_text", "text": p.Text}
	case lm15.PartImage:
		if p.Source != nil {
			if p.Source.Type == "url" {
				return map[string]any{"type": "input_image", "image_url": p.Source.URL}
			}
			if p.Source.Type == "base64" {
				return map[string]any{"type": "input_image", "image_url": fmt.Sprintf("data:%s;base64,%s", p.Source.MediaType, p.Source.Data)}
			}
		}
	case lm15.PartAudio:
		if p.Source != nil && p.Source.Type == "base64" {
			media := p.Source.MediaType
			if i := strings.LastIndex(media, "/"); i >= 0 {
				media = media[i+1:]
			}
			return map[string]any{"type": "input_audio", "audio": p.Source.Data, "format": media}
		}
	case lm15.PartDocument:
		if p.Source != nil {
			if p.Source.Type == "url" {
				return map[string]any{"type": "input_file", "file_url": p.Source.URL}
			}
			if p.Source.Type == "base64" {
				return map[string]any{"type": "input_file", "file_data": fmt.Sprintf("data:%s;base64,%s", p.Source.MediaType, p.Source.Data)}
			}
		}
	case lm15.PartToolResult:
		text := partsToText(p.Content)
		return map[string]any{"type": "input_text", "text": text}
	}
	return map[string]any{"type": "input_text", "text": p.Text}
}
