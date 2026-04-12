package provider

import (
	"encoding/json"
	"fmt"
	"strings"

	lm15 "github.com/lm15-dev/lm15-go"
)

// GeminiAdapter implements the GenerativeLanguage API.
type GeminiAdapter struct {
	lm15.BaseAdapter
	APIKey  string
	BaseURL string
}

// NewGemini creates a Gemini adapter.
func NewGemini(apiKey string, transport lm15.Transport) *GeminiAdapter {
	return &GeminiAdapter{
		BaseAdapter: lm15.BaseAdapter{Provider: "gemini", Tport: transport},
		APIKey:      apiKey,
		BaseURL:     "https://generativelanguage.googleapis.com/v1beta",
	}
}

func (a *GeminiAdapter) Manifest() lm15.ProviderManifest {
	return lm15.ProviderManifest{
		Provider: "gemini",
		Supports: lm15.EndpointSupport{
			Complete: true, Stream: true, Live: true,
			Embeddings: true, Files: true, Batches: true,
			Images: true, Audio: true,
		},
		EnvKeys: []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"},
	}
}

func (a *GeminiAdapter) modelPath(model string) string {
	if strings.HasPrefix(model, "models/") {
		return model
	}
	return "models/" + model
}

func (a *GeminiAdapter) authHeaders() map[string]string {
	return map[string]string{
		"x-goog-api-key": a.APIKey,
		"Content-Type":   "application/json",
	}
}

func isContextLengthMsg(msg string) bool {
	m := strings.ToLower(msg)
	return (strings.Contains(m, "token") && (strings.Contains(m, "limit") || strings.Contains(m, "exceed"))) ||
		strings.Contains(m, "too long") || strings.Contains(m, "context length")
}

var geminiErrorMap = map[string]string{
	"INVALID_ARGUMENT":    "invalid_request",
	"FAILED_PRECONDITION": "billing",
	"PERMISSION_DENIED":   "auth",
	"NOT_FOUND":           "invalid_request",
	"RESOURCE_EXHAUSTED":  "rate_limit",
	"INTERNAL":            "server",
	"UNAVAILABLE":         "server",
	"DEADLINE_EXCEEDED":   "timeout",
}

func (a *GeminiAdapter) NormalizeError(status int, body string) error {
	var data map[string]any
	if err := json.Unmarshal([]byte(body), &data); err == nil {
		errObj, _ := data["error"].(map[string]any)
		if errObj != nil {
			msg := toString(errObj["message"])
			errStatus := toString(errObj["status"])
			if isContextLengthMsg(msg) {
				return &lm15.ContextLengthError{lm15.InvalidRequestError{lm15.ProviderError{lm15.ULMError{msg}}}}
			}
			if code, ok := geminiErrorMap[errStatus]; ok {
				return lm15.ErrorForCode(code, msg)
			}
			if errStatus != "" && !strings.Contains(msg, errStatus) {
				msg = fmt.Sprintf("%s (%s)", msg, errStatus)
			}
			return lm15.MapHTTPError(status, msg)
		}
	}
	return lm15.MapHTTPError(status, truncate(body, 200))
}

func (a *GeminiAdapter) partPayload(p lm15.Part) map[string]any {
	switch p.Type {
	case lm15.PartText:
		return map[string]any{"text": p.Text}
	case lm15.PartImage, lm15.PartAudio, lm15.PartVideo, lm15.PartDocument:
		if p.Source != nil {
			mime := p.Source.MediaType
			if mime == "" {
				mime = "application/octet-stream"
			}
			if p.Source.Type == "url" {
				return map[string]any{"fileData": map[string]any{"mimeType": mime, "fileUri": p.Source.URL}}
			}
			if p.Source.Type == "base64" {
				return map[string]any{"inlineData": map[string]any{"mimeType": mime, "data": p.Source.Data}}
			}
			if p.Source.Type == "file" {
				return map[string]any{"fileData": map[string]any{"mimeType": mime, "fileUri": p.Source.FileID}}
			}
		}
	case lm15.PartToolResult:
		text := partsToText(p.Content)
		return map[string]any{
			"functionResponse": map[string]any{
				"name":     orDefault(p.Name, "tool"),
				"response": map[string]any{"result": map[string]any{"text": text}},
			},
		}
	}
	return map[string]any{"text": p.Text}
}

func (a *GeminiAdapter) buildPayload(request lm15.LMRequest) map[string]any {
	var contents []map[string]any
	for _, m := range request.Messages {
		role := "user"
		if m.Role == lm15.RoleAssistant {
			role = "model"
		}
		var parts []map[string]any
		for _, p := range m.Parts {
			parts = append(parts, a.partPayload(p))
		}
		contents = append(contents, map[string]any{"role": role, "parts": parts})
	}

	payload := map[string]any{"contents": contents}

	if request.System != "" {
		payload["systemInstruction"] = map[string]any{"parts": []map[string]any{{"text": request.System}}}
	}

	cfg := map[string]any{}
	if request.Config.Temperature != nil {
		cfg["temperature"] = *request.Config.Temperature
	}
	if request.Config.MaxTokens != nil {
		cfg["maxOutputTokens"] = *request.Config.MaxTokens
	}
	if len(request.Config.Stop) > 0 {
		cfg["stopSequences"] = request.Config.Stop
	}
	if len(cfg) > 0 {
		payload["generationConfig"] = cfg
	}

	if len(request.Tools) > 0 {
		var decls []map[string]any
		for _, t := range request.Tools {
			if t.Type != "function" {
				continue
			}
			params := t.Parameters
			if params == nil {
				params = map[string]any{"type": "OBJECT", "properties": map[string]any{}}
			}
			decls = append(decls, map[string]any{"name": t.Name, "description": t.Description, "parameters": params})
		}
		if len(decls) > 0 {
			payload["tools"] = []map[string]any{{"functionDeclarations": decls}}
		}
	}

	if request.Config.Provider != nil {
		output := toString(request.Config.Provider["output"])
		if output == "image" {
			if gc, ok := payload["generationConfig"].(map[string]any); ok {
				gc["responseModalities"] = []string{"IMAGE"}
			} else {
				payload["generationConfig"] = map[string]any{"responseModalities": []string{"IMAGE"}}
			}
		} else if output == "audio" {
			if gc, ok := payload["generationConfig"].(map[string]any); ok {
				gc["responseModalities"] = []string{"AUDIO"}
			} else {
				payload["generationConfig"] = map[string]any{"responseModalities": []string{"AUDIO"}}
			}
		}
		for k, v := range request.Config.Provider {
			if k == "prompt_caching" || k == "output" {
				continue
			}
			payload[k] = v
		}
	}

	return payload
}

func (a *GeminiAdapter) BuildRequest(request lm15.LMRequest, stream bool) lm15.HTTPRequest {
	endpoint := "generateContent"
	params := map[string]string{}
	if stream {
		endpoint = "streamGenerateContent"
		params["alt"] = "sse"
	}

	body, _ := json.Marshal(a.buildPayload(request))
	timeout := 60_000
	if stream {
		timeout = 120_000
	}
	return lm15.HTTPRequest{
		Method: "POST",
		URL:    fmt.Sprintf("%s/%s:%s", a.BaseURL, a.modelPath(request.Model), endpoint),
		Headers: a.authHeaders(),
		Params:  params,
		Body:    body,
		Timeout: durationMs(timeout),
	}
}

func (a *GeminiAdapter) parseCandidateParts(partsPayload []any) []lm15.Part {
	var parts []lm15.Part
	for _, pp := range partsPayload {
		p, _ := pp.(map[string]any)
		if p == nil {
			continue
		}
		if text, ok := p["text"].(string); ok {
			parts = append(parts, lm15.TextPart(text))
		} else if fc, ok := p["functionCall"].(map[string]any); ok {
			args, _ := fc["args"].(map[string]any)
			if args == nil {
				args = map[string]any{}
			}
			parts = append(parts, lm15.ToolCallPart(orDefault(toString(fc["id"]), "fc_0"), toString(fc["name"]), args))
		} else if inline, ok := p["inlineData"].(map[string]any); ok {
			mime := toString(inline["mimeType"])
			data := toString(inline["data"])
			ds := lm15.DataSource{Type: "base64", MediaType: mime, Data: data}
			if strings.HasPrefix(mime, "image/") {
				parts = append(parts, lm15.ImagePart(ds))
			} else if strings.HasPrefix(mime, "audio/") {
				parts = append(parts, lm15.AudioPart(ds))
			} else {
				parts = append(parts, lm15.DocumentPart(ds))
			}
		}
	}
	return parts
}

func (a *GeminiAdapter) ParseResponse(request lm15.LMRequest, response lm15.HTTPResponse) (lm15.LMResponse, error) {
	var data map[string]any
	if err := json.Unmarshal(response.Body, &data); err != nil {
		return lm15.LMResponse{}, err
	}

	// In-band error
	if pf, ok := data["promptFeedback"].(map[string]any); ok {
		reason := toString(pf["blockReason"])
		if reason != "" && reason != "BLOCK_REASON_UNSPECIFIED" {
			return lm15.LMResponse{}, &lm15.InvalidRequestError{lm15.ProviderError{lm15.ULMError{fmt.Sprintf("Prompt blocked: %s", reason)}}}
		}
	}

	candidates := toSlice(data["candidates"])
	candidate := map[string]any{}
	if len(candidates) > 0 {
		candidate, _ = candidates[0].(map[string]any)
	}
	content, _ := candidate["content"].(map[string]any)
	parts := a.parseCandidateParts(toSlice(content["parts"]))

	if len(parts) == 0 {
		parts = append(parts, lm15.TextPart(""))
	}

	um, _ := data["usageMetadata"].(map[string]any)
	usage := lm15.Usage{
		InputTokens:  toInt(um["promptTokenCount"]),
		OutputTokens: toInt(um["candidatesTokenCount"]),
		TotalTokens:  toInt(um["totalTokenCount"]),
	}
	if v, ok := um["cachedContentTokenCount"]; ok {
		i := toInt(v)
		usage.CacheReadTokens = &i
	}
	if v, ok := um["thoughtsTokenCount"]; ok {
		i := toInt(v)
		usage.ReasoningTokens = &i
	}

	finish := lm15.FinishStop
	for _, p := range parts {
		if p.Type == lm15.PartToolCall {
			finish = lm15.FinishToolCall
			break
		}
	}

	return lm15.LMResponse{
		ID: toString(data["responseId"]), Model: request.Model,
		Message:      lm15.Message{Role: lm15.RoleAssistant, Parts: parts},
		FinishReason: finish, Usage: usage, Provider: data,
	}, nil
}

func (a *GeminiAdapter) ParseStreamEvent(request lm15.LMRequest, raw lm15.SSEEvent) (*lm15.StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(raw.Data), &payload); err != nil {
		return nil, nil
	}

	if errObj, ok := payload["error"].(map[string]any); ok {
		code := orDefault(toString(errObj["status"]), toString(errObj["code"]))
		msg := toString(errObj["message"])
		canonCode := "provider"
		if c, ok := geminiErrorMap[code]; ok {
			canonCode = c
		}
		return &lm15.StreamEvent{Type: "error", Error: &lm15.ErrorInfo{Code: canonCode, Message: msg, ProviderCode: code}}, nil
	}

	cands := toSlice(payload["candidates"])
	if len(cands) == 0 {
		return nil, nil
	}
	cand, _ := cands[0].(map[string]any)
	content, _ := cand["content"].(map[string]any)
	partsList := toSlice(content["parts"])
	if len(partsList) == 0 {
		return nil, nil
	}
	part, _ := partsList[0].(map[string]any)

	if text, ok := part["text"].(string); ok {
		return &lm15.StreamEvent{Type: "delta", PartIndex: intPtr(0), Delta: &lm15.PartDelta{Type: "text", Text: text}}, nil
	}
	if fc, ok := part["functionCall"].(map[string]any); ok {
		args, _ := json.Marshal(fc["args"])
		return &lm15.StreamEvent{
			Type: "delta", PartIndex: intPtr(0),
			DeltaRaw: map[string]any{
				"type": "tool_call", "id": orDefault(toString(fc["id"]), "fc_0"),
				"name": toString(fc["name"]), "input": string(args),
			},
		}, nil
	}
	if inline, ok := part["inlineData"].(map[string]any); ok {
		mime := toString(inline["mimeType"])
		if strings.HasPrefix(mime, "audio/") {
			return &lm15.StreamEvent{Type: "delta", PartIndex: intPtr(0), Delta: &lm15.PartDelta{Type: "audio", Data: toString(inline["data"])}}, nil
		}
	}

	return nil, nil
}

func (a *GeminiAdapter) Embeddings(request lm15.EmbeddingRequest) (lm15.EmbeddingResponse, error) {
	modelPath := a.modelPath(request.Model)

	if len(request.Inputs) <= 1 {
		input := ""
		if len(request.Inputs) > 0 {
			input = request.Inputs[0]
		}
		payload := map[string]any{"model": modelPath, "content": map[string]any{"parts": []map[string]any{{"text": input}}}}
		body, _ := json.Marshal(payload)
		req := lm15.HTTPRequest{
			Method: "POST", URL: fmt.Sprintf("%s/%s:embedContent", a.BaseURL, modelPath),
			Headers: a.authHeaders(), Body: body, Timeout: durationMs(60_000),
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
		emb, _ := data["embedding"].(map[string]any)
		values := toFloat64Slice(emb["values"])
		return lm15.EmbeddingResponse{Model: request.Model, Vectors: [][]float64{values}, Provider: data}, nil
	}

	var requests []map[string]any
	for _, input := range request.Inputs {
		requests = append(requests, map[string]any{"model": modelPath, "content": map[string]any{"parts": []map[string]any{{"text": input}}}})
	}
	payload := map[string]any{"requests": requests}
	body, _ := json.Marshal(payload)
	req := lm15.HTTPRequest{
		Method: "POST", URL: fmt.Sprintf("%s/%s:batchEmbedContents", a.BaseURL, modelPath),
		Headers: a.authHeaders(), Body: body, Timeout: durationMs(60_000),
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
	for _, e := range toSlice(data["embeddings"]) {
		em, _ := e.(map[string]any)
		vectors = append(vectors, toFloat64Slice(em["values"]))
	}
	return lm15.EmbeddingResponse{Model: request.Model, Vectors: vectors, Provider: data}, nil
}

func (a *GeminiAdapter) FileUpload(request lm15.FileUploadRequest) (lm15.FileUploadResponse, error) {
	uploadBase := strings.Replace(a.BaseURL, "/v1beta", "/upload/v1beta", 1)
	req := lm15.HTTPRequest{
		Method: "POST", URL: uploadBase + "/files",
		Headers: map[string]string{
			"x-goog-api-key":          a.APIKey,
			"X-Goog-Upload-Protocol":  "raw",
			"X-Goog-Upload-File-Name": request.Filename,
			"Content-Type":            request.MediaType,
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
	fileID := toString(data["name"])
	if fileID == "" {
		if f, ok := data["file"].(map[string]any); ok {
			fileID = toString(f["name"])
		}
	}
	return lm15.FileUploadResponse{ID: fileID, Provider: data}, nil
}
