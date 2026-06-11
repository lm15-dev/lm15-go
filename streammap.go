package lm15

import (
	"encoding/json"
	"strings"
)

// ─── Per-provider stream-event mapping ───────────────────────────────
//
// Stateless adapters: each provider SSE frame maps to zero or more
// canonical StreamEvents. Providers may emit one end event per terminal
// frame; the MAP-3 coalescer (coalesce.go) merges them.

func decodeJSONValue(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, valueErr("invalid stream JSON: %v", err)
	}
	return v, nil
}

// ParseStreamEvents maps one SSE frame to canonical stream events.
func ParseStreamEvents(provider string, req Request, raw SSEEvent) ([]StreamEvent, error) {
	switch provider {
	case "openai":
		return openAIStreamEvents(req, raw)
	case "openai_chat":
		return openAIChatStreamEvents(req, raw)
	case "anthropic":
		return anthropicStreamEvents(req, raw)
	case "gemini":
		return geminiStreamEvents(req, raw)
	}
	return nil, valueErr("unknown provider: %s", provider)
}

// ─── Stream error details ────────────────────────────────────────────

func streamErrorDetail(code, message, providerCode string) ErrorDetail {
	if message == "" {
		message = providerCode
	}
	if message == "" {
		message = "provider error"
	}
	if providerCode == "" {
		providerCode = "provider"
	}
	return ErrorDetail{Code: code, Message: message, ProviderCode: &providerCode}
}

var openAIStreamErrorCodes = func() map[string]string {
	m := map[string]string{
		"context_length_exceeded": "context_length",
		"invalid_api_key":         "auth",
		"insufficient_quota":      "billing",
		"authentication_error":    "auth",
		"rate_limit_error":        "rate_limit",
	}
	for k, v := range openAIResponseErrorCodes {
		m[k] = v
	}
	return m
}()

func openAIStreamErrorDetail(providerCode, message string) ErrorDetail {
	code, ok := openAIStreamErrorCodes[providerCode]
	if !ok {
		code = "provider"
	}
	return streamErrorDetail(code, message, providerCode)
}

func anthropicStreamErrorDetail(providerCode, message string) ErrorDetail {
	code, ok := anthropicErrorTypeCodes[providerCode]
	if !ok {
		code = "provider"
	}
	if anthropicContextLength(message) {
		code = "context_length"
	} else if providerCode == "not_found_error" && isModelErrorMessage(message) {
		code = "unsupported_model"
	}
	return streamErrorDetail(code, message, providerCode)
}

func geminiStreamErrorDetail(providerCode, message string) ErrorDetail {
	code, ok := geminiErrorStatusCodes[providerCode]
	if !ok {
		code = "provider"
	}
	if geminiContextLength(message) {
		code = "context_length"
	} else if providerCode == "NOT_FOUND" && isModelErrorMessage(message) {
		code = "unsupported_model"
	}
	return streamErrorDetail(code, message, providerCode)
}

// ─── OpenAI Chat Completions dialect ─────────────────────────────────

func openAIChatStreamEvents(req Request, raw SSEEvent) ([]StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	if raw.Data == "[DONE]" {
		return []StreamEvent{StreamEndEvent{}}, nil
	}
	value, err := decodeJSONValue(raw.Data)
	if err != nil {
		return nil, err
	}
	payload, ok := value.(jmap)
	if !ok {
		return nil, nil
	}

	if errObj, ok := payload["error"].(jmap); ok {
		providerCode := strOrEmpty(errObj["code"])
		if providerCode == "" {
			providerCode = strOrEmpty(errObj["type"])
		}
		if providerCode == "" {
			providerCode = "provider"
		}
		return []StreamEvent{StreamErrorEvent{Error: openAIStreamErrorDetail(providerCode, strOrEmpty(errObj["message"]))}}, nil
	}

	var events []StreamEvent
	choice := jmap{}
	if choices := listAt(payload, "choices"); len(choices) > 0 {
		if c, ok := choices[0].(jmap); ok {
			choice = c
		}
	}
	delta := objAt(choice, "delta")

	reasoning := delta["reasoning_content"]
	if !truthy(reasoning) {
		reasoning = delta["reasoning"]
	}
	if truthy(reasoning) {
		events = append(events, StreamDeltaEvent{Delta: ThinkingDelta{Text: pyStr(reasoning)}})
	}

	if content, ok := delta["content"].(string); ok && content != "" {
		events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: content}})
	}

	for _, c := range listAt(delta, "tool_calls") {
		call, ok := c.(jmap)
		if !ok {
			continue
		}
		function := objAt(call, "function")
		events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{
			Input:     strOrEmpty(function["arguments"]),
			PartIndex: intOrZero(call["index"]),
			ID:        strOrNil(call["id"]),
			Name:      strOrNil(function["name"]),
		}})
	}

	finishRaw := choice["finish_reason"]
	usageData, hasUsage := payload["usage"].(jmap)
	if truthy(finishRaw) {
		finish, ok := chatFinishReasonMap[pyStr(finishRaw)]
		if !ok {
			finish = "stop"
		}
		end := StreamEndEvent{FinishReason: &finish}
		if hasUsage {
			u := usageFromChat(usageData)
			end.Usage = &u
		}
		events = append(events, end)
	} else if hasUsage {
		// Final usage-only chunk (stream_options.include_usage).
		u := usageFromChat(usageData)
		events = append(events, StreamEndEvent{Usage: &u})
	}
	return events, nil
}

// ─── OpenAI Responses API ────────────────────────────────────────────

func openAIStreamEvents(req Request, raw SSEEvent) ([]StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	if raw.Data == "[DONE]" {
		stop := "stop"
		return []StreamEvent{StreamEndEvent{FinishReason: &stop}}, nil
	}
	value, err := decodeJSONValue(raw.Data)
	if err != nil {
		return nil, err
	}
	payload, ok := value.(jmap)
	if !ok {
		return nil, valueErr("openai stream frame must be a JSON object")
	}
	et := strOrEmpty(payload["type"])
	one := func(e StreamEvent) ([]StreamEvent, error) { return []StreamEvent{e}, nil }

	switch et {
	case "response.created":
		response := objAt(payload, "response")
		model := strOrEmpty(response["model"])
		if model == "" {
			model = req.Model
		}
		return one(StreamStartEvent{ID: strOrNil(response["id"]), Model: &model})

	case "response.output_text.delta", "response.refusal.delta":
		return one(StreamDeltaEvent{Delta: TextDelta{
			Text:      strOrEmpty(payload["delta"]),
			PartIndex: intOrZero(payload["output_index"]),
		}})

	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		return one(StreamDeltaEvent{Delta: ThinkingDelta{
			Text:      strOrEmpty(payload["delta"]),
			PartIndex: intOrZero(payload["output_index"]),
		}})

	case "response.output_text.annotation.added":
		if annotation, ok := payload["annotation"].(jmap); ok {
			if c := citationFromOpenAIAnnotation(annotation, nil); c != nil {
				return one(StreamDeltaEvent{Delta: CitationDelta{
					Text: c.Text, URL: c.URL, Title: c.Title,
					PartIndex: intOrZero(payload["output_index"]),
				}})
			}
		}
		return nil, nil

	case "response.output_audio.delta":
		data := strOrEmpty(payload["delta"])
		mt := "audio/wav"
		return one(StreamDeltaEvent{Delta: AudioDelta{
			Data: &data, PartIndex: intOrZero(payload["output_index"]), MediaType: &mt,
		}})

	case "response.output_image.delta", "response.image.delta":
		data := strOrEmpty(payload["delta"])
		mt := "image/png"
		return one(StreamDeltaEvent{Delta: ImageDelta{
			Data: &data, PartIndex: intOrZero(payload["output_index"]), MediaType: &mt,
		}})

	case "response.output_item.added":
		item := objAt(payload, "item")
		if strOrEmpty(item["type"]) == "function_call" {
			id := item["call_id"]
			if !truthy(id) {
				id = item["id"]
			}
			return one(StreamDeltaEvent{Delta: ToolCallDelta{
				Input:     strOrEmpty(item["arguments"]),
				PartIndex: intOrZero(payload["output_index"]),
				ID:        strOrNil(id),
				Name:      strOrNil(item["name"]),
			}})
		}
		return nil, nil

	case "response.function_call_arguments.delta":
		id := payload["call_id"]
		if !truthy(id) {
			id = payload["id"]
		}
		return one(StreamDeltaEvent{Delta: ToolCallDelta{
			Input:     strOrEmpty(payload["delta"]),
			PartIndex: intOrZero(payload["output_index"]),
			ID:        strOrNil(id),
			Name:      strOrNil(payload["name"]),
		}})

	case "response.completed":
		response := objAt(payload, "response")
		usageData := objAt(response, "usage")
		inputDetails := objAt(usageData, "input_tokens_details")
		outputDetails := objAt(usageData, "output_tokens_details")
		usage := Usage{
			InputTokens:       i64ptr(intOrZero(usageData["input_tokens"])),
			OutputTokens:      i64ptr(intOrZero(usageData["output_tokens"])),
			TotalTokens:       intOrNilTok(usageData["total_tokens"]),
			ReasoningTokens:   intOrNilTok(outputDetails["reasoning_tokens"]),
			CacheReadTokens:   intOrNilTok(inputDetails["cached_tokens"]),
			InputAudioTokens:  intOrNilTok(inputDetails["audio_tokens"]),
			OutputAudioTokens: intOrNilTok(outputDetails["audio_tokens"]),
		}
		finish := "stop"
		for _, item := range listAt(response, "output") {
			if it, ok := item.(jmap); ok && strOrEmpty(it["type"]) == "function_call" {
				finish = "tool_call"
				break
			}
		}
		return one(StreamEndEvent{FinishReason: &finish, Usage: &usage, ProviderData: response})

	case "response.error", "error":
		providerCode, message := streamErrorEnvelope(payload)
		return one(StreamErrorEvent{Error: openAIStreamErrorDetail(providerCode, message)})
	}
	return nil, nil
}

// streamErrorEnvelope extracts (provider_code, message) from an error frame
// whose detail may live under "error" or at the top level.
func streamErrorEnvelope(payload jmap) (string, string) {
	if errObj, ok := payload["error"].(jmap); ok {
		providerCode := errObj["code"]
		if !truthy(providerCode) {
			providerCode = errObj["type"]
		}
		if !truthy(providerCode) {
			providerCode = payload["code"]
		}
		pc := strOrEmpty(providerCode)
		if pc == "" {
			pc = "provider"
		}
		message := errObj["message"]
		if !truthy(message) {
			message = payload["message"]
		}
		return pc, strOrEmpty(message)
	}
	providerCode := payload["code"]
	if !truthy(providerCode) {
		providerCode = payload["error_type"]
	}
	pc := strOrEmpty(providerCode)
	if pc == "" {
		pc = "provider"
	}
	return pc, strOrEmpty(payload["message"])
}

// ─── Anthropic Messages API ──────────────────────────────────────────

func anthropicStreamEvents(req Request, raw SSEEvent) ([]StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	value, err := decodeJSONValue(raw.Data)
	if err != nil {
		return nil, err
	}
	payload, ok := value.(jmap)
	if !ok {
		return nil, valueErr("anthropic stream frame must be a JSON object")
	}
	et := strOrEmpty(payload["type"])

	switch et {
	case "message_start":
		msg := objAt(payload, "message")
		model := strOrEmpty(msg["model"])
		if model == "" {
			model = req.Model
		}
		events := []StreamEvent{StreamStartEvent{ID: strOrNil(msg["id"]), Model: &model}}
		if truthy(msg["id"]) {
			events = append(events, StreamDeltaEvent{Delta: ContinuationDelta{
				Provider: "anthropic", Kind: "message_id",
				Data: jmap{"id": pyStr(msg["id"])},
			}})
		}
		return events, nil

	case "content_block_start":
		block := objAt(payload, "content_block")
		idx := intOrZero(payload["index"])
		switch strOrEmpty(block["type"]) {
		case "tool_use":
			var input string
			if obj, ok := block["input"].(jmap); ok {
				b, err := json.Marshal(obj)
				if err != nil {
					return nil, err
				}
				input = string(b)
			} else if truthy(block["input"]) {
				input = pyStr(block["input"])
			}
			return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{
				Input: input, PartIndex: idx,
				ID: strOrNil(block["id"]), Name: strOrNil(block["name"]),
			}}}, nil
		case "redacted_thinking":
			if block["data"] != nil {
				return []StreamEvent{
					StreamDeltaEvent{Delta: ThinkingDelta{Text: "[redacted]", PartIndex: idx}},
					StreamDeltaEvent{Delta: ContinuationDelta{
						Provider: "anthropic", Kind: "redacted_thinking",
						Data: jmap{"data": block["data"]}, PartIndex: &idx,
					}},
				}, nil
			}
		}
		return nil, nil

	case "content_block_delta":
		delta := objAt(payload, "delta")
		idx := intOrZero(payload["index"])
		switch strOrEmpty(delta["type"]) {
		case "text_delta":
			return []StreamEvent{StreamDeltaEvent{Delta: TextDelta{Text: strOrEmpty(delta["text"]), PartIndex: idx}}}, nil
		case "input_json_delta":
			return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: strOrEmpty(delta["partial_json"]), PartIndex: idx}}}, nil
		case "thinking_delta":
			return []StreamEvent{StreamDeltaEvent{Delta: ThinkingDelta{Text: strOrEmpty(delta["thinking"]), PartIndex: idx}}}, nil
		case "signature_delta":
			if truthy(delta["signature"]) {
				return []StreamEvent{StreamDeltaEvent{Delta: ContinuationDelta{
					Provider: "anthropic", Kind: "thinking_signature",
					Data: jmap{"signature": pyStr(delta["signature"])}, PartIndex: &idx,
				}}}, nil
			}
		case "citation_delta", "citations_delta":
			citation := delta
			if c, ok := delta["citation"].(jmap); ok {
				citation = c
			}
			text := citation["cited_text"]
			if !truthy(text) {
				text = citation["text"]
			}
			return []StreamEvent{StreamDeltaEvent{Delta: CitationDelta{
				PartIndex: idx,
				Text:      strOrNil(text),
				URL:       strOrNil(citation["url"]),
				Title:     strOrNil(citation["title"]),
			}}}, nil
		}
		return nil, nil

	case "message_delta":
		// Anthropic sends the authoritative stop_reason and final usage here;
		// message_stop is just the terminator and carries neither.
		delta := objAt(payload, "delta")
		usagePayload := objAt(payload, "usage")
		var usage *Usage
		if len(usagePayload) > 0 {
			input := intOrZero(usagePayload["input_tokens"])
			output := intOrZero(usagePayload["output_tokens"])
			usage = &Usage{
				InputTokens:      i64ptr(input),
				OutputTokens:     i64ptr(output),
				TotalTokens:      i64ptr(input + output),
				CacheReadTokens:  intOrNilTok(usagePayload["cache_read_input_tokens"]),
				CacheWriteTokens: intOrNilTok(usagePayload["cache_creation_input_tokens"]),
			}
		}
		var finish *string
		if delta["stop_reason"] != nil {
			f := anthropicFinishReason(delta["stop_reason"], false)
			finish = &f
		}
		if finish != nil || usage != nil {
			return []StreamEvent{StreamEndEvent{FinishReason: finish, Usage: usage}}, nil
		}
		return nil, nil

	case "message_stop":
		return []StreamEvent{StreamEndEvent{}}, nil

	case "error":
		providerCode, message := streamErrorEnvelope(payload)
		return []StreamEvent{StreamErrorEvent{Error: anthropicStreamErrorDetail(providerCode, message)}}, nil
	}
	return nil, nil
}

// ─── Gemini generateContent (streaming) ──────────────────────────────

func geminiStreamEvents(req Request, raw SSEEvent) ([]StreamEvent, error) {
	if raw.Data == "" {
		return nil, nil
	}
	value, err := decodeJSONValue(raw.Data)
	if err != nil {
		return nil, err
	}
	payload, ok := value.(jmap)
	if !ok {
		return nil, nil
	}

	if _, present := payload["error"]; present {
		providerCode, message := "provider", ""
		if errObj, ok := payload["error"].(jmap); ok {
			code := errObj["status"]
			if !truthy(code) {
				code = errObj["code"]
			}
			if truthy(code) {
				providerCode = pyStr(code)
			}
			message = strOrEmpty(errObj["message"])
		}
		return []StreamEvent{StreamErrorEvent{Error: geminiStreamErrorDetail(providerCode, message)}}, nil
	}

	if inband := geminiInbandError(payload); inband != nil {
		pc := "inband_finish_reason"
		return []StreamEvent{StreamErrorEvent{Error: ErrorDetail{
			Code: inband.Code(), ProviderCode: &pc, Message: inband.Error(),
		}}}, nil
	}

	var events []StreamEvent
	var candidate jmap
	if candidates := listAt(payload, "candidates"); len(candidates) > 0 {
		if c, ok := candidates[0].(jmap); ok {
			candidate = c
		}
	}
	yieldedDelta := false
	sawTool := false
	finish := ""
	if candidate != nil {
		content := objAt(candidate, "content")
		for i, p := range listAt(content, "parts") {
			part, ok := p.(jmap)
			if !ok {
				continue
			}
			idx := int64(i)
			_, hasText := part["text"]
			switch {
			case truthy(part["thought"]) && hasText:
				yieldedDelta = true
				events = append(events, StreamDeltaEvent{Delta: ThinkingDelta{Text: strOrEmpty(part["text"]), PartIndex: idx}})
				if part["thoughtSignature"] != nil {
					events = append(events, StreamDeltaEvent{Delta: ContinuationDelta{
						Provider: "gemini", Kind: "thought_signature",
						Data: jmap{"value": pyStr(part["thoughtSignature"])}, PartIndex: &idx,
					}})
				}
			case hasText:
				yieldedDelta = true
				events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: strOrEmpty(part["text"]), PartIndex: idx}})
			default:
				if fc, ok := part["functionCall"].(jmap); ok {
					sawTool = true
					yieldedDelta = true
					args := fc["args"]
					if args == nil {
						args = jmap{}
					}
					b, err := json.Marshal(args)
					if err != nil {
						return nil, err
					}
					events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{
						Input: string(b), PartIndex: idx,
						ID: strOrNil(fc["id"]), Name: strOrNil(fc["name"]),
					}})
					sig := part["thoughtSignature"]
					if !truthy(sig) {
						sig = fc["thoughtSignature"]
					}
					if sig != nil {
						events = append(events, StreamDeltaEvent{Delta: ContinuationDelta{
							Provider: "gemini", Kind: "thought_signature",
							Data: jmap{"value": pyStr(sig)}, PartIndex: &idx,
						}})
					}
				} else if inline, ok := part["inlineData"].(jmap); ok {
					mime := strOrEmpty(inline["mimeType"])
					if mime == "" {
						mime = "application/octet-stream"
					}
					data := strOrEmpty(inline["data"])
					if strings.HasPrefix(mime, "audio/") {
						yieldedDelta = true
						events = append(events, StreamDeltaEvent{Delta: AudioDelta{Data: &data, PartIndex: idx, MediaType: &mime}})
					} else if strings.HasPrefix(mime, "image/") {
						yieldedDelta = true
						events = append(events, StreamDeltaEvent{Delta: ImageDelta{Data: &data, PartIndex: idx, MediaType: &mime}})
					}
				}
			}
		}
		finish = strOrEmpty(candidate["finishReason"])
	}

	if payload["responseId"] != nil {
		events = append(events, StreamDeltaEvent{Delta: ContinuationDelta{
			Provider: "gemini", Kind: "response_id",
			Data: jmap{"id": pyStr(payload["responseId"])},
		}})
	}

	_, hasUsage := payload["usageMetadata"]
	if finish != "" {
		f := geminiFinishReason(finish, sawTool)
		u := geminiStreamUsage(payload)
		events = append(events, StreamEndEvent{FinishReason: &f, Usage: &u, ProviderData: payload})
	} else if !yieldedDelta && hasUsage {
		f := "stop"
		u := geminiStreamUsage(payload)
		events = append(events, StreamEndEvent{FinishReason: &f, Usage: &u, ProviderData: payload})
	}
	return events, nil
}

func geminiStreamUsage(payload jmap) Usage {
	u := objAt(payload, "usageMetadata")
	output := u["candidatesTokenCount"]
	if _, present := u["candidatesTokenCount"]; !present {
		output = u["responseTokenCount"]
	}
	return Usage{
		InputTokens:     i64ptr(intOrZero(u["promptTokenCount"])),
		OutputTokens:    i64ptr(intOrZero(output)),
		TotalTokens:     intOrNilTok(u["totalTokenCount"]),
		CacheReadTokens: intOrNilTok(u["cachedContentTokenCount"]),
		ReasoningTokens: intOrNilTok(u["thoughtsTokenCount"]),
	}
}
