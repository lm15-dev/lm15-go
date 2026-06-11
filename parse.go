package lm15

// parse.go — parse_response for the four wire providers, per the contract
// (lm15-contract spec + goldens) and the canonical mapping rules
// (lm15-python2/docs/mapping-rules.md MAP-1, MAP-2).

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// ─── Unmapped recorder ───────────────────────────────────────────────

type unmappedRecorder struct {
	entries []jmap
}

func (u *unmappedRecorder) record(path string, typ string) {
	if typ == "" {
		typ = "<missing>"
	}
	u.entries = append(u.entries, jmap{"path": path, "type": typ})
}

// pyTypeName mirrors Python's type(x).__name__ for decoded JSON values.
func pyTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "NoneType"
	case string:
		return "str"
	case bool:
		return "bool"
	case json.Number:
		return "int"
	case float64:
		return "float"
	case jmap:
		return "dict"
	case []any:
		return "list"
	}
	return fmt.Sprintf("%T", v)
}

// ─── Python-semantics helpers over decoded JSON values ───────────────

func truthy(v any) bool {
	switch t := v.(type) {
	case nil:
		return false
	case string:
		return t != ""
	case bool:
		return t
	case json.Number:
		f, err := t.Float64()
		return err != nil || f != 0
	case float64:
		return t != 0
	case jmap:
		return len(t) > 0
	case []any:
		return len(t) > 0
	}
	return true
}

// pyStr renders a decoded JSON scalar the way Python's str() would for the
// values that actually occur on these paths (strings and numbers).
func pyStr(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case json.Number:
		return t.String()
	case bool:
		if t {
			return "True"
		}
		return "False"
	case nil:
		return "None"
	}
	return fmt.Sprintf("%v", v)
}

// strOrEmpty mirrors str(x or "").
func strOrEmpty(v any) string {
	if !truthy(v) {
		return ""
	}
	return pyStr(v)
}

// strOrNil mirrors _str_or_none: None for nil or "", else str(value).
func strOrNil(v any) *string {
	if v == nil || v == "" {
		return nil
	}
	s := pyStr(v)
	return &s
}

func intVal(v any) (int64, bool) {
	switch t := v.(type) {
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i, true
		}
		if f, err := t.Float64(); err == nil {
			return int64(f), true
		}
	case float64:
		return int64(t), true
	}
	return 0, false
}

// intOrZero mirrors int(x or 0).
func intOrZero(v any) int64 {
	if i, ok := intVal(v); ok {
		return i
	}
	return 0
}

// intOrNilTok mirrors passing payload.get(k) straight into an int|None Usage
// field: nil stays nil, a number becomes the int.
func intOrNilTok(v any) *int64 {
	if i, ok := intVal(v); ok {
		return &i
	}
	return nil
}

// intIndex mirrors _int_or_none (rejects bools).
func intIndex(v any) (int64, bool) {
	if _, isBool := v.(bool); isBool {
		return 0, false
	}
	return intVal(v)
}

func i64ptr(i int64) *int64 { return &i }

func objAt(m jmap, k string) jmap {
	if o, ok := m[k].(jmap); ok {
		return o
	}
	return jmap{}
}

func listAt(m jmap, k string) []any {
	if l, ok := m[k].([]any); ok {
		return l
	}
	return nil
}

// runeSlice mirrors Python's code-point slicing source[start:end] guarded by
// 0 <= start < end <= len(source).
func runeSlice(source string, start, end int64) (string, bool) {
	runes := []rune(source)
	if start >= 0 && start < end && end <= int64(len(runes)) {
		return string(runes[start:end]), true
	}
	return "", false
}

// parseJSONObject mirrors providers/common.py parse_json_object.
func parseJSONObject(v any) jmap {
	if m, ok := v.(jmap); ok {
		return m
	}
	if s, ok := v.(string); ok && s != "" {
		var parsed any
		dec := json.NewDecoder(strings.NewReader(s))
		dec.UseNumber()
		if err := dec.Decode(&parsed); err != nil {
			return jmap{"partial_json": s}
		}
		if m, ok := parsed.(jmap); ok {
			return m
		}
		return jmap{"value": parsed}
	}
	return jmap{}
}

func hasToolCall(parts []Part) bool {
	for _, p := range parts {
		if _, ok := p.(ToolCallPart); ok {
			return true
		}
	}
	return false
}

// ─── In-body provider errors ─────────────────────────────────────────

var openAIResponseErrorCodes = map[string]string{
	"server_error":                     "server",
	"rate_limit_exceeded":              "rate_limit",
	"invalid_prompt":                   "invalid_request",
	"vector_store_timeout":             "timeout",
	"invalid_image":                    "invalid_request",
	"invalid_image_format":             "invalid_request",
	"invalid_base64_image":             "invalid_request",
	"invalid_image_url":                "invalid_request",
	"image_too_large":                  "invalid_request",
	"image_too_small":                  "invalid_request",
	"image_parse_error":                "invalid_request",
	"image_content_policy_violation":   "invalid_request",
	"invalid_image_mode":               "invalid_request",
	"image_file_too_large":             "invalid_request",
	"unsupported_image_media_type":     "invalid_request",
	"empty_image_file":                 "invalid_request",
	"failed_to_download_image":         "invalid_request",
	"image_file_not_found":             "invalid_request",
	"model_not_found":                  "unsupported_model",
	"model_not_available":              "unsupported_model",
	"unsupported_model":                "unsupported_model",
}

func openAIResponseError(code, message string) LM15Error {
	canonical, ok := openAIResponseErrorCodes[code]
	if !ok {
		canonical = "server"
	}
	msg := message
	if msg == "" {
		msg = code
	}
	if msg == "" {
		msg = "provider error"
	}
	return ErrorForCode(canonical, msg)
}

// ─── Anthropic ───────────────────────────────────────────────────────

var anthropicProviderExecutedBlocks = map[string]bool{
	"server_tool_use":            true,
	"web_search_tool_result":     true,
	"code_execution_tool_result": true,
}

func anthropicFinishReason(stopReason any, hasTool bool) string {
	if hasTool {
		return "tool_call"
	}
	reason := strings.ToLower(strOrEmpty(stopReason))
	switch reason {
	case "max_tokens", "model_context_window_exceeded":
		return "length"
	case "tool_use", "pause_turn":
		return "tool_call"
	case "refusal", "safety", "content_filter":
		return "content_filter"
	}
	return "stop"
}

func citationFromAnthropic(c jmap) *CitationPart {
	url := c["url"]
	if !truthy(url) {
		url = c["uri"]
	}
	title := c["title"]
	if !truthy(title) {
		title = c["document_title"]
	}
	if !truthy(title) {
		title = c["source_title"]
	}
	text := c["cited_text"]
	if !truthy(text) {
		text = c["text"]
	}
	if !truthy(text) {
		text = c["quote"]
	}
	var urlS, titleS, textS *string
	if truthy(url) {
		s := pyStr(url)
		urlS = &s
	}
	if truthy(title) {
		s := pyStr(title)
		titleS = &s
	}
	if truthy(text) {
		s := pyStr(text)
		textS = &s
	}
	if urlS == nil && titleS == nil && textS == nil {
		return nil
	}
	return &CitationPart{URL: urlS, Title: titleS, Text: textS}
}

func parseAnthropicResponse(req Request, data jmap, rec *unmappedRecorder) Response {
	var parts []Part
	for i, raw := range listAt(data, "content") {
		block, ok := raw.(jmap)
		if !ok {
			rec.record(fmt.Sprintf("content[%d]", i), pyTypeName(raw))
			continue
		}
		blockType, _ := block["type"].(string)
		switch blockType {
		case "text":
			parts = append(parts, TextPart{Text: strOrEmpty(block["text"])})
			for _, rawCit := range listAt(block, "citations") {
				cit, ok := rawCit.(jmap)
				if !ok {
					continue
				}
				if c := citationFromAnthropic(cit); c != nil {
					parts = append(parts, *c)
				}
			}
		case "tool_use":
			id := strOrEmpty(block["id"])
			if id == "" {
				id = fmt.Sprintf("tool_%d", len(parts))
			}
			name := strOrEmpty(block["name"])
			if name == "" {
				name = "tool"
			}
			input, _ := block["input"].(jmap)
			if input == nil {
				input = jmap{}
			}
			parts = append(parts, ToolCallPart{ID: id, Name: name, Input: input})
		case "thinking":
			var cont []ContinuationState
			if truthy(block["signature"]) {
				cont = []ContinuationState{{
					Provider: "anthropic",
					Kind:     "thinking_signature",
					Data:     jmap{"signature": pyStr(block["signature"])},
				}}
			}
			text := block["thinking"]
			if !truthy(text) {
				text = block["text"]
			}
			parts = append(parts, ThinkingPart{Text: strOrEmpty(text), Continuation: cont})
		case "redacted_thinking":
			var cont []ContinuationState
			if payload, ok := block["data"]; ok && payload != nil {
				cont = []ContinuationState{{
					Provider: "anthropic",
					Kind:     "redacted_thinking",
					Data:     jmap{"data": payload},
				}}
			}
			parts = append(parts, ThinkingPart{Text: "[redacted]", Redacted: true, Continuation: cont})
		default:
			if anthropicProviderExecutedBlocks[blockType] {
				continue
			}
			rec.record(fmt.Sprintf("content[%d]", i), blockType)
		}
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{Text: ""}} // MAP-2
	}

	usagePayload := objAt(data, "usage")
	inputTokens := intOrZero(usagePayload["input_tokens"])
	outputTokens := intOrZero(usagePayload["output_tokens"])
	usage := Usage{
		InputTokens:      i64ptr(inputTokens),
		OutputTokens:     i64ptr(outputTokens),
		TotalTokens:      i64ptr(inputTokens + outputTokens),
		CacheReadTokens:  intOrNilTok(usagePayload["cache_read_input_tokens"]),
		CacheWriteTokens: intOrNilTok(usagePayload["cache_creation_input_tokens"]),
	}

	var msgCont []ContinuationState
	var id *string
	if truthy(data["id"]) {
		s := pyStr(data["id"])
		id = &s
		msgCont = []ContinuationState{{Provider: "anthropic", Kind: "message_id", Data: jmap{"id": s}}}
	}
	model := strOrEmpty(data["model"])
	if model == "" {
		model = req.Model
	}
	return Response{
		ID:           id,
		Model:        model,
		Message:      Message{Role: "assistant", Parts: parts, Continuation: msgCont},
		FinishReason: anthropicFinishReason(data["stop_reason"], hasToolCall(parts)),
		Usage:        usage,
	}
}

// ─── OpenAI (Responses API) ──────────────────────────────────────────

var openAIProviderExecutedItems = map[string]bool{
	"web_search_call":       true,
	"file_search_call":      true,
	"code_interpreter_call": true,
	"computer_call":         true,
	"computer_use_call":     true,
}

func openAIFinishFromStatus(data jmap, hasTool bool) string {
	if hasTool {
		return "tool_call"
	}
	status := strings.ToLower(strOrEmpty(data["status"]))
	reason := ""
	if incomplete, ok := data["incomplete_details"].(jmap); ok {
		reason = strings.ToLower(strOrEmpty(incomplete["reason"]))
	}
	if status == "incomplete" && strings.Contains(reason, "token") {
		return "length"
	}
	if strings.Contains(reason, "content_filter") || strings.Contains(reason, "safety") {
		return "content_filter"
	}
	return "stop"
}

func openAIAnnotationText(annotation jmap, sourceText *string) *string {
	for _, key := range []string{"text", "snippet", "cited_text", "quote"} {
		if t := strOrNil(annotation[key]); t != nil {
			return t
		}
	}
	start, okS := intIndex(annotation["start_index"])
	end, okE := intIndex(annotation["end_index"])
	if sourceText != nil && okS && okE {
		if s, ok := runeSlice(*sourceText, start, end); ok {
			return &s
		}
	}
	return nil
}

func citationFromOpenAIAnnotation(annotation jmap, sourceText *string) *CitationPart {
	urlV := annotation["url"]
	if !truthy(urlV) {
		urlV = annotation["uri"]
	}
	url := strOrNil(urlV)
	titleV := annotation["title"]
	if !truthy(titleV) {
		titleV = annotation["filename"]
	}
	if !truthy(titleV) {
		titleV = annotation["file_id"]
	}
	title := strOrNil(titleV)
	text := openAIAnnotationText(annotation, sourceText)
	if url == nil && title == nil && text == nil {
		return nil
	}
	return &CitationPart{URL: url, Title: title, Text: text}
}

func parseOpenAIResponse(req Request, data jmap, rec *unmappedRecorder) (Response, error) {
	if respErr, ok := data["error"].(jmap); ok {
		return Response{}, openAIResponseError(strOrEmpty(respErr["code"]), strOrEmpty(respErr["message"]))
	}

	var parts []Part
	for i, rawItem := range listAt(data, "output") {
		item, ok := rawItem.(jmap)
		if !ok {
			rec.record(fmt.Sprintf("output[%d]", i), pyTypeName(rawItem))
			continue
		}
		itemType, _ := item["type"].(string)
		switch itemType {
		case "message":
			for j, rawContent := range listAt(item, "content") {
				content, ok := rawContent.(jmap)
				if !ok {
					rec.record(fmt.Sprintf("output[%d].content[%d]", i, j), pyTypeName(rawContent))
					continue
				}
				ctype, _ := content["type"].(string)
				switch ctype {
				case "output_text", "text":
					text := strOrEmpty(content["text"])
					parts = append(parts, TextPart{Text: text})
					for _, rawAnn := range listAt(content, "annotations") {
						ann, ok := rawAnn.(jmap)
						if !ok {
							continue
						}
						if c := citationFromOpenAIAnnotation(ann, &text); c != nil {
							parts = append(parts, *c)
						}
					}
				case "refusal":
					text := content["refusal"]
					if !truthy(text) {
						text = content["text"]
					}
					if s := strOrEmpty(text); s != "" {
						parts = append(parts, RefusalPart{Text: s})
					} else {
						parts = append(parts, TextPart{Text: ""})
					}
				case "output_image":
					b64 := content["b64_json"]
					if !truthy(b64) {
						b64 = content["image_base64"]
					}
					if truthy(b64) {
						s := pyStr(b64)
						parts = append(parts, ImagePart{mediaPart: mediaPart{MediaType: "image/png", Data: &s}})
					}
				case "output_audio":
					audio := objAt(content, "audio")
					b64 := audio["data"]
					if !truthy(b64) {
						b64 = content["b64_json"]
					}
					if truthy(b64) {
						s := pyStr(b64)
						parts = append(parts, AudioPart{mediaPart: mediaPart{MediaType: "audio/wav", Data: &s}})
					}
				default:
					rec.record(fmt.Sprintf("output[%d].content[%d]", i, j), ctype)
				}
			}
		case "function_call":
			id := strOrEmpty(item["call_id"])
			if id == "" {
				id = strOrEmpty(item["id"])
			}
			if id == "" {
				id = fmt.Sprintf("call_%d", len(parts))
			}
			name := strOrEmpty(item["name"])
			if name == "" {
				name = "tool"
			}
			parts = append(parts, ToolCallPart{ID: id, Name: name, Input: parseJSONObject(item["arguments"])})
		case "reasoning":
			var text string
			if summary, ok := item["summary"].([]any); ok {
				segs := make([]string, len(summary))
				for k, x := range summary {
					if xm, ok := x.(jmap); ok {
						segs[k] = pyStr(xm["text"])
					} else {
						segs[k] = pyStr(x)
					}
				}
				text = strings.Join(segs, "\n")
			} else {
				v := item["summary"]
				if !truthy(v) {
					v = item["text"]
				}
				text = strOrEmpty(v)
			}
			if text != "" {
				parts = append(parts, ThinkingPart{Text: text})
			}
		default:
			if openAIProviderExecutedItems[itemType] {
				continue
			}
			rec.record(fmt.Sprintf("output[%d]", i), itemType)
		}
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{Text: strOrEmpty(data["output_text"])}} // MAP-2
	}

	usageData := objAt(data, "usage")
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

	var msgCont []ContinuationState
	var id *string
	if truthy(data["id"]) {
		s := pyStr(data["id"])
		id = &s
		msgCont = []ContinuationState{{Provider: "openai", Kind: "response_id", Data: jmap{"id": s}}}
	}
	model := strOrEmpty(data["model"])
	if model == "" {
		model = req.Model
	}
	return Response{
		ID:           id,
		Model:        model,
		Message:      Message{Role: "assistant", Parts: parts, Continuation: msgCont},
		FinishReason: openAIFinishFromStatus(data, hasToolCall(parts)),
		Usage:        usage,
	}, nil
}

// ─── OpenAI Chat Completions ─────────────────────────────────────────

var chatFinishReasonMap = map[string]string{
	"stop":           "stop",
	"length":         "length",
	"tool_calls":     "tool_call",
	"function_call":  "tool_call",
	"content_filter": "content_filter",
}

func chatFinishReason(raw any, hasTool bool, rec *unmappedRecorder) string {
	if hasTool {
		return "tool_call"
	}
	if raw == nil || raw == "" {
		return "stop"
	}
	if mapped, ok := chatFinishReasonMap[pyStr(raw)]; ok {
		return mapped
	}
	rec.record("choices[0].finish_reason", pyStr(raw))
	return "stop"
}

func usageFromChat(usageData jmap) Usage {
	promptDetails := objAt(usageData, "prompt_tokens_details")
	completionDetails := objAt(usageData, "completion_tokens_details")
	return Usage{
		InputTokens:       i64ptr(intOrZero(usageData["prompt_tokens"])),
		OutputTokens:      i64ptr(intOrZero(usageData["completion_tokens"])),
		TotalTokens:       intOrNilTok(usageData["total_tokens"]),
		ReasoningTokens:   intOrNilTok(completionDetails["reasoning_tokens"]),
		CacheReadTokens:   intOrNilTok(promptDetails["cached_tokens"]),
		InputAudioTokens:  intOrNilTok(promptDetails["audio_tokens"]),
		OutputAudioTokens: intOrNilTok(completionDetails["audio_tokens"]),
	}
}

func parseOpenAIChatResponse(req Request, data jmap, rec *unmappedRecorder) (Response, error) {
	if respErr, ok := data["error"].(jmap); ok {
		return Response{}, openAIResponseError(strOrEmpty(respErr["code"]), strOrEmpty(respErr["message"]))
	}

	var parts []Part
	choices := listAt(data, "choices")
	choice := jmap{}
	if len(choices) > 0 {
		if c, ok := choices[0].(jmap); ok {
			choice = c
		} else {
			rec.record("choices[0]", pyTypeName(choices[0]))
		}
	}
	message := objAt(choice, "message")

	reasoningText := message["reasoning_content"]
	if !truthy(reasoningText) {
		reasoningText = message["reasoning"]
	}
	if truthy(reasoningText) {
		parts = append(parts, ThinkingPart{Text: pyStr(reasoningText)})
	}

	switch content := message["content"].(type) {
	case string:
		if content != "" {
			parts = append(parts, TextPart{Text: content})
		}
	case []any:
		for j, rawItem := range content {
			if item, ok := rawItem.(jmap); ok && item["type"] == "text" {
				parts = append(parts, TextPart{Text: strOrEmpty(item["text"])})
			} else if item, ok := rawItem.(jmap); ok {
				rec.record(fmt.Sprintf("choices[0].message.content[%d]", j), strOrEmpty(item["type"]))
			} else {
				rec.record(fmt.Sprintf("choices[0].message.content[%d]", j), pyTypeName(rawItem))
			}
		}
	case nil:
		// absent or null: nothing
	default:
		rec.record("choices[0].message.content", pyTypeName(content))
	}

	if truthy(message["refusal"]) {
		parts = append(parts, RefusalPart{Text: pyStr(message["refusal"])})
	}

	for k, rawCall := range listAt(message, "tool_calls") {
		call, ok := rawCall.(jmap)
		if !ok {
			rec.record(fmt.Sprintf("choices[0].message.tool_calls[%d]", k), pyTypeName(rawCall))
			continue
		}
		callType := call["type"]
		if !truthy(callType) {
			callType = "function"
		}
		if callType != "function" {
			rec.record(fmt.Sprintf("choices[0].message.tool_calls[%d]", k), pyStr(callType))
			continue
		}
		function := objAt(call, "function")
		id := strOrEmpty(call["id"])
		if id == "" {
			id = fmt.Sprintf("call_%d", len(parts))
		}
		name := strOrEmpty(function["name"])
		if name == "" {
			name = "tool"
		}
		parts = append(parts, ToolCallPart{ID: id, Name: name, Input: parseJSONObject(function["arguments"])})
	}

	if len(parts) == 0 {
		parts = []Part{TextPart{Text: ""}} // MAP-2
	}

	var id *string
	if truthy(data["id"]) {
		s := pyStr(data["id"])
		id = &s
	}
	model := strOrEmpty(data["model"])
	if model == "" {
		model = req.Model
	}
	return Response{
		ID:           id,
		Model:        model,
		Message:      Message{Role: "assistant", Parts: parts},
		FinishReason: chatFinishReason(choice["finish_reason"], hasToolCall(parts), rec),
		Usage:        usageFromChat(objAt(data, "usage")),
	}, nil
}

// ─── Gemini ──────────────────────────────────────────────────────────

var geminiProviderExecutedPartKeys = []string{"executableCode", "codeExecutionResult"}

var geminiCandidateFinishErrors = map[string]bool{
	"SAFETY": true, "RECITATION": true, "LANGUAGE": true, "BLOCKLIST": true,
	"PROHIBITED_CONTENT": true, "SPII": true, "MALFORMED_FUNCTION_CALL": true,
	"IMAGE_SAFETY": true, "IMAGE_PROHIBITED_CONTENT": true, "IMAGE_OTHER": true,
	"NO_IMAGE": true,
}

func geminiFinishReason(reason any, hasTool bool) string {
	if hasTool {
		return "tool_call"
	}
	r := strings.ToUpper(strOrEmpty(reason))
	if r == "MAX_TOKENS" {
		return "length"
	}
	switch r {
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	}
	return "stop"
}

func geminiInbandError(data jmap) LM15Error {
	if promptFeedback, ok := data["promptFeedback"].(jmap); ok {
		blockReason := strOrEmpty(promptFeedback["blockReason"])
		if blockReason != "" && blockReason != "BLOCK_REASON_UNSPECIFIED" {
			return ErrorForCode("invalid_request", "Prompt blocked: "+blockReason)
		}
	}
	candidates := listAt(data, "candidates")
	if len(candidates) > 0 {
		if candidate, ok := candidates[0].(jmap); ok {
			finishReason := strOrEmpty(candidate["finishReason"])
			if geminiCandidateFinishErrors[finishReason] {
				msg := strOrEmpty(candidate["finishMessage"])
				if msg == "" {
					msg = "Candidate blocked: " + finishReason
				}
				return ErrorForCode("invalid_request", msg)
			}
		}
	}
	return nil
}

func geminiThoughtContinuation(sig any) []ContinuationState {
	if sig == nil {
		return nil
	}
	return []ContinuationState{{
		Provider: "gemini",
		Kind:     "thought_signature",
		Data:     jmap{"value": pyStr(sig)},
	}}
}

func parseGeminiCandidateParts(partsPayload []any, rec *unmappedRecorder, pathPrefix string) []Part {
	var parts []Part
	for i, raw := range partsPayload {
		part, ok := raw.(jmap)
		if !ok {
			rec.record(fmt.Sprintf("%s[%d]", pathPrefix, i), pyTypeName(raw))
			continue
		}
		_, hasThought := part["thought"]
		if hasThought && truthy(part["thought"]) && truthy(part["text"]) {
			var sig any
			if v, ok := part["thoughtSignature"]; ok && v != nil {
				sig = v
			}
			parts = append(parts, ThinkingPart{Text: strOrEmpty(part["text"]), Continuation: geminiThoughtContinuation(sig)})
		} else if _, hasText := part["text"]; hasText {
			parts = append(parts, TextPart{Text: strOrEmpty(part["text"])})
		} else if fc, ok := part["functionCall"].(jmap); ok {
			var sig any
			if v, ok := part["thoughtSignature"]; ok && truthy(v) {
				sig = v
			} else if v, ok := fc["thoughtSignature"]; ok && v != nil {
				sig = v
			}
			id := strOrEmpty(fc["id"])
			if id == "" {
				id = fmt.Sprintf("fc_%d", len(parts))
			}
			name := strOrEmpty(fc["name"])
			if name == "" {
				name = "tool"
			}
			input, _ := fc["args"].(jmap)
			if input == nil {
				input = jmap{}
			}
			parts = append(parts, ToolCallPart{ID: id, Name: name, Input: input, Continuation: geminiThoughtContinuation(sig)})
		} else if inline, ok := part["inlineData"].(jmap); ok {
			mime := strOrEmpty(inline["mimeType"])
			if mime == "" {
				mime = "application/octet-stream"
			}
			data := strOrEmpty(inline["data"])
			if data == "" {
				continue
			}
			mp := mediaPart{MediaType: mime, Data: &data}
			switch {
			case strings.HasPrefix(mime, "image/"):
				parts = append(parts, ImagePart{mediaPart: mp})
			case strings.HasPrefix(mime, "audio/"):
				parts = append(parts, AudioPart{mediaPart: mp})
			default:
				parts = append(parts, DocumentPart{mediaPart: mp})
			}
		} else if fd, ok := part["fileData"].(jmap); ok {
			uri := strOrEmpty(fd["fileUri"])
			mime := strOrEmpty(fd["mimeType"])
			if mime == "" {
				mime = "application/octet-stream"
			}
			if uri == "" {
				continue
			}
			mp := mediaPart{MediaType: mime, URL: &uri}
			switch {
			case strings.HasPrefix(mime, "image/"):
				parts = append(parts, ImagePart{mediaPart: mp})
			case strings.HasPrefix(mime, "audio/"):
				parts = append(parts, AudioPart{mediaPart: mp})
			default:
				parts = append(parts, DocumentPart{mediaPart: mp})
			}
		} else if hasAnyKey(part, geminiProviderExecutedPartKeys) {
			continue
		} else {
			keys := make([]string, 0, len(part))
			for k := range part {
				keys = append(keys, k)
			}
			sortStrings(keys)
			typ := strings.Join(keys, "+")
			if typ == "" {
				typ = "<empty>"
			}
			rec.record(fmt.Sprintf("%s[%d]", pathPrefix, i), typ)
		}
	}
	return parts
}

func hasAnyKey(m jmap, keys []string) bool {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func geminiSegmentText(segment jmap, fullText string) *string {
	if t, ok := segment["text"].(string); ok && t != "" {
		return &t
	}
	start, okS := intIndex(segment["startIndex"])
	end, okE := intIndex(segment["endIndex"])
	if okS && okE {
		if s, ok := runeSlice(fullText, start, end); ok {
			return &s
		}
	}
	return nil
}

func geminiCitations(candidate jmap, fullText string) []Part {
	grounding, ok := candidate["groundingMetadata"].(jmap)
	if !ok {
		return nil
	}
	chunks := listAt(grounding, "groundingChunks")
	supports, ok := grounding["groundingSupports"].([]any)
	if !ok {
		if v, present := grounding["groundingSupports"]; present && truthy(v) {
			return nil // non-list supports → bail like the reference
		}
		supports = nil
	}
	var citations []Part
	seen := map[string]bool{}
	strKey := func(s *string) string {
		if s == nil {
			return "\x00nil"
		}
		return "\x01" + *s
	}
	for _, rawSupport := range supports {
		support, ok := rawSupport.(jmap)
		if !ok {
			continue
		}
		segment := objAt(support, "segment")
		citedText := geminiSegmentText(segment, fullText)
		indices, ok := support["groundingChunkIndices"].([]any)
		if !ok {
			continue
		}
		for _, index := range indices {
			chunk := jmap{}
			if idx, ok := intIndex(index); ok && idx >= 0 && idx < int64(len(chunks)) {
				if c, ok := chunks[idx].(jmap); ok {
					chunk = c
				}
			}
			source := jmap{}
			for _, k := range []string{"web", "retrievedContext", "googleSearch"} {
				if truthy(chunk[k]) {
					if s, ok := chunk[k].(jmap); ok {
						source = s
					}
					break
				}
			}
			urlV := source["uri"]
			if !truthy(urlV) {
				urlV = source["url"]
			}
			titleV := source["title"]
			if !truthy(titleV) {
				titleV = source["name"]
			}
			var urlS, titleS *string
			if truthy(urlV) {
				s := pyStr(urlV)
				urlS = &s
			}
			if truthy(titleV) {
				s := pyStr(titleV)
				titleS = &s
			}
			key := strKey(urlS) + strKey(titleS) + strKey(citedText)
			if seen[key] || (urlS == nil && titleS == nil && citedText == nil) {
				continue
			}
			seen[key] = true
			citations = append(citations, CitationPart{URL: urlS, Title: titleS, Text: citedText})
		}
	}
	return citations
}

func parseGeminiResponse(req Request, data jmap, rec *unmappedRecorder) (Response, error) {
	if inband := geminiInbandError(data); inband != nil {
		return Response{}, inband
	}
	candidate := jmap{}
	if candidates := listAt(data, "candidates"); len(candidates) > 0 {
		if c, ok := candidates[0].(jmap); ok {
			candidate = c
		}
	}
	content := objAt(candidate, "content")
	parts := parseGeminiCandidateParts(listAt(content, "parts"), rec, "candidates[0].content.parts")
	var fullText strings.Builder
	for _, p := range parts {
		if tp, ok := p.(TextPart); ok {
			fullText.WriteString(tp.Text)
		}
	}
	parts = append(parts, geminiCitations(candidate, fullText.String())...)
	if len(parts) == 0 {
		parts = []Part{TextPart{Text: ""}} // MAP-2
	}

	usagePayload := objAt(data, "usageMetadata")
	outputV, ok := usagePayload["candidatesTokenCount"]
	if !ok {
		outputV = usagePayload["responseTokenCount"]
	}
	usage := Usage{
		InputTokens:     i64ptr(intOrZero(usagePayload["promptTokenCount"])),
		OutputTokens:    i64ptr(intOrZero(outputV)),
		TotalTokens:     intOrNilTok(usagePayload["totalTokenCount"]),
		CacheReadTokens: intOrNilTok(usagePayload["cachedContentTokenCount"]),
		ReasoningTokens: intOrNilTok(usagePayload["thoughtsTokenCount"]),
	}

	var msgCont []ContinuationState
	var id *string
	if truthy(data["responseId"]) {
		s := pyStr(data["responseId"])
		id = &s
		msgCont = []ContinuationState{{Provider: "gemini", Kind: "response_id", Data: jmap{"id": s}}}
	}
	return Response{
		ID:           id,
		Model:        req.Model,
		Message:      Message{Role: "assistant", Parts: parts, Continuation: msgCont},
		FinishReason: geminiFinishReason(candidate["finishReason"], hasToolCall(parts)),
		Usage:        usage,
	}, nil
}

// ─── Entry points ────────────────────────────────────────────────────

// ParseResponse maps a provider HTTP response body into the canonical
// Response. The returned recorder entries are the _lm15_unmapped canary.
func ParseResponse(provider string, req Request, status int, body []byte) (Response, []jmap, error) {
	_ = status // parse_response operates on success bodies; status is informational
	dec := json.NewDecoder(strings.NewReader(string(body)))
	dec.UseNumber()
	var raw any
	if err := dec.Decode(&raw); err != nil {
		return Response{}, nil, valueErr("response body is not valid JSON: %v", err)
	}
	data, ok := raw.(jmap)
	if !ok {
		return Response{}, nil, valueErr("response body must be a JSON object")
	}
	rec := &unmappedRecorder{}
	var resp Response
	var err error
	switch provider {
	case "anthropic":
		resp = parseAnthropicResponse(req, data, rec)
	case "openai":
		resp, err = parseOpenAIResponse(req, data, rec)
	case "openai_chat":
		resp, err = parseOpenAIChatResponse(req, data, rec)
	case "gemini":
		resp, err = parseGeminiResponse(req, data, rec)
	default:
		return Response{}, nil, valueErr("unknown provider: %s", provider)
	}
	if err != nil {
		return Response{}, nil, err
	}
	resp.ProviderData = data
	return resp, rec.entries, nil
}

// ParseResponseMap is the vet-shim entry: canonical Response JSON plus the
// unmapped canary per harness/PROTOCOL.md.
func ParseResponseMap(provider string, canonical jmap, status int, bodyB64 string) (jmap, error) {
	req, err := requestFromMap(canonical)
	if err != nil {
		return nil, err
	}
	body, err := base64.StdEncoding.DecodeString(bodyB64)
	if err != nil {
		return nil, valueErr("body_b64 is not valid base64: %v", err)
	}
	resp, unmapped, err := ParseResponse(provider, req, status, body)
	if err != nil {
		return nil, err
	}
	out := jmap{"canonical_response": resp.toMap()}
	if len(unmapped) > 0 {
		entries := make([]any, len(unmapped))
		for i, e := range unmapped {
			entries[i] = e
		}
		out["unmapped"] = entries
	}
	return out, nil
}
