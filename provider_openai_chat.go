package lm15

import (
	"strconv"
	"strings"
	"time"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

// OpenAIChatLM is the OpenAI Chat Completions dialect: OpenAI's legacy
// endpoint and every OpenAI-compatible server (ollama, Groq, OpenRouter,
// vLLM, SGLang, DeepSeek, xAI, ...). Server quirks are OpenAIChatCompat
// presets; xAI's provider facts branch on the bound policy.
type OpenAIChatLM struct {
	lmCore
	compatPartial OpenAIChatCompat
	resolved      ResolvedOpenAIChatCompat
}

// XaiLM is a name: OpenAIChatLM bound to lm15.Xai with the xai preset.
type XaiLM = OpenAIChatLM

const openaiChatDefaultBaseURL = "https://api.openai.com/v1"

var groqBuiltinMap = map[string]string{"web_search": "browser_search", "code_execution": "code_interpreter"}

var chatFinishReasonMap = map[string]string{
	"stop": FinishStop, "length": FinishLength, "tool_calls": FinishToolCall, "function_call": FinishToolCall, "content_filter": FinishContentFilter,
}

// NewOpenAIChatLM constructs the Chat Completions dialect (default policy OpenAIChatAPI).
func NewOpenAIChatLM(opts ...Option) (*OpenAIChatLM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	lm := &OpenAIChatLM{}
	if err := lm.bindAccess(lm, OpenAIChatAPI, o, openaiChatDefaultBaseURL); err != nil {
		return nil, err
	}
	compat := o.compatPreset
	if compat == "" && o.chatCompat == nil {
		compat = lm.registryCompat()
	}
	switch {
	case compat != "":
		preset, err := OpenAIChatPreset(compat)
		if err != nil {
			return nil, err
		}
		lm.compatPartial = preset
		if lm.baseURL == openaiChatDefaultBaseURL {
			url, err := PresetBaseURL(OpenAIChatPresetBaseURLs, compat, "Chat Completions", "openai")
			if err != nil {
				return nil, err
			}
			lm.baseURL = url
		}
	case o.chatCompat != nil:
		if err := o.chatCompat.Validate(); err != nil {
			return nil, err
		}
		lm.compatPartial = *o.chatCompat
	}
	lm.resolved = ResolveOpenAIChatCompat(lm.compatPartial)
	return lm, nil
}

// NewXaiLM constructs the xAI adapter (Chat Completions dialect, xai preset,
// oauth-unless-explicit credential policy).
func NewXaiLM(opts ...Option) (*OpenAIChatLM, error) {
	return NewOpenAIChatLM(append([]Option{WithAccess(Xai), WithCompatPreset("xai"), WithBaseURL(DefaultXaiBaseURL)}, opts...)...)
}

func (l *OpenAIChatLM) isXai() bool { return l.provider == "xai" }

func (l *OpenAIChatLM) headers() [][2]string {
	return append([][2]string{{"Content-Type", "application/json"}}, l.access.Headers...)
}

func (l *OpenAIChatLM) compatFor(model string) ResolvedOpenAIChatCompat {
	if len(l.compatPartial.ModelOverrides) == 0 {
		return l.resolved
	}
	return ResolveOpenAIChatCompat(l.compatPartial.ForModel(model))
}

func (l *OpenAIChatLM) normalizeError(status int, body string) *Error {
	if l.isXai() {
		// xAI's own envelope is {"code": str, "error": str}: refold it.
		if data, err := DecodeJSON([]byte(body)); err == nil {
			if obj := wireObj(data); obj != nil {
				if msg, ok := obj["error"].(string); ok {
					body = jsonRaw(JSONObject{"error": JSONObject{"message": msg, "code": obj["code"]}})
				}
			}
		}
	}
	return openaiNormalizeError(&l.lmCore, status, body)
}

// ─── Models ──────────────────────────────────────────────────────────

func (l *OpenAIChatLM) modelsRequest() (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/models", headers: l.headers(), readTimeout: 30 * time.Second})
}

func (l *OpenAIChatLM) modelsFromBody(body string) ([]ModelInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	return modelInfosFromEntries(data["data"], l.provider, "openai_chat", func(e map[string]any) string { return stringOnly(e["id"]) }), nil
}

// ─── Request serialization ───────────────────────────────────────────

func chatImageBlock(img ImagePart, provider string) (JSONObject, error) {
	if img.FileID != "" {
		return nil, UnsupportedFeatureErrorf(provider, "%s: an image addressed by file_id cannot be sent on the Chat Completions wire (no file reference form); pass a URL or inline data", provider)
	}
	src := img.URL
	if src == "" {
		uri, err := mediaDataURI(img.Media)
		if err != nil {
			return nil, err
		}
		src = uri
	}
	payload := JSONObject{"url": src}
	if img.Detail != "" {
		payload["detail"] = img.Detail
	}
	return JSONObject{"type": "image_url", "image_url": payload}, nil
}

// chatContentParts maps a non-assistant message to chat content: a bare
// string for one text part, else an array. A part with no slot RAISES.
func chatContentParts(msg Message, forceArray bool, provider string) (any, error) {
	var parts []Part
	for _, p := range msg.Parts {
		switch p.(type) {
		case ToolCallPart, ToolResultPart:
			continue
		}
		parts = append(parts, p)
	}
	if len(parts) == 1 && !forceArray {
		if t, ok := parts[0].(TextPart); ok {
			return t.Text, nil
		}
	}
	var out []any
	for _, p := range parts {
		switch x := p.(type) {
		case TextPart:
			out = append(out, JSONObject{"type": "text", "text": x.Text})
		case ImagePart:
			block, err := chatImageBlock(x, provider)
			if err != nil {
				return nil, err
			}
			out = append(out, block)
		case ThinkingPart:
			continue
		default:
			if IsMediaPart(p) {
				return nil, UnsupportedFeatureErrorf(provider, "%s: a %s part in a %s message has no slot on the Chat Completions wire (text and image_url only); the OpenAI Responses, Anthropic and Gemini dialects carry it (MAP-10)", provider, p.Type(), msg.Role)
			}
			text, err := partsToText([]Part{p}, provider, "")
			if err != nil {
				return nil, err
			}
			out = append(out, JSONObject{"type": "text", "text": text})
		}
	}
	if out == nil {
		out = []any{}
	}
	return out, nil
}

func toolRowContent(provider string, part ToolResultPart, policy string) (any, error) {
	if err := checkToolResultMedia(provider, part, policy, "a Chat Completions tool row"); err != nil {
		return nil, err
	}
	if !hasMediaParts(part.Content) {
		text, err := partsToText(part.Content, provider, "a Chat Completions tool row")
		if err != nil {
			return nil, err
		}
		return toolResultErrorText(part, text), nil
	}
	var blocks []JSONObject
	for _, p := range part.Content {
		if img, ok := p.(ImagePart); ok {
			block, err := chatImageBlock(img, provider)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		} else {
			text, err := partsToText([]Part{p}, provider, "")
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, JSONObject{"type": "text", "text": text})
		}
	}
	if part.IsError {
		found := false
		for _, b := range blocks {
			if b["type"] == "text" {
				b["text"] = "[error] " + wireStr(b["text"])
				found = true
				break
			}
		}
		if !found {
			blocks = append([]JSONObject{{"type": "text", "text": "[error]"}}, blocks...)
		}
	}
	return toAnyList(blocks, func(b JSONObject) any { return b }), nil
}

func responseFormatToChat(f JSONObject) JSONObject {
	if f["type"] == "json_object" {
		return JSONObject{"type": "json_object"}
	}
	name := wireStr(f["name"])
	if name == "" {
		name = "response"
	}
	inner := JSONObject{"name": name, "schema": f["schema"]}
	if strict, ok := f["strict"]; ok {
		inner["strict"] = strict
	}
	return JSONObject{"type": "json_schema", "json_schema": inner}
}

func (l *OpenAIChatLM) buildMessages(req *Request, compat ResolvedOpenAIChatCompat) ([]any, error) {
	var messages []any
	if req.System != nil {
		text, err := systemText(req.System, l.provider)
		if err != nil {
			return nil, err
		}
		if cacheStablePrefix(req, compat.CacheControl) {
			messages = append(messages, JSONObject{"role": compat.InstructionRole, "content": []any{JSONObject{"type": "text", "text": text, "prompt_cache_breakpoint": JSONObject{"mode": "explicit"}}}})
		} else {
			messages = append(messages, JSONObject{"role": compat.InstructionRole, "content": text})
		}
	}
	breakpoint := cacheBreakpointIndex(req, compat.CacheControl)
	for msgIndex, msg := range req.Messages {
		atBreakpoint := breakpoint != nil && *breakpoint == msgIndex
		if atBreakpoint && (msg.Role == RoleAssistant || msg.Role == RoleTool) {
			return nil, breakpointUnsupported(l.provider, msgIndex, msg.Role)
		}
		switch msg.Role {
		case RoleTool:
			for _, part := range msg.Parts {
				tr, ok := part.(ToolResultPart)
				if !ok {
					continue
				}
				content, err := toolRowContent(l.provider, tr, compat.ToolResultMedia)
				if err != nil {
					return nil, err
				}
				item := JSONObject{"role": "tool", "tool_call_id": tr.ID, "content": content}
				if compat.ToolResultName == "include" && tr.Name != "" {
					item["name"] = tr.Name
				}
				messages = append(messages, item)
			}
		case RoleAssistant:
			var textBits []string
			var thinkingBits []string
			var toolCalls []any
			for _, part := range msg.Parts {
				switch p := part.(type) {
				case TextPart:
					textBits = append(textBits, p.Text)
				case RefusalPart:
					if p.Text != "" {
						textBits = append(textBits, p.Text)
					}
				case ThinkingPart:
					if compat.ThinkingReplay == "as_text" && p.Text != "" {
						textBits = append(textBits, p.Text)
					}
					if p.Text != "" {
						thinkingBits = append(thinkingBits, p.Text)
					}
				case ToolCallPart:
					toolCalls = append(toolCalls, JSONObject{"id": p.ID, "type": "function", "function": JSONObject{"name": p.Name, "arguments": jsonRaw(p.Input)}})
				}
			}
			item := JSONObject{"role": "assistant"}
			if len(textBits) > 0 {
				item["content"] = strings.Join(textBits, "\n")
			} else {
				item["content"] = nil
			}
			if compat.ThinkingReplay == "native" {
				thinking := strings.Join(thinkingBits, "\n")
				if thinking != "" || compat.AssistantReasoningContent == "include_empty" {
					item["reasoning_content"] = thinking
				}
			}
			if len(toolCalls) > 0 {
				item["tool_calls"] = toolCalls
			}
			messages = append(messages, item)
		default:
			role := msg.Role
			if role == RoleDeveloper {
				role = compat.InstructionRole
			}
			content, err := chatContentParts(msg, atBreakpoint, l.provider)
			if err != nil {
				return nil, err
			}
			if atBreakpoint {
				list, ok := content.([]any)
				if !ok || len(list) == 0 {
					return nil, breakpointUnsupported(l.provider, msgIndex, msg.Role)
				}
				last, _ := list[len(list)-1].(JSONObject)
				if last["type"] != "text" {
					return nil, breakpointUnsupported(l.provider, msgIndex, msg.Role)
				}
				last["prompt_cache_breakpoint"] = JSONObject{"mode": "explicit"}
			}
			messages = append(messages, JSONObject{"role": role, "content": content})
		}
	}
	if messages == nil {
		messages = []any{}
	}
	return messages, nil
}

func (l *OpenAIChatLM) builtinToolPayload(tool BuiltinTool, compat ResolvedOpenAIChatCompat) (JSONObject, error) {
	if compat.BuiltinTools == "groq" {
		wireType, ok := groqBuiltinMap[tool.Name]
		if !ok {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: builtin tool %q has no Groq wire mapping — supported: %v", l.provider, tool.Name, sortStrings([]string{"code_execution", "web_search"}))
		}
		entry := JSONObject{"type": wireType}
		for k, v := range tool.Config {
			entry[k] = v
		}
		return entry, nil
	}
	return nil, UnsupportedFeatureErrorf(l.provider, "%s: builtin tool %q is not supported on this server — the Chat Completions wire carries function tools only, and unproven servers may silently ignore unknown tool types. Use compat='groq' for Groq's server-executed tools, or the OpenAI Responses / Anthropic / Gemini providers", l.provider, tool.Name)
}

func (l *OpenAIChatLM) toolChoicePayload(req *Request) (any, error) {
	tc := req.Config.ToolChoice
	if tc == nil {
		return nil, nil
	}
	if tc.EffectiveMode() == "none" {
		return "none", nil
	}
	if len(tc.Allowed) > 0 {
		var builtins []string
		entries := make([]Tool, 0, len(tc.Allowed))
		for _, name := range tc.Allowed {
			t := req.ToolByName(name)
			entries = append(entries, t)
			if _, ok := t.(BuiltinTool); ok {
				builtins = append(builtins, name)
			}
		}
		if len(builtins) > 0 {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: cannot force builtin tools %v — the Chat Completions wire has no hosted-tool tool_choice form (OpenAI Responses and Anthropic carry it)", l.provider, builtins)
		}
		if len(entries) == 1 && tc.EffectiveMode() == "required" {
			return JSONObject{"type": "function", "function": JSONObject{"name": entries[0].ToolName()}}, nil
		}
		var tools []any
		for _, t := range entries {
			tools = append(tools, JSONObject{"type": "function", "function": JSONObject{"name": t.ToolName()}})
		}
		return JSONObject{"type": "allowed_tools", "allowed_tools": JSONObject{"mode": tc.EffectiveMode(), "tools": tools}}, nil
	}
	if tc.EffectiveMode() == "required" {
		return "required", nil
	}
	return "auto", nil
}

func (l *OpenAIChatLM) xaiChecks(req *Request) error {
	cfg := req.Config
	if cfg.Reasoning != nil && cfg.Reasoning.IsOff() {
		return UnsupportedFeatureErrorf(l.provider, "xai: reasoning cannot be disabled — Grok reasoning models have no off switch, and xAI silently ignores disable fields on the wire. Omit the reasoning config, or pick a non-reasoning Grok model.")
	}
	if cfg.Logprobs != nil {
		return UnsupportedFeatureErrorf(l.provider, "xai: config.logprobs is not supported — grok-4.20 and newer silently ignore logprobs/top_logprobs on the wire (docs.x.ai, verified live 2026-09-01). OpenAI and Gemini carry logprobs.")
	}
	tc := cfg.ToolChoice
	if tc != nil && len(tc.Allowed) > 0 && !(len(tc.Allowed) == 1 && tc.EffectiveMode() == "required") {
		return UnsupportedFeatureErrorf(l.provider, "xai: tool_choice.allowed subsets are silently ignored by api.x.ai (verified live 2026-09-02); force a single tool with mode='required', or send only the allowed tools in Request.tools")
	}
	if tc != nil && tc.EffectiveMode() == "required" && len(cfg.ResponseFormat) > 0 {
		return UnsupportedFeatureErrorf(l.provider, "xai: a forced tool (mode='required') cannot be combined with response_format — api.x.ai returns JSON text and drops the call (verified live 2026-09-02)")
	}
	return nil
}

func (l *OpenAIChatLM) payload(req *Request, stream bool) (JSONObject, error) {
	if l.isXai() {
		if err := l.xaiChecks(req); err != nil {
			return nil, err
		}
	}
	compat := l.compatFor(req.Model)
	messages, err := l.buildMessages(req, compat)
	if err != nil {
		return nil, err
	}
	payload := JSONObject{"model": req.Model, "messages": messages}
	if stream {
		payload["stream"] = true
		if compat.StreamUsage == "include" {
			payload["stream_options"] = JSONObject{"include_usage": true}
		}
	}
	cfg := req.Config
	if cfg.MaxTokens != nil {
		payload[compat.MaxTokensField] = *cfg.MaxTokens
	}
	if cfg.Temperature != nil {
		payload["temperature"] = jsonFloat(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		payload["top_p"] = jsonFloat(*cfg.TopP)
	}
	if cfg.TopK != nil {
		return nil, UnsupportedFeatureErrorf(l.provider, "%s: config.top_k has no field on the Chat Completions wire; servers that accept top_k take it through extensions", l.provider)
	}
	if len(cfg.Stop) > 0 {
		payload["stop"] = toAnyList(cfg.Stop, func(s string) any { return s })
	}
	if cfg.Logprobs != nil {
		payload["logprobs"] = true
		if *cfg.Logprobs > 0 {
			payload["top_logprobs"] = *cfg.Logprobs
		}
	}
	if len(req.Tools) > 0 {
		var tools []any
		for _, t := range req.Tools {
			switch x := t.(type) {
			case FunctionTool:
				fn := JSONObject{"name": x.Name, "description": nilIfEmpty(x.Description), "parameters": x.EffectiveParameters()}
				if compat.StrictTools == "include" {
					fn["strict"] = false
				}
				tools = append(tools, JSONObject{"type": "function", "function": fn})
			case BuiltinTool:
				entry, err := l.builtinToolPayload(x, compat)
				if err != nil {
					return nil, err
				}
				tools = append(tools, entry)
			}
		}
		if len(tools) > 0 {
			payload["tools"] = tools
		}
	}
	toolChoice, err := l.toolChoicePayload(req)
	if err != nil {
		return nil, err
	}
	if toolChoice != nil {
		tc := cfg.ToolChoice
		if compat.ForcedToolChoice == "reject" && (tc.EffectiveMode() != "auto" || len(tc.Allowed) > 0) {
			allowed := ""
			if len(tc.Allowed) > 0 {
				allowed = " allowed=" + fmtList(tc.Allowed)
			}
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: tool_choice mode=%q%s is silently ignored by this server (only 'auto' is honoured); omit tool_choice, or send only the tools you want callable", l.provider, tc.EffectiveMode(), allowed)
		}
		payload["tool_choice"] = toolChoice
	}
	if cfg.ToolChoice != nil && cfg.ToolChoice.Parallel != nil {
		payload["parallel_tool_calls"] = *cfg.ToolChoice.Parallel
	}
	if len(cfg.ResponseFormat) > 0 {
		if compat.JSONSchema == "reject" && cfg.ResponseFormat["type"] != "json_object" {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: response_format type %q is silently ignored by this server; use {'type': 'json_object'} and describe the shape in the prompt", l.provider, wireStr(cfg.ResponseFormat["type"]))
		}
		payload["response_format"] = responseFormatToChat(cfg.ResponseFormat)
	}
	if r := cfg.Reasoning; r != nil {
		if compat.ThinkingFormat == "none" {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: reasoning.effort=%q has no field on this server (compat thinking_format='none'); omit config.reasoning, or pass the server's own knob through extensions", l.provider, r.Effort)
		}
		if !r.IsOff() {
			if r.ThinkingBudget != nil {
				return nil, UnsupportedFeatureErrorf(l.provider, "%s: reasoning.thinking_budget is not supported — the Chat Completions wire has no thinking token budget; use effort", l.provider)
			}
			if r.Summary == "concise" || r.Summary == "detailed" {
				return nil, UnsupportedFeatureErrorf(l.provider, "%s: reasoning.summary=%q is an OpenAI Responses detail level; the Chat Completions wire has none (use 'auto')", l.provider, r.Summary)
			}
			if compat.ReasoningEfforts != nil && !inVocab(r.Effort, compat.ReasoningEfforts) {
				return nil, UnsupportedFeatureErrorf(l.provider, "%s: reasoning.effort=%q has no level on this server (it accepts %s) and would be accepted silently", l.provider, r.Effort, strings.Join(compat.ReasoningEfforts, ", "))
			}
			if compat.BuiltinTools == "groq" && r.Summary == "auto" {
				payload["reasoning_format"] = "parsed"
			}
			switch compat.ThinkingFormat {
			case "reasoning_effort", "kimi":
				payload["reasoning_effort"] = r.Effort
			case "openrouter":
				payload["reasoning"] = JSONObject{"effort": r.Effort}
			case "deepseek":
				payload["thinking"] = JSONObject{"type": "enabled"}
				payload["reasoning_effort"] = r.Effort
			case "qwen":
				payload["enable_thinking"] = true
			case "qwen_chat_template":
				payload["chat_template_kwargs"] = JSONObject{"enable_thinking": true, "preserve_thinking": true}
			}
		} else {
			switch compat.ThinkingFormat {
			case "reasoning_effort":
				payload["reasoning_effort"] = "none"
			case "openrouter":
				payload["reasoning"] = JSONObject{"enabled": false}
			case "deepseek", "kimi":
				payload["thinking"] = JSONObject{"type": "disabled"}
			case "qwen":
				payload["enable_thinking"] = false
			case "qwen_chat_template":
				payload["chat_template_kwargs"] = JSONObject{"enable_thinking": false}
			}
		}
	}
	if err := cacheCommonPayload(req, payload, compat.CacheControl, l.provider); err != nil {
		return nil, err
	}
	if compat.Routing != nil {
		payload["provider"] = compat.Routing
	}
	if cfg.ServiceTier != "" {
		payload["service_tier"] = cfg.ServiceTier
	}
	if cfg.UserID != "" {
		payload[compat.UserField] = cfg.UserID
	}
	if cfg.Store != nil {
		payload["store"] = *cfg.Store
	}
	for k, v := range cfg.Extensions {
		switch k {
		case "prompt_caching", "cache", "compat", "openai_compat", "openai_chat_compat":
			continue
		}
		payload[k] = v
	}
	return payload, nil
}

func (l *OpenAIChatLM) buildRequest(req *Request, stream bool) (*TransportRequest, error) {
	payload, err := l.payload(req, stream)
	if err != nil {
		return nil, err
	}
	timeout := 60 * time.Second
	if stream {
		timeout = 120 * time.Second
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/chat/completions", endpoint: "chat/completions", stream: stream, model: req.Model, headers: l.headers(), payload: payload, readTimeout: timeout})
}

// ─── Response parsing ────────────────────────────────────────────────

func chatFinishReason(raw any, hasToolCall bool, unmapped *[]JSONObject, path string) string {
	if hasToolCall {
		return FinishToolCall
	}
	if raw == nil || wireStr(raw) == "" {
		return FinishStop
	}
	if mapped, ok := chatFinishReasonMap[wireStr(raw)]; ok {
		return mapped
	}
	recordUnmapped(unmapped, path+".finish_reason", raw)
	return FinishStop
}

func usageFromChat(u JSONObject) Usage {
	prompt := wireObj(u["prompt_tokens_details"])
	completion := wireObj(u["completion_tokens_details"])
	return Usage{
		InputTokens:       wireIntPtr(u["prompt_tokens"]),
		OutputTokens:      wireIntPtr(u["completion_tokens"]),
		TotalTokens:       wireIntPtr(u["total_tokens"]),
		ReasoningTokens:   wireIntPtr(completion["reasoning_tokens"]),
		CacheReadTokens:   wireIntPtr(prompt["cached_tokens"]),
		CacheWriteTokens:  wireIntPtr(prompt["cache_write_tokens"]),
		InputAudioTokens:  wireIntPtr(prompt["audio_tokens"]),
		OutputAudioTokens: wireIntPtr(completion["audio_tokens"]),
	}.Normalize()
}

// responseFromChatBody is the one Chat Completions response reader.
func responseFromChatBody(provider string, data JSONObject, model string, choice *int, onError func(code, message string) *Error) (*Response, error) {
	if e := wireObj(data["error"]); e != nil {
		return nil, onError(wireStr(e["code"]), firstStr(e["message"], mustJSONString(e)))
	}
	var unmapped []JSONObject
	choices := wireList(data["choices"])
	if _, isList := data["choices"].([]any); !isList && data["choices"] != nil {
		return nil, typeErrorf("choices must be an array")
	}
	index := 0
	if choice == nil {
		if len(choices) > 1 {
			return nil, UnsupportedFeatureErrorf(provider, "%s: the body carries %d choices; a canonical Response is one message — name the choice to read (choice=i) and read each one, or send no n", provider, len(choices))
		}
	} else {
		if *choice < 0 || *choice >= len(choices) {
			return nil, valueErrorf("choice=%d but the body carries %d choice(s)", *choice, len(choices))
		}
		index = *choice
	}
	path := "choices[" + strconv.Itoa(index) + "]"
	var chosen JSONObject
	if len(choices) > 0 {
		chosen = wireObj(choices[index])
		if chosen == nil {
			recordUnmapped(&unmapped, path, jsonTypeName(choices[index]))
		}
	}
	message := wireObj(chosen["message"])
	var parts []Part
	if reasoning := firstStr(message["reasoning_content"], message["reasoning"]); reasoning != "" {
		parts = append(parts, ThinkingPart{Text: reasoning})
	}
	switch content := message["content"].(type) {
	case string:
		if content != "" {
			parts = append(parts, TextPart{Text: content})
		}
	case []any:
		for i, item := range content {
			obj := wireObj(item)
			if obj != nil && wireStr(obj["type"]) == "text" {
				parts = append(parts, TextPart{Text: wireStr(obj["text"])})
			} else if obj != nil {
				recordUnmapped(&unmapped, path+".message.content["+strconv.Itoa(i)+"]", obj["type"])
			} else {
				recordUnmapped(&unmapped, path+".message.content["+strconv.Itoa(i)+"]", jsonTypeName(item))
			}
		}
	case nil:
	default:
		recordUnmapped(&unmapped, path+".message.content", jsonTypeName(content))
	}
	if refusal := wireStr(message["refusal"]); refusal != "" && message["refusal"] != nil {
		parts = append(parts, RefusalPart{Text: refusal})
	}
	for i, rawCall := range wireList(message["tool_calls"]) {
		call := wireObj(rawCall)
		callPath := path + ".message.tool_calls[" + strconv.Itoa(i) + "]"
		if call == nil {
			recordUnmapped(&unmapped, callPath, jsonTypeName(rawCall))
			continue
		}
		callType := wireStr(call["type"])
		if callType == "" {
			callType = "function"
		}
		if callType != "function" {
			recordUnmapped(&unmapped, callPath, callType)
			continue
		}
		fn := wireObj(call["function"])
		if !truthy(fn["name"]) {
			return nil, unnamedToolCallError(provider, callPath)
		}
		id := wireStr(call["id"])
		if id == "" || call["id"] == nil {
			id = "call_" + strconv.Itoa(len(parts))
		}
		parts = append(parts, ToolCallPart{ID: id, Name: wireStr(fn["name"]), Input: parseJSONObject(fn["arguments"])})
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{}}
	}
	usage := usageFromChat(wireObj(data["usage"]))
	logprobs := openaiTokenLogprobs(wireObj(chosen["logprobs"])["content"])
	resolvedModel := wireStr(data["model"])
	if resolvedModel == "" || data["model"] == nil {
		resolvedModel = model
	}
	if resolvedModel == "" {
		return nil, valueErrorf("the body carries no model; pass model=")
	}
	return &Response{
		ID:           wireStr(data["id"]),
		Model:        resolvedModel,
		Message:      Message{Role: RoleAssistant, Parts: parts},
		FinishReason: chatFinishReason(chosen["finish_reason"], hasToolCall(parts), &unmapped, path),
		Usage:        usage,
		Logprobs:     logprobs,
		ProviderData: attachUnmapped(data, unmapped),
	}, nil
}

func (l *OpenAIChatLM) parseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	data, err := resp.JSON()
	if err != nil {
		return nil, err
	}
	return responseFromChatBody(l.provider, data, req.Model, nil, func(code, message string) *Error { return openaiResponseError(&l.lmCore, code, message) })
}

func (l *OpenAIChatLM) responseFromOpenAIChat(body JSONObject, model string, choice *int) (*Response, error) {
	return responseFromChatBody(l.provider, body, model, choice, func(code, message string) *Error { return openaiResponseError(&l.lmCore, code, message) })
}

// ResponseFromOpenAIChat reads a Chat Completions response body into a
// canonical Response (MAP-12 rule 9), under the openai-chat provider name.
func ResponseFromOpenAIChat(body JSONObject, model string, choice *int) (*Response, error) {
	return responseFromChatBody("openai-chat", body, model, choice, func(code, message string) *Error {
		kind, ok := openaiResponseErrorCodeMap[code]
		if !ok {
			kind = KindServer
		}
		msg := message
		if msg == "" {
			msg = code
		}
		if msg == "" {
			msg = "provider error"
		}
		e := providerErrorf(kind, "openai-chat", nil, msg)
		e.ProviderCode = code
		return e
	})
}

func (l *OpenAIChatLM) parseStreamEvents(_ *Request, ev sse.Event) ([]StreamEvent, error) {
	if ev.Data == "" {
		return nil, nil
	}
	if ev.Data == "[DONE]" {
		return []StreamEvent{StreamEndEvent{}}, nil
	}
	raw, err := DecodeJSON([]byte(ev.Data))
	if err != nil {
		return nil, err
	}
	payload := wireObj(raw)
	if payload == nil {
		return nil, nil
	}
	if e := wireObj(payload["error"]); e != nil {
		code := firstStr(e["code"], e["type"])
		if code == "" {
			code = "provider"
		}
		return []StreamEvent{StreamErrorEvent{Error: openaiErrorDetail(code, wireStr(e["message"]))}}, nil
	}
	var events []StreamEvent
	choices := wireList(payload["choices"])
	var choice JSONObject
	if len(choices) > 0 {
		choice = wireObj(choices[0])
	}
	delta := wireObj(choice["delta"])
	if reasoning := firstStr(delta["reasoning_content"], delta["reasoning"]); reasoning != "" {
		events = append(events, StreamDeltaEvent{Delta: ThinkingDelta{Text: reasoning}})
	}
	if content, ok := delta["content"].(string); ok && content != "" {
		events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: content, Logprobs: openaiTokenLogprobs(wireObj(choice["logprobs"])["content"])}})
	}
	for _, rawCall := range wireList(delta["tool_calls"]) {
		call := wireObj(rawCall)
		if call == nil {
			continue
		}
		fn := wireObj(call["function"])
		events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{Input: wireStr(fn["arguments"]), PartIndex: wireInt(call["index"], 0), ID: wireStr(call["id"]), Name: wireStr(fn["name"])}})
	}
	usageData := wireObj(payload["usage"])
	if finishRaw := wireStr(choice["finish_reason"]); finishRaw != "" && choice["finish_reason"] != nil {
		finish, ok := chatFinishReasonMap[finishRaw]
		if !ok {
			finish = FinishStop
		}
		var usage *Usage
		if usageData != nil {
			u := usageFromChat(usageData)
			usage = &u
		}
		events = append(events, StreamEndEvent{FinishReason: finish, Usage: usage, ProviderData: payload})
	} else if usageData != nil {
		u := usageFromChat(usageData)
		events = append(events, StreamEndEvent{Usage: &u, ProviderData: payload})
	}
	return events, nil
}

// ─── xAI: image and video generation (provider facts) ────────────────

func xaiImageInput(part ImagePart) (JSONObject, error) {
	if part.URL != "" {
		return JSONObject{"url": part.URL}, nil
	}
	if part.FileID != "" {
		return JSONObject{"file_id": part.FileID}, nil
	}
	uri, err := mediaDataURI(part.Media)
	if err != nil {
		return nil, UnsupportedFeatureErrorf("xai", "xai: input image carries no content")
	}
	return JSONObject{"url": uri}, nil
}

func (l *OpenAIChatLM) imageGenerateRequest(req *ImageGenerationRequest) (*TransportRequest, error) {
	if !l.isXai() {
		return nil, l.unsupported("image generation")
	}
	base := strings.TrimRight(l.baseURL, "/")
	payload := JSONObject{"model": req.Model, "prompt": req.Prompt}
	for k, v := range req.Extensions {
		payload[k] = v
	}
	if req.Size != "" {
		return nil, UnsupportedFeatureErrorf(l.provider, "xai: size has no wire slot; use extensions for xAI's quality/resolution fields")
	}
	if len(req.Images) == 0 {
		return l.emit(emitSpec{method: "POST", url: base + "/images/generations", headers: l.headers(), payload: payload, readTimeout: 300 * time.Second})
	}
	if len(req.Images) > 1 {
		return nil, UnsupportedFeatureErrorf(l.provider, "xai: image edits take exactly one input image; the wire has no slot for more")
	}
	img, err := xaiImageInput(req.Images[0])
	if err != nil {
		return nil, err
	}
	payload["image"] = img
	return l.emit(emitSpec{method: "POST", url: base + "/images/edits", headers: l.headers(), payload: payload, readTimeout: 300 * time.Second})
}

func (l *OpenAIChatLM) imageGenerationFromResponse(_ *ImageGenerationRequest, resp *HTTPResponse) (ImageGenerationResponse, error) {
	if !l.isXai() {
		return ImageGenerationResponse{}, l.unsupported("image generation")
	}
	data, err := resp.JSON()
	if err != nil {
		return ImageGenerationResponse{}, err
	}
	var images []ImagePart
	for _, e := range wireList(data["data"]) {
		item := wireObj(e)
		if item == nil {
			continue
		}
		mediaType := stringOnly(item["mime_type"])
		if mediaType == "" {
			mediaType = "application/octet-stream"
		}
		if b64 := stringOnly(item["b64_json"]); b64 != "" {
			images = append(images, ImagePart{Media: Media{MediaType: mediaType, Data: b64}})
		} else if url := stringOnly(item["url"]); url != "" {
			images = append(images, ImagePart{Media: Media{MediaType: mediaType, URL: url}})
		}
	}
	if len(images) == 0 {
		return ImageGenerationResponse{}, l.providerError(KindProvider, "xai: image response carries no images", 0, "", "")
	}
	return ImageGenerationResponse{Images: images, ProviderData: data}, nil
}

var xaiVideoStatusMap = map[string]string{"pending": "running", "done": "completed", "failed": "failed"}

func (l *OpenAIChatLM) videoSubmitRequest(req *VideoGenerationRequest) (*TransportRequest, error) {
	if !l.isXai() {
		return nil, l.unsupported("video generation")
	}
	if req.Seconds != nil {
		return nil, UnsupportedFeatureErrorf(l.provider, "xai: video duration has no wire slot")
	}
	if len(req.Images) > 0 {
		return nil, UnsupportedFeatureErrorf(l.provider, "xai: video input images are not mapped yet; use extensions until the mapping is live-receipted")
	}
	payload := JSONObject{"model": req.Model, "prompt": req.Prompt}
	for k, v := range req.Extensions {
		payload[k] = v
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/videos/generations", headers: l.headers(), payload: payload, readTimeout: 120 * time.Second})
}

func (l *OpenAIChatLM) videoJobFromBody(body string, videoID string) (VideoJobInfo, error) {
	if !l.isXai() {
		return VideoJobInfo{}, l.unsupported("video generation")
	}
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return VideoJobInfo{}, err
	}
	if requestID := stringOnly(data["request_id"]); requestID != "" {
		return VideoJobInfo{ID: requestID, Status: "queued", ProviderData: data}, nil
	}
	if videoID == "" {
		return VideoJobInfo{}, l.providerError(KindProvider, "xai: video body carries no request_id", 0, "", "")
	}
	wireStatus := wireStr(data["status"])
	status, ok := xaiVideoStatusMap[wireStatus]
	if !ok {
		return VideoJobInfo{}, l.providerError(KindProvider, "xai: unknown video status "+strconv.Quote(wireStatus), 0, "", "")
	}
	var progress *int
	if _, isBool := data["progress"].(bool); !isBool {
		if f, err := jsonFloat64(data["progress"], ""); err == nil {
			p := int(f)
			progress = &p
		}
	}
	return VideoJobInfo{ID: videoID, Status: status, Progress: progress, Model: stringOnly(data["model"]), ProviderData: data}, nil
}

func (l *OpenAIChatLM) videoStatusRequest(videoID string) (*TransportRequest, error) {
	if !l.isXai() {
		return nil, l.unsupported("video generation")
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/videos/" + pathID(videoID, false), headers: l.headers(), readTimeout: 60 * time.Second})
}

func (l *OpenAIChatLM) videoListRequest(int, string) (*TransportRequest, error) {
	if !l.isXai() {
		return nil, l.unsupported("video generation")
	}
	return nil, UnsupportedFeatureErrorf(l.provider, "xai: the wire has no video list endpoint (probed 2026-09-01: 404) — the ticket you stored is the only copy")
}

func (l *OpenAIChatLM) videoResultFetch(JSONObject) (*TransportRequest, error) {
	if !l.isXai() {
		return nil, l.unsupported("video generation")
	}
	return nil, nil
}

func (l *OpenAIChatLM) videoPart(statusBody JSONObject, _ *HTTPResponse) (VideoPart, error) {
	if !l.isXai() {
		return VideoPart{}, l.unsupported("video generation")
	}
	url := stringOnly(wireObj(statusBody["video"])["url"])
	if url == "" {
		return VideoPart{}, l.providerError(KindProvider, "xai: terminal video carries no url", 0, "", "")
	}
	return VideoPart{Media: Media{MediaType: "video/mp4", URL: url}}, nil
}
