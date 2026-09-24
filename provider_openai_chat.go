package lm15

import (
	"strconv"
	"strings"

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
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/models", headers: l.headers()})
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
		if d, ok := parts[0].(DataPart); ok {
			return DataPartText(d), nil // a data part is text on this wire (D3): the same string form as a lone text part
		}
	}
	var out []any
	for _, p := range parts {
		switch x := p.(type) {
		case TextPart:
			out = append(out, JSONObject{"type": "text", "text": x.Text})
		case DataPart:
			out = append(out, JSONObject{"type": "text", "text": DataPartText(x)})
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
				return nil, UnsupportedFeature(provider, "messages[*].parts["+p.Type()+"]", "%s: a %s part in a %s message has no slot on the Chat Completions wire (text and image_url only); the OpenAI Responses, Anthropic and Gemini dialects carry it (MAP-10)", provider, p.Type(), msg.Role)
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

func (l *OpenAIChatLM) buildMessages(req *Request, compat ResolvedOpenAIChatCompat, breakpoint *int) ([]any, error) {
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
			return nil, UnsupportedFeature(l.provider, "tools["+tool.Name+"]", "%s: builtin tool %q has no Groq wire mapping — supported: %v", l.provider, tool.Name, sortStrings([]string{"code_execution", "web_search"}))
		}
		entry := JSONObject{"type": wireType}
		for k, v := range tool.Config {
			entry[k] = v
		}
		return entry, nil
	}
	return nil, UnsupportedFeature(l.provider, "tools["+tool.Name+"]", "%s: builtin tool %q is not supported on this server — the Chat Completions wire carries function tools only, and unproven servers may silently ignore unknown tool types. Use compat='groq' for Groq's server-executed tools, or the OpenAI Responses / Anthropic / Gemini providers", l.provider, tool.Name)
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
			return nil, UnsupportedFeature(l.provider, "config.tool_choice.allowed", "%s: cannot force builtin tools %v — the Chat Completions wire has no hosted-tool tool_choice form (OpenAI Responses and Anthropic carry it)", l.provider, builtins)
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

// xaiAdapt applies api.x.ai's documented and live-measured gaps under
// MAP-13, returning the request as it goes to the wire.
func (l *OpenAIChatLM) xaiAdapt(req *Request, scope *adaptScope) (*Request, error) {
	out := *req
	cfg := out.Config
	if cfg.Reasoning != nil && cfg.Reasoning.IsOff() {
		// MAP-13 (decision 2026-09-14 §4.2): no off switch exists; the
		// lowest level is the closest and the spend shows in usage.
		if err := scope.substituted("config.reasoning.effort", "Grok reasoning models have no off switch and api.x.ai ignores disable fields (158 reasoning tokens on an explicit off, live 2026-09-01); the lowest level was sent", "off", "low"); err != nil {
			return nil, err
		}
		r := *cfg.Reasoning
		r.Effort = "low"
		cfg.Reasoning = &r
	}
	if cfg.Logprobs != nil {
		if err := scope.dropped("config.logprobs", "grok-4.20 and newer ignore logprobs/top_logprobs (docs.x.ai, live 2026-09-01); Response.logprobs will be absent (OpenAI and Gemini carry them)", *cfg.Logprobs); err != nil {
			return nil, err
		}
		cfg.Logprobs = nil
	}
	tc := cfg.ToolChoice
	if tc != nil && len(tc.Allowed) > 0 && !(len(tc.Allowed) == 1 && tc.EffectiveMode() == "required") {
		// api.x.ai accepts allowed_tools and ignores it (live 2026-09-02:
		// with {lookup} allowed and weather asked, it called weather).
		// MAP-13 client_side: send only the allowed tools — what the
		// allowlist means — and record it.
		var kept []Tool
		var names []any
		for _, t := range out.Tools {
			if inVocab(t.ToolName(), tc.Allowed) {
				kept = append(kept, t)
				names = append(names, t.ToolName())
			}
		}
		if err := scope.clientSide("config.tool_choice.allowed", "api.x.ai ignores tool_choice allowlists (live 2026-09-02); only the allowed tools were sent, which is what the allowlist means", toAnyList(tc.Allowed, func(s string) any { return s }), names); err != nil {
			return nil, err
		}
		out.Tools = kept
		narrowed := *tc
		narrowed.Allowed = nil
		cfg.ToolChoice = &narrowed
		tc = &narrowed
	}
	if tc != nil && tc.EffectiveMode() == "required" && len(cfg.ResponseFormat) > 0 {
		// MAP-13 rule 4(b): the program depends on the call.
		return nil, UnsupportedFeature(l.provider, "config.tool_choice.mode", "xai: a forced tool (mode='required') cannot be combined with response_format — api.x.ai returns JSON text and drops the call (verified live 2026-09-02)")
	}
	out.Config = cfg
	return &out, nil
}

func (l *OpenAIChatLM) payload(req *Request, stream bool, scope *adaptScope) (JSONObject, error) {
	if err := checkMessageMedia(req.Messages, "openai_chat", l.provider); err != nil {
		return nil, err
	}
	if l.isXai() {
		adapted, err := l.xaiAdapt(req, scope)
		if err != nil {
			return nil, err
		}
		req = adapted
	}
	compat := l.compatFor(req.Model)
	breakpoint, err := cacheBreakpointIndex(req, compat.CacheControl, scope) // once: it may record
	if err != nil {
		return nil, err
	}
	messages, err := l.buildMessages(req, compat, breakpoint)
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
		// MAP-13: a sampling hint with no field on this wire; servers that
		// take one do so through extensions.
		if err := scope.dropped("config.top_k", "the Chat Completions wire has no top_k (Anthropic and Gemini carry it; servers that accept it take it through extensions)", *cfg.TopK); err != nil {
			return nil, err
		}
	}
	if cfg.Seed != nil {
		payload["seed"] = *cfg.Seed
	}
	if cfg.FrequencyPenalty != nil {
		payload["frequency_penalty"] = jsonFloat(*cfg.FrequencyPenalty)
	}
	if cfg.PresencePenalty != nil {
		payload["presence_penalty"] = jsonFloat(*cfg.PresencePenalty)
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
			// The server documents tool_choice=auto only and ignores every
			// other form without an error (Z.AI, live 2026-09-03: required →
			// text answer, none → a tool call). MAP-13: "none" and an
			// allowlist have a client-side form — send no tools / only
			// those tools — and are recorded; "required" cannot be forced
			// and the program depends on the call (rule 4b): refused.
			switch {
			case tc.EffectiveMode() == "required":
				return nil, UnsupportedFeature(l.provider, "config.tool_choice.mode", "%s: tool_choice mode='required' is silently ignored by this server (only 'auto' is honoured) and a forced call cannot be reproduced client-side", l.provider)
			case tc.EffectiveMode() == "none":
				if err := scope.clientSide("config.tool_choice.mode", "this server ignores tool_choice='none'; no tools were sent, which is the same outcome", "none", "no tools sent"); err != nil {
					return nil, err
				}
				delete(payload, "tools")
			default:
				var kept []any
				var names []any
				for _, t := range wireList(payload["tools"]) {
					name := wireStr(wireObj(wireObj(t)["function"])["name"])
					if inVocab(name, tc.Allowed) {
						kept = append(kept, t)
						names = append(names, name)
					}
				}
				if err := scope.clientSide("config.tool_choice.allowed", "this server ignores tool_choice allowlists; only the allowed tools were sent, which is what the allowlist means", toAnyList(tc.Allowed, func(s string) any { return s }), names); err != nil {
					return nil, err
				}
				payload["tools"] = kept
			}
			toolChoice = "auto"
		}
		payload["tool_choice"] = toolChoice
	}
	if cfg.ToolChoice != nil && cfg.ToolChoice.Parallel != nil {
		payload["parallel_tool_calls"] = *cfg.ToolChoice.Parallel
	}
	if len(cfg.ResponseFormat) > 0 {
		if compat.JSONSchema == "reject" && cfg.ResponseFormat["type"] != "json_object" {
			// The server accepts response_format.type=json_schema and ignores
			// it (Z.AI, live 2026-09-03). MAP-13: omit and record.
			if err := scope.dropped("config.response_format", "this server accepts response_format type "+strconv.Quote(wireStr(cfg.ResponseFormat["type"]))+" and does not apply it; use {'type': 'json_object'} and describe the shape in the prompt", cfg.ResponseFormat); err != nil {
				return nil, err
			}
		} else {
			// MAP-14: the judgment convention goes verbatim on the chat
			// dialect; a server that scores named tokens delivers
			// probabilities through the trie driver, every other one
			// answers with the pick only.
			if !l.scoresNamedTokens() {
				if err := noteUnmeasurableProbabilities(scope, req, l.provider); err != nil {
					return nil, err
				}
			}
			payload["response_format"] = responseFormatToChat(cfg.ResponseFormat)
		}
	}
	if r := cfg.Reasoning; r != nil {
		if compat.ThinkingFormat == "none" {
			// No reasoning dial on this server. MAP-13: the dial is dropped
			// and recorded — the model may reason at its own default and
			// the tokens show in usage.
			if err := scope.dropped("config.reasoning", "this server has no reasoning dial on its wire (compat thinking_format='none'); the model reasons at its own default; pass the server's own knob through extensions", JSONObject{"effort": r.Effort}); err != nil {
				return nil, err
			}
			r = nil
		}
		if r != nil && !r.IsOff() {
			rr := *r
			r = &rr
			if r.ThinkingBudget != nil {
				if err := scope.dropped("config.reasoning.thinking_budget", "the Chat Completions wire has no thinking token budget; effort carries the intent", *r.ThinkingBudget); err != nil {
					return nil, err
				}
			}
			if r.Summary == "concise" || r.Summary == "detailed" {
				if err := scope.substituted("config.reasoning.summary", "the Chat Completions wire has no summary detail levels; 'auto' is what it shows", r.Summary, "auto"); err != nil {
					return nil, err
				}
				r.Summary = "auto"
			}
			if compat.ReasoningEfforts != nil && !inVocab(r.Effort, compat.ReasoningEfforts) {
				// MAP-13: clamp to the nearest declared level; the server
				// would have accepted the word silently (Moonshot kimi-k3
				// answered 200 to `medium` and to `bogus`, live 2026-09-03).
				nearest, err := nearestEffort(r.Effort, compat.ReasoningEfforts)
				if err != nil {
					return nil, err
				}
				if err := scope.clamped("config.reasoning.effort", "this server has no "+strconv.Quote(r.Effort)+" level (it accepts "+strings.Join(compat.ReasoningEfforts, ", ")+") and would have accepted the word silently", r.Effort, nearest); err != nil {
					return nil, err
				}
				r.Effort = nearest
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
		} else if r != nil {
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
	if err := cacheCommonPayload(req, payload, compat.CacheControl, l.provider, breakpoint, scope); err != nil {
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

func (l *OpenAIChatLM) buildRequest(req *Request, stream bool, scope *adaptScope) (*TransportRequest, error) {
	payload, err := l.payload(req, stream, scope)
	if err != nil {
		return nil, err
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/chat/completions", endpoint: "chat/completions", stream: stream, model: req.Model, headers: l.headers(), payload: payload, scope: scope})
}

// scoresNamedTokens reports whether this server can deliver a distribution
// over declared keys (MAP-14 §4: honours logprob_token_ids; vLLM ≥ 0.29).
func (l *OpenAIChatLM) scoresNamedTokens() bool {
	return l.resolved.TokenScoring == "logprob_token_ids"
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
	data, err := l.jsonBody(resp)
	if err != nil {
		return nil, err
	}
	out, err := responseFromChatBody(l.provider, data, req.Model, nil, func(code, message string) *Error { return openaiResponseError(&l.lmCore, code, message) })
	if err != nil {
		return nil, err
	}
	return foldJudgments(out, req.Config.ResponseFormat), nil
}

// foldJudgments is MAP-14 §3: the single text part of a judgment answer
// becomes a DataPart.
func foldJudgments(resp *Response, responseFormat JSONObject) *Response {
	if resp == nil || responseFormat == nil || responseFormat["type"] != "json_schema" {
		return resp
	}
	found := JudgmentsInSchema(responseFormat["schema"])
	if len(found) == 0 {
		return resp
	}
	resp.Message.Parts = ReplaceTextWithData(resp.Message.Parts, found)
	return resp
}

func (l *OpenAIChatLM) responseFromOpenAIChat(body JSONObject, model string, choice *int, responseFormat JSONObject) (*Response, error) {
	out, err := responseFromChatBody(l.provider, body, model, choice, func(code, message string) *Error { return openaiResponseError(&l.lmCore, code, message) })
	if err != nil {
		return nil, err
	}
	return foldJudgments(out, responseFormat), nil
}

// ResponseFromOpenAIChat reads a Chat Completions response body into a
// canonical Response (MAP-12 rule 9), under the openai-chat provider name.
// responseFormat (optional) is the request's, so a judgment answer folds
// into a DataPart (MAP-14 §3).
func ResponseFromOpenAIChat(body JSONObject, model string, choice *int, responseFormat JSONObject) (*Response, error) {
	out, err := responseFromOpenAIChatBody(body, model, choice)
	if err != nil {
		return nil, err
	}
	return foldJudgments(out, responseFormat), nil
}

func responseFromOpenAIChatBody(body JSONObject, model string, choice *int) (*Response, error) {
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
		return l.emit(emitSpec{method: "POST", url: base + "/images/generations", headers: l.headers(), payload: payload})
	}
	if len(req.Images) > 1 {
		return nil, UnsupportedFeatureErrorf(l.provider, "xai: image edits take exactly one input image; the wire has no slot for more")
	}
	img, err := xaiImageInput(req.Images[0])
	if err != nil {
		return nil, err
	}
	payload["image"] = img
	return l.emit(emitSpec{method: "POST", url: base + "/images/edits", headers: l.headers(), payload: payload})
}

func (l *OpenAIChatLM) imageGenerationFromResponse(_ *ImageGenerationRequest, resp *HTTPResponse) (ImageGenerationResponse, error) {
	if !l.isXai() {
		return ImageGenerationResponse{}, l.unsupported("image generation")
	}
	data, err := l.jsonBody(resp)
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
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/videos/generations", headers: l.headers(), payload: payload})
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
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/videos/" + pathID(videoID, false), headers: l.headers()})
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
