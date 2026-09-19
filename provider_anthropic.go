package lm15

import (
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

// AnthropicLM is the Anthropic Messages dialect bound to an access policy.
// ClaudeCodeLM is the same type bound to lm15.ClaudeCode.
type AnthropicLM struct {
	lmCore
	apiVersion string
	resolved   ResolvedAnthropicCompat
}

// ClaudeCodeLM is a name: AnthropicLM bound to lm15.ClaudeCode.
type ClaudeCodeLM = AnthropicLM

const anthropicDefaultBaseURL = "https://api.anthropic.com/v1"

var anthropicBuiltinMap = map[string]string{"web_search": "web_search_20250305", "code_execution": "code_execution_20250522"}
var anthropicProviderExecutedBlocks = map[string]bool{"server_tool_use": true, "web_search_tool_result": true, "code_execution_tool_result": true}

const (
	anthropicDefaultVisibleTokens = 1024
)

var adaptiveClassMarkers = []string{"sonnet-5", "opus-5", "sonnet-4-6", "opus-4-6", "opus-4-7", "opus-4-8", "fable", "mythos", "haiku-5"}

// AnthropicAdaptiveClass reports models that take thinking.type=adaptive (MAP-7 rule 10).
func AnthropicAdaptiveClass(model string) bool {
	lowered := strings.ToLower(model)
	for _, m := range adaptiveClassMarkers {
		if strings.Contains(lowered, m) {
			return true
		}
	}
	return false
}

// NewAnthropicLM constructs the Messages dialect (default policy AnthropicAPI).
func NewAnthropicLM(opts ...Option) (*AnthropicLM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	lm := &AnthropicLM{apiVersion: "2023-06-01"}
	if o.apiVersion != "" {
		lm.apiVersion = o.apiVersion
	}
	if err := lm.bindAccess(lm, AnthropicAPI, o, anthropicDefaultBaseURL); err != nil {
		return nil, err
	}
	compat := o.compatPreset
	if compat == "" && o.anthropicCompat == nil {
		compat = lm.registryCompat()
	}
	switch {
	case compat != "":
		preset, err := AnthropicPreset(compat)
		if err != nil {
			return nil, err
		}
		lm.resolved = ResolveAnthropicCompat(preset)
		if lm.baseURL == anthropicDefaultBaseURL {
			url, err := PresetBaseURL(AnthropicPresetBaseURLs, compat, "Messages", "anthropic")
			if err != nil {
				return nil, err
			}
			lm.baseURL = url
		}
	case o.anthropicCompat != nil:
		lm.resolved = ResolveAnthropicCompat(*o.anthropicCompat)
	default:
		lm.resolved = ResolveAnthropicCompat(AnthropicCompat{})
	}
	return lm, nil
}

// NewClaudeCodeLM constructs the Messages dialect on a Claude Code login.
func NewClaudeCodeLM(opts ...Option) (*AnthropicLM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	policy := ClaudeCode
	if o.claudeCodeVersion != "" && o.claudeCodeVersion != DefaultClaudeCodeVersion {
		policy = policy.WithHeaders([][2]string{{"user-agent", "claude-cli/" + o.claudeCodeVersion}})
	}
	return NewAnthropicLM(append([]Option{WithAccess(policy)}, opts...)...)
}

var anthropicErrorTypeMap = map[string]ErrorKind{
	"authentication_error": KindAuth, "permission_error": KindAuth, "billing_error": KindBilling, "rate_limit_error": KindRateLimit,
	"request_too_large": KindInvalidRequest, "not_found_error": KindInvalidRequest, "resource_not_found_error": KindInvalidRequest,
	"DeploymentNotFound": KindUnsupportedModel, "invalid_authentication_error": KindAuth, "invalid_request_error": KindInvalidRequest,
	"api_error": KindServer, "overloaded_error": KindServer, "timeout_error": KindTimeout,
}

func anthropicContextLengthMessage(msg string) bool {
	l := strings.ToLower(msg)
	return strings.Contains(l, "prompt is too long") || strings.Contains(l, "too many tokens") || strings.Contains(l, "context window") ||
		strings.Contains(l, "context length") || (strings.Contains(l, "token") && (strings.Contains(l, "limit") || strings.Contains(l, "exceed")))
}

func (l *AnthropicLM) errorDetail(providerCode, message string) ErrorDetail {
	kind, ok := anthropicErrorTypeMap[providerCode]
	if !ok {
		kind = KindProvider
	}
	if anthropicContextLengthMessage(message) {
		kind = KindContextLength
	} else if providerCode == "not_found_error" && isModelErrorMessage(message) {
		kind = KindUnsupportedModel
	}
	msg := message
	if msg == "" {
		msg = providerCode
	}
	if msg == "" {
		msg = "provider error"
	}
	if providerCode == "" {
		providerCode = "provider"
	}
	return ErrorDetail{Code: kind.CanonicalCode(), Message: msg, ProviderCode: providerCode}
}

func (l *AnthropicLM) normalizeError(status int, body string) *Error {
	data, err := DecodeJSON([]byte(body))
	obj := wireObj(data)
	if err != nil || (obj == nil && data != nil && err == nil) {
		return l.lmCore.normalizeError(status, body)
	}
	var msg, errType, requestID string
	if obj != nil {
		var errObj any = obj["error"]
		switch e := errObj.(type) {
		case map[string]any:
			msg = wireStr(e["message"])
			errType = firstStr(e["type"], e["code"])
		case string:
			msg = e
		default:
			// Azure Foundry's gateway uses top-level {code, message}.
			msg = wireStr(obj["message"])
			errType = firstStr(obj["type"], obj["code"])
		}
		requestID = wireStr(obj["request_id"])
	}
	switch {
	case anthropicContextLengthMessage(msg):
		return l.providerError(KindContextLength, msg, status, errType, requestID)
	case errType == "DeploymentNotFound" || ((errType == "not_found_error" || errType == "resource_not_found_error") && isModelErrorMessage(msg)):
		return l.providerError(KindUnsupportedModel, msg, status, errType, requestID)
	}
	if kind, ok := anthropicErrorTypeMap[errType]; ok {
		return l.providerError(kind, msg, status, errType, requestID)
	}
	if errType != "" && !strings.Contains(msg, errType) {
		msg = msg + " (" + errType + ")"
	}
	if msg == "" {
		msg = strings.TrimSpace(body)
		if len(msg) > 500 {
			msg = msg[:500]
		}
		if msg == "" {
			msg = "HTTP " + strconv.Itoa(status)
		}
	}
	return l.withLoginHint(MapHTTPError(status, msg, l.provider, l.access.EnvKeys, errType, requestID, nil))
}

func (l *AnthropicLM) headers(req *Request) [][2]string {
	headers := [][2]string{{"anthropic-version", l.apiVersion}, {"content-type", "application/json"}}
	var betas []string
	for _, h := range l.access.Headers {
		if strings.EqualFold(h[0], "anthropic-beta") {
			for _, b := range strings.Split(h[1], ",") {
				if b != "" {
					betas = append(betas, b)
				}
			}
		} else {
			headers = append(headers, h)
		}
	}
	if req != nil {
		for _, t := range req.Tools {
			if bt, ok := t.(BuiltinTool); ok && bt.Name == "code_execution" {
				betas = append(betas, "code-execution-2025-05-22")
			}
		}
	}
	if len(betas) > 0 {
		headers = append(headers, [2]string{"anthropic-beta", strings.Join(betas, ",")})
	}
	return headers
}

// ─── Request serialization ───────────────────────────────────────────

func (l *AnthropicLM) toolResultContent(p Part) (JSONObject, error) {
	switch x := p.(type) {
	case TextPart:
		return JSONObject{"type": "text", "text": x.Text}, nil
	case ImagePart:
		src, err := anthropicSource(x.Media)
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "image", "source": src}, nil
	case DocumentPart:
		src, err := anthropicSource(x.Media)
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "document", "source": src}, nil
	}
	if IsMediaPart(p) {
		return nil, UnsupportedFeatureErrorf(l.provider, "%s: a %s part cannot reach a tool_result block (text, image and document only; MAP-10)", l.provider, p.Type())
	}
	text, err := partsToText([]Part{p}, l.provider, "")
	if err != nil {
		return nil, err
	}
	return JSONObject{"type": "text", "text": text}, nil
}

func (l *AnthropicLM) part(p Part) (JSONObject, error) {
	switch x := p.(type) {
	case TextPart:
		return JSONObject{"type": "text", "text": x.Text}, nil
	case ImagePart:
		src, err := anthropicSource(x.Media)
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "image", "source": src}, nil
	case DocumentPart:
		src, err := anthropicSource(x.Media)
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "document", "source": src}, nil
	case ToolCallPart:
		return JSONObject{"type": "tool_use", "id": x.ID, "name": x.Name, "input": x.Input}, nil
	case ToolResultPart:
		if err := checkToolResultMedia(l.provider, x, l.resolved.ToolResultMedia, "a tool_result block"); err != nil {
			return nil, err
		}
		var blocks []JSONObject
		for _, c := range x.Content {
			b, err := l.toolResultContent(c)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, b)
		}
		out := JSONObject{"type": "tool_result", "tool_use_id": x.ID}
		if len(blocks) == 1 && blocks[0]["type"] == "text" {
			out["content"] = blocks[0]["text"]
		} else if len(blocks) > 0 {
			out["content"] = toAnyList(blocks, func(b JSONObject) any { return b })
		}
		if x.IsError {
			out["is_error"] = true
		}
		return out, nil
	case ThinkingPart:
		if redacted := ContinuationData(x.Continuation, "anthropic", "redacted_thinking"); redacted != nil {
			out := JSONObject{"type": "redacted_thinking"}
			for k, v := range redacted {
				out[k] = v
			}
			return out, nil
		}
		if sig := ContinuationData(x.Continuation, "anthropic", "thinking_signature"); sig != nil && truthy(sig["signature"]) {
			return JSONObject{"type": "thinking", "thinking": x.Text, "signature": sig["signature"]}, nil
		}
		if l.resolved.ThinkingReplay == "unsigned" && x.Text != "" {
			return JSONObject{"type": "thinking", "thinking": x.Text}, nil
		}
		return JSONObject{"type": "text", "text": x.Text}, nil
	}
	return JSONObject{"type": "text", "text": partText(p)}, nil
}

func (l *AnthropicLM) message(msg Message) (JSONObject, error) {
	role := "user"
	if msg.Role == RoleAssistant {
		role = "assistant"
	}
	if msg.Role == RoleDeveloper {
		text, err := partsToText(msg.Parts, "", "")
		if err != nil {
			return nil, err
		}
		return JSONObject{"role": role, "content": []any{JSONObject{"type": "text", "text": "[developer]\n" + text}}}, nil
	}
	var parts []any
	for _, p := range msg.Parts {
		block, err := l.part(p)
		if err != nil {
			return nil, err
		}
		parts = append(parts, block)
	}
	return JSONObject{"role": role, "content": parts}, nil
}

func (l *AnthropicLM) toolChoicePayload(req *Request) (JSONObject, error) {
	tc := req.Config.ToolChoice
	if tc == nil {
		return nil, nil
	}
	payload := JSONObject{}
	switch {
	case tc.EffectiveMode() == "none":
		payload["type"] = "none"
	case len(tc.Allowed) > 0:
		if len(tc.Allowed) == 1 && tc.EffectiveMode() == "required" {
			payload["type"] = "tool"
			payload["name"] = tc.Allowed[0]
		} else if sameNameSet(tc.Allowed, req.Tools) {
			if tc.EffectiveMode() == "required" {
				payload["type"] = "any"
			} else {
				payload["type"] = "auto"
			}
		} else {
			return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: tool_choice.allowed subsets are not supported — the Messages API can force one named tool or allow all declared tools, but cannot restrict to a subset. Send only the allowed tools in Request.tools instead")
		}
	case tc.EffectiveMode() == "required":
		payload["type"] = "any"
	default:
		payload["type"] = "auto"
	}
	if tc.Parallel != nil && !*tc.Parallel && payload["type"] != "none" {
		payload["disable_parallel_tool_use"] = true
	}
	return payload, nil
}

func sameNameSet(allowed []string, tools []Tool) bool {
	a := map[string]bool{}
	for _, n := range allowed {
		a[n] = true
	}
	b := map[string]bool{}
	for _, t := range tools {
		b[t.ToolName()] = true
	}
	if len(a) != len(b) {
		return false
	}
	for n := range a {
		if !b[n] {
			return false
		}
	}
	return true
}

func anthropicResponseFormat(provider string, f JSONObject) (JSONObject, error) {
	if f["type"] == "json_object" {
		return nil, UnsupportedFeatureErrorf(provider, "anthropic: response_format json_object is not supported — the Messages API has no any-JSON mode; give a json_schema (objects need additionalProperties: false)")
	}
	return JSONObject{"format": JSONObject{"type": "json_schema", "schema": f["schema"]}}, nil
}

func (l *AnthropicLM) payload(req *Request, stream bool) (JSONObject, error) {
	compat := l.resolved
	cfg := req.Config
	if compat.ModelPrefixes != nil {
		ok := false
		for _, p := range compat.ModelPrefixes {
			if strings.HasPrefix(req.Model, p) {
				ok = true
			}
		}
		if !ok {
			e := providerErrorf(KindUnsupportedModel, l.provider, nil, l.provider+": model "+strconv.Quote(req.Model)+" is not one this endpoint serves as typed (expected a prefix in "+fmtList(compat.ModelPrefixes)+"); it would be silently substituted by another model. Name the model you actually want.")
			return nil, e
		}
	}
	cacheCfg := cfg.Cache
	useCache := cacheCfg != nil && cacheCfg.EffectiveMode() != "off" && compat.CacheControl == "anthropic"
	longCache := cacheCfg != nil && cacheCfg.Retention == "long" && compat.CacheControl == "anthropic"

	var messages []JSONObject
	for _, m := range req.Messages {
		wm, err := l.message(m)
		if err != nil {
			return nil, err
		}
		messages = append(messages, wm)
	}
	if useCache {
		if cacheCfg.Key != "" {
			return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: cache.key is not supported — the Messages API has no cache affinity key (OpenAI's prompt_cache_key); marks on blocks are the mechanism (prefix / prefix_until_index)")
		}
		if cacheCfg.Resource != "" {
			return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: cache.resource is not supported — the Messages API has no stored-cache tier; it caches by marks on blocks")
		}
		idx := -1
		if cacheCfg.PrefixUntilIndex != nil {
			idx = *cacheCfg.PrefixUntilIndex
			if idx > len(messages)-1 {
				idx = len(messages) - 1
			}
		} else if cacheCfg.Prefix == "history" {
			idx = len(messages) - 1
		}
		if idx >= 0 {
			if content, ok := messages[idx]["content"].([]any); ok && len(content) > 0 {
				if last, ok := content[len(content)-1].(JSONObject); ok {
					marker := JSONObject{"type": "ephemeral"}
					if longCache {
						marker["ttl"] = "1h"
					}
					if _, has := last["cache_control"]; !has {
						last["cache_control"] = marker
					}
				}
			}
		}
	}

	reasoning := cfg.Reasoning
	deepseekThinking := compat.ThinkingFormat == "deepseek"
	alwaysAdaptive := compat.ThinkingFormat == "adaptive"
	effortOnly := compat.ThinkingFormat == "effort"
	adaptive := reasoning != nil && !reasoning.IsOff() && (deepseekThinking || alwaysAdaptive || effortOnly || AnthropicAdaptiveClass(req.Model))
	if reasoning != nil && !reasoning.IsOff() {
		if compat.ReasoningEfforts != nil && !inVocab(reasoning.Effort, compat.ReasoningEfforts) {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: reasoning.effort=%q has no level on this server (it accepts %s) and would be accepted silently", l.provider, reasoning.Effort, strings.Join(compat.ReasoningEfforts, ", "))
		}
		if reasoning.Summary == "concise" || reasoning.Summary == "detailed" {
			return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: reasoning.summary=%q is an OpenAI detail level; the Messages API returns thinking blocks whenever thinking runs (use 'auto' or None)", reasoning.Summary)
		}
		if adaptive {
			if reasoning.ThinkingBudget != nil {
				why := "this model class takes thinking.type 'adaptive' with output_config.effort; budget_tokens is rejected by the API (live 2026-09-02)"
				if deepseekThinking {
					why = "this server ignores budget_tokens (a silent no-op); effort is the dial"
				} else if alwaysAdaptive {
					why = "this server accepts budget_tokens without translating it (a silent no-op); effort is the dial (protocols--messages.md)"
				}
				return nil, UnsupportedFeatureErrorf(l.provider, "%s: reasoning.thinking_budget is not supported on %s — %s", l.provider, req.Model, why)
			}
			if reasoning.Effort == "minimal" && !(deepseekThinking || alwaysAdaptive || effortOnly) {
				return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: reasoning.effort='minimal' has no level on this model class (output_config.effort is low|medium|high|xhigh|max); 'low' is the floor")
			}
		}
	}
	var thinkingBudget *int
	if !adaptive && reasoning != nil && !reasoning.IsOff() {
		if reasoning.ThinkingBudget != nil {
			thinkingBudget = reasoning.ThinkingBudget
		} else {
			b := EffortThinkingBudgets[reasoning.Effort]
			thinkingBudget = &b
		}
	}
	maxTokens := anthropicDefaultVisibleTokens
	if cfg.MaxTokens != nil {
		maxTokens = *cfg.MaxTokens
	}
	if thinkingBudget != nil {
		maxTokens += *thinkingBudget
	}
	payload := JSONObject{"model": req.Model, "messages": toAnyList(messages, func(m JSONObject) any { return m }), "stream": stream, "max_tokens": maxTokens}
	if req.System != nil {
		text, err := systemText(req.System, l.provider)
		if err != nil {
			return nil, err
		}
		if useCache {
			marker := JSONObject{"type": "ephemeral"}
			if longCache {
				marker["ttl"] = "1h"
			}
			payload["system"] = []any{JSONObject{"type": "text", "text": text, "cache_control": marker}}
		} else {
			payload["system"] = text
		}
	}
	if compat.SamplingParams == "reject" {
		for name, set := range map[string]bool{"temperature": cfg.Temperature != nil, "top_p": cfg.TopP != nil, "top_k": cfg.TopK != nil} {
			if set {
				return nil, UnsupportedFeatureErrorf(l.provider, "%s: config.%s is silently ignored by this server (the model's sampling is fixed); omit it", l.provider, name)
			}
		}
	}
	if cfg.Temperature != nil {
		payload["temperature"] = jsonFloat(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		payload["top_p"] = jsonFloat(*cfg.TopP)
	}
	if cfg.TopK != nil {
		payload["top_k"] = *cfg.TopK
	}
	if len(cfg.Stop) > 0 {
		payload["stop_sequences"] = toAnyList(cfg.Stop, func(s string) any { return s })
	}
	if len(req.Tools) > 0 {
		var tools []any
		for _, t := range req.Tools {
			switch x := t.(type) {
			case FunctionTool:
				tools = append(tools, JSONObject{"name": x.Name, "description": nilIfEmpty(x.Description), "input_schema": x.EffectiveParameters()})
			case BuiltinTool:
				wt := x.Name
				if mapped, ok := anthropicBuiltinMap[x.Name]; ok {
					wt = mapped
				}
				out := JSONObject{"type": wt, "name": x.Name}
				for k, v := range x.Config {
					out[k] = v
				}
				tools = append(tools, out)
			}
		}
		payload["tools"] = tools
	}
	toolChoice, err := l.toolChoicePayload(req)
	if err != nil {
		return nil, err
	}
	if toolChoice != nil {
		if compat.ParallelToolCalls == "reject" && cfg.ToolChoice.Parallel != nil {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: tool_choice.parallel is silently ignored by this server (disable_parallel_tool_use is not applied); omit it", l.provider)
		}
		payload["tool_choice"] = toolChoice
	}
	switch {
	case deepseekThinking:
		if reasoning != nil && reasoning.IsOff() {
			payload["thinking"] = JSONObject{"type": "disabled"}
		} else if reasoning != nil {
			payload["thinking"] = JSONObject{"type": "enabled"}
			payload["output_config"] = JSONObject{"effort": reasoning.Effort}
		}
	case effortOnly:
		if reasoning != nil && reasoning.IsOff() {
			payload["thinking"] = JSONObject{"type": "disabled"}
		} else if reasoning != nil {
			payload["output_config"] = JSONObject{"effort": reasoning.Effort}
		}
	case alwaysAdaptive && reasoning != nil && reasoning.IsOff():
		payload["thinking"] = JSONObject{"type": "disabled"}
	case adaptive:
		payload["thinking"] = JSONObject{"type": "adaptive"}
		payload["output_config"] = JSONObject{"effort": reasoning.Effort}
	case thinkingBudget != nil:
		payload["thinking"] = JSONObject{"type": "enabled", "budget_tokens": *thinkingBudget}
	}
	if len(cfg.ResponseFormat) > 0 {
		if compat.StructuredOutput == "reject" {
			return nil, UnsupportedFeatureErrorf(l.provider, "%s: response_format is silently ignored by this server (output_config.format is accepted and not applied); describe the shape in the prompt", l.provider)
		}
		oc, err := anthropicResponseFormat(l.provider, cfg.ResponseFormat)
		if err != nil {
			return nil, err
		}
		existing := wireObj(payload["output_config"])
		merged := copyObject(existing)
		for k, v := range oc {
			merged[k] = v
		}
		payload["output_config"] = merged
	}
	if cfg.ServiceTier != "" {
		payload["service_tier"] = cfg.ServiceTier
	}
	if cfg.UserID != "" {
		payload["metadata"] = JSONObject{"user_id": cfg.UserID}
	}
	if cfg.Store != nil {
		return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: config.store is not supported — the Messages API has no response-storage opt-out field (OpenAI and Gemini carry it)")
	}
	if cfg.Logprobs != nil {
		return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: config.logprobs is not supported — the Messages API does not expose token log probabilities (OpenAI and Gemini carry them)")
	}
	for k, v := range cfg.Extensions {
		if k != "prompt_caching" {
			payload[k] = v
		}
	}
	if l.access.SystemPrefix != "" {
		prefix := JSONObject{"type": "text", "text": l.access.SystemPrefix}
		switch existing := payload["system"].(type) {
		case nil:
			payload["system"] = []any{prefix}
		case []any:
			payload["system"] = append([]any{prefix}, existing...)
		default:
			payload["system"] = []any{prefix, JSONObject{"type": "text", "text": wireStr(existing)}}
		}
	}
	return payload, nil
}

func (l *AnthropicLM) buildRequest(req *Request, stream bool) (*TransportRequest, error) {
	payload, err := l.payload(req, stream)
	if err != nil {
		return nil, err
	}
	timeout := 60 * time.Second
	if stream {
		timeout = 120 * time.Second
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/messages", headers: l.headers(req), payload: payload, endpoint: "messages", stream: stream, model: req.Model, readTimeout: timeout})
}

// ─── Response parsing ────────────────────────────────────────────────

func anthropicFinish(stopReason string, hasToolCall bool) string {
	if hasToolCall {
		return FinishToolCall
	}
	switch strings.ToLower(stopReason) {
	case "max_tokens", "model_context_window_exceeded":
		return FinishLength
	case "tool_use", "pause_turn":
		return FinishToolCall
	case "refusal", "safety", "content_filter":
		return FinishContentFilter
	}
	return FinishStop
}

func anthropicReasoningTokens(usage JSONObject) *int {
	details := wireObj(usage["output_tokens_details"])
	if details == nil || details["thinking_tokens"] == nil {
		return nil
	}
	return wireIntPtr(details["thinking_tokens"])
}

func anthropicUsage(u JSONObject) Usage {
	return Usage{
		InputTokens:      wireIntPtr(u["input_tokens"]),
		OutputTokens:     wireIntPtr(u["output_tokens"]),
		CacheReadTokens:  wireIntPtr(u["cache_read_input_tokens"]),
		CacheWriteTokens: wireIntPtr(u["cache_creation_input_tokens"]),
		ReasoningTokens:  anthropicReasoningTokens(u),
	}.Normalize()
}

func citationFromAnthropic(c JSONObject) (CitationPart, bool) {
	url := firstStr(c["url"], c["uri"])
	title := firstStr(c["title"], c["document_title"], c["source_title"])
	text := firstStr(c["cited_text"], c["text"], c["quote"])
	if url == "" && title == "" && text == "" {
		return CitationPart{}, false
	}
	return CitationPart{URL: url, Title: title, Text: text}, true
}

func (l *AnthropicLM) parseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	data, err := resp.JSON()
	if err != nil {
		return nil, err
	}
	var parts []Part
	var unmapped []JSONObject
	for i, raw := range wireList(data["content"]) {
		block := wireObj(raw)
		path := "content[" + strconv.Itoa(i) + "]"
		if block == nil {
			recordUnmapped(&unmapped, path, jsonTypeName(raw))
			continue
		}
		bt := wireStr(block["type"])
		switch {
		case bt == "text":
			parts = append(parts, TextPart{Text: wireStr(block["text"])})
			for _, rawC := range wireList(block["citations"]) {
				if c := wireObj(rawC); c != nil {
					if cit, ok := citationFromAnthropic(c); ok {
						parts = append(parts, cit)
					}
				}
			}
		case bt == "tool_use":
			if !truthy(block["name"]) {
				return nil, unnamedToolCallError(l.provider, path)
			}
			id := wireStr(block["id"])
			if id == "" || block["id"] == nil {
				id = "tool_" + strconv.Itoa(len(parts))
			}
			input := wireObj(block["input"])
			if input == nil {
				input = JSONObject{}
			}
			parts = append(parts, ToolCallPart{ID: id, Name: wireStr(block["name"]), Input: input})
		case bt == "thinking":
			var continuation []ContinuationState
			if truthy(block["signature"]) {
				continuation = []ContinuationState{{Provider: "anthropic", Kind: "thinking_signature", Data: JSONObject{"signature": wireStr(block["signature"])}}}
			}
			parts = append(parts, ThinkingPart{Text: firstStr(block["thinking"], block["text"]), Continuation: continuation})
		case bt == "redacted_thinking":
			var continuation []ContinuationState
			if block["data"] != nil {
				continuation = []ContinuationState{{Provider: "anthropic", Kind: "redacted_thinking", Data: JSONObject{"data": block["data"]}}}
			}
			parts = append(parts, ThinkingPart{Text: "", Continuation: continuation})
		case anthropicProviderExecutedBlocks[bt]:
		default:
			recordUnmapped(&unmapped, path, block["type"])
		}
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{}}
	}
	model := wireStr(data["model"])
	if model == "" {
		model = req.Model
	}
	return &Response{
		ID:           wireStr(data["id"]),
		Model:        model,
		Message:      Message{Role: RoleAssistant, Parts: parts},
		FinishReason: anthropicFinish(wireStr(data["stop_reason"]), hasToolCall(parts)),
		Usage:        anthropicUsage(wireObj(data["usage"])),
		ProviderData: attachUnmapped(data, unmapped),
	}, nil
}

func (l *AnthropicLM) parseStreamEvents(req *Request, ev sse.Event) ([]StreamEvent, error) {
	if ev.Data == "" {
		return nil, nil
	}
	raw, err := DecodeJSON([]byte(ev.Data))
	if err != nil {
		return nil, err
	}
	payload := wireObj(raw)
	if payload == nil {
		return nil, nil
	}
	idx := wireInt(payload["index"], 0)
	switch wireStr(payload["type"]) {
	case "message_start":
		msg := wireObj(payload["message"])
		model := wireStr(msg["model"])
		if model == "" {
			model = req.Model
		}
		return []StreamEvent{StreamStartEvent{ID: wireStr(msg["id"]), Model: model}}, nil
	case "content_block_start":
		block := wireObj(payload["content_block"])
		switch wireStr(block["type"]) {
		case "tool_use":
			input := ""
			switch start := block["input"].(type) {
			case map[string]any:
				if len(start) > 0 {
					input = jsonRaw(start)
				}
			case nil:
			default:
				input = wireStr(start)
			}
			return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: input, PartIndex: idx, ID: wireStr(block["id"]), Name: wireStr(block["name"])}}}, nil
		case "redacted_thinking":
			if block["data"] != nil {
				i := idx
				return []StreamEvent{
					StreamDeltaEvent{Delta: ThinkingDelta{Text: "", PartIndex: idx}},
					StreamDeltaEvent{Delta: ContinuationDelta{Provider: "anthropic", Kind: "redacted_thinking", Data: JSONObject{"data": block["data"]}, PartIndex: &i}},
				}, nil
			}
		}
		return nil, nil
	case "content_block_delta":
		delta := wireObj(payload["delta"])
		switch dtype := wireStr(delta["type"]); dtype {
		case "text_delta":
			return []StreamEvent{StreamDeltaEvent{Delta: TextDelta{Text: wireStr(delta["text"]), PartIndex: idx}}}, nil
		case "input_json_delta":
			return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: wireStr(delta["partial_json"]), PartIndex: idx}}}, nil
		case "thinking_delta":
			return []StreamEvent{StreamDeltaEvent{Delta: ThinkingDelta{Text: wireStr(delta["thinking"]), PartIndex: idx}}}, nil
		case "signature_delta":
			if truthy(delta["signature"]) {
				i := idx
				return []StreamEvent{StreamDeltaEvent{Delta: ContinuationDelta{Provider: "anthropic", Kind: "thinking_signature", Data: JSONObject{"signature": wireStr(delta["signature"])}, PartIndex: &i}}}, nil
			}
		case "citation_delta", "citations_delta":
			c := wireObj(delta["citation"])
			if c == nil {
				c = delta
			}
			d := CitationDelta{PartIndex: idx}
			if t := firstStr(c["cited_text"], c["text"]); t != "" {
				d.Text = S(t)
			}
			if u := wireStr(c["url"]); u != "" {
				d.URL = S(u)
			}
			if t := wireStr(c["title"]); t != "" {
				d.Title = S(t)
			}
			return []StreamEvent{StreamDeltaEvent{Delta: d}}, nil
		}
		return nil, nil
	case "message_delta":
		delta := wireObj(payload["delta"])
		var usage *Usage
		if up := wireObj(payload["usage"]); len(up) > 0 {
			u := anthropicUsage(up)
			usage = &u
		}
		stopReason, hasStop := delta["stop_reason"]
		if (hasStop && stopReason != nil) || usage != nil {
			finish := ""
			if hasStop && stopReason != nil {
				finish = anthropicFinish(wireStr(stopReason), false)
			}
			return []StreamEvent{StreamEndEvent{FinishReason: finish, Usage: usage, ProviderData: payload}}, nil
		}
		return nil, nil
	case "message_stop":
		return []StreamEvent{StreamEndEvent{}}, nil
	case "error":
		var code, message string
		if e := wireObj(payload["error"]); e != nil {
			code = firstStr(e["type"], e["code"], payload["code"])
			message = firstStr(e["message"], payload["message"])
		} else {
			code = firstStr(payload["code"], payload["error_type"])
			message = wireStr(payload["message"])
		}
		if code == "" {
			code = "provider"
		}
		return []StreamEvent{StreamErrorEvent{Error: l.errorDetail(code, message)}}, nil
	}
	return nil, nil
}

// ─── Models ──────────────────────────────────────────────────────────

func (l *AnthropicLM) modelsRequest() (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/models", params: map[string]string{"limit": "1000"}, headers: l.headers(nil), readTimeout: 30 * time.Second})
}

func (l *AnthropicLM) modelsFromBody(body string) ([]ModelInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	return modelInfosFromEntries(data["data"], l.provider, "anthropic_messages", func(e map[string]any) string { return stringOnly(e["id"]) }), nil
}

// ─── Files ───────────────────────────────────────────────────────────

func (l *AnthropicLM) fileUploadRequest(req *FileUploadRequest) (*TransportRequest, error) {
	var fields [][2]string
	for _, k := range sortedKeys(req.Extensions) {
		fields = append(fields, [2]string{k, wireStr(req.Extensions[k])})
	}
	content, err := req.Content()
	if err != nil {
		return nil, err
	}
	ct, body := multipartFormBody(fields, []multipartFile{{Field: "file", Filename: req.Filename, ContentType: req.EffectiveMediaType(), Data: content}})
	headers := l.headers(nil)
	for i := range headers {
		if headers[i][0] == "content-type" {
			headers[i][1] = ct
		}
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/files", headers: headers, body: body, readTimeout: 300 * time.Second})
}

func (l *AnthropicLM) fileInfo(data JSONObject) (FileInfo, error) {
	id := stringOnly(data["id"])
	if id == "" {
		return FileInfo{}, l.providerError(KindProvider, "anthropic: file object carries no id", 0, "", "")
	}
	var size *int
	if v, ok := data["size_bytes"]; ok {
		if _, isBool := v.(bool); !isBool {
			if i, err := jsonInt(v, "size_bytes"); err == nil {
				size = &i
			}
		}
	}
	var downloadable *bool
	if b, ok := data["downloadable"].(bool); ok {
		downloadable = &b
	}
	return FileInfo{
		ID: id, Filename: stringOnly(data["filename"]), MediaType: stringOnly(data["mime_type"]), SizeBytes: size,
		CreatedAt: isoUTC(data["created_at"]), ExpiresAt: isoUTC(data["expires_at"]), Readiness: "ready",
		Downloadable: downloadable, ProviderData: data,
	}, nil
}

func (l *AnthropicLM) fileInfoFromBody(body string) (FileInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FileInfo{}, err
	}
	return l.fileInfo(data)
}

func (l *AnthropicLM) fileGetRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(fileID, false), headers: l.headers(nil), readTimeout: 60 * time.Second})
}

func (l *AnthropicLM) fileListRequest(limit int, cursor string) (*TransportRequest, error) {
	params := map[string]string{"limit": strconv.Itoa(limit)}
	if cursor != "" {
		params["page"] = cursor
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files", params: params, headers: l.headers(nil), readTimeout: 60 * time.Second})
}

func (l *AnthropicLM) filePageFromListBody(body string) (FilePage, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FilePage{}, err
	}
	var items []FileInfo
	for _, e := range wireList(data["data"]) {
		if obj := wireObj(e); obj != nil {
			info, err := l.fileInfo(obj)
			if err != nil {
				return FilePage{}, err
			}
			items = append(items, info)
		}
	}
	return FilePage{Items: items, NextCursor: stringOnly(data["next_page"])}, nil
}

func (l *AnthropicLM) fileDeleteRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "DELETE", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(fileID, false), headers: l.headers(nil), readTimeout: 60 * time.Second})
}

func (l *AnthropicLM) fileDownloadRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(fileID, false) + "/content", headers: l.headers(nil), readTimeout: 300 * time.Second})
}

// ─── Batch ───────────────────────────────────────────────────────────

func anthropicBatchStatus(data JSONObject) string {
	switch strings.ToLower(wireStr(data["processing_status"])) {
	case "in_progress":
		return BatchRunning
	case "canceling":
		return BatchCancelling
	case "ended":
		counts := wireObj(data["request_counts"])
		n := func(k string) int { return wireInt(counts[k], 0) }
		if n("canceled") > 0 && n("succeeded") == 0 && n("errored") == 0 && n("expired") == 0 {
			return BatchCancelled
		}
		if n("expired") > 0 && n("succeeded") == 0 && n("errored") == 0 && n("canceled") == 0 {
			return BatchExpired
		}
		return BatchCompleted
	}
	return BatchQueued
}

func (l *AnthropicLM) batchSubmitRequest(req *BatchRequest, _ JSONObject) (*TransportRequest, error) {
	if req.Label != "" {
		return nil, UnsupportedFeatureErrorf(l.provider, "anthropic: batch labels are not supported — the Message Batches create body has no metadata field (verified live 2026-08-31); submit without a label and correlate by id")
	}
	var requests []any
	for i, nested := range req.Requests {
		params, err := l.payload(nested, false)
		if err != nil {
			return nil, err
		}
		requests = append(requests, JSONObject{"custom_id": strconv.Itoa(i), "params": params})
	}
	payload := JSONObject{"requests": requests}
	for k, v := range req.Extensions {
		payload[k] = v
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/messages/batches", headers: l.headers(nil), payload: payload, readTimeout: 120 * time.Second})
}

func (l *AnthropicLM) batchJobInfo(data JSONObject) (BatchJobInfo, error) {
	id := stringOnly(data["id"])
	if id == "" {
		return BatchJobInfo{}, l.providerError(KindProvider, "anthropic: batch object carries no id", 0, "", "")
	}
	return BatchJobInfo{ID: id, Status: anthropicBatchStatus(data), CreatedAt: isoUTC(data["created_at"]), ProviderData: data}, nil
}

func (l *AnthropicLM) batchJobFromBody(body string) (BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return BatchJobInfo{}, err
	}
	return l.batchJobInfo(data)
}

func (l *AnthropicLM) batchStatusRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/messages/batches/" + pathID(batchID, false), headers: l.headers(nil), readTimeout: 60 * time.Second})
}

func (l *AnthropicLM) batchCancelRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/messages/batches/" + pathID(batchID, false) + "/cancel", headers: l.headers(nil), readTimeout: 60 * time.Second})
}

func (l *AnthropicLM) batchResultFetches(statusBody JSONObject) ([]*TransportRequest, error) {
	url := stringOnly(statusBody["results_url"])
	if url == "" {
		return nil, l.providerError(KindProvider, "anthropic: ended batch carries no results_url", 0, "", "")
	}
	req, err := l.emit(emitSpec{method: "GET", url: url, headers: l.headers(nil), readTimeout: 300 * time.Second})
	if err != nil {
		return nil, err
	}
	return []*TransportRequest{req}, nil
}

func (l *AnthropicLM) batchEntries(_ JSONObject, fetched []string) ([]BatchEntry, error) {
	var entries []BatchEntry
	if len(fetched) == 0 {
		return entries, nil
	}
	for _, line := range strings.Split(fetched[0], "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		item, err := DecodeJSONObject([]byte(line))
		if err != nil {
			return nil, err
		}
		index, err := strconv.Atoi(wireStr(item["custom_id"]))
		if err != nil {
			return nil, valueErrorf("batch entry custom_id is not an int")
		}
		result := wireObj(item["result"])
		switch rtype := wireStr(result["type"]); rtype {
		case "succeeded":
			message := wireObj(result["message"])
			resp, err := l.parseResponse(batchEntryRequest(stringOnly(message["model"])), JSONResponse(200, message))
			if err != nil {
				return nil, err
			}
			entries = append(entries, BatchEntry{Index: index, Outcome: "succeeded", Response: resp})
		case "errored":
			raw := result["error"]
			var envelope JSONObject
			if obj := wireObj(raw); obj != nil {
				if _, has := obj["error"]; has {
					envelope = obj
				} else {
					envelope = JSONObject{"error": obj}
				}
			} else {
				envelope = JSONObject{"error": raw}
			}
			e := l.normalizeError(400, jsonRaw(envelope))
			msg := e.Message
			if msg == "" {
				msg = "batch entry errored"
			}
			entries = append(entries, BatchEntry{Index: index, Outcome: "errored", Error: &ErrorDetail{Code: e.Code, Message: msg, ProviderCode: e.ProviderCode}})
		case "canceled":
			entries = append(entries, BatchEntry{Index: index, Outcome: "cancelled"})
		case "expired":
			entries = append(entries, BatchEntry{Index: index, Outcome: "expired"})
		default:
			entries = append(entries, BatchEntry{Index: index, Outcome: "errored", Error: &ErrorDetail{Code: CodeProvider, Message: "unrecognized batch result type " + strconv.Quote(rtype)}})
		}
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].Index < entries[j].Index })
	return entries, nil
}

func (l *AnthropicLM) batchListRequest(limit int) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/messages/batches", params: map[string]string{"limit": strconv.Itoa(limit)}, headers: l.headers(nil), readTimeout: 60 * time.Second})
}

func (l *AnthropicLM) batchJobsFromListBody(body string) ([]BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []BatchJobInfo
	for _, e := range wireList(data["data"]) {
		if obj := wireObj(e); obj != nil {
			info, err := l.batchJobInfo(obj)
			if err != nil {
				return nil, err
			}
			out = append(out, info)
		}
	}
	return out, nil
}
