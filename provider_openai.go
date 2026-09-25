package lm15

import (
	"context"
	"encoding/base64"
	"fmt"
	"iter"
	"regexp"
	"strconv"
	"strings"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

// OpenAILM is the OpenAI Responses dialect bound to an access policy.
// OpenAICodexLM is the same type bound to the Codex login policy: the
// backend branches are keyed on access.Backend (AUTH-10).
type OpenAILM struct {
	lmCore
	compatBase OpenAIResponsesCompat
}

// OpenAICodexLM is a name: OpenAILM bound to lm15.OpenAICodex.
type OpenAICodexLM = OpenAILM

const openaiDefaultBaseURL = "https://api.openai.com/v1"

// modelListHint is appended to Codex unknown-model errors.
const modelListHint = "List the models your subscription accepts: call .list_models() on this client."

// NewOpenAILM constructs the Responses dialect (default policy OpenAIAPI).
func NewOpenAILM(opts ...Option) (*OpenAILM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	lm := &OpenAILM{}
	if err := lm.bindAccess(lm, OpenAIAPI, o, openaiDefaultBaseURL); err != nil {
		return nil, err
	}
	compat := o.compatPreset
	if compat == "" && o.responsesCompat == nil {
		compat = lm.registryCompat()
	}
	switch {
	case compat != "":
		preset, err := OpenAIResponsesPreset(compat)
		if err != nil {
			return nil, err
		}
		lm.compatBase = preset
		if lm.baseURL == openaiDefaultBaseURL {
			url, err := PresetBaseURL(OpenAIResponsesPresetBaseURLs, compat, "Responses", "openai")
			if err != nil {
				return nil, err
			}
			lm.baseURL = url
		}
	case o.responsesCompat != nil:
		lm.compatBase = *o.responsesCompat
	default:
		lm.compatBase, _ = OpenAIResponsesPreset("openai")
	}
	if lm.isCodex() {
		if lm.accountID == "" {
			if static, ok := lm.credential.(StaticCredential); ok {
				if key, ok := static.Value.(APIKey); ok {
					lm.accountID = ExtractChatGPTAccountID(key.Value)
				}
			}
		}
		if lm.accountID == "" {
			return nil, NotConfiguredErrorf(lm.provider, nil, lm.access.LoginHint, "No ChatGPT account id found in the Codex OAuth token.")
		}
	}
	return lm, nil
}

// NewOpenAICodexLM constructs the Responses dialect on a Codex CLI login.
func NewOpenAICodexLM(opts ...Option) (*OpenAILM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	policy := OpenAICodex
	if o.codexOriginator != "" && o.codexOriginator != DefaultCodexOriginator {
		policy = policy.WithHeaders([][2]string{{"originator", o.codexOriginator}})
	}
	if o.codexClientVersion != "" && o.codexClientVersion != DefaultCodexClientVersion {
		policy = policy.WithBackendOptions(map[string]string{"client_version": o.codexClientVersion})
	}
	return NewOpenAILM(append([]Option{WithAccess(policy)}, opts...)...)
}

// ─── Shared OpenAI-family helpers ────────────────────────────────────

var openaiBuiltinMap = map[string]string{
	"web_search": "web_search_preview", "code_execution": "code_interpreter", "file_search": "file_search", "computer_use": "computer_use_preview",
}

var openaiProviderExecutedItems = map[string]bool{
	"web_search_call": true, "file_search_call": true, "code_interpreter_call": true, "computer_call": true, "computer_use_call": true,
}

func builtinTypeOpenAI(tool BuiltinTool, compat ResolvedOpenAIResponsesCompat) string {
	if compat.BuiltinTools == "openai" {
		if t, ok := openaiBuiltinMap[tool.Name]; ok {
			return t
		}
	}
	return tool.Name
}

func attachUnmapped(providerData JSONObject, unmapped []JSONObject) JSONObject {
	if len(unmapped) == 0 {
		return providerData
	}
	out := copyObject(providerData)
	out.Set("_lm15_unmapped", toAnyList(unmapped, func(u JSONObject) any { return u }))
	return out
}

func recordUnmapped(unmapped *[]JSONObject, path string, typ any) {
	t := wireStr(typ)
	if !truthy(typ) {
		t = "<missing>"
	}
	*unmapped = append(*unmapped, JSONObject{{"path", path}, {"type", t}})
}

var gptVersionRe = regexp.MustCompile(`^gpt-(\d+)\.(\d+)`)

// openaiModelHasCacheOptions is true for the GPT-5.6-and-later class (MAP-6).
func openaiModelHasCacheOptions(model string) bool {
	m := gptVersionRe.FindStringSubmatch(strings.ToLower(model))
	if m == nil {
		return false
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	return major > 5 || (major == 5 && minor >= 6)
}

// cacheBreakpointIndex is the message the prompt_cache_breakpoint mark goes
// on, computed ONCE per build (it may record). MAP-13: the wire carries the
// mark on a text block of a user/developer message only; a mark asked for
// elsewhere walks back to the nearest eligible message ("cache up to here"
// — the nearest boundary before "here" is the obvious answer); with none,
// the mark is dropped and implicit caching still applies.
func cacheBreakpointIndex(req *Request, cacheControl string, scope *adaptScope) (*int, error) {
	cfg := req.Config.Cache
	if cfg == nil || cfg.EffectiveMode() == "off" || cfg.PrefixUntilIndex == nil || cacheControl != "openai" {
		return nil, nil
	}
	asked := *cfg.PrefixUntilIndex
	if asked > len(req.Messages)-1 {
		asked = len(req.Messages) - 1
	}
	for index := asked; index >= 0; index-- {
		msg := req.Messages[index]
		if msg.Role == RoleAssistant || msg.Role == RoleTool || len(msg.Parts) == 0 {
			continue
		}
		if _, ok := msg.Parts[len(msg.Parts)-1].(TextPart); !ok {
			continue
		}
		if index != asked {
			if err := scope.substituted("config.cache.prefix_until_index",
				fmt.Sprintf("message %d is a %s message or does not end with text; the Responses wire marks text blocks of user/developer messages only, so the mark moved to the nearest eligible message before it", asked, req.Messages[asked].Role),
				asked, index); err != nil {
				return nil, err
			}
		}
		i := index
		return &i, nil
	}
	if err := scope.dropped("config.cache.prefix_until_index",
		fmt.Sprintf("no user/developer message ending with text at or before message %d; the Responses wire marks text blocks only (implicit caching still applies)", asked), asked); err != nil {
		return nil, err
	}
	return nil, nil
}

func cacheStablePrefix(req *Request, cacheControl string) bool {
	cfg := req.Config.Cache
	return cfg != nil && cfg.EffectiveMode() != "off" && cfg.Prefix == "stable" && cacheControl == "openai"
}

func hasExplicitBreakpoint(req *Request, cacheControl string, breakpoint *int) bool {
	return breakpoint != nil || (cacheStablePrefix(req, cacheControl) && req.System != nil)
}

// cacheCommonPayload adds the shared MAP-6 fields for both OpenAI dialects.
func cacheCommonPayload(req *Request, payload *JSONObject, cacheControl, provider string, breakpoint *int, scope *adaptScope) error {
	cfg := req.Config.Cache
	if cfg == nil {
		return nil
	}
	if cacheControl != "openai" && cacheControl != "openai_implicit" {
		// MAP-6 rule 7: a stored-cache resource where there is no such tier
		// RAISES; dropping it would send the request without the prompt
		// prefix it holds (MAP-13: refuse when a guess could hurt).
		if cfg.Resource != "" {
			return UnsupportedFeature(provider, "config.cache.resource", "%s: cache.resource is not supported — this provider has no stored-cache tier; sending without it would drop the prompt prefix the resource holds", provider)
		}
		// MAP-13: the key and the lifetime have no home on a server without
		// OpenAI's cache fields; dropped and recorded.
		if cfg.Key != "" {
			if err := scope.dropped("config.cache.key", "this server has no cache affinity field; implicit caching still applies", cfg.Key); err != nil {
				return err
			}
		}
		if cfg.Retention == "long" {
			if err := scope.dropped("config.cache.retention", "this server has no in-request cache lifetime knob; implicit caching still applies", "long"); err != nil {
				return err
			}
		}
		return nil
	}
	if cacheControl == "openai_implicit" {
		if cfg.EffectiveMode() != "off" {
			if cfg.Key != "" {
				payload.Set("prompt_cache_key", cfg.Key)
			}
			if cfg.Retention == "long" {
				payload.Set("prompt_cache_retention", "24h")
			}
		}
		if cfg.Resource != "" {
			return UnsupportedFeature(provider, "config.cache.resource", "%s: cache.resource is not supported — this provider has no stored-cache tier; it caches every prompt prefix automatically", provider)
		}
		return nil
	}
	if cfg.EffectiveMode() == "off" {
		if openaiModelHasCacheOptions(req.Model) {
			payload.Set("prompt_cache_options", JSONObject{{"mode", "explicit"}})
		}
		return nil
	}
	if cfg.Key != "" {
		payload.Set("prompt_cache_key", cfg.Key)
	}
	if cfg.Retention == "long" {
		payload.Set("prompt_cache_retention", "24h")
	}
	if openaiModelHasCacheOptions(req.Model) && hasExplicitBreakpoint(req, cacheControl, breakpoint) {
		payload.Set("prompt_cache_options", JSONObject{{"mode", "explicit"}})
	}
	if cfg.Resource != "" {
		return UnsupportedFeature(provider, "config.cache.resource", "%s: cache.resource is not supported — this provider has no stored-cache tier; it caches by marks on blocks (prefix / prefix_until_index) and automatically", provider)
	}
	return nil
}

func breakpointUnsupported(provider string, index int, role string) *Error {
	return UnsupportedFeature(provider, "config.cache.prefix_until_index", "%s: cache.prefix_until_index=%d points at a %s message whose last block is not text — the wire carries prompt_cache_breakpoint on text input blocks only. Point the prefix at a user/developer message that ends with text, or omit prefix_until_index (implicit caching still applies).", provider, index, role)
}

func responseFormatToOpenAIText(f JSONObject) JSONObject {
	if f.Get("type") == "json_object" {
		return JSONObject{{"format", JSONObject{{"type", "json_object"}}}}
	}
	name := wireStr(f.Get("name"))
	if name == "" {
		name = "response"
	}
	fmtObj := JSONObject{{"type", "json_schema"}, {"name", name}, {"schema", f.Get("schema")}}
	if strict, ok := f.Lookup("strict"); ok {
		fmtObj.Set("strict", strict)
	}
	return JSONObject{{"format", fmtObj}}
}

func openaiFinishFromStatus(data JSONObject, hasToolCall bool) string {
	if hasToolCall {
		return FinishToolCall
	}
	status := strings.ToLower(wireStr(data.Get("status")))
	reason := ""
	if inc := wireObj(data.Get("incomplete_details")); inc != nil {
		reason = strings.ToLower(wireStr(inc.Get("reason")))
	}
	if status == "incomplete" && strings.Contains(reason, "token") {
		return FinishLength
	}
	if strings.Contains(reason, "content_filter") || strings.Contains(reason, "safety") {
		return FinishContentFilter
	}
	return FinishStop
}

func openaiBatchStatus(status string) string {
	status = strings.ToLower(status)
	switch status {
	case "completed", "failed", "cancelled", "expired":
		return status
	case "cancelling", "canceling":
		return BatchCancelling
	case "in_progress", "finalizing":
		return BatchRunning
	}
	return BatchQueued
}

func annotationText(a JSONObject, source string, hasSource bool) string {
	for _, key := range []string{"text", "snippet", "cited_text", "quote"} {
		if s := wireStr(a.Get(key)); s != "" && a.Get(key) != nil {
			return s
		}
	}
	start, end := wireIntPtr(a.Get("start_index")), wireIntPtr(a.Get("end_index"))
	if hasSource && start != nil && end != nil && 0 <= *start && *start < *end && *end <= len(source) {
		return source[*start:*end]
	}
	return ""
}

func citationFromOpenAIAnnotation(a JSONObject, source string, hasSource bool) (CitationPart, bool) {
	url := firstStr(a.Get("url"), a.Get("uri"))
	title := firstStr(a.Get("title"), a.Get("filename"), a.Get("file_id"))
	text := annotationText(a, source, hasSource)
	if url == "" && title == "" && text == "" {
		return CitationPart{}, false
	}
	return CitationPart{URL: url, Title: title, Text: text}, true
}

func citationDeltaFromAnnotation(a JSONObject, partIndex int) (CitationDelta, bool) {
	c, ok := citationFromOpenAIAnnotation(a, "", false)
	if !ok {
		return CitationDelta{}, false
	}
	d := CitationDelta{PartIndex: partIndex}
	if c.Text != "" {
		d.Text = S(c.Text)
	}
	if c.URL != "" {
		d.URL = S(c.URL)
	}
	if c.Title != "" {
		d.Title = S(c.Title)
	}
	return d, true
}

func openaiUsage(usageData JSONObject) Usage {
	in := wireObj(usageData.Get("input_tokens_details"))
	if in == nil {
		in = wireObj(usageData.Get("input_token_details"))
	}
	out := wireObj(usageData.Get("output_tokens_details"))
	if out == nil {
		out = wireObj(usageData.Get("output_token_details"))
	}
	return Usage{
		InputTokens:       wireIntPtr(usageData.Get("input_tokens")),
		OutputTokens:      wireIntPtr(usageData.Get("output_tokens")),
		TotalTokens:       wireIntPtr(usageData.Get("total_tokens")),
		ReasoningTokens:   wireIntPtr(out.Get("reasoning_tokens")),
		CacheReadTokens:   wireIntPtr(in.Get("cached_tokens")),
		CacheWriteTokens:  wireIntPtr(in.Get("cache_write_tokens")),
		InputAudioTokens:  wireIntPtr(in.Get("audio_tokens")),
		OutputAudioTokens: wireIntPtr(out.Get("audio_tokens")),
	}.Normalize()
}

// OpenAI error envelope code tables (shared by the chat dialect).
var openaiResponseErrorCodeMap = map[string]ErrorKind{
	"server_error": KindServer, "rate_limit_exceeded": KindRateLimit, "invalid_prompt": KindInvalidRequest,
	"vector_store_timeout": KindTimeout, "invalid_image": KindInvalidRequest, "invalid_image_format": KindInvalidRequest,
	"invalid_base64_image": KindInvalidRequest, "invalid_image_url": KindInvalidRequest, "image_too_large": KindInvalidRequest,
	"image_too_small": KindInvalidRequest, "image_parse_error": KindInvalidRequest, "image_content_policy_violation": KindInvalidRequest,
	"invalid_image_mode": KindInvalidRequest, "image_file_too_large": KindInvalidRequest, "unsupported_image_media_type": KindInvalidRequest,
	"empty_image_file": KindInvalidRequest, "failed_to_download_image": KindInvalidRequest, "image_file_not_found": KindInvalidRequest,
	"model_not_found": KindUnsupportedModel, "model_not_available": KindUnsupportedModel, "unsupported_model": KindUnsupportedModel,
	"DeploymentNotFound": KindUnsupportedModel,
	// Azure documents these on Responses error frames even under HTTP 200.
	"no_capacity": KindRateLimit, "too_many_requests": KindRateLimit,
}

var openaiModelErrorCodes = map[string]bool{"model_not_found": true, "model_not_available": true, "unsupported_model": true, "DeploymentNotFound": true}

var openaiStreamErrorCodeMap = func() map[string]ErrorKind {
	m := map[string]ErrorKind{}
	for k, v := range openaiResponseErrorCodeMap {
		m[k] = v
	}
	m["context_length_exceeded"] = KindContextLength
	m["invalid_api_key"] = KindAuth
	m["insufficient_quota"] = KindBilling
	m["1113"] = KindBilling
	m["exceeded_current_quota_error"] = KindBilling
	m["authentication_error"] = KindAuth
	m["rate_limit_error"] = KindRateLimit
	return m
}()

var modelErrorMarkers = []string{"not found", "does not exist", "not exist", "not supported", "unsupported", "not available", "unknown"}

func isModelErrorMessage(message string, codes ...string) bool {
	lowered := strings.ToLower(strings.Join(append([]string{message}, codes...), " "))
	if !strings.Contains(lowered, "model") {
		return false
	}
	for _, m := range modelErrorMarkers {
		if strings.Contains(lowered, m) {
			return true
		}
	}
	return false
}

// openaiResponseError maps an in-body error envelope (complete path).
func openaiResponseError(c *lmCore, code, message string) *Error {
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
	return c.providerError(kind, msg, 0, code, "")
}

func openaiErrorDetail(providerCode, message string) ErrorDetail {
	kind, ok := openaiStreamErrorCodeMap[providerCode]
	if !ok {
		kind = KindProvider
	}
	if IsPinnedModelNotFound(providerCode, message) { // MAP-15
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

// openaiNormalizeError parses the OpenAI error envelope (shared by both
// OpenAI dialects and xAI).
func openaiNormalizeError(c *lmCore, status int, body string) *Error {
	if c.isCodex() {
		if e := codexDetailError(c, status, body); e != nil {
			return e
		}
	}
	data, err := DecodeJSON([]byte(body))
	if err != nil {
		return c.normalizeError(status, body)
	}
	obj := wireObj(data)
	var msg, code, errType string
	if obj != nil {
		switch e := jsonView(obj.Get("error")).(type) {
		case JSONObject:
			msg = wireStr(e.Get("message"))
			code = wireStr(e.Get("code"))
			errType = wireStr(e.Get("type"))
			if e.Get("code") == nil {
				code = ""
			}
			if e.Get("type") == nil {
				errType = ""
			}
		case nil:
		default:
			msg = wireStr(e)
		}
	}
	providerCode := code
	if providerCode == "" {
		providerCode = errType
	}
	switch {
	case code == "context_length_exceeded":
		return c.providerError(KindContextLength, msg, status, providerCode, "")
	case openaiModelErrorCodes[code] || (status == 404 && isModelErrorMessage(msg, code, errType)) || IsPinnedModelNotFound(providerCode, msg): // MAP-15
		return c.providerError(KindUnsupportedModel, msg, status, providerCode, "")
	case code == "insufficient_quota" || code == "1113" || errType == "insufficient_quota" || errType == "exceeded_current_quota_error":
		return c.providerError(KindBilling, msg, status, providerCode, "")
	case code == "invalid_api_key" || errType == "authentication_error":
		return c.providerError(KindAuth, msg, status, providerCode, "")
	case code == "rate_limit_exceeded" || errType == "rate_limit_error":
		return c.providerError(KindRateLimit, msg, status, providerCode, "")
	}
	if code != "" && !strings.Contains(msg, code) {
		msg = msg + " (" + code + ")"
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
	return c.withLoginHint(MapHTTPError(status, msg, c.provider, c.access.EnvKeys, providerCode, "", nil))
}

func codexDetailError(c *lmCore, status int, body string) *Error {
	data, err := DecodeJSON([]byte(body))
	if err != nil {
		return nil
	}
	obj := wireObj(data)
	if obj == nil {
		return nil
	}
	detail, ok := obj.Get("detail").(string)
	if !ok || strings.TrimSpace(detail) == "" {
		return nil
	}
	detail = strings.TrimSpace(detail)
	if isModelErrorMessage(detail) {
		return c.providerError(KindUnsupportedModel, detail+"\n"+modelListHint, status, "", "")
	}
	return c.withLoginHint(MapHTTPError(status, detail, c.provider, nil, "", "", nil))
}

func (l *OpenAILM) normalizeError(status int, body string) *Error {
	return openaiNormalizeError(&l.lmCore, status, body)
}

func (l *OpenAILM) headers(contentType string) [][2]string {
	if contentType == "" {
		contentType = "application/json"
	}
	headers := [][2]string{{"Content-Type", contentType}}
	if l.isCodex() && l.accountID != "" {
		headers = append(headers, [2]string{"chatgpt-account-id", l.accountID})
	}
	headers = append(headers, l.access.Headers...)
	return headers
}

func (l *OpenAILM) compat(req *Request) ResolvedOpenAIResponsesCompat {
	return ResolveOpenAIResponsesCompat(MergeOpenAIResponsesCompat(l.compatBase, openaiResponsesCompatFromExtensions(req.Config.Extensions)))
}

// ─── Request serialization ───────────────────────────────────────────

func (l *OpenAILM) buildInput(messages []Message, compat ResolvedOpenAIResponsesCompat, breakpoint *int) ([]any, error) {
	var items []any
	for msgIndex, msg := range messages {
		atBreakpoint := breakpoint != nil && *breakpoint == msgIndex
		if atBreakpoint && (msg.Role == RoleAssistant || msg.Role == RoleTool) {
			return nil, breakpointUnsupported(l.provider, msgIndex, msg.Role)
		}
		if msg.Role == RoleTool {
			for _, part := range msg.Parts {
				tr, ok := part.(ToolResultPart)
				if !ok {
					continue
				}
				output, err := toolResultOutputOpenAI(l.provider, tr, compat.ToolResultMedia)
				if err != nil {
					return nil, err
				}
				item := JSONObject{{"type", "function_call_output"}, {"call_id", tr.ID}, {"output", output}}
				if compat.ToolResultName == "include" && tr.Name != "" {
					item.Set("name", tr.Name)
				}
				items = append(items, item)
			}
			continue
		}
		var content []JSONObject
		if msg.Role == RoleAssistant {
			for _, part := range msg.Parts {
				switch p := part.(type) {
				case TextPart:
					content = append(content, JSONObject{{"type", "output_text"}, {"text", p.Text}})
				case RefusalPart:
					content = append(content, JSONObject{{"type", "refusal"}, {"refusal", p.Text}})
				case ThinkingPart:
					state := ContinuationData(p.Continuation, "openai", "reasoning_item")
					if len(state) > 0 {
						item := JSONObject{{"type", "reasoning"}}
						for _, k := range []string{"id", "encrypted_content"} {
							if v, ok := state.Lookup(k); ok {
								item.Set(k, v)
							}
						}
						if p.Text != "" {
							item.Set("summary", []any{JSONObject{{"type", "summary_text"}, {"text", p.Text}}})
						} else {
							item.Set("summary", []any{})
						}
						items = append(items, item)
					} else if p.Text != "" {
						content = append(content, JSONObject{{"type", "output_text"}, {"text", p.Text}})
					}
				}
			}
		} else {
			for _, part := range msg.Parts {
				switch part.(type) {
				case ToolCallPart, ToolResultPart:
					continue
				}
				block, err := partToOpenAIInput(part, l.provider)
				if err != nil {
					return nil, err
				}
				content = append(content, block)
			}
		}
		if atBreakpoint {
			if len(content) == 0 || content[len(content)-1].Get("type") != "input_text" {
				return nil, breakpointUnsupported(l.provider, msgIndex, msg.Role)
			}
			content[len(content)-1].Set("prompt_cache_breakpoint", JSONObject{{"mode", "explicit"}})
		}
		if len(content) > 0 {
			role := msg.Role
			if role == RoleDeveloper {
				role = compat.DeveloperRole
			}
			item := JSONObject{{"role", role}, {"content", toAnyList(content, func(b JSONObject) any { return b })}}
			if compat.CommentaryPhase == "tag" && msg.Role == RoleAssistant && hasToolCall(msg.Parts) {
				item.Set("phase", "commentary")
			}
			items = append(items, item)
		}
		for _, part := range msg.Parts {
			if tc, ok := part.(ToolCallPart); ok {
				items = append(items, JSONObject{{"type", "function_call"}, {"call_id", tc.ID}, {"name", tc.Name}, {"arguments", jsonRaw(tc.Input)}})
			}
		}
	}
	if items == nil {
		items = []any{}
	}
	return items, nil
}

func hasToolCall(parts []Part) bool {
	for _, p := range parts {
		if _, ok := p.(ToolCallPart); ok {
			return true
		}
	}
	return false
}

func (l *OpenAILM) toolChoicePayload(req *Request, compat ResolvedOpenAIResponsesCompat) any {
	tc := req.Config.ToolChoice
	if tc == nil {
		return nil
	}
	if tc.EffectiveMode() == "none" {
		return "none"
	}
	if len(tc.Allowed) > 0 {
		entries := make([]Tool, 0, len(tc.Allowed))
		for _, name := range tc.Allowed {
			entries = append(entries, req.ToolByName(name))
		}
		if len(entries) == 1 && tc.EffectiveMode() == "required" {
			if bt, ok := entries[0].(BuiltinTool); ok {
				return JSONObject{{"type", builtinTypeOpenAI(bt, compat)}}
			}
			return JSONObject{{"type", "function"}, {"name", entries[0].ToolName()}}
		}
		var wire []any
		for _, t := range entries {
			if bt, ok := t.(BuiltinTool); ok {
				wire = append(wire, JSONObject{{"type", builtinTypeOpenAI(bt, compat)}})
			} else {
				wire = append(wire, JSONObject{{"type", "function"}, {"name", t.ToolName()}})
			}
		}
		return JSONObject{{"type", "allowed_tools"}, {"mode", tc.EffectiveMode()}, {"tools", wire}}
	}
	if tc.EffectiveMode() == "required" {
		return "required"
	}
	return "auto"
}

func (l *OpenAILM) payload(req *Request, stream bool, scope *adaptScope) (JSONObject, error) {
	if err := checkMessageMedia(req.Messages, "openai", l.provider); err != nil {
		return nil, err
	}
	compat := l.compat(req)
	breakpoint, err := cacheBreakpointIndex(req, compat.CacheControl, scope) // once: it may record
	if err != nil {
		return nil, err
	}
	input, err := l.buildInput(req.Messages, compat, breakpoint)
	if err != nil {
		return nil, err
	}
	payload := JSONObject{{"model", req.Model}, {"input", input}, {"stream", stream}}
	cfg := req.Config
	if req.System != nil {
		text, err := systemText(req.System, l.provider)
		if err != nil {
			return nil, err
		}
		if cacheStablePrefix(req, compat.CacheControl) {
			first := JSONObject{{"role", compat.DeveloperRole}, {"content", []any{JSONObject{{"type", "input_text"}, {"text", text}, {"prompt_cache_breakpoint", JSONObject{{"mode", "explicit"}}}}}}}
			payload.Set("input", append([]any{first}, input...))
		} else {
			payload.Set("instructions", text)
		}
	}
	if cfg.MaxTokens != nil {
		payload.Set(compat.MaxOutputTokensField, *cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		payload.Set("temperature", jsonFloat(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		payload.Set("top_p", jsonFloat(*cfg.TopP))
	}
	if len(cfg.Stop) > 0 {
		// MAP-13 client_side: the Responses wire has no stop field; the text
		// is cut at the first sequence after the wire (complete) or as it
		// streams (the source is closed at the cut).
		stop := toAnyList(cfg.Stop, func(s string) any { return s })
		if err := scope.clientSide("config.stop", "the Responses wire has no stop field; the reply is streamed and the connection closed at the first stop sequence (whether the provider then stops generating, and billing, is its own behaviour); the usage report rides only the final frame, so it is not reported when the cut happens (never estimated)", stop, stop); err != nil {
			return nil, err
		}
	}
	if cfg.TopK != nil {
		if err := scope.dropped("config.top_k", "the Responses wire has no top_k (Anthropic and Gemini carry it)", *cfg.TopK); err != nil {
			return nil, err
		}
	}
	for _, knob := range []struct {
		name string
		set  bool
		val  any
	}{{"seed", cfg.Seed != nil, deref(cfg.Seed)}, {"frequency_penalty", cfg.FrequencyPenalty != nil, deref(cfg.FrequencyPenalty)}, {"presence_penalty", cfg.PresencePenalty != nil, deref(cfg.PresencePenalty)}} {
		if knob.set {
			// The Responses API dropped these from the Chat Completions wire
			// (no field in the reference); the chat dialect carries them.
			if err := scope.dropped("config."+knob.name, "the Responses wire has no "+knob.name+" field (the Chat Completions dialect carries it)", knob.val); err != nil {
				return nil, err
			}
		}
	}
	if cfg.Logprobs != nil {
		payload.Set("top_logprobs", *cfg.Logprobs)
		payload.Set("include", []any{"message.output_text.logprobs"})
	}
	if len(req.Tools) > 0 {
		var tools []any
		for _, t := range req.Tools {
			switch x := t.(type) {
			case FunctionTool:
				tp := JSONObject{{"type", "function"}, {"name", x.Name}, {"description", nilIfEmpty(x.Description)}, {"parameters", x.EffectiveParameters()}}
				if compat.StrictTools == "include" {
					tp.Set("strict", false)
				}
				tools = append(tools, tp)
			case BuiltinTool:
				out := JSONObject{{"type", builtinTypeOpenAI(x, compat)}}
				for k, v := range x.Config.All() {
					out.Set(k, v)
				}
				tools = append(tools, out)
			}
		}
		payload.Set("tools", tools)
	}
	if tc := l.toolChoicePayload(req, compat); tc != nil {
		payload.Set("tool_choice", tc)
	}
	if cfg.ToolChoice != nil && cfg.ToolChoice.Parallel != nil {
		payload.Set("parallel_tool_calls", *cfg.ToolChoice.Parallel)
	}
	if len(cfg.ResponseFormat) > 0 {
		// MAP-14: the judgment convention goes verbatim (strict honours
		// anyOf/const/title, receipted 2026-09-17); probabilities cannot be
		// measured here.
		if err := noteUnmeasurableProbabilities(scope, req, l.provider); err != nil {
			return nil, err
		}
		payload.Set("text", responseFormatToOpenAIText(cfg.ResponseFormat))
	}
	if r := cfg.Reasoning; r != nil {
		if !r.IsOff() {
			if r.ThinkingBudget != nil {
				// MAP-13: effort carries the intent (MAP-7 rule 5); no budget
				// field exists on this wire.
				if err := scope.dropped("config.reasoning.thinking_budget", "this wire has no thinking token budget; effort carries the intent (Anthropic's manual class and Gemini take a budget)", *r.ThinkingBudget); err != nil {
					return nil, err
				}
			}
			summary := r.Summary
			if (summary == "concise" || summary == "detailed") && compat.ReasoningFormat != "responses_reasoning" {
				if err := scope.substituted("config.reasoning.summary", "this wire has no summary detail levels; 'auto' is what it shows", summary, "auto"); err != nil {
					return nil, err
				}
				summary = "auto"
			}
			switch compat.ReasoningFormat {
			case "responses_reasoning":
				rp := JSONObject{{"effort", r.Effort}}
				if summary != "" {
					rp.Set("summary", summary)
				}
				payload.Set("reasoning", rp)
			case "reasoning_effort":
				payload.Set("reasoning_effort", r.Effort)
			case "openrouter":
				payload.Set("reasoning", JSONObject{{"effort", r.Effort}})
			case "deepseek":
				payload.Set("thinking", JSONObject{{"type", "enabled"}})
				payload.Set("reasoning_effort", r.Effort)
			case "qwen", "zai":
				payload.Set("enable_thinking", true)
			case "qwen_chat_template":
				payload.Set("chat_template_kwargs", JSONObject{{"enable_thinking", true}, {"preserve_thinking", true}})
			}
		} else {
			switch compat.ReasoningFormat {
			case "responses_reasoning":
				payload.Set("reasoning", JSONObject{{"effort", "none"}})
			case "reasoning_effort":
				payload.Set("reasoning_effort", "none")
			case "openrouter":
				payload.Set("reasoning", JSONObject{{"enabled", false}})
			case "deepseek":
				payload.Set("thinking", JSONObject{{"type", "disabled"}})
			case "qwen", "zai":
				payload.Set("enable_thinking", false)
			case "qwen_chat_template":
				payload.Set("chat_template_kwargs", JSONObject{{"enable_thinking", false}})
			}
		}
	}
	if err := cacheCommonPayload(req, &payload, compat.CacheControl, l.provider, breakpoint, scope); err != nil {
		return nil, err
	}
	if compat.Routing != nil {
		payload.Set("provider", compat.Routing)
	}
	if cfg.ServiceTier != "" {
		payload.Set("service_tier", cfg.ServiceTier)
	}
	if cfg.UserID != "" {
		payload.Set("safety_identifier", cfg.UserID)
	}
	if cfg.Store != nil {
		payload.Set("store", *cfg.Store)
	}
	for k, v := range cfg.Extensions.All() {
		switch k {
		case "prompt_caching", "cache", "compat", "openai_compat", "openai_responses_compat":
			continue
		}
		payload.Set(k, v)
	}
	if l.isCodex() {
		// An explicit cap or store=true is refused, never stripped: dropping a
		// cap means unbounded spend (MAP-13 rule 4; Rust, R, Python and TS refuse the same).
		if req.Config.MaxTokens != nil {
			return nil, UnsupportedFeature(l.provider, "config.max_tokens", "%s: config.max_tokens: this backend has no output cap; dropping it risks unbounded paid generation", l.provider)
		}
		if req.Config.Store != nil && *req.Config.Store {
			return nil, UnsupportedFeature(l.provider, "config.store", "%s: config.store: this backend cannot store a retrievable response; the program may depend on retrieval", l.provider)
		}
		if l.access.SystemPrefix != "" {
			if _, has := payload.Lookup("instructions"); !has {
				payload.Set("instructions", l.access.SystemPrefix)
			}
		}
		payload.Set("store", false)
		payload.Set("stream", true)
		payload.Delete("max_output_tokens")
		payload.Delete("max_completion_tokens")
		payload.Delete("max_tokens")
	}
	return payload, nil
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func (l *OpenAILM) buildRequest(req *Request, stream bool, scope *adaptScope) (*TransportRequest, error) {
	payload, err := l.payload(req, stream, scope)
	if err != nil {
		return nil, err
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/responses", endpoint: "responses", stream: stream, model: req.Model, headers: l.headers(""), payload: payload, scope: scope})
}

// ─── Response parsing ────────────────────────────────────────────────

func (l *OpenAILM) parseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	data, err := l.jsonBody(resp)
	if err != nil {
		return nil, err
	}
	if e := wireObj(data.Get("error")); e != nil {
		return nil, openaiResponseError(&l.lmCore, wireStr(e.Get("code")), firstStr(e.Get("message"), mustJSONString(e)))
	}
	var parts []Part
	var unmapped []JSONObject
	var logprobs []TokenLogprob
	for i, rawItem := range wireList(data.Get("output")) {
		item := wireObj(rawItem)
		if item == nil {
			recordUnmapped(&unmapped, "output["+strconv.Itoa(i)+"]", jsonTypeName(rawItem))
			continue
		}
		itemType := wireStr(item.Get("type"))
		switch {
		case itemType == "message":
			for j, rawContent := range wireList(item.Get("content")) {
				content := wireObj(rawContent)
				path := "output[" + strconv.Itoa(i) + "].content[" + strconv.Itoa(j) + "]"
				if content == nil {
					recordUnmapped(&unmapped, path, jsonTypeName(rawContent))
					continue
				}
				switch ctype := wireStr(content.Get("type")); ctype {
				case "output_text", "text":
					text := wireStr(content.Get("text"))
					parts = append(parts, TextPart{Text: text})
					logprobs = append(logprobs, openaiTokenLogprobs(content.Get("logprobs"))...)
					for _, rawA := range wireList(content.Get("annotations")) {
						if a := wireObj(rawA); a != nil {
							if c, ok := citationFromOpenAIAnnotation(a, text, true); ok {
								parts = append(parts, c)
							}
						}
					}
				case "refusal":
					text := firstStr(content.Get("refusal"), content.Get("text"))
					if text != "" {
						parts = append(parts, RefusalPart{Text: text})
					} else {
						parts = append(parts, TextPart{})
					}
				case "output_image":
					if b64 := firstStr(content.Get("b64_json"), content.Get("image_base64")); b64 != "" {
						parts = append(parts, ImagePart{Media: Media{MediaType: "image/png", Data: b64}})
					}
				case "output_audio":
					audio := wireObj(content.Get("audio"))
					if b64 := firstStr(audio.Get("data"), content.Get("b64_json")); b64 != "" {
						parts = append(parts, AudioPart{Media: Media{MediaType: "audio/wav", Data: b64}})
					}
				default:
					recordUnmapped(&unmapped, path, content.Get("type"))
				}
			}
		case itemType == "function_call":
			if !truthy(item.Get("name")) {
				return nil, unnamedToolCallError(l.provider, "output["+strconv.Itoa(i)+"]")
			}
			id := firstStr(item.Get("call_id"), item.Get("id"))
			if id == "" {
				id = "call_" + strconv.Itoa(len(parts))
			}
			parts = append(parts, ToolCallPart{ID: id, Name: wireStr(item.Get("name")), Input: parseJSONObject(item.Get("arguments"))})
		case itemType == "reasoning":
			text := ""
			if summary, ok := item.Get("summary").([]any); ok {
				var lines []string
				for _, x := range summary {
					if obj := wireObj(x); obj != nil {
						lines = append(lines, wireStr(obj.Get("text")))
					} else {
						lines = append(lines, wireStr(x))
					}
				}
				text = strings.Join(lines, "\n")
			} else {
				text = firstStr(item.Get("summary"), item.Get("text"))
			}
			state := JSONObject{}
			if truthy(item.Get("id")) {
				state.Set("id", wireStr(item.Get("id")))
			}
			if truthy(item.Get("encrypted_content")) {
				state.Set("encrypted_content", wireStr(item.Get("encrypted_content")))
			}
			var continuation []ContinuationState
			if len(state) > 0 {
				continuation = []ContinuationState{{Provider: "openai", Kind: "reasoning_item", Data: state}}
			}
			if text != "" || len(continuation) > 0 {
				parts = append(parts, ThinkingPart{Text: text, Continuation: continuation})
			}
		case openaiProviderExecutedItems[itemType]:
		default:
			recordUnmapped(&unmapped, "output["+strconv.Itoa(i)+"]", item.Get("type"))
		}
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{Text: wireStr(data.Get("output_text"))}}
	}
	usage := openaiUsage(wireObj(data.Get("usage")))
	model := wireStr(data.Get("model"))
	if model == "" {
		model = req.Model
	}
	return &Response{
		ID:           wireStr(data.Get("id")),
		Model:        model,
		Message:      Message{Role: RoleAssistant, Parts: ReplaceTextWithData(parts, RequestJudgments(req))},
		FinishReason: openaiFinishFromStatus(data, hasToolCall(parts)),
		Usage:        usage,
		Logprobs:     logprobs,
		ProviderData: attachUnmapped(data, unmapped),
	}, nil
}

func mustJSONString(v any) string { return string(mustJSON(v)) }

func (l *OpenAILM) parseStreamEvents(req *Request, ev sse.Event) ([]StreamEvent, error) {
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
	et := wireStr(payload.Get("type"))
	outputIndex := wireInt(payload.Get("output_index"), 0)
	if et == "response.output_item.added" || et == "response.output_item.done" {
		if item := wireObj(payload.Get("item")); item != nil && wireStr(item.Get("type")) == "reasoning" {
			if et == "response.output_item.added" {
				return []StreamEvent{StreamDeltaEvent{Delta: ThinkingDelta{Text: "", PartIndex: outputIndex}}}, nil
			}
			state := JSONObject{}
			for _, k := range []string{"id", "encrypted_content"} {
				if truthy(item.Get(k)) {
					state.Set(k, item.Get(k))
				}
			}
			if len(state) > 0 {
				idx := outputIndex
				return []StreamEvent{StreamDeltaEvent{Delta: ContinuationDelta{Provider: "openai", Kind: "reasoning_item", Data: state, PartIndex: &idx}}}, nil
			}
			return nil, nil
		}
	}
	switch et {
	case "response.created":
		resp := wireObj(payload.Get("response"))
		model := wireStr(resp.Get("model"))
		if model == "" {
			model = req.Model
		}
		return []StreamEvent{StreamStartEvent{ID: wireStr(resp.Get("id")), Model: model}}, nil
	case "response.output_text.delta", "response.refusal.delta":
		return []StreamEvent{StreamDeltaEvent{Delta: TextDelta{Text: wireStr(payload.Get("delta")), PartIndex: outputIndex, Logprobs: openaiTokenLogprobs(payload.Get("logprobs"))}}}, nil
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		return []StreamEvent{StreamDeltaEvent{Delta: ThinkingDelta{Text: wireStr(payload.Get("delta")), PartIndex: outputIndex}}}, nil
	case "response.output_text.annotation.added":
		if a := wireObj(payload.Get("annotation")); a != nil {
			if d, ok := citationDeltaFromAnnotation(a, outputIndex); ok {
				return []StreamEvent{StreamDeltaEvent{Delta: d}}, nil
			}
		}
		return nil, nil
	case "response.output_audio.delta":
		return []StreamEvent{StreamDeltaEvent{Delta: AudioDelta{Data: S(wireStr(payload.Get("delta"))), PartIndex: outputIndex, MediaType: "audio/wav"}}}, nil
	case "response.output_image.delta", "response.image.delta":
		return []StreamEvent{StreamDeltaEvent{Delta: ImageDelta{Data: S(wireStr(payload.Get("delta"))), PartIndex: outputIndex, MediaType: "image/png"}}}, nil
	case "response.output_item.added":
		item := wireObj(payload.Get("item"))
		if wireStr(item.Get("type")) == "function_call" {
			return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: wireStr(item.Get("arguments")), PartIndex: outputIndex, ID: firstStr(item.Get("call_id"), item.Get("id")), Name: wireStr(item.Get("name"))}}}, nil
		}
		return nil, nil
	case "response.function_call_arguments.delta":
		return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: wireStr(payload.Get("delta")), PartIndex: outputIndex, ID: firstStr(payload.Get("call_id"), payload.Get("id")), Name: wireStr(payload.Get("name"))}}}, nil
	case "response.completed":
		resp := wireObj(payload.Get("response"))
		usage := openaiUsage(wireObj(resp.Get("usage")))
		hasTool := false
		for _, rawItem := range wireList(resp.Get("output")) {
			if item := wireObj(rawItem); item != nil && wireStr(item.Get("type")) == "function_call" {
				hasTool = true
			}
		}
		finish := FinishStop
		if hasTool {
			finish = FinishToolCall
		}
		return []StreamEvent{StreamEndEvent{FinishReason: finish, Usage: &usage, ProviderData: resp}}, nil
	case "response.error", "error":
		code, message := openaiStreamErrorFields(payload)
		return []StreamEvent{StreamErrorEvent{Error: openaiErrorDetail(code, message)}}, nil
	}
	return nil, nil
}

func openaiStreamErrorFields(payload JSONObject) (string, string) {
	if e := wireObj(payload.Get("error")); e != nil {
		code := firstStr(e.Get("code"), e.Get("type"), payload.Get("code"))
		if code == "" {
			code = "provider"
		}
		return code, firstStr(e.Get("message"), payload.Get("message"))
	}
	code := firstStr(payload.Get("code"), payload.Get("error_type"))
	if code == "" {
		code = "provider"
	}
	return code, wireStr(payload.Get("message"))
}

// ─── Codex / live overrides ──────────────────────────────────────────

func (l *OpenAILM) completeOverride(ctx context.Context, req *Request) (*Response, bool, error) {
	if !l.isCodex() {
		return nil, false, nil
	}
	resp, err := MaterializeResponse(l.Stream(ctx, req), req)
	return resp, true, err
}

func (l *OpenAILM) streamOverride(ctx context.Context, req *Request) (iter.Seq2[StreamEvent, error], bool) {
	if l.isCodex() || !l.shouldUseLiveCompletion(req) {
		return nil, false
	}
	return l.streamViaLiveCompletion(ctx, req), true
}

func (l *OpenAILM) shouldUseLiveCompletion(req *Request) bool {
	mode := strings.ToLower(wireStr(req.Config.Extensions.Get("transport")))
	if mode == "live" || mode == "websocket" || mode == "ws" {
		return true
	}
	model := strings.ToLower(req.Model)
	return strings.Contains(model, "realtime") || strings.Contains(model, "-live")
}

// ─── Models ──────────────────────────────────────────────────────────

func (l *OpenAILM) modelsRequest() (*TransportRequest, error) {
	var params map[string]string
	if l.isCodex() {
		params = map[string]string{"client_version": l.access.BackendOptions["client_version"]}
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/models", params: params, headers: l.headers("")})
}

func (l *OpenAILM) modelsFromBody(body string) ([]ModelInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	if l.isCodex() {
		return modelInfosFromEntries(data.Get("models"), l.provider, "openai_responses", func(e JSONObject) string { return stringOnly(e.Get("slug")) }), nil
	}
	return modelInfosFromEntries(data.Get("data"), l.provider, "openai_responses", func(e JSONObject) string { return stringOnly(e.Get("id")) }), nil
}

func stringOnly(v any) string {
	s, _ := v.(string)
	return s
}

// ─── Files ───────────────────────────────────────────────────────────

func (l *OpenAILM) fileUploadRequest(req *FileUploadRequest) (*TransportRequest, error) {
	purpose := "user_data"
	var fields [][2]string
	if p, ok := req.Extensions.Lookup("purpose"); ok {
		purpose = wireStr(p)
	}
	fields = append(fields, [2]string{"purpose", purpose})
	for k, v := range req.Extensions.All() {
		if k != "purpose" {
			fields = append(fields, [2]string{k, wireStr(v)})
		}
	}
	content, err := req.Content()
	if err != nil {
		return nil, err
	}
	ct, body := multipartFormBody(fields, []multipartFile{{Field: "file", Filename: req.Filename, ContentType: req.EffectiveMediaType(), Data: content}})
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/files", headers: l.headers(ct), body: body})
}

func (l *OpenAILM) fileInfo(data JSONObject) (FileInfo, error) {
	id := stringOnly(data.Get("id"))
	if id == "" {
		return FileInfo{}, l.providerError(KindProvider, "openai: file object carries no id", 0, "", "")
	}
	var size *int
	if v, ok := data.Lookup("bytes"); ok {
		if i, err := jsonInt(v, "bytes"); err == nil {
			size = &i
		}
	}
	return FileInfo{
		ID: id, Filename: stringOnly(data.Get("filename")), SizeBytes: size,
		CreatedAt: isoUTC(data.Get("created_at")), ExpiresAt: isoUTC(data.Get("expires_at")),
		Readiness: openaiFileReadinessOf(data.Get("status")), ProviderData: data,
	}, nil
}

func (l *OpenAILM) fileInfoFromBody(body string) (FileInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FileInfo{}, err
	}
	return l.fileInfo(data)
}

func (l *OpenAILM) fileGetRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(fileID, false), headers: l.headers("")})
}

func (l *OpenAILM) fileListRequest(limit int, cursor string) (*TransportRequest, error) {
	params := map[string]string{"limit": strconv.Itoa(limit)}
	if cursor != "" {
		params["after"] = cursor
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files", params: params, headers: l.headers("")})
}

func (l *OpenAILM) filePageFromListBody(body string) (FilePage, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FilePage{}, err
	}
	var items []FileInfo
	for _, e := range wireList(data.Get("data")) {
		if obj := wireObj(e); obj != nil {
			info, err := l.fileInfo(obj)
			if err != nil {
				return FilePage{}, err
			}
			items = append(items, info)
		}
	}
	cursor := ""
	if truthy(data.Get("has_more")) && len(items) > 0 {
		cursor = stringOnly(data.Get("last_id"))
	}
	return FilePage{Items: items, NextCursor: cursor}, nil
}

func (l *OpenAILM) fileDeleteRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "DELETE", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(fileID, false), headers: l.headers("")})
}

func (l *OpenAILM) fileDownloadRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(fileID, false) + "/content", headers: l.headers("")})
}

// ─── Batch ───────────────────────────────────────────────────────────

func (l *OpenAILM) batchUploadRequest(req *BatchRequest, scope *adaptScope) (*TransportRequest, error) {
	var lines []string
	for i, nested := range req.Requests {
		body, err := l.payload(nested, false, scope)
		if err != nil {
			return nil, err
		}
		lines = append(lines, jsonRaw(JSONObject{{"custom_id", strconv.Itoa(i)}, {"method", "POST"}, {"url", "/v1/responses"}, {"body", body}}))
	}
	data := []byte(strings.Join(lines, "\n") + "\n")
	ct, body := multipartFormBody([][2]string{{"purpose", "batch"}}, []multipartFile{{Field: "file", Filename: "lm15-batch.jsonl", ContentType: "application/jsonl", Data: data}})
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/files", headers: l.headers(ct), body: body, scope: scope})
}

func (l *OpenAILM) batchSubmitRequest(req *BatchRequest, uploadBody JSONObject, scope *adaptScope) (*TransportRequest, error) {
	inputFileID := stringOnly(uploadBody.Get("id"))
	if inputFileID == "" {
		return nil, l.providerError(KindProvider, "openai: batch input file upload returned no id", 0, "", "")
	}
	payload := JSONObject{{"input_file_id", inputFileID}, {"endpoint", "/v1/responses"}, {"completion_window", "24h"}}
	ext := copyObject(req.Extensions)
	if v, ok := ext.Lookup("endpoint"); ok {
		payload.Set("endpoint", v)
		ext.Delete("endpoint")
	}
	if v, ok := ext.Lookup("completion_window"); ok {
		payload.Set("completion_window", v)
		ext.Delete("completion_window")
	}
	if req.Label != "" {
		payload.Set("metadata", JSONObject{{"label", req.Label}})
	}
	for k, v := range ext.All() {
		payload.Set(k, v)
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/batches", headers: l.headers(""), payload: payload, scope: scope})
}

func (l *OpenAILM) batchJobInfo(data JSONObject) (BatchJobInfo, error) {
	id := stringOnly(data.Get("id"))
	if id == "" {
		return BatchJobInfo{}, l.providerError(KindProvider, "openai: batch object carries no id", 0, "", "")
	}
	label := stringOnly(wireObj(data.Get("metadata")).Get("label"))
	return BatchJobInfo{ID: id, Status: openaiBatchStatus(wireStr(data.Get("status"))), Label: label, CreatedAt: isoUTC(data.Get("created_at")), ProviderData: data}, nil
}

func (l *OpenAILM) batchJobFromBody(body string) (BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return BatchJobInfo{}, err
	}
	return l.batchJobInfo(data)
}

func (l *OpenAILM) batchStatusRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/batches/" + pathID(batchID, false), headers: l.headers("")})
}

func (l *OpenAILM) batchCancelRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/batches/" + pathID(batchID, false) + "/cancel", headers: l.headers("")})
}

func (l *OpenAILM) batchResultFetches(statusBody JSONObject) ([]*TransportRequest, error) {
	var out []*TransportRequest
	for _, key := range []string{"output_file_id", "error_file_id"} {
		if id := stringOnly(statusBody.Get(key)); id != "" {
			req, err := l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files/" + pathID(id, false) + "/content", headers: l.headers("")})
			if err != nil {
				return nil, err
			}
			out = append(out, req)
		}
	}
	return out, nil
}

func (l *OpenAILM) batchEntries(statusBody JSONObject, fetched []string) ([]BatchEntry, error) {
	jobStatus := openaiBatchStatus(wireStr(statusBody.Get("status")))
	found := map[int]BatchEntry{}
	maxIndex := -1
	for _, text := range fetched {
		for _, line := range strings.Split(text, "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			item, err := DecodeJSONObject([]byte(line))
			if err != nil {
				return nil, err
			}
			index, err := strconv.Atoi(wireStr(item.Get("custom_id")))
			if err != nil {
				return nil, valueErrorf("batch entry custom_id is not an int")
			}
			if index > maxIndex {
				maxIndex = index
			}
			respObj := wireObj(item.Get("response"))
			statusCode := wireInt(respObj.Get("status_code"), 0)
			bodyObj := wireObj(respObj.Get("body"))
			if statusCode == 200 && len(bodyObj) > 0 {
				resp, err := l.parseResponse(batchEntryRequest(stringOnly(bodyObj.Get("model"))), JSONResponse(200, bodyObj))
				if err != nil {
					return nil, err
				}
				found[index] = BatchEntry{Index: index, Outcome: "succeeded", Response: resp}
			} else {
				var errSource any = bodyObj
				if len(bodyObj) == 0 {
					errSource = item.Get("error")
					if errSource == nil {
						errSource = JSONObject{}
					}
				}
				if statusCode == 0 {
					statusCode = 400
				}
				e := l.normalizeError(statusCode, jsonRaw(errSource))
				msg := e.Message
				if msg == "" {
					msg = "batch entry errored"
				}
				found[index] = BatchEntry{Index: index, Outcome: "errored", Error: &ErrorDetail{Code: e.Code, Message: msg, ProviderCode: e.ProviderCode}}
			}
		}
	}
	total := wireInt(wireObj(statusBody.Get("request_counts")).Get("total"), 0)
	if total == 0 && maxIndex >= 0 {
		total = maxIndex + 1
	}
	fill := "errored"
	if jobStatus == BatchExpired {
		fill = "expired"
	} else if jobStatus == BatchCancelled {
		fill = "cancelled"
	}
	var entries []BatchEntry
	for i := 0; i < total; i++ {
		if e, ok := found[i]; ok {
			entries = append(entries, e)
		} else if fill == "errored" {
			entries = append(entries, BatchEntry{Index: i, Outcome: "errored", Error: &ErrorDetail{Code: CodeProvider, Message: "entry missing from batch output files"}})
		} else {
			entries = append(entries, BatchEntry{Index: i, Outcome: fill})
		}
	}
	return entries, nil
}

func (l *OpenAILM) batchListRequest(limit int) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/batches", params: map[string]string{"limit": strconv.Itoa(limit)}, headers: l.headers("")})
}

func (l *OpenAILM) batchJobsFromListBody(body string) ([]BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []BatchJobInfo
	for _, e := range wireList(data.Get("data")) {
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

// ─── Video (Sora) ────────────────────────────────────────────────────

var openaiVideoStatusMap = map[string]string{"queued": "queued", "in_progress": "running", "completed": "completed", "failed": "failed", "cancelled": "cancelled"}

func (l *OpenAILM) videoSubmitRequest(req *VideoGenerationRequest) (*TransportRequest, error) {
	if len(req.Images) > 0 {
		return nil, UnsupportedFeatureErrorf(l.provider, "openai: video input images (input_reference) are not mapped yet; use the provider door until the mapping is live-receipted")
	}
	payload := JSONObject{{"model", req.Model}, {"prompt", req.Prompt}}
	for k, v := range req.Extensions.All() {
		payload.Set(k, v)
	}
	if req.Seconds != nil {
		payload.Set("seconds", strconv.Itoa(*req.Seconds))
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/videos", headers: l.headers(""), payload: payload})
}

func (l *OpenAILM) videoJobInfo(data JSONObject) (VideoJobInfo, error) {
	id := stringOnly(data.Get("id"))
	if id == "" {
		return VideoJobInfo{}, l.providerError(KindProvider, "openai: video object carries no id", 0, "", "")
	}
	wireStatus := wireStr(data.Get("status"))
	status, ok := openaiVideoStatusMap[wireStatus]
	if !ok {
		return VideoJobInfo{}, l.providerError(KindProvider, "openai: unknown video status "+strconv.Quote(wireStatus), 0, "", "")
	}
	var progress *int
	if _, isBool := data.Get("progress").(bool); !isBool {
		if f, err := jsonFloat64(data.Get("progress"), ""); err == nil {
			p := int(f)
			progress = &p
		}
	}
	return VideoJobInfo{ID: id, Status: status, Progress: progress, CreatedAt: isoUTC(data.Get("created_at")), Model: stringOnly(data.Get("model")), ProviderData: data}, nil
}

func (l *OpenAILM) videoJobFromBody(body string, _ string) (VideoJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return VideoJobInfo{}, err
	}
	return l.videoJobInfo(data)
}

func (l *OpenAILM) videoStatusRequest(videoID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/videos/" + pathID(videoID, false), headers: l.headers("")})
}

func (l *OpenAILM) videoResultFetch(statusBody JSONObject) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/videos/" + pathID(wireStr(statusBody.Get("id")), false) + "/content", headers: l.headers("")})
}

func (l *OpenAILM) videoPart(_ JSONObject, fetched *HTTPResponse) (VideoPart, error) {
	if fetched == nil {
		return VideoPart{}, l.providerError(KindProvider, "openai: video content fetch is required", 0, "", "")
	}
	ct := contentTypeOf(fetched.Headers)
	if ct == "" {
		return VideoPart{}, l.providerError(KindProvider, "openai: video content carries no content-type", 0, "", "")
	}
	return VideoPart{Media: Media{MediaType: ct, Data: base64.StdEncoding.EncodeToString(fetched.Body)}}, nil
}

func (l *OpenAILM) videoListRequest(limit int, _ string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/videos", params: map[string]string{"limit": strconv.Itoa(limit)}, headers: l.headers("")})
}

func (l *OpenAILM) videoJobsFromListBody(body string) ([]VideoJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []VideoJobInfo
	for _, e := range wireList(data.Get("data")) {
		if obj := wireObj(e); obj != nil {
			info, err := l.videoJobInfo(obj)
			if err != nil {
				return nil, err
			}
			out = append(out, info)
		}
	}
	return out, nil
}

// ─── Generation ──────────────────────────────────────────────────────

func (l *OpenAILM) imageGenerateRequest(req *ImageGenerationRequest) (*TransportRequest, error) {
	base := strings.TrimRight(l.baseURL, "/")
	compat := ResolveOpenAIResponsesCompat(l.compatBase)
	if len(req.Images) == 0 {
		payload := JSONObject{{"model", req.Model}, {"prompt", req.Prompt}}
		if req.Size != "" {
			payload.Set("size", req.Size)
		}
		for k, v := range req.Extensions.All() {
			if v != nil {
				payload.Set(k, v)
			}
		}
		return l.emit(emitSpec{method: "POST", url: base + "/images/generations", headers: l.headers(""), payload: payload})
	}
	for _, img := range req.Images {
		if img.Data == "" && img.Path == "" {
			return nil, UnsupportedFeatureErrorf(l.provider, "openai: image edits take inline data or a local path; url/file_id-addressed input images have no wire slot")
		}
	}
	fields := [][2]string{{"model", req.Model}, {"prompt", req.Prompt}}
	if req.Size != "" {
		fields = append(fields, [2]string{"size", req.Size})
	}
	for k, v := range req.Extensions.All() {
		fields = append(fields, [2]string{k, wireStr(v)})
	}
	var files []multipartFile
	for i, img := range req.Images {
		data, err := img.Bytes()
		if err != nil {
			return nil, err
		}
		field := "image[]"
		if compat.EditImageField == "indexed" {
			field = "image[" + strconv.Itoa(i) + "]"
		}
		files = append(files, multipartFile{Field: field, Filename: "image-" + strconv.Itoa(i), ContentType: img.MediaType, Data: data})
	}
	ct, body := multipartFormBody(fields, files)
	return l.emit(emitSpec{method: "POST", url: base + "/images/edits", headers: l.headers(ct), body: body})
}

func (l *OpenAILM) imageGenerationFromResponse(_ *ImageGenerationRequest, resp *HTTPResponse) (ImageGenerationResponse, error) {
	data, err := l.jsonBody(resp)
	if err != nil {
		return ImageGenerationResponse{}, err
	}
	mediaType := ""
	if f := stringOnly(data.Get("output_format")); f != "" {
		mediaType = "image/" + f
	}
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	var images []ImagePart
	for _, e := range wireList(data.Get("data")) {
		item := wireObj(e)
		if item == nil {
			continue
		}
		if b64 := stringOnly(item.Get("b64_json")); b64 != "" {
			images = append(images, ImagePart{Media: Media{MediaType: mediaType, Data: b64}})
		} else if url := stringOnly(item.Get("url")); url != "" {
			images = append(images, ImagePart{Media: Media{MediaType: mediaType, URL: url}})
		}
	}
	u := wireObj(data.Get("usage"))
	usage := Usage{InputTokens: wireIntPtr(u.Get("input_tokens")), OutputTokens: wireIntPtr(u.Get("output_tokens")), TotalTokens: wireIntPtr(u.Get("total_tokens"))}.Normalize()
	out := ImageGenerationResponse{Images: images, Usage: usage, ProviderData: data}
	return out, out.Validate()
}

func (l *OpenAILM) speechGenerateRequest(req *SpeechGenerationRequest) (*TransportRequest, error) {
	payload := JSONObject{{"model", req.Model}, {"input", req.Prompt}}
	for k, v := range req.Extensions.All() {
		payload.Set(k, v)
	}
	if req.Voice != "" {
		payload.Set("voice", req.Voice)
	}
	if req.Format != "" {
		payload.Set("response_format", req.Format)
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/audio/speech", headers: l.headers(""), payload: payload})
}

func (l *OpenAILM) speechGenerationFromResponse(_ *SpeechGenerationRequest, resp *HTTPResponse) (SpeechGenerationResponse, error) {
	ct := contentTypeOf(resp.Headers)
	if ct == "" {
		return SpeechGenerationResponse{}, l.providerError(KindProvider, "openai: speech response carries no content-type", 0, "", "")
	}
	audio := AudioPart{Media: Media{MediaType: ct, Data: base64.StdEncoding.EncodeToString(resp.Body)}}
	return SpeechGenerationResponse{Audio: audio, ProviderData: JSONObject{{"content_type", ct}}}, nil
}
