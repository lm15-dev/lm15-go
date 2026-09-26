package lm15

import (
	"encoding/base64"
	"encoding/binary"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

// GeminiLM is the Google Gemini GenerateContent dialect.
type GeminiLM struct {
	lmCore
	uploadBaseURL string
}

const (
	geminiDefaultBaseURL   = "https://generativelanguage.googleapis.com/v1beta"
	geminiDefaultUploadURL = "https://generativelanguage.googleapis.com/upload/v1beta"
)

var geminiBuiltinMap = map[string]string{"web_search": "googleSearch", "code_execution": "codeExecution"}
var geminiProviderExecutedKeys = []string{"executableCode", "codeExecutionResult"}

// NewGeminiLM constructs the Gemini dialect (default policy GeminiAPI).
func NewGeminiLM(opts ...Option) (*GeminiLM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	lm := &GeminiLM{uploadBaseURL: geminiDefaultUploadURL}
	if o.uploadBaseURL != "" {
		lm.uploadBaseURL = o.uploadBaseURL
	}
	lm.apiKeyHeader = "x-goog-api-key"
	if err := lm.bindAccess(lm, GeminiAPI, o, geminiDefaultBaseURL); err != nil {
		return nil, err
	}
	return lm, nil
}

// GeminiLevelClass reports the Gemini 3.x class (thinkingLevel, no full off).
func GeminiLevelClass(model string) bool {
	lowered := strings.ToLower(model)
	return strings.HasPrefix(lowered, "models/gemini-3") || strings.HasPrefix(lowered, "gemini-3")
}

var geminiErrorStatusMap = map[string]ErrorKind{
	"INVALID_ARGUMENT": KindInvalidRequest, "FAILED_PRECONDITION": KindBilling, "PERMISSION_DENIED": KindAuth, "UNAUTHENTICATED": KindAuth,
	"NOT_FOUND": KindInvalidRequest, "RESOURCE_EXHAUSTED": KindRateLimit, "INTERNAL": KindServer, "UNAVAILABLE": KindServer, "DEADLINE_EXCEEDED": KindTimeout,
}

var geminiCandidateFinishErrors = map[string]bool{
	"SAFETY": true, "RECITATION": true, "LANGUAGE": true, "BLOCKLIST": true, "PROHIBITED_CONTENT": true, "SPII": true,
	"MALFORMED_FUNCTION_CALL": true, "IMAGE_SAFETY": true, "IMAGE_PROHIBITED_CONTENT": true, "IMAGE_OTHER": true, "NO_IMAGE": true,
	"IMAGE_RECITATION": true, "UNEXPECTED_TOOL_CALL": true, "TOO_MANY_TOOL_CALLS": true, "MISSING_THOUGHT_SIGNATURE": true, "MALFORMED_RESPONSE": true,
}

func geminiContextLengthMessage(msg string) bool {
	l := strings.ToLower(msg)
	return (strings.Contains(l, "token") && (strings.Contains(l, "limit") || strings.Contains(l, "exceed"))) ||
		strings.Contains(l, "too long") || strings.Contains(l, "context is too long") || strings.Contains(l, "context length")
}

func (l *GeminiLM) errorDetail(providerCode, message string) ErrorDetail {
	kind, ok := geminiErrorStatusMap[providerCode]
	if !ok {
		kind = KindProvider
	}
	if geminiContextLengthMessage(message) {
		kind = KindContextLength
	} else if providerCode == "NOT_FOUND" && isModelErrorMessage(message) {
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

func (l *GeminiLM) inbandError(data JSONObject) *Error {
	if pf := wireObj(data.Get("promptFeedback")); pf != nil {
		if reason := wireStr(pf.Get("blockReason")); reason != "" && reason != "BLOCK_REASON_UNSPECIFIED" {
			return l.providerError(KindInvalidRequest, "Prompt blocked: "+reason, 0, "promptFeedback", "")
		}
	}
	candidates := wireList(data.Get("candidates"))
	if len(candidates) > 0 {
		if c := wireObj(candidates[0]); c != nil {
			if fr := wireStr(c.Get("finishReason")); geminiCandidateFinishErrors[fr] {
				msg := wireStr(c.Get("finishMessage"))
				if msg == "" {
					msg = "Candidate blocked: " + fr
				}
				return l.providerError(KindInvalidRequest, msg, 0, fr, "")
			}
		}
	}
	return nil
}

func (l *GeminiLM) normalizeError(status int, body string) *Error {
	data, err := DecodeJSON([]byte(body))
	if err != nil {
		return l.lmCore.normalizeError(status, body)
	}
	obj := wireObj(data)
	var msg, errStatus string
	if obj != nil {
		switch e := jsonView(obj.Get("error")).(type) {
		case JSONObject:
			msg = wireStr(e.Get("message"))
			errStatus = wireStr(e.Get("status"))
		case nil:
		default:
			msg = wireStr(e)
		}
	}
	if geminiContextLengthMessage(msg) {
		return l.providerError(KindContextLength, msg, status, errStatus, "")
	}
	if errStatus == "NOT_FOUND" && isModelErrorMessage(msg) {
		return l.providerError(KindUnsupportedModel, msg, status, errStatus, "")
	}
	if kind, ok := geminiErrorStatusMap[errStatus]; ok {
		return l.providerError(kind, msg, status, errStatus, "")
	}
	if errStatus != "" && !strings.Contains(msg, errStatus) {
		msg = msg + " (" + errStatus + ")"
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
	return MapHTTPError(status, msg, l.provider, l.access.EnvKeys, errStatus, "", nil)
}

func (l *GeminiLM) modelPath(model string) string {
	model = quoteSafe(model, "/:@")
	if strings.HasPrefix(model, "models/") {
		return model
	}
	return "models/" + model
}

func (l *GeminiLM) authHeaders(extra [][2]string) [][2]string {
	return append([][2]string(nil), extra...)
}

func geminiNumber(v float64) any {
	if v == float64(int64(v)) {
		return int64(v)
	}
	return jsonFloat(v)
}

// geminiSchemaFields are the keys of Gemini's Schema object
// (generate-content#v1beta.Schema): what its OpenAPI fields parse.
var geminiSchemaFields = map[string]bool{
	"type": true, "format": true, "title": true, "description": true, "nullable": true, "enum": true,
	"maxItems": true, "minItems": true, "properties": true, "required": true, "minProperties": true,
	"maxProperties": true, "minLength": true, "maxLength": true, "pattern": true, "example": true,
	"anyOf": true, "propertyOrdering": true, "default": true, "items": true, "minimum": true, "maximum": true,
}

// jsonListOf reads any Go slice or array (a decoded []any, or a []string or
// []int a caller wrote) as a list; bytes are not a JSON list.
func jsonListOf(v any) ([]any, bool) {
	if l, ok := v.([]any); ok {
		return l, true
	}
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || (rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array) || rv.Type().Elem().Kind() == reflect.Uint8 {
		return nil, false
	}
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = rv.Index(i).Interface()
	}
	return out, true
}

// geminiOpenAPISchema is MAP-16: can Gemini's OpenAPI field
// (responseSchema, functionDeclarations[].parameters) carry schema? No,
// when a schema node — the root, a value of properties, items, an element
// of anyOf (or anyOf itself when it is one object) — is a boolean, has a
// key that is not a field of Gemini's Schema object, has a list type, or
// has an enum list with an element that is not a string: the OpenAPI field
// answers 400 there and the JSON Schema field accepts it (live 2026-09-26,
// lm15-contract mapping/gemini-schema-field.json). Anything else stays on
// the OpenAPI field; example and default are values, never walked.
func geminiOpenAPISchema(schema any) bool {
	stack := []any{schema}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if _, ok := node.(bool); ok {
			return false
		}
		obj, ok := jsonView(node).(JSONObject)
		if !ok {
			continue
		}
		for _, m := range obj {
			if !geminiSchemaFields[m.Key] {
				return false
			}
			switch m.Key {
			case "type":
				if _, isList := jsonListOf(m.Value); isList {
					return false
				}
			case "enum":
				if items, isList := jsonListOf(m.Value); isList {
					for _, item := range items {
						if _, isString := item.(string); !isString {
							return false
						}
					}
				}
			case "properties":
				if props, isObject := jsonView(m.Value).(JSONObject); isObject {
					for _, p := range props {
						stack = append(stack, p.Value)
					}
				}
			case "items":
				stack = append(stack, m.Value)
			case "anyOf":
				if items, isList := jsonListOf(m.Value); isList {
					stack = append(stack, items...)
				} else {
					stack = append(stack, m.Value)
				}
			}
		}
	}
	return true
}

// geminiFunctionDeclaration is MAP-16 for a tool: parameters or
// parametersJsonSchema, the schema verbatim either way (INV-002).
func geminiFunctionDeclaration(ft FunctionTool) JSONObject {
	params := ft.EffectiveParameters()
	field := "parameters"
	if !geminiOpenAPISchema(params) {
		field = "parametersJsonSchema"
	}
	return JSONObject{{"name", ft.Name}, {"description", nilIfEmpty(ft.Description)}, {field, params}}
}

func geminiResponseFormat(f JSONObject) JSONObject {
	if f.Get("type") == "json_object" {
		return JSONObject{{"responseMimeType", "application/json"}}
	}
	schema := f.Get("schema")
	field := "responseSchema"
	if !geminiOpenAPISchema(schema) {
		field = "responseJsonSchema"
	}
	return JSONObject{{"responseMimeType", "application/json"}, {field, schema}}
}

func modalityTokens(details any, modality string) *int {
	list, ok := details.([]any)
	if !ok {
		return nil
	}
	sum := 0
	found := false
	for _, e := range list {
		if obj := wireObj(e); obj != nil && wireStr(obj.Get("modality")) == modality {
			sum += wireInt(obj.Get("tokenCount"), 0)
			found = true
		}
	}
	if !found {
		return nil
	}
	return &sum
}

func geminiUsage(usage JSONObject, outputKeys ...string) Usage {
	if len(usage) == 0 {
		return Usage{}
	}
	zero := 0
	output := &zero
	for _, k := range outputKeys {
		if v, ok := usage.Lookup(k); ok {
			output = wireIntPtr(v)
			if output == nil {
				output = &zero
			}
			break
		}
	}
	input := wireIntPtr(usage.Get("promptTokenCount"))
	if input == nil {
		z := 0
		input = &z
	}
	outputDetails := usage.Get("candidatesTokensDetails")
	if outputDetails == nil {
		outputDetails = usage.Get("responseTokensDetails")
	}
	return Usage{
		InputTokens:       input,
		OutputTokens:      output,
		TotalTokens:       wireIntPtr(usage.Get("totalTokenCount")),
		CacheReadTokens:   wireIntPtr(usage.Get("cachedContentTokenCount")),
		ReasoningTokens:   wireIntPtr(usage.Get("thoughtsTokenCount")),
		InputAudioTokens:  modalityTokens(usage.Get("promptTokensDetails"), "AUDIO"),
		OutputAudioTokens: modalityTokens(outputDetails, "AUDIO"),
	}.Normalize()
}

func thoughtSignatureState(part JSONObject) []ContinuationState {
	sig, ok := part.Lookup("thoughtSignature")
	if !ok || sig == nil {
		return nil
	}
	return []ContinuationState{{Provider: "gemini", Kind: "thought_signature", Data: JSONObject{{"value", wireStr(sig)}}}}
}

func geminiFinish(reason string, hasToolCall bool) string {
	if hasToolCall {
		return FinishToolCall
	}
	switch strings.ToUpper(reason) {
	case "MAX_TOKENS":
		return FinishLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return FinishContentFilter
	}
	return FinishStop
}

func geminiBatchStatus(data JSONObject) string {
	state := strings.ToUpper(wireStr(wireObj(data.Get("metadata")).Get("state")))
	mapping := map[string]string{
		"BATCH_STATE_PENDING": BatchQueued, "BATCH_STATE_RUNNING": BatchRunning, "BATCH_STATE_CANCELLING": BatchCancelling,
		"BATCH_STATE_SUCCEEDED": BatchCompleted, "BATCH_STATE_FAILED": BatchFailed, "BATCH_STATE_CANCELLED": BatchCancelled, "BATCH_STATE_EXPIRED": BatchExpired,
	}
	if s, ok := mapping[state]; ok {
		return s
	}
	if truthy(data.Get("done")) {
		return BatchCompleted
	}
	return BatchQueued
}

func geminiTokenLogprobs(result any) []TokenLogprob {
	lr := wireObj(result)
	if lr == nil {
		return nil
	}
	chosen := wireList(lr.Get("chosenCandidates"))
	topSteps := wireList(lr.Get("topCandidates"))
	var out []TokenLogprob
	for i, c := range chosen {
		cand := wireObj(c)
		if cand == nil {
			continue
		}
		var top []TopLogprob
		if i < len(topSteps) {
			for _, a := range wireList(wireObj(topSteps[i]).Get("candidates")) {
				if alt := wireObj(a); alt != nil {
					top = append(top, TopLogprob{Token: wireStr(alt.Get("token")), Logprob: wireFloat(alt.Get("logProbability"), 0), TokenID: wireIntPtr(alt.Get("tokenId"))})
				}
			}
		}
		out = append(out, TokenLogprob{Token: wireStr(cand.Get("token")), Logprob: wireFloat(cand.Get("logProbability"), 0), TokenID: wireIntPtr(cand.Get("tokenId")), Top: top})
	}
	return out
}

func geminiSegmentText(segment JSONObject, full string) string {
	if t := stringOnly(segment.Get("text")); t != "" {
		return t
	}
	start, end := wireIntPtr(segment.Get("startIndex")), wireIntPtr(segment.Get("endIndex"))
	if start != nil && end != nil && 0 <= *start && *start < *end && *end <= len(full) {
		return full[*start:*end]
	}
	return ""
}

func geminiCitations(candidate JSONObject, full string) []Part {
	grounding := wireObj(candidate.Get("groundingMetadata"))
	if grounding == nil {
		return nil
	}
	chunks := wireList(grounding.Get("groundingChunks"))
	supports, ok := grounding.Get("groundingSupports").([]any)
	if !ok {
		return nil
	}
	seen := map[[3]string]bool{}
	var out []Part
	for _, s := range supports {
		support := wireObj(s)
		if support == nil {
			continue
		}
		cited := geminiSegmentText(wireObj(support.Get("segment")), full)
		indices, ok := support.Get("groundingChunkIndices").([]any)
		if !ok {
			continue
		}
		for _, raw := range indices {
			idx := wireIntPtr(raw)
			var chunk JSONObject
			if idx != nil && *idx >= 0 && *idx < len(chunks) {
				chunk = wireObj(chunks[*idx])
			}
			source := wireObj(chunk.Get("web"))
			if source == nil {
				source = wireObj(chunk.Get("retrievedContext"))
			}
			if source == nil {
				source = wireObj(chunk.Get("googleSearch"))
			}
			url := firstStr(source.Get("uri"), source.Get("url"))
			title := firstStr(source.Get("title"), source.Get("name"))
			key := [3]string{url, title, cited}
			if seen[key] || (url == "" && title == "" && cited == "") {
				continue
			}
			seen[key] = true
			out = append(out, CitationPart{URL: url, Title: title, Text: cited})
		}
	}
	return out
}

// ─── Request serialization ───────────────────────────────────────────

func (l *GeminiLM) functionResponse(part ToolResultPart, names map[string]string) (JSONObject, error) {
	var textParts, mediaParts []Part
	for _, p := range part.Content {
		if IsMediaPart(p) {
			if p.Type() != "image" && p.Type() != "document" {
				return nil, UnsupportedFeature(l.provider, "messages[*].tool_result["+part.ID+"].content["+p.Type()+"]", "%s: a %s part in tool_result %q cannot reach a functionResponse — multimodal function responses take images (png/jpeg/webp) and documents (pdf, text/plain) only (MAP-10)", l.provider, p.Type(), part.ID)
			}
			mediaParts = append(mediaParts, p)
		} else {
			textParts = append(textParts, p)
		}
	}
	name := part.Name
	if name == "" {
		name = names[part.ID]
	}
	if name == "" {
		return nil, UnsupportedFeature(l.provider, "messages[*].tool_result["+part.ID+"].name", "%s: tool_result %q needs a function name on the Gemini wire and no preceding assistant tool_call with that id is in the transcript; set ToolResultPart.name (MAP-10 rule 6)", l.provider, part.ID)
	}
	text, err := partsToText(textParts, l.provider, "functionResponse.response")
	if err != nil {
		return nil, err
	}
	var response JSONObject
	switch {
	case part.IsError:
		response = JSONObject{{"error", text}}
	case len(mediaParts) > 0 && len(textParts) == 0:
		response = JSONObject{}
	default:
		response = JSONObject{{"result", text}}
	}
	fr := JSONObject{{"name", name}, {"response", response}}
	if part.ID != "" {
		fr.Set("id", part.ID)
	}
	if len(mediaParts) > 0 {
		var blocks []any
		for _, p := range mediaParts {
			b, err := l.part(p, nil)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, b)
		}
		fr.Set("parts", blocks)
	}
	return JSONObject{{"functionResponse", fr}}, nil
}

func (l *GeminiLM) part(p Part, names map[string]string) (JSONObject, error) {
	switch x := p.(type) {
	case TextPart:
		out := JSONObject{{"text", x.Text}}
		if thought := ContinuationData(x.Continuation, "gemini", "thought_signature"); thought != nil && truthy(thought.Get("value")) {
			out.Set("thoughtSignature", thought.Get("value"))
		}
		return out, nil
	case DataPart:
		// 2026-09-19 D3: a data part on a text wire is its compact JSON.
		return JSONObject{{"text", DataPartText(x)}}, nil
	case ToolCallPart:
		fc := JSONObject{{"name", x.Name}, {"args", x.Input}}
		if x.ID != "" {
			fc.Set("id", x.ID)
		}
		out := JSONObject{{"functionCall", fc}}
		if thought := ContinuationData(x.Continuation, "gemini", "thought_signature"); thought != nil && truthy(thought.Get("value")) {
			out.Set("thoughtSignature", thought.Get("value"))
		}
		return out, nil
	case ToolResultPart:
		return l.functionResponse(x, names)
	case ThinkingPart:
		out := JSONObject{{"text", x.Text}}
		if thought := ContinuationData(x.Continuation, "gemini", "thought_signature"); thought != nil && truthy(thought.Get("value")) {
			out.Set("thought", true)
			out.Set("thoughtSignature", thought.Get("value"))
		}
		return out, nil
	}
	if m, ok := MediaOf(p); ok {
		mime := m.MediaType
		if mime == "" {
			mime = "application/octet-stream"
		}
		switch {
		case m.URL != "":
			return JSONObject{{"fileData", JSONObject{{"mimeType", mime}, {"fileUri", m.URL}}}}, nil
		case m.FileID != "":
			return JSONObject{{"fileData", JSONObject{{"mimeType", mime}, {"fileUri", m.FileID}}}}, nil
		default:
			b64, err := m.Base64()
			if err != nil {
				return nil, err
			}
			return JSONObject{{"inlineData", JSONObject{{"mimeType", mime}, {"data", b64}}}}, nil
		}
	}
	return JSONObject{{"text", partText(p)}}, nil
}

func (l *GeminiLM) message(msg Message, names map[string]string) (JSONObject, error) {
	if msg.Role == RoleDeveloper {
		text, err := partsToText(msg.Parts, l.provider, "a developer turn")
		if err != nil {
			return nil, err
		}
		return JSONObject{{"role", "user"}, {"parts", []any{JSONObject{{"text", "[developer]\n" + text}}}}}, nil
	}
	role := "user"
	if msg.Role == RoleAssistant {
		role = "model"
	}
	var parts []any
	for _, p := range msg.Parts {
		b, err := l.part(p, names)
		if err != nil {
			return nil, err
		}
		parts = append(parts, b)
	}
	return JSONObject{{"role", role}, {"parts", parts}}, nil
}

func callNames(messages []Message) map[string]string {
	names := map[string]string{}
	for _, m := range messages {
		for _, p := range m.Parts {
			if tc, ok := p.(ToolCallPart); ok && tc.ID != "" {
				names[tc.ID] = tc.Name
			}
		}
	}
	return names
}

func (l *GeminiLM) toolConfigPayload(req *Request, scope *adaptScope) (JSONObject, error) {
	tc := req.Config.ToolChoice
	if tc == nil {
		return nil, nil
	}
	if tc.Parallel != nil && !*tc.Parallel {
		// MAP-13: no wire knob (live 2026-09-02: two calls came back on 2.5
		// and 3.7 with the preference set). A preference; agent loops
		// iterate tool-call parts as a list anyway. Dropped and recorded.
		if err := scope.dropped("config.tool_choice.parallel", "GenerateContent has no parallel-tool-calls knob and may return several calls (OpenAI and Anthropic carry it)", false); err != nil {
			return nil, err
		}
	}
	mode := map[string]string{"none": "NONE", "required": "ANY", "auto": "AUTO"}[tc.EffectiveMode()]
	cfg := JSONObject{{"mode", mode}}
	if len(tc.Allowed) > 0 {
		var builtins []string
		for _, name := range tc.Allowed {
			if _, ok := req.ToolByName(name).(BuiltinTool); ok {
				builtins = append(builtins, name)
			}
		}
		if len(builtins) > 0 {
			// MAP-13 rule 4(b): the program depends on the forced tool running.
			return nil, UnsupportedFeature(l.provider, "config.tool_choice.allowed", "gemini: cannot force builtin tools %v — functionCallingConfig addresses function declarations only; googleSearch/codeExecution have no tool_choice form (OpenAI Responses and Anthropic carry builtin forcing)", builtins)
		}
		cfg.Set("allowedFunctionNames", toAnyList(tc.Allowed, func(s string) any { return s }))
		if tc.EffectiveMode() == "auto" {
			cfg.Set("mode", "VALIDATED")
		}
	}
	return JSONObject{{"functionCallingConfig", cfg}}, nil
}

func (l *GeminiLM) cacheResource(cacheID string) string {
	if strings.HasPrefix(cacheID, "cachedContents/") {
		return cacheID
	}
	return "cachedContents/" + cacheID
}

func (l *GeminiLM) payload(req *Request, scope *adaptScope) (JSONObject, error) {
	cfg := req.Config
	ext := copyObject(cfg.Extensions)
	resource := ""
	suffixFrom := 0
	if c := cfg.Cache; c != nil && c.EffectiveMode() != "off" {
		if c.Key != "" {
			if err := scope.dropped("config.cache.key", "GenerateContent has no cache affinity key; implicit caching applies, and a stored cache (lm.cache(prefix), cache.resource) is the explicit tier", c.Key); err != nil {
				return nil, err
			}
		}
		if c.Retention != "" && c.Retention != "short" {
			if err := scope.dropped("config.cache.retention", "GenerateContent takes no lifetime in-request; it belongs to the stored cache (cache_create(..., ttl_seconds=...) / cache_update)", c.Retention); err != nil {
				return nil, err
			}
		}
		if c.Resource != "" {
			resource = c.Resource
			if c.PrefixUntilIndex != nil {
				idx := *c.PrefixUntilIndex
				if idx > len(req.Messages)-1 {
					idx = len(req.Messages) - 1
				}
				suffixFrom = idx + 1
			}
		}
	}
	wireMessages := req.Messages[suffixFrom:]
	if resource != "" && len(wireMessages) == 0 {
		return nil, valueErrorf("gemini: a request against a stored cache needs at least one message after the prefix")
	}
	names := callNames(req.Messages)
	var contents []any
	for _, m := range wireMessages {
		wm, err := l.message(m, names)
		if err != nil {
			return nil, err
		}
		contents = append(contents, wm)
	}
	if contents == nil {
		contents = []any{}
	}
	payload := JSONObject{{"contents", contents}}
	if resource != "" {
		payload.Set("cachedContent", l.cacheResource(resource))
	}
	if req.System != nil && resource == "" {
		text, err := systemText(req.System, l.provider)
		if err != nil {
			return nil, err
		}
		payload.Set("systemInstruction", JSONObject{{"parts", []any{JSONObject{{"text", text}}}}})
	}
	gen := JSONObject{}
	if cfg.Temperature != nil {
		gen.Set("temperature", geminiNumber(*cfg.Temperature))
	}
	if cfg.MaxTokens != nil {
		gen.Set("maxOutputTokens", *cfg.MaxTokens)
	}
	if cfg.TopP != nil {
		gen.Set("topP", geminiNumber(*cfg.TopP))
	}
	if cfg.TopK != nil {
		gen.Set("topK", *cfg.TopK)
	}
	if len(cfg.Stop) > 0 {
		gen.Set("stopSequences", toAnyList(cfg.Stop, func(s string) any { return s }))
	}
	if cfg.Seed != nil {
		gen.Set("seed", *cfg.Seed)
	}
	if cfg.FrequencyPenalty != nil {
		gen.Set("frequencyPenalty", geminiNumber(*cfg.FrequencyPenalty))
	}
	if cfg.PresencePenalty != nil {
		gen.Set("presencePenalty", geminiNumber(*cfg.PresencePenalty))
	}
	if cfg.Logprobs != nil {
		gen.Set("responseLogprobs", true)
		if *cfg.Logprobs > 0 {
			gen.Set("logprobs", *cfg.Logprobs)
		}
	}
	if len(cfg.ResponseFormat) > 0 {
		// MAP-14 §2: judgment properties go as enum with descriptions folded
		// into the property description — responseJsonSchema ignores
		// anyOf/const (receipted 2026-09-17); probabilities cannot be
		// measured here.
		if err := noteUnmeasurableProbabilities(scope, req, l.provider); err != nil {
			return nil, err
		}
		format := cfg.ResponseFormat
		if found := RequestJudgments(req); len(found) > 0 {
			if schema, ok := asObject(format.Get("schema")); ok {
				format = copyObject(format)
				format.Set("schema", geminiSchema(schema, found))
			}
		}
		for k, v := range geminiResponseFormat(format).All() {
			gen.Set(k, v)
		}
	}
	if r := cfg.Reasoning; r != nil {
		rr := *r
		r = &rr
		levelClass := GeminiLevelClass(req.Model)
		if r.IsOff() && levelClass {
			// MAP-13 (decision 2026-09-14 §4.2): the Gemini 3 class has no
			// honoured off switch (thinkingBudget 0 accepted, 58 tokens still
			// spent on 3.7 Flash, live 2026-09-02); the closest to "none" is
			// the lowest level, and the spend shows in usage.reasoning_tokens.
			if err := scope.substituted("config.reasoning.effort", req.Model+" cannot disable thinking (the Gemini 3 class honours no off switch); the lowest level was sent and the thinking spend is visible in usage", "off", "minimal"); err != nil {
				return nil, err
			}
			r.Effort = "minimal"
		}
		if r.IsOff() {
			gen.Set("thinkingConfig", JSONObject{{"thinkingBudget", 0}})
		} else {
			if r.Summary == "concise" || r.Summary == "detailed" {
				if err := scope.substituted("config.reasoning.summary", "GenerateContent has includeThoughts only, no detail levels; 'auto' shows the thoughts", r.Summary, "auto"); err != nil {
					return nil, err
				}
				r.Summary = "auto"
			}
			thinking := JSONObject{}
			if r.Summary != "" {
				thinking.Set("includeThoughts", true)
			}
			switch {
			case r.ThinkingBudget != nil:
				thinking.Set("thinkingBudget", *r.ThinkingBudget)
			case levelClass:
				effort := r.Effort
				if effort == "xhigh" || effort == "max" {
					if err := scope.clamped("config.reasoning.effort", "the Gemini 3 class has thinkingLevel minimal|low|medium|high; 'high' is the ceiling", effort, "high"); err != nil {
						return nil, err
					}
					effort = "high"
				}
				thinking.Set("thinkingLevel", effort)
			default:
				thinking.Set("thinkingBudget", EffortThinkingBudgets[r.Effort])
			}
			gen.Set("thinkingConfig", thinking)
		}
	}
	if len(gen) > 0 {
		payload.Set("generationConfig", gen)
	}
	if len(req.Tools) > 0 && resource == "" {
		var declarations []any
		var tools []any
		for _, t := range req.Tools {
			if ft, ok := t.(FunctionTool); ok {
				declarations = append(declarations, geminiFunctionDeclaration(ft))
			}
		}
		if len(declarations) > 0 {
			tools = append(tools, JSONObject{{"functionDeclarations", declarations}})
		}
		for _, t := range req.Tools {
			if bt, ok := t.(BuiltinTool); ok {
				tools = append(tools, geminiBuiltin(bt))
			}
		}
		payload.Set("tools", tools)
	}
	if resource == "" {
		tc, err := l.toolConfigPayload(req, scope)
		if err != nil {
			return nil, err
		}
		if tc != nil {
			payload.Set("toolConfig", tc)
		}
	}
	switch wireStr(ext.Get("output")) {
	case "image":
		setIn(&payload, []any{"IMAGE"}, "generationConfig", "responseModalities")
	case "audio":
		setIn(&payload, []any{"AUDIO"}, "generationConfig", "responseModalities")
	}
	if cfg.Store != nil {
		payload.Set("store", *cfg.Store)
	}
	if cfg.ServiceTier != "" {
		payload.Set("serviceTier", cfg.ServiceTier)
	}
	if cfg.UserID != "" {
		// MAP-13 (decision 2026-09-14 §4.5): attribution has no field here
		// and nothing in the program depends on it at run time; a
		// compliance policy sets adaptations="refuse".
		if err := scope.dropped("config.user_id", "GenerateContent has no end-user attribution field (OpenAI and Anthropic carry it)", cfg.UserID); err != nil {
			return nil, err
		}
	}
	for k, v := range ext.All() {
		if k != "prompt_caching" && k != "output" {
			payload.Set(k, v)
		}
	}
	return payload, nil
}

func geminiBuiltin(t BuiltinTool) JSONObject {
	key := t.Name
	if mapped, ok := geminiBuiltinMap[t.Name]; ok {
		key = mapped
	}
	cfg := t.Config
	if cfg == nil {
		cfg = JSONObject{}
	}
	return JSONObject{{key, cfg}}
}

func (l *GeminiLM) buildRequest(req *Request, stream bool, scope *adaptScope) (*TransportRequest, error) {
	payload, err := l.payload(req, scope)
	if err != nil {
		return nil, err
	}
	endpoint := "generateContent"
	var params map[string]string
	if stream {
		endpoint = "streamGenerateContent"
		params = map[string]string{"alt": "sse"}
	}
	return l.emit(emitSpec{
		method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(req.Model) + ":" + endpoint,
		endpoint: "generateContent", stream: stream, model: req.Model,
		headers: [][2]string{{"Content-Type", "application/json"}}, params: params, payload: payload, scope: scope,
	})
}

// ─── Response parsing ────────────────────────────────────────────────

func (l *GeminiLM) parseCandidateParts(partsPayload []any, unmapped *[]JSONObject, pathPrefix string) ([]Part, error) {
	var parts []Part
	for i, raw := range partsPayload {
		part := wireObj(raw)
		path := pathPrefix + "[" + strconv.Itoa(i) + "]"
		if part == nil {
			if unmapped != nil {
				recordUnmapped(unmapped, path, jsonTypeName(raw))
			}
			continue
		}
		_, hasText := part.Lookup("text")
		switch {
		case truthy(part.Get("thought")) && hasText:
			parts = append(parts, ThinkingPart{Text: wireStr(part.Get("text")), Continuation: thoughtSignatureState(part)})
		case hasText:
			parts = append(parts, TextPart{Text: wireStr(part.Get("text")), Continuation: thoughtSignatureState(part)})
		case wireObj(part.Get("functionCall")) != nil:
			fc := wireObj(part.Get("functionCall"))
			var continuation []ContinuationState
			sig := part.Get("thoughtSignature")
			if sig == nil {
				sig = fc.Get("thoughtSignature")
			}
			if sig != nil {
				continuation = []ContinuationState{{Provider: "gemini", Kind: "thought_signature", Data: JSONObject{{"value", wireStr(sig)}}}}
			}
			if !truthy(fc.Get("name")) {
				return nil, unnamedToolCallError(l.provider, path)
			}
			id := wireStr(fc.Get("id"))
			if id == "" || fc.Get("id") == nil {
				id = "tool_call_" + strconv.Itoa(len(parts))
			}
			args := wireObj(fc.Get("args"))
			if args == nil {
				args = JSONObject{}
			}
			parts = append(parts, ToolCallPart{ID: id, Name: wireStr(fc.Get("name")), Input: args, Continuation: continuation})
		case wireObj(part.Get("inlineData")) != nil:
			inline := wireObj(part.Get("inlineData"))
			mime := wireStr(inline.Get("mimeType"))
			if mime == "" {
				mime = "application/octet-stream"
			}
			data := wireStr(inline.Get("data"))
			if data == "" {
				continue
			}
			parts = append(parts, mediaPartFor(mime, Media{MediaType: mime, Data: data}))
		case wireObj(part.Get("fileData")) != nil:
			fd := wireObj(part.Get("fileData"))
			uri := wireStr(fd.Get("fileUri"))
			mime := wireStr(fd.Get("mimeType"))
			if mime == "" {
				mime = "application/octet-stream"
			}
			if uri == "" {
				continue
			}
			parts = append(parts, mediaPartFor(mime, Media{MediaType: mime, URL: uri}))
		case hasAnyKey(part, geminiProviderExecutedKeys):
		default:
			if unmapped != nil {
				keys := sortedKeys(part)
				t := strings.Join(keys, "+")
				if t == "" {
					t = "<empty>"
				}
				recordUnmapped(unmapped, path, t)
			}
		}
	}
	return parts, nil
}

func mediaPartFor(mime string, m Media) Part {
	switch {
	case strings.HasPrefix(mime, "image/"):
		return ImagePart{Media: m}
	case strings.HasPrefix(mime, "audio/"):
		return AudioPart{Media: m}
	}
	return DocumentPart{Media: m}
}

func hasAnyKey(m JSONObject, keys []string) bool {
	for _, k := range keys {
		if _, ok := m.Lookup(k); ok {
			return true
		}
	}
	return false
}

func (l *GeminiLM) parseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	data, err := l.jsonBody(resp)
	if err != nil {
		return nil, err
	}
	if e := l.inbandError(data); e != nil {
		return nil, e
	}
	var candidate JSONObject
	if cands := wireList(data.Get("candidates")); len(cands) > 0 {
		candidate = wireObj(cands[0])
	}
	content := wireObj(candidate.Get("content"))
	var unmapped []JSONObject
	parts, err := l.parseCandidateParts(wireList(content.Get("parts")), &unmapped, "candidates[0].content.parts")
	if err != nil {
		return nil, err
	}
	var full strings.Builder
	for _, p := range parts {
		if t, ok := p.(TextPart); ok {
			full.WriteString(t.Text)
		}
	}
	parts = append(parts, geminiCitations(candidate, full.String())...)
	if len(parts) == 0 {
		parts = []Part{TextPart{}}
	}
	return &Response{
		ID:           wireStr(data.Get("responseId")),
		Model:        req.Model,
		Message:      Message{Role: RoleAssistant, Parts: ReplaceTextWithData(parts, RequestJudgments(req))},
		FinishReason: geminiFinish(wireStr(candidate.Get("finishReason")), hasToolCall(parts)),
		Usage:        geminiUsage(wireObj(data.Get("usageMetadata")), "candidatesTokenCount", "responseTokenCount"),
		Logprobs:     geminiTokenLogprobs(candidate.Get("logprobsResult")),
		ProviderData: attachUnmapped(data, unmapped),
	}, nil
}

func (l *GeminiLM) usageFromPayload(payload JSONObject) Usage {
	return geminiUsage(wireObj(payload.Get("usageMetadata")), "candidatesTokenCount", "responseTokenCount")
}

func (l *GeminiLM) parseStreamEvents(_ *Request, ev sse.Event) ([]StreamEvent, error) {
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
	if errRaw, ok := payload.Lookup("error"); ok {
		e := wireObj(errRaw)
		code := firstStr(e.Get("status"), e.Get("code"))
		if code == "" {
			code = "provider"
		}
		return []StreamEvent{StreamErrorEvent{Error: l.errorDetail(code, wireStr(e.Get("message")))}}, nil
	}
	if inband := l.inbandError(payload); inband != nil {
		return []StreamEvent{StreamErrorEvent{Error: ErrorDetail{Code: inband.Code, ProviderCode: "inband_finish_reason", Message: inband.Error()}}}, nil
	}
	var events []StreamEvent
	var candidate JSONObject
	if cands := wireList(payload.Get("candidates")); len(cands) > 0 {
		candidate = wireObj(cands[0])
	}
	yielded := false
	sawTool := false
	finish := ""
	if candidate != nil {
		content := wireObj(candidate.Get("content"))
		chunkLogprobs := geminiTokenLogprobs(candidate.Get("logprobsResult"))
		for idx, raw := range wireList(content.Get("parts")) {
			part := wireObj(raw)
			if part == nil {
				continue
			}
			i := idx
			signatureDelta := func(sig any) StreamEvent {
				return StreamDeltaEvent{Delta: ContinuationDelta{Provider: "gemini", Kind: "thought_signature", Data: JSONObject{{"value", wireStr(sig)}}, PartIndex: &i}}
			}
			_, hasText := part.Lookup("text")
			switch {
			case truthy(part.Get("thought")) && hasText:
				yielded = true
				events = append(events, StreamDeltaEvent{Delta: ThinkingDelta{Text: wireStr(part.Get("text")), PartIndex: idx}})
				if part.Get("thoughtSignature") != nil {
					events = append(events, signatureDelta(part.Get("thoughtSignature")))
				}
			case hasText:
				yielded = true
				events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: wireStr(part.Get("text")), PartIndex: idx, Logprobs: chunkLogprobs}})
				chunkLogprobs = nil
				if part.Get("thoughtSignature") != nil {
					events = append(events, signatureDelta(part.Get("thoughtSignature")))
				}
			case wireObj(part.Get("functionCall")) != nil:
				fc := wireObj(part.Get("functionCall"))
				sawTool = true
				yielded = true
				args := fc.Get("args")
				if args == nil {
					args = JSONObject{}
				}
				events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{Input: jsonRaw(args), PartIndex: idx, ID: wireStr(fc.Get("id")), Name: wireStr(fc.Get("name"))}})
				sig := part.Get("thoughtSignature")
				if sig == nil {
					sig = fc.Get("thoughtSignature")
				}
				if sig != nil {
					events = append(events, signatureDelta(sig))
				}
			case wireObj(part.Get("inlineData")) != nil:
				inline := wireObj(part.Get("inlineData"))
				mime := wireStr(inline.Get("mimeType"))
				if mime == "" {
					mime = "application/octet-stream"
				}
				data := wireStr(inline.Get("data"))
				if strings.HasPrefix(mime, "audio/") {
					yielded = true
					events = append(events, StreamDeltaEvent{Delta: AudioDelta{Data: S(data), PartIndex: idx, MediaType: mime}})
				} else if strings.HasPrefix(mime, "image/") {
					yielded = true
					events = append(events, StreamDeltaEvent{Delta: ImageDelta{Data: S(data), PartIndex: idx, MediaType: mime}})
				}
			}
		}
		finish = wireStr(candidate.Get("finishReason"))
	}
	if finish != "" {
		usage := l.usageFromPayload(payload)
		events = append(events, StreamEndEvent{FinishReason: geminiFinish(finish, sawTool), Usage: &usage, ProviderData: payload})
	} else if _, has := payload.Lookup("usageMetadata"); !yielded && has {
		usage := l.usageFromPayload(payload)
		events = append(events, StreamEndEvent{FinishReason: FinishStop, Usage: &usage, ProviderData: payload})
	}
	return events, nil
}

// ─── Models ──────────────────────────────────────────────────────────

func (l *GeminiLM) modelsRequest() (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/models", params: map[string]string{"pageSize": "1000"}})
}

func (l *GeminiLM) modelsFromBody(body string) ([]ModelInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	return modelInfosFromEntries(data.Get("models"), l.provider, "gemini_generate_content", func(e JSONObject) string {
		return strings.TrimPrefix(stringOnly(e.Get("name")), "models/")
	}), nil
}

// ─── Files ───────────────────────────────────────────────────────────

func geminiFileResource(fileID string) string {
	if strings.Contains(fileID, "://") {
		trimmed := strings.TrimRight(fileID, "/")
		if i := strings.LastIndex(trimmed, "/files/"); i >= 0 && trimmed[i+len("/files/"):] != "" {
			return "files/" + trimmed[i+len("/files/"):]
		}
	}
	if strings.HasPrefix(fileID, "files/") {
		return fileID
	}
	return "files/" + fileID
}

func (l *GeminiLM) fileUploadRequest(req *FileUploadRequest) (*TransportRequest, error) {
	params := map[string]string{}
	for k, v := range req.Extensions.All() {
		if v != nil {
			params[k] = wireStr(v)
		}
	}
	content, err := req.Content()
	if err != nil {
		return nil, err
	}
	ct, body := multipartRelatedBody(JSONObject{{"file", JSONObject{{"display_name", req.Filename}}}}, req.EffectiveMediaType(), content)
	return l.emit(emitSpec{method: "POST", url: buildURL(strings.TrimRight(l.uploadBaseURL, "/")+"/files", params), headers: [][2]string{{"X-Goog-Upload-Protocol", "multipart"}, {"Content-Type", ct}}, body: body})
}

func (l *GeminiLM) fileInfo(data JSONObject) (FileInfo, error) {
	id := stringOnly(data.Get("uri"))
	if id == "" {
		id = stringOnly(data.Get("name"))
	}
	if id == "" {
		return FileInfo{}, l.providerError(KindProvider, "gemini: file object carries no uri or name", 0, "", "")
	}
	state := wireStr(data.Get("state"))
	readiness := "ready"
	if strings.HasSuffix(state, "PROCESSING") {
		readiness = "pending"
	} else if strings.HasSuffix(state, "FAILED") {
		readiness = "failed"
	}
	var downloadable *bool
	if stringOnly(data.Get("downloadUri")) != "" {
		downloadable = B(true)
	} else if data.Get("source") == "UPLOADED" {
		downloadable = B(false)
	}
	var size *int
	switch raw := data.Get("sizeBytes").(type) {
	case string:
		if i, err := strconv.Atoi(raw); err == nil {
			size = &i
		}
	case bool:
	default:
		if raw != nil {
			if i, err := jsonInt(raw, ""); err == nil {
				size = &i
			}
		}
	}
	return FileInfo{
		ID: id, Filename: stringOnly(data.Get("displayName")), MediaType: stringOnly(data.Get("mimeType")), SizeBytes: size,
		CreatedAt: isoUTC(data.Get("createTime")), ExpiresAt: isoUTC(data.Get("expirationTime")), Readiness: readiness, Downloadable: downloadable, ProviderData: data,
	}, nil
}

func (l *GeminiLM) fileInfoFromBody(body string) (FileInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FileInfo{}, err
	}
	if f := wireObj(data.Get("file")); f != nil {
		return l.fileInfo(f)
	}
	return l.fileInfo(data)
}

func (l *GeminiLM) fileGetRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(geminiFileResource(fileID), true)})
}

func (l *GeminiLM) fileListRequest(limit int, cursor string) (*TransportRequest, error) {
	params := map[string]string{"pageSize": strconv.Itoa(limit)}
	if cursor != "" {
		params["pageToken"] = cursor
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files", params: params})
}

func (l *GeminiLM) filePageFromListBody(body string) (FilePage, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FilePage{}, err
	}
	var items []FileInfo
	for _, e := range wireList(data.Get("files")) {
		if obj := wireObj(e); obj != nil {
			info, err := l.fileInfo(obj)
			if err != nil {
				return FilePage{}, err
			}
			items = append(items, info)
		}
	}
	return FilePage{Items: items, NextCursor: stringOnly(data.Get("nextPageToken"))}, nil
}

func (l *GeminiLM) fileDeleteRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "DELETE", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(geminiFileResource(fileID), true)})
}

func (l *GeminiLM) fileDownloadRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(geminiFileResource(fileID), true) + ":download", params: map[string]string{"alt": "media"}})
}

// ─── Cache resources ─────────────────────────────────────────────────

func (l *GeminiLM) cacheCreateRequest(prefix *Request, ttlSeconds *int, label string) (*TransportRequest, error) {
	names := callNames(prefix.Messages)
	var contents []any
	for _, m := range prefix.Messages {
		wm, err := l.message(m, names)
		if err != nil {
			return nil, err
		}
		contents = append(contents, wm)
	}
	body := JSONObject{{"model", l.modelPath(prefix.Model)}, {"contents", contents}}
	if prefix.System != nil {
		text, err := systemText(prefix.System, l.provider)
		if err != nil {
			return nil, err
		}
		body.Set("systemInstruction", JSONObject{{"parts", []any{JSONObject{{"text", text}}}}})
	}
	if len(prefix.Tools) > 0 {
		var declarations, tools []any
		for _, t := range prefix.Tools {
			if ft, ok := t.(FunctionTool); ok {
				declarations = append(declarations, geminiFunctionDeclaration(ft))
			}
		}
		if len(declarations) > 0 {
			tools = append(tools, JSONObject{{"functionDeclarations", declarations}})
		}
		for _, t := range prefix.Tools {
			if bt, ok := t.(BuiltinTool); ok {
				tools = append(tools, geminiBuiltin(bt))
			}
		}
		body.Set("tools", tools)
	}
	if ttlSeconds != nil {
		body.Set("ttl", strconv.Itoa(*ttlSeconds)+"s")
	}
	if label != "" {
		body.Set("displayName", label)
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/cachedContents", headers: [][2]string{{"Content-Type", "application/json"}}, payload: body})
}

func (l *GeminiLM) cacheInfo(data JSONObject) (CacheInfo, error) {
	name := stringOnly(data.Get("name"))
	if name == "" {
		return CacheInfo{}, l.providerError(KindProvider, "gemini: cache object carries no name", 0, "", "")
	}
	model := strings.TrimPrefix(wireStr(data.Get("model")), "models/")
	if model == "" {
		return CacheInfo{}, l.providerError(KindProvider, "gemini: cache object carries no model", 0, "", "")
	}
	var tokens *int
	switch raw := wireObj(data.Get("usageMetadata")).Get("totalTokenCount").(type) {
	case string:
		if i, err := strconv.Atoi(raw); err == nil && i >= 0 {
			tokens = &i
		}
	case bool, nil:
	default:
		if i, err := jsonInt(raw, ""); err == nil && i >= 0 {
			tokens = &i
		}
	}
	return CacheInfo{ID: name, Model: model, Tokens: tokens, CreatedAt: isoUTC(data.Get("createTime")), ExpiresAt: isoUTC(data.Get("expireTime")), Label: stringOnly(data.Get("displayName")), ProviderData: data}, nil
}

func (l *GeminiLM) cacheInfoFromBody(body string) (CacheInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return CacheInfo{}, err
	}
	return l.cacheInfo(data)
}

func (l *GeminiLM) cacheGetRequest(cacheID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(l.cacheResource(cacheID), true)})
}

func (l *GeminiLM) cacheListRequest(limit int, cursor string) (*TransportRequest, error) {
	params := map[string]string{"pageSize": strconv.Itoa(limit)}
	if cursor != "" {
		params["pageToken"] = cursor
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/cachedContents", params: params})
}

func (l *GeminiLM) cachePageFromListBody(body string) (CachePage, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return CachePage{}, err
	}
	var items []CacheInfo
	for _, e := range wireList(data.Get("cachedContents")) {
		if obj := wireObj(e); obj != nil {
			info, err := l.cacheInfo(obj)
			if err != nil {
				return CachePage{}, err
			}
			items = append(items, info)
		}
	}
	return CachePage{Items: items, NextCursor: stringOnly(data.Get("nextPageToken"))}, nil
}

func (l *GeminiLM) cacheDeleteRequest(cacheID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "DELETE", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(l.cacheResource(cacheID), true)})
}

func (l *GeminiLM) cacheUpdateRequest(cacheID string, ttlSeconds int) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "PATCH", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(l.cacheResource(cacheID), true), headers: [][2]string{{"Content-Type", "application/json"}}, payload: JSONObject{{"ttl", strconv.Itoa(ttlSeconds) + "s"}}})
}

// ─── Batch ───────────────────────────────────────────────────────────

func (l *GeminiLM) batchSubmitRequest(req *BatchRequest, _ JSONObject, scope *adaptScope) (*TransportRequest, error) {
	model := req.EffectiveModel()
	var requests []any
	for i, nested := range req.Requests {
		p, err := l.payload(nested, scope)
		if err != nil {
			return nil, err
		}
		requests = append(requests, JSONObject{{"request", p}, {"metadata", JSONObject{{"key", strconv.Itoa(i)}}}})
	}
	batch := JSONObject{{"inputConfig", JSONObject{{"requests", JSONObject{{"requests", requests}}}}}}
	if req.Label != "" {
		batch.Set("displayName", req.Label)
	}
	payload := JSONObject{{"batch", batch}}
	for k, v := range req.Extensions.All() {
		payload.Set(k, v)
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(model) + ":batchGenerateContent", headers: [][2]string{{"Content-Type", "application/json"}}, payload: payload, scope: scope})
}

func (l *GeminiLM) batchJobInfo(data JSONObject) (BatchJobInfo, error) {
	name := stringOnly(data.Get("name"))
	if name == "" {
		return BatchJobInfo{}, l.providerError(KindProvider, "gemini: batch operation carries no name", 0, "", "")
	}
	metadata := wireObj(data.Get("metadata"))
	return BatchJobInfo{ID: name, Status: geminiBatchStatus(data), Label: stringOnly(metadata.Get("displayName")), CreatedAt: isoUTC(metadata.Get("createTime")), ProviderData: data}, nil
}

func (l *GeminiLM) batchJobFromBody(body string) (BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return BatchJobInfo{}, err
	}
	return l.batchJobInfo(data)
}

func (l *GeminiLM) batchStatusRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(batchID, true)})
}

func (l *GeminiLM) batchCancelRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(batchID, true) + ":cancel", headers: [][2]string{{"Content-Type", "application/json"}}, payload: JSONObject{}})
}

func (l *GeminiLM) batchResultFetches(JSONObject) ([]*TransportRequest, error) { return nil, nil }

func (l *GeminiLM) batchEntries(statusBody JSONObject, _ []string) ([]BatchEntry, error) {
	response := wireObj(statusBody.Get("response"))
	var inlined []any
	switch x := jsonView(response.Get("inlinedResponses")).(type) {
	case JSONObject:
		inlined = wireList(x.Get("inlinedResponses"))
	case []any:
		inlined = x
	}
	var entries []BatchEntry
	for position, raw := range inlined {
		item := wireObj(raw)
		if item == nil {
			continue
		}
		index := position
		if key := wireStr(wireObj(item.Get("metadata")).Get("key")); key != "" {
			if i, err := strconv.Atoi(key); err == nil {
				index = i
			}
		}
		if body := wireObj(item.Get("response")); body != nil {
			resp, err := l.parseResponse(batchEntryRequest(stringOnly(body.Get("modelVersion"))), JSONResponse(200, body))
			if err != nil {
				return nil, err
			}
			entries = append(entries, BatchEntry{Index: index, Outcome: "succeeded", Response: resp})
		} else {
			e := wireObj(item.Get("error"))
			msg := wireStr(e.Get("message"))
			if msg == "" {
				msg = "batch entry errored"
			}
			code := ""
			if e.Get("status") != nil {
				code = wireStr(e.Get("status"))
			} else if e.Get("code") != nil {
				code = wireStr(e.Get("code"))
			}
			entries = append(entries, BatchEntry{Index: index, Outcome: "errored", Error: &ErrorDetail{Code: CodeProvider, Message: msg, ProviderCode: code}})
		}
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].Index < entries[j].Index })
	return entries, nil
}

func (l *GeminiLM) batchListRequest(limit int) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/batches", params: map[string]string{"pageSize": strconv.Itoa(limit)}})
}

func (l *GeminiLM) batchJobsFromListBody(body string) ([]BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []BatchJobInfo
	for _, e := range wireList(data.Get("operations")) {
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

// ─── Video (Veo) ─────────────────────────────────────────────────────

func (l *GeminiLM) videoSubmitRequest(req *VideoGenerationRequest) (*TransportRequest, error) {
	if len(req.Images) > 0 {
		return nil, UnsupportedFeatureErrorf(l.provider, "gemini: video input images are not mapped yet; use extensions until the mapping is live-receipted")
	}
	payload := JSONObject{{"instances", []any{JSONObject{{"prompt", req.Prompt}}}}}
	for k, v := range req.Extensions.All() {
		payload.Set(k, v)
	}
	if req.Seconds != nil {
		if _, has := payload.Lookup("parameters"); !has {
			payload.Set("parameters", JSONObject{{"durationSeconds", *req.Seconds}})
		}
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(req.Model) + ":predictLongRunning", headers: [][2]string{{"Content-Type", "application/json"}}, payload: payload})
}

func (l *GeminiLM) videoJobInfo(data JSONObject) (VideoJobInfo, error) {
	name := stringOnly(data.Get("name"))
	if name == "" {
		return VideoJobInfo{}, l.providerError(KindProvider, "gemini: video operation carries no name", 0, "", "")
	}
	status := "running"
	if data.Get("done") == true {
		status = "completed"
		if wireObj(data.Get("error")) != nil {
			status = "failed"
		}
	}
	return VideoJobInfo{ID: name, Status: status, ProviderData: data}, nil
}

func (l *GeminiLM) videoJobFromBody(body string, _ string) (VideoJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return VideoJobInfo{}, err
	}
	return l.videoJobInfo(data)
}

func (l *GeminiLM) videoStatusRequest(videoID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(videoID, true)})
}

func (l *GeminiLM) videoResultURI(statusBody JSONObject) (string, error) {
	gvr := wireObj(wireObj(statusBody.Get("response")).Get("generateVideoResponse"))
	if samples := wireList(gvr.Get("generatedSamples")); len(samples) > 0 {
		if uri := stringOnly(wireObj(wireObj(samples[0]).Get("video")).Get("uri")); uri != "" {
			return uri, nil
		}
	}
	return "", l.providerError(KindProvider, "gemini: terminal video operation carries no video uri", 0, "", "")
}

func (l *GeminiLM) videoResultFetch(statusBody JSONObject) (*TransportRequest, error) {
	uri, err := l.videoResultURI(statusBody)
	if err != nil {
		return nil, err
	}
	return l.emit(emitSpec{method: "GET", url: uri})
}

func (l *GeminiLM) videoPart(_ JSONObject, fetched *HTTPResponse) (VideoPart, error) {
	if fetched == nil {
		return VideoPart{}, l.providerError(KindProvider, "gemini: video content fetch is required", 0, "", "")
	}
	ct := contentTypeOf(fetched.Headers)
	if ct == "" {
		return VideoPart{}, l.providerError(KindProvider, "gemini: video download carries no content-type", 0, "", "")
	}
	return VideoPart{Media: Media{MediaType: ct, Data: base64.StdEncoding.EncodeToString(fetched.Body)}}, nil
}

func (l *GeminiLM) videoListRequest(limit int, model string) (*TransportRequest, error) {
	if model == "" {
		return nil, UnsupportedFeatureErrorf(l.provider, "gemini: video jobs list per model — pass model= (operations live under models/<model>/operations)")
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(model) + "/operations", params: map[string]string{"pageSize": strconv.Itoa(limit)}})
}

func (l *GeminiLM) videoJobsFromListBody(body string) ([]VideoJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []VideoJobInfo
	for _, e := range wireList(data.Get("operations")) {
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

// ─── Generation (image / speech through generateContent) ─────────────

func (l *GeminiLM) imageGenerationLMRequest(req *ImageGenerationRequest) *Request {
	ext := copyObject(req.Extensions)
	if req.Size != "" {
		gen := copyObject(wireObj(ext.Get("generationConfig")))
		imageCfg := copyObject(wireObj(gen.Get("imageConfig")))
		if _, has := imageCfg.Lookup("aspectRatio"); !has {
			imageCfg.Set("aspectRatio", req.Size)
		}
		gen.Set("imageConfig", imageCfg)
		ext.Set("generationConfig", gen)
	}
	parts := []Part{TextPart{Text: req.Prompt}}
	for _, img := range req.Images {
		parts = append(parts, img)
	}
	cfg := Config{}
	if len(ext) > 0 {
		cfg.Extensions = ext
	}
	return &Request{Model: req.Model, Messages: []Message{{Role: RoleUser, Parts: parts}}, Config: cfg}
}

func (l *GeminiLM) imageGenerateRequest(req *ImageGenerationRequest) (*TransportRequest, error) {
	return l.buildRequest(l.imageGenerationLMRequest(req), false, nil)
}

func (l *GeminiLM) imageGenerationFromResponse(req *ImageGenerationRequest, resp *HTTPResponse) (ImageGenerationResponse, error) {
	chat, err := l.parseResponse(l.imageGenerationLMRequest(req), resp)
	if err != nil {
		return ImageGenerationResponse{}, err
	}
	var images []ImagePart
	var texts []string
	for _, p := range chat.Message.Parts {
		switch x := p.(type) {
		case ImagePart:
			images = append(images, x)
		case TextPart:
			if x.Text != "" {
				texts = append(texts, x.Text)
			}
		}
	}
	if len(images) == 0 {
		return ImageGenerationResponse{}, l.providerError(KindProvider, "gemini: model returned no image parts", 0, "", "")
	}
	return ImageGenerationResponse{Images: images, Text: strings.Join(texts, ""), ID: chat.ID, Model: chat.Model, Usage: chat.Usage, ProviderData: chat.ProviderData}, nil
}

func (l *GeminiLM) speechGenerationLMRequest(req *SpeechGenerationRequest) (*Request, error) {
	if req.Format != "" {
		return nil, UnsupportedFeatureErrorf(l.provider, "gemini: speech format cannot be chosen; the wire always returns PCM")
	}
	gen := JSONObject{{"responseModalities", []any{"AUDIO"}}}
	if req.Voice != "" {
		gen.Set("speechConfig", JSONObject{{"voiceConfig", JSONObject{{"prebuiltVoiceConfig", JSONObject{{"voiceName", req.Voice}}}}}})
	}
	ext := JSONObject{{"generationConfig", gen}}
	for k, v := range req.Extensions.All() {
		ext.Set(k, v)
	}
	return &Request{Model: req.Model, Messages: []Message{UserMessage(req.Prompt)}, Config: Config{Extensions: ext}}, nil
}

func (l *GeminiLM) speechGenerateRequest(req *SpeechGenerationRequest) (*TransportRequest, error) {
	lmReq, err := l.speechGenerationLMRequest(req)
	if err != nil {
		return nil, err
	}
	return l.buildRequest(lmReq, false, nil)
}

func (l *GeminiLM) speechGenerationFromResponse(req *SpeechGenerationRequest, resp *HTTPResponse) (SpeechGenerationResponse, error) {
	lmReq, err := l.speechGenerationLMRequest(req)
	if err != nil {
		return SpeechGenerationResponse{}, err
	}
	chat, err := l.parseResponse(lmReq, resp)
	if err != nil {
		return SpeechGenerationResponse{}, err
	}
	for _, p := range chat.Message.Parts {
		if audio, ok := p.(AudioPart); ok {
			return SpeechGenerationResponse{Audio: audio, ID: chat.ID, Model: chat.Model, Usage: chat.Usage, ProviderData: chat.ProviderData}, nil
		}
	}
	return SpeechGenerationResponse{}, l.providerError(KindProvider, "gemini: model returned no audio part", 0, "", "")
}

// wavToPCM strips a RIFF header, returning (pcm, sample rate).
func wavToPCM(data []byte) ([]byte, int) {
	if len(data) >= 44 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE" {
		rate := int(binary.LittleEndian.Uint32(data[24:28]))
		pos := 12
		for pos+8 <= len(data) {
			chunkID := string(data[pos : pos+4])
			size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
			if chunkID == "data" {
				end := pos + 8 + size
				if end > len(data) {
					end = len(data)
				}
				return data[pos+8 : end], rate
			}
			pos += 8 + size
		}
		return data[44:], rate
	}
	return data, 16000
}
