package lm15

import (
	"encoding/base64"
	"encoding/binary"
	"sort"
	"strconv"
	"strings"
	"time"

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
	if pf := wireObj(data["promptFeedback"]); pf != nil {
		if reason := wireStr(pf["blockReason"]); reason != "" && reason != "BLOCK_REASON_UNSPECIFIED" {
			return l.providerError(KindInvalidRequest, "Prompt blocked: "+reason, 0, "promptFeedback", "")
		}
	}
	candidates := wireList(data["candidates"])
	if len(candidates) > 0 {
		if c := wireObj(candidates[0]); c != nil {
			if fr := wireStr(c["finishReason"]); geminiCandidateFinishErrors[fr] {
				msg := wireStr(c["finishMessage"])
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
		switch e := obj["error"].(type) {
		case map[string]any:
			msg = wireStr(e["message"])
			errStatus = wireStr(e["status"])
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

func containsKey(v any, key string) bool {
	switch x := v.(type) {
	case map[string]any:
		if _, ok := x[key]; ok {
			return true
		}
		for _, item := range x {
			if containsKey(item, key) {
				return true
			}
		}
	case []any:
		for _, item := range x {
			if containsKey(item, key) {
				return true
			}
		}
	}
	return false
}

func geminiResponseFormat(f JSONObject) JSONObject {
	if f["type"] == "json_object" {
		return JSONObject{"responseMimeType": "application/json"}
	}
	schema := f["schema"]
	field := "responseSchema"
	if containsKey(schema, "additionalProperties") {
		field = "responseJsonSchema"
	}
	return JSONObject{"responseMimeType": "application/json", field: schema}
}

func modalityTokens(details any, modality string) *int {
	list, ok := details.([]any)
	if !ok {
		return nil
	}
	sum := 0
	found := false
	for _, e := range list {
		if obj := wireObj(e); obj != nil && wireStr(obj["modality"]) == modality {
			sum += wireInt(obj["tokenCount"], 0)
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
		if v, ok := usage[k]; ok {
			output = wireIntPtr(v)
			if output == nil {
				output = &zero
			}
			break
		}
	}
	input := wireIntPtr(usage["promptTokenCount"])
	if input == nil {
		z := 0
		input = &z
	}
	outputDetails := usage["candidatesTokensDetails"]
	if outputDetails == nil {
		outputDetails = usage["responseTokensDetails"]
	}
	return Usage{
		InputTokens:       input,
		OutputTokens:      output,
		TotalTokens:       wireIntPtr(usage["totalTokenCount"]),
		CacheReadTokens:   wireIntPtr(usage["cachedContentTokenCount"]),
		ReasoningTokens:   wireIntPtr(usage["thoughtsTokenCount"]),
		InputAudioTokens:  modalityTokens(usage["promptTokensDetails"], "AUDIO"),
		OutputAudioTokens: modalityTokens(outputDetails, "AUDIO"),
	}.Normalize()
}

func thoughtSignatureState(part JSONObject) []ContinuationState {
	sig, ok := part["thoughtSignature"]
	if !ok || sig == nil {
		return nil
	}
	return []ContinuationState{{Provider: "gemini", Kind: "thought_signature", Data: JSONObject{"value": wireStr(sig)}}}
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
	state := strings.ToUpper(wireStr(wireObj(data["metadata"])["state"]))
	mapping := map[string]string{
		"BATCH_STATE_PENDING": BatchQueued, "BATCH_STATE_RUNNING": BatchRunning, "BATCH_STATE_CANCELLING": BatchCancelling,
		"BATCH_STATE_SUCCEEDED": BatchCompleted, "BATCH_STATE_FAILED": BatchFailed, "BATCH_STATE_CANCELLED": BatchCancelled, "BATCH_STATE_EXPIRED": BatchExpired,
	}
	if s, ok := mapping[state]; ok {
		return s
	}
	if truthy(data["done"]) {
		return BatchCompleted
	}
	return BatchQueued
}

func geminiTokenLogprobs(result any) []TokenLogprob {
	lr := wireObj(result)
	if lr == nil {
		return nil
	}
	chosen := wireList(lr["chosenCandidates"])
	topSteps := wireList(lr["topCandidates"])
	var out []TokenLogprob
	for i, c := range chosen {
		cand := wireObj(c)
		if cand == nil {
			continue
		}
		var top []TopLogprob
		if i < len(topSteps) {
			for _, a := range wireList(wireObj(topSteps[i])["candidates"]) {
				if alt := wireObj(a); alt != nil {
					top = append(top, TopLogprob{Token: wireStr(alt["token"]), Logprob: wireFloat(alt["logProbability"], 0), TokenID: wireIntPtr(alt["tokenId"])})
				}
			}
		}
		out = append(out, TokenLogprob{Token: wireStr(cand["token"]), Logprob: wireFloat(cand["logProbability"], 0), TokenID: wireIntPtr(cand["tokenId"]), Top: top})
	}
	return out
}

func geminiSegmentText(segment JSONObject, full string) string {
	if t := stringOnly(segment["text"]); t != "" {
		return t
	}
	start, end := wireIntPtr(segment["startIndex"]), wireIntPtr(segment["endIndex"])
	if start != nil && end != nil && 0 <= *start && *start < *end && *end <= len(full) {
		return full[*start:*end]
	}
	return ""
}

func geminiCitations(candidate JSONObject, full string) []Part {
	grounding := wireObj(candidate["groundingMetadata"])
	if grounding == nil {
		return nil
	}
	chunks := wireList(grounding["groundingChunks"])
	supports, ok := grounding["groundingSupports"].([]any)
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
		cited := geminiSegmentText(wireObj(support["segment"]), full)
		indices, ok := support["groundingChunkIndices"].([]any)
		if !ok {
			continue
		}
		for _, raw := range indices {
			idx := wireIntPtr(raw)
			var chunk JSONObject
			if idx != nil && *idx >= 0 && *idx < len(chunks) {
				chunk = wireObj(chunks[*idx])
			}
			source := wireObj(chunk["web"])
			if source == nil {
				source = wireObj(chunk["retrievedContext"])
			}
			if source == nil {
				source = wireObj(chunk["googleSearch"])
			}
			url := firstStr(source["uri"], source["url"])
			title := firstStr(source["title"], source["name"])
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
				return nil, UnsupportedFeatureErrorf(l.provider, "%s: a %s part in tool_result %q cannot reach a functionResponse — multimodal function responses take images (png/jpeg/webp) and documents (pdf, text/plain) only (MAP-10)", l.provider, p.Type(), part.ID)
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
		return nil, UnsupportedFeatureErrorf(l.provider, "%s: tool_result %q needs a function name on the Gemini wire and no preceding assistant tool_call with that id is in the transcript; set ToolResultPart.name (MAP-10 rule 6)", l.provider, part.ID)
	}
	text, err := partsToText(textParts, l.provider, "functionResponse.response")
	if err != nil {
		return nil, err
	}
	var response JSONObject
	switch {
	case part.IsError:
		response = JSONObject{"error": text}
	case len(mediaParts) > 0 && len(textParts) == 0:
		response = JSONObject{}
	default:
		response = JSONObject{"result": text}
	}
	fr := JSONObject{"name": name, "response": response}
	if part.ID != "" {
		fr["id"] = part.ID
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
		fr["parts"] = blocks
	}
	return JSONObject{"functionResponse": fr}, nil
}

func (l *GeminiLM) part(p Part, names map[string]string) (JSONObject, error) {
	switch x := p.(type) {
	case TextPart:
		out := JSONObject{"text": x.Text}
		if thought := ContinuationData(x.Continuation, "gemini", "thought_signature"); thought != nil && truthy(thought["value"]) {
			out["thoughtSignature"] = thought["value"]
		}
		return out, nil
	case ToolCallPart:
		fc := JSONObject{"name": x.Name, "args": x.Input}
		if x.ID != "" {
			fc["id"] = x.ID
		}
		out := JSONObject{"functionCall": fc}
		if thought := ContinuationData(x.Continuation, "gemini", "thought_signature"); thought != nil && truthy(thought["value"]) {
			out["thoughtSignature"] = thought["value"]
		}
		return out, nil
	case ToolResultPart:
		return l.functionResponse(x, names)
	case ThinkingPart:
		out := JSONObject{"text": x.Text}
		if thought := ContinuationData(x.Continuation, "gemini", "thought_signature"); thought != nil && truthy(thought["value"]) {
			out["thought"] = true
			out["thoughtSignature"] = thought["value"]
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
			return JSONObject{"fileData": JSONObject{"mimeType": mime, "fileUri": m.URL}}, nil
		case m.FileID != "":
			return JSONObject{"fileData": JSONObject{"mimeType": mime, "fileUri": m.FileID}}, nil
		default:
			b64, err := m.Base64()
			if err != nil {
				return nil, err
			}
			return JSONObject{"inlineData": JSONObject{"mimeType": mime, "data": b64}}, nil
		}
	}
	return JSONObject{"text": partText(p)}, nil
}

func (l *GeminiLM) message(msg Message, names map[string]string) (JSONObject, error) {
	if msg.Role == RoleDeveloper {
		text, err := partsToText(msg.Parts, l.provider, "a developer turn")
		if err != nil {
			return nil, err
		}
		return JSONObject{"role": "user", "parts": []any{JSONObject{"text": "[developer]\n" + text}}}, nil
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
	return JSONObject{"role": role, "parts": parts}, nil
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

func (l *GeminiLM) toolConfigPayload(req *Request) (JSONObject, error) {
	tc := req.Config.ToolChoice
	if tc == nil {
		return nil, nil
	}
	if tc.Parallel != nil && !*tc.Parallel {
		return nil, UnsupportedFeatureErrorf(l.provider, "gemini: tool_choice.parallel=False is not supported — GenerateContent has no parallel-tool-calls knob and returns several calls regardless (OpenAI and Anthropic carry it)")
	}
	mode := map[string]string{"none": "NONE", "required": "ANY", "auto": "AUTO"}[tc.EffectiveMode()]
	cfg := JSONObject{"mode": mode}
	if len(tc.Allowed) > 0 {
		var builtins []string
		for _, name := range tc.Allowed {
			if _, ok := req.ToolByName(name).(BuiltinTool); ok {
				builtins = append(builtins, name)
			}
		}
		if len(builtins) > 0 {
			return nil, UnsupportedFeatureErrorf(l.provider, "gemini: cannot force builtin tools %v — functionCallingConfig addresses function declarations only; googleSearch/codeExecution have no tool_choice form (OpenAI Responses and Anthropic carry builtin forcing)", builtins)
		}
		cfg["allowedFunctionNames"] = toAnyList(tc.Allowed, func(s string) any { return s })
		if tc.EffectiveMode() == "auto" {
			cfg["mode"] = "VALIDATED"
		}
	}
	return JSONObject{"functionCallingConfig": cfg}, nil
}

func (l *GeminiLM) cacheResource(cacheID string) string {
	if strings.HasPrefix(cacheID, "cachedContents/") {
		return cacheID
	}
	return "cachedContents/" + cacheID
}

func (l *GeminiLM) payload(req *Request) (JSONObject, error) {
	cfg := req.Config
	ext := copyObject(cfg.Extensions)
	resource := ""
	suffixFrom := 0
	if c := cfg.Cache; c != nil && c.EffectiveMode() != "off" {
		if c.Key != "" {
			return nil, UnsupportedFeatureErrorf(l.provider, "gemini: cache.key is not supported — GenerateContent has no cache affinity key; use cache.resource with a stored cache (lm.cache(prefix))")
		}
		if c.Retention != "" && c.Retention != "short" {
			return nil, UnsupportedFeatureErrorf(l.provider, "gemini: cache.retention is not supported in-request — lifetime belongs to the stored cache (cache_create(..., ttl_seconds=...) / cache_update)")
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
	payload := JSONObject{"contents": contents}
	if resource != "" {
		payload["cachedContent"] = l.cacheResource(resource)
	}
	if req.System != nil && resource == "" {
		text, err := systemText(req.System, l.provider)
		if err != nil {
			return nil, err
		}
		payload["systemInstruction"] = JSONObject{"parts": []any{JSONObject{"text": text}}}
	}
	gen := JSONObject{}
	if cfg.Temperature != nil {
		gen["temperature"] = geminiNumber(*cfg.Temperature)
	}
	if cfg.MaxTokens != nil {
		gen["maxOutputTokens"] = *cfg.MaxTokens
	}
	if cfg.TopP != nil {
		gen["topP"] = geminiNumber(*cfg.TopP)
	}
	if cfg.TopK != nil {
		gen["topK"] = *cfg.TopK
	}
	if len(cfg.Stop) > 0 {
		gen["stopSequences"] = toAnyList(cfg.Stop, func(s string) any { return s })
	}
	if cfg.Logprobs != nil {
		gen["responseLogprobs"] = true
		if *cfg.Logprobs > 0 {
			gen["logprobs"] = *cfg.Logprobs
		}
	}
	if len(cfg.ResponseFormat) > 0 {
		for k, v := range geminiResponseFormat(cfg.ResponseFormat) {
			gen[k] = v
		}
	}
	if r := cfg.Reasoning; r != nil {
		levelClass := GeminiLevelClass(req.Model)
		if r.IsOff() {
			if levelClass {
				return nil, UnsupportedFeatureErrorf(l.provider, "gemini: reasoning cannot be disabled on %s — the Gemini 3 class has no full off switch (thinkingBudget 0 is accepted but not honoured); use effort='low' or a 2.5 model", req.Model)
			}
			gen["thinkingConfig"] = JSONObject{"thinkingBudget": 0}
		} else {
			if r.Summary == "concise" || r.Summary == "detailed" {
				return nil, UnsupportedFeatureErrorf(l.provider, "gemini: reasoning.summary=%q is an OpenAI detail level; GenerateContent has includeThoughts only (use 'auto')", r.Summary)
			}
			thinking := JSONObject{}
			if r.Summary != "" {
				thinking["includeThoughts"] = true
			}
			switch {
			case r.ThinkingBudget != nil:
				thinking["thinkingBudget"] = *r.ThinkingBudget
			case levelClass:
				if r.Effort == "xhigh" || r.Effort == "max" {
					return nil, UnsupportedFeatureErrorf(l.provider, "gemini: reasoning.effort=%q has no thinkingLevel on the Gemini 3 class (minimal|low|medium|high); 'high' is the ceiling", r.Effort)
				}
				thinking["thinkingLevel"] = r.Effort
			default:
				thinking["thinkingBudget"] = EffortThinkingBudgets[r.Effort]
			}
			gen["thinkingConfig"] = thinking
		}
	}
	if len(gen) > 0 {
		payload["generationConfig"] = gen
	}
	if len(req.Tools) > 0 && resource == "" {
		var declarations []any
		var tools []any
		for _, t := range req.Tools {
			if ft, ok := t.(FunctionTool); ok {
				declarations = append(declarations, JSONObject{"name": ft.Name, "description": nilIfEmpty(ft.Description), "parameters": ft.EffectiveParameters()})
			}
		}
		if len(declarations) > 0 {
			tools = append(tools, JSONObject{"functionDeclarations": declarations})
		}
		for _, t := range req.Tools {
			if bt, ok := t.(BuiltinTool); ok {
				tools = append(tools, geminiBuiltin(bt))
			}
		}
		payload["tools"] = tools
	}
	if resource == "" {
		tc, err := l.toolConfigPayload(req)
		if err != nil {
			return nil, err
		}
		if tc != nil {
			payload["toolConfig"] = tc
		}
	}
	switch wireStr(ext["output"]) {
	case "image":
		genCfg := wireObj(payload["generationConfig"])
		if genCfg == nil {
			genCfg = JSONObject{}
			payload["generationConfig"] = genCfg
		}
		genCfg["responseModalities"] = []any{"IMAGE"}
	case "audio":
		genCfg := wireObj(payload["generationConfig"])
		if genCfg == nil {
			genCfg = JSONObject{}
			payload["generationConfig"] = genCfg
		}
		genCfg["responseModalities"] = []any{"AUDIO"}
	}
	if cfg.Store != nil {
		payload["store"] = *cfg.Store
	}
	if cfg.ServiceTier != "" {
		payload["serviceTier"] = cfg.ServiceTier
	}
	if cfg.UserID != "" {
		return nil, UnsupportedFeatureErrorf(l.provider, "gemini: config.user_id is not supported — GenerateContent has no end-user attribution field (OpenAI and Anthropic carry it)")
	}
	for k, v := range ext {
		if k != "prompt_caching" && k != "output" {
			payload[k] = v
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
	return JSONObject{key: cfg}
}

func (l *GeminiLM) buildRequest(req *Request, stream bool) (*TransportRequest, error) {
	payload, err := l.payload(req)
	if err != nil {
		return nil, err
	}
	endpoint := "generateContent"
	var params map[string]string
	timeout := 60 * time.Second
	if stream {
		endpoint = "streamGenerateContent"
		params = map[string]string{"alt": "sse"}
		timeout = 120 * time.Second
	}
	return l.emit(emitSpec{
		method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(req.Model) + ":" + endpoint,
		endpoint: "generateContent", stream: stream, model: req.Model,
		headers: [][2]string{{"Content-Type", "application/json"}}, params: params, payload: payload, readTimeout: timeout,
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
		_, hasText := part["text"]
		switch {
		case truthy(part["thought"]) && hasText:
			parts = append(parts, ThinkingPart{Text: wireStr(part["text"]), Continuation: thoughtSignatureState(part)})
		case hasText:
			parts = append(parts, TextPart{Text: wireStr(part["text"]), Continuation: thoughtSignatureState(part)})
		case wireObj(part["functionCall"]) != nil:
			fc := wireObj(part["functionCall"])
			var continuation []ContinuationState
			sig := part["thoughtSignature"]
			if sig == nil {
				sig = fc["thoughtSignature"]
			}
			if sig != nil {
				continuation = []ContinuationState{{Provider: "gemini", Kind: "thought_signature", Data: JSONObject{"value": wireStr(sig)}}}
			}
			if !truthy(fc["name"]) {
				return nil, unnamedToolCallError(l.provider, path)
			}
			id := wireStr(fc["id"])
			if id == "" || fc["id"] == nil {
				id = "tool_call_" + strconv.Itoa(len(parts))
			}
			args := wireObj(fc["args"])
			if args == nil {
				args = JSONObject{}
			}
			parts = append(parts, ToolCallPart{ID: id, Name: wireStr(fc["name"]), Input: args, Continuation: continuation})
		case wireObj(part["inlineData"]) != nil:
			inline := wireObj(part["inlineData"])
			mime := wireStr(inline["mimeType"])
			if mime == "" {
				mime = "application/octet-stream"
			}
			data := wireStr(inline["data"])
			if data == "" {
				continue
			}
			parts = append(parts, mediaPartFor(mime, Media{MediaType: mime, Data: data}))
		case wireObj(part["fileData"]) != nil:
			fd := wireObj(part["fileData"])
			uri := wireStr(fd["fileUri"])
			mime := wireStr(fd["mimeType"])
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
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

func (l *GeminiLM) parseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	data, err := resp.JSON()
	if err != nil {
		return nil, err
	}
	if e := l.inbandError(data); e != nil {
		return nil, e
	}
	var candidate JSONObject
	if cands := wireList(data["candidates"]); len(cands) > 0 {
		candidate = wireObj(cands[0])
	}
	content := wireObj(candidate["content"])
	var unmapped []JSONObject
	parts, err := l.parseCandidateParts(wireList(content["parts"]), &unmapped, "candidates[0].content.parts")
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
		ID:           wireStr(data["responseId"]),
		Model:        req.Model,
		Message:      Message{Role: RoleAssistant, Parts: parts},
		FinishReason: geminiFinish(wireStr(candidate["finishReason"]), hasToolCall(parts)),
		Usage:        geminiUsage(wireObj(data["usageMetadata"]), "candidatesTokenCount", "responseTokenCount"),
		Logprobs:     geminiTokenLogprobs(candidate["logprobsResult"]),
		ProviderData: attachUnmapped(data, unmapped),
	}, nil
}

func (l *GeminiLM) usageFromPayload(payload JSONObject) Usage {
	return geminiUsage(wireObj(payload["usageMetadata"]), "candidatesTokenCount", "responseTokenCount")
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
	if errRaw, ok := payload["error"]; ok {
		e := wireObj(errRaw)
		code := firstStr(e["status"], e["code"])
		if code == "" {
			code = "provider"
		}
		return []StreamEvent{StreamErrorEvent{Error: l.errorDetail(code, wireStr(e["message"]))}}, nil
	}
	if inband := l.inbandError(payload); inband != nil {
		return []StreamEvent{StreamErrorEvent{Error: ErrorDetail{Code: inband.Code, ProviderCode: "inband_finish_reason", Message: inband.Error()}}}, nil
	}
	var events []StreamEvent
	var candidate JSONObject
	if cands := wireList(payload["candidates"]); len(cands) > 0 {
		candidate = wireObj(cands[0])
	}
	yielded := false
	sawTool := false
	finish := ""
	if candidate != nil {
		content := wireObj(candidate["content"])
		chunkLogprobs := geminiTokenLogprobs(candidate["logprobsResult"])
		for idx, raw := range wireList(content["parts"]) {
			part := wireObj(raw)
			if part == nil {
				continue
			}
			i := idx
			signatureDelta := func(sig any) StreamEvent {
				return StreamDeltaEvent{Delta: ContinuationDelta{Provider: "gemini", Kind: "thought_signature", Data: JSONObject{"value": wireStr(sig)}, PartIndex: &i}}
			}
			_, hasText := part["text"]
			switch {
			case truthy(part["thought"]) && hasText:
				yielded = true
				events = append(events, StreamDeltaEvent{Delta: ThinkingDelta{Text: wireStr(part["text"]), PartIndex: idx}})
				if part["thoughtSignature"] != nil {
					events = append(events, signatureDelta(part["thoughtSignature"]))
				}
			case hasText:
				yielded = true
				events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: wireStr(part["text"]), PartIndex: idx, Logprobs: chunkLogprobs}})
				chunkLogprobs = nil
				if part["thoughtSignature"] != nil {
					events = append(events, signatureDelta(part["thoughtSignature"]))
				}
			case wireObj(part["functionCall"]) != nil:
				fc := wireObj(part["functionCall"])
				sawTool = true
				yielded = true
				args := fc["args"]
				if args == nil {
					args = JSONObject{}
				}
				events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{Input: jsonRaw(args), PartIndex: idx, ID: wireStr(fc["id"]), Name: wireStr(fc["name"])}})
				sig := part["thoughtSignature"]
				if sig == nil {
					sig = fc["thoughtSignature"]
				}
				if sig != nil {
					events = append(events, signatureDelta(sig))
				}
			case wireObj(part["inlineData"]) != nil:
				inline := wireObj(part["inlineData"])
				mime := wireStr(inline["mimeType"])
				if mime == "" {
					mime = "application/octet-stream"
				}
				data := wireStr(inline["data"])
				if strings.HasPrefix(mime, "audio/") {
					yielded = true
					events = append(events, StreamDeltaEvent{Delta: AudioDelta{Data: S(data), PartIndex: idx, MediaType: mime}})
				} else if strings.HasPrefix(mime, "image/") {
					yielded = true
					events = append(events, StreamDeltaEvent{Delta: ImageDelta{Data: S(data), PartIndex: idx, MediaType: mime}})
				}
			}
		}
		finish = wireStr(candidate["finishReason"])
	}
	if finish != "" {
		usage := l.usageFromPayload(payload)
		events = append(events, StreamEndEvent{FinishReason: geminiFinish(finish, sawTool), Usage: &usage, ProviderData: payload})
	} else if _, has := payload["usageMetadata"]; !yielded && has {
		usage := l.usageFromPayload(payload)
		events = append(events, StreamEndEvent{FinishReason: FinishStop, Usage: &usage, ProviderData: payload})
	}
	return events, nil
}

// ─── Models ──────────────────────────────────────────────────────────

func (l *GeminiLM) modelsRequest() (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/models", params: map[string]string{"pageSize": "1000"}, readTimeout: 30 * time.Second})
}

func (l *GeminiLM) modelsFromBody(body string) ([]ModelInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	return modelInfosFromEntries(data["models"], l.provider, "gemini_generate_content", func(e map[string]any) string {
		return strings.TrimPrefix(stringOnly(e["name"]), "models/")
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
	for k, v := range req.Extensions {
		if v != nil {
			params[k] = wireStr(v)
		}
	}
	content, err := req.Content()
	if err != nil {
		return nil, err
	}
	ct, body := multipartRelatedBody(JSONObject{"file": JSONObject{"display_name": req.Filename}}, req.EffectiveMediaType(), content)
	return l.emit(emitSpec{method: "POST", url: buildURL(strings.TrimRight(l.uploadBaseURL, "/")+"/files", params), headers: [][2]string{{"X-Goog-Upload-Protocol", "multipart"}, {"Content-Type", ct}}, body: body, readTimeout: 300 * time.Second})
}

func (l *GeminiLM) fileInfo(data JSONObject) (FileInfo, error) {
	id := stringOnly(data["uri"])
	if id == "" {
		id = stringOnly(data["name"])
	}
	if id == "" {
		return FileInfo{}, l.providerError(KindProvider, "gemini: file object carries no uri or name", 0, "", "")
	}
	state := wireStr(data["state"])
	readiness := "ready"
	if strings.HasSuffix(state, "PROCESSING") {
		readiness = "pending"
	} else if strings.HasSuffix(state, "FAILED") {
		readiness = "failed"
	}
	var downloadable *bool
	if stringOnly(data["downloadUri"]) != "" {
		downloadable = B(true)
	} else if data["source"] == "UPLOADED" {
		downloadable = B(false)
	}
	var size *int
	switch raw := data["sizeBytes"].(type) {
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
		ID: id, Filename: stringOnly(data["displayName"]), MediaType: stringOnly(data["mimeType"]), SizeBytes: size,
		CreatedAt: isoUTC(data["createTime"]), ExpiresAt: isoUTC(data["expirationTime"]), Readiness: readiness, Downloadable: downloadable, ProviderData: data,
	}, nil
}

func (l *GeminiLM) fileInfoFromBody(body string) (FileInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FileInfo{}, err
	}
	if f := wireObj(data["file"]); f != nil {
		return l.fileInfo(f)
	}
	return l.fileInfo(data)
}

func (l *GeminiLM) fileGetRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(geminiFileResource(fileID), true), readTimeout: 60 * time.Second})
}

func (l *GeminiLM) fileListRequest(limit int, cursor string) (*TransportRequest, error) {
	params := map[string]string{"pageSize": strconv.Itoa(limit)}
	if cursor != "" {
		params["pageToken"] = cursor
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/files", params: params, readTimeout: 60 * time.Second})
}

func (l *GeminiLM) filePageFromListBody(body string) (FilePage, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return FilePage{}, err
	}
	var items []FileInfo
	for _, e := range wireList(data["files"]) {
		if obj := wireObj(e); obj != nil {
			info, err := l.fileInfo(obj)
			if err != nil {
				return FilePage{}, err
			}
			items = append(items, info)
		}
	}
	return FilePage{Items: items, NextCursor: stringOnly(data["nextPageToken"])}, nil
}

func (l *GeminiLM) fileDeleteRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "DELETE", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(geminiFileResource(fileID), true), readTimeout: 60 * time.Second})
}

func (l *GeminiLM) fileDownloadRequest(fileID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(geminiFileResource(fileID), true) + ":download", params: map[string]string{"alt": "media"}, readTimeout: 300 * time.Second})
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
	body := JSONObject{"model": l.modelPath(prefix.Model), "contents": contents}
	if prefix.System != nil {
		text, err := systemText(prefix.System, l.provider)
		if err != nil {
			return nil, err
		}
		body["systemInstruction"] = JSONObject{"parts": []any{JSONObject{"text": text}}}
	}
	if len(prefix.Tools) > 0 {
		var declarations, tools []any
		for _, t := range prefix.Tools {
			if ft, ok := t.(FunctionTool); ok {
				declarations = append(declarations, JSONObject{"name": ft.Name, "description": nilIfEmpty(ft.Description), "parameters": ft.EffectiveParameters()})
			}
		}
		if len(declarations) > 0 {
			tools = append(tools, JSONObject{"functionDeclarations": declarations})
		}
		for _, t := range prefix.Tools {
			if bt, ok := t.(BuiltinTool); ok {
				tools = append(tools, geminiBuiltin(bt))
			}
		}
		body["tools"] = tools
	}
	if ttlSeconds != nil {
		body["ttl"] = strconv.Itoa(*ttlSeconds) + "s"
	}
	if label != "" {
		body["displayName"] = label
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/cachedContents", headers: [][2]string{{"Content-Type", "application/json"}}, payload: body, readTimeout: 120 * time.Second})
}

func (l *GeminiLM) cacheInfo(data JSONObject) (CacheInfo, error) {
	name := stringOnly(data["name"])
	if name == "" {
		return CacheInfo{}, l.providerError(KindProvider, "gemini: cache object carries no name", 0, "", "")
	}
	model := strings.TrimPrefix(wireStr(data["model"]), "models/")
	if model == "" {
		return CacheInfo{}, l.providerError(KindProvider, "gemini: cache object carries no model", 0, "", "")
	}
	var tokens *int
	switch raw := wireObj(data["usageMetadata"])["totalTokenCount"].(type) {
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
	return CacheInfo{ID: name, Model: model, Tokens: tokens, CreatedAt: isoUTC(data["createTime"]), ExpiresAt: isoUTC(data["expireTime"]), Label: stringOnly(data["displayName"]), ProviderData: data}, nil
}

func (l *GeminiLM) cacheInfoFromBody(body string) (CacheInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return CacheInfo{}, err
	}
	return l.cacheInfo(data)
}

func (l *GeminiLM) cacheGetRequest(cacheID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(l.cacheResource(cacheID), true), readTimeout: 60 * time.Second})
}

func (l *GeminiLM) cacheListRequest(limit int, cursor string) (*TransportRequest, error) {
	params := map[string]string{"pageSize": strconv.Itoa(limit)}
	if cursor != "" {
		params["pageToken"] = cursor
	}
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/cachedContents", params: params, readTimeout: 60 * time.Second})
}

func (l *GeminiLM) cachePageFromListBody(body string) (CachePage, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return CachePage{}, err
	}
	var items []CacheInfo
	for _, e := range wireList(data["cachedContents"]) {
		if obj := wireObj(e); obj != nil {
			info, err := l.cacheInfo(obj)
			if err != nil {
				return CachePage{}, err
			}
			items = append(items, info)
		}
	}
	return CachePage{Items: items, NextCursor: stringOnly(data["nextPageToken"])}, nil
}

func (l *GeminiLM) cacheDeleteRequest(cacheID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "DELETE", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(l.cacheResource(cacheID), true), readTimeout: 60 * time.Second})
}

func (l *GeminiLM) cacheUpdateRequest(cacheID string, ttlSeconds int) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "PATCH", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(l.cacheResource(cacheID), true), headers: [][2]string{{"Content-Type", "application/json"}}, payload: JSONObject{"ttl": strconv.Itoa(ttlSeconds) + "s"}, readTimeout: 60 * time.Second})
}

// ─── Batch ───────────────────────────────────────────────────────────

func (l *GeminiLM) batchSubmitRequest(req *BatchRequest, _ JSONObject) (*TransportRequest, error) {
	model := req.EffectiveModel()
	var requests []any
	for i, nested := range req.Requests {
		p, err := l.payload(nested)
		if err != nil {
			return nil, err
		}
		requests = append(requests, JSONObject{"request": p, "metadata": JSONObject{"key": strconv.Itoa(i)}})
	}
	batch := JSONObject{"inputConfig": JSONObject{"requests": JSONObject{"requests": requests}}}
	if req.Label != "" {
		batch["displayName"] = req.Label
	}
	payload := JSONObject{"batch": batch}
	for k, v := range req.Extensions {
		payload[k] = v
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(model) + ":batchGenerateContent", headers: [][2]string{{"Content-Type", "application/json"}}, payload: payload, readTimeout: 120 * time.Second})
}

func (l *GeminiLM) batchJobInfo(data JSONObject) (BatchJobInfo, error) {
	name := stringOnly(data["name"])
	if name == "" {
		return BatchJobInfo{}, l.providerError(KindProvider, "gemini: batch operation carries no name", 0, "", "")
	}
	metadata := wireObj(data["metadata"])
	return BatchJobInfo{ID: name, Status: geminiBatchStatus(data), Label: stringOnly(metadata["displayName"]), CreatedAt: isoUTC(metadata["createTime"]), ProviderData: data}, nil
}

func (l *GeminiLM) batchJobFromBody(body string) (BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return BatchJobInfo{}, err
	}
	return l.batchJobInfo(data)
}

func (l *GeminiLM) batchStatusRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(batchID, true), readTimeout: 60 * time.Second})
}

func (l *GeminiLM) batchCancelRequest(batchID string) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(batchID, true) + ":cancel", headers: [][2]string{{"Content-Type", "application/json"}}, payload: JSONObject{}, readTimeout: 60 * time.Second})
}

func (l *GeminiLM) batchResultFetches(JSONObject) ([]*TransportRequest, error) { return nil, nil }

func (l *GeminiLM) batchEntries(statusBody JSONObject, _ []string) ([]BatchEntry, error) {
	response := wireObj(statusBody["response"])
	var inlined []any
	switch x := response["inlinedResponses"].(type) {
	case map[string]any:
		inlined = wireList(x["inlinedResponses"])
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
		if key := wireStr(wireObj(item["metadata"])["key"]); key != "" {
			if i, err := strconv.Atoi(key); err == nil {
				index = i
			}
		}
		if body := wireObj(item["response"]); body != nil {
			resp, err := l.parseResponse(batchEntryRequest(stringOnly(body["modelVersion"])), JSONResponse(200, body))
			if err != nil {
				return nil, err
			}
			entries = append(entries, BatchEntry{Index: index, Outcome: "succeeded", Response: resp})
		} else {
			e := wireObj(item["error"])
			msg := wireStr(e["message"])
			if msg == "" {
				msg = "batch entry errored"
			}
			code := ""
			if e["status"] != nil {
				code = wireStr(e["status"])
			} else if e["code"] != nil {
				code = wireStr(e["code"])
			}
			entries = append(entries, BatchEntry{Index: index, Outcome: "errored", Error: &ErrorDetail{Code: CodeProvider, Message: msg, ProviderCode: code}})
		}
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].Index < entries[j].Index })
	return entries, nil
}

func (l *GeminiLM) batchListRequest(limit int) (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/batches", params: map[string]string{"pageSize": strconv.Itoa(limit)}, readTimeout: 60 * time.Second})
}

func (l *GeminiLM) batchJobsFromListBody(body string) ([]BatchJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []BatchJobInfo
	for _, e := range wireList(data["operations"]) {
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
	payload := JSONObject{"instances": []any{JSONObject{"prompt": req.Prompt}}}
	for k, v := range req.Extensions {
		payload[k] = v
	}
	if req.Seconds != nil {
		if _, has := payload["parameters"]; !has {
			payload["parameters"] = JSONObject{"durationSeconds": *req.Seconds}
		}
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(req.Model) + ":predictLongRunning", headers: [][2]string{{"Content-Type", "application/json"}}, payload: payload, readTimeout: 120 * time.Second})
}

func (l *GeminiLM) videoJobInfo(data JSONObject) (VideoJobInfo, error) {
	name := stringOnly(data["name"])
	if name == "" {
		return VideoJobInfo{}, l.providerError(KindProvider, "gemini: video operation carries no name", 0, "", "")
	}
	status := "running"
	if data["done"] == true {
		status = "completed"
		if wireObj(data["error"]) != nil {
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
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + pathID(videoID, true), readTimeout: 60 * time.Second})
}

func (l *GeminiLM) videoResultURI(statusBody JSONObject) (string, error) {
	gvr := wireObj(wireObj(statusBody["response"])["generateVideoResponse"])
	if samples := wireList(gvr["generatedSamples"]); len(samples) > 0 {
		if uri := stringOnly(wireObj(wireObj(samples[0])["video"])["uri"]); uri != "" {
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
	return l.emit(emitSpec{method: "GET", url: uri, readTimeout: 600 * time.Second})
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
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/" + l.modelPath(model) + "/operations", params: map[string]string{"pageSize": strconv.Itoa(limit)}, readTimeout: 60 * time.Second})
}

func (l *GeminiLM) videoJobsFromListBody(body string) ([]VideoJobInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	var out []VideoJobInfo
	for _, e := range wireList(data["operations"]) {
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
		gen := copyObject(wireObj(ext["generationConfig"]))
		imageCfg := copyObject(wireObj(gen["imageConfig"]))
		if _, has := imageCfg["aspectRatio"]; !has {
			imageCfg["aspectRatio"] = req.Size
		}
		gen["imageConfig"] = imageCfg
		ext["generationConfig"] = gen
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
	return l.buildRequest(l.imageGenerationLMRequest(req), false)
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
	gen := JSONObject{"responseModalities": []any{"AUDIO"}}
	if req.Voice != "" {
		gen["speechConfig"] = JSONObject{"voiceConfig": JSONObject{"prebuiltVoiceConfig": JSONObject{"voiceName": req.Voice}}}
	}
	ext := JSONObject{"generationConfig": gen}
	for k, v := range req.Extensions {
		ext[k] = v
	}
	return &Request{Model: req.Model, Messages: []Message{UserMessage(req.Prompt)}, Config: Config{Extensions: ext}}, nil
}

func (l *GeminiLM) speechGenerateRequest(req *SpeechGenerationRequest) (*TransportRequest, error) {
	lmReq, err := l.speechGenerationLMRequest(req)
	if err != nil {
		return nil, err
	}
	return l.buildRequest(lmReq, false)
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
