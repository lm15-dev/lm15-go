package lm15

import (
	"encoding/base64"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

// The vet shim (lm15-contract/harness/PROTOCOL.md): one JSON request per
// line in, one reply per line out. The shim only transforms; the harness
// compares. Nothing here touches the network.

// Version is this port's implementation version.
const Version = "1.1.0-rc.1"

const parseOnlyKey = "vet-parse-only"

// VetHandlers lists the ops this shim answers.
var VetHandlers map[string]func(JSONObject) (JSONObject, error)

func init() {
	VetHandlers = map[string]func(JSONObject) (JSONObject, error){
		"capabilities":          vetCapabilities,
		"build_request":         vetBuildRequest,
		"ingest_openai_chat":    vetIngestOpenAIChat,
		"parse_response":        vetParseResponse,
		"replay_stream":         vetReplayStream,
		"normalize_error":       vetNormalizeError,
		"serde_roundtrip":       vetSerdeRoundtrip,
		"validate":              vetValidate,
		"surface_dump":          func(JSONObject) (JSONObject, error) { return SurfaceDump(), nil },
		"explain_auth":          vetExplainAuth,
		"resolve_model":         vetResolveModel,
		"token_exchange_build":  vetTokenExchangeBuild,
		"token_exchange_parse":  vetTokenExchangeParse,
		"sigv4_sign":            vetSigV4Sign,
		"build_models_request":  vetBuildModelsRequest,
		"parse_models_response": vetParseModelsResponse,
		"replay_live":           vetReplayLive,
		"generation_build":      vetGenerationBuild,
		"generation_parse":      vetGenerationParse,
		"file_op_build":         vetFileOpBuild,
		"file_op_parse":         vetFileOpParse,
		"video_op_build":        vetVideoOpBuild,
		"video_op_parse":        vetVideoOpParse,
		"batch_op_build":        vetBatchOpBuild,
		"batch_op_parse":        vetBatchOpParse,
		"cache_op_build":        vetCacheOpBuild,
		"cache_op_parse":        vetCacheOpParse,
		"managed_run":           vetManagedRun,
	}
}

// vetFailure carries extra fields for the error envelope (replay_stream's events).
type vetFailure struct {
	cause error
	extra JSONObject
}

func (f *vetFailure) Error() string { return f.cause.Error() }

// HandleVetLine answers one protocol line.
func HandleVetLine(line []byte) []byte {
	raw, err := DecodeJSON(line)
	if err != nil {
		return mustJSON(vetErrorReply(nil, err))
	}
	msg, ok := asObject(raw)
	if !ok {
		return mustJSON(vetErrorReply(nil, valueErrorf("request must be a JSON object")))
	}
	id := msg.Get("id")
	op := wireStr(msg.Get("op"))
	handler, ok := VetHandlers[op]
	if !ok {
		return mustJSON(vetErrorReply(id, valueErrorf("unknown op: %s", op)))
	}
	result, err := func() (result JSONObject, err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("panic: %v", r)
			}
		}()
		return handler(msg)
	}()
	if err != nil {
		return mustJSON(vetErrorReply(id, err))
	}
	return mustJSON(JSONObject{{"id", id}, {"ok", true}, {"result", result}})
}

func vetErrorReply(id any, err error) JSONObject {
	extra := JSONObject{}
	if f, ok := err.(*vetFailure); ok {
		extra = f.extra
		err = f.cause
	}
	// The reference writes type, message, code, then the extras.
	errObj := JSONObject{{"type", nil}, {"message", err.Error()}}
	if e := AsError(err); e != nil {
		errObj.Set("type", e.ClassName())
		errObj.Set("message", e.Message)
		if e.Code != "" {
			errObj.Set("code", e.Code)
		}
		if e.Kind.IsA(KindStreamAssembly) && e.Partial != nil {
			errObj.Set("partial_response", ResponseToDict(e.Partial, false))
		}
		if e.Feature != "" {
			// MAP-13: the config path a refusal is about, so a policy layer
			// can act on it; pinned by cases as expect_lm15.raises.feature.
			errObj.Set("feature", e.Feature)
		}
		if e.Kind.IsA(KindUnknownModel) || e.Kind.IsA(KindAmbiguousModel) {
			errObj.Set("model", e.Model)
			if e.Kind.IsA(KindAmbiguousModel) {
				errObj.Set("providers", toAnyList(e.Providers, func(s string) any { return s }))
			}
		}
	} else if kind := NativeErrorKind(err); kind != "" {
		errObj.Set("type", kind)
	} else {
		errObj.Set("type", fmt.Sprintf("%T", err))
	}
	for k, v := range extra.All() {
		errObj.Set(k, v)
	}
	return JSONObject{{"id", id}, {"ok", false}, {"error", errObj}}
}

// ─── Adapter construction ────────────────────────────────────────────

func vetCredential(msg JSONObject) (CredentialLike, error) {
	if c := wireObj(msg.Get("credential")); c != nil {
		return CredentialFromDict(c)
	}
	if k, ok := msg.Lookup("api_key"); ok {
		return wireStr(k), nil
	}
	return parseOnlyKey, nil
}

func vetClock(msg JSONObject) (func() time.Time, error) {
	if msg.Get("now") == nil {
		return nil, nil
	}
	fixed, err := ParseRFC3339(wireStr(msg.Get("now")))
	if err != nil {
		return nil, err
	}
	return func() time.Time { return fixed }, nil
}

func vetSettings(msg JSONObject) map[string]string {
	obj := wireObj(msg.Get("settings"))
	if obj == nil {
		return nil
	}
	out := map[string]string{}
	for k, v := range obj.All() {
		out[k] = wireStr(v)
	}
	return out
}

// AdapterForProvider constructs the LM a contract case's provider names,
// exactly as the router would, with the given credential and base URL.
// Extra options can supply an embedding's transport or adaptation policy.
func AdapterForProvider(provider string, credential CredentialLike, baseURL string, settings map[string]string, clock func() time.Time, extra ...Option) (LM, error) {
	def, ok := LookupProvider(provider)
	if !ok {
		return nil, valueErrorf("unknown provider: %s", provider)
	}
	opts := []Option{WithAPIKey(credential)}
	if baseURL != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}
	if def.Bound() {
		opts = append(opts, WithAccess(def.Access))
		if def.Compat != "" {
			opts = append(opts, WithCompatPreset(def.Compat))
		}
	}
	if def.Hosted() && settings != nil {
		opts = append(opts, WithSettings(settings))
	}
	if clock != nil {
		opts = append(opts, WithClock(clock))
	}
	return construct(def, append(opts, extra...))
}

func vetAdapter(msg JSONObject, parseOnly bool) (LM, error) {
	var cred CredentialLike = parseOnlyKey
	if !parseOnly {
		c, err := vetCredential(msg)
		if err != nil {
			return nil, err
		}
		cred = c
	}
	clock, err := vetClock(msg)
	if err != nil {
		return nil, err
	}
	var opts []Option
	if def, ok := LookupProvider(wireStr(msg.Get("provider"))); ok && def.ID == "openai-codex" {
		// Fixture identity belongs to the vet shim, never a real SDK caller.
		opts = append(opts, WithAccountID("test-account"))
	}
	return AdapterForProvider(wireStr(msg.Get("provider")), cred, wireStr(msg.Get("base_url")), vetSettings(msg), clock, opts...)
}

// NormalizeTransportRequest renders a wire request in the protocol's shape.
func NormalizeTransportRequest(req *TransportRequest) JSONObject {
	u, params := splitURL(req.URL)
	headers := JSONObject{}
	for _, h := range req.Headers {
		headers.Set(strings.ToLower(h[0]), h[1])
	}
	paramsObj := JSONObject{}
	for _, k := range sortedMapKeys(params) {
		paramsObj = append(paramsObj, Member{k, params[k]})
	}
	out := JSONObject{{"method", req.Method}, {"url", u}, {"params", paramsObj}, {"headers", headers}, {"body", nil}}
	if len(req.Body) > 0 {
		if strings.Contains(strings.ToLower(wireStr(headers.Get("content-type"))), "json") {
			if decoded, err := DecodeJSON(req.Body); err == nil {
				out.Set("body", decoded)
				return out
			}
		}
		out.Set("body_b64", base64.StdEncoding.EncodeToString(req.Body))
	}
	return out
}

func normalizedOrErr(req *TransportRequest, err error) (JSONObject, error) {
	if err != nil {
		return nil, err
	}
	return NormalizeTransportRequest(req), nil
}

func vetBody(msg JSONObject) ([]byte, error) {
	return base64.StdEncoding.DecodeString(wireStr(msg.Get("body_b64")))
}

func vetHeaders(msg JSONObject) [][2]string {
	var out [][2]string
	for k, v := range wireObj(msg.Get("headers")).All() {
		out = append(out, [2]string{k, wireStr(v)})
	}
	return out
}

func responseResult(resp *Response) JSONObject {
	result := JSONObject{{"canonical_response", ResponseToDict(resp, false)}}
	if unmapped, ok := resp.ProviderData.Lookup("_lm15_unmapped"); ok && unmapped != nil {
		result.Set("unmapped", unmapped)
	}
	return result
}

func vetRequest(msg JSONObject) (*Request, error) {
	obj := wireObj(msg.Get("canonical_request"))
	if obj == nil {
		return nil, keyError("canonical_request")
	}
	return RequestFromDict(obj)
}

// ─── Ops ─────────────────────────────────────────────────────────────

func vetCapabilities(JSONObject) (JSONObject, error) {
	ops := make([]string, 0, len(VetHandlers))
	for k := range VetHandlers {
		ops = append(ops, k)
	}
	sort.Strings(ops)
	return JSONObject{{"language", "go"}, {"ops", toAnyList(ops, func(s string) any { return s })}, {"impl_version", Version}}, nil
}

func vetBuildRequest(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	req, err := vetRequest(msg)
	if err != nil {
		return nil, err
	}
	wire, adaptations, err := lm.Build(req, truthy(msg.Get("stream")))
	if err != nil {
		return nil, err
	}
	out := NormalizeTransportRequest(wire)
	if len(adaptations) > 0 {
		// MAP-13: the record, without the adapter's own wording (never
		// pinned).
		out.Set("adaptations", toAnyList(adaptations, func(a Adaptation) any {
			d := AdaptationToDict(a)
			d.Delete("reason")
			return d
		}))
	}
	return out, nil
}

func vetIngestOpenAIChat(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	body := wireObj(msg.Get("body"))
	if body == nil {
		return nil, typeErrorf("a Chat Completions request body is a JSON object, got %s", jsonTypeName(msg.Get("body")))
	}
	req, err := lm.RequestFromOpenAIChat(body)
	if err != nil {
		return nil, err
	}
	return JSONObject{{"canonical_request", RequestToDict(req)}}, nil
}

func vetParseResponse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	req, err := vetRequest(msg)
	if err != nil {
		return nil, err
	}
	body, err := vetBody(msg)
	if err != nil {
		return nil, err
	}
	resp, err := lm.ParseResponse(req, &HTTPResponse{Status: wireInt(msg.Get("status"), 200), Reason: "OK", Headers: [][2]string{{"content-type", "application/json"}}, Body: body})
	if err != nil {
		return nil, err
	}
	return responseResult(resp), nil
}

// ParseStreamBody replays an SSE body through the adapter and the
// coalescer: the canonical post-coalesce event trace.
func ParseStreamBody(lm LM, req *Request, body []byte) ([]StreamEvent, error) {
	rawEvents, err := sse.ParseAll(body)
	if err != nil {
		return nil, err
	}
	var events []StreamEvent
	var failure error
	CoalesceStream(func(yield func(StreamEvent, error) bool) {
		for _, raw := range rawEvents {
			parsed, err := lm.ParseStreamEvents(req, raw)
			if err != nil {
				yield(nil, err)
				return
			}
			for _, ev := range parsed {
				if ev != nil && !yield(ev, nil) {
					return
				}
			}
		}
	}, req.Model)(func(ev StreamEvent, err error) bool {
		if err != nil {
			failure = err
			return false
		}
		events = append(events, ev)
		return true
	})
	return events, failure
}

func vetReplayStream(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	req, err := vetRequest(msg)
	if err != nil {
		return nil, err
	}
	body, err := vetBody(msg)
	if err != nil {
		return nil, err
	}
	events, err := ParseStreamBody(lm, req, body)
	if err != nil {
		return nil, err
	}
	dicts := toAnyList(events, func(e StreamEvent) any { return StreamEventToDict(e) })
	resp, err := MaterializeResponse(SliceSeq(events), req)
	if err != nil {
		if IsKind(err, KindStreamAssembly) {
			return nil, &vetFailure{cause: err, extra: JSONObject{{"events", dicts}}}
		}
		return nil, err
	}
	result := JSONObject{{"events", dicts}}
	for k, v := range responseResult(resp).All() {
		result.Set(k, v)
	}
	return result, nil
}

func vetNormalizeError(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	e := lm.NormalizeError(wireInt(msg.Get("status"), 0), wireStr(msg.Get("body_text")))
	var providerCode any
	if e.ProviderCode != "" {
		providerCode = e.ProviderCode
	}
	return JSONObject{{"class", e.ClassName()}, {"code", e.Code}, {"provider_code", providerCode}, {"message", e.Message}}, nil
}

// SerdeKinds maps the protocol's serde kinds to (from_dict, to_dict).
var SerdeKinds = map[string]struct {
	From func(JSONObject) (any, error)
	To   func(any) JSONObject
}{
	"part":                       {func(d JSONObject) (any, error) { return PartFromDict(d) }, func(v any) JSONObject { return PartToDict(v.(Part)) }},
	"message":                    {func(d JSONObject) (any, error) { return MessageFromDict(d) }, func(v any) JSONObject { return MessageToDict(v.(Message)) }},
	"tool":                       {func(d JSONObject) (any, error) { return ToolFromDict(d) }, func(v any) JSONObject { return ToolToDict(v.(Tool)) }},
	"tool_choice":                {func(d JSONObject) (any, error) { return ToolChoiceFromDict(d) }, func(v any) JSONObject { return ToolChoiceToDict(v.(ToolChoice)) }},
	"reasoning":                  {func(d JSONObject) (any, error) { return ReasoningFromDict(d) }, func(v any) JSONObject { return ReasoningToDict(v.(Reasoning)) }},
	"config":                     {func(d JSONObject) (any, error) { return ConfigFromDict(d) }, func(v any) JSONObject { return ConfigToDict(v.(Config)) }},
	"cache_config":               {func(d JSONObject) (any, error) { return CacheConfigFromDict(d) }, func(v any) JSONObject { return CacheConfigToDict(v.(CacheConfig)) }},
	"cache_info":                 {func(d JSONObject) (any, error) { return CacheInfoFromDict(d) }, func(v any) JSONObject { return CacheInfoToDict(v.(CacheInfo)) }},
	"cache_page":                 {func(d JSONObject) (any, error) { return CachePageFromDict(d) }, func(v any) JSONObject { return CachePageToDict(v.(CachePage)) }},
	"cached_prefix":              {func(d JSONObject) (any, error) { return CachedPrefixFromDict(d) }, func(v any) JSONObject { return CachedPrefixToDict(v.(CachedPrefix)) }},
	"token_logprob":              {func(d JSONObject) (any, error) { return TokenLogprobFromDict(d) }, func(v any) JSONObject { return TokenLogprobToDict(v.(TokenLogprob)) }},
	"continuation_state":         {func(d JSONObject) (any, error) { return ContinuationFromDict(d) }, func(v any) JSONObject { return ContinuationToDict(v.(ContinuationState)) }},
	"error_detail":               {func(d JSONObject) (any, error) { return ErrorDetailFromDict(d) }, func(v any) JSONObject { return ErrorDetailToDict(v.(ErrorDetail)) }},
	"delta":                      {func(d JSONObject) (any, error) { return DeltaFromDict(d) }, func(v any) JSONObject { return DeltaToDict(v.(Delta)) }},
	"usage":                      {func(d JSONObject) (any, error) { return UsageFromDict(d) }, func(v any) JSONObject { return UsageToDict(v.(Usage)) }},
	"credential":                 {func(d JSONObject) (any, error) { return CredentialFromDict(d) }, func(v any) JSONObject { return CredentialToDict(v.(Credential)) }},
	"stream_event":               {func(d JSONObject) (any, error) { return StreamEventFromDict(d) }, func(v any) JSONObject { return StreamEventToDict(v.(StreamEvent)) }},
	"request":                    {func(d JSONObject) (any, error) { return RequestFromDict(d) }, func(v any) JSONObject { return RequestToDict(v.(*Request)) }},
	"response":                   {func(d JSONObject) (any, error) { return ResponseFromDict(d) }, func(v any) JSONObject { return ResponseToDict(v.(*Response), false) }},
	"model_info":                 {func(d JSONObject) (any, error) { return ModelInfoFromDict(d) }, func(v any) JSONObject { return ModelInfoToDict(v.(ModelInfo)) }},
	"batch_request":              {func(d JSONObject) (any, error) { return BatchRequestFromDict(d) }, func(v any) JSONObject { return BatchRequestToDict(v.(BatchRequest)) }},
	"batch_job":                  {func(d JSONObject) (any, error) { return BatchJobFromDict(d) }, func(v any) JSONObject { return BatchJobToDict(v.(BatchJobInfo)) }},
	"batch_entry":                {func(d JSONObject) (any, error) { return BatchEntryFromDict(d) }, func(v any) JSONObject { return BatchEntryToDict(v.(BatchEntry)) }},
	"file_upload_request":        {func(d JSONObject) (any, error) { return FileUploadRequestFromDict(d) }, func(v any) JSONObject { return FileUploadRequestToDict(v.(FileUploadRequest)) }},
	"file_info":                  {func(d JSONObject) (any, error) { return FileInfoFromDict(d) }, func(v any) JSONObject { return FileInfoToDict(v.(FileInfo)) }},
	"file_page":                  {func(d JSONObject) (any, error) { return FilePageFromDict(d) }, func(v any) JSONObject { return FilePageToDict(v.(FilePage)) }},
	"image_generation_request":   {func(d JSONObject) (any, error) { return ImageGenerationRequestFromDict(d) }, func(v any) JSONObject { return ImageGenerationRequestToDict(v.(ImageGenerationRequest)) }},
	"image_generation_response":  {func(d JSONObject) (any, error) { return ImageGenerationResponseFromDict(d) }, func(v any) JSONObject { return ImageGenerationResponseToDict(v.(ImageGenerationResponse)) }},
	"speech_generation_request":  {func(d JSONObject) (any, error) { return SpeechGenerationRequestFromDict(d) }, func(v any) JSONObject { return SpeechGenerationRequestToDict(v.(SpeechGenerationRequest)) }},
	"speech_generation_response": {func(d JSONObject) (any, error) { return SpeechGenerationResponseFromDict(d) }, func(v any) JSONObject { return SpeechGenerationResponseToDict(v.(SpeechGenerationResponse)) }},
	"video_generation_request":   {func(d JSONObject) (any, error) { return VideoGenerationRequestFromDict(d) }, func(v any) JSONObject { return VideoGenerationRequestToDict(v.(VideoGenerationRequest)) }},
	"video_job":                  {func(d JSONObject) (any, error) { return VideoJobFromDict(d) }, func(v any) JSONObject { return VideoJobToDict(v.(VideoJobInfo)) }},
	"audio_format":               {func(d JSONObject) (any, error) { return AudioFormatFromDict(d) }, func(v any) JSONObject { return AudioFormatToDict(v.(AudioFormat)) }},
	"live_config":                {func(d JSONObject) (any, error) { return LiveConfigFromDict(d) }, func(v any) JSONObject { return LiveConfigToDict(v.(LiveConfig)) }},
	"live_client_event":          {func(d JSONObject) (any, error) { return LiveClientEventFromDict(d) }, func(v any) JSONObject { return LiveClientEventToDict(v.(LiveClientEvent)) }},
	"live_server_event":          {func(d JSONObject) (any, error) { return LiveServerEventFromDict(d) }, func(v any) JSONObject { return LiveServerEventToDict(v.(LiveServerEvent)) }},
}

func serdeRoundtrip(msg JSONObject) (JSONObject, error) {
	kind := wireStr(msg.Get("kind"))
	entry, ok := SerdeKinds[kind]
	if !ok {
		return nil, valueErrorf("unknown kind: %s", kind)
	}
	value, ok := asObject(msg.Get("value"))
	if !ok {
		return nil, typeErrorf("value must be a JSON object")
	}
	obj, err := entry.From(value)
	if err != nil {
		return nil, err
	}
	return entry.To(obj), nil
}

func vetSerdeRoundtrip(msg JSONObject) (JSONObject, error) {
	out, err := serdeRoundtrip(msg)
	if err != nil {
		return nil, err
	}
	return JSONObject{{"value", out}}, nil
}

func vetValidate(msg JSONObject) (JSONObject, error) {
	out, err := serdeRoundtrip(msg)
	if err != nil {
		return nil, err
	}
	return JSONObject{{"ok", true}, {"normalized", out}}, nil
}

func vetResolveModel(msg JSONObject) (JSONObject, error) {
	env := map[string]string{}
	for k, v := range wireObj(msg.Get("env")).All() {
		env[k] = wireStr(v)
	}
	var registry *ModelRegistry
	if _, has := msg.Lookup("catalog"); has {
		registry = NewModelRegistry()
		for _, entry := range wireList(msg.Get("catalog")) {
			obj, ok := asObject(entry)
			if !ok {
				return nil, typeErrorf("catalog entries must be objects")
			}
			info, err := ModelInfoFromDict(obj)
			if err != nil {
				return nil, err
			}
			if err := registry.Add(info, false); err != nil {
				return nil, err
			}
		}
	}
	res, err := Resolve(wireStr(msg.Get("model")), RouterConfig{Registry: registry, Env: env})
	if err != nil {
		return nil, err
	}
	return JSONObject{{"provider", res.Provider}, {"model", res.Model}, {"source", res.Source}}, nil
}

func vetBuildModelsRequest(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	return normalizedOrErr(hooks(lm).modelsRequest())
}

func vetParseModelsResponse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	body, err := vetBody(msg)
	if err != nil {
		return nil, err
	}
	if status := wireInt(msg.Get("status"), 200); status >= 400 {
		return nil, lm.NormalizeError(status, string(body))
	}
	models, err := hooks(lm).modelsFromBody(string(body))
	if err != nil {
		return nil, err
	}
	return JSONObject{{"models", toAnyList(models, func(m ModelInfo) any { return ModelInfoToDict(m) })}}, nil
}

// hooks exposes the dialect hooks of a constructed adapter.
func hooks(lm LM) dialect { return lm.(dialect) }

func vetGenerationBuild(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	gr := wireObj(msg.Get("generation_request"))
	switch wireStr(msg.Get("kind")) {
	case "image":
		req, err := ImageGenerationRequestFromDict(gr)
		if err != nil {
			return nil, err
		}
		return normalizedOrErr(hooks(lm).imageGenerateRequest(&req))
	case "speech":
		req, err := SpeechGenerationRequestFromDict(gr)
		if err != nil {
			return nil, err
		}
		return normalizedOrErr(hooks(lm).speechGenerateRequest(&req))
	}
	return nil, valueErrorf("unknown generation kind: %s", wireStr(msg.Get("kind")))
}

func vetGenerationParse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	body, err := vetBody(msg)
	if err != nil {
		return nil, err
	}
	status := wireInt(msg.Get("status"), 200)
	if status >= 400 {
		return nil, lm.NormalizeError(status, string(body))
	}
	resp := &HTTPResponse{Status: status, Reason: "OK", Headers: vetHeaders(msg), Body: body}
	gr := wireObj(msg.Get("generation_request"))
	switch wireStr(msg.Get("kind")) {
	case "image":
		req, err := ImageGenerationRequestFromDict(gr)
		if err != nil {
			return nil, err
		}
		out, err := hooks(lm).imageGenerationFromResponse(&req, resp)
		if err != nil {
			return nil, err
		}
		return ImageGenerationResponseToDict(out), nil
	case "speech":
		req, err := SpeechGenerationRequestFromDict(gr)
		if err != nil {
			return nil, err
		}
		out, err := hooks(lm).speechGenerationFromResponse(&req, resp)
		if err != nil {
			return nil, err
		}
		return SpeechGenerationResponseToDict(out), nil
	}
	return nil, valueErrorf("unknown generation kind: %s", wireStr(msg.Get("kind")))
}

func vetLimit(msg JSONObject) int {
	if v, ok := msg.Lookup("limit"); ok && v != nil {
		return wireInt(v, 20)
	}
	return 20
}

func vetFileOpBuild(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	switch op := wireStr(msg.Get("file_op")); op {
	case "upload":
		req, err := FileUploadRequestFromDict(wireObj(msg.Get("upload_request")))
		if err != nil {
			return nil, err
		}
		return normalizedOrErr(h.fileUploadRequest(&req))
	case "get":
		return normalizedOrErr(h.fileGetRequest(wireStr(msg.Get("file_id"))))
	case "list":
		return normalizedOrErr(h.fileListRequest(vetLimit(msg), stringOnly(msg.Get("cursor"))))
	case "delete":
		return normalizedOrErr(h.fileDeleteRequest(wireStr(msg.Get("file_id"))))
	case "download":
		return normalizedOrErr(h.fileDownloadRequest(wireStr(msg.Get("file_id"))))
	default:
		return nil, valueErrorf("unknown file_op: %s", op)
	}
}

func vetFileOpParse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	body, err := vetBody(msg)
	if err != nil {
		return nil, err
	}
	if status := wireInt(msg.Get("status"), 200); status >= 400 {
		return nil, lm.NormalizeError(status, string(body))
	}
	switch kind := wireStr(msg.Get("kind")); kind {
	case "info":
		info, err := hooks(lm).fileInfoFromBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"file", FileInfoToDict(info)}}, nil
	case "page":
		page, err := hooks(lm).filePageFromListBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"page", FilePageToDict(page)}}, nil
	default:
		return nil, valueErrorf("unknown file parse kind: %s", kind)
	}
}

func vetCacheOpBuild(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	var ttl *int
	if v, ok := msg.Lookup("ttl_seconds"); ok && v != nil {
		n, err := jsonInt(v, "ttl_seconds")
		if err != nil {
			return nil, err
		}
		ttl = &n
	}
	switch op := wireStr(msg.Get("cache_op")); op {
	case "create":
		prefix, err := RequestFromDict(wireObj(msg.Get("prefix_request")))
		if err != nil {
			return nil, err
		}
		if err := checkCachePrefix(prefix, ttl); err != nil {
			return nil, err
		}
		return normalizedOrErr(h.cacheCreateRequest(prefix, ttl, stringOnly(msg.Get("label"))))
	case "get":
		return normalizedOrErr(h.cacheGetRequest(wireStr(msg.Get("cache_id"))))
	case "list":
		return normalizedOrErr(h.cacheListRequest(vetLimit(msg), stringOnly(msg.Get("cursor"))))
	case "delete":
		return normalizedOrErr(h.cacheDeleteRequest(wireStr(msg.Get("cache_id"))))
	case "update":
		if ttl == nil {
			return nil, keyError("ttl_seconds")
		}
		return normalizedOrErr(h.cacheUpdateRequest(wireStr(msg.Get("cache_id")), *ttl))
	default:
		return nil, valueErrorf("unknown cache_op: %s", op)
	}
}

func vetCacheOpParse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	body, err := vetBody(msg)
	if err != nil {
		return nil, err
	}
	if status := wireInt(msg.Get("status"), 200); status >= 400 {
		return nil, lm.NormalizeError(status, string(body))
	}
	switch kind := wireStr(msg.Get("kind")); kind {
	case "info":
		info, err := hooks(lm).cacheInfoFromBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"cache", CacheInfoToDict(info)}}, nil
	case "page":
		page, err := hooks(lm).cachePageFromListBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"page", CachePageToDict(page)}}, nil
	default:
		return nil, valueErrorf("unknown cache parse kind: %s", kind)
	}
}

func requestsList(reqs []*TransportRequest) JSONObject {
	return JSONObject{{"requests", toAnyList(reqs, func(r *TransportRequest) any { return NormalizeTransportRequest(r) })}}
}

func vetVideoOpBuild(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	switch action := wireStr(msg.Get("action")); action {
	case "submit":
		req, err := VideoGenerationRequestFromDict(wireObj(msg.Get("video_request")))
		if err != nil {
			return nil, err
		}
		wire, err := h.videoSubmitRequest(&req)
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "status":
		wire, err := h.videoStatusRequest(wireStr(msg.Get("video_id")))
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "result_fetch":
		wire, err := h.videoResultFetch(wireObj(msg.Get("status_body")))
		if err != nil {
			return nil, err
		}
		if wire == nil {
			return requestsList(nil), nil
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "list":
		wire, err := h.videoListRequest(vetLimit(msg), stringOnly(msg.Get("model")))
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	default:
		return nil, valueErrorf("unknown video action: %s", action)
	}
}

func vetVideoOpParse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	switch kind := wireStr(msg.Get("kind")); kind {
	case "job":
		body, err := vetBody(msg)
		if err != nil {
			return nil, err
		}
		if status := wireInt(msg.Get("status"), 200); status >= 400 {
			return nil, lm.NormalizeError(status, string(body))
		}
		job, err := h.videoJobFromBody(string(body), stringOnly(msg.Get("video_id")))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"job", VideoJobToDict(job)}}, nil
	case "list":
		body, err := vetBody(msg)
		if err != nil {
			return nil, err
		}
		jobs, err := h.videoJobsFromListBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"jobs", toAnyList(jobs, func(j VideoJobInfo) any { return VideoJobToDict(j) })}}, nil
	case "part":
		var fetched *HTTPResponse
		if msg.Get("fetched_b64") != nil {
			raw, err := base64.StdEncoding.DecodeString(wireStr(msg.Get("fetched_b64")))
			if err != nil {
				return nil, err
			}
			fetched = &HTTPResponse{Status: 200, Reason: "OK", Headers: vetHeaders(msg), Body: raw}
		}
		part, err := h.videoPart(wireObj(msg.Get("status_body")), fetched)
		if err != nil {
			return nil, err
		}
		return JSONObject{{"part", PartToDict(part)}}, nil
	default:
		return nil, valueErrorf("unknown video parse kind: %s", kind)
	}
}

func vetBatchOpBuild(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, false)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	switch action := wireStr(msg.Get("action")); action {
	case "upload":
		req, err := BatchRequestFromDict(wireObj(msg.Get("batch_request")))
		if err != nil {
			return nil, err
		}
		wire, err := h.batchUploadRequest(&req, nil)
		if err != nil {
			return nil, err
		}
		if wire == nil {
			return requestsList(nil), nil
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "submit":
		req, err := BatchRequestFromDict(wireObj(msg.Get("batch_request")))
		if err != nil {
			return nil, err
		}
		wire, err := h.batchSubmitRequest(&req, wireObj(msg.Get("upload_body")), nil)
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "status":
		wire, err := h.batchStatusRequest(wireStr(msg.Get("batch_id")))
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "cancel":
		wire, err := h.batchCancelRequest(wireStr(msg.Get("batch_id")))
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "list":
		wire, err := h.batchListRequest(vetLimit(msg))
		if err != nil {
			return nil, err
		}
		return requestsList([]*TransportRequest{wire}), nil
	case "result_fetches":
		fetches, err := h.batchResultFetches(wireObj(msg.Get("status_body")))
		if err != nil {
			return nil, err
		}
		return requestsList(fetches), nil
	default:
		return nil, valueErrorf("unknown batch action: %s", action)
	}
}

func vetBatchOpParse(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	switch kind := wireStr(msg.Get("kind")); kind {
	case "job":
		body, err := vetBody(msg)
		if err != nil {
			return nil, err
		}
		if status := wireInt(msg.Get("status"), 200); status >= 400 {
			return nil, lm.NormalizeError(status, string(body))
		}
		job, err := h.batchJobFromBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"job", BatchJobToDict(job)}}, nil
	case "list":
		body, err := vetBody(msg)
		if err != nil {
			return nil, err
		}
		jobs, err := h.batchJobsFromListBody(string(body))
		if err != nil {
			return nil, err
		}
		return JSONObject{{"jobs", toAnyList(jobs, func(j BatchJobInfo) any { return BatchJobToDict(j) })}}, nil
	case "entries":
		var fetched []string
		for _, b := range wireList(msg.Get("fetched_b64")) {
			raw, err := base64.StdEncoding.DecodeString(wireStr(b))
			if err != nil {
				return nil, err
			}
			fetched = append(fetched, string(raw))
		}
		entries, err := h.batchEntries(wireObj(msg.Get("status_body")), fetched)
		if err != nil {
			return nil, err
		}
		return JSONObject{{"entries", toAnyList(entries, func(e BatchEntry) any { return BatchEntryToDict(e) })}}, nil
	default:
		return nil, valueErrorf("unknown batch parse kind: %s", kind)
	}
}

func vetReplayLive(msg JSONObject) (JSONObject, error) {
	lm, err := vetAdapter(msg, true)
	if err != nil {
		return nil, err
	}
	h := hooks(lm)
	config, err := LiveConfigFromDict(wireObj(msg.Get("live_config")))
	if err != nil {
		return nil, err
	}
	encoder := h.liveEncoder(&config)
	clientFrames := []any{}
	for _, raw := range wireList(msg.Get("client_events")) {
		obj, ok := asObject(raw)
		if !ok {
			return nil, typeErrorf("client_events must contain objects")
		}
		event, err := LiveClientEventFromDict(obj)
		if err != nil {
			return nil, err
		}
		frames, err := encoder(event)
		if err != nil {
			return nil, err
		}
		clientFrames = append(clientFrames, toAnyList(frames, func(f JSONObject) any { return f }))
	}
	events := []any{}
	for _, b64 := range wireList(msg.Get("server_frames_b64")) {
		raw, err := base64.StdEncoding.DecodeString(wireStr(b64))
		if err != nil {
			return nil, err
		}
		decoded, err := h.liveDecode(raw)
		if err != nil {
			return nil, err
		}
		events = append(events, toAnyList(decoded, func(e LiveServerEvent) any { return LiveServerEventToDict(e) }))
	}
	setup, err := h.liveSetupFrames(&config)
	if err != nil {
		return nil, err
	}
	return JSONObject{{"setup_frames", toAnyList(setup, func(f JSONObject) any { return f })}, {"client_frames", clientFrames}, {"events", events}}, nil
}

func vetExplainAuth(msg JSONObject) (JSONObject, error) {
	provider := wireStr(msg.Get("provider"))
	sentinel := wireStr(msg.Get("sentinel"))
	env := map[string]string{}
	for k, v := range wireObj(msg.Get("env")).All() {
		env[k] = wireStr(v)
	}
	opts := ExplainOptions{Env: env}
	if providers := wireList(msg.Get("api_keys_providers")); len(providers) > 0 {
		opts.APIKeys = map[string]CredentialLike{}
		for _, p := range providers {
			opts.APIKeys[wireStr(p)] = sentinel
		}
	}
	if files := wireObj(msg.Get("files")); files != nil {
		opts.Files = map[string]string{}
		for k, v := range files.All() {
			opts.Files[k] = wireStr(v)
		}
	}
	if env["HOME"] != "" {
		opts.Home = env["HOME"]
	}
	opts.Settings = vetSettings(msg)
	opts.Credential = wireStr(msg.Get("credential"))
	opts.BaseURL = wireStr(msg.Get("base_url"))
	if cp := msg.Get("credentials_path"); cp != nil {
		switch provider {
		case "claude-code":
			opts.ClaudeCredentialsPath = wireStr(cp)
		case "xai":
			opts.XaiCredentialsPath = wireStr(cp)
		default:
			opts.CodexAuthPath = wireStr(cp)
		}
	}
	report, err := ExplainAuth(provider, opts)
	if err != nil {
		return nil, err
	}
	steps := toAnyList(report.Steps, func(s AuthStep) any { return JSONObject{{"kind", s.Kind}, {"state", s.State}} })
	text := report.Describe()
	out := JSONObject{{"configured", report.Configured}, {"steps", steps}, {"report_text", strings.Join([]string{text, fmt.Sprintf("%+v", report.Steps), text}, "\n")}}
	if report.BaseURL != "" {
		out.Set("base_url", report.BaseURL)
	}
	return out, nil
}

func vetTokenExchangeBuild(msg JSONObject) (JSONObject, error) {
	def, ok := LookupProvider(wireStr(msg.Get("provider")))
	if !ok {
		return nil, valueErrorf("unknown provider: %s", wireStr(msg.Get("provider")))
	}
	inputs := wireObj(msg.Get("input"))
	if inputs == nil {
		inputs = wireObj(msg.Get("credential"))
	}
	if inputs == nil {
		inputs = JSONObject{}
	}
	env := map[string]string{}
	for k, v := range wireObj(inputs.Get("env")).All() {
		env[k] = wireStr(v)
	}
	var files map[string]string
	if pem := stringOnly(inputs.Get("certificate_pem")); pem != "" && env["AZURE_CLIENT_CERTIFICATE_PATH"] != "" {
		files = map[string]string{env["AZURE_CLIENT_CERTIFICATE_PATH"]: pem + "\n" + stringOnly(inputs.Get("private_key_pem"))}
	}
	fixed, err := ParseRFC3339(wireStr(msg.Get("now")))
	if err != nil {
		return nil, err
	}
	settings := map[string]string{}
	settingsSrc := wireObj(inputs.Get("settings"))
	if settingsSrc == nil {
		settingsSrc = wireObj(msg.Get("settings"))
	}
	for k, v := range settingsSrc.All() {
		settings[k] = wireStr(v)
	}
	ctx := &ChainContext{Env: env, Files: files, Now: func() time.Time { return fixed }, Settings: settings}
	return TokenExchangeBuild(def.Access, wireStr(msg.Get("rung")), inputs, ctx)
}

func vetTokenExchangeParse(msg JSONObject) (JSONObject, error) {
	def, ok := LookupProvider(wireStr(msg.Get("provider")))
	if !ok {
		return nil, valueErrorf("unknown provider: %s", wireStr(msg.Get("provider")))
	}
	fixed, err := ParseRFC3339(wireStr(msg.Get("now")))
	if err != nil {
		return nil, err
	}
	body := wireObj(msg.Get("body"))
	if body == nil && msg.Get("body_b64") != nil {
		raw, err := base64.StdEncoding.DecodeString(wireStr(msg.Get("body_b64")))
		if err != nil {
			return nil, err
		}
		if decoded, err := DecodeJSON(raw); err == nil {
			body = wireObj(decoded)
		}
	}
	if body == nil {
		body = JSONObject{}
	}
	status := 200
	if v, ok := msg.Lookup("status"); ok && v != nil {
		status = wireInt(v, 200)
	}
	ctx := &ChainContext{Env: map[string]string{}, Now: func() time.Time { return fixed }}
	cred, err := TokenExchangeParse(def.Access, wireStr(msg.Get("rung")), status, body, ctx)
	if err != nil {
		if e := AsError(err); e != nil {
			return JSONObject{{"ok", false}, {"error", JSONObject{{"class", e.ClassName()}, {"code", e.Code}}}}, nil
		}
		return nil, err
	}
	return JSONObject{{"ok", true}, {"credential", CredentialToDict(cred)}}, nil
}

func vetSigV4Sign(msg JSONObject) (JSONObject, error) {
	cred, err := CredentialFromDict(wireObj(msg.Get("credential")))
	if err != nil {
		return nil, err
	}
	aws, ok := cred.(AwsCredentials)
	if !ok {
		return nil, typeErrorf("credential must be aws")
	}
	req := wireObj(msg.Get("request"))
	headers := map[string]string{}
	for k, v := range wireObj(req.Get("headers")).All() {
		lk := strings.ToLower(k)
		if lk == "host" || lk == "x-amz-date" || lk == "x-amz-security-token" {
			continue
		}
		if list, ok := v.([]any); ok {
			parts := make([]string, 0, len(list))
			for _, item := range list {
				parts = append(parts, wireStr(item))
			}
			headers[k] = strings.Join(parts, ",")
		} else {
			headers[k] = wireStr(v)
		}
	}
	now, err := ParseRFC3339(wireStr(msg.Get("now")))
	if err != nil {
		return nil, err
	}
	sig := SigV4Sign(wireStr(req.Get("method")), wireStr(req.Get("url")), headers, []byte(wireStr(req.Get("body"))), aws, wireStr(msg.Get("region")), wireStr(msg.Get("service")), now)
	hdrs := JSONObject{}
	for _, k := range sortedMapKeys(sig.Headers) {
		hdrs = append(hdrs, Member{k, sig.Headers[k]})
	}
	return JSONObject{{"canonical_request", sig.CanonicalRequest}, {"string_to_sign", sig.StringToSign}, {"authorization", sig.Authorization}, {"headers", hdrs}}, nil
}
