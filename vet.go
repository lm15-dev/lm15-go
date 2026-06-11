package lm15

import (
	"reflect"
	"sort"
	"strings"
)

// ImplVersion is the lm15-go implementation version reported by the shim.
const ImplVersion = "2.0.0-dev"

// SerdeRoundtrip implements the vet protocol's serde_roundtrip op:
// to_dict(from_dict(value)) for the given kind, no cleaning beyond the
// canonical serializers themselves.
func SerdeRoundtrip(kind string, value jmap) (jmap, error) {
	switch kind {
	case "part":
		p, err := PartFromMap(value)
		if err != nil {
			return nil, err
		}
		return p.toMap(), nil
	case "message":
		v, err := messageFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "tool":
		v, err := ToolFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "tool_choice":
		v, err := toolChoiceFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "reasoning":
		v, err := reasoningFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "config":
		v, err := configFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "cache_config":
		v, err := cacheConfigFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "continuation_state":
		v, err := continuationStateFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "error_detail":
		v, err := errorDetailFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "delta":
		v, err := DeltaFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "usage":
		v, err := usageFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "stream_event":
		v, err := StreamEventFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "request":
		v, err := requestFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap()
	case "response":
		v, err := responseFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "model_info":
		v, err := modelInfoFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "audio_format":
		v, err := audioFormatFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "live_config":
		v, err := liveConfigFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap()
	case "live_client_event":
		v, err := LiveClientEventFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	case "live_server_event":
		v, err := LiveServerEventFromMap(value)
		if err != nil {
			return nil, err
		}
		return v.toMap(), nil
	}
	return nil, valueErr("unknown kind: %s", kind)
}

// ─── surface_dump ────────────────────────────────────────────────────

// surfaceTypes is an explicit type registry; field names are derived from
// struct reflection over `json` tags (deviation from the reference, which
// reflects dataclasses directly — noted per the port instructions).
var surfaceTypes = []any{
	ContinuationState{}, TextPart{}, ThinkingPart{}, RefusalPart{},
	CitationPart{}, ImagePart{}, AudioPart{}, VideoPart{}, DocumentPart{},
	BinaryPart{}, ToolCallPart{}, ToolResultPart{}, Message{},
	FunctionTool{}, BuiltinTool{}, ToolChoice{}, Reasoning{}, CacheConfig{},
	Config{}, ErrorDetail{}, TextDelta{}, ThinkingDelta{}, AudioDelta{},
	ImageDelta{}, ToolCallDelta{}, CitationDelta{}, ContinuationDelta{},
	Usage{}, StreamStartEvent{}, StreamDeltaEvent{}, StreamEndEvent{},
	StreamErrorEvent{}, Request{}, Response{}, AudioFormat{}, LiveConfig{},
	LiveClientTurnEvent{}, LiveClientAudioEvent{}, LiveClientImageEvent{},
	LiveClientTextEvent{}, LiveClientToolResultEvent{},
	LiveClientInterruptEvent{}, LiveClientEndAudioEvent{},
	LiveServerAudioEvent{}, LiveServerTextEvent{}, LiveServerToolCallEvent{},
	LiveServerToolCallDeltaEvent{}, LiveServerInterruptedEvent{},
	LiveServerTurnEndEvent{}, LiveServerErrorEvent{},
	EmbeddingRequest{}, EmbeddingResponse{}, FileUploadRequest{},
	FileUploadResponse{}, BatchRequest{}, BatchResponse{},
	ImageGenerationRequest{}, ImageGenerationResponse{},
	AudioGenerationRequest{}, AudioGenerationResponse{}, ToolCallInfo{},
	ModelInfo{}, ModelOrigin{}, InferenceModelInfo{}, TrainingModelInfo{},
	InferencePricing{}, TrainingPricing{},
}

func reflectFields(t reflect.Type) []string {
	var out []string
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.Anonymous && f.Type.Kind() == reflect.Struct {
			out = append(out, reflectFields(f.Type)...)
			continue
		}
		tag := strings.Split(f.Tag.Get("json"), ",")[0]
		if tag == "" || tag == "-" {
			continue
		}
		out = append(out, tag)
	}
	return out
}

// SurfaceDump returns the type/enum surface for the vet protocol.
func SurfaceDump() jmap {
	types := jmap{}
	for _, v := range surfaceTypes {
		t := reflect.TypeOf(v)
		fields := reflectFields(t)
		anyFields := make([]any, len(fields))
		for i, f := range fields {
			anyFields[i] = f
		}
		types[t.Name()] = jmap{"fields": anyFields}
	}
	enums := jmap{}
	for name, values := range map[string][]string{
		"Role":                RoleValues,
		"PartType":            PartTypes,
		"DeltaType":           DeltaTypes,
		"FinishReason":        FinishReasons,
		"ReasoningEffort":     ReasoningEfforts,
		"ReasoningSummary":    ReasoningSummaries,
		"ErrorCode":           ErrorCodes,
		"StreamEventType":     StreamEventTypes,
		"BatchStatus":         BatchStatuses,
		"AudioEncoding":       AudioEncodings,
		"ToolChoiceMode":      ToolChoiceModes,
		"CacheMode":           CacheModes,
		"CacheRetention":      CacheRetentions,
		"LiveClientEventType": LiveClientEventTypes,
		"LiveServerEventType": LiveServerEventTypes,
	} {
		sorted := append([]string(nil), values...)
		sort.Strings(sorted)
		anyValues := make([]any, len(sorted))
		for i, v := range sorted {
			anyValues[i] = v
		}
		enums[name] = anyValues
	}
	return jmap{"types": types, "enums": enums}
}
