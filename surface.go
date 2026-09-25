package lm15

import (
	"reflect"
	"sort"
	"strings"
	"unicode"
)

// SurfaceDump is the reflection-driven surface the vet protocol's
// surface_dump op reports: every canonical type's fields (snake_case, from
// the struct fields), every vocabulary, and every provider's access policy.
func SurfaceDump() JSONObject {
	types := JSONObject{}
	for _, t := range canonicalTypes() {
		rt := reflect.TypeOf(t)
		var fields []any
		for _, f := range flattenFields(rt) {
			fields = append(fields, f)
		}
		types.Set(rt.Name(), JSONObject{{"fields", fields}})
	}
	enums := JSONObject{}
	for name, values := range Vocabularies {
		enums.Set(name, toAnyList(values, func(s string) any { return s }))
	}
	providers := JSONObject{}
	for _, id := range ProviderIDs() {
		def := Providers[id]
		s := def.Access.Supports
		supports := JSONObject{}
		for _, name := range endpointSupportFields {
			supports.Set(name, s.SupportsEndpoint(name))
		}
		extra := append([]string(nil), s.Extra...)
		sort.Strings(extra)
		supports.Set("extra", toAnyList(extra, func(x string) any { return x }))
		providers.Set(id, JSONObject{
			{"supports", supports},
			{"auth_modes", toAnyList(def.Access.AuthModes, func(x string) any { return x })},
			{"env_keys", toAnyList(def.Access.EnvKeys, func(x string) any { return x })},
		})
	}
	return JSONObject{{"types", types}, {"enums", enums}, {"providers", providers}}
}

// canonicalTypes lists one zero value per public canonical type; the field
// names themselves come from reflection.
func canonicalTypes() []any {
	return []any{
		ContinuationState{}, TextPart{}, ImagePart{}, AudioPart{}, VideoPart{}, DocumentPart{}, BinaryPart{},
		ToolCallPart{}, ToolResultPart{}, ThinkingPart{}, RefusalPart{}, CitationPart{}, DataPart{}, Message{},
		TextDelta{}, ThinkingDelta{}, AudioDelta{}, ImageDelta{}, ToolCallDelta{}, CitationDelta{}, ContinuationDelta{},
		ErrorDetail{}, HTTPResponseDetail{}, StreamStartEvent{}, StreamDeltaEvent{}, StreamEndEvent{}, StreamErrorEvent{},
		Adaptation{}, CredentialSource{},
		FunctionTool{}, BuiltinTool{}, Reasoning{}, CacheConfig{}, ToolChoice{}, Config{}, Request{},
		TopLogprob{}, TokenLogprob{}, Usage{}, Response{}, FileUploadRequest{}, FileInfo{}, FilePage{},
		CacheInfo{}, CachePage{}, CachedPrefix{}, BatchRequest{}, BatchJobInfo{}, BatchEntry{},
		ImageGenerationRequest{}, ImageGenerationResponse{}, SpeechGenerationRequest{}, SpeechGenerationResponse{},
		VideoGenerationRequest{}, VideoJobInfo{}, AudioFormat{}, LiveConfig{},
		LiveClientTurnEvent{}, LiveClientAudioEvent{}, LiveClientImageEvent{}, LiveClientTextEvent{}, LiveClientToolResultEvent{},
		LiveClientInterruptEvent{}, LiveClientEndAudioEvent{}, LiveServerAudioEvent{}, LiveServerTextEvent{}, LiveServerToolCallEvent{},
		LiveServerToolCallDeltaEvent{}, LiveServerInterruptedEvent{}, LiveServerTurnEndEvent{}, LiveServerUsageEvent{}, LiveServerErrorEvent{},
		ToolCallInfo{},
	}
}

func flattenFields(rt reflect.Type) []string {
	var out []string
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		if f.Anonymous {
			out = append(out, flattenFields(f.Type)...)
			continue
		}
		if !f.IsExported() {
			continue
		}
		out = append(out, snakeCase(f.Name))
	}
	return out
}

var initialisms = map[string]string{"ID": "id", "URL": "url", "TopP": "top_p", "TopK": "top_k", "APIFamily": "api_family", "UserID": "user_id", "FileID": "file_id", "TokenID": "token_id", "SizeBytes": "size_bytes", "IsError": "is_error", "HTTPResponse": "http_response", "LogprobsIncomplete": "logprobs_complete"}

// snakeCase turns a Go field name into its canonical JSON key.
func snakeCase(name string) string {
	if v, ok := initialisms[name]; ok {
		return v
	}
	name = strings.ReplaceAll(name, "ID", "Id")
	name = strings.ReplaceAll(name, "URL", "Url")
	var b strings.Builder
	runes := []rune(name)
	for i, r := range runes {
		if unicode.IsUpper(r) {
			if i > 0 && (unicode.IsLower(runes[i-1]) || unicode.IsDigit(runes[i-1]) || (i+1 < len(runes) && unicode.IsLower(runes[i+1]) && unicode.IsUpper(runes[i-1]))) {
				b.WriteByte('_')
			}
			b.WriteRune(unicode.ToLower(r))
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}
