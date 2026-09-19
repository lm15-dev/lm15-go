package lm15

import "strings"

// Compatibility policies: how a dialect serializes a canonical request for
// one server's quirks (reference lm15/compat.py). A field left "" inherits;
// "auto" is the dialect's own default. Every value is pinned by a receipt
// or the server's documentation, cited in the reference.

// ─── OpenAI Responses ────────────────────────────────────────────────

// OpenAIResponsesCompat is a partial policy for Responses-family servers.
type OpenAIResponsesCompat struct {
	DeveloperRole        string // auto | developer | system
	MaxOutputTokensField string // auto | max_output_tokens | max_completion_tokens | max_tokens
	ReasoningFormat      string // auto | none | responses_reasoning | reasoning_effort | openrouter | deepseek | qwen | qwen_chat_template | zai
	ToolResultName       string // auto | include | omit
	StrictTools          string // auto | include | omit
	CacheControl         string // auto | none | openai | openai_implicit | anthropic
	CommentaryPhase      string // auto | omit | tag
	EditImageField       string // auto | array | indexed
	BuiltinTools         string // auto | openai | verbatim
	ToolResultMedia      string // auto | native | images | reject
	Routing              JSONObject
	Extensions           JSONObject
}

// ResolvedOpenAIResponsesCompat is a fully resolved Responses policy.
type ResolvedOpenAIResponsesCompat struct {
	DeveloperRole        string
	MaxOutputTokensField string
	ReasoningFormat      string
	ToolResultName       string
	StrictTools          string
	CacheControl         string
	CommentaryPhase      string
	EditImageField       string
	BuiltinTools         string
	ToolResultMedia      string
	Routing              JSONObject
	Extensions           JSONObject
}

func pick(value, fallback string) string {
	if value == "" || value == "auto" {
		return fallback
	}
	return value
}

// ResolveOpenAIResponsesCompat fills every knob.
func ResolveOpenAIResponsesCompat(p OpenAIResponsesCompat) ResolvedOpenAIResponsesCompat {
	return ResolvedOpenAIResponsesCompat{
		DeveloperRole:        pick(p.DeveloperRole, "developer"),
		MaxOutputTokensField: pick(p.MaxOutputTokensField, "max_output_tokens"),
		ReasoningFormat:      pick(p.ReasoningFormat, "responses_reasoning"),
		ToolResultName:       pick(p.ToolResultName, "omit"),
		StrictTools:          pick(p.StrictTools, "omit"),
		CacheControl:         pick(p.CacheControl, "openai"),
		CommentaryPhase:      pick(p.CommentaryPhase, "omit"),
		EditImageField:       pick(p.EditImageField, "array"),
		BuiltinTools:         pick(p.BuiltinTools, "openai"),
		ToolResultMedia:      pick(p.ToolResultMedia, "native"),
		Routing:              p.Routing,
		Extensions:           p.Extensions,
	}
}

// MergeOpenAIResponsesCompat layers override on base ("" inherits).
func MergeOpenAIResponsesCompat(base OpenAIResponsesCompat, override *OpenAIResponsesCompat) OpenAIResponsesCompat {
	if override == nil {
		return base
	}
	out := base
	ov := func(dst *string, v string) {
		if v != "" {
			*dst = v
		}
	}
	ov(&out.DeveloperRole, override.DeveloperRole)
	ov(&out.MaxOutputTokensField, override.MaxOutputTokensField)
	ov(&out.ReasoningFormat, override.ReasoningFormat)
	ov(&out.ToolResultName, override.ToolResultName)
	ov(&out.StrictTools, override.StrictTools)
	ov(&out.CacheControl, override.CacheControl)
	ov(&out.CommentaryPhase, override.CommentaryPhase)
	ov(&out.EditImageField, override.EditImageField)
	ov(&out.BuiltinTools, override.BuiltinTools)
	ov(&out.ToolResultMedia, override.ToolResultMedia)
	if override.Routing != nil {
		out.Routing = override.Routing
	}
	out.Extensions = mergeJSON(base.Extensions, override.Extensions)
	return out
}

func mergeJSON(a, b JSONObject) JSONObject {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	out := copyObject(a)
	for k, v := range b {
		out[k] = v
	}
	return out
}

var openaiResponsesPresets = map[string]OpenAIResponsesCompat{
	"openai":     {DeveloperRole: "developer", MaxOutputTokensField: "max_output_tokens", ReasoningFormat: "responses_reasoning", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai"},
	"openrouter": {DeveloperRole: "developer", MaxOutputTokensField: "max_tokens", ReasoningFormat: "openrouter", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai", ToolResultMedia: "reject"},
	"ollama":     {DeveloperRole: "system", MaxOutputTokensField: "max_tokens", ReasoningFormat: "none", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"vllm":       {DeveloperRole: "system", MaxOutputTokensField: "max_tokens", ReasoningFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"sglang":     {DeveloperRole: "system", MaxOutputTokensField: "max_tokens", ReasoningFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"qwen":       {DeveloperRole: "system", MaxOutputTokensField: "max_tokens", ReasoningFormat: "qwen", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"deepseek":   {DeveloperRole: "system", MaxOutputTokensField: "max_tokens", ReasoningFormat: "deepseek", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"zai":        {DeveloperRole: "system", MaxOutputTokensField: "max_tokens", ReasoningFormat: "zai", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"meta":       {DeveloperRole: "developer", MaxOutputTokensField: "max_output_tokens", ReasoningFormat: "responses_reasoning", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai_implicit", CommentaryPhase: "tag", EditImageField: "indexed", BuiltinTools: "verbatim", ToolResultMedia: "native"},
	"moonshotai": {DeveloperRole: "developer", MaxOutputTokensField: "max_output_tokens", ReasoningFormat: "responses_reasoning", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai_implicit", BuiltinTools: "verbatim", ToolResultMedia: "images"},
}

// OpenAIResponsesPresetBaseURLs are the addresses the Responses presets name.
var OpenAIResponsesPresetBaseURLs = map[string]string{
	"openai":     "https://api.openai.com/v1",
	"ollama":     "http://localhost:11434/v1",
	"lmstudio":   "http://localhost:1234/v1",
	"vllm":       "http://localhost:8000/v1",
	"sglang":     "http://localhost:30000/v1",
	"openrouter": "https://openrouter.ai/api/v1",
	"meta":       "https://api.meta.ai/v1",
	"moonshotai": "https://api.moonshot.ai/v1",
}

func init() {
	openaiResponsesPresets["lmstudio"] = openaiResponsesPresets["ollama"]
	openaiChatPresets["lmstudio"] = openaiChatPresets["ollama"]
}

// OpenAIResponsesPreset returns the named Responses preset.
func OpenAIResponsesPreset(name string) (OpenAIResponsesCompat, error) {
	p, ok := openaiResponsesPresets[presetKey(name)]
	if !ok {
		return OpenAIResponsesCompat{}, valueErrorf("unknown OpenAIResponsesCompat preset: %q", name)
	}
	return p, nil
}

// ─── OpenAI Chat Completions ─────────────────────────────────────────

var chatOverridable = map[string]bool{
	"instruction_role": true, "max_tokens_field": true, "stream_usage": true, "thinking_format": true, "thinking_replay": true,
	"assistant_reasoning_content": true, "strict_tools": true, "cache_control": true, "user_field": true,
	"forced_tool_choice": true, "json_schema": true, "reasoning_efforts": true, "tool_result_media": true,
}

// ModelOverride is a per-model-family knob override (first matching prefix wins).
type ModelOverride struct {
	Prefix string
	Knobs  map[string]string
}

// OpenAIChatCompat is a partial policy for Chat Completions servers.
type OpenAIChatCompat struct {
	InstructionRole           string // auto | developer | system
	MaxTokensField            string // auto | max_completion_tokens | max_tokens
	StreamUsage               string // auto | include | omit
	ToolResultName            string // auto | include | omit
	AssistantAfterToolResult  string // auto | insert | omit
	ThinkingFormat            string // auto | none | reasoning_effort | openrouter | deepseek | kimi | qwen | qwen_chat_template
	ThinkingReplay            string // auto | native | as_text | omit
	AssistantReasoningContent string // auto | include_empty | omit
	StrictTools               string // auto | include | omit
	BuiltinTools              string // auto | reject | groq
	ToolResultMedia           string // auto | native | images | reject
	CacheControl              string // auto | none | openai | openai_implicit | anthropic
	UserField                 string // auto | user | user_id | safety_identifier
	ForcedToolChoice          string // auto | send | reject
	JSONSchema                string // auto | send | reject
	ReasoningEfforts          []string
	Routing                   JSONObject
	Extensions                JSONObject
	ModelOverrides            []ModelOverride
}

// ResolvedOpenAIChatCompat is a fully resolved Chat policy.
type ResolvedOpenAIChatCompat struct {
	InstructionRole           string
	MaxTokensField            string
	StreamUsage               string
	ToolResultName            string
	AssistantAfterToolResult  string
	ThinkingFormat            string
	ThinkingReplay            string
	AssistantReasoningContent string
	StrictTools               string
	BuiltinTools              string
	ToolResultMedia           string
	CacheControl              string
	UserField                 string
	ForcedToolChoice          string
	JSONSchema                string
	ReasoningEfforts          []string
	Routing                   JSONObject
	Extensions                JSONObject
}

// Validate checks the override knob names.
func (c OpenAIChatCompat) Validate() error {
	for _, o := range c.ModelOverrides {
		if o.Prefix == "" {
			return valueErrorf("model_overrides: each prefix is a non-empty string")
		}
		for name := range o.Knobs {
			if !chatOverridable[name] {
				return valueErrorf("model_overrides: %q is not an overridable knob", name)
			}
		}
	}
	for _, w := range c.ReasoningEfforts {
		if !inVocab(w, ReasoningEfforts) || w == "off" {
			return valueErrorf("reasoning_efforts must be ReasoningEffort words other than 'off'; got %v", c.ReasoningEfforts)
		}
	}
	return nil
}

// ForModel applies the first matching model override.
func (c OpenAIChatCompat) ForModel(model string) OpenAIChatCompat {
	for _, o := range c.ModelOverrides {
		if strings.HasPrefix(model, o.Prefix) {
			out := c
			out.ModelOverrides = nil
			for name, value := range o.Knobs {
				switch name {
				case "instruction_role":
					out.InstructionRole = value
				case "max_tokens_field":
					out.MaxTokensField = value
				case "stream_usage":
					out.StreamUsage = value
				case "thinking_format":
					out.ThinkingFormat = value
				case "thinking_replay":
					out.ThinkingReplay = value
				case "assistant_reasoning_content":
					out.AssistantReasoningContent = value
				case "strict_tools":
					out.StrictTools = value
				case "cache_control":
					out.CacheControl = value
				case "user_field":
					out.UserField = value
				case "forced_tool_choice":
					out.ForcedToolChoice = value
				case "json_schema":
					out.JSONSchema = value
				case "tool_result_media":
					out.ToolResultMedia = value
				case "reasoning_efforts":
					out.ReasoningEfforts = strings.Split(value, ",")
				}
			}
			return out
		}
	}
	return c
}

// ResolveOpenAIChatCompat fills every knob.
func ResolveOpenAIChatCompat(p OpenAIChatCompat) ResolvedOpenAIChatCompat {
	return ResolvedOpenAIChatCompat{
		InstructionRole:           pick(p.InstructionRole, "system"),
		MaxTokensField:            pick(p.MaxTokensField, "max_completion_tokens"),
		StreamUsage:               pick(p.StreamUsage, "include"),
		ToolResultName:            pick(p.ToolResultName, "omit"),
		AssistantAfterToolResult:  pick(p.AssistantAfterToolResult, "omit"),
		ThinkingFormat:            pick(p.ThinkingFormat, "reasoning_effort"),
		ThinkingReplay:            pick(p.ThinkingReplay, "as_text"),
		AssistantReasoningContent: pick(p.AssistantReasoningContent, "omit"),
		StrictTools:               pick(p.StrictTools, "omit"),
		BuiltinTools:              pick(p.BuiltinTools, "reject"),
		ToolResultMedia:           pick(p.ToolResultMedia, "reject"),
		CacheControl:              pick(p.CacheControl, "openai"),
		UserField:                 pick(p.UserField, "user"),
		ForcedToolChoice:          pick(p.ForcedToolChoice, "send"),
		JSONSchema:                pick(p.JSONSchema, "send"),
		ReasoningEfforts:          p.ReasoningEfforts,
		Routing:                   p.Routing,
		Extensions:                p.Extensions,
	}
}

var openaiChatPresets = map[string]OpenAIChatCompat{
	"openai":     {InstructionRole: "system", MaxTokensField: "max_completion_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai", ToolResultMedia: "reject"},
	"ollama":     {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "none", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"groq":       {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", BuiltinTools: "groq", CacheControl: "none", ToolResultMedia: "reject"},
	"openrouter": {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "openrouter", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai", ToolResultMedia: "reject"},
	"xai":        {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "deepseek", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "images"},
	"vllm":       {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"sglang":     {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", ToolResultMedia: "reject"},
	"deepseek":   {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "deepseek", ThinkingReplay: "native", AssistantReasoningContent: "include_empty", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", UserField: "user_id", ToolResultMedia: "reject"},
	"qwen":       {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "qwen", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none"},
	"bedrock": {InstructionRole: "system", MaxTokensField: "max_completion_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", UserField: "user", ForcedToolChoice: "send", JSONSchema: "send",
		ModelOverrides: []ModelOverride{
			{Prefix: "openai.gpt-oss", Knobs: map[string]string{"forced_tool_choice": "reject", "json_schema": "reject"}},
			{Prefix: "google.gemma", Knobs: map[string]string{"forced_tool_choice": "reject"}},
		}, ToolResultMedia: "reject"},
	"bedrock_mantle": {InstructionRole: "system", MaxTokensField: "max_completion_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", UserField: "user", ForcedToolChoice: "send", JSONSchema: "send",
		ModelOverrides: []ModelOverride{
			{Prefix: "openai.gpt-oss", Knobs: map[string]string{"forced_tool_choice": "reject", "json_schema": "reject"}},
		}},
	"zai":        {InstructionRole: "system", MaxTokensField: "max_tokens", StreamUsage: "include", ThinkingFormat: "deepseek", ThinkingReplay: "native", ToolResultName: "omit", StrictTools: "omit", CacheControl: "none", UserField: "user_id", ForcedToolChoice: "reject", JSONSchema: "reject", ToolResultMedia: "images"},
	"meta":       {InstructionRole: "developer", MaxTokensField: "max_completion_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai_implicit", UserField: "safety_identifier", ToolResultMedia: "reject"},
	"moonshotai": {InstructionRole: "system", MaxTokensField: "max_completion_tokens", StreamUsage: "include", ThinkingFormat: "kimi", ThinkingReplay: "native", ToolResultName: "omit", StrictTools: "omit", CacheControl: "openai_implicit", UserField: "safety_identifier", ReasoningEfforts: []string{"low", "high", "max"}, ToolResultMedia: "images"},
}

// OpenAIChatPresetBaseURLs are the addresses the Chat presets name.
var OpenAIChatPresetBaseURLs = map[string]string{
	"openai":     "https://api.openai.com/v1",
	"ollama":     "http://localhost:11434/v1",
	"lmstudio":   "http://localhost:1234/v1",
	"groq":       "https://api.groq.com/openai/v1",
	"openrouter": "https://openrouter.ai/api/v1",
	"xai":        "https://api.x.ai/v1",
	"vllm":       "http://localhost:8000/v1",
	"sglang":     "http://localhost:30000/v1",
	"deepseek":   "https://api.deepseek.com",
	"zai":        "https://api.z.ai/api/paas/v4",
	"meta":       "https://api.meta.ai/v1",
	"moonshotai": "https://api.moonshot.ai/v1",
}

// OpenAIChatPreset returns the named Chat preset.
func OpenAIChatPreset(name string) (OpenAIChatCompat, error) {
	p, ok := openaiChatPresets[presetKey(name)]
	if !ok {
		return OpenAIChatCompat{}, valueErrorf("unknown OpenAIChatCompat preset: %q", name)
	}
	return p, nil
}

// MergeOpenAIChatCompat layers override on base ("" inherits).
func MergeOpenAIChatCompat(base OpenAIChatCompat, override *OpenAIChatCompat) OpenAIChatCompat {
	if override == nil {
		return base
	}
	out := base
	ov := func(dst *string, v string) {
		if v != "" {
			*dst = v
		}
	}
	ov(&out.InstructionRole, override.InstructionRole)
	ov(&out.MaxTokensField, override.MaxTokensField)
	ov(&out.StreamUsage, override.StreamUsage)
	ov(&out.ToolResultName, override.ToolResultName)
	ov(&out.AssistantAfterToolResult, override.AssistantAfterToolResult)
	ov(&out.ThinkingFormat, override.ThinkingFormat)
	ov(&out.ThinkingReplay, override.ThinkingReplay)
	ov(&out.AssistantReasoningContent, override.AssistantReasoningContent)
	ov(&out.StrictTools, override.StrictTools)
	ov(&out.BuiltinTools, override.BuiltinTools)
	ov(&out.ToolResultMedia, override.ToolResultMedia)
	ov(&out.CacheControl, override.CacheControl)
	ov(&out.UserField, override.UserField)
	ov(&out.ForcedToolChoice, override.ForcedToolChoice)
	ov(&out.JSONSchema, override.JSONSchema)
	if override.ReasoningEfforts != nil {
		out.ReasoningEfforts = override.ReasoningEfforts
	}
	if override.Routing != nil {
		out.Routing = override.Routing
	}
	if override.ModelOverrides != nil {
		out.ModelOverrides = override.ModelOverrides
	}
	out.Extensions = mergeJSON(base.Extensions, override.Extensions)
	return out
}

// ─── Anthropic Messages ──────────────────────────────────────────────

// AnthropicCompat is a partial policy for Messages-family servers.
type AnthropicCompat struct {
	ThinkingFormat    string // auto | anthropic | deepseek | adaptive | effort
	ThinkingReplay    string // auto | signed | unsigned
	CacheControl      string // auto | anthropic | none
	StructuredOutput  string // auto | send | reject
	ParallelToolCalls string // auto | send | reject
	SamplingParams    string // auto | send | reject
	ToolResultMedia   string // auto | native | images | reject
	ReasoningEfforts  []string
	ModelPrefixes     []string
	Extensions        JSONObject
}

// ResolvedAnthropicCompat is a fully resolved Messages policy.
type ResolvedAnthropicCompat struct {
	ThinkingFormat    string
	ThinkingReplay    string
	CacheControl      string
	StructuredOutput  string
	ParallelToolCalls string
	SamplingParams    string
	ToolResultMedia   string
	ReasoningEfforts  []string
	ModelPrefixes     []string
	Extensions        JSONObject
}

// ResolveAnthropicCompat fills every knob.
func ResolveAnthropicCompat(p AnthropicCompat) ResolvedAnthropicCompat {
	return ResolvedAnthropicCompat{
		ThinkingFormat:    pick(p.ThinkingFormat, "anthropic"),
		ThinkingReplay:    pick(p.ThinkingReplay, "signed"),
		CacheControl:      pick(p.CacheControl, "anthropic"),
		StructuredOutput:  pick(p.StructuredOutput, "send"),
		ParallelToolCalls: pick(p.ParallelToolCalls, "send"),
		SamplingParams:    pick(p.SamplingParams, "send"),
		ToolResultMedia:   pick(p.ToolResultMedia, "native"),
		ReasoningEfforts:  p.ReasoningEfforts,
		ModelPrefixes:     p.ModelPrefixes,
		Extensions:        p.Extensions,
	}
}

var anthropicPresets = map[string]AnthropicCompat{
	"anthropic":  {},
	"deepseek":   {ThinkingFormat: "deepseek", CacheControl: "none", StructuredOutput: "reject", ParallelToolCalls: "reject", ModelPrefixes: []string{"deepseek-"}, ToolResultMedia: "reject"},
	"meta":       {ThinkingFormat: "adaptive", CacheControl: "none", StructuredOutput: "send", ParallelToolCalls: "send", ToolResultMedia: "native"},
	"moonshotai": {ThinkingFormat: "effort", ThinkingReplay: "unsigned", CacheControl: "none", StructuredOutput: "send", ParallelToolCalls: "reject", SamplingParams: "reject", ReasoningEfforts: []string{"low", "high", "max"}, ModelPrefixes: []string{"kimi-"}, ToolResultMedia: "images"},
}

// AnthropicPresetBaseURLs are the addresses the Anthropic presets name.
var AnthropicPresetBaseURLs = map[string]string{
	"anthropic":  "https://api.anthropic.com/v1",
	"deepseek":   "https://api.deepseek.com/anthropic/v1",
	"meta":       "https://api.meta.ai/v1",
	"moonshotai": "https://api.moonshot.ai/anthropic/v1",
}

// AnthropicPreset returns the named Anthropic preset.
func AnthropicPreset(name string) (AnthropicCompat, error) {
	p, ok := anthropicPresets[presetKey(name)]
	if !ok {
		return AnthropicCompat{}, valueErrorf("unknown AnthropicCompat preset: %q", name)
	}
	return p, nil
}

// ─── Shared ──────────────────────────────────────────────────────────

var presetAliases = map[string]string{
	"openai_chat":      "openai",
	"chat":             "openai",
	"chat_completions": "openai",
	"responses":        "openai",
	"openai_responses": "openai",
	"lm_studio":        "lmstudio",
	"dashscope_qwen":   "qwen",
	"z_ai":             "zai",
}

func presetKey(name string) string {
	key := strings.ToLower(name)
	key = strings.NewReplacer("-", "_", " ", "_", ".", "_").Replace(key)
	if alias, ok := presetAliases[key]; ok {
		return alias
	}
	return key
}

// PresetBaseURL is the address a preset name supplies for one dialect. A
// name with no entry (other than the dialect's own default) is refused with
// NotConfiguredError: a request for a named server never goes to the cloud
// default with whatever key is around (2026-09-11).
func PresetBaseURL(table map[string]string, name, dialect, defaultPreset string) (string, error) {
	key := presetKey(name)
	if url, ok := table[key]; ok {
		return url, nil
	}
	if key == defaultPreset {
		return table[defaultPreset], nil
	}
	return "", NotConfiguredErrorf("", nil, "",
		"compat %q names a server whose %s address lm15 does not know; pass base_url= (the server's OpenAI-compatible root, e.g. 'http://localhost:PORT/v1')",
		name, dialect)
}

// openaiResponsesCompatFromExtensions reads the per-request Responses hatch:
// extensions["openai_responses_compat"] / ["openai_compat"] / ["compat"]["openai_responses"|"openai"].
func openaiResponsesCompatFromExtensions(ext JSONObject) *OpenAIResponsesCompat {
	if len(ext) == 0 {
		return nil
	}
	raw := ext["openai_responses_compat"]
	if raw == nil {
		raw = ext["openai_compat"]
	}
	if raw == nil {
		if compat, ok := ext["compat"].(map[string]any); ok {
			raw = compat["openai_responses"]
			if raw == nil {
				raw = compat["openai"]
			}
		}
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return nil
	}
	get := func(key string) string {
		s, _ := m[key].(string)
		return s
	}
	out := &OpenAIResponsesCompat{
		DeveloperRole:        get("developer_role"),
		MaxOutputTokensField: get("max_output_tokens_field"),
		ReasoningFormat:      get("reasoning_format"),
		ToolResultName:       get("tool_result_name"),
		StrictTools:          get("strict_tools"),
		CacheControl:         get("cache_control"),
		CommentaryPhase:      get("commentary_phase"),
		EditImageField:       get("edit_image_field"),
		BuiltinTools:         get("builtin_tools"),
		ToolResultMedia:      get("tool_result_media"),
	}
	if r, ok := m["routing"].(map[string]any); ok {
		out.Routing = r
	}
	if e, ok := m["extensions"].(map[string]any); ok {
		out.Extensions = e
	}
	return out
}
