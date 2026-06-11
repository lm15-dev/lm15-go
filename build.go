package lm15

// build.go — request building for the four wire adapters (vet op
// build_request). Behavior is pinned by lm15-contract cases/ wire fixtures;
// the canonical→provider mapping mirrors the contract spec (and, where the
// spec is silent, the Python reference adapters).

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// TransportRequest is the pure build_request output: the wire request the
// transport would send. Params are decoded query parameters (PROTOCOL.md:
// encoding is the transport's responsibility); header names are lowercase.
type TransportRequest struct {
	Method  string
	URL     string
	Params  map[string]string
	Headers map[string]string
	Body    jmap
}

// DefaultBaseURLs per provider.
var DefaultBaseURLs = map[string]string{
	"openai":      "https://api.openai.com/v1",
	"openai_chat": "https://api.openai.com/v1",
	"anthropic":   "https://api.anthropic.com/v1",
	"gemini":      "https://generativelanguage.googleapis.com/v1beta",
}

const anthropicAPIVersion = "2023-06-01"

// BuildRequest builds the provider wire request for a canonical Request.
// baseURL == "" means the provider default.
func BuildRequest(provider string, req Request, stream bool, apiKey, baseURL string) (TransportRequest, error) {
	if baseURL == "" {
		baseURL = DefaultBaseURLs[provider]
	}
	baseURL = strings.TrimRight(baseURL, "/")
	switch provider {
	case "openai":
		return buildOpenAI(req, stream, apiKey, baseURL)
	case "openai_chat":
		return buildOpenAIChat(req, stream, apiKey, baseURL, defaultChatCompat())
	case "anthropic":
		return buildAnthropic(req, stream, apiKey, baseURL)
	case "gemini":
		return buildGemini(req, stream, apiKey, baseURL)
	}
	return TransportRequest{}, valueErr("unknown provider: %s", provider)
}

// ─── shared helpers ──────────────────────────────────────────────────

// partsToText is the lossy text rendering for text-only provider fields.
func partsToText(parts []Part) string {
	var out []string
	for _, p := range parts {
		switch x := p.(type) {
		case TextPart:
			out = append(out, x.Text)
		case ThinkingPart:
			if x.Text != "" {
				out = append(out, x.Text)
			}
		case CitationPart:
			var bits []string
			for _, s := range []*string{x.Title, x.URL, x.Text} {
				if s != nil && *s != "" {
					bits = append(bits, *s)
				}
			}
			if len(bits) > 0 {
				out = append(out, strings.Join(bits, " — "))
			}
		}
	}
	return strings.Join(out, "\n")
}

func systemText(system any) string {
	switch s := system.(type) {
	case string:
		return s
	case []Part:
		return partsToText(s)
	}
	return ""
}

func mediaOf(p Part) (mediaPart, bool) {
	switch x := p.(type) {
	case ImagePart:
		return x.mediaPart, true
	case AudioPart:
		return x.mediaPart, true
	case VideoPart:
		return x.mediaPart, true
	case DocumentPart:
		return x.mediaPart, true
	case BinaryPart:
		return x.mediaPart, true
	}
	return mediaPart{}, false
}

func mediaDataURI(m mediaPart) (string, error) {
	if m.Data == nil {
		return "", valueErr("media part has no inline data")
	}
	return fmt.Sprintf("data:%s;base64,%s", m.MediaType, *m.Data), nil
}

// compactJSON renders v as compact JSON without HTML escaping (the wire
// form of json.dumps(..., separators=(",", ":"), ensure_ascii=False)).
// Map keys are emitted in sorted order.
func compactJSON(v any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "{}"
	}
	return strings.TrimRight(buf.String(), "\n")
}

func emptyToolResultOutput(content []Part) string {
	items := make([]string, len(content))
	for i, p := range content {
		items[i] = fmt.Sprintf(`{"type": "%s"}`, p.PartType())
	}
	return "[" + strings.Join(items, ", ") + "]"
}

func toolResultOutput(p ToolResultPart) string {
	out := partsToText(p.Content)
	if out == "" {
		out = emptyToolResultOutput(p.Content)
	}
	return out
}

// continuationData finds the data of a part's continuation state matching
// provider+kind, or nil.
func continuationData(p Part, provider, kind string) jmap {
	for _, c := range p.Continuations() {
		if c.Provider == provider && c.Kind == kind {
			return c.Data
		}
	}
	return nil
}

// copyMap shallow-copies a jmap.
func copyMap(m jmap) jmap {
	out := jmap{}
	for k, v := range m {
		out[k] = v
	}
	return out
}

func anyList(items []string) []any {
	out := make([]any, len(items))
	for i, s := range items {
		out[i] = s
	}
	return out
}

func mappedEffort(effort string) string {
	switch effort {
	case "adaptive":
		return "medium"
	case "xhigh":
		return "high"
	}
	return effort
}

func toolDescription(t FunctionTool) any {
	if t.Description == nil {
		return nil
	}
	return *t.Description
}

func toolParameters(t FunctionTool) jmap {
	if t.Parameters == nil {
		return DefaultParameters()
	}
	return t.Parameters
}

func extensionsPassthrough(payload jmap, extensions jmap, reserved map[string]bool) {
	for k, v := range extensions {
		if !reserved[k] {
			payload[k] = v
		}
	}
}

// ─── OpenAI (Responses API) ──────────────────────────────────────────

var openAIBuiltinMap = map[string]string{
	"web_search":     "web_search_preview",
	"code_execution": "code_interpreter",
	"file_search":    "file_search",
	"computer_use":   "computer_use_preview",
}

func builtinToOpenAI(t BuiltinTool) jmap {
	name, ok := openAIBuiltinMap[t.Name]
	if !ok {
		name = t.Name
	}
	out := jmap{"type": name}
	for k, v := range t.Config {
		out[k] = v
	}
	return out
}

func partToOpenAIInput(p Part) (jmap, error) {
	switch x := p.(type) {
	case TextPart:
		return jmap{"type": "input_text", "text": x.Text}, nil
	case ImagePart:
		switch {
		case x.URL != nil:
			out := jmap{"type": "input_image", "image_url": *x.URL}
			if x.Detail != nil && *x.Detail != "" {
				out["detail"] = *x.Detail
			}
			return out, nil
		case x.Data != nil:
			uri, err := mediaDataURI(x.mediaPart)
			if err != nil {
				return nil, err
			}
			out := jmap{"type": "input_image", "image_url": uri}
			if x.Detail != nil && *x.Detail != "" {
				out["detail"] = *x.Detail
			}
			return out, nil
		case x.FileID != nil:
			return jmap{"type": "input_image", "file_id": *x.FileID}, nil
		}
	case AudioPart:
		switch {
		case x.Data != nil:
			mt := x.MediaType
			if mt == "" {
				mt = "audio/wav"
			}
			media := mt[strings.Index(mt, "/")+1:]
			if media == "mpeg" || media == "mp3" {
				media = "mp3"
			}
			return jmap{"type": "input_audio", "audio": *x.Data, "format": media}, nil
		case x.URL != nil:
			return jmap{"type": "input_audio", "audio_url": *x.URL}, nil
		case x.FileID != nil:
			return jmap{"type": "input_audio", "file_id": *x.FileID}, nil
		}
	case DocumentPart, BinaryPart:
		m, _ := mediaOf(p)
		switch {
		case m.URL != nil:
			return jmap{"type": "input_file", "file_url": *m.URL}, nil
		case m.Data != nil:
			mt := m.MediaType
			if mt == "" {
				mt = "application/octet-stream"
			}
			ext := mt[strings.Index(mt, "/")+1:]
			if i := strings.Index(ext, "+"); i >= 0 {
				ext = ext[:i]
			}
			if ext == "" {
				ext = "bin"
			}
			uri, err := mediaDataURI(m)
			if err != nil {
				return nil, err
			}
			return jmap{"type": "input_file", "filename": "file." + ext, "file_data": uri}, nil
		case m.FileID != nil:
			return jmap{"type": "input_file", "file_id": *m.FileID}, nil
		}
	case VideoPart:
		switch {
		case x.URL != nil:
			return jmap{"type": "input_video", "video_url": *x.URL}, nil
		case x.Data != nil:
			uri, err := mediaDataURI(x.mediaPart)
			if err != nil {
				return nil, err
			}
			return jmap{"type": "input_video", "video_data": uri}, nil
		case x.FileID != nil:
			return jmap{"type": "input_video", "file_id": *x.FileID}, nil
		}
	case ToolResultPart:
		return jmap{"type": "input_text", "text": partsToText(x.Content)}, nil
	case CitationPart:
		return jmap{"type": "input_text", "text": partsToText([]Part{x})}, nil
	case ThinkingPart:
		return jmap{"type": "input_text", "text": x.Text}, nil
	case RefusalPart:
		return jmap{"type": "input_text", "text": x.Text}, nil
	}
	return jmap{"type": "input_text", "text": ""}, nil
}

func openAIBuildInput(messages []Message) ([]any, error) {
	var items []any
	for _, msg := range messages {
		if msg.Role == "tool" {
			for _, p := range msg.Parts {
				if tr, ok := p.(ToolResultPart); ok {
					items = append(items, jmap{
						"type":    "function_call_output",
						"call_id": tr.ID,
						"output":  toolResultOutput(tr),
					})
				}
			}
			continue
		}
		var content []any
		if msg.Role == "assistant" {
			for _, p := range msg.Parts {
				switch x := p.(type) {
				case TextPart:
					content = append(content, jmap{"type": "output_text", "text": x.Text})
				case RefusalPart:
					content = append(content, jmap{"type": "refusal", "refusal": x.Text})
				}
			}
		} else {
			for _, p := range msg.Parts {
				switch p.(type) {
				case ToolCallPart, ToolResultPart:
					continue
				}
				item, err := partToOpenAIInput(p)
				if err != nil {
					return nil, err
				}
				content = append(content, item)
			}
		}
		if len(content) > 0 {
			role := msg.Role
			if role == "developer" {
				role = "developer" // resolved compat developer_role
			}
			items = append(items, jmap{"role": role, "content": content})
		}
		for _, p := range msg.Parts {
			if tc, ok := p.(ToolCallPart); ok {
				input := tc.Input
				if input == nil {
					input = jmap{}
				}
				items = append(items, jmap{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Name,
					"arguments": compactJSON(input),
				})
			}
		}
	}
	return items, nil
}

func openAIToolChoice(tc *ToolChoice) any {
	if tc == nil {
		return nil
	}
	if tc.Mode == "none" {
		return "none"
	}
	if len(tc.Allowed) == 1 {
		return jmap{"type": "function", "name": tc.Allowed[0]}
	}
	if tc.Mode == "required" {
		return "required"
	}
	return "auto"
}

func responseFormatToOpenAIText(fc jmap) jmap {
	if text, ok := fc["text"].(jmap); ok {
		return text
	}
	if _, ok := fc["format"].(jmap); ok {
		return copyMap(fc)
	}
	switch fc["type"] {
	case "json_schema":
		format := copyMap(fc)
		if _, ok := format["name"]; !ok {
			format["name"] = "response"
		}
		return jmap{"format": format}
	case "json_object":
		return jmap{"format": copyMap(fc)}
	}
	schema, ok := fc["schema"].(jmap)
	if !ok {
		schema = fc
	}
	name := "response"
	if n, ok := fc["name"].(string); ok && n != "" {
		name = n
	}
	return jmap{"format": jmap{"type": "json_schema", "name": name, "schema": schema}}
}

func buildOpenAI(req Request, stream bool, apiKey, baseURL string) (TransportRequest, error) {
	cfg := req.Config
	if cfg == nil {
		cfg = &Config{}
	}
	input, err := openAIBuildInput(req.Messages)
	if err != nil {
		return TransportRequest{}, err
	}
	payload := jmap{"model": req.Model, "input": input, "stream": stream}
	if req.System != nil {
		payload["instructions"] = systemText(req.System)
	}
	if cfg.MaxTokens != nil {
		payload["max_output_tokens"] = jsonInt(*cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		payload["temperature"] = jsonFloat(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		payload["top_p"] = jsonFloat(*cfg.TopP)
	}
	if len(req.Tools) > 0 {
		var tools []any
		for _, t := range req.Tools {
			switch x := t.(type) {
			case FunctionTool:
				tools = append(tools, jmap{
					"type":        "function",
					"name":        x.Name,
					"description": toolDescription(x),
					"parameters":  toolParameters(x),
				})
			case BuiltinTool:
				tools = append(tools, builtinToOpenAI(x))
			}
		}
		payload["tools"] = tools
	}
	if tc := openAIToolChoice(cfg.ToolChoice); tc != nil {
		payload["tool_choice"] = tc
	}
	if cfg.ToolChoice != nil && cfg.ToolChoice.Parallel != nil {
		payload["parallel_tool_calls"] = *cfg.ToolChoice.Parallel
	}
	if len(cfg.ResponseFormat) > 0 {
		payload["text"] = responseFormatToOpenAIText(cfg.ResponseFormat)
	}
	if r := cfg.Reasoning; r != nil && !r.IsOff() {
		reasoning := jmap{"effort": mappedEffort(r.Effort)}
		if r.Summary != nil {
			reasoning["summary"] = *r.Summary
		}
		payload["reasoning"] = reasoning
	}
	if cc := cfg.Cache; cc != nil && cc.Mode != "off" {
		if cc.Key != nil && *cc.Key != "" {
			payload["prompt_cache_key"] = *cc.Key
		}
		if cc.Retention != nil && *cc.Retention == "long" {
			payload["prompt_cache_retention"] = "24h"
		}
	}
	extensionsPassthrough(payload, cfg.Extensions, map[string]bool{
		"prompt_caching": true, "cache": true, "compat": true,
		"openai_compat": true, "openai_responses_compat": true,
	})
	return TransportRequest{
		Method: "POST",
		URL:    baseURL + "/responses",
		Params: map[string]string{},
		Headers: map[string]string{
			"authorization": "Bearer " + apiKey,
			"content-type":  "application/json",
		},
		Body: payload,
	}, nil
}

// ─── OpenAI Chat Completions ─────────────────────────────────────────

// ChatCompat is the resolved OpenAI Chat Completions compatibility policy.
type ChatCompat struct {
	InstructionRole string // "system" | "developer"
	MaxTokensField  string // "max_completion_tokens" | "max_tokens"
	StreamUsage     string // "include" | "omit"
	ThinkingFormat  string // "none" | "reasoning_effort" | "openrouter" | ...
	ToolResultName  string // "include" | "omit"
	StrictTools     string // "include" | "omit"
	CacheControl    string // "none" | "openai"
	Routing         jmap
}

func defaultChatCompat() ChatCompat {
	return ChatCompat{
		InstructionRole: "system",
		MaxTokensField:  "max_completion_tokens",
		StreamUsage:     "include",
		ThinkingFormat:  "reasoning_effort",
		ToolResultName:  "omit",
		StrictTools:     "omit",
		CacheControl:    "openai",
	}
}

// ChatPresetBaseURLs gives each preset's default server base URL.
var ChatPresetBaseURLs = map[string]string{
	"openai":     "https://api.openai.com/v1",
	"ollama":     "http://localhost:11434/v1",
	"groq":       "https://api.groq.com/openai/v1",
	"openrouter": "https://openrouter.ai/api/v1",
	"vllm":       "http://localhost:8000/v1",
	"sglang":     "http://localhost:30000/v1",
}

// ChatCompatPreset resolves a named OpenAI-compatible server preset
// (ollama, groq, openrouter, vllm, sglang, openai, ...).
func ChatCompatPreset(name string) (ChatCompat, error) {
	key := strings.NewReplacer("-", "_", " ", "_").Replace(strings.ToLower(name))
	c := defaultChatCompat()
	switch key {
	case "openai", "openai_chat", "chat", "chat_completions":
		return c, nil
	case "ollama", "lmstudio", "lm_studio":
		c.MaxTokensField = "max_tokens"
		c.ThinkingFormat = "none"
		c.CacheControl = "none"
		return c, nil
	case "groq", "vllm", "sglang":
		c.MaxTokensField = "max_tokens"
		c.CacheControl = "none"
		return c, nil
	case "openrouter":
		c.MaxTokensField = "max_tokens"
		c.ThinkingFormat = "openrouter"
		return c, nil
	}
	return ChatCompat{}, valueErr("unknown OpenAIChat compat preset: %q", name)
}

func chatContentParts(msg Message) (any, error) {
	var parts []Part
	for _, p := range msg.Parts {
		switch p.(type) {
		case ToolCallPart, ToolResultPart:
			continue
		}
		parts = append(parts, p)
	}
	if len(parts) == 1 {
		if t, ok := parts[0].(TextPart); ok {
			return t.Text, nil
		}
	}
	out := []any{}
	for _, p := range parts {
		switch x := p.(type) {
		case TextPart:
			out = append(out, jmap{"type": "text", "text": x.Text})
		case ImagePart:
			var url string
			if x.URL != nil {
				url = *x.URL
			} else {
				u, err := mediaDataURI(x.mediaPart)
				if err != nil {
					return nil, err
				}
				url = u
			}
			payload := jmap{"url": url}
			if x.Detail != nil && *x.Detail != "" {
				payload["detail"] = *x.Detail
			}
			out = append(out, jmap{"type": "image_url", "image_url": payload})
		case ThinkingPart:
			continue // thinking is never replayed as user content
		default:
			if text := partsToText([]Part{p}); text != "" {
				out = append(out, jmap{"type": "text", "text": text})
			}
		}
	}
	return out, nil
}

func chatBuildMessages(req Request, compat ChatCompat) ([]any, error) {
	var messages []any
	if req.System != nil {
		messages = append(messages, jmap{"role": compat.InstructionRole, "content": systemText(req.System)})
	}
	for _, msg := range req.Messages {
		if msg.Role == "tool" {
			for _, p := range msg.Parts {
				if tr, ok := p.(ToolResultPart); ok {
					item := jmap{"role": "tool", "tool_call_id": tr.ID, "content": toolResultOutput(tr)}
					if compat.ToolResultName == "include" && tr.Name != nil && *tr.Name != "" {
						item["name"] = *tr.Name
					}
					messages = append(messages, item)
				}
			}
			continue
		}
		if msg.Role == "assistant" {
			var textBits []string
			var toolCalls []any
			for _, p := range msg.Parts {
				switch x := p.(type) {
				case TextPart:
					textBits = append(textBits, x.Text)
				case RefusalPart:
					if x.Text != "" {
						textBits = append(textBits, x.Text)
					}
				case ToolCallPart:
					input := x.Input
					if input == nil {
						input = jmap{}
					}
					toolCalls = append(toolCalls, jmap{
						"id":   x.ID,
						"type": "function",
						"function": jmap{
							"name":      x.Name,
							"arguments": compactJSON(input),
						},
					})
				}
			}
			item := jmap{"role": "assistant"}
			if len(textBits) > 0 {
				item["content"] = strings.Join(textBits, "\n")
			} else {
				item["content"] = nil
			}
			if len(toolCalls) > 0 {
				item["tool_calls"] = toolCalls
			}
			messages = append(messages, item)
			continue
		}
		role := msg.Role
		if role == "developer" {
			role = compat.InstructionRole
		}
		content, err := chatContentParts(msg)
		if err != nil {
			return nil, err
		}
		if list, ok := content.([]any); ok && len(list) == 0 {
			continue
		}
		messages = append(messages, jmap{"role": role, "content": content})
	}
	return messages, nil
}

func chatToolChoice(tc *ToolChoice) any {
	if tc == nil {
		return nil
	}
	if tc.Mode == "none" {
		return "none"
	}
	if len(tc.Allowed) == 1 {
		return jmap{"type": "function", "function": jmap{"name": tc.Allowed[0]}}
	}
	if tc.Mode == "required" {
		return "required"
	}
	return "auto"
}

func responseFormatToChat(fc jmap) jmap {
	if fc["type"] == "json_object" {
		return jmap{"type": "json_object"}
	}
	if _, ok := fc["json_schema"].(jmap); ok {
		return copyMap(fc)
	}
	if fc["type"] == "json_schema" {
		inner := jmap{}
		for k, v := range fc {
			if k != "type" {
				inner[k] = v
			}
		}
		if _, ok := inner["name"]; !ok {
			inner["name"] = "response"
		}
		return jmap{"type": "json_schema", "json_schema": inner}
	}
	schema, ok := fc["schema"].(jmap)
	if !ok {
		schema = fc
	}
	name := "response"
	if n, ok := fc["name"].(string); ok && n != "" {
		name = n
	}
	return jmap{"type": "json_schema", "json_schema": jmap{"name": name, "schema": schema}}
}

func buildOpenAIChat(req Request, stream bool, apiKey, baseURL string, compat ChatCompat) (TransportRequest, error) {
	cfg := req.Config
	if cfg == nil {
		cfg = &Config{}
	}
	messages, err := chatBuildMessages(req, compat)
	if err != nil {
		return TransportRequest{}, err
	}
	payload := jmap{"model": req.Model, "messages": messages}
	if stream {
		payload["stream"] = true
		if compat.StreamUsage == "include" {
			payload["stream_options"] = jmap{"include_usage": true}
		}
	}
	if cfg.MaxTokens != nil {
		payload[compat.MaxTokensField] = jsonInt(*cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		payload["temperature"] = jsonFloat(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		payload["top_p"] = jsonFloat(*cfg.TopP)
	}
	if len(cfg.Stop) > 0 {
		payload["stop"] = anyList(cfg.Stop)
	}
	if len(req.Tools) > 0 {
		var tools []any
		for _, t := range req.Tools {
			if x, ok := t.(FunctionTool); ok {
				fn := jmap{
					"name":        x.Name,
					"description": toolDescription(x),
					"parameters":  toolParameters(x),
				}
				if compat.StrictTools == "include" {
					fn["strict"] = false
				}
				tools = append(tools, jmap{"type": "function", "function": fn})
			}
		}
		if len(tools) > 0 {
			payload["tools"] = tools
		}
	}
	if tc := chatToolChoice(cfg.ToolChoice); tc != nil {
		payload["tool_choice"] = tc
	}
	if cfg.ToolChoice != nil && cfg.ToolChoice.Parallel != nil {
		payload["parallel_tool_calls"] = *cfg.ToolChoice.Parallel
	}
	if len(cfg.ResponseFormat) > 0 {
		payload["response_format"] = responseFormatToChat(cfg.ResponseFormat)
	}
	if r := cfg.Reasoning; r != nil {
		effort := mappedEffort(r.Effort)
		if !r.IsOff() {
			switch compat.ThinkingFormat {
			case "reasoning_effort":
				payload["reasoning_effort"] = effort
			case "openrouter":
				payload["reasoning"] = jmap{"effort": effort}
			}
		}
	}
	if cc := cfg.Cache; cc != nil && cc.Mode != "off" && compat.CacheControl == "openai" {
		if cc.Key != nil && *cc.Key != "" {
			payload["prompt_cache_key"] = *cc.Key
		}
		if cc.Retention != nil && *cc.Retention == "long" {
			payload["prompt_cache_retention"] = "24h"
		}
	}
	if compat.Routing != nil {
		payload["provider"] = compat.Routing
	}
	extensionsPassthrough(payload, cfg.Extensions, map[string]bool{
		"prompt_caching": true, "cache": true, "compat": true,
		"openai_compat": true, "openai_chat_compat": true,
	})
	return TransportRequest{
		Method: "POST",
		URL:    baseURL + "/chat/completions",
		Params: map[string]string{},
		Headers: map[string]string{
			"authorization": "Bearer " + apiKey,
			"content-type":  "application/json",
		},
		Body: payload,
	}, nil
}

// ─── Anthropic ───────────────────────────────────────────────────────

var anthropicBuiltinMap = map[string]string{
	"web_search":     "web_search_20250305",
	"code_execution": "code_execution_20250522",
}

const (
	defaultAnthropicVisibleTokens  int64 = 1024
	defaultAnthropicThinkingBudget int64 = 1024
)

func builtinToAnthropic(t BuiltinTool) jmap {
	typ, ok := anthropicBuiltinMap[t.Name]
	if !ok {
		typ = t.Name
	}
	out := jmap{"type": typ, "name": t.Name}
	for k, v := range t.Config {
		out[k] = v
	}
	return out
}

func anthropicSource(m mediaPart) (jmap, error) {
	switch {
	case m.URL != nil:
		return jmap{"type": "url", "url": *m.URL}, nil
	case m.FileID != nil:
		return jmap{"type": "file", "file_id": *m.FileID}, nil
	case m.Data != nil:
		return jmap{"type": "base64", "media_type": m.MediaType, "data": *m.Data}, nil
	}
	return nil, valueErr("media part has no usable source")
}

func anthropicToolResultContent(p Part) (jmap, error) {
	switch x := p.(type) {
	case TextPart:
		return jmap{"type": "text", "text": x.Text}, nil
	case ImagePart:
		src, err := anthropicSource(x.mediaPart)
		if err != nil {
			return nil, err
		}
		return jmap{"type": "image", "source": src}, nil
	case DocumentPart:
		src, err := anthropicSource(x.mediaPart)
		if err != nil {
			return nil, err
		}
		return jmap{"type": "document", "source": src}, nil
	}
	return jmap{"type": "text", "text": partText(p)}, nil
}

// partText extracts a part's text-ish field, or "".
func partText(p Part) string {
	switch x := p.(type) {
	case TextPart:
		return x.Text
	case ThinkingPart:
		return x.Text
	case RefusalPart:
		return x.Text
	case CitationPart:
		if x.Text != nil {
			return *x.Text
		}
	}
	return ""
}

func anthropicPart(p Part) (jmap, error) {
	switch x := p.(type) {
	case TextPart:
		return jmap{"type": "text", "text": x.Text}, nil
	case ImagePart:
		src, err := anthropicSource(x.mediaPart)
		if err != nil {
			return nil, err
		}
		return jmap{"type": "image", "source": src}, nil
	case DocumentPart:
		src, err := anthropicSource(x.mediaPart)
		if err != nil {
			return nil, err
		}
		return jmap{"type": "document", "source": src}, nil
	case ToolCallPart:
		input := x.Input
		if input == nil {
			input = jmap{}
		}
		return jmap{"type": "tool_use", "id": x.ID, "name": x.Name, "input": input}, nil
	case ToolResultPart:
		var blocks []jmap
		for _, c := range x.Content {
			b, err := anthropicToolResultContent(c)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, b)
		}
		out := jmap{"type": "tool_result", "tool_use_id": x.ID}
		if len(blocks) > 0 {
			if len(blocks) == 1 && blocks[0]["type"] == "text" {
				out["content"] = blocks[0]["text"]
			} else {
				list := make([]any, len(blocks))
				for i, b := range blocks {
					list[i] = b
				}
				out["content"] = list
			}
		}
		if x.IsError {
			out["is_error"] = true
		}
		return out, nil
	case ThinkingPart:
		if redacted := continuationData(x, "anthropic", "redacted_thinking"); redacted != nil {
			out := jmap{"type": "redacted_thinking"}
			for k, v := range redacted {
				out[k] = v
			}
			return out, nil
		}
		if sig := continuationData(x, "anthropic", "thinking_signature"); sig != nil {
			if s, ok := sig["signature"].(string); ok && s != "" {
				return jmap{"type": "thinking", "thinking": x.Text, "signature": s}, nil
			}
		}
		return jmap{"type": "text", "text": x.Text}, nil
	}
	return jmap{"type": "text", "text": partText(p)}, nil
}

func anthropicMessage(msg Message) (jmap, error) {
	role := "user"
	if msg.Role == "assistant" {
		role = "assistant"
	}
	if msg.Role == "developer" {
		return jmap{"role": "user", "content": []any{
			jmap{"type": "text", "text": "[developer]\n" + partsToText(msg.Parts)},
		}}, nil
	}
	content := make([]any, len(msg.Parts))
	for i, p := range msg.Parts {
		block, err := anthropicPart(p)
		if err != nil {
			return nil, err
		}
		content[i] = block
	}
	return jmap{"role": role, "content": content}, nil
}

func anthropicToolChoice(tc *ToolChoice) jmap {
	if tc == nil {
		return nil
	}
	payload := jmap{}
	switch {
	case tc.Mode == "none":
		payload["type"] = "none"
	case len(tc.Allowed) == 1:
		payload["type"] = "tool"
		payload["name"] = tc.Allowed[0]
	case len(tc.Allowed) > 1:
		if tc.Mode == "required" {
			payload["type"] = "any"
		} else {
			payload["type"] = "auto"
		}
	case tc.Mode == "required":
		payload["type"] = "any"
	default:
		payload["type"] = "auto"
	}
	if tc.Parallel != nil && !*tc.Parallel && payload["type"] != "none" {
		payload["disable_parallel_tool_use"] = true
	}
	return payload
}

func responseFormatToAnthropicOutputConfig(fc jmap) jmap {
	if oc, ok := fc["output_config"].(jmap); ok {
		return copyMap(oc)
	}
	if _, ok := fc["format"].(jmap); ok {
		return copyMap(fc)
	}
	switch fc["type"] {
	case "json_schema":
		schema, ok := fc["schema"].(jmap)
		if !ok {
			schema = jmap{}
		}
		return jmap{"format": jmap{"type": "json_schema", "schema": schema}}
	case "json_object":
		return jmap{"format": jmap{"type": "json_schema", "schema": jmap{"type": "object"}}}
	}
	schema, ok := fc["schema"].(jmap)
	if !ok {
		schema = fc
	}
	return jmap{"format": jmap{"type": "json_schema", "schema": schema}}
}

func anthropicThinkingBudget(cfg *Config) *int64 {
	r := cfg.Reasoning
	if r == nil || r.IsOff() {
		return nil
	}
	budget := defaultAnthropicThinkingBudget
	if r.ThinkingBudget != nil && *r.ThinkingBudget != 0 {
		budget = *r.ThinkingBudget
	}
	return &budget
}

func anthropicMaxTokens(cfg *Config, thinkingBudget *int64) (int64, error) {
	visible := defaultAnthropicVisibleTokens
	if cfg.MaxTokens != nil && *cfg.MaxTokens != 0 {
		visible = *cfg.MaxTokens
	}
	if thinkingBudget == nil {
		return visible, nil
	}
	if r := cfg.Reasoning; r != nil && r.TotalBudget != nil {
		if *r.TotalBudget <= *thinkingBudget {
			return 0, valueErr("Anthropic requires Reasoning.total_budget to be greater than Reasoning.thinking_budget because max_tokens includes thinking tokens")
		}
		return *r.TotalBudget, nil
	}
	return *thinkingBudget + visible, nil
}

func buildAnthropic(req Request, stream bool, apiKey, baseURL string) (TransportRequest, error) {
	cfg := req.Config
	if cfg == nil {
		cfg = &Config{}
	}
	useCache := cfg.Cache != nil && cfg.Cache.Mode != "off"
	longCache := cfg.Cache != nil && cfg.Cache.Retention != nil && *cfg.Cache.Retention == "long"

	messages := make([]any, len(req.Messages))
	for i, m := range req.Messages {
		msg, err := anthropicMessage(m)
		if err != nil {
			return TransportRequest{}, err
		}
		messages[i] = msg
	}
	if useCache && cfg.Cache.PrefixUntilIndex != nil {
		idx := *cfg.Cache.PrefixUntilIndex
		if idx > int64(len(messages)-1) {
			idx = int64(len(messages) - 1)
		}
		if idx >= 0 {
			if content, ok := messages[idx].(jmap)["content"].([]any); ok && len(content) > 0 {
				if last, ok := content[len(content)-1].(jmap); ok {
					if _, present := last["cache_control"]; !present {
						last["cache_control"] = jmap{"type": "ephemeral"}
					}
				}
			}
		}
	}

	thinkingBudget := anthropicThinkingBudget(cfg)
	maxTokens, err := anthropicMaxTokens(cfg, thinkingBudget)
	if err != nil {
		return TransportRequest{}, err
	}
	payload := jmap{
		"model":      req.Model,
		"messages":   messages,
		"stream":     stream,
		"max_tokens": jsonInt(maxTokens),
	}
	if req.System != nil {
		text := systemText(req.System)
		if useCache {
			marker := jmap{"type": "ephemeral"}
			if longCache {
				marker["ttl"] = "1h"
			}
			payload["system"] = []any{jmap{"type": "text", "text": text, "cache_control": marker}}
		} else {
			payload["system"] = text
		}
	}
	if cfg.Temperature != nil {
		payload["temperature"] = jsonFloat(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		payload["top_p"] = jsonFloat(*cfg.TopP)
	}
	if cfg.TopK != nil {
		payload["top_k"] = jsonInt(*cfg.TopK)
	}
	if len(cfg.Stop) > 0 {
		payload["stop_sequences"] = anyList(cfg.Stop)
	}
	hasCodeExecution := false
	if len(req.Tools) > 0 {
		var tools []any
		for _, t := range req.Tools {
			switch x := t.(type) {
			case FunctionTool:
				tools = append(tools, jmap{
					"name":         x.Name,
					"description":  toolDescription(x),
					"input_schema": toolParameters(x),
				})
			case BuiltinTool:
				if x.Name == "code_execution" {
					hasCodeExecution = true
				}
				tools = append(tools, builtinToAnthropic(x))
			}
		}
		payload["tools"] = tools
	}
	if tc := anthropicToolChoice(cfg.ToolChoice); tc != nil {
		payload["tool_choice"] = tc
	}
	if thinkingBudget != nil {
		payload["thinking"] = jmap{"type": "enabled", "budget_tokens": jsonInt(*thinkingBudget)}
	}
	if len(cfg.ResponseFormat) > 0 {
		payload["output_config"] = responseFormatToAnthropicOutputConfig(cfg.ResponseFormat)
	}
	extensionsPassthrough(payload, cfg.Extensions, map[string]bool{"prompt_caching": true})

	headers := map[string]string{
		"x-api-key":         apiKey,
		"anthropic-version": anthropicAPIVersion,
		"content-type":      "application/json",
	}
	if hasCodeExecution {
		headers["anthropic-beta"] = "code-execution-2025-05-22"
	}
	return TransportRequest{
		Method:  "POST",
		URL:     baseURL + "/messages",
		Params:  map[string]string{},
		Headers: headers,
		Body:    payload,
	}, nil
}

// ─── Gemini ──────────────────────────────────────────────────────────

var geminiBuiltinMap = map[string]string{
	"web_search":     "googleSearch",
	"code_execution": "codeExecution",
}

func builtinToGemini(t BuiltinTool) jmap {
	key, ok := geminiBuiltinMap[t.Name]
	if !ok {
		key = t.Name
	}
	cfg := t.Config
	if cfg == nil {
		cfg = jmap{}
	}
	return jmap{key: cfg}
}

// geminiNumber renders Gemini's proto3-JSON wire form for float knobs:
// integral values are sent in integer form.
func geminiNumber(f float64) json.Number {
	if f == float64(int64(f)) {
		return jsonInt(int64(f))
	}
	return jsonFloat(f)
}

func geminiModelPath(model string) string {
	if strings.HasPrefix(model, "models/") {
		return model
	}
	return "models/" + model
}

func containsKey(v any, key string) bool {
	switch x := v.(type) {
	case jmap:
		if _, ok := x[key]; ok {
			return true
		}
		for _, vv := range x {
			if containsKey(vv, key) {
				return true
			}
		}
	case []any:
		for _, vv := range x {
			if containsKey(vv, key) {
				return true
			}
		}
	}
	return false
}

func geminiSchemaField(schema jmap) string {
	if containsKey(schema, "additionalProperties") {
		return "responseJsonSchema"
	}
	return "responseSchema"
}

func responseFormatToGeminiConfig(fc jmap) jmap {
	if gc, ok := fc["generationConfig"].(jmap); ok {
		return copyMap(gc)
	}
	out := jmap{}
	for _, k := range []string{"responseMimeType", "response_mime_type"} {
		if v, ok := fc[k].(string); ok && v != "" {
			out["responseMimeType"] = v
			break
		}
	}
	for _, k := range []string{"responseSchema", "response_schema"} {
		if v, ok := fc[k].(jmap); ok {
			out["responseSchema"] = v
			break
		}
	}
	for _, k := range []string{"responseJsonSchema", "response_json_schema"} {
		if v, ok := fc[k].(jmap); ok {
			out["responseJsonSchema"] = v
			break
		}
	}
	setSchema := func(schema jmap) {
		field := geminiSchemaField(schema)
		out[field] = schema
		if field == "responseJsonSchema" {
			delete(out, "responseSchema")
		} else {
			delete(out, "responseJsonSchema")
		}
		if _, ok := out["responseMimeType"]; !ok {
			out["responseMimeType"] = "application/json"
		}
	}
	switch fc["type"] {
	case "json_object":
		if _, ok := out["responseMimeType"]; !ok {
			out["responseMimeType"] = "application/json"
		}
		return out
	case "json_schema":
		if schema, ok := fc["schema"].(jmap); ok {
			setSchema(schema)
		} else if _, ok := out["responseMimeType"]; !ok {
			out["responseMimeType"] = "application/json"
		}
		return out
	}
	if schema, ok := fc["schema"].(jmap); ok {
		setSchema(schema)
		return out
	}
	_, hasType := fc["type"]
	_, hasProps := fc["properties"]
	_, hasItems := fc["items"]
	if hasType || hasProps || hasItems {
		setSchema(copyMap(fc))
	}
	if len(out) == 0 {
		return copyMap(fc)
	}
	return out
}

func geminiPart(p Part) (jmap, error) {
	switch x := p.(type) {
	case TextPart:
		return jmap{"text": x.Text}, nil
	case ToolCallPart:
		input := x.Input
		if input == nil {
			input = jmap{}
		}
		fc := jmap{"name": x.Name, "args": input}
		if x.ID != "" {
			fc["id"] = x.ID
		}
		out := jmap{"functionCall": fc}
		if thought := continuationData(x, "gemini", "thought_signature"); thought != nil {
			if v, ok := thought["value"].(string); ok && v != "" {
				out["thoughtSignature"] = v
			}
		}
		return out, nil
	case ToolResultPart:
		name := "tool"
		if x.Name != nil && *x.Name != "" {
			name = *x.Name
		}
		fr := jmap{"name": name, "response": jmap{"result": partsToText(x.Content)}}
		if x.ID != "" {
			fr["id"] = x.ID
		}
		return jmap{"functionResponse": fr}, nil
	case ThinkingPart:
		out := jmap{"text": x.Text}
		if thought := continuationData(x, "gemini", "thought_signature"); thought != nil {
			if v, ok := thought["value"].(string); ok && v != "" {
				out["thought"] = true
				out["thoughtSignature"] = v
			}
		}
		return out, nil
	}
	if m, ok := mediaOf(p); ok {
		mime := m.MediaType
		if mime == "" {
			mime = "application/octet-stream"
		}
		switch {
		case m.URL != nil:
			return jmap{"fileData": jmap{"mimeType": mime, "fileUri": *m.URL}}, nil
		case m.FileID != nil:
			return jmap{"fileData": jmap{"mimeType": mime, "fileUri": *m.FileID}}, nil
		case m.Data != nil:
			return jmap{"inlineData": jmap{"mimeType": mime, "data": *m.Data}}, nil
		}
	}
	return jmap{"text": partText(p)}, nil
}

func geminiMessage(msg Message) (jmap, error) {
	if msg.Role == "developer" {
		return jmap{"role": "user", "parts": []any{jmap{"text": "[developer]\n" + partsToText(msg.Parts)}}}, nil
	}
	role := "user"
	if msg.Role == "assistant" {
		role = "model"
	}
	parts := make([]any, len(msg.Parts))
	for i, p := range msg.Parts {
		gp, err := geminiPart(p)
		if err != nil {
			return nil, err
		}
		parts[i] = gp
	}
	return jmap{"role": role, "parts": parts}, nil
}

func buildGemini(req Request, stream bool, apiKey, baseURL string) (TransportRequest, error) {
	cfg := req.Config
	if cfg == nil {
		cfg = &Config{}
	}
	contents := make([]any, len(req.Messages))
	for i, m := range req.Messages {
		msg, err := geminiMessage(m)
		if err != nil {
			return TransportRequest{}, err
		}
		contents[i] = msg
	}
	payload := jmap{"contents": contents}
	if req.System != nil {
		payload["systemInstruction"] = jmap{"parts": []any{jmap{"text": systemText(req.System)}}}
	}
	generationConfig := jmap{}
	if cfg.Temperature != nil {
		generationConfig["temperature"] = geminiNumber(*cfg.Temperature)
	}
	if cfg.MaxTokens != nil {
		generationConfig["maxOutputTokens"] = jsonInt(*cfg.MaxTokens)
	}
	if cfg.TopP != nil {
		generationConfig["topP"] = geminiNumber(*cfg.TopP)
	}
	if cfg.TopK != nil {
		generationConfig["topK"] = jsonInt(*cfg.TopK)
	}
	if len(cfg.Stop) > 0 {
		generationConfig["stopSequences"] = anyList(cfg.Stop)
	}
	if len(cfg.ResponseFormat) > 0 {
		for k, v := range responseFormatToGeminiConfig(cfg.ResponseFormat) {
			generationConfig[k] = v
		}
	}
	if r := cfg.Reasoning; r != nil {
		if r.IsOff() {
			generationConfig["thinkingConfig"] = jmap{"thinkingBudget": jsonInt(0)}
		} else {
			thinking := jmap{"includeThoughts": true}
			if r.ThinkingBudget != nil {
				thinking["thinkingBudget"] = jsonInt(*r.ThinkingBudget)
			}
			generationConfig["thinkingConfig"] = thinking
		}
	}
	if len(generationConfig) > 0 {
		payload["generationConfig"] = generationConfig
	}
	if len(req.Tools) > 0 {
		var declarations []any
		var tools []any
		for _, t := range req.Tools {
			if x, ok := t.(FunctionTool); ok {
				declarations = append(declarations, jmap{
					"name":        x.Name,
					"description": toolDescription(x),
					"parameters":  toolParameters(x),
				})
			}
		}
		if len(declarations) > 0 {
			tools = append(tools, jmap{"functionDeclarations": declarations})
		}
		for _, t := range req.Tools {
			if x, ok := t.(BuiltinTool); ok {
				tools = append(tools, builtinToGemini(x))
			}
		}
		payload["tools"] = tools
	}
	if tc := cfg.ToolChoice; tc != nil {
		mode := map[string]string{"none": "NONE", "required": "ANY", "auto": "AUTO"}[tc.Mode]
		fcc := jmap{"mode": mode}
		if len(tc.Allowed) > 0 {
			fcc["allowedFunctionNames"] = anyList(tc.Allowed)
		}
		payload["toolConfig"] = jmap{"functionCallingConfig": fcc}
	}
	if output, ok := cfg.Extensions["output"].(string); ok {
		switch output {
		case "image":
			ensureObj(payload, "generationConfig")["responseModalities"] = []any{"IMAGE"}
		case "audio":
			ensureObj(payload, "generationConfig")["responseModalities"] = []any{"AUDIO"}
		}
	}
	extensionsPassthrough(payload, cfg.Extensions, map[string]bool{"prompt_caching": true, "output": true})

	endpoint := "generateContent"
	params := map[string]string{}
	if stream {
		endpoint = "streamGenerateContent"
		params["alt"] = "sse"
	}
	return TransportRequest{
		Method: "POST",
		URL:    baseURL + "/" + geminiModelPath(req.Model) + ":" + endpoint,
		Params: params,
		Headers: map[string]string{
			"x-goog-api-key": apiKey,
			"content-type":   "application/json",
		},
		Body: payload,
	}, nil
}

func ensureObj(m jmap, key string) jmap {
	if o, ok := m[key].(jmap); ok {
		return o
	}
	o := jmap{}
	m[key] = o
	return o
}

// BuildRequestMap is the vet-protocol entry: canonical request JSON in,
// protocol-shaped transport request out.
func BuildRequestMap(provider string, canonical jmap, stream bool, apiKey, baseURL string) (jmap, error) {
	req, err := requestFromMap(canonical)
	if err != nil {
		return nil, err
	}
	tr, err := BuildRequest(provider, req, stream, apiKey, baseURL)
	if err != nil {
		return nil, err
	}
	var body any
	if tr.Body != nil {
		body = tr.Body
	}
	return jmap{
		"method":  tr.Method,
		"url":     tr.URL,
		"params":  tr.Params,
		"headers": tr.Headers,
		"body":    body,
	}, nil
}
