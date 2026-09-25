package lm15

import (
	"mime"
	"path"
	"strconv"
	"strings"
)

// MAP-12: a Chat Completions request body → canonical Request, under ONE
// preset's spellings. Every key has one verdict (map / extensions / refuse /
// call-mode / default); malformed input is a native error.

var ingestExtensionsKeys = map[string]bool{
	"logit_bias": true, "metadata": true, "verbosity": true, "moderation": true, "provider": true,
	// prediction: a latency hint (predicted outputs); harmless verbatim on
	// OpenAI, dropped-with-note elsewhere (decision 2026-09-14 §4.9).
	"prediction": true,
}

var ingestRefusedKeys = map[string]string{
	"n":                  "lm15 reads one choice per response; n>1 would silently lose choices — fan out in the caller",
	"audio":              "audio output parameters have no canonical slot on the chat surface",
	"modalities":         "output modality selection has no canonical slot on the chat surface",
	"web_search_options": "a server-executed search the chat dialect cannot map to parts (MAP-1); the Responses dialect carries web_search as a BuiltinTool",
}

var ingestCallModeKeys = map[string]bool{"stream": true, "stream_options": true}

var ingestConfigKeys = map[string]bool{
	"model": true, "messages": true, "tools": true, "tool_choice": true, "parallel_tool_calls": true,
	// functions / function_call: the deprecated function-calling shape,
	// translated to tools / tool_choice (MAP-13: a pure spelling change).
	"functions": true, "function_call": true,
	"max_completion_tokens": true, "max_tokens": true, "temperature": true, "top_p": true, "top_k": true, "stop": true,
	"seed": true, "frequency_penalty": true, "presence_penalty": true,
	"logprobs": true, "top_logprobs": true, "response_format": true, "service_tier": true, "store": true,
	"user": true, "safety_identifier": true, "user_id": true,
	"reasoning_effort": true, "reasoning": true, "thinking": true, "enable_thinking": true, "chat_template_kwargs": true, "reasoning_format": true,
	"prompt_cache_key": true, "prompt_cache_retention": true, "prompt_cache_options": true,
}

var ingestGroqBuiltinInverse = map[string]string{"browser_search": "web_search", "code_interpreter": "code_execution"}
var ingestAudioMediaTypes = map[string]string{"wav": "audio/wav", "mp3": "audio/mpeg"}
var ingestClientObjectKeys = []string{"provider_specific_fields", "thinking_blocks", "images"}

func ingestUnsupported(provider, what, why string) *Error {
	return UnsupportedFeatureErrorf(provider, "%s: %s cannot be carried by a canonical Request — %s", provider, what, why)
}

func ingestStr(v any, where string) (string, error) {
	s, ok := v.(string)
	if !ok {
		return "", typeErrorf("%s must be a string, got %s", where, jsonTypeName(v))
	}
	return s, nil
}

func ingestObject(v any, where string) (JSONObject, error) {
	m, ok := asObject(v)
	if !ok {
		return nil, typeErrorf("%s must be a JSON object, got %s", where, jsonTypeName(v))
	}
	return m, nil
}

func ingestOnlyKeys(provider string, obj JSONObject, allowed []string, where string) error {
	allowedSet := map[string]bool{}
	for _, k := range allowed {
		allowedSet[k] = true
	}
	var extra []string
	for k := range obj.All() {
		if !allowedSet[k] {
			extra = append(extra, k)
		}
	}
	if len(extra) > 0 {
		extra = sortStrings(extra)
		return ingestUnsupported(provider, where+" key "+strconv.Quote(extra[0]), "no canonical slot for it")
	}
	return nil
}

func ingestDataURI(value, where string) (string, string, error) {
	if !strings.HasPrefix(value, "data:") {
		return "", "", valueErrorf("%s must be a base64 data URI", where)
	}
	head, payload, ok := strings.Cut(value[5:], ",")
	if !ok || !strings.HasSuffix(head, ";base64") || payload == "" {
		return "", "", valueErrorf("%s must be a base64 data URI (data:<media-type>;base64,<payload>)", where)
	}
	mediaType := strings.TrimSuffix(head, ";base64")
	if mediaType == "" {
		return "", "", valueErrorf("%s data URI has no media type", where)
	}
	return mediaType, payload, nil
}

func ingestImageBlock(provider string, block JSONObject, where string) (ImagePart, error) {
	if err := ingestOnlyKeys(provider, block, []string{"type", "image_url", "prompt_cache_breakpoint"}, where); err != nil {
		return ImagePart{}, err
	}
	spec, err := ingestObject(block.Get("image_url"), where+".image_url")
	if err != nil {
		return ImagePart{}, err
	}
	if err := ingestOnlyKeys(provider, spec, []string{"url", "detail"}, where+".image_url"); err != nil {
		return ImagePart{}, err
	}
	url, err := ingestStr(spec.Get("url"), where+".image_url.url")
	if err != nil {
		return ImagePart{}, err
	}
	detail := stringOnly(spec.Get("detail"))
	if strings.HasPrefix(url, "data:") {
		mediaType, payload, err := ingestDataURI(url, where+".image_url.url")
		if err != nil {
			return ImagePart{}, err
		}
		return ImagePart{Media: Media{MediaType: mediaType, Data: payload}, Detail: detail}, nil
	}
	guessed := ""
	if ext := path.Ext(strings.SplitN(strings.SplitN(url, "?", 2)[0], "#", 2)[0]); ext != "" {
		guessed, _, _ = strings.Cut(mime.TypeByExtension(strings.ToLower(ext)), ";")
	}
	img := Image(WithURL(url))
	if strings.HasPrefix(guessed, "image/") {
		img.MediaType = guessed
	}
	img.Detail = detail
	return img, nil
}

func ingestTextBlock(provider string, block JSONObject, where string) (TextPart, error) {
	if err := ingestOnlyKeys(provider, block, []string{"type", "text", "prompt_cache_breakpoint"}, where); err != nil {
		return TextPart{}, err
	}
	text, err := ingestStr(block.Get("text"), where+".text")
	if err != nil {
		return TextPart{}, err
	}
	return TextPart{Text: text}, nil
}

func ingestHasBreakpoint(block JSONObject, where string) (bool, error) {
	mark, present := block.Lookup("prompt_cache_breakpoint")
	if !present || mark == nil {
		return false, nil
	}
	obj, err := ingestObject(mark, where+".prompt_cache_breakpoint")
	if err != nil {
		return false, err
	}
	if len(obj) != 1 || obj.Get("mode") != "explicit" {
		return false, valueErrorf("%s.prompt_cache_breakpoint must be {\"mode\": \"explicit\"}", where)
	}
	if block.Get("type") != "text" {
		return false, valueErrorf("%s: a prompt_cache_breakpoint rides on a text block, not %q", where, wireStr(block.Get("type")))
	}
	return true, nil
}

func ingestContentBlocks(provider string, content any, role, where string) ([]Part, bool, error) {
	if s, ok := content.(string); ok {
		return []Part{TextPart{Text: s}}, false, nil
	}
	list, ok := content.([]any)
	if !ok {
		return nil, false, typeErrorf("%s.content must be a string or an array of content parts", where)
	}
	var parts []Part
	breakpointAtEnd := false
	for index, raw := range list {
		blockWhere := where + ".content[" + strconv.Itoa(index) + "]"
		block, err := ingestObject(raw, blockWhere)
		if err != nil {
			return nil, false, err
		}
		kind := wireStr(block.Get("type"))
		marked, err := ingestHasBreakpoint(block, blockWhere)
		if err != nil {
			return nil, false, err
		}
		if marked && index != len(list)-1 {
			return nil, false, valueErrorf("%s: a prompt_cache_breakpoint marks the end of a message; it must be on the last block", blockWhere)
		}
		breakpointAtEnd = breakpointAtEnd || marked
		switch {
		case kind == "text":
			p, err := ingestTextBlock(provider, block, blockWhere)
			if err != nil {
				return nil, false, err
			}
			parts = append(parts, p)
		case kind == "image_url" && (role == "user" || role == "tool"):
			p, err := ingestImageBlock(provider, block, blockWhere)
			if err != nil {
				return nil, false, err
			}
			parts = append(parts, p)
		case kind == "input_audio" && role == "user":
			if err := ingestOnlyKeys(provider, block, []string{"type", "input_audio", "prompt_cache_breakpoint"}, blockWhere); err != nil {
				return nil, false, err
			}
			spec, err := ingestObject(block.Get("input_audio"), blockWhere+".input_audio")
			if err != nil {
				return nil, false, err
			}
			if err := ingestOnlyKeys(provider, spec, []string{"data", "format"}, blockWhere+".input_audio"); err != nil {
				return nil, false, err
			}
			format, err := ingestStr(spec.Get("format"), blockWhere+".input_audio.format")
			if err != nil {
				return nil, false, err
			}
			mediaType, ok := ingestAudioMediaTypes[format]
			if !ok {
				return nil, false, valueErrorf("%s.input_audio.format must be one of [mp3 wav]", blockWhere)
			}
			data, err := ingestStr(spec.Get("data"), blockWhere+".input_audio.data")
			if err != nil {
				return nil, false, err
			}
			parts = append(parts, AudioPart{Media: Media{MediaType: mediaType, Data: data}})
		case kind == "file" && role == "user":
			if err := ingestOnlyKeys(provider, block, []string{"type", "file", "prompt_cache_breakpoint"}, blockWhere); err != nil {
				return nil, false, err
			}
			spec, err := ingestObject(block.Get("file"), blockWhere+".file")
			if err != nil {
				return nil, false, err
			}
			if err := ingestOnlyKeys(provider, spec, []string{"file_data", "file_id", "filename"}, blockWhere+".file"); err != nil {
				return nil, false, err
			}
			if spec.Get("filename") != nil {
				return nil, false, ingestUnsupported(provider, blockWhere+".file.filename", "DocumentPart has no filename slot")
			}
			switch {
			case spec.Get("file_id") != nil && spec.Get("file_data") == nil:
				id, err := ingestStr(spec.Get("file_id"), blockWhere+".file.file_id")
				if err != nil {
					return nil, false, err
				}
				parts = append(parts, Document(WithFileID(id)))
			case spec.Get("file_data") != nil && spec.Get("file_id") == nil:
				raw, err := ingestStr(spec.Get("file_data"), blockWhere+".file.file_data")
				if err != nil {
					return nil, false, err
				}
				mediaType, payload, err := ingestDataURI(raw, blockWhere+".file.file_data")
				if err != nil {
					return nil, false, err
				}
				parts = append(parts, DocumentPart{Media: Media{MediaType: mediaType, Data: payload}})
			default:
				return nil, false, valueErrorf("%s.file needs exactly one of file_data / file_id", blockWhere)
			}
		case kind == "refusal" && role == "assistant":
			if err := ingestOnlyKeys(provider, block, []string{"type", "refusal"}, blockWhere); err != nil {
				return nil, false, err
			}
			text, err := ingestStr(block.Get("refusal"), blockWhere+".refusal")
			if err != nil {
				return nil, false, err
			}
			parts = append(parts, RefusalPart{Text: text})
		default:
			return nil, false, ingestUnsupported(provider, blockWhere+" of type "+strconv.Quote(kind)+" in a "+role+" message",
				"no canonical part for that block on this wire (a part is not a knob: there is no extensions door for content)")
		}
	}
	return parts, breakpointAtEnd, nil
}

func ingestEmptyContainer(v any) bool {
	switch x := jsonView(v).(type) {
	case []any:
		return len(x) == 0
	case JSONObject:
		return len(x) == 0
	}
	return false
}

func ingestAllEmpty(v any) bool {
	m, ok := asObject(v)
	if !ok {
		return false
	}
	for _, val := range m.All() {
		switch x := jsonView(val).(type) {
		case nil:
		case []any:
			if len(x) > 0 {
				return false
			}
		case JSONObject:
			if len(x) > 0 {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func ingestAnnotations(provider string, raw any, contentText *string, where string) ([]Part, error) {
	list, ok := raw.([]any)
	if !ok {
		return nil, typeErrorf("%s.annotations must be an array", where)
	}
	var out []Part
	for index, item := range list {
		entryWhere := where + ".annotations[" + strconv.Itoa(index) + "]"
		entry, err := ingestObject(item, entryWhere)
		if err != nil {
			return nil, err
		}
		if entry.Get("type") != "url_citation" {
			return nil, ingestUnsupported(provider, entryWhere+" of type "+strconv.Quote(wireStr(entry.Get("type"))), "only url_citation annotations have a canonical part (CitationPart)")
		}
		if err := ingestOnlyKeys(provider, entry, []string{"type", "url_citation"}, entryWhere); err != nil {
			return nil, err
		}
		spec, err := ingestObject(entry.Get("url_citation"), entryWhere+".url_citation")
		if err != nil {
			return nil, err
		}
		if err := ingestOnlyKeys(provider, spec, []string{"url", "title", "start_index", "end_index"}, entryWhere+".url_citation"); err != nil {
			return nil, err
		}
		text := ""
		start, end := wireIntPtr(spec.Get("start_index")), wireIntPtr(spec.Get("end_index"))
		if _, sb := spec.Get("start_index").(bool); !sb {
			if _, eb := spec.Get("end_index").(bool); !eb && contentText != nil && start != nil && end != nil && 0 <= *start && *start <= *end && *end <= len(*contentText) {
				text = (*contentText)[*start:*end]
			}
		}
		url, err := ingestStr(spec.Get("url"), entryWhere+".url_citation.url")
		if err != nil {
			return nil, err
		}
		title := ""
		if spec.Get("title") != nil {
			if title, err = ingestStr(spec.Get("title"), entryWhere+".url_citation.title"); err != nil {
				return nil, err
			}
		}
		out = append(out, CitationPart{URL: url, Title: title, Text: text})
	}
	return out, nil
}

func ingestToolCalls(provider string, calls any, where string) ([]Part, error) {
	list, ok := calls.([]any)
	if !ok {
		return nil, typeErrorf("%s.tool_calls must be an array", where)
	}
	var out []Part
	for index, raw := range list {
		callWhere := where + ".tool_calls[" + strconv.Itoa(index) + "]"
		call, err := ingestObject(raw, callWhere)
		if err != nil {
			return nil, err
		}
		kind := "function"
		if k, present := call.Lookup("type"); present {
			kind = wireStr(k)
		}
		if kind != "function" {
			return nil, ingestUnsupported(provider, callWhere+" of type "+strconv.Quote(kind), "only function tool calls have a canonical part")
		}
		if err := ingestOnlyKeys(provider, call, []string{"id", "type", "function"}, callWhere); err != nil {
			return nil, err
		}
		fn, err := ingestObject(call.Get("function"), callWhere+".function")
		if err != nil {
			return nil, err
		}
		if err := ingestOnlyKeys(provider, fn, []string{"name", "arguments"}, callWhere+".function"); err != nil {
			return nil, err
		}
		var parsed any = JSONObject{}
		if args, present := fn.Lookup("arguments"); present {
			if s, ok := args.(string); ok {
				if s != "" {
					parsed, err = DecodeJSON([]byte(s))
					if err != nil {
						return nil, valueErrorf("%s.function.arguments is not JSON: %v", callWhere, err)
					}
				}
			} else {
				parsed = args
			}
		}
		input, ok := asObject(parsed)
		if !ok {
			return nil, valueErrorf("%s.function.arguments must encode a JSON object", callWhere)
		}
		id, err := ingestStr(call.Get("id"), callWhere+".id")
		if err != nil {
			return nil, err
		}
		name, err := ingestStr(fn.Get("name"), callWhere+".function.name")
		if err != nil {
			return nil, err
		}
		out = append(out, ToolCallPart{ID: id, Name: name, Input: input})
	}
	return out, nil
}

type ingestMessages struct {
	system           *SystemPrompt
	messages         []Message
	systemBreakpoint bool
	breakpointIndex  *int
}

func ingestRows(provider string, rows any) (ingestMessages, error) {
	list, ok := rows.([]any)
	if !ok {
		return ingestMessages{}, typeErrorf("messages must be an array")
	}
	var out ingestMessages
	var pending []Part
	flush := func() {
		if len(pending) > 0 {
			out.messages = append(out.messages, Message{Role: RoleTool, Parts: pending})
			pending = nil
		}
	}
	for index, raw := range list {
		where := "messages[" + strconv.Itoa(index) + "]"
		row, err := ingestObject(raw, where)
		if err != nil {
			return out, err
		}
		role := wireStr(row.Get("role"))
		if row.Get("name") != nil && role != "tool" {
			return out, ingestUnsupported(provider, where+".name", "a per-message participant name has no canonical slot")
		}
		switch role {
		case "system", "developer":
			flush()
			if err := ingestOnlyKeys(provider, row, []string{"role", "content"}, where); err != nil {
				return out, err
			}
			parts, marked, err := ingestContentBlocks(provider, row.Get("content"), "system", where)
			if err != nil {
				return out, err
			}
			if index == 0 {
				if marked {
					out.systemBreakpoint = true
				}
				if len(parts) == 1 {
					if t, ok := parts[0].(TextPart); ok {
						out.system = System(t.Text)
						continue
					}
				}
				out.system = SystemParts(parts...)
			} else {
				if marked {
					idx := len(out.messages)
					out.breakpointIndex = &idx
				}
				out.messages = append(out.messages, Message{Role: RoleDeveloper, Parts: parts})
			}
		case "user":
			flush()
			if err := ingestOnlyKeys(provider, row, []string{"role", "content", "name"}, where); err != nil {
				return out, err
			}
			parts, marked, err := ingestContentBlocks(provider, row.Get("content"), "user", where)
			if err != nil {
				return out, err
			}
			if marked {
				if out.breakpointIndex != nil || out.systemBreakpoint {
					return out, valueErrorf("%s: a request carries at most one prompt_cache_breakpoint", where)
				}
				idx := len(out.messages)
				out.breakpointIndex = &idx
			}
			out.messages = append(out.messages, Message{Role: RoleUser, Parts: parts})
		case "assistant":
			flush()
			allowed := append([]string{"role", "content", "tool_calls", "refusal", "reasoning_content", "name", "audio", "function_call", "annotations"}, ingestClientObjectKeys...)
			if err := ingestOnlyKeys(provider, row, allowed, where); err != nil {
				return out, err
			}
			if row.Get("audio") != nil {
				return out, ingestUnsupported(provider, where+".audio", "an assistant audio reference has no canonical part")
			}
			if row.Get("function_call") != nil {
				return out, ingestUnsupported(provider, where+".function_call", "the deprecated function-calling shape; use tool_calls")
			}
			for _, key := range ingestClientObjectKeys {
				if v, present := row.Lookup(key); present && v != nil && !ingestEmptyContainer(v) && !ingestAllEmpty(v) {
					return out, ingestUnsupported(provider, where+"."+key, "a client library's own field with no canonical part; only its empty form reads as absent")
				}
			}
			var parts []Part
			if rt := row.Get("reasoning_content"); rt != nil {
				text, err := ingestStr(rt, where+".reasoning_content")
				if err != nil {
					return out, err
				}
				parts = append(parts, ThinkingPart{Text: text})
			}
			var contentText *string
			if content := row.Get("content"); content != nil {
				textParts, marked, err := ingestContentBlocks(provider, content, "assistant", where)
				if err != nil {
					return out, err
				}
				if marked {
					return out, valueErrorf("%s: a prompt_cache_breakpoint cannot mark an assistant message (the builder refuses the same cell)", where)
				}
				parts = append(parts, textParts...)
				if s, ok := content.(string); ok {
					contentText = &s
				}
			}
			if rt := row.Get("refusal"); rt != nil {
				text, err := ingestStr(rt, where+".refusal")
				if err != nil {
					return out, err
				}
				parts = append(parts, RefusalPart{Text: text})
			}
			if row.Get("tool_calls") != nil {
				calls, err := ingestToolCalls(provider, row.Get("tool_calls"), where)
				if err != nil {
					return out, err
				}
				parts = append(parts, calls...)
			}
			if row.Get("annotations") != nil {
				cits, err := ingestAnnotations(provider, row.Get("annotations"), contentText, where)
				if err != nil {
					return out, err
				}
				parts = append(parts, cits...)
			}
			if len(parts) == 0 {
				parts = append(parts, TextPart{})
			}
			out.messages = append(out.messages, Message{Role: RoleAssistant, Parts: parts})
		case "tool":
			if err := ingestOnlyKeys(provider, row, []string{"role", "content", "tool_call_id", "name"}, where); err != nil {
				return out, err
			}
			parts, marked, err := ingestContentBlocks(provider, row.Get("content"), "tool", where)
			if err != nil {
				return out, err
			}
			if marked {
				return out, valueErrorf("%s: a prompt_cache_breakpoint cannot mark a tool message (the builder refuses the same cell)", where)
			}
			id, err := ingestStr(row.Get("tool_call_id"), where+".tool_call_id")
			if err != nil {
				return out, err
			}
			name := ""
			if row.Get("name") != nil {
				if name, err = ingestStr(row.Get("name"), where+".name"); err != nil {
					return out, err
				}
			}
			pending = append(pending, ToolResultPart{ID: id, Content: parts, Name: name})
		case "function":
			return out, ingestUnsupported(provider, where+" with role 'function'", "the deprecated function-calling shape; use a tool row with tool_call_id")
		default:
			return out, valueErrorf("%s.role must be one of system, developer, user, assistant, tool; got %q", where, role)
		}
	}
	flush()
	return out, nil
}

func ingestTools(provider string, raw any, compat ResolvedOpenAIChatCompat) ([]Tool, error) {
	if raw == nil {
		return nil, nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, typeErrorf("tools must be an array")
	}
	var tools []Tool
	for index, item := range list {
		where := "tools[" + strconv.Itoa(index) + "]"
		entry, err := ingestObject(item, where)
		if err != nil {
			return nil, err
		}
		kind := wireStr(entry.Get("type"))
		switch {
		case kind == "function":
			if err := ingestOnlyKeys(provider, entry, []string{"type", "function"}, where); err != nil {
				return nil, err
			}
			fn, err := ingestObject(entry.Get("function"), where+".function")
			if err != nil {
				return nil, err
			}
			if err := ingestOnlyKeys(provider, fn, []string{"name", "description", "parameters", "strict"}, where+".function"); err != nil {
				return nil, err
			}
			if fn.Get("strict") == true {
				return nil, ingestUnsupported(provider, where+".function.strict = true", "no per-tool strict slot (compat.strict_tools is a preset policy)")
			}
			name, err := ingestStr(fn.Get("name"), where+".function.name")
			if err != nil {
				return nil, err
			}
			tool := FunctionTool{Name: name}
			if fn.Get("description") != nil {
				if tool.Description, err = ingestStr(fn.Get("description"), where+".function.description"); err != nil {
					return nil, err
				}
			}
			if fn.Get("parameters") != nil {
				if tool.Parameters, err = ingestObject(fn.Get("parameters"), where+".function.parameters"); err != nil {
					return nil, err
				}
			}
			tools = append(tools, tool)
		case ingestGroqBuiltinInverse[kind] != "" && compat.BuiltinTools == "groq":
			config := JSONObject{}
			for k, v := range entry.All() {
				if k != "type" {
					config.Set(k, v)
				}
			}
			if len(config) == 0 {
				config = nil
			}
			tools = append(tools, BuiltinTool{Name: ingestGroqBuiltinInverse[kind], Config: config})
		default:
			return nil, ingestUnsupported(provider, where+" of type "+strconv.Quote(kind), "only function tools (and, on the groq preset, its server-executed tools) have a canonical form")
		}
	}
	return tools, nil
}

// toolChoiceFromFunctionCall translates the deprecated function_call
// spelling to the tool_choice shape.
func toolChoiceFromFunctionCall(raw any) (any, error) {
	if raw == "none" || raw == "auto" {
		return raw, nil
	}
	if obj, ok := asObject(raw); ok {
		if name, has := obj.Lookup("name"); has {
			return JSONObject{{"type", "function"}, {"function", JSONObject{{"name", name}}}}, nil
		}
	}
	return nil, valueErrorf("function_call must be 'none', 'auto', or {name}; got %s", jsonRaw(raw))
}

func ingestToolChoice(provider string, raw any, parallel any) (*ToolChoice, error) {
	mode := ""
	var allowed []string
	if raw != nil {
		switch x := jsonView(raw).(type) {
		case string:
			if x != "none" && x != "auto" && x != "required" {
				return nil, valueErrorf("tool_choice must be none, auto, required, or an object")
			}
			mode = x
		case JSONObject:
			switch wireStr(x.Get("type")) {
			case "function":
				if err := ingestOnlyKeys(provider, x, []string{"type", "function"}, "tool_choice"); err != nil {
					return nil, err
				}
				fn, err := ingestObject(x.Get("function"), "tool_choice.function")
				if err != nil {
					return nil, err
				}
				if err := ingestOnlyKeys(provider, fn, []string{"name"}, "tool_choice.function"); err != nil {
					return nil, err
				}
				name, err := ingestStr(fn.Get("name"), "tool_choice.function.name")
				if err != nil {
					return nil, err
				}
				mode, allowed = "required", []string{name}
			case "allowed_tools":
				if err := ingestOnlyKeys(provider, x, []string{"type", "allowed_tools"}, "tool_choice"); err != nil {
					return nil, err
				}
				spec, err := ingestObject(x.Get("allowed_tools"), "tool_choice.allowed_tools")
				if err != nil {
					return nil, err
				}
				if err := ingestOnlyKeys(provider, spec, []string{"mode", "tools"}, "tool_choice.allowed_tools"); err != nil {
					return nil, err
				}
				if mode, err = ingestStr(spec.Get("mode"), "tool_choice.allowed_tools.mode"); err != nil {
					return nil, err
				}
				entries, ok := spec.Get("tools").([]any)
				if !ok || len(entries) == 0 {
					return nil, valueErrorf("tool_choice.allowed_tools.tools must be a non-empty array")
				}
				for i, e := range entries {
					entryWhere := "tool_choice.allowed_tools.tools[" + strconv.Itoa(i) + "]"
					entry, err := ingestObject(e, entryWhere)
					if err != nil {
						return nil, err
					}
					if entry.Get("type") != "function" {
						return nil, ingestUnsupported(provider, entryWhere+" of type "+strconv.Quote(wireStr(entry.Get("type"))), "only function tools can be allowed on this wire")
					}
					fn, err := ingestObject(entry.Get("function"), entryWhere+".function")
					if err != nil {
						return nil, err
					}
					name, err := ingestStr(fn.Get("name"), entryWhere+".function.name")
					if err != nil {
						return nil, err
					}
					allowed = append(allowed, name)
				}
			case "custom":
				return nil, ingestUnsupported(provider, "tool_choice of type 'custom'", "custom tools have no canonical form")
			default:
				return nil, valueErrorf("tool_choice.type must be function or allowed_tools; got %q", wireStr(x.Get("type")))
			}
		default:
			return nil, valueErrorf("tool_choice must be none, auto, required, or an object")
		}
	}
	var par *bool
	if parallel != nil {
		b, ok := parallel.(bool)
		if !ok {
			return nil, typeErrorf("parallel_tool_calls must be a boolean")
		}
		par = &b
	}
	if mode == "" && par == nil {
		return nil, nil
	}
	if mode == "" {
		mode = "auto"
	}
	return &ToolChoice{Mode: mode, Allowed: allowed, Parallel: par}, nil
}

func ingestResponseFormat(provider string, raw any) (JSONObject, error) {
	obj, err := ingestObject(raw, "response_format")
	if err != nil {
		return nil, err
	}
	switch wireStr(obj.Get("type")) {
	case "text":
		return nil, ingestOnlyKeys(provider, obj, []string{"type"}, "response_format")
	case "json_object":
		if err := ingestOnlyKeys(provider, obj, []string{"type"}, "response_format"); err != nil {
			return nil, err
		}
		return JSONObject{{"type", "json_object"}}, nil
	case "json_schema":
		if err := ingestOnlyKeys(provider, obj, []string{"type", "json_schema"}, "response_format"); err != nil {
			return nil, err
		}
		inner, err := ingestObject(obj.Get("json_schema"), "response_format.json_schema")
		if err != nil {
			return nil, err
		}
		if err := ingestOnlyKeys(provider, inner, []string{"name", "schema", "strict", "description"}, "response_format.json_schema"); err != nil {
			return nil, err
		}
		if inner.Get("description") != nil {
			return nil, ingestUnsupported(provider, "response_format.json_schema.description", "the canonical response_format has no description slot (INV-050)")
		}
		schema, err := ingestObject(inner.Get("schema"), "response_format.json_schema.schema")
		if err != nil {
			return nil, err
		}
		out := JSONObject{{"type", "json_schema"}, {"schema", schema}}
		if name := inner.Get("name"); name != nil && name != "response" {
			s, err := ingestStr(name, "response_format.json_schema.name")
			if err != nil {
				return nil, err
			}
			out.Set("name", s)
		}
		if strict := inner.Get("strict"); strict != nil {
			b, ok := strict.(bool)
			if !ok {
				return nil, typeErrorf("response_format.json_schema.strict must be a boolean")
			}
			out.Set("strict", b)
		}
		return out, nil
	}
	return nil, valueErrorf("response_format.type must be text, json_object or json_schema; got %q", wireStr(obj.Get("type")))
}

func ingestReasoning(provider string, body JSONObject, compat ResolvedOpenAIChatCompat) (*Reasoning, JSONObject, error) {
	present := map[string]bool{}
	for _, k := range []string{"reasoning_effort", "reasoning", "thinking", "enable_thinking", "chat_template_kwargs", "reasoning_format"} {
		if _, ok := body.Lookup(k); ok {
			present[k] = true
		}
	}
	if len(present) == 0 {
		return nil, JSONObject{}, nil
	}
	spelledBy := map[string]bool{}
	switch compat.ThinkingFormat {
	case "reasoning_effort":
		spelledBy["reasoning_effort"] = true
	case "openrouter":
		spelledBy["reasoning"] = true
	case "deepseek", "kimi":
		spelledBy["thinking"], spelledBy["reasoning_effort"] = true, true
	case "qwen":
		spelledBy["enable_thinking"] = true
	case "qwen_chat_template":
		spelledBy["chat_template_kwargs"] = true
	}
	if compat.BuiltinTools == "groq" {
		spelledBy["reasoning_format"] = true
	}
	var foreign []string
	for k := range present {
		if !spelledBy[k] {
			foreign = append(foreign, k)
		}
	}
	if len(foreign) > 0 {
		foreign = sortStrings(foreign)
		var spelled []string
		for k := range spelledBy {
			spelled = append(spelled, k)
		}
		spelling := "nowhere (no dial)"
		if len(spelled) > 0 {
			spelling = fmtList(sortStrings(spelled))
		}
		return nil, nil, ingestUnsupported(provider, strconv.Quote(foreign[0]), "this server's reasoning dial is spelled "+spelling+"; another server's spelling would be sent and ignored")
	}
	extensions := JSONObject{}
	effort := ""
	off := false
	if present["reasoning_effort"] {
		word, err := ingestStr(body.Get("reasoning_effort"), "reasoning_effort")
		if err != nil {
			return nil, nil, err
		}
		if word == "none" {
			off = true
		} else {
			effort = word
		}
	}
	if present["thinking"] {
		spec, err := ingestObject(body.Get("thinking"), "thinking")
		if err != nil {
			return nil, nil, err
		}
		if err := ingestOnlyKeys(provider, spec, []string{"type"}, "thinking"); err != nil {
			return nil, nil, err
		}
		switch spec.Get("type") {
		case "disabled":
			if effort != "" {
				return nil, nil, valueErrorf("thinking.type=disabled next to a reasoning_effort level is contradictory")
			}
			off = true
		case "enabled":
			if effort == "" && !off {
				return nil, nil, ingestUnsupported(provider, "thinking.type=enabled without reasoning_effort", "lm15's dial is a level (MAP-7); set config.reasoning with an effort word")
			}
		default:
			return nil, nil, valueErrorf("thinking.type must be enabled or disabled; got %q", wireStr(spec.Get("type")))
		}
	}
	if present["reasoning"] {
		spec, err := ingestObject(body.Get("reasoning"), "reasoning")
		if err != nil {
			return nil, nil, err
		}
		if err := ingestOnlyKeys(provider, spec, []string{"effort", "enabled"}, "reasoning"); err != nil {
			return nil, nil, err
		}
		if spec.Get("enabled") == false {
			off = true
		} else if spec.Get("effort") != nil {
			if effort, err = ingestStr(spec.Get("effort"), "reasoning.effort"); err != nil {
				return nil, nil, err
			}
		} else {
			return nil, nil, valueErrorf("reasoning must carry effort or enabled: false")
		}
	}
	if present["enable_thinking"] {
		switch body.Get("enable_thinking") {
		case false:
			off = true
		case true:
			return nil, nil, ingestUnsupported(provider, "enable_thinking = true", "this wire has no effort level; lm15's dial is a level (MAP-7) — set config.reasoning yourself")
		default:
			return nil, nil, typeErrorf("enable_thinking must be a boolean")
		}
	}
	if present["chat_template_kwargs"] {
		spec, err := ingestObject(body.Get("chat_template_kwargs"), "chat_template_kwargs")
		if err != nil {
			return nil, nil, err
		}
		if err := ingestOnlyKeys(provider, spec, []string{"enable_thinking", "preserve_thinking"}, "chat_template_kwargs"); err != nil {
			return nil, nil, err
		}
		switch spec.Get("enable_thinking") {
		case false:
			off = true
		case true:
			return nil, nil, ingestUnsupported(provider, "chat_template_kwargs.enable_thinking = true", "this wire has no effort level; lm15's dial is a level (MAP-7) — set config.reasoning yourself")
		default:
			return nil, nil, typeErrorf("chat_template_kwargs.enable_thinking must be a boolean")
		}
	}
	summary := ""
	if present["reasoning_format"] {
		value := body.Get("reasoning_format")
		if value != "parsed" {
			return nil, nil, ingestUnsupported(provider, "reasoning_format = "+strconv.Quote(wireStr(value)), "only 'parsed' maps (Reasoning.summary='auto', MAP-7 rule 7)")
		}
		if effort == "" {
			extensions.Set("reasoning_format", value)
		} else {
			summary = "auto"
		}
	}
	if off {
		return &Reasoning{Effort: "off"}, extensions, nil
	}
	if effort == "" {
		return nil, extensions, nil
	}
	return &Reasoning{Effort: effort, Summary: summary}, extensions, nil
}

func ingestCache(provider string, body JSONObject, compat ResolvedOpenAIChatCompat, systemBreakpoint bool, breakpointIndex *int) (*CacheConfig, error) {
	var keys []string
	for _, k := range []string{"prompt_cache_key", "prompt_cache_retention", "prompt_cache_options"} {
		if _, ok := body.Lookup(k); ok {
			keys = append(keys, k)
		}
	}
	marked := systemBreakpoint || breakpointIndex != nil
	if len(keys) == 0 && !marked {
		return nil, nil
	}
	if compat.CacheControl != "openai" && compat.CacheControl != "openai_implicit" {
		what := "prompt_cache_breakpoint"
		if len(keys) > 0 {
			what = sortStrings(keys)[0]
		}
		return nil, ingestUnsupported(provider, strconv.Quote(what), "this server has no OpenAI prompt-cache control (compat.cache_control)")
	}
	if marked && compat.CacheControl != "openai" {
		return nil, ingestUnsupported(provider, "prompt_cache_breakpoint", "this server swallows an explicit breakpoint silently (compat.cache_control=openai_implicit)")
	}
	key := stringOnly(body.Get("prompt_cache_key"))
	retention := ""
	if v, ok := body.Lookup("prompt_cache_retention"); ok {
		if v != "24h" {
			return nil, ingestUnsupported(provider, "prompt_cache_retention = "+strconv.Quote(wireStr(v)), "only '24h' has a canonical value (CacheConfig.retention='long')")
		}
		retention = "long"
	}
	explicit := false
	if v, ok := body.Lookup("prompt_cache_options"); ok {
		spec, err := ingestObject(v, "prompt_cache_options")
		if err != nil {
			return nil, err
		}
		if err := ingestOnlyKeys(provider, spec, []string{"mode", "ttl"}, "prompt_cache_options"); err != nil {
			return nil, err
		}
		if spec.Get("ttl") != nil {
			return nil, ingestUnsupported(provider, "prompt_cache_options.ttl", "CacheConfig.retention names 24h only")
		}
		switch spec.Get("mode") {
		case "explicit":
			explicit = true
		case "implicit":
			return nil, ingestUnsupported(provider, "prompt_cache_options.mode = 'implicit'", "the server default; a canonical CacheConfig names auto or off")
		default:
			return nil, valueErrorf("prompt_cache_options.mode must be explicit or implicit; got %q", wireStr(spec.Get("mode")))
		}
	}
	if explicit && !marked {
		if key != "" || retention != "" {
			return nil, valueErrorf("prompt_cache_options.mode=explicit with no breakpoint is the off switch; it cannot carry a key or retention (INV-027)")
		}
		return &CacheConfig{Mode: "off"}, nil
	}
	if systemBreakpoint {
		return &CacheConfig{Prefix: "stable", Key: key, Retention: retention}, nil
	}
	if breakpointIndex != nil {
		return &CacheConfig{PrefixUntilIndex: breakpointIndex, Key: key, Retention: retention}, nil
	}
	return &CacheConfig{Key: key, Retention: retention}, nil
}

func ingestConfig(provider string, body JSONObject, compat ResolvedOpenAIChatCompat, systemBreakpoint bool, breakpointIndex *int) (Config, error) {
	var cfg Config
	var err error
	_, hasMCT := body.Lookup("max_completion_tokens")
	_, hasMT := body.Lookup("max_tokens")
	if hasMCT || hasMT {
		if hasMCT && hasMT && !jsonEqual(body.Get("max_completion_tokens"), body.Get("max_tokens")) {
			return cfg, valueErrorf("max_tokens and max_completion_tokens disagree")
		}
		v := body.Get("max_completion_tokens")
		if !hasMCT {
			v = body.Get("max_tokens")
		}
		if v != nil {
			n, err := jsonInt(v, "max_tokens")
			if err != nil {
				return cfg, err
			}
			cfg.MaxTokens = &n
		}
	}
	if cfg.Temperature, err = optFloat(body, "temperature"); err != nil {
		return cfg, err
	}
	if cfg.TopP, err = optFloat(body, "top_p"); err != nil {
		return cfg, err
	}
	if cfg.TopK, err = optInt(body, "top_k"); err != nil {
		return cfg, err
	}
	if cfg.Seed, err = optInt(body, "seed"); err != nil {
		return cfg, err
	}
	if cfg.FrequencyPenalty, err = optFloat(body, "frequency_penalty"); err != nil {
		return cfg, err
	}
	if cfg.PresencePenalty, err = optFloat(body, "presence_penalty"); err != nil {
		return cfg, err
	}
	if cfg.ServiceTier, err = optString(body, "service_tier"); err != nil {
		return cfg, err
	}
	if cfg.Store, err = optBool(body, "store"); err != nil {
		return cfg, err
	}
	if cfg.Stop, err = stringList(body.Get("stop"), "stop"); err != nil {
		return cfg, err
	}
	switch lp := body.Get("logprobs"); lp {
	case true:
		top := 0
		if v, ok := body.Lookup("top_logprobs"); ok && v != nil {
			if top, err = jsonInt(v, "top_logprobs"); err != nil {
				return cfg, err
			}
		}
		cfg.Logprobs = &top
	case nil, false:
		if _, ok := body.Lookup("top_logprobs"); ok {
			return cfg, valueErrorf("top_logprobs requires logprobs: true")
		}
	default:
		return cfg, typeErrorf("logprobs must be a boolean")
	}
	if rf, ok := body.Lookup("response_format"); ok {
		if cfg.ResponseFormat, err = ingestResponseFormat(provider, rf); err != nil {
			return cfg, err
		}
	}
	_, hasFunctionCall := body.Lookup("function_call")
	_, hasToolChoice := body.Lookup("tool_choice")
	if hasFunctionCall && hasToolChoice {
		return cfg, valueErrorf("function_call and tool_choice cannot both be given")
	}
	rawToolChoice := body.Get("tool_choice")
	if hasFunctionCall {
		if rawToolChoice, err = toolChoiceFromFunctionCall(body.Get("function_call")); err != nil {
			return cfg, err
		}
	}
	if cfg.ToolChoice, err = ingestToolChoice(provider, rawToolChoice, body.Get("parallel_tool_calls")); err != nil {
		return cfg, err
	}
	var userKeys []string
	for _, k := range []string{"user", "safety_identifier", "user_id"} {
		if _, ok := body.Lookup(k); ok {
			userKeys = append(userKeys, k)
		}
	}
	if inVocab("user_id", userKeys) && compat.UserField != "user_id" {
		return cfg, ingestUnsupported(provider, "'user_id'", "this server spells the end-user field "+strconv.Quote(compat.UserField))
	}
	if len(userKeys) > 1 {
		return cfg, valueErrorf("one end-user identifier only; got %v", userKeys)
	}
	if len(userKeys) == 1 {
		if cfg.UserID, err = optString(body, userKeys[0]); err != nil {
			return cfg, err
		}
	}
	reasoning, extensions, err := ingestReasoning(provider, body, compat)
	if err != nil {
		return cfg, err
	}
	cfg.Reasoning = reasoning
	if cfg.Cache, err = ingestCache(provider, body, compat, systemBreakpoint, breakpointIndex); err != nil {
		return cfg, err
	}
	for k, v := range body.All() {
		if ingestExtensionsKeys[k] {
			extensions.Set(k, v)
		}
	}
	if len(extensions) > 0 {
		cfg.Extensions = extensions
	}
	return cfg, cfg.Validate()
}

func ingestOpenAIChat(provider string, body JSONObject, compat ResolvedOpenAIChatCompat) (*Request, error) {
	if body == nil {
		return nil, typeErrorf("a Chat Completions request body is a JSON object, got null")
	}
	for k := range body.All() {
		if why, refused := ingestRefusedKeys[k]; refused {
			return nil, ingestUnsupported(provider, strconv.Quote(k), why)
		}
		if !ingestConfigKeys[k] && !ingestExtensionsKeys[k] && !ingestCallModeKeys[k] {
			return nil, ingestUnsupported(provider, strconv.Quote(k), "no verdict for this key (lm15-contract/tools/openai-chat-ingest-verdicts.json); lm15 never drops a key silently")
		}
	}
	model := stringOnly(body.Get("model"))
	if model == "" {
		return nil, valueErrorf("model must be a non-empty string")
	}
	if _, ok := body.Lookup("messages"); !ok {
		return nil, valueErrorf("messages is required")
	}
	rows, err := ingestRows(provider, body.Get("messages"))
	if err != nil {
		return nil, err
	}
	_, hasFunctions := body.Lookup("functions")
	_, hasTools := body.Lookup("tools")
	if hasFunctions && hasTools {
		return nil, valueErrorf("functions and tools cannot both be given")
	}
	rawTools := body.Get("tools")
	if hasFunctions {
		list, ok := body.Get("functions").([]any)
		if !ok {
			return nil, typeErrorf("functions must be an array")
		}
		rawTools = toAnyList(list, func(fn any) any { return JSONObject{{"type", "function"}, {"function", fn}} })
	}
	tools, err := ingestTools(provider, rawTools, compat)
	if err != nil {
		return nil, err
	}
	cfg, err := ingestConfig(provider, body, compat, rows.systemBreakpoint, rows.breakpointIndex)
	if err != nil {
		return nil, err
	}
	req := &Request{Model: model, Messages: rows.messages, System: rows.system, Tools: tools, Config: cfg}
	return req, req.Validate()
}

// RequestFromOpenAIChat reads a Chat Completions request body into a
// canonical Request under the named preset's spellings ("" = OpenAI's own).
func RequestFromOpenAIChat(body JSONObject, compatPreset string) (*Request, error) {
	partial := OpenAIChatCompat{}
	if compatPreset != "" {
		p, err := OpenAIChatPreset(compatPreset)
		if err != nil {
			return nil, err
		}
		partial = p
	}
	model := stringOnly(body.Get("model"))
	return ingestOpenAIChat("openai-chat", body, ResolveOpenAIChatCompat(partial.ForModel(model)))
}

// RequestFromOpenAIChatCompat reads a body under an explicit compat value.
func RequestFromOpenAIChatCompat(body JSONObject, compat OpenAIChatCompat) (*Request, error) {
	model := stringOnly(body.Get("model"))
	return ingestOpenAIChat("openai-chat", body, ResolveOpenAIChatCompat(compat.ForModel(model)))
}

func (l *OpenAIChatLM) requestFromOpenAIChat(body JSONObject) (*Request, error) {
	compat := l.resolved
	if model := stringOnly(body.Get("model")); model != "" {
		compat = l.compatFor(model)
	}
	return ingestOpenAIChat(l.provider, body, compat)
}
