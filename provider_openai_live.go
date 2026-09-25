package lm15

import (
	"context"
	"iter"
	"net/url"
	"strings"
)

// OpenAI Realtime: the pure codec (setup frames, client event → frames,
// server frame → events) and the socket-backed session and completion.

func liveUsageFromResponse(response JSONObject) *Usage {
	usageData := wireObj(response.Get("usage"))
	if usageData == nil {
		return nil
	}
	u := openaiUsage(usageData)
	return &u
}

func (l *OpenAILM) liveURL(model string) string {
	u, err := url.Parse(l.baseURL)
	if err != nil {
		return l.baseURL
	}
	scheme := "ws"
	if u.Scheme == "https" {
		scheme = "wss"
	}
	basePath := strings.TrimRight(u.Path, "/")
	path := "/realtime"
	if basePath != "" {
		path = basePath + "/realtime"
	}
	out := url.URL{Scheme: scheme, Host: u.Host, Path: path, RawQuery: url.Values{"model": {model}}.Encode()}
	return out.String()
}

func (l *OpenAILM) liveHeaders(ctx context.Context) ([][2]string, error) {
	cred, err := l.resolveCredential(ctx)
	if err != nil {
		return nil, err
	}
	headers := append([][2]string(nil), l.access.Headers...)
	if cred != nil {
		name, value, ok, err := AuthHeaderFor(l.access, cred, l.apiKeyHeader)
		if err != nil {
			return nil, err
		}
		if ok {
			headers = append(headers, [2]string{name, value})
		}
	}
	return headers, nil
}

func liveAudioFormat(f AudioFormat) JSONObject {
	if f.Encoding == "pcm16" {
		return JSONObject{{"type", "audio/pcm"}, {"rate", f.SampleRate}}
	}
	return JSONObject{{"type", "audio/" + f.Encoding}}
}

func (l *OpenAILM) liveSessionUpdatePayload(config *LiveConfig) (JSONObject, error) {
	session := JSONObject{{"type", "realtime"}}
	if config.System != nil {
		text, err := systemText(config.System, l.provider)
		if err != nil {
			return nil, err
		}
		session.Set("instructions", text)
	}
	audio := JSONObject{}
	if config.OutputFormat != nil || config.Voice != "" {
		session.Set("output_modalities", []any{"audio"})
		output := JSONObject{}
		if config.OutputFormat != nil {
			output.Set("format", liveAudioFormat(*config.OutputFormat))
		}
		if config.Voice != "" {
			output.Set("voice", config.Voice)
		}
		audio.Set("output", output)
	} else {
		session.Set("output_modalities", []any{"text"})
	}
	if config.InputFormat != nil {
		audio.Set("input", JSONObject{{"format", liveAudioFormat(*config.InputFormat)}, {"turn_detection", nil}})
	}
	if len(audio) > 0 {
		session.Set("audio", audio)
	}
	if len(config.Tools) > 0 {
		var tools []any
		for _, t := range config.Tools {
			if ft, ok := t.(FunctionTool); ok {
				tools = append(tools, JSONObject{{"type", "function"}, {"name", ft.Name}, {"description", nilIfEmpty(ft.Description)}, {"parameters", ft.EffectiveParameters()}})
			}
		}
		session.Set("tools", tools)
	}
	for k, v := range config.Extensions.All() {
		session.Set(k, v)
	}
	return JSONObject{{"type", "session.update"}, {"session", session}}, nil
}

func (l *OpenAILM) liveSetupFrames(config *LiveConfig) ([]JSONObject, error) {
	frame, err := l.liveSessionUpdatePayload(config)
	if err != nil {
		return nil, err
	}
	return []JSONObject{frame}, nil
}

func (l *OpenAILM) liveEncoder(*LiveConfig) func(LiveClientEvent) ([]JSONObject, error) {
	return l.encodeLiveClientEvent
}

func (l *OpenAILM) encodeLiveClientEvent(event LiveClientEvent) ([]JSONObject, error) {
	userMessage := func(content []any) JSONObject {
		return JSONObject{{"type", "conversation.item.create"}, {"item", JSONObject{{"type", "message"}, {"role", "user"}, {"content", content}}}}
	}
	switch e := event.(type) {
	case LiveClientAudioEvent:
		return []JSONObject{{{"type", "input_audio_buffer.append"}, {"audio", e.Data}}}, nil
	case LiveClientEndAudioEvent:
		return []JSONObject{{{"type", "input_audio_buffer.commit"}}, {{"type", "response.create"}}}, nil
	case LiveClientInterruptEvent:
		return []JSONObject{{{"type", "response.cancel"}}}, nil
	case LiveClientTextEvent:
		return []JSONObject{userMessage([]any{JSONObject{{"type", "input_text"}, {"text", e.Text}}}), {{"type", "response.create"}}}, nil
	case LiveClientTurnEvent:
		var content []any
		for _, p := range e.Parts {
			block, err := partToOpenAIInput(p, "")
			if err != nil {
				return nil, err
			}
			content = append(content, block)
		}
		if e.TurnComplete {
			return []JSONObject{userMessage(content), {{"type", "response.create"}}}, nil
		}
		return []JSONObject{userMessage(content)}, nil
	case LiveClientImageEvent:
		return []JSONObject{userMessage([]any{JSONObject{{"type", "input_image"}, {"image_url", "data:" + e.EffectiveMediaType() + ";base64," + e.Data}}}), {{"type", "response.create"}}}, nil
	case LiveClientToolResultEvent:
		output, err := partsToText(e.Content, l.provider, "a Realtime function_call_output")
		if err != nil {
			return nil, err
		}
		return []JSONObject{{{"type", "conversation.item.create"}, {"item", JSONObject{{"type", "function_call_output"}, {"call_id", e.ID}, {"output", output}}}}, {{"type", "response.create"}}}, nil
	}
	return nil, nil
}

func (l *OpenAILM) liveDecode(raw []byte) ([]LiveServerEvent, error) {
	decoded, err := DecodeJSON(raw)
	if err != nil {
		return nil, nil
	}
	payload := wireObj(decoded)
	if payload == nil {
		return nil, nil
	}
	et := wireStr(payload.Get("type"))
	var events []LiveServerEvent
	switch et {
	case "response.output_text.delta", "response.text.delta", "response.output_audio_transcript.delta", "response.audio_transcript.delta":
		if delta := firstStr(payload.Get("delta"), payload.Get("text")); delta != "" {
			events = append(events, LiveServerTextEvent{Text: delta})
		}
	case "response.output_audio.delta":
		if delta := wireStr(payload.Get("delta")); delta != "" {
			events = append(events, LiveServerAudioEvent{Data: delta})
		}
	case "response.function_call_arguments.delta":
		if delta := wireStr(payload.Get("delta")); delta != "" {
			events = append(events, LiveServerToolCallDeltaEvent{InputDelta: delta, ID: firstStr(payload.Get("call_id"), payload.Get("id")), Name: wireStr(payload.Get("name"))})
		}
	case "response.output_item.done":
		item := wireObj(payload.Get("item"))
		if wireStr(item.Get("type")) == "function_call" {
			if callID := firstStr(item.Get("call_id"), item.Get("id")); callID != "" {
				name := wireStr(item.Get("name"))
				if name == "" {
					name = "tool"
				}
				events = append(events, LiveServerToolCallEvent{ID: callID, Name: name, Input: parseJSONObject(item.Get("arguments"))})
			}
		}
	case "response.done", "response.completed":
		response := wireObj(payload.Get("response"))
		usage := liveUsageFromResponse(response)
		hasCall := false
		for _, i := range wireList(response.Get("output")) {
			if obj := wireObj(i); obj != nil && wireStr(obj.Get("type")) == "function_call" {
				hasCall = true
			}
		}
		switch {
		case wireStr(response.Get("status")) == "cancelled":
			if usage != nil {
				events = append(events, LiveServerUsageEvent{Usage: *usage})
			}
			events = append(events, LiveServerInterruptedEvent{})
		case hasCall:
			if usage != nil {
				events = append(events, LiveServerUsageEvent{Usage: *usage})
			}
		default:
			u := Usage{}
			if usage != nil {
				u = *usage
			}
			events = append(events, LiveServerTurnEndEvent{Usage: u})
		}
	case "response.cancelled", "response.canceled":
		events = append(events, LiveServerInterruptedEvent{})
	case "error", "response.error":
		code, message := openaiStreamErrorFields(payload)
		if code == "response_cancel_not_active" {
			return events, nil
		}
		events = append(events, LiveServerErrorEvent{Error: openaiErrorDetail(code, message)})
	}
	return events, nil
}

func (l *OpenAILM) live(ctx context.Context, config *LiveConfig) (LiveSession, error) {
	headers, err := l.liveHeaders(ctx)
	if err != nil {
		return nil, err
	}
	conn, err := dialWebSocket(ctx, l.liveURL(config.Model), headers)
	if err != nil {
		return nil, err
	}
	frames, err := l.liveSetupFrames(config)
	if err != nil {
		conn.Close()
		return nil, err
	}
	for _, f := range frames {
		if err := conn.Send(ctx, mustJSON(f)); err != nil {
			conn.Close()
			return nil, err
		}
	}
	return newWebSocketLiveSession(conn, l.encodeLiveClientEvent, l.liveDecode), nil
}

// ─── Chat completion over Realtime (live models) ─────────────────────

func (l *OpenAILM) liveMessageFramesForRequest(req *Request) ([]JSONObject, error) {
	var frames []JSONObject
	for _, m := range req.Messages {
		if m.Role == RoleTool {
			for _, p := range m.Parts {
				tr, ok := p.(ToolResultPart)
				if !ok {
					continue
				}
				text, err := partsToText(tr.Content, l.provider, "a Realtime function_call_output")
				if err != nil {
					return nil, err
				}
				frames = append(frames, JSONObject{{"type", "conversation.item.create"}, {"item", JSONObject{{"type", "function_call_output"}, {"call_id", tr.ID}, {"output", toolResultErrorText(tr, text)}}}})
			}
			continue
		}
		var content []any
		for _, p := range m.Parts {
			switch p.(type) {
			case ToolCallPart, ToolResultPart:
				continue
			}
			block, err := partToOpenAIInput(p, l.provider)
			if err != nil {
				return nil, err
			}
			content = append(content, block)
		}
		if len(content) > 0 {
			frames = append(frames, JSONObject{{"type", "conversation.item.create"}, {"item", JSONObject{{"type", "message"}, {"role", m.Role}, {"content", content}}}})
		}
		for _, p := range m.Parts {
			if tc, ok := p.(ToolCallPart); ok {
				frames = append(frames, JSONObject{{"type", "conversation.item.create"}, {"item", JSONObject{{"type", "function_call"}, {"call_id", tc.ID}, {"name", tc.Name}, {"arguments", jsonRaw(tc.Input)}}}})
			}
		}
	}
	create := JSONObject{{"type", "response.create"}}
	if wireStr(req.Config.Extensions.Get("output")) == "audio" {
		create.Set("response", JSONObject{{"output_modalities", []any{"audio"}}})
	}
	return append(frames, create), nil
}

func (l *OpenAILM) decodeLiveCompletionEvents(raw []byte) []StreamEvent {
	decoded, err := DecodeJSON(raw)
	if err != nil {
		return nil
	}
	payload := wireObj(decoded)
	if payload == nil {
		return nil
	}
	et := wireStr(payload.Get("type"))
	switch et {
	case "response.output_text.delta", "response.text.delta", "response.output_audio_transcript.delta", "response.audio_transcript.delta":
		if delta := firstStr(payload.Get("delta"), payload.Get("text")); delta != "" {
			return []StreamEvent{StreamDeltaEvent{Delta: TextDelta{Text: delta}}}
		}
	case "response.output_audio.delta":
		if delta := wireStr(payload.Get("delta")); delta != "" {
			return []StreamEvent{StreamDeltaEvent{Delta: AudioDelta{Data: S(delta), MediaType: "audio/wav"}}}
		}
	case "response.output_item.added", "response.output_item.done", "response.function_call_arguments.delta", "response.function_call_arguments.done":
		var callID, name, args string
		if et == "response.output_item.added" || et == "response.output_item.done" {
			item := wireObj(payload.Get("item"))
			if wireStr(item.Get("type")) != "function_call" {
				return nil
			}
			callID, name = firstStr(item.Get("call_id"), item.Get("id")), wireStr(item.Get("name"))
			args = wireStr(item.Get("arguments"))
		} else {
			callID, name = firstStr(payload.Get("call_id"), payload.Get("id")), wireStr(payload.Get("name"))
			if strings.HasSuffix(et, "delta") {
				args = wireStr(payload.Get("delta"))
			} else {
				args = wireStr(payload.Get("arguments"))
			}
		}
		if name == "" {
			name = "tool"
		}
		return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: args, ID: callID, Name: name}}}
	case "response.done", "response.completed":
		response := wireObj(payload.Get("response"))
		usage := openaiUsage(wireObj(response.Get("usage")))
		return []StreamEvent{StreamEndEvent{FinishReason: FinishStop, Usage: &usage, ProviderData: response}}
	case "error", "response.error":
		code, message := openaiStreamErrorFields(payload)
		return []StreamEvent{StreamErrorEvent{Error: openaiErrorDetail(code, message)}}
	}
	return nil
}

func (l *OpenAILM) streamViaLiveCompletion(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		headers, err := l.liveHeaders(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		conn, err := dialWebSocket(ctx, l.liveURL(req.Model), headers)
		if err != nil {
			yield(nil, err)
			return
		}
		defer conn.Close()
		ext := copyObject(req.Config.Extensions)
		ext.Delete("transport")
		ext.Delete("prompt_caching")
		ext.Delete("output")
		if len(ext) == 0 {
			ext = nil
		}
		config := &LiveConfig{Model: req.Model, System: req.System, Tools: req.Tools, Extensions: ext}
		setup, err := l.liveSessionUpdatePayload(config)
		if err != nil {
			yield(nil, err)
			return
		}
		if err := conn.Send(ctx, mustJSON(setup)); err != nil {
			yield(nil, err)
			return
		}
		frames, err := l.liveMessageFramesForRequest(req)
		if err != nil {
			yield(nil, err)
			return
		}
		for _, f := range frames {
			if err := conn.Send(ctx, mustJSON(f)); err != nil {
				yield(nil, err)
				return
			}
		}
		if !yield(StreamStartEvent{Model: req.Model}, nil) {
			return
		}
		sawTool := false
		usage := Usage{}
		for {
			raw, err := conn.Recv(ctx)
			if err != nil {
				yield(nil, newError(KindTransport, err.Error()).WithCause(err))
				return
			}
			for _, event := range l.decodeLiveCompletionEvents(raw) {
				switch e := event.(type) {
				case StreamDeltaEvent:
					if _, ok := e.Delta.(ToolCallDelta); ok {
						sawTool = true
					}
					if !yield(e, nil) {
						return
					}
				case StreamErrorEvent:
					yield(e, nil)
					return
				case StreamEndEvent:
					if e.Usage != nil {
						usage = *e.Usage
					}
					finish := e.FinishReason
					if sawTool {
						finish = FinishToolCall
					} else if finish == "" {
						finish = FinishStop
					}
					yield(StreamEndEvent{FinishReason: finish, Usage: &usage}, nil)
					return
				}
			}
		}
	}
}
