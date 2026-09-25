package lm15

import (
	"context"
	"encoding/base64"
	"iter"
	"net/url"
	"strconv"
	"strings"
)

// Gemini Live (BidiGenerateContent): the pure codec plus the socket-backed
// session and the live-model completion path.

func geminiAudioNativeLiveModel(model string) bool {
	lowered := strings.ToLower(model)
	return strings.Contains(lowered, "live-preview") || strings.Contains(lowered, "native-audio")
}

func (l *GeminiLM) shouldUseLiveCompletion(req *Request) bool {
	mode := strings.ToLower(wireStr(req.Config.Extensions.Get("transport")))
	if mode == "live" || mode == "websocket" || mode == "ws" {
		return true
	}
	model := strings.ToLower(req.Model)
	return strings.Contains(model, "-live") || strings.HasSuffix(model, "live")
}

func (l *GeminiLM) streamOverride(ctx context.Context, req *Request) (iter.Seq2[StreamEvent, error], bool) {
	if !l.shouldUseLiveCompletion(req) {
		return nil, false
	}
	return l.streamViaLiveCompletion(ctx, req), true
}

func (l *GeminiLM) liveURL(ctx context.Context) (string, error) {
	key, err := l.credentialString(ctx)
	if err != nil {
		return "", err
	}
	u, err := url.Parse(l.baseURL)
	if err != nil {
		return "", err
	}
	scheme := "ws"
	if u.Scheme == "https" {
		scheme = "wss"
	}
	out := url.URL{Scheme: scheme, Host: u.Host, Path: "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent", RawQuery: url.Values{"key": {key}}.Encode()}
	return out.String(), nil
}

func (l *GeminiLM) liveSetupPayload(config *LiveConfig) (JSONObject, error) {
	setup := JSONObject{{"model", l.modelPath(config.Model)}}
	if config.System != nil {
		text, err := systemText(config.System, l.provider)
		if err != nil {
			return nil, err
		}
		setup.Set("systemInstruction", JSONObject{{"parts", []any{JSONObject{{"text", text}}}}})
	}
	var functions []any
	for _, t := range config.Tools {
		if ft, ok := t.(FunctionTool); ok {
			functions = append(functions, JSONObject{{"name", ft.Name}, {"description", nilIfEmpty(ft.Description)}, {"parameters", ft.EffectiveParameters()}})
		}
	}
	if len(functions) > 0 {
		setup.Set("tools", []any{JSONObject{{"functionDeclarations", functions}}})
	}
	gen := JSONObject{}
	if config.OutputFormat != nil || geminiAudioNativeLiveModel(config.Model) {
		gen.Set("responseModalities", []any{"AUDIO"})
	}
	if config.Voice != "" {
		gen.Set("speechConfig", JSONObject{{"voiceConfig", JSONObject{{"prebuiltVoiceConfig", JSONObject{{"voiceName", config.Voice}}}}}})
	}
	if len(gen) > 0 {
		setup.Set("generationConfig", gen)
	}
	for k, v := range config.Extensions.All() {
		setup.Set(k, v)
	}
	return JSONObject{{"setup", setup}}, nil
}

func (l *GeminiLM) liveSetupFrames(config *LiveConfig) ([]JSONObject, error) {
	payload, err := l.liveSetupPayload(config)
	if err != nil {
		return nil, err
	}
	if geminiAudioNativeLiveModel(config.Model) {
		setIn(&payload, JSONObject{}, "setup", "outputAudioTranscription")
	}
	return []JSONObject{payload}, nil
}

func (l *GeminiLM) liveEncoder(config *LiveConfig) func(LiveClientEvent) ([]JSONObject, error) {
	audioNative := geminiAudioNativeLiveModel(config.Model)
	return func(event LiveClientEvent) ([]JSONObject, error) {
		if t, ok := event.(LiveClientTextEvent); ok && audioNative {
			return []JSONObject{{{"realtimeInput", JSONObject{{"text", t.Text}}}}}, nil
		}
		return l.encodeLiveClientEvent(event)
	}
}

func (l *GeminiLM) encodeLiveClientEvent(event LiveClientEvent) ([]JSONObject, error) {
	switch e := event.(type) {
	case LiveClientTurnEvent:
		var parts []any
		for _, p := range e.Parts {
			b, err := l.part(p, nil)
			if err != nil {
				return nil, err
			}
			parts = append(parts, b)
		}
		return []JSONObject{{{"clientContent", JSONObject{{"turns", []any{JSONObject{{"role", "user"}, {"parts", parts}}}}, {"turnComplete", e.TurnComplete}}}}}, nil
	case LiveClientAudioEvent:
		return []JSONObject{{{"realtimeInput", JSONObject{{"audio", JSONObject{{"mimeType", e.EffectiveMediaType()}, {"data", e.Data}}}}}}}, nil
	case LiveClientImageEvent:
		return []JSONObject{{{"realtimeInput", JSONObject{{"video", JSONObject{{"mimeType", e.EffectiveMediaType()}, {"data", e.Data}}}}}}}, nil
	case LiveClientInterruptEvent:
		return []JSONObject{{{"clientContent", JSONObject{{"turnComplete", true}}}}}, nil
	case LiveClientEndAudioEvent:
		return []JSONObject{{{"realtimeInput", JSONObject{{"audioStreamEnd", true}}}}}, nil
	case LiveClientTextEvent:
		return []JSONObject{{{"clientContent", JSONObject{{"turns", []any{JSONObject{{"role", "user"}, {"parts", []any{JSONObject{{"text", e.Text}}}}}}}, {"turnComplete", true}}}}}, nil
	case LiveClientToolResultEvent:
		text, err := partsToText(e.Content, "", "")
		if err != nil {
			return nil, err
		}
		return []JSONObject{{{"toolResponse", JSONObject{{"functionResponses", []any{JSONObject{{"id", e.ID}, {"response", JSONObject{{"output", []any{JSONObject{{"text", text}}}}}}}}}}}}}, nil
	}
	return nil, nil
}

func (l *GeminiLM) liveUsage(payload, server JSONObject) Usage {
	usage := wireObj(payload.Get("usageMetadata"))
	if usage == nil && server != nil {
		usage = wireObj(server.Get("usageMetadata"))
	}
	return geminiUsage(usage, "responseTokenCount", "candidatesTokenCount")
}

func (l *GeminiLM) liveDecode(raw []byte) ([]LiveServerEvent, error) {
	decoded, err := DecodeJSON(raw)
	if err != nil {
		return nil, nil
	}
	payload := wireObj(decoded)
	if payload == nil {
		return nil, nil
	}
	if errRaw, ok := payload.Lookup("error"); ok {
		e := wireObj(errRaw)
		code := firstStr(e.Get("status"), e.Get("code"))
		if code == "" {
			code = "provider"
		}
		return []LiveServerEvent{LiveServerErrorEvent{Error: l.errorDetail(code, wireStr(e.Get("message")))}}, nil
	}
	var events []LiveServerEvent
	if tc := wireObj(payload.Get("toolCall")); tc != nil {
		for _, raw := range wireList(tc.Get("functionCalls")) {
			if fc := wireObj(raw); fc != nil {
				events = append(events, LiveServerToolCallEvent{ID: firstOr(wireStr(fc.Get("id")), "fc_0"), Name: firstOr(wireStr(fc.Get("name")), "tool"), Input: objOrEmpty(fc.Get("args"))})
			}
		}
	}
	server := wireObj(payload.Get("serverContent"))
	if server == nil {
		return events, nil
	}
	if modelTurn := wireObj(server.Get("modelTurn")); modelTurn != nil {
		for _, raw := range wireList(modelTurn.Get("parts")) {
			part := wireObj(raw)
			if part == nil {
				continue
			}
			if _, has := part.Lookup("text"); has {
				events = append(events, LiveServerTextEvent{Text: wireStr(part.Get("text"))})
			} else if inline := wireObj(part.Get("inlineData")); inline != nil {
				mime := wireStr(inline.Get("mimeType"))
				if strings.HasPrefix(mime, "audio/") {
					events = append(events, LiveServerAudioEvent{Data: wireStr(inline.Get("data")), MediaType: mime})
				}
			} else if fc := wireObj(part.Get("functionCall")); fc != nil {
				events = append(events, LiveServerToolCallEvent{ID: firstOr(wireStr(fc.Get("id")), "fc_0"), Name: firstOr(wireStr(fc.Get("name")), "tool"), Input: objOrEmpty(fc.Get("args"))})
			}
		}
	}
	if tx := wireObj(server.Get("outputTranscription")); tx != nil && truthy(tx.Get("text")) {
		events = append(events, LiveServerTextEvent{Text: wireStr(tx.Get("text"))})
	}
	hasUsage := wireObj(payload.Get("usageMetadata")) != nil || wireObj(server.Get("usageMetadata")) != nil
	if hasUsage && !truthy(server.Get("turnComplete")) {
		events = append(events, LiveServerUsageEvent{Usage: l.liveUsage(payload, server)})
	}
	if truthy(server.Get("interrupted")) {
		events = append(events, LiveServerInterruptedEvent{})
	}
	if truthy(server.Get("turnComplete")) {
		events = append(events, LiveServerTurnEndEvent{Usage: l.liveUsage(payload, server)})
	}
	return events, nil
}

func firstOr(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

func objOrEmpty(v any) JSONObject {
	if o := wireObj(v); o != nil {
		return o
	}
	return JSONObject{}
}

// liveSetupStatus: true = setupComplete, false = keep waiting; error on failure.
func (l *GeminiLM) liveSetupStatus(raw []byte) (bool, error) {
	decoded, err := DecodeJSON(raw)
	if err != nil {
		return false, nil
	}
	payload := wireObj(decoded)
	if payload == nil {
		return false, nil
	}
	if _, ok := payload.Lookup("setupComplete"); ok {
		return true, nil
	}
	if errRaw, ok := payload.Lookup("error"); ok {
		e := wireObj(errRaw)
		msg := wireStr(errRaw)
		code := "live_setup"
		if e != nil {
			msg = wireStr(e.Get("message"))
			code = firstOr(wireStr(e.Get("status")), "live_setup")
		}
		return false, l.providerError(KindInvalidRequest, "Live setup failed: "+msg, 0, code, "")
	}
	return false, nil
}

func (l *GeminiLM) waitForSetupComplete(ctx context.Context, conn wsConn) error {
	for {
		raw, err := conn.Recv(ctx)
		if err != nil {
			return newError(KindTransport, err.Error()).WithCause(err)
		}
		done, err := l.liveSetupStatus(raw)
		if err != nil {
			return err
		}
		if done {
			return nil
		}
	}
}

func (l *GeminiLM) live(ctx context.Context, config *LiveConfig) (LiveSession, error) {
	u, err := l.liveURL(ctx)
	if err != nil {
		return nil, err
	}
	conn, err := dialWebSocket(ctx, u, nil)
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
	if err := l.waitForSetupComplete(ctx, conn); err != nil {
		conn.Close()
		return nil, err
	}
	return newWebSocketLiveSession(conn, l.liveEncoder(config), l.liveDecode), nil
}

// ─── Completion over Live ────────────────────────────────────────────

func (l *GeminiLM) liveSetupPayloadFromRequest(req *Request) (JSONObject, error) {
	ext := copyObject(req.Config.Extensions)
	ext.Delete("transport")
	ext.Delete("prompt_caching")
	ext.Delete("output")
	if len(ext) == 0 {
		ext = nil
	}
	payload, err := l.liveSetupPayload(&LiveConfig{Model: req.Model, System: req.System, Tools: req.Tools, Extensions: ext})
	if err != nil {
		return nil, err
	}
	output := wireStr(req.Config.Extensions.Get("output"))
	audioNative := geminiAudioNativeLiveModel(req.Model)
	if output == "audio" || audioNative {
		setIn(&payload, []any{"AUDIO"}, "setup", "generationConfig", "responseModalities")
		if output != "audio" {
			setIn(&payload, JSONObject{}, "setup", "outputAudioTranscription")
		}
		hasMedia := false
		for _, m := range req.Messages {
			for _, p := range m.Parts {
				switch p.(type) {
				case AudioPart, VideoPart:
					hasMedia = true
				}
			}
		}
		if hasMedia {
			setIn(&payload, true, "setup", "realtimeInputConfig", "automaticActivityDetection", "disabled")
		}
	} else if output == "image" {
		setIn(&payload, []any{"IMAGE"}, "setup", "generationConfig", "responseModalities")
	}
	return payload, nil
}

func (l *GeminiLM) liveClientContentFromRequest(req *Request) ([]JSONObject, error) {
	if geminiAudioNativeLiveModel(req.Model) {
		return l.realtimeInputPayloads(req)
	}
	if len(req.Messages) == 1 && req.Messages[0].Role == RoleUser {
		allText := true
		for _, p := range req.Messages[0].Parts {
			if _, ok := p.(TextPart); !ok {
				allText = false
			}
		}
		if allText {
			text, _ := partsToText(req.Messages[0].Parts, "", "")
			return []JSONObject{{{"realtimeInput", JSONObject{{"text", text}}}}}, nil
		}
	}
	names := callNames(req.Messages)
	var turns []any
	for _, m := range req.Messages {
		wm, err := l.message(m, names)
		if err != nil {
			return nil, err
		}
		turns = append(turns, wm)
	}
	return []JSONObject{{{"clientContent", JSONObject{{"turns", turns}, {"turnComplete", true}}}}}, nil
}

func (l *GeminiLM) realtimeInputPayloads(req *Request) ([]JSONObject, error) {
	var textPayloads, mediaPayloads []JSONObject
	var contentParts []any
	sentMedia := false
	for _, m := range req.Messages {
		for _, p := range m.Parts {
			switch x := p.(type) {
			case TextPart:
				if x.Text != "" {
					textPayloads = append(textPayloads, JSONObject{{"realtimeInput", JSONObject{{"text", x.Text}}}})
				}
			case AudioPart:
				if x.Data == "" && x.Path == "" {
					continue
				}
				mime := firstOr(x.MediaType, "audio/pcm")
				raw, err := x.Bytes()
				if err != nil {
					return nil, err
				}
				if strings.Contains(mime, "wav") || strings.Contains(mime, "wave") {
					pcm, rate := wavToPCM(raw)
					mediaPayloads = append(mediaPayloads, JSONObject{{"realtimeInput", JSONObject{{"audio", JSONObject{{"mimeType", "audio/pcm;rate=" + strconv.Itoa(rate)}, {"data", base64.StdEncoding.EncodeToString(pcm)}}}}}})
				} else {
					data := x.Data
					if data == "" {
						data = base64.StdEncoding.EncodeToString(raw)
					}
					mediaPayloads = append(mediaPayloads, JSONObject{{"realtimeInput", JSONObject{{"audio", JSONObject{{"mimeType", mime}, {"data", data}}}}}})
				}
				sentMedia = true
			case VideoPart:
				if x.Data == "" && x.Path == "" {
					continue
				}
				data, err := x.Base64()
				if err != nil {
					return nil, err
				}
				mediaPayloads = append(mediaPayloads, JSONObject{{"realtimeInput", JSONObject{{"video", JSONObject{{"mimeType", firstOr(x.MediaType, "video/mp4")}, {"data", data}}}}}})
				sentMedia = true
			case ImagePart, DocumentPart, BinaryPart:
				b, err := l.part(p, nil)
				if err != nil {
					return nil, err
				}
				contentParts = append(contentParts, b)
			}
		}
	}
	var payloads []JSONObject
	if len(contentParts) > 0 {
		payloads = append(payloads, JSONObject{{"clientContent", JSONObject{{"turns", []any{JSONObject{{"role", "user"}, {"parts", contentParts}}}}, {"turnComplete", false}}}})
	}
	payloads = append(payloads, textPayloads...)
	payloads = append(payloads, mediaPayloads...)
	if sentMedia {
		payloads = append([]JSONObject{{{"realtimeInput", JSONObject{{"activityStart", JSONObject{}}}}}}, payloads...)
		payloads = append(payloads, JSONObject{{"realtimeInput", JSONObject{{"activityEnd", JSONObject{}}}}})
	}
	if len(payloads) == 0 {
		payloads = append(payloads, JSONObject{{"realtimeInput", JSONObject{{"text", ""}}}})
	}
	return payloads, nil
}

func (l *GeminiLM) decodeLiveCompletionEvents(raw []byte) ([]StreamEvent, bool, Usage) {
	decoded, err := DecodeJSON(raw)
	if err != nil {
		return nil, false, Usage{}
	}
	payload := wireObj(decoded)
	if payload == nil {
		return nil, false, Usage{}
	}
	if errRaw, ok := payload.Lookup("error"); ok {
		e := wireObj(errRaw)
		code := firstOr(firstStr(e.Get("status"), e.Get("code")), "provider")
		return []StreamEvent{StreamErrorEvent{Error: l.errorDetail(code, wireStr(e.Get("message")))}}, false, Usage{}
	}
	var events []StreamEvent
	if tc := wireObj(payload.Get("toolCall")); tc != nil {
		for idx, raw := range wireList(tc.Get("functionCalls")) {
			if fc := wireObj(raw); fc != nil {
				events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{Input: jsonRaw(objOrEmpty(fc.Get("args"))), PartIndex: idx, ID: firstOr(wireStr(fc.Get("id")), "fc_"+strconv.Itoa(idx)), Name: firstOr(wireStr(fc.Get("name")), "tool")}})
			}
		}
	}
	server := wireObj(payload.Get("serverContent"))
	if server == nil {
		return events, false, l.liveUsage(payload, nil)
	}
	if modelTurn := wireObj(server.Get("modelTurn")); modelTurn != nil {
		for idx, raw := range wireList(modelTurn.Get("parts")) {
			part := wireObj(raw)
			if part == nil {
				continue
			}
			if _, has := part.Lookup("text"); has {
				events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: wireStr(part.Get("text")), PartIndex: idx}})
			} else if fc := wireObj(part.Get("functionCall")); fc != nil {
				events = append(events, StreamDeltaEvent{Delta: ToolCallDelta{Input: jsonRaw(objOrEmpty(fc.Get("args"))), PartIndex: idx, ID: firstOr(wireStr(fc.Get("id")), "fc_0"), Name: firstOr(wireStr(fc.Get("name")), "tool")}})
			} else if inline := wireObj(part.Get("inlineData")); inline != nil {
				mime := wireStr(inline.Get("mimeType"))
				data := wireStr(inline.Get("data"))
				if strings.HasPrefix(mime, "audio/") {
					events = append(events, StreamDeltaEvent{Delta: AudioDelta{Data: S(data), PartIndex: idx, MediaType: mime}})
				} else if strings.HasPrefix(mime, "image/") {
					events = append(events, StreamDeltaEvent{Delta: ImageDelta{Data: S(data), PartIndex: idx, MediaType: mime}})
				}
			}
		}
	}
	if tx := wireObj(server.Get("outputTranscription")); tx != nil && truthy(tx.Get("text")) {
		events = append(events, StreamDeltaEvent{Delta: TextDelta{Text: wireStr(tx.Get("text"))}})
	}
	return events, truthy(server.Get("turnComplete")), l.liveUsage(payload, server)
}

func maxInt(a, b *int) *int {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	if *a > *b {
		return a
	}
	return b
}

func (l *GeminiLM) streamViaLiveCompletion(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		u, err := l.liveURL(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		conn, err := dialWebSocket(ctx, u, nil)
		if err != nil {
			yield(nil, err)
			return
		}
		defer conn.Close()
		setup, err := l.liveSetupPayloadFromRequest(req)
		if err != nil {
			yield(nil, err)
			return
		}
		if !geminiAudioNativeLiveModel(req.Model) {
			g := wireObj(wireObj(setup.Get("setup")).Get("generationConfig"))
			if !g.Has("responseModalities") {
				setIn(&setup, []any{"TEXT"}, "setup", "generationConfig", "responseModalities")
			}
		}
		if err := conn.Send(ctx, mustJSON(setup)); err != nil {
			yield(nil, err)
			return
		}
		if err := l.waitForSetupComplete(ctx, conn); err != nil {
			yield(nil, err)
			return
		}
		frames, err := l.liveClientContentFromRequest(req)
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
		acc := Usage{}
		for {
			raw, err := conn.Recv(ctx)
			if err != nil {
				yield(nil, newError(KindTransport, err.Error()).WithCause(err))
				return
			}
			events, turnComplete, usage := l.decodeLiveCompletionEvents(raw)
			acc = Usage{InputTokens: maxInt(acc.InputTokens, usage.InputTokens), OutputTokens: maxInt(acc.OutputTokens, usage.OutputTokens), TotalTokens: maxInt(acc.TotalTokens, usage.TotalTokens)}
			for _, event := range events {
				if de, ok := event.(StreamDeltaEvent); ok {
					if _, isTool := de.Delta.(ToolCallDelta); isTool {
						sawTool = true
					}
				}
				if _, isErr := event.(StreamErrorEvent); isErr {
					yield(event, nil)
					return
				}
				if !yield(event, nil) {
					return
				}
			}
			if turnComplete {
				finish := FinishStop
				if sawTool {
					finish = FinishToolCall
				}
				u := acc.Normalize()
				yield(StreamEndEvent{FinishReason: finish, Usage: &u}, nil)
				return
			}
		}
	}
}
