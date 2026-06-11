package lm15

import "encoding/json"

// ─── AudioFormat ─────────────────────────────────────────────────────

type AudioFormat struct {
	Encoding   string `json:"encoding"`
	SampleRate int64  `json:"sample_rate"`
	Channels   int64  `json:"channels"`
}

func (af AudioFormat) toMap() jmap {
	m := jmap{"encoding": af.Encoding, "sample_rate": jsonInt(af.SampleRate)}
	if af.Channels != 0 {
		m["channels"] = jsonInt(af.Channels)
	}
	return m
}

func audioFormatFromMap(m jmap) (AudioFormat, error) {
	encoding, err := reqString(m, "encoding")
	if err != nil {
		return AudioFormat{}, err
	}
	if !inVocab(AudioEncodings, encoding) {
		return AudioFormat{}, valueErr("unsupported audio encoding: %s", encoding)
	}
	rate, err := optInt(m, "sample_rate")
	if err != nil {
		return AudioFormat{}, err
	}
	if rate == nil {
		return AudioFormat{}, keyErr("sample_rate")
	}
	if *rate <= 0 {
		return AudioFormat{}, valueErr("sample_rate must be positive")
	}
	channels, err := optInt(m, "channels")
	if err != nil {
		return AudioFormat{}, err
	}
	ch := int64(1)
	if channels != nil {
		ch = *channels
	}
	if ch <= 0 {
		return AudioFormat{}, valueErr("channels must be positive")
	}
	return AudioFormat{Encoding: encoding, SampleRate: *rate, Channels: ch}, nil
}

func (af AudioFormat) MarshalJSON() ([]byte, error) { return json.Marshal(af.toMap()) }

func (af *AudioFormat) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := audioFormatFromMap(m)
	if err != nil {
		return err
	}
	*af = v
	return nil
}

// ─── LiveConfig ──────────────────────────────────────────────────────

type LiveConfig struct {
	Model        string       `json:"model"`
	System       any          `json:"system"` // string or []Part
	Tools        []Tool       `json:"tools"`
	Voice        *string      `json:"voice"`
	InputFormat  *AudioFormat `json:"input_format"`
	OutputFormat *AudioFormat `json:"output_format"`
	Extensions   jmap         `json:"extensions"`
}

func (lc LiveConfig) toMap() (jmap, error) {
	m := jmap{"model": lc.Model}
	system, err := systemToJSON(lc.System)
	if err != nil {
		return nil, err
	}
	putOpt(m, "system", system)
	putOpt(m, "tools", toolsToJSON(lc.Tools))
	if lc.Voice != nil && *lc.Voice != "" {
		m["voice"] = *lc.Voice
	}
	if lc.InputFormat != nil {
		putOpt(m, "input_format", lc.InputFormat.toMap())
	}
	if lc.OutputFormat != nil {
		putOpt(m, "output_format", lc.OutputFormat.toMap())
	}
	putOpt(m, "extensions", lc.Extensions)
	return m, nil
}

func liveConfigFromMap(m jmap) (LiveConfig, error) {
	var lc LiveConfig
	var err error
	if lc.Model, err = reqString(m, "model"); err != nil {
		return lc, err
	}
	if lc.Model == "" {
		return lc, valueErr("live config model must be non-empty")
	}
	if lc.System, err = systemFromJSON(m); err != nil {
		return lc, err
	}
	if lc.Tools, err = toolsFromJSON(m); err != nil {
		return lc, err
	}
	if lc.Voice, err = optStringPtr(m, "voice"); err != nil {
		return lc, err
	}
	if obj, err := optObj(m, "input_format"); err != nil {
		return lc, err
	} else if obj != nil {
		af, err := audioFormatFromMap(obj)
		if err != nil {
			return lc, err
		}
		lc.InputFormat = &af
	}
	if obj, err := optObj(m, "output_format"); err != nil {
		return lc, err
	} else if obj != nil {
		af, err := audioFormatFromMap(obj)
		if err != nil {
			return lc, err
		}
		lc.OutputFormat = &af
	}
	if lc.Extensions, err = optObj(m, "extensions"); err != nil {
		return lc, err
	}
	return lc, nil
}

func (lc LiveConfig) MarshalJSON() ([]byte, error) {
	m, err := lc.toMap()
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

func (lc *LiveConfig) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := liveConfigFromMap(m)
	if err != nil {
		return err
	}
	*lc = v
	return nil
}

// ─── Live client events ──────────────────────────────────────────────

// LiveClientEvent is the tagged union of client→server live events.
type LiveClientEvent interface {
	LiveClientEventType() string
	toMap() jmap
	isLiveClientEvent()
}

type LiveClientTurnEvent struct {
	Parts        []Part `json:"parts"`
	TurnComplete bool   `json:"turn_complete"`
}

func (e LiveClientTurnEvent) LiveClientEventType() string { return "turn" }
func (e LiveClientTurnEvent) isLiveClientEvent()          {}
func (e LiveClientTurnEvent) toMap() jmap {
	parts := make([]any, len(e.Parts))
	for i, p := range e.Parts {
		parts[i] = p.toMap()
	}
	// Serialized without cleaning: turn_complete is emitted even when false.
	return jmap{"type": "turn", "parts": parts, "turn_complete": e.TurnComplete}
}

type LiveClientAudioEvent struct {
	Data      string `json:"data"`
	MediaType string `json:"media_type"`
}

func (e LiveClientAudioEvent) LiveClientEventType() string { return "audio" }
func (e LiveClientAudioEvent) isLiveClientEvent()          {}
func (e LiveClientAudioEvent) toMap() jmap {
	return jmap{"type": "audio", "data": e.Data, "media_type": e.MediaType}
}

type LiveClientImageEvent struct {
	Data      string `json:"data"`
	MediaType string `json:"media_type"`
}

func (e LiveClientImageEvent) LiveClientEventType() string { return "image" }
func (e LiveClientImageEvent) isLiveClientEvent()          {}
func (e LiveClientImageEvent) toMap() jmap {
	return jmap{"type": "image", "data": e.Data, "media_type": e.MediaType}
}

type LiveClientTextEvent struct {
	Text string `json:"text"`
}

func (e LiveClientTextEvent) LiveClientEventType() string { return "text" }
func (e LiveClientTextEvent) isLiveClientEvent()          {}
func (e LiveClientTextEvent) toMap() jmap {
	return jmap{"type": "text", "text": e.Text}
}

type LiveClientToolResultEvent struct {
	ID      string `json:"id"`
	Content []Part `json:"content"`
}

func (e LiveClientToolResultEvent) LiveClientEventType() string { return "tool_result" }
func (e LiveClientToolResultEvent) isLiveClientEvent()          {}
func (e LiveClientToolResultEvent) toMap() jmap {
	content := make([]any, len(e.Content))
	for i, p := range e.Content {
		content[i] = p.toMap()
	}
	return jmap{"type": "tool_result", "id": e.ID, "content": content}
}

type LiveClientInterruptEvent struct{}

func (e LiveClientInterruptEvent) LiveClientEventType() string { return "interrupt" }
func (e LiveClientInterruptEvent) isLiveClientEvent()          {}
func (e LiveClientInterruptEvent) toMap() jmap                 { return jmap{"type": "interrupt"} }

type LiveClientEndAudioEvent struct{}

func (e LiveClientEndAudioEvent) LiveClientEventType() string { return "end_audio" }
func (e LiveClientEndAudioEvent) isLiveClientEvent()          {}
func (e LiveClientEndAudioEvent) toMap() jmap                 { return jmap{"type": "end_audio"} }

// LiveClientEventFromMap dispatches on "type".
func LiveClientEventFromMap(m jmap) (LiveClientEvent, error) {
	t, err := reqString(m, "type")
	if err != nil {
		return nil, err
	}
	switch t {
	case "turn":
		raw, err := optList(m, "parts")
		if err != nil {
			return nil, err
		}
		var parts []Part
		for _, item := range raw {
			obj, ok := item.(jmap)
			if !ok {
				return nil, typeErr("turn parts must be objects")
			}
			p, err := PartFromMap(obj)
			if err != nil {
				return nil, err
			}
			parts = append(parts, p)
		}
		if len(parts) == 0 {
			return nil, valueErr("turn parts must be non-empty")
		}
		complete, err := optBool(m, "turn_complete", true)
		if err != nil {
			return nil, err
		}
		return LiveClientTurnEvent{Parts: parts, TurnComplete: complete}, nil
	case "audio":
		data, err := reqString(m, "data")
		if err != nil {
			return nil, err
		}
		mediaType, err := optString(m, "media_type", "audio/pcm;rate=16000")
		if err != nil {
			return nil, err
		}
		return LiveClientAudioEvent{Data: data, MediaType: mediaType}, nil
	case "image":
		data, err := reqString(m, "data")
		if err != nil {
			return nil, err
		}
		mediaType, err := optString(m, "media_type", "image/jpeg")
		if err != nil {
			return nil, err
		}
		return LiveClientImageEvent{Data: data, MediaType: mediaType}, nil
	case "text":
		text, err := optString(m, "text", "")
		if err != nil {
			return nil, err
		}
		return LiveClientTextEvent{Text: text}, nil
	case "tool_result":
		id, err := reqString(m, "id")
		if err != nil {
			return nil, err
		}
		raw, err := optList(m, "content")
		if err != nil {
			return nil, err
		}
		var content []Part
		for _, item := range raw {
			obj, ok := item.(jmap)
			if !ok {
				return nil, typeErr("tool_result content entries must be objects")
			}
			p, err := PartFromMap(obj)
			if err != nil {
				return nil, err
			}
			content = append(content, p)
		}
		if len(content) == 0 {
			return nil, valueErr("tool_result content must be non-empty")
		}
		return LiveClientToolResultEvent{ID: id, Content: content}, nil
	case "interrupt":
		return LiveClientInterruptEvent{}, nil
	case "end_audio":
		return LiveClientEndAudioEvent{}, nil
	}
	return nil, valueErr("unsupported live client event type: %s", t)
}

// ─── Live server events ──────────────────────────────────────────────

// LiveServerEvent is the tagged union of server→client live events.
type LiveServerEvent interface {
	LiveServerEventType() string
	toMap() jmap
	isLiveServerEvent()
}

type LiveServerAudioEvent struct {
	Data      string  `json:"data"`
	MediaType *string `json:"media_type"`
}

func (e LiveServerAudioEvent) LiveServerEventType() string { return "audio" }
func (e LiveServerAudioEvent) isLiveServerEvent()          {}
func (e LiveServerAudioEvent) toMap() jmap {
	m := jmap{"type": "audio", "data": e.Data}
	if e.MediaType != nil && *e.MediaType != "" {
		m["media_type"] = *e.MediaType
	}
	return m
}

type LiveServerTextEvent struct {
	Text string `json:"text"`
}

func (e LiveServerTextEvent) LiveServerEventType() string { return "text" }
func (e LiveServerTextEvent) isLiveServerEvent()          {}
func (e LiveServerTextEvent) toMap() jmap {
	return jmap{"type": "text", "text": e.Text}
}

type LiveServerToolCallEvent struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input jmap   `json:"input"`
}

func (e LiveServerToolCallEvent) LiveServerEventType() string { return "tool_call" }
func (e LiveServerToolCallEvent) isLiveServerEvent()          {}
func (e LiveServerToolCallEvent) toMap() jmap {
	input := e.Input
	if input == nil {
		input = jmap{}
	}
	return jmap{"type": "tool_call", "id": e.ID, "name": e.Name, "input": input}
}

type LiveServerToolCallDeltaEvent struct {
	InputDelta string  `json:"input_delta"`
	ID         *string `json:"id"`
	Name       *string `json:"name"`
}

func (e LiveServerToolCallDeltaEvent) LiveServerEventType() string { return "tool_call_delta" }
func (e LiveServerToolCallDeltaEvent) isLiveServerEvent()          {}
func (e LiveServerToolCallDeltaEvent) toMap() jmap {
	m := jmap{"type": "tool_call_delta"}
	putOpt(m, "input_delta", e.InputDelta)
	if e.ID != nil && *e.ID != "" {
		m["id"] = *e.ID
	}
	if e.Name != nil && *e.Name != "" {
		m["name"] = *e.Name
	}
	return m
}

type LiveServerInterruptedEvent struct{}

func (e LiveServerInterruptedEvent) LiveServerEventType() string { return "interrupted" }
func (e LiveServerInterruptedEvent) isLiveServerEvent()          {}
func (e LiveServerInterruptedEvent) toMap() jmap                 { return jmap{"type": "interrupted"} }

type LiveServerTurnEndEvent struct {
	Usage Usage `json:"usage"`
}

func (e LiveServerTurnEndEvent) LiveServerEventType() string { return "turn_end" }
func (e LiveServerTurnEndEvent) isLiveServerEvent()          {}
func (e LiveServerTurnEndEvent) toMap() jmap {
	m := jmap{"type": "turn_end"}
	putOpt(m, "usage", e.Usage.toMap())
	return m
}

type LiveServerErrorEvent struct {
	Error ErrorDetail `json:"error"`
}

func (e LiveServerErrorEvent) LiveServerEventType() string { return "error" }
func (e LiveServerErrorEvent) isLiveServerEvent()          {}
func (e LiveServerErrorEvent) toMap() jmap {
	return jmap{"type": "error", "error": e.Error.toMap()}
}

// LiveServerEventFromMap dispatches on "type".
func LiveServerEventFromMap(m jmap) (LiveServerEvent, error) {
	t, err := reqString(m, "type")
	if err != nil {
		return nil, err
	}
	switch t {
	case "audio":
		data, err := reqString(m, "data")
		if err != nil {
			return nil, err
		}
		mediaType, err := optStringPtr(m, "media_type")
		if err != nil {
			return nil, err
		}
		return LiveServerAudioEvent{Data: data, MediaType: mediaType}, nil
	case "text":
		text, err := optString(m, "text", "")
		if err != nil {
			return nil, err
		}
		return LiveServerTextEvent{Text: text}, nil
	case "tool_call":
		id, err := reqString(m, "id")
		if err != nil {
			return nil, err
		}
		name, err := reqString(m, "name")
		if err != nil {
			return nil, err
		}
		input, err := optObj(m, "input")
		if err != nil {
			return nil, err
		}
		if input == nil {
			input = jmap{}
		}
		return LiveServerToolCallEvent{ID: id, Name: name, Input: input}, nil
	case "tool_call_delta":
		inputDelta, err := optString(m, "input_delta", "")
		if err != nil {
			return nil, err
		}
		id, err := optStringPtr(m, "id")
		if err != nil {
			return nil, err
		}
		name, err := optStringPtr(m, "name")
		if err != nil {
			return nil, err
		}
		return LiveServerToolCallDeltaEvent{InputDelta: inputDelta, ID: id, Name: name}, nil
	case "interrupted":
		return LiveServerInterruptedEvent{}, nil
	case "turn_end":
		var usage Usage
		if obj, err := optObj(m, "usage"); err != nil {
			return nil, err
		} else if obj != nil {
			if usage, err = usageFromMap(obj); err != nil {
				return nil, err
			}
		}
		return LiveServerTurnEndEvent{Usage: usage}, nil
	case "error":
		obj, err := optObj(m, "error")
		if err != nil {
			return nil, err
		}
		if obj == nil {
			return nil, keyErr("error")
		}
		detail, err := errorDetailFromMap(obj)
		if err != nil {
			return nil, err
		}
		return LiveServerErrorEvent{Error: detail}, nil
	}
	return nil, valueErr("unsupported live server event type: %s", t)
}
