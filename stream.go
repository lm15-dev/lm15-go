package lm15

import "encoding/json"

// ─── Usage ───────────────────────────────────────────────────────────

// Usage counters are *int64: nil means "not reported by the provider",
// distinct from a reported 0.
type Usage struct {
	InputTokens       *int64 `json:"input_tokens"`
	OutputTokens      *int64 `json:"output_tokens"`
	TotalTokens       *int64 `json:"total_tokens"`
	CacheReadTokens   *int64 `json:"cache_read_tokens"`
	CacheWriteTokens  *int64 `json:"cache_write_tokens"`
	ReasoningTokens   *int64 `json:"reasoning_tokens"`
	InputAudioTokens  *int64 `json:"input_audio_tokens"`
	OutputAudioTokens *int64 `json:"output_audio_tokens"`
}

func (u Usage) toMap() jmap {
	m := jmap{}
	put := func(k string, v *int64) {
		if v != nil {
			m[k] = jsonInt(*v)
		}
	}
	put("input_tokens", u.InputTokens)
	put("output_tokens", u.OutputTokens)
	put("total_tokens", u.TotalTokens)
	put("cache_read_tokens", u.CacheReadTokens)
	put("cache_write_tokens", u.CacheWriteTokens)
	put("reasoning_tokens", u.ReasoningTokens)
	put("input_audio_tokens", u.InputAudioTokens)
	put("output_audio_tokens", u.OutputAudioTokens)
	return m
}

func usageFromMap(m jmap) (Usage, error) {
	var u Usage
	for _, f := range []struct {
		key string
		dst **int64
	}{
		{"input_tokens", &u.InputTokens},
		{"output_tokens", &u.OutputTokens},
		{"total_tokens", &u.TotalTokens},
		{"cache_read_tokens", &u.CacheReadTokens},
		{"cache_write_tokens", &u.CacheWriteTokens},
		{"reasoning_tokens", &u.ReasoningTokens},
		{"input_audio_tokens", &u.InputAudioTokens},
		{"output_audio_tokens", &u.OutputAudioTokens},
	} {
		v, err := optInt(m, f.key)
		if err != nil {
			return Usage{}, err
		}
		if v != nil && *v < 0 {
			return Usage{}, valueErr("%s must be >= 0", f.key)
		}
		*f.dst = v
	}
	return u, nil
}

func (u Usage) MarshalJSON() ([]byte, error) { return json.Marshal(u.toMap()) }

func (u *Usage) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := usageFromMap(m)
	if err != nil {
		return err
	}
	*u = v
	return nil
}

// ─── ErrorDetail ─────────────────────────────────────────────────────

type ErrorDetail struct {
	Code         string  `json:"code"`
	Message      string  `json:"message"`
	ProviderCode *string `json:"provider_code"`
}

func (e ErrorDetail) toMap() jmap {
	m := jmap{"code": e.Code}
	putOpt(m, "message", e.Message)
	if e.ProviderCode != nil && *e.ProviderCode != "" {
		m["provider_code"] = *e.ProviderCode
	}
	return m
}

func errorDetailFromMap(m jmap) (ErrorDetail, error) {
	code, err := reqString(m, "code")
	if err != nil {
		return ErrorDetail{}, err
	}
	if !inVocab(ErrorCodes, code) {
		return ErrorDetail{}, valueErr("unsupported error code: %s", code)
	}
	message, err := optString(m, "message", "")
	if err != nil {
		return ErrorDetail{}, err
	}
	providerCode, err := optStringPtr(m, "provider_code")
	if err != nil {
		return ErrorDetail{}, err
	}
	return ErrorDetail{Code: code, Message: message, ProviderCode: providerCode}, nil
}

func (e ErrorDetail) MarshalJSON() ([]byte, error) { return json.Marshal(e.toMap()) }

func (e *ErrorDetail) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := errorDetailFromMap(m)
	if err != nil {
		return err
	}
	*e = v
	return nil
}

// ─── Deltas ──────────────────────────────────────────────────────────

// Delta is the tagged union of streaming deltas, dispatching on "type".
type Delta interface {
	DeltaType() string
	toMap() jmap
	isDelta()
}

type TextDelta struct {
	Text      string `json:"text"`
	PartIndex int64  `json:"part_index"`
}

func (d TextDelta) DeltaType() string { return "text" }
func (d TextDelta) isDelta()          {}
func (d TextDelta) toMap() jmap {
	return jmap{"type": "text", "text": d.Text, "part_index": jsonInt(d.PartIndex)}
}

type ThinkingDelta struct {
	Text      string `json:"text"`
	PartIndex int64  `json:"part_index"`
}

func (d ThinkingDelta) DeltaType() string { return "thinking" }
func (d ThinkingDelta) isDelta()          {}
func (d ThinkingDelta) toMap() jmap {
	return jmap{"type": "thinking", "text": d.Text, "part_index": jsonInt(d.PartIndex)}
}

type AudioDelta struct {
	Data      *string `json:"data"`
	URL       *string `json:"url"`
	FileID    *string `json:"file_id"`
	PartIndex int64   `json:"part_index"`
	MediaType *string `json:"media_type"`
}

func (d AudioDelta) DeltaType() string { return "audio" }
func (d AudioDelta) isDelta()          {}
func (d AudioDelta) toMap() jmap {
	return mediaDeltaToMap("audio", d.Data, d.URL, d.FileID, d.PartIndex, d.MediaType)
}

type ImageDelta struct {
	Data      *string `json:"data"`
	URL       *string `json:"url"`
	FileID    *string `json:"file_id"`
	PartIndex int64   `json:"part_index"`
	MediaType *string `json:"media_type"`
}

func (d ImageDelta) DeltaType() string { return "image" }
func (d ImageDelta) isDelta()          {}
func (d ImageDelta) toMap() jmap {
	return mediaDeltaToMap("image", d.Data, d.URL, d.FileID, d.PartIndex, d.MediaType)
}

func mediaDeltaToMap(typ string, data, url, fileID *string, partIndex int64, mediaType *string) jmap {
	m := jmap{"type": typ, "part_index": jsonInt(partIndex)}
	if data != nil {
		m["data"] = *data
	}
	if url != nil {
		m["url"] = *url
	}
	if fileID != nil {
		m["file_id"] = *fileID
	}
	if mediaType != nil {
		m["media_type"] = *mediaType
	}
	return m
}

type ToolCallDelta struct {
	Input     string  `json:"input"`
	PartIndex int64   `json:"part_index"`
	ID        *string `json:"id"`
	Name      *string `json:"name"`
}

func (d ToolCallDelta) DeltaType() string { return "tool_call" }
func (d ToolCallDelta) isDelta()          {}
func (d ToolCallDelta) toMap() jmap {
	m := jmap{"type": "tool_call", "input": d.Input, "part_index": jsonInt(d.PartIndex)}
	if d.ID != nil {
		m["id"] = *d.ID
	}
	if d.Name != nil {
		m["name"] = *d.Name
	}
	return m
}

type CitationDelta struct {
	Text      *string `json:"text"`
	URL       *string `json:"url"`
	Title     *string `json:"title"`
	PartIndex int64   `json:"part_index"`
}

func (d CitationDelta) DeltaType() string { return "citation" }
func (d CitationDelta) isDelta()          {}
func (d CitationDelta) toMap() jmap {
	m := jmap{"type": "citation", "part_index": jsonInt(d.PartIndex)}
	if d.Text != nil {
		m["text"] = *d.Text
	}
	if d.URL != nil {
		m["url"] = *d.URL
	}
	if d.Title != nil {
		m["title"] = *d.Title
	}
	return m
}

type ContinuationDelta struct {
	Provider  string `json:"provider"`
	Kind      string `json:"kind"`
	Data      jmap   `json:"data"`
	PartIndex *int64 `json:"part_index"`
}

func (d ContinuationDelta) DeltaType() string { return "continuation" }
func (d ContinuationDelta) isDelta()          {}
func (d ContinuationDelta) toMap() jmap {
	data := d.Data
	if data == nil {
		data = jmap{}
	}
	m := jmap{"type": "continuation", "provider": d.Provider, "kind": d.Kind, "data": data}
	if d.PartIndex != nil {
		m["part_index"] = jsonInt(*d.PartIndex)
	}
	return m
}

// ToState converts a ContinuationDelta to its ContinuationState.
func (d ContinuationDelta) ToState() ContinuationState {
	return ContinuationState{Provider: d.Provider, Kind: d.Kind, Data: d.Data}
}

// DeltaFromMap deserializes a Delta, dispatching on "type".
func DeltaFromMap(m jmap) (Delta, error) {
	t, err := reqString(m, "type")
	if err != nil {
		return nil, err
	}
	partIndexPtr, err := optInt(m, "part_index")
	if err != nil {
		return nil, err
	}
	partIndex := int64(0)
	if partIndexPtr != nil {
		partIndex = *partIndexPtr
	}
	if t != "continuation" && partIndex < 0 {
		return nil, valueErr("part_index must be >= 0")
	}
	switch t {
	case "text", "thinking":
		text, err := optString(m, "text", "")
		if err != nil {
			return nil, err
		}
		if t == "text" {
			return TextDelta{Text: text, PartIndex: partIndex}, nil
		}
		return ThinkingDelta{Text: text, PartIndex: partIndex}, nil
	case "audio", "image":
		data, err := optStringPtr(m, "data")
		if err != nil {
			return nil, err
		}
		url, err := optStringPtr(m, "url")
		if err != nil {
			return nil, err
		}
		fileID, err := optStringPtr(m, "file_id")
		if err != nil {
			return nil, err
		}
		mediaType, err := optStringPtr(m, "media_type")
		if err != nil {
			return nil, err
		}
		sources := 0
		for _, s := range []*string{data, url, fileID} {
			if s != nil {
				sources++
			}
		}
		if sources > 1 {
			return nil, valueErr("at most one of data/url/file_id")
		}
		if t == "audio" {
			return AudioDelta{Data: data, URL: url, FileID: fileID, PartIndex: partIndex, MediaType: mediaType}, nil
		}
		return ImageDelta{Data: data, URL: url, FileID: fileID, PartIndex: partIndex, MediaType: mediaType}, nil
	case "tool_call":
		input, err := optString(m, "input", "")
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
		return ToolCallDelta{Input: input, PartIndex: partIndex, ID: id, Name: name}, nil
	case "citation":
		text, err := optStringPtr(m, "text")
		if err != nil {
			return nil, err
		}
		url, err := optStringPtr(m, "url")
		if err != nil {
			return nil, err
		}
		title, err := optStringPtr(m, "title")
		if err != nil {
			return nil, err
		}
		if text == nil && url == nil && title == nil {
			return nil, valueErr("citation delta requires at least one of text/url/title")
		}
		return CitationDelta{Text: text, URL: url, Title: title, PartIndex: partIndex}, nil
	case "continuation":
		provider, err := reqString(m, "provider")
		if err != nil {
			return nil, err
		}
		kind, err := reqString(m, "kind")
		if err != nil {
			return nil, err
		}
		if provider == "" || kind == "" {
			return nil, valueErr("continuation delta provider/kind must be non-empty")
		}
		data, err := optObj(m, "data")
		if err != nil {
			return nil, err
		}
		if data == nil {
			data = jmap{}
		}
		return ContinuationDelta{Provider: provider, Kind: kind, Data: data, PartIndex: partIndexPtr}, nil
	}
	return nil, valueErr("unsupported delta type: %s", t)
}

func (d TextDelta) MarshalJSON() ([]byte, error)         { return json.Marshal(d.toMap()) }
func (d ThinkingDelta) MarshalJSON() ([]byte, error)     { return json.Marshal(d.toMap()) }
func (d AudioDelta) MarshalJSON() ([]byte, error)        { return json.Marshal(d.toMap()) }
func (d ImageDelta) MarshalJSON() ([]byte, error)        { return json.Marshal(d.toMap()) }
func (d ToolCallDelta) MarshalJSON() ([]byte, error)     { return json.Marshal(d.toMap()) }
func (d CitationDelta) MarshalJSON() ([]byte, error)     { return json.Marshal(d.toMap()) }
func (d ContinuationDelta) MarshalJSON() ([]byte, error) { return json.Marshal(d.toMap()) }

// ─── Stream events ───────────────────────────────────────────────────

// StreamEvent is the tagged union of stream events, dispatching on "type".
type StreamEvent interface {
	StreamEventType() string
	toMap() jmap
	isStreamEvent()
}

type StreamStartEvent struct {
	ID    *string `json:"id"`
	Model *string `json:"model"`
}

func (e StreamStartEvent) StreamEventType() string { return "start" }
func (e StreamStartEvent) isStreamEvent()          {}
func (e StreamStartEvent) toMap() jmap {
	m := jmap{"type": "start"}
	if e.ID != nil && *e.ID != "" {
		m["id"] = *e.ID
	}
	if e.Model != nil && *e.Model != "" {
		m["model"] = *e.Model
	}
	return m
}

type StreamDeltaEvent struct {
	Delta Delta `json:"delta"`
}

func (e StreamDeltaEvent) StreamEventType() string { return "delta" }
func (e StreamDeltaEvent) isStreamEvent()          {}
func (e StreamDeltaEvent) toMap() jmap {
	return jmap{"type": "delta", "delta": e.Delta.toMap()}
}

type StreamEndEvent struct {
	FinishReason *string `json:"finish_reason"`
	Usage        *Usage  `json:"usage"`
	ProviderData jmap    `json:"provider_data"`
}

func (e StreamEndEvent) StreamEventType() string { return "end" }
func (e StreamEndEvent) isStreamEvent()          {}
func (e StreamEndEvent) toMap() jmap {
	m := jmap{"type": "end"}
	if e.FinishReason != nil && *e.FinishReason != "" {
		m["finish_reason"] = *e.FinishReason
	}
	if e.Usage != nil {
		putOpt(m, "usage", e.Usage.toMap())
	}
	putOpt(m, "provider_data", e.ProviderData)
	return m
}

type StreamErrorEvent struct {
	Error ErrorDetail `json:"error"`
}

func (e StreamErrorEvent) StreamEventType() string { return "error" }
func (e StreamErrorEvent) isStreamEvent()          {}
func (e StreamErrorEvent) toMap() jmap {
	return jmap{"type": "error", "error": e.Error.toMap()}
}

// StreamEventFromMap deserializes a StreamEvent, dispatching on "type".
func StreamEventFromMap(m jmap) (StreamEvent, error) {
	t, err := reqString(m, "type")
	if err != nil {
		return nil, err
	}
	switch t {
	case "start":
		id, err := optStringPtr(m, "id")
		if err != nil {
			return nil, err
		}
		model, err := optStringPtr(m, "model")
		if err != nil {
			return nil, err
		}
		return StreamStartEvent{ID: id, Model: model}, nil
	case "delta":
		obj, err := optObj(m, "delta")
		if err != nil {
			return nil, err
		}
		if obj == nil {
			return nil, keyErr("delta")
		}
		delta, err := DeltaFromMap(obj)
		if err != nil {
			return nil, err
		}
		return StreamDeltaEvent{Delta: delta}, nil
	case "end":
		finish, err := optStringPtr(m, "finish_reason")
		if err != nil {
			return nil, err
		}
		if finish != nil && !inVocab(FinishReasons, *finish) {
			return nil, valueErr("unsupported finish reason: %s", *finish)
		}
		var usage *Usage
		if obj, err := optObj(m, "usage"); err == nil && obj != nil {
			u, err := usageFromMap(obj)
			if err != nil {
				return nil, err
			}
			usage = &u
		} else if err != nil {
			return nil, err
		}
		providerData, err := optObj(m, "provider_data")
		if err != nil {
			return nil, err
		}
		return StreamEndEvent{FinishReason: finish, Usage: usage, ProviderData: providerData}, nil
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
		return StreamErrorEvent{Error: detail}, nil
	}
	return nil, valueErr("unsupported stream event type: %s", t)
}

func (e StreamStartEvent) MarshalJSON() ([]byte, error) { return json.Marshal(e.toMap()) }
func (e StreamDeltaEvent) MarshalJSON() ([]byte, error) { return json.Marshal(e.toMap()) }
func (e StreamEndEvent) MarshalJSON() ([]byte, error)   { return json.Marshal(e.toMap()) }
func (e StreamErrorEvent) MarshalJSON() ([]byte, error) { return json.Marshal(e.toMap()) }
