package lm15

import "encoding/json"

// ─── ContinuationState ───────────────────────────────────────────────

type ContinuationState struct {
	Provider string `json:"provider"`
	Kind     string `json:"kind"`
	Data     jmap   `json:"data"`
}

func (c ContinuationState) toMap() jmap {
	data := c.Data
	if data == nil {
		data = jmap{}
	}
	return jmap{"provider": c.Provider, "kind": c.Kind, "data": data}
}

func continuationStateFromMap(m jmap) (ContinuationState, error) {
	provider, err := reqString(m, "provider")
	if err != nil {
		return ContinuationState{}, err
	}
	kind, err := reqString(m, "kind")
	if err != nil {
		return ContinuationState{}, err
	}
	if provider == "" || kind == "" {
		return ContinuationState{}, valueErr("continuation provider/kind must be non-empty")
	}
	data, err := optObj(m, "data")
	if err != nil {
		return ContinuationState{}, err
	}
	if data == nil {
		data = jmap{}
	}
	return ContinuationState{Provider: provider, Kind: kind, Data: data}, nil
}

func (c ContinuationState) MarshalJSON() ([]byte, error) { return json.Marshal(c.toMap()) }

func (c *ContinuationState) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := continuationStateFromMap(m)
	if err != nil {
		return err
	}
	*c = v
	return nil
}

func continuationToJSON(states []ContinuationState) []any {
	if len(states) == 0 {
		return nil
	}
	out := make([]any, len(states))
	for i, s := range states {
		out[i] = s.toMap()
	}
	return out
}

func continuationFromJSON(m jmap) ([]ContinuationState, error) {
	raw, err := optList(m, "continuation")
	if err != nil {
		return nil, typeErr("continuation must be a list")
	}
	var out []ContinuationState
	for _, item := range raw {
		obj, ok := item.(jmap)
		if !ok {
			return nil, typeErr("continuation entries must be objects")
		}
		s, err := continuationStateFromMap(obj)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// ─── Part ────────────────────────────────────────────────────────────

// Part is the canonical tagged union of message content. Concrete structs
// carry custom JSON marshalling dispatching on the "type" discriminator.
type Part interface {
	PartType() string
	Continuations() []ContinuationState
	toMap() jmap
	isPart()
}

type TextPart struct {
	Text         string              `json:"text"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p TextPart) PartType() string                   { return "text" }
func (p TextPart) Continuations() []ContinuationState { return p.Continuation }
func (p TextPart) isPart()                            {}
func (p TextPart) toMap() jmap {
	m := jmap{"type": "text", "text": p.Text}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

type ThinkingPart struct {
	Text         string              `json:"text"`
	Redacted     bool                `json:"redacted"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p ThinkingPart) PartType() string                   { return "thinking" }
func (p ThinkingPart) Continuations() []ContinuationState { return p.Continuation }
func (p ThinkingPart) isPart()                            {}
func (p ThinkingPart) toMap() jmap {
	m := jmap{"type": "thinking", "text": p.Text}
	if p.Redacted {
		m["redacted"] = true
	}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

type RefusalPart struct {
	Text         string              `json:"text"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p RefusalPart) PartType() string                   { return "refusal" }
func (p RefusalPart) Continuations() []ContinuationState { return p.Continuation }
func (p RefusalPart) isPart()                            {}
func (p RefusalPart) toMap() jmap {
	m := jmap{"type": "refusal", "text": p.Text}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

type CitationPart struct {
	URL          *string             `json:"url"`
	Title        *string             `json:"title"`
	Text         *string             `json:"text"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p CitationPart) PartType() string                   { return "citation" }
func (p CitationPart) Continuations() []ContinuationState { return p.Continuation }
func (p CitationPart) isPart()                            {}
func (p CitationPart) toMap() jmap {
	m := jmap{"type": "citation"}
	if p.Text != nil {
		m["text"] = *p.Text
	}
	if p.URL != nil {
		m["url"] = *p.URL
	}
	if p.Title != nil {
		m["title"] = *p.Title
	}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

// mediaPart holds the shared fields of the five media part types.
type mediaPart struct {
	MediaType    string              `json:"media_type"`
	Data         *string             `json:"data"`
	URL          *string             `json:"url"`
	FileID       *string             `json:"file_id"`
	Path         *string             `json:"path"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p mediaPart) Continuations() []ContinuationState { return p.Continuation }

func (p mediaPart) mediaToMap(typ string) jmap {
	m := jmap{"type": typ, "media_type": p.MediaType}
	if p.Data != nil {
		m["data"] = *p.Data
	}
	if p.URL != nil {
		m["url"] = *p.URL
	}
	if p.FileID != nil {
		m["file_id"] = *p.FileID
	}
	if p.Path != nil {
		m["path"] = *p.Path
	}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

type ImagePart struct {
	mediaPart
	Detail *string `json:"detail"`
}

func (p ImagePart) PartType() string { return "image" }
func (p ImagePart) isPart()          {}
func (p ImagePart) toMap() jmap {
	m := p.mediaToMap("image")
	if p.Detail != nil {
		m["detail"] = *p.Detail
	}
	return m
}

type AudioPart struct{ mediaPart }

func (p AudioPart) PartType() string { return "audio" }
func (p AudioPart) isPart()          {}
func (p AudioPart) toMap() jmap      { return p.mediaToMap("audio") }

type VideoPart struct{ mediaPart }

func (p VideoPart) PartType() string { return "video" }
func (p VideoPart) isPart()          {}
func (p VideoPart) toMap() jmap      { return p.mediaToMap("video") }

type DocumentPart struct{ mediaPart }

func (p DocumentPart) PartType() string { return "document" }
func (p DocumentPart) isPart()          {}
func (p DocumentPart) toMap() jmap      { return p.mediaToMap("document") }

type BinaryPart struct{ mediaPart }

func (p BinaryPart) PartType() string { return "binary" }
func (p BinaryPart) isPart()          {}
func (p BinaryPart) toMap() jmap      { return p.mediaToMap("binary") }

type ToolCallPart struct {
	ID           string              `json:"id"`
	Name         string              `json:"name"`
	Input        jmap                `json:"input"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p ToolCallPart) PartType() string                   { return "tool_call" }
func (p ToolCallPart) Continuations() []ContinuationState { return p.Continuation }
func (p ToolCallPart) isPart()                            {}
func (p ToolCallPart) toMap() jmap {
	input := p.Input
	if input == nil {
		input = jmap{}
	}
	m := jmap{"type": "tool_call", "id": p.ID, "name": p.Name, "input": input}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

type ToolResultPart struct {
	ID           string              `json:"id"`
	Content      []Part              `json:"content"`
	Name         *string             `json:"name"`
	IsError      bool                `json:"is_error"`
	Continuation []ContinuationState `json:"continuation"`
}

func (p ToolResultPart) PartType() string                   { return "tool_result" }
func (p ToolResultPart) Continuations() []ContinuationState { return p.Continuation }
func (p ToolResultPart) isPart()                            {}
func (p ToolResultPart) toMap() jmap {
	m := jmap{"type": "tool_result", "id": p.ID}
	if p.Name != nil {
		m["name"] = *p.Name
	}
	content := make([]any, len(p.Content))
	for i, c := range p.Content {
		content[i] = c.toMap()
	}
	m["content"] = content // required-with-shape: emitted even when []
	if p.IsError {
		m["is_error"] = true
	}
	putOpt(m, "continuation", continuationToJSON(p.Continuation))
	return m
}

// ─── Part serde ──────────────────────────────────────────────────────

// PartToMap serializes a Part to its canonical JSON object form.
func PartToMap(p Part) jmap { return p.toMap() }

func mediaFromMap(m jmap) (mediaPart, error) {
	mt, err := optString(m, "media_type", "")
	if err != nil {
		return mediaPart{}, err
	}
	if mt == "" {
		return mediaPart{}, valueErr("media_type must be a non-empty string")
	}
	data, err := optStringPtr(m, "data")
	if err != nil {
		return mediaPart{}, err
	}
	url, err := optStringPtr(m, "url")
	if err != nil {
		return mediaPart{}, err
	}
	fileID, err := optStringPtr(m, "file_id")
	if err != nil {
		return mediaPart{}, err
	}
	path, err := optStringPtr(m, "path")
	if err != nil {
		return mediaPart{}, err
	}
	sources := 0
	for _, s := range []*string{data, url, fileID, path} {
		if s != nil {
			sources++
		}
	}
	if sources != 1 {
		return mediaPart{}, valueErr("exactly one of data/url/file_id/path is required")
	}
	cont, err := continuationFromJSON(m)
	if err != nil {
		return mediaPart{}, err
	}
	return mediaPart{MediaType: mt, Data: data, URL: url, FileID: fileID, Path: path, Continuation: cont}, nil
}

// PartFromMap deserializes a Part from canonical JSON, dispatching on "type".
func PartFromMap(m jmap) (Part, error) {
	t, err := reqString(m, "type")
	if err != nil {
		return nil, err
	}
	cont, err := continuationFromJSON(m)
	if err != nil {
		return nil, err
	}
	switch t {
	case "text":
		text, err := optString(m, "text", "")
		if err != nil {
			return nil, err
		}
		return TextPart{Text: text, Continuation: cont}, nil
	case "thinking":
		text, err := optString(m, "text", "")
		if err != nil {
			return nil, err
		}
		redacted, err := optBool(m, "redacted", false)
		if err != nil {
			return nil, err
		}
		return ThinkingPart{Text: text, Redacted: redacted, Continuation: cont}, nil
	case "refusal":
		text, err := optString(m, "text", "")
		if err != nil {
			return nil, err
		}
		if text == "" {
			return nil, valueErr("refusal text must be non-empty")
		}
		return RefusalPart{Text: text, Continuation: cont}, nil
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
			return nil, valueErr("citation requires at least one of url/title/text")
		}
		return CitationPart{Text: text, URL: url, Title: title, Continuation: cont}, nil
	case "image":
		media, err := mediaFromMap(m)
		if err != nil {
			return nil, err
		}
		detail, err := optStringPtr(m, "detail")
		if err != nil {
			return nil, err
		}
		if detail != nil && !inVocab([]string{"low", "high", "auto"}, *detail) {
			return nil, valueErr("detail must be one of low/high/auto")
		}
		return ImagePart{mediaPart: media, Detail: detail}, nil
	case "audio":
		media, err := mediaFromMap(m)
		if err != nil {
			return nil, err
		}
		return AudioPart{media}, nil
	case "video":
		media, err := mediaFromMap(m)
		if err != nil {
			return nil, err
		}
		return VideoPart{media}, nil
	case "document":
		media, err := mediaFromMap(m)
		if err != nil {
			return nil, err
		}
		return DocumentPart{media}, nil
	case "binary":
		media, err := mediaFromMap(m)
		if err != nil {
			return nil, err
		}
		return BinaryPart{media}, nil
	case "tool_call":
		id, err := reqString(m, "id")
		if err != nil {
			return nil, err
		}
		name, err := reqString(m, "name")
		if err != nil {
			return nil, err
		}
		if id == "" || name == "" {
			return nil, valueErr("tool_call id and name must be non-empty")
		}
		input, err := optObj(m, "input")
		if err != nil {
			return nil, err
		}
		if input == nil {
			input = jmap{}
		}
		return ToolCallPart{ID: id, Name: name, Input: input, Continuation: cont}, nil
	case "tool_result":
		id, err := reqString(m, "id")
		if err != nil {
			return nil, err
		}
		if id == "" {
			return nil, valueErr("tool_result id must be non-empty")
		}
		name, err := optStringPtr(m, "name")
		if err != nil {
			return nil, err
		}
		isErr, err := optBool(m, "is_error", false)
		if err != nil {
			return nil, err
		}
		rawContent, err := optList(m, "content")
		if err != nil {
			return nil, err
		}
		var content []Part
		for _, c := range rawContent {
			obj, ok := c.(jmap)
			if !ok {
				return nil, typeErr("tool_result content entries must be objects")
			}
			part, err := PartFromMap(obj)
			if err != nil {
				return nil, err
			}
			content = append(content, part)
		}
		if len(content) == 0 {
			return nil, valueErr("tool_result content must be non-empty")
		}
		return ToolResultPart{ID: id, Content: content, Name: name, IsError: isErr, Continuation: cont}, nil
	}
	return nil, valueErr("unsupported part type: %s", t)
}

// MarshalPart serializes any Part to canonical JSON bytes.
func MarshalPart(p Part) ([]byte, error) { return json.Marshal(p.toMap()) }

// UnmarshalPart deserializes canonical JSON bytes into a Part.
func UnmarshalPart(data []byte) (Part, error) {
	m, err := decodeMap(data)
	if err != nil {
		return nil, err
	}
	return PartFromMap(m)
}

func (p TextPart) MarshalJSON() ([]byte, error)       { return MarshalPart(p) }
func (p ThinkingPart) MarshalJSON() ([]byte, error)   { return MarshalPart(p) }
func (p RefusalPart) MarshalJSON() ([]byte, error)    { return MarshalPart(p) }
func (p CitationPart) MarshalJSON() ([]byte, error)   { return MarshalPart(p) }
func (p ImagePart) MarshalJSON() ([]byte, error)      { return MarshalPart(p) }
func (p AudioPart) MarshalJSON() ([]byte, error)      { return MarshalPart(p) }
func (p VideoPart) MarshalJSON() ([]byte, error)      { return MarshalPart(p) }
func (p DocumentPart) MarshalJSON() ([]byte, error)   { return MarshalPart(p) }
func (p BinaryPart) MarshalJSON() ([]byte, error)     { return MarshalPart(p) }
func (p ToolCallPart) MarshalJSON() ([]byte, error)   { return MarshalPart(p) }
func (p ToolResultPart) MarshalJSON() ([]byte, error) { return MarshalPart(p) }

// ─── Message ─────────────────────────────────────────────────────────

type Message struct {
	Role         string              `json:"role"`
	Parts        []Part              `json:"parts"`
	Continuation []ContinuationState `json:"continuation"`
}

func (msg Message) toMap() jmap {
	parts := make([]any, len(msg.Parts))
	for i, p := range msg.Parts {
		parts[i] = p.toMap()
	}
	m := jmap{"role": msg.Role, "parts": parts}
	putOpt(m, "continuation", continuationToJSON(msg.Continuation))
	return m
}

func messageFromMap(m jmap) (Message, error) {
	role, err := reqString(m, "role")
	if err != nil {
		return Message{}, err
	}
	if !inVocab(RoleValues, role) {
		return Message{}, valueErr("unsupported role: %s", role)
	}
	rawParts, err := optList(m, "parts")
	if err != nil {
		return Message{}, err
	}
	var parts []Part
	for _, p := range rawParts {
		obj, ok := p.(jmap)
		if !ok {
			return Message{}, typeErr("message parts must be objects")
		}
		part, err := PartFromMap(obj)
		if err != nil {
			return Message{}, err
		}
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		return Message{}, valueErr("message for role '%s' has no parts", role)
	}
	cont, err := continuationFromJSON(m)
	if err != nil {
		return Message{}, err
	}
	return Message{Role: role, Parts: parts, Continuation: cont}, nil
}

func (msg Message) MarshalJSON() ([]byte, error) { return json.Marshal(msg.toMap()) }

func (msg *Message) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := messageFromMap(m)
	if err != nil {
		return err
	}
	*msg = v
	return nil
}
