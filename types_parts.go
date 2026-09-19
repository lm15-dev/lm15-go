package lm15

import (
	"encoding/base64"
	"mime"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// ContinuationState is opaque provider-owned state needed to continue or
// replay a transcript. Provider is the dialect id (openai, anthropic,
// gemini, xai), never the access door (MAP-7.8).
type ContinuationState struct {
	Provider string
	Kind     string
	Data     JSONObject
}

// Validate checks the state.
func (s ContinuationState) Validate() error {
	if s.Provider == "" {
		return valueErrorf("ContinuationState.provider cannot be empty")
	}
	if s.Kind == "" {
		return valueErrorf("ContinuationState.kind cannot be empty")
	}
	if s.Data == nil {
		return typeErrorf("data must be a JSON object")
	}
	return checkJSONObject(s.Data, "data", true)
}

func validateContinuation(states []ContinuationState) error {
	for _, s := range states {
		if err := s.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ContinuationData returns the data of the first matching state, or nil.
func ContinuationData(states []ContinuationState, provider, kind string) JSONObject {
	for _, s := range states {
		if s.Provider == provider && s.Kind == kind {
			if s.Data == nil {
				return JSONObject{}
			}
			return s.Data
		}
	}
	return nil
}

// ─── Parts ───────────────────────────────────────────────────────────

// Part is the atom of content: a closed sum of eleven variants. Switch on
// the concrete type, or on Type().
type Part interface {
	Type() string
	ContinuationStates() []ContinuationState
	// WithContinuation returns a copy carrying these states.
	WithContinuation([]ContinuationState) Part
	Validate() error
	sealedPart()
}

// TextPart is a block of text.
type TextPart struct {
	Text         string
	Continuation []ContinuationState
}

// ThinkingPart is a model reasoning trace. Hidden thinking is empty Text
// with replay state in Continuation (MAP-7 rule 11).
type ThinkingPart struct {
	Text         string
	Continuation []ContinuationState
}

// RefusalPart is an explicit refusal (non-empty).
type RefusalPart struct {
	Text         string
	Continuation []ContinuationState
}

// CitationPart references source material.
type CitationPart struct {
	URL          string
	Title        string
	Text         string
	Continuation []ContinuationState
}

// Media holds the fields every media part shares: exactly one of Data
// (base64), URL, FileID, Path.
type Media struct {
	MediaType    string
	Data         string
	URL          string
	FileID       string
	Path         string
	Continuation []ContinuationState
}

// ImagePart is an image; Detail is one of low/high/auto.
type ImagePart struct {
	Media
	Detail string
}

// AudioPart is audio content.
type AudioPart struct{ Media }

// VideoPart is video content.
type VideoPart struct{ Media }

// DocumentPart is a document (PDF, ...).
type DocumentPart struct{ Media }

// BinaryPart is arbitrary bytes.
type BinaryPart struct{ Media }

// ToolCallPart is the model's request for an external computation.
type ToolCallPart struct {
	ID           string
	Name         string
	Input        JSONObject
	Continuation []ContinuationState
}

// ToolResultPart is the result of an external computation, sent back.
type ToolResultPart struct {
	ID           string
	Content      []Part
	Name         string
	IsError      bool
	Continuation []ContinuationState
}

func (TextPart) Type() string       { return PartTypeText }
func (ThinkingPart) Type() string   { return PartTypeThinking }
func (RefusalPart) Type() string    { return PartTypeRefusal }
func (CitationPart) Type() string   { return PartTypeCitation }
func (ImagePart) Type() string      { return PartTypeImage }
func (AudioPart) Type() string      { return PartTypeAudio }
func (VideoPart) Type() string      { return PartTypeVideo }
func (DocumentPart) Type() string   { return PartTypeDocument }
func (BinaryPart) Type() string     { return PartTypeBinary }
func (ToolCallPart) Type() string   { return PartTypeToolCall }
func (ToolResultPart) Type() string { return PartTypeToolResult }

func (TextPart) sealedPart()       {}
func (ThinkingPart) sealedPart()   {}
func (RefusalPart) sealedPart()    {}
func (CitationPart) sealedPart()   {}
func (ImagePart) sealedPart()      {}
func (AudioPart) sealedPart()      {}
func (VideoPart) sealedPart()      {}
func (DocumentPart) sealedPart()   {}
func (BinaryPart) sealedPart()     {}
func (ToolCallPart) sealedPart()   {}
func (ToolResultPart) sealedPart() {}

func (p TextPart) ContinuationStates() []ContinuationState       { return p.Continuation }
func (p ThinkingPart) ContinuationStates() []ContinuationState   { return p.Continuation }
func (p RefusalPart) ContinuationStates() []ContinuationState    { return p.Continuation }
func (p CitationPart) ContinuationStates() []ContinuationState   { return p.Continuation }
func (p ImagePart) ContinuationStates() []ContinuationState      { return p.Continuation }
func (p AudioPart) ContinuationStates() []ContinuationState      { return p.Continuation }
func (p VideoPart) ContinuationStates() []ContinuationState      { return p.Continuation }
func (p DocumentPart) ContinuationStates() []ContinuationState   { return p.Continuation }
func (p BinaryPart) ContinuationStates() []ContinuationState     { return p.Continuation }
func (p ToolCallPart) ContinuationStates() []ContinuationState   { return p.Continuation }
func (p ToolResultPart) ContinuationStates() []ContinuationState { return p.Continuation }

func (p TextPart) WithContinuation(c []ContinuationState) Part       { p.Continuation = c; return p }
func (p ThinkingPart) WithContinuation(c []ContinuationState) Part   { p.Continuation = c; return p }
func (p RefusalPart) WithContinuation(c []ContinuationState) Part    { p.Continuation = c; return p }
func (p CitationPart) WithContinuation(c []ContinuationState) Part   { p.Continuation = c; return p }
func (p ImagePart) WithContinuation(c []ContinuationState) Part      { p.Continuation = c; return p }
func (p AudioPart) WithContinuation(c []ContinuationState) Part      { p.Continuation = c; return p }
func (p VideoPart) WithContinuation(c []ContinuationState) Part      { p.Continuation = c; return p }
func (p DocumentPart) WithContinuation(c []ContinuationState) Part   { p.Continuation = c; return p }
func (p BinaryPart) WithContinuation(c []ContinuationState) Part     { p.Continuation = c; return p }
func (p ToolCallPart) WithContinuation(c []ContinuationState) Part   { p.Continuation = c; return p }
func (p ToolResultPart) WithContinuation(c []ContinuationState) Part { p.Continuation = c; return p }

// Validate implements Part.
func (p TextPart) Validate() error { return validateContinuation(p.Continuation) }

// Validate implements Part.
func (p ThinkingPart) Validate() error { return validateContinuation(p.Continuation) }

// Validate implements Part (INV-016).
func (p RefusalPart) Validate() error {
	if p.Text == "" {
		return valueErrorf("RefusalPart.text cannot be empty")
	}
	return validateContinuation(p.Continuation)
}

// Validate implements Part (INV-017).
func (p CitationPart) Validate() error {
	if p.URL == "" && p.Title == "" && p.Text == "" {
		return valueErrorf("CitationPart requires at least one of url, title, or text")
	}
	return validateContinuation(p.Continuation)
}

var base64Re = regexp.MustCompile(`^[A-Za-z0-9+/]*={0,2}$`)

// base64Payload strips a data-URI prefix and embedded whitespace (INV-012).
func base64Payload(partType, data string) (string, error) {
	if data == "" {
		return "", valueErrorf("%s.data cannot be empty", partType)
	}
	if strings.HasPrefix(data, "data:") && strings.Contains(data, ";base64,") {
		data = data[strings.Index(data, ";base64,")+len(";base64,"):]
	}
	if strings.ContainsAny(data, " \n\r\t\v\f") {
		data = strings.Join(strings.Fields(data), "")
	}
	return data, nil
}

func validateBase64(partType, data string) error {
	payload, err := base64Payload(partType, data)
	if err != nil {
		return err
	}
	if len(payload)%4 != 0 || !base64Re.MatchString(payload) {
		return valueErrorf("%s.data must be a valid base64 string", partType)
	}
	return nil
}

func decodeBase64Payload(partType, data string) ([]byte, error) {
	payload, err := base64Payload(partType, data)
	if err != nil {
		return nil, err
	}
	out, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, valueErrorf("%s.data must be a valid base64 string", partType)
	}
	return out, nil
}

func (m Media) validate(partType string) error {
	if m.MediaType == "" {
		return valueErrorf("%s requires media_type", partType)
	}
	count := 0
	for _, v := range []string{m.Data, m.URL, m.FileID, m.Path} {
		if v != "" {
			count++
		}
	}
	if count != 1 {
		return valueErrorf("%s requires exactly one of data, url, file_id, or path", partType)
	}
	if m.Data != "" {
		if err := validateBase64(partType, m.Data); err != nil {
			return err
		}
	}
	return validateContinuation(m.Continuation)
}

// Bytes returns the media bytes: inline data decoded, or the path read now.
func (m Media) Bytes() ([]byte, error) {
	if m.Data != "" {
		return decodeBase64Payload("media", m.Data)
	}
	if m.Path != "" {
		return os.ReadFile(m.Path)
	}
	return nil, valueErrorf("media part has no inline data or path; fetch url/file_id-addressed media before decoding")
}

// Base64 returns the media bytes as base64 (inline data verbatim, a path read now).
func (m Media) Base64() (string, error) {
	if m.Data != "" {
		return m.Data, nil
	}
	if m.Path != "" {
		raw, err := os.ReadFile(m.Path)
		if err != nil {
			return "", err
		}
		return base64.StdEncoding.EncodeToString(raw), nil
	}
	return "", valueErrorf("media part has no inline data or path")
}

// Validate implements Part.
func (p ImagePart) Validate() error {
	if err := p.Media.validate("ImagePart"); err != nil {
		return err
	}
	if p.Detail != "" && p.Detail != "low" && p.Detail != "high" && p.Detail != "auto" {
		return valueErrorf("unsupported ImagePart.detail: %s", p.Detail)
	}
	return nil
}

// Validate implements Part.
func (p AudioPart) Validate() error { return p.Media.validate("AudioPart") }

// Validate implements Part.
func (p VideoPart) Validate() error { return p.Media.validate("VideoPart") }

// Validate implements Part.
func (p DocumentPart) Validate() error { return p.Media.validate("DocumentPart") }

// Validate implements Part.
func (p BinaryPart) Validate() error { return p.Media.validate("BinaryPart") }

// Validate implements Part.
func (p ToolCallPart) Validate() error {
	if p.ID == "" {
		return valueErrorf("ToolCallPart.id cannot be empty")
	}
	if p.Name == "" {
		return valueErrorf("ToolCallPart.name cannot be empty")
	}
	if p.Input == nil {
		return typeErrorf("input must be a JSON object")
	}
	if err := checkJSONObject(p.Input, "input", true); err != nil {
		return err
	}
	return validateContinuation(p.Continuation)
}

// Validate implements Part (INV-013, INV-014).
func (p ToolResultPart) Validate() error {
	if p.ID == "" {
		return valueErrorf("ToolResultPart.id cannot be empty")
	}
	if len(p.Content) == 0 {
		return valueErrorf("ToolResultPart requires content")
	}
	for _, c := range p.Content {
		if c == nil {
			return typeErrorf("ToolResultPart.content must contain Part objects")
		}
		if isToolResultForbidden(c) {
			return typeErrorf("ToolResultPart.content cannot contain tool calls, nested tool results, thinking parts, or refusals")
		}
		if err := c.Validate(); err != nil {
			return err
		}
	}
	return validateContinuation(p.Continuation)
}

func isToolResultForbidden(p Part) bool {
	switch p.(type) {
	case ToolCallPart, ToolResultPart, ThinkingPart, RefusalPart:
		return true
	}
	return false
}

func isPromptForbidden(p Part) bool {
	switch p.(type) {
	case ToolCallPart, ToolResultPart, ThinkingPart, RefusalPart, CitationPart:
		return true
	}
	return false
}

// IsMediaPart reports whether p carries bytes or an address rather than words.
func IsMediaPart(p Part) bool {
	switch p.(type) {
	case ImagePart, AudioPart, VideoPart, DocumentPart, BinaryPart:
		return true
	}
	return false
}

// MediaOf returns the Media of a media part.
func MediaOf(p Part) (Media, bool) {
	switch x := p.(type) {
	case ImagePart:
		return x.Media, true
	case AudioPart:
		return x.Media, true
	case VideoPart:
		return x.Media, true
	case DocumentPart:
		return x.Media, true
	case BinaryPart:
		return x.Media, true
	}
	return Media{}, false
}

// ─── Factories ───────────────────────────────────────────────────────

// Text creates a text part.
func Text(content string) TextPart { return TextPart{Text: content} }

// Thinking creates a thinking part.
func Thinking(content string) ThinkingPart { return ThinkingPart{Text: content} }

// Refusal creates a refusal part.
func Refusal(content string) RefusalPart { return RefusalPart{Text: content} }

// Citation creates a citation part.
func Citation(url, title, text string) CitationPart {
	return CitationPart{URL: url, Title: title, Text: text}
}

// ToolCall creates a tool call part.
func ToolCall(id, name string, input JSONObject) ToolCallPart {
	if input == nil {
		input = JSONObject{}
	}
	return ToolCallPart{ID: id, Name: name, Input: input}
}

// ToolResult creates a tool result carrying one text part.
func ToolResult(id, output string) ToolResultPart {
	return ToolResultPart{ID: id, Content: []Part{Text(output)}}
}

// ToolResultParts creates a tool result from parts.
func ToolResultParts(id string, content ...Part) ToolResultPart {
	return ToolResultPart{ID: id, Content: content}
}

// ToolError creates an error tool result.
func ToolError(id, output string) ToolResultPart {
	return ToolResultPart{ID: id, Content: []Part{Text(output)}, IsError: true}
}

// MediaOption configures a media factory.
type MediaOption func(*Media)

// WithURL addresses the media by URL.
func WithURL(url string) MediaOption { return func(m *Media) { m.URL = url } }

// WithFileID addresses the media by a provider file id.
func WithFileID(id string) MediaOption { return func(m *Media) { m.FileID = id } }

// WithPath addresses the media by a local path (read at request time).
func WithPath(path string) MediaOption { return func(m *Media) { m.Path = path } }

// WithData embeds raw bytes (base64-encoded).
func WithData(data []byte) MediaOption {
	return func(m *Media) { m.Data = base64.StdEncoding.EncodeToString(data) }
}

// WithBase64 embeds an already-encoded base64 string (or data URI).
func WithBase64(data string) MediaOption { return func(m *Media) { m.Data = data } }

// WithMediaType sets the MIME type.
func WithMediaType(mediaType string) MediaOption { return func(m *Media) { m.MediaType = mediaType } }

func buildMedia(defaultType string, opts []MediaOption) Media {
	var m Media
	for _, o := range opts {
		o(&m)
	}
	if m.MediaType == "" && m.Path != "" {
		if guessed := mime.TypeByExtension(strings.ToLower(filepath.Ext(m.Path))); guessed != "" {
			m.MediaType, _, _ = strings.Cut(guessed, ";")
		}
	}
	if m.MediaType == "" {
		m.MediaType = defaultType
	}
	return m
}

// Image creates an image part (default media type image/png).
func Image(opts ...MediaOption) ImagePart { return ImagePart{Media: buildMedia("image/png", opts)} }

// Audio creates an audio part (default media type audio/wav).
func Audio(opts ...MediaOption) AudioPart { return AudioPart{Media: buildMedia("audio/wav", opts)} }

// Video creates a video part (default media type video/mp4).
func Video(opts ...MediaOption) VideoPart { return VideoPart{Media: buildMedia("video/mp4", opts)} }

// Document creates a document part (default media type application/pdf).
func Document(opts ...MediaOption) DocumentPart {
	return DocumentPart{Media: buildMedia("application/pdf", opts)}
}

// Binary creates a binary part (default media type application/octet-stream).
func Binary(opts ...MediaOption) BinaryPart {
	return BinaryPart{Media: buildMedia("application/octet-stream", opts)}
}

// partText returns the text of a text-bearing part ("" otherwise).
func partText(p Part) string {
	switch x := p.(type) {
	case TextPart:
		return x.Text
	case ThinkingPart:
		return x.Text
	case RefusalPart:
		return x.Text
	case CitationPart:
		return x.Text
	}
	return ""
}
