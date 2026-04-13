// Package lm15 provides a universal interface for OpenAI, Anthropic, and Gemini.
// Zero dependencies beyond the Go standard library.
package lm15

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
)

// ── Enums ──────────────────────────────────────────────────────────

// Role of a message participant.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// PartType identifies the kind of content in a Part.
type PartType string

const (
	PartText       PartType = "text"
	PartImage      PartType = "image"
	PartAudio      PartType = "audio"
	PartVideo      PartType = "video"
	PartDocument   PartType = "document"
	PartToolCall   PartType = "tool_call"
	PartToolResult PartType = "tool_result"
	PartThinking   PartType = "thinking"
	PartRefusal    PartType = "refusal"
	PartCitation   PartType = "citation"
)

// FinishReason indicates why the model stopped generating.
type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCall      FinishReason = "tool_call"
	FinishContentFilter FinishReason = "content_filter"
	FinishError         FinishReason = "error"
)

// ── DataSource ─────────────────────────────────────────────────────

// DataSource describes where media data comes from.
type DataSource struct {
	Type      string `json:"type"`                 // "base64", "url", "file"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`       // base64-encoded
	URL       string `json:"url,omitempty"`
	FileID    string `json:"file_id,omitempty"`
	Detail    string `json:"detail,omitempty"`      // "low", "high", "auto"
}

// Bytes decodes base64 data. Only valid for Type == "base64".
func (ds DataSource) Bytes() ([]byte, error) {
	if ds.Type != "base64" || ds.Data == "" {
		return nil, fmt.Errorf("DataSource(type=%q) has no inline bytes", ds.Type)
	}
	return base64.StdEncoding.DecodeString(ds.Data)
}

// ── Part ───────────────────────────────────────────────────────────

// Part is a single piece of content in a message.
type Part struct {
	Type     PartType         `json:"type"`
	Text     string           `json:"text,omitempty"`
	Source   *DataSource      `json:"source,omitempty"`
	ID       string           `json:"id,omitempty"`
	Name     string           `json:"name,omitempty"`
	Input    map[string]any   `json:"input,omitempty"`
	Content  []Part           `json:"content,omitempty"`
	IsError  *bool            `json:"is_error,omitempty"`
	Redacted *bool            `json:"redacted,omitempty"`
	Summary  string           `json:"summary,omitempty"`
	URL      string           `json:"url,omitempty"`
	Title    string           `json:"title,omitempty"`
	Metadata map[string]any   `json:"metadata,omitempty"`
}

// TextPart creates a text part.
func TextPart(text string) Part {
	return Part{Type: PartText, Text: text}
}

// ThinkingPart creates a thinking/reasoning part.
func ThinkingPart(text string) Part {
	return Part{Type: PartThinking, Text: text}
}

// RefusalPart creates a refusal part.
func RefusalPart(text string) Part {
	return Part{Type: PartRefusal, Text: text}
}

// CitationPart creates a citation part.
func CitationPart(text, url, title string) Part {
	return Part{Type: PartCitation, Text: text, URL: url, Title: title}
}

// ImagePart creates an image part from a DataSource.
func ImagePart(source DataSource) Part {
	return Part{Type: PartImage, Source: &source}
}

// AudioPart creates an audio part from a DataSource.
func AudioPart(source DataSource) Part {
	return Part{Type: PartAudio, Source: &source}
}

// VideoPart creates a video part from a DataSource.
func VideoPart(source DataSource) Part {
	return Part{Type: PartVideo, Source: &source}
}

// DocumentPart creates a document part from a DataSource.
func DocumentPart(source DataSource) Part {
	return Part{Type: PartDocument, Source: &source}
}

// ToolCallPart creates a tool call part.
func ToolCallPart(id, name string, input map[string]any) Part {
	return Part{Type: PartToolCall, ID: id, Name: name, Input: input}
}

// ToolResultPart creates a tool result part.
func ToolResultPart(id string, content []Part, name string) Part {
	return Part{Type: PartToolResult, ID: id, Content: content, Name: name}
}

// ImageURL creates an image part from a URL.
func ImageURL(url string) Part {
	return ImagePart(DataSource{Type: "url", URL: url, MediaType: "image/png"})
}

// ImageBase64 creates an image part from base64 data.
func ImageBase64(data, mediaType string) Part {
	return ImagePart(DataSource{Type: "base64", Data: data, MediaType: mediaType})
}

// ImageBytes creates an image part from raw bytes.
func ImageBytes(data []byte, mediaType string) Part {
	return ImageBase64(base64.StdEncoding.EncodeToString(data), mediaType)
}

// AudioURL creates an audio part from a URL.
func AudioURL(url string) Part {
	return AudioPart(DataSource{Type: "url", URL: url, MediaType: "audio/wav"})
}

// AudioBase64 creates an audio part from base64 data.
func AudioBase64(data, mediaType string) Part {
	return AudioPart(DataSource{Type: "base64", Data: data, MediaType: mediaType})
}

// DocumentURL creates a document part from a URL.
func DocumentURL(url string) Part {
	return DocumentPart(DataSource{Type: "url", URL: url, MediaType: "application/pdf"})
}

// ── Tool ───────────────────────────────────────────────────────────

// Tool defines a function or builtin tool the model can call.
type Tool struct {
	Type        string         `json:"type"`                  // "function" or "builtin"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Fn          func(map[string]any) (any, error) `json:"-"` // auto-execute function
}

// FunctionTool creates a function tool definition.
func FunctionTool(name, description string, parameters map[string]any) Tool {
	if parameters == nil {
		parameters = map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return Tool{Type: "function", Name: name, Description: description, Parameters: parameters}
}

// BuiltinTool creates a builtin tool reference (e.g., "web_search").
func BuiltinTool(name string) Tool {
	return Tool{Type: "builtin", Name: name}
}

// ToolCallInfo describes a pending tool call for the on_tool_call callback.
type ToolCallInfo struct {
	ID    string
	Name  string
	Input map[string]any
}

// ── Config ─────────────────────────────────────────────────────────

// Config holds generation parameters.
type Config struct {
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	Reasoning      map[string]any  `json:"reasoning,omitempty"`
	Provider       map[string]any  `json:"provider,omitempty"`
	ResponseFormat map[string]any  `json:"response_format,omitempty"`
}

// ── Message ────────────────────────────────────────────────────────

// Message is a single turn in a conversation.
type Message struct {
	Role  Role   `json:"role"`
	Parts []Part `json:"parts"`
	Name  string `json:"name,omitempty"`
}

// UserMessage creates a user message with text content.
func UserMessage(text string) Message {
	return Message{Role: RoleUser, Parts: []Part{TextPart(text)}}
}

// AssistantMessage creates an assistant message with text content.
func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Parts: []Part{TextPart(text)}}
}

// ToolResultMessage creates a tool result message from a map of {callID: result}.
func ToolResultMessage(results map[string]string) Message {
	var parts []Part
	for callID, result := range results {
		parts = append(parts, ToolResultPart(callID, []Part{TextPart(result)}, ""))
	}
	return Message{Role: RoleTool, Parts: parts}
}

// ── Canonical JSON serialization ───────────────────────────────────

// PartFromDict creates a Part from a canonical JSON map.
func PartFromDict(d map[string]any) Part {
	pt := toString(d["type"])
	switch PartType(pt) {
	case PartText:
		return Part{Type: PartText, Text: toString(d["text"])}
	case PartThinking:
		p := Part{Type: PartThinking, Text: toString(d["text"]), Summary: toString(d["summary"])}
		if v, ok := d["redacted"].(bool); ok {
			p.Redacted = &v
		}
		return p
	case PartRefusal:
		return Part{Type: PartRefusal, Text: toString(d["text"])}
	case PartCitation:
		return Part{Type: PartCitation, Text: toString(d["text"]), URL: toString(d["url"]), Title: toString(d["title"])}
	case PartImage, PartAudio, PartVideo, PartDocument:
		var source *DataSource
		if src, ok := d["source"].(map[string]any); ok {
			source = &DataSource{
				Type: toString(src["type"]), URL: toString(src["url"]),
				Data: toString(src["data"]), MediaType: toString(src["media_type"]),
				FileID: toString(src["file_id"]), Detail: toString(src["detail"]),
			}
		}
		return Part{Type: PartType(pt), Source: source}
	case PartToolCall:
		input, _ := d["arguments"].(map[string]any)
		return Part{Type: PartToolCall, ID: toString(d["id"]), Name: toString(d["name"]), Input: input}
	case PartToolResult:
		var content []Part
		switch c := d["content"].(type) {
		case string:
			if c != "" {
				content = []Part{TextPart(c)}
			}
		case []any:
			for _, item := range c {
				if m, ok := item.(map[string]any); ok {
					content = append(content, PartFromDict(m))
				}
			}
		}
		return Part{Type: PartToolResult, ID: toString(d["id"]), Name: toString(d["name"]), Content: content}
	}
	return Part{Type: PartText, Text: toString(d["text"])}
}

// MessageFromDict creates a Message from a canonical JSON map.
func MessageFromDict(d map[string]any) Message {
	role := Role(toString(d["role"]))
	var parts []Part
	if ps, ok := d["parts"].([]any); ok {
		for _, item := range ps {
			if m, ok := item.(map[string]any); ok {
				parts = append(parts, PartFromDict(m))
			}
		}
	}
	return Message{Role: role, Parts: parts, Name: toString(d["name"])}
}

// MessagesFromJSON converts a JSON array of canonical message dicts to Messages.
func MessagesFromJSON(data []any) []Message {
	var msgs []Message
	for _, item := range data {
		if m, ok := item.(map[string]any); ok {
			msgs = append(msgs, MessageFromDict(m))
		}
	}
	return msgs
}

func toString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// ── Request / Response ─────────────────────────────────────────────

// LMRequest is the normalized request sent to any provider.
type LMRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	System   string    `json:"system,omitempty"`
	Tools    []Tool    `json:"tools,omitempty"`
	Config   Config    `json:"config,omitempty"`
}

// Usage reports token counts.
type Usage struct {
	InputTokens      int  `json:"input_tokens"`
	OutputTokens     int  `json:"output_tokens"`
	TotalTokens      int  `json:"total_tokens"`
	CacheReadTokens  *int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens *int `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  *int `json:"reasoning_tokens,omitempty"`
	InputAudioTokens *int `json:"input_audio_tokens,omitempty"`
	OutputAudioTokens *int `json:"output_audio_tokens,omitempty"`
}

// LMResponse is the normalized response from any provider.
type LMResponse struct {
	ID           string         `json:"id"`
	Model        string         `json:"model"`
	Message      Message        `json:"message"`
	FinishReason FinishReason   `json:"finish_reason"`
	Usage        Usage          `json:"usage"`
	Provider     map[string]any `json:"provider,omitempty"`
}

// Text returns concatenated text from the response, or empty string if none.
func (r LMResponse) Text() string {
	var texts []string
	for _, p := range r.Message.Parts {
		if p.Type == PartText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	if len(texts) == 0 {
		return ""
	}
	result := texts[0]
	for _, t := range texts[1:] {
		result += "\n" + t
	}
	return result
}

// Thinking returns concatenated thinking parts, or empty string if none.
func (r LMResponse) Thinking() string {
	var texts []string
	for _, p := range r.Message.Parts {
		if p.Type == PartThinking && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	if len(texts) == 0 {
		return ""
	}
	result := texts[0]
	for _, t := range texts[1:] {
		result += "\n" + t
	}
	return result
}

// ToolCalls returns all tool call parts from the response.
func (r LMResponse) ToolCalls() []Part {
	var calls []Part
	for _, p := range r.Message.Parts {
		if p.Type == PartToolCall {
			calls = append(calls, p)
		}
	}
	return calls
}

// Image returns the first image part, or nil.
func (r LMResponse) Image() *Part {
	for _, p := range r.Message.Parts {
		if p.Type == PartImage {
			return &p
		}
	}
	return nil
}

// Audio returns the first audio part, or nil.
func (r LMResponse) Audio() *Part {
	for _, p := range r.Message.Parts {
		if p.Type == PartAudio {
			return &p
		}
	}
	return nil
}

// JSON parses the response text as JSON into the given target.
func (r LMResponse) JSON(target any) error {
	text := r.Text()
	if text == "" {
		return errors.New("response contains no text")
	}
	return json.Unmarshal([]byte(text), target)
}

// ── Streaming ──────────────────────────────────────────────────────

// ErrorInfo describes a stream error.
type ErrorInfo struct {
	Code         string `json:"code"`
	Message      string `json:"message"`
	ProviderCode string `json:"provider_code,omitempty"`
}

// PartDelta is a partial update during streaming.
type PartDelta struct {
	Type  string `json:"type"`            // "text", "tool_call", "thinking", "audio"
	Text  string `json:"text,omitempty"`
	Data  string `json:"data,omitempty"`
	Input string `json:"input,omitempty"`
}

// StreamEvent is a single event from a streaming response.
type StreamEvent struct {
	Type         string         `json:"type"` // "start", "delta", "part_start", "part_end", "end", "error"
	ID           string         `json:"id,omitempty"`
	Model        string         `json:"model,omitempty"`
	PartIndex    *int           `json:"part_index,omitempty"`
	Delta        *PartDelta     `json:"delta,omitempty"`
	DeltaRaw     map[string]any `json:"-"` // for tool_call deltas with structured fields
	PartType     string         `json:"part_type,omitempty"`
	FinishReason FinishReason   `json:"finish_reason,omitempty"`
	Usage        *Usage         `json:"usage,omitempty"`
	Error        *ErrorInfo     `json:"error,omitempty"`
}

// StreamChunk is a higher-level event emitted by Result.
type StreamChunk struct {
	Type     string       // "text", "thinking", "audio", "tool_call", "tool_result", "finished"
	Text     string
	Name     string
	Input    map[string]any
	Response *LMResponse
}

// ── Live Sessions ──────────────────────────────────────────────────

// AudioFormat describes audio encoding parameters.
type AudioFormat struct {
	Encoding   string `json:"encoding"`    // "pcm16", "opus", "mp3", "aac"
	SampleRate int    `json:"sample_rate"`
	Channels   int    `json:"channels"`
}

// LiveConfig configures a live (real-time) session.
type LiveConfig struct {
	Model        string         `json:"model"`
	System       string         `json:"system,omitempty"`
	Tools        []Tool         `json:"tools,omitempty"`
	Voice        string         `json:"voice,omitempty"`
	InputFormat  *AudioFormat   `json:"input_format,omitempty"`
	OutputFormat *AudioFormat   `json:"output_format,omitempty"`
	Provider     map[string]any `json:"provider,omitempty"`
}

// LiveClientEvent is sent from client to server in a live session.
type LiveClientEvent struct {
	Type    string `json:"type"` // "audio", "video", "text", "tool_result", "interrupt", "end_audio"
	Data    string `json:"data,omitempty"`
	Text    string `json:"text,omitempty"`
	ID      string `json:"id,omitempty"`
	Content []Part `json:"content,omitempty"`
}

// LiveServerEvent is received from server in a live session.
type LiveServerEvent struct {
	Type  string         `json:"type"` // "audio", "text", "tool_call", "interrupted", "turn_end", "error"
	Data  string         `json:"data,omitempty"`
	Text  string         `json:"text,omitempty"`
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
	Usage *Usage         `json:"usage,omitempty"`
	Error *ErrorInfo     `json:"error,omitempty"`
}

// ── Auxiliary Request/Response ──────────────────────────────────────

// EmbeddingRequest asks for text embeddings.
type EmbeddingRequest struct {
	Model    string         `json:"model"`
	Inputs   []string       `json:"inputs"`
	Provider map[string]any `json:"provider,omitempty"`
}

// EmbeddingResponse contains embedding vectors.
type EmbeddingResponse struct {
	Model    string         `json:"model"`
	Vectors  [][]float64    `json:"vectors"`
	Usage    Usage          `json:"usage,omitempty"`
	Provider map[string]any `json:"provider,omitempty"`
}

// FileUploadRequest uploads a file to a provider.
type FileUploadRequest struct {
	Model     string         `json:"model,omitempty"`
	Filename  string         `json:"filename"`
	BytesData []byte         `json:"-"`
	MediaType string         `json:"media_type"`
	Provider  map[string]any `json:"provider,omitempty"`
}

// FileUploadResponse contains the uploaded file ID.
type FileUploadResponse struct {
	ID       string         `json:"id"`
	Provider map[string]any `json:"provider,omitempty"`
}

// ImageGenerationRequest generates images from a prompt.
type ImageGenerationRequest struct {
	Model    string         `json:"model"`
	Prompt   string         `json:"prompt"`
	Size     string         `json:"size,omitempty"`
	Provider map[string]any `json:"provider,omitempty"`
}

// ImageGenerationResponse contains generated images.
type ImageGenerationResponse struct {
	Images   []DataSource   `json:"images"`
	Provider map[string]any `json:"provider,omitempty"`
}

// AudioGenerationRequest generates audio from text.
type AudioGenerationRequest struct {
	Model    string         `json:"model"`
	Prompt   string         `json:"prompt"`
	Voice    string         `json:"voice,omitempty"`
	Format   string         `json:"format,omitempty"`
	Provider map[string]any `json:"provider,omitempty"`
}

// AudioGenerationResponse contains generated audio.
type AudioGenerationResponse struct {
	Audio    DataSource     `json:"audio"`
	Provider map[string]any `json:"provider,omitempty"`
}
