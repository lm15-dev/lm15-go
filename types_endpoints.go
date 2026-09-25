package lm15

import (
	"os"
	"strings"
)

// ─── Files ───────────────────────────────────────────────────────────

// FileUploadRequest uploads bytes or a local path. Files are account-scoped
// (no model). MediaType "" reads as application/octet-stream.
type FileUploadRequest struct {
	Filename   string
	Bytes      []byte
	MediaType  string
	Extensions JSONObject
	Path       string
}

// EffectiveMediaType returns MediaType or the octet-stream default.
func (r FileUploadRequest) EffectiveMediaType() string {
	if r.MediaType == "" {
		return "application/octet-stream"
	}
	return r.MediaType
}

// Validate checks exactly one of Bytes / Path.
func (r FileUploadRequest) Validate() error {
	if r.Filename == "" {
		return valueErrorf("FileUploadRequest.filename cannot be empty")
	}
	if r.Bytes == nil && r.Path == "" {
		return typeErrorf("FileUploadRequest requires bytes_data or path")
	}
	if r.Bytes != nil && r.Path != "" {
		return valueErrorf("FileUploadRequest requires exactly one of bytes_data or path")
	}
	if r.Bytes != nil && len(r.Bytes) == 0 {
		return valueErrorf("bytes_data is required")
	}
	_, err := normalizeExtensions(r.Extensions)
	return err
}

// Content returns the bytes (a path is read now).
func (r FileUploadRequest) Content() ([]byte, error) {
	if r.Bytes != nil {
		return r.Bytes, nil
	}
	if r.Path != "" {
		return os.ReadFile(r.Path)
	}
	return nil, valueErrorf("FileUploadRequest has neither bytes_data nor path")
}

// FileInfo is a snapshot of one stored file. Readiness "" reads as "ready".
type FileInfo struct {
	ID           string
	Filename     string
	MediaType    string
	SizeBytes    *int
	CreatedAt    string
	ExpiresAt    string
	Readiness    string
	Downloadable *bool
	ProviderData JSONObject
}

// EffectiveReadiness returns Readiness or "ready".
func (f FileInfo) EffectiveReadiness() string {
	if f.Readiness == "" {
		return "ready"
	}
	return f.Readiness
}

// Ready reports readiness == ready.
func (f FileInfo) Ready() bool { return f.EffectiveReadiness() == "ready" }

// Validate checks the snapshot.
func (f FileInfo) Validate() error {
	if f.ID == "" {
		return valueErrorf("FileInfo.id cannot be empty")
	}
	if f.SizeBytes != nil && *f.SizeBytes < 0 {
		return valueErrorf("FileInfo.size_bytes must be >= 0")
	}
	if !inVocab(f.EffectiveReadiness(), FileReadinessValues) {
		return valueErrorf("unsupported file readiness: %s", f.Readiness)
	}
	return checkJSONObject(f.ProviderData, "provider_data", false)
}

// FilePage is one page of stored files.
type FilePage struct {
	Items      []FileInfo
	NextCursor string
}

// Validate checks every item.
func (p FilePage) Validate() error {
	for _, f := range p.Items {
		if err := f.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ─── Cache resources ─────────────────────────────────────────────────

// CacheInfo is a snapshot of one stored cache object (MAP-6 resource tier).
type CacheInfo struct {
	ID           string
	Model        string
	Tokens       *int
	CreatedAt    string
	ExpiresAt    string
	Label        string
	ProviderData JSONObject
}

// Validate checks the snapshot.
func (c CacheInfo) Validate() error {
	if c.ID == "" {
		return valueErrorf("CacheInfo.id cannot be empty")
	}
	if c.Model == "" {
		return valueErrorf("CacheInfo.model cannot be empty")
	}
	if c.Tokens != nil && *c.Tokens < 0 {
		return valueErrorf("CacheInfo.tokens must be >= 0")
	}
	return checkJSONObject(c.ProviderData, "provider_data", false)
}

// CachePage is one page of cache objects.
type CachePage struct {
	Items      []CacheInfo
	NextCursor string
}

// Validate checks every item.
func (p CachePage) Validate() error {
	for _, c := range p.Items {
		if err := c.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// CachedPrefix is a reusable prompt beginning: the prefix Request and, on
// providers with the resource tier, the stored object.
type CachedPrefix struct {
	Prefix   *Request
	Resource *CacheInfo
	// Provider (optional) is the router destination the prefix was cached
	// through (LMRouter.Cache sets it; router-local names included). Prefix
	// and Resource models stay wire names; Request emits "provider:model".
	// It carries no credential, endpoint or provider declaration: reuse it
	// with the same router configuration.
	Provider string
}

// Validate checks the prefix carries a default Config and the resource matches.
func (c CachedPrefix) Validate() error {
	if c.Provider != "" && strings.ContainsAny(c.Provider, " \t\n\r:/") {
		return valueErrorf("CachedPrefix.provider must be a non-empty provider name without routing separators")
	}
	if c.Prefix == nil {
		return typeErrorf("CachedPrefix.prefix must be a Request")
	}
	if err := c.Prefix.Validate(); err != nil {
		return err
	}
	if !c.Prefix.Config.IsDefault() {
		return valueErrorf("CachedPrefix.prefix must carry a default Config: a cached object has no generation settings")
	}
	if c.Resource != nil {
		if err := c.Resource.Validate(); err != nil {
			return err
		}
		if c.Resource.Model != c.Prefix.Model {
			return valueErrorf("CachedPrefix.resource.model must equal the prefix model (a stored cache belongs to one model)")
		}
	}
	return nil
}

// ID is the resource id, or "".
func (c CachedPrefix) ID() string {
	if c.Resource == nil {
		return ""
	}
	return c.Resource.ID
}

// ExpiresAt is the resource expiry, or "".
func (c CachedPrefix) ExpiresAt() string {
	if c.Resource == nil {
		return ""
	}
	return c.Resource.ExpiresAt
}

// CacheConfigFor marks the seam between prefix and suffix.
func (c CachedPrefix) CacheConfigFor() CacheConfig {
	idx := len(c.Prefix.Messages) - 1
	return CacheConfig{PrefixUntilIndex: &idx, Resource: c.ID()}
}

// Request appends messages to the prefix and sets the cache boundary.
// config supplies generation settings; its Cache must be unset.
func (c CachedPrefix) Request(messages []Message, config *Config) (*Request, error) {
	if len(messages) == 0 {
		return nil, typeErrorf("messages must be a Message or a non-empty sequence of Messages")
	}
	base := Config{}
	if config != nil {
		base = *config
	}
	if base.Cache != nil {
		return nil, valueErrorf("config.cache is decided by the CachedPrefix; leave it unset")
	}
	cc := c.CacheConfigFor()
	base.Cache = &cc
	all := append(append([]Message(nil), c.Prefix.Messages...), messages...)
	model := c.Prefix.Model
	if c.Provider != "" {
		model = CanonicalProvider(c.Provider) + ":" + c.Prefix.Model
	}
	req := &Request{Model: model, System: c.Prefix.System, Tools: c.Prefix.Tools, Messages: all, Config: base}
	return req, req.Validate()
}

// RequestFrom appends a suffix Request (same model, no system, no tools).
func (c CachedPrefix) RequestFrom(suffix *Request, config *Config) (*Request, error) {
	head, rest, qualified := strings.Cut(suffix.Model, ":")
	sameRoute := c.Provider != "" && qualified && CanonicalProvider(head) == CanonicalProvider(c.Provider) && rest == c.Prefix.Model
	if suffix.Model != c.Prefix.Model && !sameRoute {
		return nil, valueErrorf("suffix Request model must equal the prefix model")
	}
	if suffix.System != nil || len(suffix.Tools) > 0 {
		return nil, valueErrorf("suffix Request cannot redefine system or tools: the prefix owns them")
	}
	if config == nil && !suffix.Config.IsDefault() {
		cfg := suffix.Config
		config = &cfg
	}
	return c.Request(suffix.Messages, config)
}

// ─── Batch ───────────────────────────────────────────────────────────

// BatchRequest is a batch of requests. Model "" infers from Requests[0].
type BatchRequest struct {
	Model      string
	Requests   []*Request
	Label      string
	Extensions JSONObject
}

// EffectiveModel returns Model or the first request's model (INV-032).
func (b BatchRequest) EffectiveModel() string {
	if b.Model != "" || len(b.Requests) == 0 {
		return b.Model
	}
	return b.Requests[0].Model
}

// Validate checks the batch.
func (b BatchRequest) Validate() error {
	if len(b.Requests) == 0 {
		return valueErrorf("requests cannot be empty")
	}
	for _, r := range b.Requests {
		if r == nil {
			return typeErrorf("BatchRequest.requests must contain Request objects")
		}
		if err := r.Validate(); err != nil {
			return err
		}
	}
	_, err := normalizeExtensions(b.Extensions)
	return err
}

// BatchJobInfo is the ticket: a snapshot of one provider-side job.
type BatchJobInfo struct {
	ID           string
	Status       string
	Label        string
	CreatedAt    string
	ProviderData JSONObject
}

// Done reports a terminal status.
func (j BatchJobInfo) Done() bool { return inVocab(j.Status, BatchTerminalStatuses) }

// Validate checks the snapshot.
func (j BatchJobInfo) Validate() error {
	if j.ID == "" {
		return valueErrorf("BatchJobInfo.id cannot be empty")
	}
	if !inVocab(j.Status, BatchStatuses) {
		return valueErrorf("unsupported batch status: %s", j.Status)
	}
	return checkJSONObject(j.ProviderData, "provider_data", false)
}

// BatchEntry is the fate of one request, in submission order.
type BatchEntry struct {
	Index    int
	Outcome  string
	Response *Response
	Error    *ErrorDetail
}

// OK reports outcome == succeeded.
func (e BatchEntry) OK() bool { return e.Outcome == "succeeded" }

// Validate checks the outcome/payload pairing.
func (e BatchEntry) Validate() error {
	if e.Index < 0 {
		return valueErrorf("BatchEntry.index must be a non-negative int")
	}
	if !inVocab(e.Outcome, BatchOutcomes) {
		return valueErrorf("unsupported batch outcome: %s", e.Outcome)
	}
	switch e.Outcome {
	case "succeeded":
		if e.Response == nil || e.Error != nil {
			return valueErrorf("succeeded entries carry a Response and no error")
		}
		return e.Response.Validate()
	case "errored":
		if e.Error == nil || e.Response != nil {
			return valueErrorf("errored entries carry an ErrorDetail and no response")
		}
		return e.Error.Validate()
	}
	if e.Response != nil || e.Error != nil {
		return valueErrorf("%s entries carry neither response nor error", e.Outcome)
	}
	return nil
}

// ─── Generation ──────────────────────────────────────────────────────

// ImageGenerationRequest: text (and optional input images) in, images out.
type ImageGenerationRequest struct {
	Model      string
	Prompt     string
	Size       string
	Images     []ImagePart
	Extensions JSONObject
}

// Validate checks the request.
func (r ImageGenerationRequest) Validate() error {
	if r.Model == "" {
		return valueErrorf("model is required")
	}
	if r.Prompt == "" {
		return valueErrorf("prompt is required")
	}
	for _, img := range r.Images {
		if err := img.Validate(); err != nil {
			return err
		}
	}
	_, err := normalizeExtensions(r.Extensions)
	return err
}

// ImageGenerationResponse carries generated images plus any narration.
type ImageGenerationResponse struct {
	Images       []ImagePart
	Text         string
	ID           string
	Model        string
	Usage        Usage
	ProviderData JSONObject
}

// Validate checks at least one image.
func (r ImageGenerationResponse) Validate() error {
	if len(r.Images) == 0 {
		return valueErrorf("ImageGenerationResponse requires at least one image")
	}
	for _, img := range r.Images {
		if err := img.Validate(); err != nil {
			return err
		}
	}
	if err := r.Usage.Validate(); err != nil {
		return err
	}
	return checkJSONObject(r.ProviderData, "provider_data", false)
}

// SpeechGenerationRequest is text-to-speech. Omitted voice/format mean the
// server's defaults.
type SpeechGenerationRequest struct {
	Model      string
	Prompt     string
	Voice      string
	Format     string
	Extensions JSONObject
}

// Validate checks the request.
func (r SpeechGenerationRequest) Validate() error {
	if r.Model == "" {
		return valueErrorf("model is required")
	}
	if r.Prompt == "" {
		return valueErrorf("prompt is required")
	}
	_, err := normalizeExtensions(r.Extensions)
	return err
}

// SpeechGenerationResponse carries the synthesized audio.
type SpeechGenerationResponse struct {
	Audio        AudioPart
	ID           string
	Model        string
	Usage        Usage
	ProviderData JSONObject
}

// Validate checks the response.
func (r SpeechGenerationResponse) Validate() error {
	if err := r.Audio.Validate(); err != nil {
		return err
	}
	if err := r.Usage.Validate(); err != nil {
		return err
	}
	return checkJSONObject(r.ProviderData, "provider_data", false)
}

// VideoGenerationRequest submits a video job.
type VideoGenerationRequest struct {
	Model      string
	Prompt     string
	Seconds    *int
	Images     []ImagePart
	Extensions JSONObject
}

// Validate checks the request.
func (r VideoGenerationRequest) Validate() error {
	if r.Model == "" {
		return valueErrorf("model is required")
	}
	if r.Prompt == "" {
		return valueErrorf("prompt is required")
	}
	if r.Seconds != nil && *r.Seconds <= 0 {
		return valueErrorf("VideoGenerationRequest.seconds must be a positive int")
	}
	for _, img := range r.Images {
		if err := img.Validate(); err != nil {
			return err
		}
	}
	_, err := normalizeExtensions(r.Extensions)
	return err
}

// VideoJobInfo is the ticket of a video job.
type VideoJobInfo struct {
	ID           string
	Status       string
	Progress     *int
	CreatedAt    string
	Model        string
	ProviderData JSONObject
}

// Done reports a terminal status.
func (j VideoJobInfo) Done() bool { return inVocab(j.Status, VideoTerminalStatuses) }

// Validate checks the snapshot.
func (j VideoJobInfo) Validate() error {
	if j.ID == "" {
		return valueErrorf("VideoJobInfo.id cannot be empty")
	}
	if !inVocab(j.Status, VideoStatuses) {
		return valueErrorf("unsupported video status: %s", j.Status)
	}
	if j.Progress != nil && (*j.Progress < 0 || *j.Progress > 100) {
		return valueErrorf("VideoJobInfo.progress must be an int percentage 0-100")
	}
	return checkJSONObject(j.ProviderData, "provider_data", false)
}

// ─── Audio / Live ────────────────────────────────────────────────────

// AudioFormat describes a live audio stream. Channels 0 reads as 1.
type AudioFormat struct {
	Encoding   string
	SampleRate int
	Channels   int
}

// EffectiveChannels returns Channels or 1.
func (a AudioFormat) EffectiveChannels() int {
	if a.Channels == 0 {
		return 1
	}
	return a.Channels
}

// Validate checks the format.
func (a AudioFormat) Validate() error {
	if !inVocab(a.Encoding, AudioEncodings) {
		return valueErrorf("unsupported audio encoding: %s", a.Encoding)
	}
	if a.SampleRate <= 0 {
		return valueErrorf("sample_rate must be > 0")
	}
	if a.EffectiveChannels() <= 0 {
		return valueErrorf("channels must be > 0")
	}
	return nil
}

// LiveConfig configures a live session.
type LiveConfig struct {
	Model        string
	System       *SystemPrompt
	Tools        []Tool
	Voice        string
	InputFormat  *AudioFormat
	OutputFormat *AudioFormat
	Extensions   JSONObject
}

// Validate checks the config.
func (c LiveConfig) Validate() error {
	if c.Model == "" {
		return valueErrorf("model is required")
	}
	if err := c.System.Validate(); err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, t := range c.Tools {
		if t == nil {
			return typeErrorf("LiveConfig.tools must contain Tool objects")
		}
		if err := t.Validate(); err != nil {
			return err
		}
		if seen[t.ToolName()] {
			return valueErrorf("LiveConfig.tools cannot contain duplicate tool names")
		}
		seen[t.ToolName()] = true
	}
	if c.InputFormat != nil {
		if err := c.InputFormat.Validate(); err != nil {
			return err
		}
	}
	if c.OutputFormat != nil {
		if err := c.OutputFormat.Validate(); err != nil {
			return err
		}
	}
	_, err := normalizeExtensions(c.Extensions)
	return err
}

// LiveClientEvent is a closed sum of the events a client sends.
type LiveClientEvent interface {
	Type() string
	Validate() error
	sealedLiveClient()
}

// LiveClientTurnEvent sends prompt parts; TurnComplete says whether the
// model may answer now (set it explicitly in literals; NewLiveClientTurnEvent
// defaults it to true).
type LiveClientTurnEvent struct {
	Parts        []Part
	TurnComplete bool
}

// NewLiveClientTurnEvent creates a complete turn.
func NewLiveClientTurnEvent(parts ...Part) LiveClientTurnEvent {
	return LiveClientTurnEvent{Parts: parts, TurnComplete: true}
}

// LiveClientAudioEvent streams input audio (base64). MediaType "" reads as audio/pcm;rate=16000.
type LiveClientAudioEvent struct {
	Data      string
	MediaType string
}

// LiveClientImageEvent streams an input frame (base64). MediaType "" reads as image/jpeg.
type LiveClientImageEvent struct {
	Data      string
	MediaType string
}

// LiveClientTextEvent sends text.
type LiveClientTextEvent struct{ Text string }

// LiveClientToolResultEvent answers a tool call.
type LiveClientToolResultEvent struct {
	ID      string
	Content []Part
}

// LiveClientInterruptEvent interrupts the model.
type LiveClientInterruptEvent struct{}

// LiveClientEndAudioEvent ends the audio input turn.
type LiveClientEndAudioEvent struct{}

func (LiveClientTurnEvent) Type() string       { return "turn" }
func (LiveClientAudioEvent) Type() string      { return "audio" }
func (LiveClientImageEvent) Type() string      { return "image" }
func (LiveClientTextEvent) Type() string       { return "text" }
func (LiveClientToolResultEvent) Type() string { return "tool_result" }
func (LiveClientInterruptEvent) Type() string  { return "interrupt" }
func (LiveClientEndAudioEvent) Type() string   { return "end_audio" }

func (LiveClientTurnEvent) sealedLiveClient()       {}
func (LiveClientAudioEvent) sealedLiveClient()      {}
func (LiveClientImageEvent) sealedLiveClient()      {}
func (LiveClientTextEvent) sealedLiveClient()       {}
func (LiveClientToolResultEvent) sealedLiveClient() {}
func (LiveClientInterruptEvent) sealedLiveClient()  {}
func (LiveClientEndAudioEvent) sealedLiveClient()   {}

// EffectiveMediaType returns MediaType or the default.
func (e LiveClientAudioEvent) EffectiveMediaType() string {
	if e.MediaType == "" {
		return "audio/pcm;rate=16000"
	}
	return e.MediaType
}

// EffectiveMediaType returns MediaType or the default.
func (e LiveClientImageEvent) EffectiveMediaType() string {
	if e.MediaType == "" {
		return "image/jpeg"
	}
	return e.MediaType
}

func (e LiveClientTurnEvent) Validate() error {
	if len(e.Parts) == 0 {
		return valueErrorf("LiveClientTurnEvent requires at least one part")
	}
	for _, p := range e.Parts {
		if p == nil {
			return typeErrorf("LiveClientTurnEvent.parts must contain Part objects")
		}
		if isPromptForbidden(p) {
			return typeErrorf("LiveClientTurnEvent.parts cannot contain model/tool protocol parts")
		}
		if err := p.Validate(); err != nil {
			return err
		}
	}
	return validateInputDataParts("live input", e.Parts)
}

func (e LiveClientAudioEvent) Validate() error {
	if e.Data == "" {
		return valueErrorf("LiveClientAudioEvent.data cannot be empty")
	}
	if err := validateBase64("LiveClientAudioEvent", e.Data); err != nil {
		return err
	}
	if !strings.HasPrefix(e.EffectiveMediaType(), "audio/") {
		return valueErrorf("LiveClientAudioEvent.media_type must start with 'audio/'")
	}
	return nil
}

func (e LiveClientImageEvent) Validate() error {
	if e.Data == "" {
		return valueErrorf("LiveClientImageEvent.data cannot be empty")
	}
	if err := validateBase64("LiveClientImageEvent", e.Data); err != nil {
		return err
	}
	if !strings.HasPrefix(e.EffectiveMediaType(), "image/") {
		return valueErrorf("LiveClientImageEvent.media_type must start with 'image/'")
	}
	return nil
}

func (LiveClientTextEvent) Validate() error { return nil }

func (e LiveClientToolResultEvent) Validate() error {
	if e.ID == "" {
		return valueErrorf("LiveClientToolResultEvent.id cannot be empty")
	}
	if len(e.Content) == 0 {
		return valueErrorf("LiveClientToolResultEvent requires content")
	}
	for _, p := range e.Content {
		if p == nil {
			return typeErrorf("LiveClientToolResultEvent.content must contain Part objects")
		}
		if isToolResultForbidden(p) {
			return typeErrorf("LiveClientToolResultEvent.content cannot contain model or protocol parts")
		}
		if err := p.Validate(); err != nil {
			return err
		}
	}
	return validateInputDataParts("live tool results", e.Content)
}

func (LiveClientInterruptEvent) Validate() error { return nil }
func (LiveClientEndAudioEvent) Validate() error  { return nil }

// LiveServerEvent is a closed sum of the events a server sends.
type LiveServerEvent interface {
	Type() string
	Validate() error
	sealedLiveServer()
}

// LiveServerAudioEvent carries output audio (base64).
type LiveServerAudioEvent struct {
	Data      string
	MediaType string
}

// LiveServerTextEvent carries output text (or a transcript).
type LiveServerTextEvent struct{ Text string }

// LiveServerToolCallEvent asks the client to run a tool.
type LiveServerToolCallEvent struct {
	ID    string
	Name  string
	Input JSONObject
}

// LiveServerToolCallDeltaEvent streams tool-call arguments.
type LiveServerToolCallDeltaEvent struct {
	InputDelta string
	ID         string
	Name       string
}

// LiveServerInterruptedEvent reports a barge-in.
type LiveServerInterruptedEvent struct{}

// LiveServerTurnEndEvent ends a turn with its usage.
type LiveServerTurnEndEvent struct{ Usage Usage }

// LiveServerUsageEvent bills a response that did not end the turn.
type LiveServerUsageEvent struct{ Usage Usage }

// LiveServerErrorEvent reports a failure.
type LiveServerErrorEvent struct{ Error ErrorDetail }

func (LiveServerAudioEvent) Type() string         { return "audio" }
func (LiveServerTextEvent) Type() string          { return "text" }
func (LiveServerToolCallEvent) Type() string      { return "tool_call" }
func (LiveServerToolCallDeltaEvent) Type() string { return "tool_call_delta" }
func (LiveServerInterruptedEvent) Type() string   { return "interrupted" }
func (LiveServerTurnEndEvent) Type() string       { return "turn_end" }
func (LiveServerUsageEvent) Type() string         { return "usage" }
func (LiveServerErrorEvent) Type() string         { return "error" }

func (LiveServerAudioEvent) sealedLiveServer()         {}
func (LiveServerTextEvent) sealedLiveServer()          {}
func (LiveServerToolCallEvent) sealedLiveServer()      {}
func (LiveServerToolCallDeltaEvent) sealedLiveServer() {}
func (LiveServerInterruptedEvent) sealedLiveServer()   {}
func (LiveServerTurnEndEvent) sealedLiveServer()       {}
func (LiveServerUsageEvent) sealedLiveServer()         {}
func (LiveServerErrorEvent) sealedLiveServer()         {}

func (e LiveServerAudioEvent) Validate() error {
	if e.Data == "" {
		return valueErrorf("LiveServerAudioEvent.data cannot be empty")
	}
	if err := validateBase64("LiveServerAudioEvent", e.Data); err != nil {
		return err
	}
	if e.MediaType != "" && !strings.HasPrefix(e.MediaType, "audio/") {
		return valueErrorf("LiveServerAudioEvent.media_type must start with 'audio/'")
	}
	return nil
}

func (LiveServerTextEvent) Validate() error { return nil }

func (e LiveServerToolCallEvent) Validate() error {
	if e.ID == "" {
		return valueErrorf("LiveServerToolCallEvent.id cannot be empty")
	}
	if e.Name == "" {
		return valueErrorf("LiveServerToolCallEvent.name cannot be empty")
	}
	if e.Input == nil {
		return typeErrorf("input must be a JSON object")
	}
	return checkJSONObject(e.Input, "input", true)
}

func (LiveServerToolCallDeltaEvent) Validate() error { return nil }
func (LiveServerInterruptedEvent) Validate() error   { return nil }
func (e LiveServerTurnEndEvent) Validate() error     { return e.Usage.Validate() }
func (e LiveServerUsageEvent) Validate() error       { return e.Usage.Validate() }
func (e LiveServerErrorEvent) Validate() error       { return e.Error.Validate() }
