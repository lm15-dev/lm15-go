package lm15

// ─── Deltas ──────────────────────────────────────────────────────────

// Delta is a typed fragment of a Part arriving during streaming; a closed
// sum. PartIndex names a slot (MAP-9).
type Delta interface {
	Type() string
	Validate() error
	sealedDelta()
}

// TextDelta is a text fragment (with the fragment's own logprobs when
// streamed). LogprobsIncomplete (wire logprobs_complete=false) means local
// text editing removed scores that cannot describe the retained text (a
// stop inside a token); scores always describe original, whole provider
// tokens. Materialization ANDs the flag across text events.
type TextDelta struct {
	Text               string
	PartIndex          int
	Logprobs           []TokenLogprob
	LogprobsIncomplete bool
}

// ThinkingDelta is a reasoning fragment.
type ThinkingDelta struct {
	Text      string
	PartIndex int
}

// AudioDelta is a partial audio chunk. Data may be unaligned base64; an empty
// string is data (emitted), nil is absent.
type AudioDelta struct {
	Data      *string
	URL       *string
	FileID    *string
	PartIndex int
	MediaType string
}

// ImageDelta is a partial image chunk.
type ImageDelta struct {
	Data      *string
	URL       *string
	FileID    *string
	PartIndex int
	MediaType string
}

// ToolCallDelta is a tool-call input fragment, optionally carrying identity.
type ToolCallDelta struct {
	Input     string
	PartIndex int
	ID        string
	Name      string
}

// CitationDelta is a citation fragment.
type CitationDelta struct {
	Text      *string
	URL       *string
	Title     *string
	PartIndex int
}

// ContinuationDelta is opaque replay state. PartIndex nil attaches to the
// message; an int attaches to that completed part.
type ContinuationDelta struct {
	Provider  string
	Kind      string
	Data      JSONObject
	PartIndex *int
}

func (TextDelta) Type() string         { return DeltaTypeText }
func (ThinkingDelta) Type() string     { return DeltaTypeThinking }
func (AudioDelta) Type() string        { return DeltaTypeAudio }
func (ImageDelta) Type() string        { return DeltaTypeImage }
func (ToolCallDelta) Type() string     { return DeltaTypeToolCall }
func (CitationDelta) Type() string     { return DeltaTypeCitation }
func (ContinuationDelta) Type() string { return DeltaTypeContinuation }

func (TextDelta) sealedDelta()         {}
func (ThinkingDelta) sealedDelta()     {}
func (AudioDelta) sealedDelta()        {}
func (ImageDelta) sealedDelta()        {}
func (ToolCallDelta) sealedDelta()     {}
func (CitationDelta) sealedDelta()     {}
func (ContinuationDelta) sealedDelta() {}

func validatePartIndex(i int) error {
	if i < 0 {
		return valueErrorf("part_index must be >= 0")
	}
	return nil
}

func (d TextDelta) Validate() error {
	if err := validatePartIndex(d.PartIndex); err != nil {
		return err
	}
	for _, lp := range d.Logprobs {
		if err := lp.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (d ThinkingDelta) Validate() error { return validatePartIndex(d.PartIndex) }

func mediaDeltaAddresses(partType string, data, url, fileID *string) error {
	n := 0
	for _, p := range []*string{data, url, fileID} {
		if p != nil {
			n++
		}
	}
	if n > 1 {
		return valueErrorf("%s can include at most one of data, url, or file_id", partType)
	}
	return nil
}

func (d AudioDelta) Validate() error {
	if err := validatePartIndex(d.PartIndex); err != nil {
		return err
	}
	return mediaDeltaAddresses("AudioDelta", d.Data, d.URL, d.FileID)
}

func (d ImageDelta) Validate() error {
	if err := validatePartIndex(d.PartIndex); err != nil {
		return err
	}
	return mediaDeltaAddresses("ImageDelta", d.Data, d.URL, d.FileID)
}

func (d ToolCallDelta) Validate() error { return validatePartIndex(d.PartIndex) }

func (d CitationDelta) Validate() error {
	if err := validatePartIndex(d.PartIndex); err != nil {
		return err
	}
	if d.Text == nil && d.URL == nil && d.Title == nil {
		return valueErrorf("CitationDelta requires at least one of text, url, or title")
	}
	return nil
}

func (d ContinuationDelta) Validate() error {
	if d.Provider == "" {
		return valueErrorf("ContinuationDelta.provider cannot be empty")
	}
	if d.Kind == "" {
		return valueErrorf("ContinuationDelta.kind cannot be empty")
	}
	if d.Data == nil {
		return typeErrorf("data must be a JSON object")
	}
	if err := checkJSONObject(d.Data, "data", true); err != nil {
		return err
	}
	if d.PartIndex != nil {
		return validatePartIndex(*d.PartIndex)
	}
	return nil
}

// ToState converts the delta to its ContinuationState.
func (d ContinuationDelta) ToState() ContinuationState {
	return ContinuationState{Provider: d.Provider, Kind: d.Kind, Data: d.Data}
}

// ─── Stream events ───────────────────────────────────────────────────

// ErrorDetail is structured error information. HTTPResponse is the bounded
// handshake diagnostics of an in-stream error (2026-09-19); empty means no
// HTTP evidence, not success.
type ErrorDetail struct {
	Code         string
	Message      string
	ProviderCode string
	HTTPResponse HTTPResponseDetail
}

// Validate checks the code vocabulary.
func (e ErrorDetail) Validate() error {
	if !inVocab(e.Code, ErrorCodes) {
		return valueErrorf("unsupported error code: %s", e.Code)
	}
	return e.HTTPResponse.Validate()
}

// StreamEvent is one of StreamStartEvent, StreamDeltaEvent, StreamEndEvent,
// StreamErrorEvent.
type StreamEvent interface {
	Type() string
	Validate() error
	sealedStreamEvent()
}

// StreamStartEvent opens a stream (exactly one, MAP-4). Adaptations (MAP-13)
// is what the wire got that differs from what was asked, known before the
// first byte and so carried by the first event.
type StreamStartEvent struct {
	ID          string
	Model       string
	Adaptations []Adaptation
}

// StreamDeltaEvent carries a typed delta.
type StreamDeltaEvent struct{ Delta Delta }

// StreamEndEvent closes a stream (exactly one, final, MAP-3).
type StreamEndEvent struct {
	FinishReason string
	Usage        *Usage
	ProviderData JSONObject
}

// StreamErrorEvent reports a failure.
type StreamErrorEvent struct{ Error ErrorDetail }

func (StreamStartEvent) Type() string { return "start" }
func (StreamDeltaEvent) Type() string { return "delta" }
func (StreamEndEvent) Type() string   { return "end" }
func (StreamErrorEvent) Type() string { return "error" }

func (StreamStartEvent) sealedStreamEvent() {}
func (StreamDeltaEvent) sealedStreamEvent() {}
func (StreamEndEvent) sealedStreamEvent()   {}
func (StreamErrorEvent) sealedStreamEvent() {}

func (e StreamStartEvent) Validate() error { return validateAdaptations(e.Adaptations) }

func (e StreamDeltaEvent) Validate() error {
	if e.Delta == nil {
		return typeErrorf("StreamDeltaEvent.delta must be a Delta")
	}
	return e.Delta.Validate()
}

func (e StreamEndEvent) Validate() error {
	if e.FinishReason != "" && !inVocab(e.FinishReason, FinishReasons) {
		return valueErrorf("unsupported finish reason: %s", e.FinishReason)
	}
	if e.Usage != nil {
		if err := e.Usage.Validate(); err != nil {
			return err
		}
	}
	return checkJSONObject(e.ProviderData, "provider_data", false)
}

func (e StreamErrorEvent) Validate() error { return e.Error.Validate() }
