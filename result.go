package lm15

import (
	"encoding/base64"
	"encoding/binary"
	"iter"
	"sort"
	"sync"
)

// Stream materialization: one engine (StreamAccumulator), the MAP-3/MAP-4
// coalescer, the lazy ResponseStream, and the one-shot MaterializeResponse.

// StreamAccumulator folds canonical stream events into a Response (MAP-9).
type StreamAccumulator struct {
	request             *Request
	startedID           string
	startedModel        string
	finishReason        string
	usage               *Usage
	textParts           map[int][]string
	thinkingParts       map[int][]string
	audioChunks         map[int][]string
	audioMediaTypes     map[int]string
	imageParts          map[int]ImagePart
	citationParts       map[int][]CitationPart
	toolCallRaw         map[int]string
	toolCallID          map[int]string
	toolCallName        map[int]string
	toolCallSeen        map[int]bool
	messageContinuation []ContinuationState
	partContinuation    map[int][]ContinuationState
	logprobSeq          []TokenLogprob
	logprobsIncomplete  bool
	providerData        JSONObject
	adaptations         []Adaptation
}

// NewStreamAccumulator creates an accumulator for a request.
func NewStreamAccumulator(request *Request) *StreamAccumulator {
	return &StreamAccumulator{
		request:          request,
		textParts:        map[int][]string{},
		thinkingParts:    map[int][]string{},
		audioChunks:      map[int][]string{},
		audioMediaTypes:  map[int]string{},
		imageParts:       map[int]ImagePart{},
		citationParts:    map[int][]CitationPart{},
		toolCallRaw:      map[int]string{},
		toolCallID:       map[int]string{},
		toolCallName:     map[int]string{},
		toolCallSeen:     map[int]bool{},
		partContinuation: map[int][]ContinuationState{},
	}
}

// Push folds one event. Error events are ignored (the caller decides).
func (a *StreamAccumulator) Push(event StreamEvent) {
	switch e := event.(type) {
	case StreamStartEvent:
		if e.ID != "" {
			a.startedID = e.ID
		}
		if e.Model != "" {
			a.startedModel = e.Model
		}
		if len(e.Adaptations) > 0 {
			a.adaptations = e.Adaptations
		}
	case StreamEndEvent:
		if e.FinishReason != "" {
			a.finishReason = e.FinishReason
		}
		if e.Usage != nil {
			a.usage = e.Usage
		}
		if e.ProviderData != nil {
			a.providerData = e.ProviderData
		}
	case StreamDeltaEvent:
		a.pushDelta(e.Delta)
	}
}

func (a *StreamAccumulator) pushDelta(delta Delta) {
	switch d := delta.(type) {
	case TextDelta:
		a.textParts[d.PartIndex] = append(a.textParts[d.PartIndex], d.Text)
		a.logprobSeq = append(a.logprobSeq, d.Logprobs...)
		// AND across text events: a later true never erases false.
		a.logprobsIncomplete = a.logprobsIncomplete || d.LogprobsIncomplete
	case ThinkingDelta:
		a.thinkingParts[d.PartIndex] = append(a.thinkingParts[d.PartIndex], d.Text)
	case AudioDelta:
		data := ""
		if d.Data != nil {
			data = *d.Data
		}
		a.audioChunks[d.PartIndex] = append(a.audioChunks[d.PartIndex], data)
		if _, seen := a.audioMediaTypes[d.PartIndex]; !seen {
			a.audioMediaTypes[d.PartIndex] = d.MediaType
		}
	case ToolCallDelta:
		a.toolCallSeen[d.PartIndex] = true
		if d.ID != "" {
			a.toolCallID[d.PartIndex] = d.ID
		}
		if d.Name != "" {
			a.toolCallName[d.PartIndex] = d.Name
		}
		a.toolCallRaw[d.PartIndex] += d.Input
	case ImageDelta:
		mt := d.MediaType
		if mt == "" {
			mt = "image/png"
		}
		switch {
		case d.Data != nil:
			a.imageParts[d.PartIndex] = ImagePart{Media: Media{MediaType: mt, Data: *d.Data}}
		case d.URL != nil:
			a.imageParts[d.PartIndex] = ImagePart{Media: Media{MediaType: mt, URL: *d.URL}}
		case d.FileID != nil:
			a.imageParts[d.PartIndex] = ImagePart{Media: Media{MediaType: mt, FileID: *d.FileID}}
		}
	case CitationDelta:
		c := CitationPart{}
		if d.Text != nil {
			c.Text = *d.Text
		}
		if d.URL != nil {
			c.URL = *d.URL
		}
		if d.Title != nil {
			c.Title = *d.Title
		}
		a.citationParts[d.PartIndex] = append(a.citationParts[d.PartIndex], c)
	case ContinuationDelta:
		state := d.ToState()
		if d.PartIndex == nil {
			a.messageContinuation = append(a.messageContinuation, state)
		} else {
			a.partContinuation[*d.PartIndex] = append(a.partContinuation[*d.PartIndex], state)
		}
	}
}

// Response builds the Response, or a StreamAssemblyError when a tool call
// never carried a name (MAP-9); the error carries the partial.
func (a *StreamAccumulator) Response() (*Response, error) {
	var unnamed []int
	for idx := range a.toolCallSeen {
		if a.toolCallName[idx] == "" {
			unnamed = append(unnamed, idx)
		}
	}
	if len(unnamed) > 0 {
		sort.Ints(unnamed)
		skip := map[int]bool{}
		for _, i := range unnamed {
			skip[i] = true
		}
		partial := a.assemble(skip)
		e := newError(KindStreamAssembly, "tool call at part "+intToStr(unnamed[0])+" arrived without a name; the adapter that produced this stream must set ToolCallDelta.name on the call's first fragment (MAP-9: lm15 does not guess which tool the model meant)")
		e.Partial = partial
		idx := unnamed[0]
		e.PartIndex = &idx
		return nil, e
	}
	return a.assemble(map[int]bool{}), nil
}

func (a *StreamAccumulator) assemble(skip map[int]bool) *Response {
	indexSet := map[int]bool{}
	for _, m := range []map[int]bool{a.toolCallSeen} {
		for k := range m {
			indexSet[k] = true
		}
	}
	for k := range a.thinkingParts {
		indexSet[k] = true
	}
	for k := range a.textParts {
		indexSet[k] = true
	}
	for k := range a.imageParts {
		indexSet[k] = true
	}
	for k := range a.audioChunks {
		indexSet[k] = true
	}
	for k := range a.citationParts {
		indexSet[k] = true
	}
	for k := range a.partContinuation {
		indexSet[k] = true
	}
	indexes := make([]int, 0, len(indexSet))
	for k := range indexSet {
		indexes = append(indexes, k)
	}
	sort.Ints(indexes)

	var parts []Part
	for _, idx := range indexes {
		continuation := a.partContinuation[idx]
		hasTool := a.toolCallSeen[idx] && !skip[idx]
		if chunks, ok := a.thinkingParts[idx]; ok {
			parts = append(parts, ThinkingPart{Text: joinStrings(chunks), Continuation: continuation})
		}
		if chunks, ok := a.textParts[idx]; ok {
			parts = append(parts, TextPart{Text: joinStrings(chunks), Continuation: continuation})
		}
		if img, ok := a.imageParts[idx]; ok {
			img.Continuation = continuation
			parts = append(parts, img)
		}
		if chunks, ok := a.audioChunks[idx]; ok {
			raw := concatB64Chunks(chunks)
			mt := a.audioMediaTypes[idx]
			if mt == "" || mt == "audio/pcm" || mt == "audio/pcm16" {
				parts = append(parts, AudioPart{Media: Media{MediaType: "audio/wav", Data: base64.StdEncoding.EncodeToString(pcmToWav(raw, 24000, 1, 16)), Continuation: continuation}})
			} else {
				parts = append(parts, AudioPart{Media: Media{MediaType: mt, Data: base64.StdEncoding.EncodeToString(raw), Continuation: continuation}})
			}
		}
		if cits, ok := a.citationParts[idx]; ok {
			for _, c := range cits {
				c.Continuation = continuation
				parts = append(parts, c)
			}
		}
		if hasTool {
			id := a.toolCallID[idx]
			if id == "" {
				id = "tool_call_" + intToStr(idx) // the lm15 correlator (Gemini sends none)
			}
			parts = append(parts, ToolCallPart{ID: id, Name: a.toolCallName[idx], Input: parseJSONBestEffort(a.toolCallRaw[idx]), Continuation: continuation})
		} else if !skip[idx] {
			_, t := a.thinkingParts[idx]
			_, x := a.textParts[idx]
			_, i := a.imageParts[idx]
			_, au := a.audioChunks[idx]
			_, c := a.citationParts[idx]
			if !t && !x && !i && !au && !c {
				parts = append(parts, TextPart{Text: "", Continuation: continuation})
			}
		}
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{Text: ""}}
	}
	finish := a.finishReason
	hasToolCalls := false
	for _, p := range parts {
		if _, ok := p.(ToolCallPart); ok {
			hasToolCalls = true
		}
	}
	if finish == "" {
		if hasToolCalls {
			finish = FinishToolCall
		} else {
			finish = FinishStop
		}
	} else if finish == FinishStop && hasToolCalls {
		finish = FinishToolCall
	}
	model := a.startedModel
	if model == "" && a.request != nil {
		model = a.request.Model
	}
	usage := Usage{}
	if a.usage != nil {
		usage = *a.usage
	}
	var logprobs []TokenLogprob
	if len(a.logprobSeq) > 0 {
		logprobs = a.logprobSeq
	}
	// MAP-14 §3: the single text part of a judgment answer becomes a DataPart.
	if a.request != nil {
		parts = ReplaceTextWithData(parts, RequestJudgments(a.request))
	}
	return &Response{
		ID:                 a.startedID,
		Model:              model,
		Message:            Message{Role: RoleAssistant, Parts: parts, Continuation: a.messageContinuation},
		FinishReason:       finish,
		Usage:              usage.Normalize(),
		Logprobs:           logprobs,
		LogprobsIncomplete: a.logprobsIncomplete,
		ProviderData:       a.providerData,
		Adaptations:        a.adaptations,
	}
}

func joinStrings(chunks []string) string {
	n := 0
	for _, c := range chunks {
		n += len(c)
	}
	buf := make([]byte, 0, n)
	for _, c := range chunks {
		buf = append(buf, c...)
	}
	return string(buf)
}

func concatB64Chunks(chunks []string) []byte {
	var raw []byte
	for _, chunk := range chunks {
		if chunk == "" {
			continue
		}
		decoded, err := base64.StdEncoding.DecodeString(chunk)
		if err != nil {
			padded := chunk
			for len(padded)%4 != 0 {
				padded += "="
			}
			if decoded, err = base64.StdEncoding.DecodeString(padded); err != nil {
				continue
			}
		}
		raw = append(raw, decoded...)
	}
	return raw
}

func pcmToWav(pcm []byte, sampleRate, channels, bits int) []byte {
	byteRate := sampleRate * channels * bits / 8
	blockAlign := channels * bits / 8
	header := make([]byte, 44)
	copy(header[0:], "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(36+len(pcm)))
	copy(header[8:], "WAVE")
	copy(header[12:], "fmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], uint16(channels))
	binary.LittleEndian.PutUint32(header[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(header[28:], uint32(byteRate))
	binary.LittleEndian.PutUint16(header[32:], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:], uint16(bits))
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(len(pcm)))
	return append(header, pcm...)
}

// ─── Coalescer (MAP-3 / MAP-4) ───────────────────────────────────────

// CoalesceStream enforces one leading start and one final merged end event
// over a raw adapter stream; delta and error events pass through.
func CoalesceStream(events iter.Seq2[StreamEvent, error], model string) iter.Seq2[StreamEvent, error] {
	return CoalesceStreamWith(events, model, nil)
}

// stampStart puts the build's adaptations (MAP-13) on the start event,
// provider-sent or synthesized: they are known before the first byte.
func stampStart(e StreamStartEvent, adaptations []Adaptation) StreamStartEvent {
	if len(adaptations) > 0 && len(e.Adaptations) == 0 {
		e.Adaptations = adaptations
	}
	return e
}

// CoalesceStreamWith is CoalesceStream with the build's adaptations stamped
// on the start event.
func CoalesceStreamWith(events iter.Seq2[StreamEvent, error], model string, adaptations []Adaptation) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		started := false
		sawEnd := false
		finish := ""
		var usage *Usage
		var endData JSONObject
		endRank := -1
		for event, err := range events {
			if err != nil {
				yield(nil, err)
				return
			}
			switch e := event.(type) {
			case StreamStartEvent:
				if started {
					continue
				}
				started = true
				if !yield(stampStart(e, adaptations), nil) {
					return
				}
				continue
			case StreamEndEvent:
				sawEnd = true
				if e.FinishReason != "" {
					finish = e.FinishReason
				}
				if e.Usage != nil {
					usage = e.Usage
				}
				if e.ProviderData != nil {
					rank := 0
					if e.Usage != nil {
						rank = 2
					} else if e.FinishReason != "" {
						rank = 1
					}
					if rank >= endRank {
						endData = e.ProviderData
						endRank = rank
					}
				}
				continue
			case StreamDeltaEvent:
				if !started {
					started = true
					if !yield(StreamStartEvent{Model: model, Adaptations: adaptations}, nil) {
						return
					}
				}
			}
			if !yield(event, nil) {
				return
			}
		}
		if sawEnd {
			if !started {
				if !yield(StreamStartEvent{Model: model, Adaptations: adaptations}, nil) {
					return
				}
			}
			yield(StreamEndEvent{FinishReason: finish, Usage: usage, ProviderData: endData}, nil)
		}
	}
}

// ─── One-shot materialization ────────────────────────────────────────

func errorFromEvent(detail ErrorDetail) *Error {
	e := newError(ErrorKindForCode(detail.Code), detail.Message)
	e.Code = detail.Code
	e.ProviderCode = detail.ProviderCode
	// The handshake diagnostics propagate into the error's metadata; the
	// status stays absent (an in-stream error is never labelled 200).
	e.RequestID = detail.HTTPResponse.RequestID
	e.RetryAfter = detail.HTTPResponse.RetryAfter
	e.RateLimitHeaders = detail.HTTPResponse.RateLimitHeaders.Clone()
	return e
}

func incompleteError(acc *StreamAccumulator, message string) *Error {
	e := newError(KindStreamAssembly, message)
	partial, err := acc.Response()
	if err != nil {
		if ae := AsError(err); ae != nil {
			partial = ae.Partial
		}
	}
	e.Partial = partial
	return e
}

// MaterializeResponse consumes a complete stream, requiring a final end event.
func MaterializeResponse(events iter.Seq2[StreamEvent, error], request *Request) (*Response, error) {
	acc := NewStreamAccumulator(request)
	var response *Response
	for event, err := range events {
		if err != nil {
			if response != nil {
				// A failure after the end event is the connection's afterlife (2026-09-11).
				break
			}
			return nil, err
		}
		if response != nil {
			e := newError(KindStreamAssembly, "Stream emitted an event after its end event (MAP-3: the end event is final); the source that produced this stream is defective")
			e.Partial = response
			return nil, e
		}
		if errEvent, ok := event.(StreamErrorEvent); ok {
			return nil, errorFromEvent(errEvent.Error)
		}
		acc.Push(event)
		if _, ok := event.(StreamEndEvent); ok {
			resp, err := acc.Response()
			if err != nil {
				return nil, err
			}
			response = resp
		}
	}
	if response == nil {
		return nil, incompleteError(acc, "Stream ended without an end event: its finish reason and usage never arrived, so the text is not a finished turn (MAP-3)")
	}
	return response, nil
}

// ─── ResponseStream ──────────────────────────────────────────────────

// ResponseStream is a lazy stream-backed assembler: iterate Text() or
// Events(), then Response() for the same Response Complete returns.
type ResponseStream struct {
	mu       sync.Mutex
	acc      *StreamAccumulator
	next     func() (StreamEvent, error, bool)
	stop     func()
	response *Response
	failure  error
	done     bool
	// CleanupErrors are failures that followed the end event (a read error
	// while draining); the Response is complete regardless.
	CleanupErrors []error
}

// NewResponseStream wraps a stream of canonical events.
func NewResponseStream(events iter.Seq2[StreamEvent, error], request *Request) *ResponseStream {
	next, stop := iter.Pull2(events)
	return &ResponseStream{acc: NewStreamAccumulator(request), next: next, stop: stop}
}

// pump reads the next event, teeing it through the accumulator.
func (s *ResponseStream) pump() (StreamEvent, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return nil, false, s.failure
	}
	event, err, ok := s.next()
	if !ok {
		s.finish(nil)
		return nil, false, s.failure
	}
	if err != nil {
		if s.response != nil {
			s.CleanupErrors = append(s.CleanupErrors, err)
			s.finish(nil)
			return nil, false, nil
		}
		s.finish(err)
		return nil, false, err
	}
	if s.response != nil {
		e := newError(KindStreamAssembly, "Stream emitted an event after its end event (MAP-3: the end event is final); the source that produced this stream is defective")
		e.Partial = s.response
		s.finish(e)
		return nil, false, e
	}
	if errEvent, ok := event.(StreamErrorEvent); ok {
		e := errorFromEvent(errEvent.Error)
		s.finish(e)
		return nil, false, e
	}
	s.acc.Push(event)
	if _, ok := event.(StreamEndEvent); ok {
		resp, err := s.acc.Response()
		if err != nil {
			s.finish(err)
			return nil, false, err
		}
		s.response = resp
	}
	return event, true, nil
}

func (s *ResponseStream) finish(err error) {
	if s.done {
		return
	}
	s.done = true
	if err != nil {
		s.failure = err
	} else if s.response == nil && s.failure == nil {
		s.failure = incompleteError(s.acc, "Stream ended without an end event: its finish reason and usage never arrived, so the text is not a finished turn (MAP-3)")
	}
	s.stop()
}

// Events yields the canonical events as they arrive.
func (s *ResponseStream) Events() iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		for {
			event, ok, err := s.pump()
			if err != nil {
				yield(nil, err)
				return
			}
			if !ok {
				return
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}

// Text yields text fragments as they arrive.
func (s *ResponseStream) Text() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for event, err := range s.Events() {
			if err != nil {
				yield("", err)
				return
			}
			if de, ok := event.(StreamDeltaEvent); ok {
				if td, ok := de.Delta.(TextDelta); ok {
					if !yield(td.Text, nil) {
						return
					}
				}
			}
		}
	}
}

// Response drains the stream (if needed) and returns the assembled Response.
func (s *ResponseStream) Response() (*Response, error) {
	for {
		s.mu.Lock()
		done, failure, resp := s.done, s.failure, s.response
		s.mu.Unlock()
		if failure != nil {
			return nil, failure
		}
		if done {
			return resp, nil
		}
		if _, ok, err := s.pump(); err != nil {
			return nil, err
		} else if !ok {
			continue
		}
	}
}

// Close stops reading without draining. An unfinished stream has no
// complete response (MAP-3).
func (s *ResponseStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.done && s.response == nil && s.failure == nil {
		s.failure = incompleteError(s.acc, "Stream closed before its end event: the response was not completed (close() was called while the stream was still open; MAP-3)")
	}
	s.done = true
	s.stop()
	return nil
}

// ─── Conversion ──────────────────────────────────────────────────────

// ResponseToEvents converts a Response to stream events (lossless for the
// Delta vocabulary; a non-streamable part is an error).
func ResponseToEvents(r *Response) ([]StreamEvent, error) {
	events := []StreamEvent{StreamStartEvent{ID: r.ID, Model: r.Model, Adaptations: r.Adaptations}}
	pending := r.Logprobs
	pendingIncomplete := r.LogprobsIncomplete
	if pendingIncomplete {
		hasText := false
		for _, part := range r.Message.Parts {
			if _, ok := part.(TextPart); ok {
				hasText = true
			}
		}
		if !hasText {
			return nil, typeErrorf("Cannot stream incomplete logprobs without a TextPart to carry their coverage")
		}
	}
	for idx, part := range r.Message.Parts {
		var delta Delta
		switch p := part.(type) {
		case TextPart:
			delta = TextDelta{Text: p.Text, PartIndex: idx, Logprobs: pending, LogprobsIncomplete: pendingIncomplete}
			pending = nil
			pendingIncomplete = false
		case ThinkingPart:
			delta = ThinkingDelta{Text: p.Text, PartIndex: idx}
		case ToolCallPart:
			delta = ToolCallDelta{Input: jsonRaw(p.Input), PartIndex: idx, ID: p.ID, Name: p.Name}
		case ImagePart:
			d := ImageDelta{PartIndex: idx, MediaType: p.MediaType}
			if p.Data != "" {
				d.Data = S(p.Data)
			}
			if p.URL != "" {
				d.URL = S(p.URL)
			}
			if p.FileID != "" {
				d.FileID = S(p.FileID)
			}
			delta = d
		case AudioPart:
			if p.Data == "" {
				return nil, typeErrorf("Cannot convert AudioPart to StreamEvent: AudioDelta only supports inline data")
			}
			delta = AudioDelta{Data: S(p.Data), PartIndex: idx, MediaType: p.MediaType}
		case CitationPart:
			d := CitationDelta{PartIndex: idx}
			if p.Text != "" {
				d.Text = S(p.Text)
			}
			if p.URL != "" {
				d.URL = S(p.URL)
			}
			if p.Title != "" {
				d.Title = S(p.Title)
			}
			delta = d
		default:
			return nil, typeErrorf("Cannot convert %T to StreamEvent: no %q Delta variant exists", part, part.Type())
		}
		events = append(events, StreamDeltaEvent{Delta: delta})
		for _, state := range part.ContinuationStates() {
			i := idx
			events = append(events, StreamDeltaEvent{Delta: ContinuationDelta{Provider: state.Provider, Kind: state.Kind, Data: state.Data, PartIndex: &i}})
		}
	}
	for _, state := range r.Message.Continuation {
		events = append(events, StreamDeltaEvent{Delta: ContinuationDelta{Provider: state.Provider, Kind: state.Kind, Data: state.Data}})
	}
	usage := r.Usage
	events = append(events, StreamEndEvent{FinishReason: r.FinishReason, Usage: &usage, ProviderData: r.ProviderData})
	return events, nil
}

// SliceSeq turns a slice of events into a stream.
func SliceSeq(events []StreamEvent) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		for _, e := range events {
			if !yield(e, nil) {
				return
			}
		}
	}
}
