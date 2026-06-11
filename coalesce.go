package lm15

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ─── MAP-3 coalescer ─────────────────────────────────────────────────

// CoalesceStream enforces MAP-3 (docs/mapping-rules.md): a stream yields
// exactly one StreamEndEvent, as the final event. Adapters are stateless
// and may emit one end event per provider terminal frame (finish_reason
// chunk, post-finish usage-only chunk, [DONE], message_delta +
// message_stop). This pass absorbs every end event's fields — a later
// non-nil field replaces the accumulated value, a nil field never erases
// one — and appends the single merged end event. If no end event was
// seen, none is fabricated.
func CoalesceStream(events []StreamEvent) []StreamEvent {
	out := make([]StreamEvent, 0, len(events))
	sawEnd := false
	var finish *string
	var usage *Usage
	var providerData jmap
	for _, event := range events {
		end, ok := event.(StreamEndEvent)
		if !ok {
			out = append(out, event)
			continue
		}
		sawEnd = true
		if end.FinishReason != nil {
			finish = end.FinishReason
		}
		if end.Usage != nil {
			usage = end.Usage
		}
		if end.ProviderData != nil {
			providerData = end.ProviderData
		}
	}
	if sawEnd {
		out = append(out, StreamEndEvent{FinishReason: finish, Usage: usage, ProviderData: providerData})
	}
	return out
}

// ─── Stream materialization ──────────────────────────────────────────

type toolCallAccum struct {
	id    *string
	name  *string
	raw   string
	input jmap
}

// roundState accumulates stream deltas into a complete Response
// (mirror of the reference lm15.result._RoundState).
type roundState struct {
	request          Request
	startedID        *string
	startedModel     *string
	finishReason     *string
	usage            *Usage
	textParts        map[int64][]string
	thinkingParts    map[int64][]string
	audioChunks      map[int64][]string
	audioMediaTypes  map[int64]*string
	imageParts       map[int64]ImagePart
	citationParts    map[int64][]CitationPart
	toolCalls        map[int64]*toolCallAccum
	messageCont      []ContinuationState
	partCont         map[int64][]ContinuationState
	providerData     jmap
	sawProviderData  bool
}

func newRoundState(req Request) *roundState {
	return &roundState{
		request:         req,
		textParts:       map[int64][]string{},
		thinkingParts:   map[int64][]string{},
		audioChunks:     map[int64][]string{},
		audioMediaTypes: map[int64]*string{},
		imageParts:      map[int64]ImagePart{},
		citationParts:   map[int64][]CitationPart{},
		toolCalls:       map[int64]*toolCallAccum{},
		partCont:        map[int64][]ContinuationState{},
	}
}

func (s *roundState) apply(event StreamEvent) {
	switch e := event.(type) {
	case StreamStartEvent:
		if e.ID != nil && *e.ID != "" {
			s.startedID = e.ID
		}
		if e.Model != nil && *e.Model != "" {
			s.startedModel = e.Model
		}
	case StreamEndEvent:
		if e.FinishReason != nil && *e.FinishReason != "" {
			s.finishReason = e.FinishReason
		}
		if e.Usage != nil {
			s.usage = e.Usage
		}
		if e.ProviderData != nil {
			s.providerData = e.ProviderData
			s.sawProviderData = true
		}
	case StreamDeltaEvent:
		s.applyDelta(e.Delta)
	}
}

func (s *roundState) applyDelta(delta Delta) {
	switch d := delta.(type) {
	case TextDelta:
		s.textParts[d.PartIndex] = append(s.textParts[d.PartIndex], d.Text)
	case ThinkingDelta:
		s.thinkingParts[d.PartIndex] = append(s.thinkingParts[d.PartIndex], d.Text)
	case AudioDelta:
		data := ""
		if d.Data != nil {
			data = *d.Data
		}
		s.audioChunks[d.PartIndex] = append(s.audioChunks[d.PartIndex], data)
		if _, ok := s.audioMediaTypes[d.PartIndex]; !ok {
			s.audioMediaTypes[d.PartIndex] = d.MediaType
		}
	case ToolCallDelta:
		acc, ok := s.toolCalls[d.PartIndex]
		if !ok {
			acc = &toolCallAccum{}
			s.toolCalls[d.PartIndex] = acc
		}
		if d.ID != nil {
			acc.id = d.ID
		}
		if d.Name != nil {
			acc.name = d.Name
		}
		acc.raw += d.Input
		acc.input = parseJSONBestEffort(acc.raw)
	case ImageDelta:
		mt := "image/png"
		if d.MediaType != nil && *d.MediaType != "" {
			mt = *d.MediaType
		}
		part := ImagePart{mediaPart: mediaPart{MediaType: mt}}
		switch {
		case d.Data != nil:
			part.Data = d.Data
		case d.URL != nil:
			part.URL = d.URL
		case d.FileID != nil:
			part.FileID = d.FileID
		default:
			return
		}
		s.imageParts[d.PartIndex] = part
	case CitationDelta:
		s.citationParts[d.PartIndex] = append(s.citationParts[d.PartIndex],
			CitationPart{Text: d.Text, URL: d.URL, Title: d.Title})
	case ContinuationDelta:
		state := d.ToState()
		if d.PartIndex == nil {
			s.messageCont = append(s.messageCont, state)
		} else {
			s.partCont[*d.PartIndex] = append(s.partCont[*d.PartIndex], state)
		}
	}
}

// parseJSONBestEffort mirrors the reference's tolerant tool-input parsing:
// a JSON object passes through, another JSON value is wrapped as
// {"value": v}, unparseable text as {"partial_json": raw}.
func parseJSONBestEffort(raw string) jmap {
	if raw == "" {
		return jmap{}
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return jmap{"partial_json": raw}
	}
	if m, ok := v.(jmap); ok {
		return m
	}
	return jmap{"value": v}
}

func (s *roundState) materialize() Response {
	var toolNames []string
	for _, t := range s.request.Tools {
		if ft, ok := t.(FunctionTool); ok {
			toolNames = append(toolNames, ft.Name)
		}
	}

	indexSet := map[int64]bool{}
	for idx := range s.thinkingParts {
		indexSet[idx] = true
	}
	for idx := range s.textParts {
		indexSet[idx] = true
	}
	for idx := range s.imageParts {
		indexSet[idx] = true
	}
	for idx := range s.audioChunks {
		indexSet[idx] = true
	}
	for idx := range s.citationParts {
		indexSet[idx] = true
	}
	for idx := range s.toolCalls {
		indexSet[idx] = true
	}
	for idx := range s.partCont {
		indexSet[idx] = true
	}
	indexes := make([]int64, 0, len(indexSet))
	for idx := range indexSet {
		indexes = append(indexes, idx)
	}
	sort.Slice(indexes, func(i, j int) bool { return indexes[i] < indexes[j] })

	var parts []Part
	for pos, idx := range indexes {
		continuation := s.partCont[idx]
		matched := false
		if texts, ok := s.thinkingParts[idx]; ok {
			matched = true
			parts = append(parts, ThinkingPart{Text: strings.Join(texts, ""), Continuation: continuation})
		}
		if texts, ok := s.textParts[idx]; ok {
			matched = true
			parts = append(parts, TextPart{Text: strings.Join(texts, ""), Continuation: continuation})
		}
		if img, ok := s.imageParts[idx]; ok {
			matched = true
			img.Continuation = continuation
			parts = append(parts, img)
		}
		if chunks, ok := s.audioChunks[idx]; ok {
			matched = true
			raw := concatB64Chunks(chunks)
			mediaType := s.audioMediaTypes[idx]
			var part AudioPart
			if mediaType == nil || *mediaType == "audio/pcm" || *mediaType == "audio/pcm16" {
				enc := base64.StdEncoding.EncodeToString(pcmToWav(raw, 24000, 1, 16))
				part = AudioPart{mediaPart{MediaType: "audio/wav", Data: &enc}}
			} else {
				enc := base64.StdEncoding.EncodeToString(raw)
				part = AudioPart{mediaPart{MediaType: *mediaType, Data: &enc}}
			}
			part.Continuation = continuation
			parts = append(parts, part)
		}
		if citations, ok := s.citationParts[idx]; ok {
			matched = true
			for _, c := range citations {
				c.Continuation = continuation
				parts = append(parts, c)
			}
		}
		if acc, ok := s.toolCalls[idx]; ok {
			payload := acc.input
			if payload == nil {
				payload = parseJSONBestEffort(acc.raw)
			}
			name := ""
			if acc.name != nil {
				name = *acc.name
			}
			if name == "" {
				switch {
				case len(toolNames) == 1:
					name = toolNames[0]
				case pos < len(toolNames):
					name = toolNames[pos]
				default:
					name = "tool"
				}
			}
			id := ""
			if acc.id != nil {
				id = *acc.id
			}
			if id == "" {
				id = fmt.Sprintf("tool_call_%d", idx)
			}
			parts = append(parts, ToolCallPart{ID: id, Name: name, Input: payload, Continuation: continuation})
		} else if !matched {
			parts = append(parts, TextPart{Text: "", Continuation: continuation})
		}
	}

	if len(parts) == 0 {
		parts = []Part{TextPart{Text: ""}}
	}

	finish := ""
	if s.finishReason != nil {
		finish = *s.finishReason
	}
	hasTool := hasToolCall(parts)
	if finish == "" {
		if hasTool {
			finish = "tool_call"
		} else {
			finish = "stop"
		}
	} else if finish == "stop" && hasTool {
		finish = "tool_call"
	}

	model := s.request.Model
	if s.startedModel != nil && *s.startedModel != "" {
		model = *s.startedModel
	}
	usage := Usage{}
	if s.usage != nil {
		usage = *s.usage
	}
	return Response{
		ID:           s.startedID,
		Model:        model,
		Message:      Message{Role: "assistant", Parts: parts, Continuation: s.messageCont},
		FinishReason: finish,
		Usage:        usage,
		ProviderData: s.providerData,
	}
}

// concatB64Chunks decodes each base64 chunk (padding-tolerant) and
// concatenates the raw bytes.
func concatB64Chunks(chunks []string) []byte {
	var raw []byte
	for _, chunk := range chunks {
		if chunk == "" {
			continue
		}
		if b, err := base64.StdEncoding.DecodeString(chunk); err == nil {
			raw = append(raw, b...)
			continue
		}
		padded := chunk + strings.Repeat("=", (4-len(chunk)%4)%4)
		if b, err := base64.StdEncoding.DecodeString(padded); err == nil {
			raw = append(raw, b...)
		}
	}
	return raw
}

// pcmToWav wraps raw PCM bytes in a WAV header.
func pcmToWav(pcm []byte, sampleRate, channels, bits int) []byte {
	byteRate := sampleRate * channels * bits / 8
	blockAlign := channels * bits / 8
	dataSize := len(pcm)
	buf := make([]byte, 0, 44+dataSize)
	w := func(b []byte) { buf = append(buf, b...) }
	u32 := func(v uint32) { var b [4]byte; binary.LittleEndian.PutUint32(b[:], v); w(b[:]) }
	u16 := func(v uint16) { var b [2]byte; binary.LittleEndian.PutUint16(b[:], v); w(b[:]) }
	w([]byte("RIFF"))
	u32(uint32(36 + dataSize))
	w([]byte("WAVE"))
	w([]byte("fmt "))
	u32(16)
	u16(1)
	u16(uint16(channels))
	u32(uint32(sampleRate))
	u32(uint32(byteRate))
	u16(uint16(blockAlign))
	u16(uint16(bits))
	w([]byte("data"))
	u32(uint32(dataSize))
	w(pcm)
	return buf
}

// MaterializeResponse consumes a canonical event trace and builds the
// complete Response.
func MaterializeResponse(events []StreamEvent, req Request) Response {
	state := newRoundState(req)
	for _, event := range events {
		state.apply(event)
		if _, ok := event.(StreamEndEvent); ok {
			break
		}
	}
	return state.materialize()
}

// ─── replay_stream ───────────────────────────────────────────────────

// ReplayStream parses a captured SSE body for the given provider into the
// POST-coalesce canonical event trace plus the materialized Response.
func ReplayStream(provider string, req Request, body []byte) ([]StreamEvent, Response, error) {
	sseEvents, err := ParseSSE(body)
	if err != nil {
		return nil, Response{}, err
	}
	var events []StreamEvent
	for _, raw := range sseEvents {
		mapped, err := ParseStreamEvents(provider, req, raw)
		if err != nil {
			return nil, Response{}, err
		}
		events = append(events, mapped...)
	}
	// MAP-3: the canonical trace is the post-coalesce trace — exactly one
	// merged StreamEndEvent, final.
	events = CoalesceStream(events)
	return events, MaterializeResponse(events, req), nil
}

// ReplayStreamMap is the vet-shim entry for the replay_stream op.
func ReplayStreamMap(provider string, canonical jmap, bodyB64 string) (jmap, error) {
	req, err := requestFromMap(canonical)
	if err != nil {
		return nil, err
	}
	body, err := base64.StdEncoding.DecodeString(bodyB64)
	if err != nil {
		return nil, valueErr("body_b64 is not valid base64: %v", err)
	}
	events, resp, err := ReplayStream(provider, req, body)
	if err != nil {
		return nil, err
	}
	eventMaps := make([]any, len(events))
	for i, e := range events {
		eventMaps[i] = e.toMap()
	}
	return jmap{"events": eventMaps, "canonical_response": resp.toMap()}, nil
}
