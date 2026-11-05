package lm15

import (
	"strings"

	"github.com/lm15-dev/lm15-go/v2/internal/sse"
)

// streamReader reads ONE stream's raw provider events into canonical
// events (MAP-20). A dialect's per-event mapping (parseStreamEvents) holds
// no state; a reader may hold what one stream needs and nothing across
// streams: a dialect that restates an item in a final copy completes the
// item's fragments from it. feed is called once per raw event, in order.
type streamReader interface {
	feed(ev sse.Event) ([]StreamEvent, error)
}

// readerDialect is a dialect whose streams need a reader of their own.
type readerDialect interface {
	streamReader(req *Request) streamReader
}

type statelessReader struct {
	d   dialect
	req *Request
}

func (r statelessReader) feed(ev sse.Event) ([]StreamEvent, error) {
	return r.d.parseStreamEvents(r.req, ev)
}

// newStreamReader is the reader for one stream of req: every path that
// turns raw events into canonical events goes through it (the stream and
// the vet shim's ParseStreamBody).
func (c *lmCore) newStreamReader(req *Request) streamReader {
	if rd, ok := c.self.(readerDialect); ok {
		return rd.streamReader(req)
	}
	return statelessReader{d: c.self, req: req}
}

// ─── Final copies (MAP-20) ──────────────────────────────────────────────

// finalCopies is what one stream's fragments delivered, per part slot and
// channel. A final copy is the provider restating a value it may also have
// sent in fragments; remainder is what the fragments did not deliver: the
// whole value when none came, the rest when they are its beginning, nothing
// when they are all of it. Fragments that are not its beginning contradict
// it and the stream is refused: lm15 does not choose.
type finalCopies struct {
	seen map[finalKey]string
}

type finalKey struct {
	index   int
	channel string
}

func newFinalCopies() *finalCopies { return &finalCopies{seen: map[finalKey]string{}} }

func (f *finalCopies) saw(index int, channel, text string) {
	if text != "" {
		k := finalKey{index, channel}
		f.seen[k] += text
	}
}

func (f *finalCopies) remainder(index int, channel, final string, jsonValue bool) (string, error) {
	k := finalKey{index, channel}
	got := f.seen[k]
	if strings.HasPrefix(final, got) {
		f.seen[k] = final
		return final[len(got):], nil
	}
	// The same JSON value, members in the same order: the text differs
	// (whitespace, escapes), the arguments do not.
	if jsonValue && sameJSON(got, final) {
		return "", nil
	}
	e := newError(KindStreamAssembly, "the streamed "+channel+" of part "+intToStr(index)+
		" is not the beginning of the provider's final copy of it ("+intToStr(len(got))+" bytes streamed, "+
		intToStr(len(final))+" in the final copy); lm15 does not choose between them (MAP-20)")
	idx := index
	e.PartIndex = &idx
	return "", e
}

func sameJSON(a, b string) bool {
	va, err := DecodeJSON([]byte(a))
	if err != nil {
		return false
	}
	vb, err := DecodeJSON([]byte(b))
	if err != nil {
		return false
	}
	ea, err1 := EncodeJSON(va)
	eb, err2 := EncodeJSON(vb)
	return err1 == nil && err2 == nil && string(ea) == string(eb)
}

func argumentsText(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	}
	b, err := EncodeJSON(v)
	if err != nil {
		return ""
	}
	return string(b)
}

// messageFinalText is the text a message item's fragments add up to: its
// output_text and refusal contents in order (both stream as text into the
// item's slot).
func messageFinalText(item JSONObject) string {
	var b strings.Builder
	for _, raw := range wireList(item.Get("content")) {
		c := wireObj(raw)
		if c == nil {
			continue
		}
		switch wireStr(c.Get("type")) {
		case "output_text", "text":
			b.WriteString(wireStr(c.Get("text")))
		case "refusal":
			b.WriteString(firstStr(c.Get("refusal"), c.Get("text")))
		}
	}
	return b.String()
}

// responsesStreamReader reads one OpenAI Responses stream (MAP-20).
// response.output_item.done is each item's final copy; at it the reader
// emits what the item's fragments did not deliver: a function call's
// arguments (the Codex backend sends parallel calls' arguments only there),
// a message's text, a reasoning item's summary. A summary's parts are joined
// with a newline, as the complete path joins them (INV-051); the newline
// goes before the first fragment of each later part.
type responsesStreamReader struct {
	lm          *OpenAILM
	req         *Request
	copies      *finalCopies
	summaryPart map[int]int
	named       map[int]bool
}

func (l *OpenAILM) streamReader(req *Request) streamReader {
	return &responsesStreamReader{lm: l, req: req, copies: newFinalCopies(), summaryPart: map[int]int{}, named: map[int]bool{}}
}

func (r *responsesStreamReader) feed(ev sse.Event) ([]StreamEvent, error) {
	events, err := r.lm.parseStreamEvents(r.req, ev)
	if err != nil || ev.Data == "" || ev.Data == "[DONE]" {
		return events, err
	}
	raw, err := DecodeJSON([]byte(ev.Data))
	if err != nil {
		return events, nil
	}
	payload := wireObj(raw)
	if payload == nil {
		return events, nil
	}
	et := wireStr(payload.Get("type"))
	index := wireInt(payload.Get("output_index"), 0)
	switch et {
	case "response.reasoning_summary_text.delta":
		part := wireInt(payload.Get("summary_index"), 0)
		at := r.summaryPart[index]
		prefix := ""
		if part > at {
			r.summaryPart[index] = part
			prefix = strings.Repeat("\n", part-at)
		}
		for i, e := range events {
			if de, ok := e.(StreamDeltaEvent); ok {
				if td, ok := de.Delta.(ThinkingDelta); ok {
					td.Text = prefix + td.Text
					events[i] = StreamDeltaEvent{Delta: td}
					r.copies.saw(index, "summary", td.Text)
				}
			}
		}
	case "response.output_text.delta", "response.refusal.delta":
		r.copies.saw(index, "text", wireStr(payload.Get("delta")))
	case "response.output_item.added", "response.function_call_arguments.delta":
		for _, e := range events {
			if de, ok := e.(StreamDeltaEvent); ok {
				if tc, ok := de.Delta.(ToolCallDelta); ok {
					r.copies.saw(index, "arguments", tc.Input)
					if tc.Name != "" {
						r.named[index] = true
					}
				}
			}
		}
	case "response.output_item.done":
		if item := wireObj(payload.Get("item")); item != nil {
			fill, err := r.complete(index, item)
			if err != nil {
				return nil, err
			}
			events = append(fill, events...)
		}
	}
	return events, nil
}

func (r *responsesStreamReader) complete(index int, item JSONObject) ([]StreamEvent, error) {
	switch wireStr(item.Get("type")) {
	case "function_call":
		rest, err := r.copies.remainder(index, "arguments", argumentsText(item.Get("arguments")), true)
		if err != nil {
			return nil, err
		}
		if rest == "" && r.named[index] {
			return nil, nil
		}
		name := wireStr(item.Get("name"))
		if name != "" {
			r.named[index] = true
		}
		return []StreamEvent{StreamDeltaEvent{Delta: ToolCallDelta{Input: rest, PartIndex: index, ID: firstStr(item.Get("call_id"), item.Get("id")), Name: name}}}, nil
	case "message":
		rest, err := r.copies.remainder(index, "text", messageFinalText(item), false)
		if err != nil || rest == "" {
			return nil, err
		}
		return []StreamEvent{StreamDeltaEvent{Delta: TextDelta{Text: rest, PartIndex: index}}}, nil
	case "reasoning":
		summary, ok := item.Get("summary").([]any)
		if !ok {
			return nil, nil
		}
		texts := make([]string, 0, len(summary))
		for _, x := range summary {
			if o := wireObj(x); o != nil {
				texts = append(texts, wireStr(o.Get("text")))
			} else {
				texts = append(texts, wireStr(x))
			}
		}
		rest, err := r.copies.remainder(index, "summary", strings.Join(texts, "\n"), false)
		if err != nil || rest == "" {
			return nil, err
		}
		return []StreamEvent{StreamDeltaEvent{Delta: ThinkingDelta{Text: rest, PartIndex: index}}}, nil
	}
	return nil, nil
}

// ─── Gemini: part slots across chunks (MAP-20) ──────────────────────────

func geminiPartKind(part JSONObject) string {
	if part == nil {
		return ""
	}
	if part.Has("text") {
		if truthy(part.Get("thought")) {
			return "thought"
		}
		return "text"
	}
	if wireObj(part.Get("functionCall")) != nil {
		return "call"
	}
	if wireObj(part.Get("inlineData")) != nil {
		return "media"
	}
	return ""
}

// geminiStreamReader gives each chunk's parts their slot in the stream. A
// chunk's parts are numbered from 0 in every chunk. The first part of a
// chunk continues the previous chunk's last part when both are text (or
// both thought text): that is how a long text arrives. Every other part is
// a new part with a slot of its own, in arrival order. A function call is
// never a continuation: gemini-2.5-pro sends each parallel call in a chunk
// of its own (cases/gemini/streaming_parallel_tool_calls_chunked.json).
type geminiStreamReader struct {
	lm       *GeminiLM
	req      *Request
	lastSlot int
	lastKind string
	next     int
}

func (l *GeminiLM) streamReader(req *Request) streamReader {
	return &geminiStreamReader{lm: l, req: req}
}

func (r *geminiStreamReader) feed(ev sse.Event) ([]StreamEvent, error) {
	events, err := r.lm.parseStreamEvents(r.req, ev)
	if err != nil || ev.Data == "" {
		return events, err
	}
	raw, err := DecodeJSON([]byte(ev.Data))
	if err != nil {
		return events, nil
	}
	payload := wireObj(raw)
	if payload == nil {
		return events, nil
	}
	var parts []any
	if cands := wireList(payload.Get("candidates")); len(cands) > 0 {
		if cand := wireObj(cands[0]); cand != nil {
			parts = wireList(wireObj(cand.Get("content")).Get("parts"))
		}
	}
	slots := map[int]int{}
	for idx, rawPart := range parts {
		kind := geminiPartKind(wireObj(rawPart))
		if kind == "" {
			continue
		}
		var slot int
		if idx == 0 && (kind == "text" || kind == "thought") && r.lastKind == kind {
			slot = r.lastSlot
		} else {
			slot = r.next
			r.next++
		}
		slots[idx] = slot
		r.lastSlot, r.lastKind = slot, kind
	}
	for i, e := range events {
		if de, ok := e.(StreamDeltaEvent); ok {
			events[i] = StreamDeltaEvent{Delta: withSlot(de.Delta, slots)}
		}
	}
	return events, nil
}

func withSlot(d Delta, slots map[int]int) Delta {
	remap := func(i int) int {
		if s, ok := slots[i]; ok {
			return s
		}
		return i
	}
	switch x := d.(type) {
	case TextDelta:
		x.PartIndex = remap(x.PartIndex)
		return x
	case ThinkingDelta:
		x.PartIndex = remap(x.PartIndex)
		return x
	case AudioDelta:
		x.PartIndex = remap(x.PartIndex)
		return x
	case ImageDelta:
		x.PartIndex = remap(x.PartIndex)
		return x
	case ToolCallDelta:
		x.PartIndex = remap(x.PartIndex)
		return x
	case CitationDelta:
		x.PartIndex = remap(x.PartIndex)
		return x
	case ContinuationDelta:
		if x.PartIndex != nil {
			s := remap(*x.PartIndex)
			x.PartIndex = &s
		}
		return x
	}
	return d
}
