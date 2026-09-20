package lm15

// MAP-13 client-side stop.
//
// A wire with no stop field (OpenAI Responses) gets the sequence applied
// here: the visible text is cut at the first occurrence, the finish reason
// becomes "stop", and the source is closed at the cut. Whether the
// provider then stops generating (and billing) on a closed connection is
// the provider's behaviour, not a promise lm15 can make; what lm15 does
// promise is that the end event carries no usage after a cut (the final
// frame was never read) — "not reported", never estimated.
//
// Scores (changes/2026-09-15-stop-filter-score-preservation.md): original
// events pass through unchanged until their text is safe; only the event
// actually cut is rebuilt. Whole tokens entirely before the cut keep their
// scores; a token the cut splits loses its score and LogprobsIncomplete
// is set; nothing is ever split, rescaled or invented.

import (
	"iter"
	"strings"
)

// firstStop finds the earliest occurrence of any stop sequence in text.
func firstStop(text string, stop []string) (int, string, bool) {
	best, bestSeq, found := 0, "", false
	for _, seq := range stop {
		if seq == "" {
			continue
		}
		if idx := strings.Index(text, seq); idx >= 0 && (!found || idx < best) {
			best, bestSeq, found = idx, seq, true
		}
	}
	return best, bestSeq, found
}

// scoresBeforeCut keeps original scores for whole retained tokens; never
// scores a token fragment. Byte boundaries matter: a provider token can
// itself contain only part of a Unicode character. Token spellings are
// used only when their concatenated UTF-8 bytes exactly reproduce the
// original text; otherwise the scores cannot be placed and are omitted
// with coverage marked incomplete.
func scoresBeforeCut(scores []TokenLogprob, text string, cutAt int) ([]TokenLogprob, bool) {
	if len(scores) == 0 || cutAt == 0 {
		return nil, false
	}
	if cutAt >= len(text) {
		return scores, false
	}
	tokenBytes := make([][]byte, len(scores))
	total := 0
	for i, s := range scores {
		if s.Bytes != nil {
			b := make([]byte, len(s.Bytes))
			for j, v := range s.Bytes {
				if v < 0 || v > 255 {
					return nil, true
				}
				b[j] = byte(v)
			}
			tokenBytes[i] = b
		} else {
			tokenBytes[i] = []byte(s.Token)
		}
		total += len(tokenBytes[i])
	}
	if total != len(text) {
		return nil, true
	}
	joined := make([]byte, 0, total)
	for _, b := range tokenBytes {
		joined = append(joined, b...)
	}
	if string(joined) != text {
		return nil, true
	}
	end := 0
	for i, b := range tokenBytes {
		if end == cutAt {
			return scores[:i], false
		}
		end += len(b)
		if end > cutAt {
			return scores[:i], true
		}
	}
	return scores, false
}

// ApplyClientSideStop cuts the response's visible text at the first stop
// sequence. The text parts are one stream in document order: a sequence
// that starts at the end of one part and finishes at the start of the next
// is a hit. The part holding the start is cut there; every later part is
// removed. Scores follow the rule above.
func ApplyClientSideStop(response *Response, stop []string) *Response {
	if len(stop) == 0 || response == nil {
		return response
	}
	type textAt struct {
		index int
		part  TextPart
	}
	var textParts []textAt
	var joined strings.Builder
	for i, p := range response.Message.Parts {
		if t, ok := p.(TextPart); ok {
			textParts = append(textParts, textAt{i, t})
			joined.WriteString(t.Text)
		}
	}
	if len(textParts) == 0 {
		return response
	}
	text := joined.String()
	hit, _, found := firstStop(text, stop)
	if !found {
		return response
	}
	offset := 0
	cutIndex, cutAt := textParts[len(textParts)-1].index, 0
	for _, tp := range textParts {
		if offset+len(tp.part.Text) > hit {
			cutIndex, cutAt = tp.index, hit-offset
			break
		}
		offset += len(tp.part.Text)
	}
	var parts []Part
	for i, p := range response.Message.Parts {
		if i > cutIndex {
			break
		}
		if i == cutIndex {
			t := p.(TextPart)
			t.Text = t.Text[:cutAt]
			parts = append(parts, t)
		} else {
			parts = append(parts, p)
		}
	}
	if len(parts) == 0 {
		parts = []Part{TextPart{Text: ""}}
	}
	out := *response
	out.Message = Message{Role: response.Message.Role, Parts: parts, Continuation: response.Message.Continuation}
	out.FinishReason = FinishStop
	scores, incomplete := scoresBeforeCut(response.Logprobs, text, hit)
	out.Logprobs = scores
	out.LogprobsIncomplete = response.LogprobsIncomplete || incomplete
	return &out
}

// stopCutter keeps original events until their text is safe, then passes
// them unchanged. Text deltas form one text stream. A possible stop suffix
// holds its entire event, plus intervening events to preserve order. Only
// the event actually cut is rebuilt.
type stopCutter struct {
	stop     []string
	hold     int
	segments []StreamEvent
	cut      bool
}

func newStopCutter(stop []string) *stopCutter {
	c := &stopCutter{}
	for _, s := range stop {
		if s != "" {
			c.stop = append(c.stop, s)
		}
	}
	longest := 1
	for _, s := range c.stop {
		if len(s) > longest {
			longest = len(s)
		}
	}
	c.hold = longest - 1
	return c
}

func textDeltaOf(event StreamEvent) (TextDelta, bool) {
	de, ok := event.(StreamDeltaEvent)
	if !ok {
		return TextDelta{}, false
	}
	td, ok := de.Delta.(TextDelta)
	return td, ok
}

// take releases count text characters (bytes) from the head of the queue,
// together with the non-text events between them.
func (c *stopCutter) take(count int, cutting bool) []StreamEvent {
	var out []StreamEvent
	for len(c.segments) > 0 {
		event := c.segments[0]
		delta, isText := textDeltaOf(event)
		if !isText {
			out = append(out, event)
			c.segments = c.segments[1:]
			continue
		}
		if cutting && count == 0 {
			break
		}
		if len(delta.Text) <= count {
			out = append(out, event)
			c.segments = c.segments[1:]
			count -= len(delta.Text)
		} else if cutting {
			scores, incomplete := scoresBeforeCut(delta.Logprobs, delta.Text, count)
			delta.Text = delta.Text[:count]
			delta.Logprobs = scores
			delta.LogprobsIncomplete = delta.LogprobsIncomplete || incomplete
			out = append(out, StreamDeltaEvent{Delta: delta})
			break
		} else {
			break
		}
	}
	return out
}

func (c *stopCutter) feed(event StreamEvent) []StreamEvent {
	if _, isText := textDeltaOf(event); !isText {
		if len(c.segments) == 0 {
			return []StreamEvent{event}
		}
		c.segments = append(c.segments, event)
		return nil
	}
	c.segments = append(c.segments, event)
	var buf strings.Builder
	for _, e := range c.segments {
		if td, ok := textDeltaOf(e); ok {
			buf.WriteString(td.Text)
		}
	}
	text := buf.String()
	if hit, _, found := firstStop(text, c.stop); found {
		out := c.take(hit, true)
		c.segments = nil
		c.cut = true
		return out
	}
	release := len(text) - c.hold
	if release < 0 {
		release = 0
	}
	return c.take(release, false)
}

func (c *stopCutter) flush() []StreamEvent {
	out := c.segments
	c.segments = nil
	return out
}

// TruncateStreamAtStop applies a client-side stop to a canonical stream:
// text is cut at the first sequence, a synthesized end event (finish_reason
// "stop", no usage) follows, and the source is closed at the cut.
func TruncateStreamAtStop(events iter.Seq2[StreamEvent, error], stop []string) iter.Seq2[StreamEvent, error] {
	cutter := newStopCutter(stop)
	if len(cutter.stop) == 0 {
		return events
	}
	return func(yield func(StreamEvent, error) bool) {
		for event, err := range events {
			if err != nil {
				yield(nil, err)
				return
			}
			var out []StreamEvent
			switch event.(type) {
			case StreamEndEvent, StreamErrorEvent:
				out = append(cutter.flush(), event)
			default:
				out = cutter.feed(event)
			}
			for _, e := range out {
				if !yield(e, nil) {
					return
				}
			}
			if cutter.cut {
				yield(StreamEndEvent{FinishReason: FinishStop}, nil)
				return // returning closes the source at the cut
			}
		}
	}
}
