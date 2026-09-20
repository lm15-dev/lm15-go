package lm15

// Bounded live-turn collection (changes/2026-09-15-live-collection-limits.md).
//
// A turn view collects the server events of one turn until its boundary.
// A peer that never emits a boundary would otherwise grow that collection
// without end, so every view has a byte and an event budget; on overflow
// it fails with a local, non-retryable CollectionLimitError that keeps
// every accepted event (and, on a byte overflow, the event that did not
// fit) for the application to process, continue from with raw reads, or
// abandon. The session stays open, under application control.

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"unicode/utf8"
)

// The shared defaults: a round conservative payload budget for an
// explicitly buffering convenience, not a measured percentile.
const (
	DefaultTurnMaxBytes  = 16 * 1024 * 1024
	DefaultTurnMaxEvents = 10000
)

// TurnLimits bounds one turn view's collection: MaxBytes charges the
// compact ASCII JSON of every accepted canonical event; MaxEvents counts
// them. Zero takes the default; both must be positive.
type TurnLimits struct {
	MaxBytes  int
	MaxEvents int
}

func (l TurnLimits) withDefaults() (TurnLimits, error) {
	if l.MaxBytes == 0 {
		l.MaxBytes = DefaultTurnMaxBytes
	}
	if l.MaxEvents == 0 {
		l.MaxEvents = DefaultTurnMaxEvents
	}
	if l.MaxBytes < 0 || l.MaxEvents < 0 {
		return l, valueErrorf("turn limits must be positive integers, got max_bytes=%d max_events=%d", l.MaxBytes, l.MaxEvents)
	}
	return l, nil
}

// ─── The byte charge ─────────────────────────────────────────────────

// eventByteCharge is the length of the event serialized to compact ASCII
// JSON: no optional whitespace, separators "," and ":", quotes and
// backslashes escaped, the standard short control escapes, every other
// control and non-ASCII code unit as \uXXXX (surrogate pairs above the
// BMP), "/" unescaped. Omission and number rules are the canonical serde
// rules. Key order does not affect the size. Counting stops once the
// budget is exceeded (limit > 0) — no second copy of the history is built.
func eventByteCharge(event LiveServerEvent, limit int) int {
	return asciiJSONSize(LiveServerEventToDict(event), limit)
}

func asciiJSONSize(v any, limit int) int {
	n := 0
	var walk func(v any) bool // false = stop, over the limit
	walk = func(v any) bool {
		switch x := v.(type) {
		case nil:
			n += 4
		case bool:
			if x {
				n += 4
			} else {
				n += 5
			}
		case string:
			n += asciiStringSize(x)
		case json.Number:
			n += len(x.String())
		case int:
			n += len(strconv.Itoa(x))
		case int64:
			n += len(strconv.FormatInt(x, 10))
		case float64:
			b, _ := jsonFloat(x).MarshalJSON()
			n += len(b)
		case jsonFloat:
			b, _ := x.MarshalJSON()
			n += len(b)
		case map[string]any:
			n += 2 // {}
			first := true
			for k, val := range x {
				if !first {
					n++ // ,
				}
				first = false
				n += asciiStringSize(k) + 1 // key and :
				if !walk(val) {
					return false
				}
			}
		case []any:
			n += 2
			for i, val := range x {
				if i > 0 {
					n++
				}
				if !walk(val) {
					return false
				}
			}
		default:
			rv := reflect.ValueOf(v)
			switch rv.Kind() {
			case reflect.Slice, reflect.Array:
				n += 2
				for i := 0; i < rv.Len(); i++ {
					if i > 0 {
						n++
					}
					if !walk(rv.Index(i).Interface()) {
						return false
					}
				}
			case reflect.Map:
				n += 2
				first := true
				iter := rv.MapRange()
				for iter.Next() {
					if !first {
						n++
					}
					first = false
					n += asciiStringSize(fmt.Sprint(iter.Key().Interface())) + 1
					if !walk(iter.Value().Interface()) {
						return false
					}
				}
			case reflect.Pointer, reflect.Interface:
				if rv.IsNil() {
					n += 4
				} else if !walk(rv.Elem().Interface()) {
					return false
				}
			default:
				b, err := EncodeJSON(v)
				if err == nil {
					n += len(b)
				}
			}
		}
		return limit <= 0 || n <= limit
	}
	walk(v)
	return n
}

// asciiStringSize is the length of a JSON string literal under the
// compact ASCII rule (quotes included).
func asciiStringSize(s string) int {
	n := 2
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == '"' || r == '\\' || r == '\n' || r == '\r' || r == '\t' || r == '\b' || r == '\f':
			n += 2
		case r < 0x20 || r == 0x7f:
			n += 6
		case r < 0x80:
			n++
		case r == utf8.RuneError && size == 1:
			n += 6 // an invalid byte would be a \ufffd on the wire
		case r > 0xFFFF:
			n += 12 // a surrogate pair
		default:
			n += 6
		}
		i += size
	}
	return n
}

// ─── The view ────────────────────────────────────────────────────────

// eventSource is what a view reads from: the session's Recv.
type eventSource interface {
	Recv(ctx context.Context) (LiveServerEvent, error)
}

// TurnView collects one turn under a budget. Iterate with Next until ok
// is false, or call Result for the materialized Turn; both stop at the
// turn's boundary (turn_end, interrupted, error) or a tool_call the
// caller must answer (LIVE-1). A view that overflowed is sealed: every
// later call returns the same CollectionLimitError, even after Close.
// Closing a view never closes the session.
type TurnView struct {
	source        eventSource
	limits        TurnLimits
	accepted      []LiveServerEvent
	retainedBytes int
	done          bool
	failure       *Error
	turn          *Turn
}

// NewTurnView creates a view over a session (or any event source) with
// the given limits (zero fields take the defaults).
func NewTurnView(source eventSource, limits TurnLimits) (*TurnView, error) {
	effective, err := limits.withDefaults()
	if err != nil {
		return nil, err
	}
	return &TurnView{source: source, limits: effective}, nil
}

func (v *TurnView) collectionError(limit string, maximum int, rejected LiveServerEvent) *Error {
	// Never payload text in the message: the numbers say what happened.
	e := newError(KindCollectionLimit, fmt.Sprintf("live turn collection reached its %s budget (%d): %d events, %d bytes retained; the session is still open — process the accepted events (and the rejected one, if any), continue with raw reads, interrupt, or close", limit, maximum, len(v.accepted), v.retainedBytes))
	e.Limit = limit
	e.Maximum = maximum
	e.RetainedBytes = v.retainedBytes
	e.PartialEvents = append([]LiveServerEvent(nil), v.accepted...)
	e.RejectedEvent = rejected
	return e
}

// Next receives one event under the budget. ok is false at the turn's
// boundary (after the boundary event itself was returned) or on failure.
func (v *TurnView) Next(ctx context.Context) (event LiveServerEvent, ok bool, err error) {
	if v.failure != nil {
		return nil, false, v.failure
	}
	if v.done {
		return nil, false, nil
	}
	// Before receiving another event, a reached event cap fails without
	// consuming that next event: a cap is a cap.
	if len(v.accepted) >= v.limits.MaxEvents {
		v.failure = v.collectionError("max_events", v.limits.MaxEvents, nil)
		return nil, false, v.failure
	}
	ev, err := v.source.Recv(ctx)
	if err != nil {
		return nil, false, err
	}
	// Determine whether the byte charge fits, and only then append.
	remaining := v.limits.MaxBytes - v.retainedBytes
	charge := eventByteCharge(ev, remaining)
	if charge > remaining {
		v.failure = v.collectionError("max_bytes", v.limits.MaxBytes, ev)
		return nil, false, v.failure
	}
	v.accepted = append(v.accepted, ev)
	v.retainedBytes += charge
	switch ev.Type() {
	case "turn_end", "interrupted", "error", "tool_call":
		v.done = true
	}
	return ev, true, nil
}

// Result drains the view to the turn's boundary and returns the Turn
// (the same one on repeat).
func (v *TurnView) Result(ctx context.Context) (*Turn, error) {
	if v.turn != nil {
		return v.turn, nil
	}
	for {
		_, ok, err := v.Next(ctx)
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
	}
	v.turn = MaterializeTurn(v.accepted)
	return v.turn, nil
}

// Snapshot materializes the accepted events so far, on demand, as an
// incomplete Turn (ok=false) unless the boundary was reached.
func (v *TurnView) Snapshot() *Turn {
	t := MaterializeTurn(v.accepted)
	if !v.done || v.failure != nil {
		t.EndedBy = "incomplete"
	}
	return t
}

// Events are the accepted events so far, in order.
func (v *TurnView) Events() []LiveServerEvent { return append([]LiveServerEvent(nil), v.accepted...) }

// RetainedBytes is the byte charge of the accepted events.
func (v *TurnView) RetainedBytes() int { return v.retainedBytes }

// Close seals the view; it never closes the session.
func (v *TurnView) Close() error {
	v.done = true
	return nil
}

// PartialTurn materializes a CollectionLimitError's accepted events as an
// incomplete Turn, on demand (raw events are always available on the
// error without decoding).
func (e *Error) PartialTurn() *Turn {
	if e == nil || !e.Kind.IsA(KindCollectionLimit) {
		return nil
	}
	t := MaterializeTurn(e.PartialEvents)
	t.EndedBy = "incomplete"
	return t
}
