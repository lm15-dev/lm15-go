package lm15

import (
	"strings"
)

// ─── Server-Sent Events parsing ──────────────────────────────────────
//
// Mirrors the reference SSE grammar (lm15.sse): `event:` names the event,
// `data:` lines accumulate (joined with "\n"), a blank line flushes, `:`
// comment lines are skipped, and oversized lines/events are transport
// errors.

// SSEEvent is one parsed server-sent event.
type SSEEvent struct {
	Event *string
	Data  string
}

const (
	sseMaxLineBytes  = 64 * 1024
	sseMaxEventBytes = 1024 * 1024
)

// ParseSSE parses a complete SSE body into events.
func ParseSSE(body []byte) ([]SSEEvent, error) {
	var out []SSEEvent
	var eventName *string
	var dataLines []string
	eventBytes := 0

	flush := func() {
		if len(dataLines) > 0 {
			out = append(out, SSEEvent{Event: eventName, Data: strings.Join(dataLines, "\n")})
		}
		eventName = nil
		dataLines = nil
		eventBytes = 0
	}

	for _, raw := range strings.SplitAfter(string(body), "\n") {
		if raw == "" {
			continue
		}
		if len(raw) > sseMaxLineBytes {
			return nil, TransportError{ErrorMeta{Message: "SSE line exceeds limit"}}
		}
		line := strings.TrimRight(raw, "\r\n")
		eventBytes += len(raw)
		if eventBytes > sseMaxEventBytes {
			return nil, TransportError{ErrorMeta{Message: "SSE event exceeds limit"}}
		}
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, ":"):
		case strings.HasPrefix(line, "event:"):
			name := strings.TrimSpace(line[len("event:"):])
			eventName = &name
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimLeft(line[len("data:"):], " \t\n\r\v\f"))
		}
	}
	flush()
	return out, nil
}
