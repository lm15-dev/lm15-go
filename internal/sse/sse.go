// Package sse parses Server-Sent Events from a byte stream.
package sse

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Event is one SSE event: the optional event name and the joined data lines.
type Event struct {
	Name string
	Data string
}

// Limits bound a single line and a single event (defaults 64 KiB / 1 MiB).
type Limits struct {
	MaxLineBytes  int
	MaxEventBytes int
}

// DefaultLimits are the reference's limits.
var DefaultLimits = Limits{MaxLineBytes: 64 * 1024, MaxEventBytes: 1024 * 1024}

// ErrLimit is returned when a line or event exceeds its limit.
var ErrLimit = errors.New("sse: limit exceeded")

// Parser reads events from lines.
type Parser struct {
	limits    Limits
	eventName string
	dataLines []string
	eventLen  int
}

// NewParser creates a parser with the given limits (zero = defaults).
func NewParser(limits Limits) *Parser {
	if limits.MaxLineBytes == 0 {
		limits.MaxLineBytes = DefaultLimits.MaxLineBytes
	}
	if limits.MaxEventBytes == 0 {
		limits.MaxEventBytes = DefaultLimits.MaxEventBytes
	}
	return &Parser{limits: limits}
}

// Feed consumes one raw line (with or without its terminator) and returns
// the completed event, if this line closed one.
func (p *Parser) Feed(raw []byte) (Event, bool, error) {
	if len(raw) > p.limits.MaxLineBytes {
		return Event{}, false, fmt.Errorf("%w: SSE line exceeds limit (%d > %d)", ErrLimit, len(raw), p.limits.MaxLineBytes)
	}
	p.eventLen += len(raw)
	if p.eventLen > p.limits.MaxEventBytes {
		return Event{}, false, fmt.Errorf("%w: SSE event exceeds limit (%d > %d)", ErrLimit, p.eventLen, p.limits.MaxEventBytes)
	}
	line := strings.TrimRight(string(raw), "\r\n")
	if line == "" {
		ev, ok := p.flush()
		return ev, ok, nil
	}
	switch {
	case strings.HasPrefix(line, ":"):
	case strings.HasPrefix(line, "event:"):
		p.eventName = strings.TrimSpace(line[len("event:"):])
	case strings.HasPrefix(line, "data:"):
		p.dataLines = append(p.dataLines, strings.TrimLeft(line[len("data:"):], " \t"))
	}
	return Event{}, false, nil
}

// Flush returns the pending event at end of input, if any.
func (p *Parser) Flush() (Event, bool) { return p.flush() }

func (p *Parser) flush() (Event, bool) {
	defer func() {
		p.eventName = ""
		p.dataLines = nil
		p.eventLen = 0
	}()
	if len(p.dataLines) == 0 {
		return Event{}, false
	}
	return Event{Name: p.eventName, Data: strings.Join(p.dataLines, "\n")}, true
}

// ParseAll parses a complete body into events.
func ParseAll(body []byte) ([]Event, error) {
	p := NewParser(Limits{})
	var out []Event
	for _, line := range splitLines(body) {
		ev, ok, err := p.Feed(line)
		if err != nil {
			return out, err
		}
		if ok {
			out = append(out, ev)
		}
	}
	if ev, ok := p.Flush(); ok {
		out = append(out, ev)
	}
	return out, nil
}

// splitLines splits keeping terminators (like Python's splitlines(keepends=True) on \n).
func splitLines(body []byte) [][]byte {
	var out [][]byte
	for len(body) > 0 {
		idx := bytes.IndexByte(body, '\n')
		if idx < 0 {
			out = append(out, body)
			break
		}
		out = append(out, body[:idx+1])
		body = body[idx+1:]
	}
	return out
}

// Reader yields events from an io.Reader as they arrive.
type Reader struct {
	scanner *bufio.Reader
	parser  *Parser
	done    bool
}

// NewReader wraps a streaming body.
func NewReader(r io.Reader, limits Limits) *Reader {
	return &Reader{scanner: bufio.NewReaderSize(r, 64*1024), parser: NewParser(limits)}
}

// Next returns the next event; io.EOF when the body is exhausted.
func (r *Reader) Next() (Event, error) {
	for !r.done {
		line, err := r.scanner.ReadBytes('\n')
		if len(line) > 0 {
			ev, ok, ferr := r.parser.Feed(line)
			if ferr != nil {
				return Event{}, ferr
			}
			if ok {
				return ev, nil
			}
		}
		if err != nil {
			r.done = true
			if err != io.EOF {
				return Event{}, err
			}
		}
	}
	if ev, ok := r.parser.Flush(); ok {
		return ev, nil
	}
	return Event{}, io.EOF
}
