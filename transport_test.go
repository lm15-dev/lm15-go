package lm15

import (
	"strings"
	"testing"
)

func TestParseSSE(t *testing.T) {
	input := "data: hello\n\ndata: world\n\n"
	ch := ParseSSE(strings.NewReader(input))

	events := make([]SSEEvent, 0)
	for e := range ch {
		events = append(events, e)
	}

	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}
	if events[0].Data != "hello" {
		t.Errorf("expected hello, got %s", events[0].Data)
	}
	if events[1].Data != "world" {
		t.Errorf("expected world, got %s", events[1].Data)
	}
}

func TestParseSSENamedEvents(t *testing.T) {
	input := "event: message\ndata: hi\n\n"
	ch := ParseSSE(strings.NewReader(input))

	e := <-ch
	if e.Event != "message" || e.Data != "hi" {
		t.Errorf("unexpected: %+v", e)
	}
}

func TestParseSSEMultiLine(t *testing.T) {
	input := "data: line1\ndata: line2\n\n"
	ch := ParseSSE(strings.NewReader(input))

	e := <-ch
	if e.Data != "line1\nline2" {
		t.Errorf("expected multi-line data, got %q", e.Data)
	}
}

func TestParseSSEComments(t *testing.T) {
	input := ": comment\ndata: hi\n\n"
	ch := ParseSSE(strings.NewReader(input))

	e := <-ch
	if e.Data != "hi" {
		t.Errorf("expected hi, got %s", e.Data)
	}
}

func TestParseSSEDone(t *testing.T) {
	input := "data: {\"text\":\"hi\"}\n\ndata: [DONE]\n\n"
	ch := ParseSSE(strings.NewReader(input))

	events := make([]SSEEvent, 0)
	for e := range ch {
		events = append(events, e)
	}

	if len(events) != 2 || events[1].Data != "[DONE]" {
		t.Errorf("unexpected events: %+v", events)
	}
}
