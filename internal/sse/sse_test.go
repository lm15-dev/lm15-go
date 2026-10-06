package sse

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

// chunkReader hands out its body a few bytes per Read, like a network body.
type chunkReader struct {
	body []byte
	size int
}

func (c *chunkReader) Read(p []byte) (int, error) {
	if len(c.body) == 0 {
		return 0, io.EOF
	}
	n := c.size
	if n > len(p) {
		n = len(p)
	}
	if n > len(c.body) {
		n = len(c.body)
	}
	copy(p, c.body[:n])
	c.body = c.body[n:]
	return n, nil
}

func readAll(t *testing.T, r *Reader) []Event {
	t.Helper()
	var out []Event
	for {
		ev, err := r.Next()
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		out = append(out, ev)
	}
}

// INV-056: no default limit. The 64 KiB / 1 MiB defaults refused real
// streams (OpenAI Responses echoes the whole response; Gemini sends a 4K
// image as one 29.7 MB line).
func TestALineOverTheFormerLimitsParsesByDefault(t *testing.T) {
	text := strings.Repeat("x", 3*1024*1024)
	payload, _ := json.Marshal(map[string]string{"text": text})
	body := append(append([]byte("event: response.completed\ndata: "), payload...), "\n\n"...)

	events := readAll(t, NewReader(&chunkReader{body: body, size: 16 * 1024}, Limits{}))
	if len(events) != 1 || events[0].Name != "response.completed" {
		t.Fatalf("events = %d", len(events))
	}
	var got map[string]string
	if err := json.Unmarshal([]byte(events[0].Data), &got); err != nil || got["text"] != text {
		t.Fatalf("data did not round-trip: %v", err)
	}
	all, err := ParseAll(body)
	if err != nil || len(all) != 1 || all[0].Data != events[0].Data {
		t.Fatalf("ParseAll: %v", err)
	}
}

func TestCapsAreOptInAndStillRefuse(t *testing.T) {
	if _, _, err := NewParser(Limits{MaxLineBytes: 4}).Feed([]byte("data: too long\n")); !errors.Is(err, ErrLimit) {
		t.Fatalf("line cap: %v", err)
	}
	p := NewParser(Limits{MaxEventBytes: 8})
	if _, _, err := p.Feed([]byte("data: 1\n")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := p.Feed([]byte("data: 2\n")); !errors.Is(err, ErrLimit) {
		t.Fatalf("event cap: %v", err)
	}
}

// A cap the caller set is enforced while the line is still arriving: an
// endless line without '\n' stops at the cap, not at the end of memory.
func TestALineCapStopsAnUnterminatedLineEarly(t *testing.T) {
	endless := io.MultiReader(strings.NewReader("data: "), neverEnding('a'))
	_, err := NewReader(endless, Limits{MaxLineBytes: 1 << 20}).Next()
	if !errors.Is(err, ErrLimit) {
		t.Fatalf("err = %v", err)
	}
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func TestReaderMatchesParseAllForAnyChunking(t *testing.T) {
	pieces := []string{"a", "\n", "bc", "\r\n", "\n\n", "data: {}\n", "data: x\n\n", strings.Repeat("z", 300)}
	seed := uint32(56)
	rand := func(n int) int {
		seed = seed*1103515245 + 12345
		return int(seed>>8) % n
	}
	for trial := 0; trial < 300; trial++ {
		var b bytes.Buffer
		for k := rand(40); k > 0; k-- {
			b.WriteString(pieces[rand(len(pieces))])
		}
		body := b.Bytes()
		want, err := ParseAll(body)
		if err != nil {
			t.Fatal(err)
		}
		got := readAll(t, NewReader(&chunkReader{body: append([]byte(nil), body...), size: 1 + rand(7)}, Limits{}))
		if len(got) != len(want) {
			t.Fatalf("trial %d: %d events, want %d", trial, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d event %d: %+v, want %+v", trial, i, got[i], want[i])
			}
		}
	}
}

func TestAThirtyMegabyteLineReadsInLinearTime(t *testing.T) {
	body := append(bytes.Repeat([]byte("a"), 30*1024*1024), "\n\n"...)
	body = append([]byte("data: "), body...)
	started := time.Now()
	events := readAll(t, NewReader(&chunkReader{body: body, size: 16 * 1024}, Limits{}))
	if len(events) != 1 || len(events[0].Data) != 30*1024*1024 {
		t.Fatalf("events = %d", len(events))
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("took %v", elapsed)
	}
}
