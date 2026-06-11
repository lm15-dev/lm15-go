package lm15

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMessageConstructors(t *testing.T) {
	m := UserMessage("hi")
	if m.Role != "user" || len(m.Parts) != 1 || m.Parts[0].(TextPart).Text != "hi" {
		t.Fatalf("UserMessage: %+v", m)
	}
	if a := AssistantMessage("yo"); a.Role != "assistant" {
		t.Fatalf("AssistantMessage role: %s", a.Role)
	}
	tr := ToolResults(ToolResult("call_1", "22C"))
	if tr.Role != "tool" {
		t.Fatalf("ToolResults role: %s", tr.Role)
	}
	p := tr.Parts[0].(ToolResultPart)
	if p.ID != "call_1" || p.Content[0].(TextPart).Text != "22C" {
		t.Fatalf("ToolResult: %+v", p)
	}
}

func TestResponseAccessors(t *testing.T) {
	resp := Response{Message: Message{Role: "assistant", Parts: []Part{
		ThinkingPart{Text: "hmm"},
		TextPart{Text: "hello"},
		ToolCallPart{ID: "c1", Name: "f", Input: jmap{}},
	}}}
	if resp.Text() != "" {
		t.Fatalf("Text() must be empty when non-text/citation/thinking parts present, got %q", resp.Text())
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].Name != "f" {
		t.Fatalf("ToolCalls: %+v", resp.ToolCalls())
	}
	textOnly := Response{Message: Message{Role: "assistant", Parts: []Part{
		ThinkingPart{Text: "hmm"}, TextPart{Text: "a"}, TextPart{Text: "b"},
	}}}
	if textOnly.Text() != "a\nb" {
		t.Fatalf("Text(): %q", textOnly.Text())
	}
}

func TestCompatPresetSetsBaseURL(t *testing.T) {
	lm, err := NewOpenAIChatLM("k", WithCompat(CompatOllama))
	if err != nil {
		t.Fatal(err)
	}
	if lm.baseURL != "http://localhost:11434/v1" {
		t.Fatalf("ollama base URL: %q", lm.baseURL)
	}
	lm2, err := NewOpenAIChatLM("k", WithCompat(CompatGroq), WithBaseURL("http://example.test/v1"))
	if err != nil {
		t.Fatal(err)
	}
	if lm2.baseURL != "http://example.test/v1" {
		t.Fatalf("explicit base URL must win: %q", lm2.baseURL)
	}
	if _, err := NewOpenAILM("k", WithCompat(CompatGroq)); err == nil {
		t.Fatal("WithCompat on OpenAILM must error")
	}
}

func TestClientCompleteAgainstFakeServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer k" {
			t.Errorf("auth header: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer srv.Close()
	lm, err := NewOpenAIChatLM("k", WithBaseURL(srv.URL+"/v1"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := lm.Complete(context.Background(), Request{Model: "m", Messages: []Message{UserMessage("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "hello there" || resp.FinishReason != "stop" {
		t.Fatalf("resp: %q %q", resp.Text(), resp.FinishReason)
	}
	if resp.Usage.TotalTokens == nil || *resp.Usage.TotalTokens != 5 {
		t.Fatalf("usage: %+v", resp.Usage)
	}
}

func TestClientCompleteErrorNormalized(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
		w.Write([]byte(`{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`))
	}))
	defer srv.Close()
	lm, _ := NewOpenAIChatLM("k", WithBaseURL(srv.URL+"/v1"))
	_, err := lm.Complete(context.Background(), Request{Model: "m", Messages: []Message{UserMessage("hi")}})
	le, ok := err.(LM15Error)
	if !ok || le.Code() != "auth" {
		t.Fatalf("want AuthError, got %T %v", err, err)
	}
}

// The MAP-3 gate: a chat stream with a finish chunk, a usage-only chunk and
// [DONE] must yield text deltas then EXACTLY ONE end event, last, merged.
func TestClientStreamSingleEndEvent(t *testing.T) {
	chunks := []string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
		`data: [DONE]`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			w.Write([]byte(c + "\n\n"))
		}
	}))
	defer srv.Close()
	lm, _ := NewOpenAIChatLM("k", WithBaseURL(srv.URL+"/v1"))
	var text strings.Builder
	ends := 0
	last := ""
	var endUsage *Usage
	for ev, err := range lm.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserMessage("hi")}}) {
		if err != nil {
			t.Fatal(err)
		}
		last = ev.StreamEventType()
		switch e := ev.(type) {
		case StreamDeltaEvent:
			if td, ok := e.Delta.(TextDelta); ok {
				text.WriteString(td.Text)
			}
		case StreamEndEvent:
			ends++
			endUsage = e.Usage
		}
	}
	if text.String() != "Hello" {
		t.Fatalf("text: %q", text.String())
	}
	if ends != 1 || last != "end" {
		t.Fatalf("MAP-3 violated: ends=%d last=%s", ends, last)
	}
	if endUsage == nil || endUsage.TotalTokens == nil || *endUsage.TotalTokens != 6 {
		t.Fatalf("end usage: %+v", endUsage)
	}
}

func TestClientStreamEarlyBreakCleansUp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for i := 0; i < 50; i++ {
			w.Write([]byte(`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}` + "\n\n"))
			f.Flush()
		}
	}))
	defer srv.Close()
	lm, _ := NewOpenAIChatLM("k", WithBaseURL(srv.URL+"/v1"))
	n := 0
	for ev, err := range lm.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserMessage("hi")}}) {
		if err != nil {
			t.Fatal(err)
		}
		_ = ev
		n++
		if n == 3 {
			break
		}
	}
	if n != 3 {
		t.Fatalf("n=%d", n)
	}
}
