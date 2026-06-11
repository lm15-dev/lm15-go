package lm15

// Live smoke tests. CI-safe: every test skips under -short, when its API
// key env var is absent, or (ollama) when the local server isn't running.
// Keep spend tiny: small max_tokens everywhere.

import (
	"context"
	"net"
	"os"
	"strings"
	"testing"
	"time"
)

func liveKey(t *testing.T, name string) string {
	t.Helper()
	if testing.Short() {
		t.Skip("-short: skipping live test")
	}
	key := os.Getenv(name)
	if key == "" {
		t.Skipf("%s not set", name)
	}
	return key
}

func requireOllama(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("-short: skipping live test")
	}
	conn, err := net.DialTimeout("tcp", "localhost:11434", time.Second)
	if err != nil {
		t.Skip("ollama not reachable on localhost:11434")
	}
	conn.Close()
}

func liveCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func checkComplete(t *testing.T, lm interface {
	Complete(context.Context, Request) (Response, error)
}, req Request) Response {
	t.Helper()
	resp, err := lm.Complete(liveCtx(t), req)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Fatalf("complete: empty text, parts=%+v", resp.Message.Parts)
	}
	u := resp.Usage
	if u.InputTokens == nil || *u.InputTokens <= 0 || u.OutputTokens == nil || *u.OutputTokens <= 0 {
		t.Fatalf("complete: insane usage %+v", u)
	}
	t.Logf("complete text=%q usage in=%d out=%d", resp.Text(), *u.InputTokens, *u.OutputTokens)
	return resp
}

func checkStream(t *testing.T, lm *providerClient, req Request) {
	t.Helper()
	textDeltas, ends := 0, 0
	last := ""
	var text strings.Builder
	var endUsage *Usage
	for ev, err := range lm.Stream(liveCtx(t), req) {
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		last = ev.StreamEventType()
		switch e := ev.(type) {
		case StreamDeltaEvent:
			if td, ok := e.Delta.(TextDelta); ok {
				textDeltas++
				text.WriteString(td.Text)
			}
		case StreamEndEvent:
			ends++
			endUsage = e.Usage
		}
	}
	if textDeltas == 0 || strings.TrimSpace(text.String()) == "" {
		t.Fatalf("stream: no text deltas")
	}
	if ends != 1 || last != "end" {
		t.Fatalf("MAP-3 violated: ends=%d last=%s", ends, last)
	}
	if endUsage == nil || endUsage.OutputTokens == nil || *endUsage.OutputTokens <= 0 {
		t.Fatalf("stream end usage missing/insane: %+v", endUsage)
	}
	t.Logf("stream deltas=%d text=%q end usage out=%d", textDeltas, text.String(), *endUsage.OutputTokens)
}

// ── (a) local ollama via the openai_chat adapter ─────────────────────

func ollamaRequest() Request {
	return Request{
		Model:    "qwen3.5:0.8b",
		Messages: []Message{UserMessage("Say hello in five words or fewer.")},
		Config: &Config{
			MaxTokens:  i64(80),
			Extensions: jmap{"reasoning_effort": "none"},
		},
	}
}

func TestLiveOllamaComplete(t *testing.T) {
	requireOllama(t)
	lm, err := NewOpenAIChatLM("ollama", WithCompat(CompatOllama))
	if err != nil {
		t.Fatal(err)
	}
	checkComplete(t, lm, ollamaRequest())
}

func TestLiveOllamaStream(t *testing.T) {
	requireOllama(t)
	lm, err := NewOpenAIChatLM("ollama", WithCompat(CompatOllama))
	if err != nil {
		t.Fatal(err)
	}
	checkStream(t, &lm.providerClient, ollamaRequest())
}

// ── (b) Groq via the openai_chat adapter ─────────────────────────────

func groqRequest() Request {
	return Request{
		Model:    "llama-3.1-8b-instant",
		Messages: []Message{UserMessage("Say hello in two words.")},
		Config:   &Config{MaxTokens: i64(8)},
	}
}

func TestLiveGroqComplete(t *testing.T) {
	key := liveKey(t, "GROQ_API_KEY")
	lm, err := NewOpenAIChatLM(key, WithCompat(CompatGroq))
	if err != nil {
		t.Fatal(err)
	}
	checkComplete(t, lm, groqRequest())
}

func TestLiveGroqStream(t *testing.T) {
	key := liveKey(t, "GROQ_API_KEY")
	lm, err := NewOpenAIChatLM(key, WithCompat(CompatGroq))
	if err != nil {
		t.Fatal(err)
	}
	checkStream(t, &lm.providerClient, groqRequest())
}

// ── (c) first-party: OpenAI Responses API ────────────────────────────

func openAIRequest() Request {
	return Request{
		Model:    "gpt-4.1-mini",
		Messages: []Message{UserMessage("Say hello in two words.")},
		Config:   &Config{MaxTokens: i64(16)},
	}
}

func TestLiveOpenAIComplete(t *testing.T) {
	key := liveKey(t, "OPENAI_API_KEY")
	lm, err := NewOpenAILM(key)
	if err != nil {
		t.Fatal(err)
	}
	checkComplete(t, lm, openAIRequest())
}

func TestLiveOpenAIStream(t *testing.T) {
	key := liveKey(t, "OPENAI_API_KEY")
	lm, err := NewOpenAILM(key)
	if err != nil {
		t.Fatal(err)
	}
	checkStream(t, &lm.providerClient, openAIRequest())
}

// Full function-tool round-trip: model asks -> we answer -> final text.
func TestLiveOpenAIToolsRoundTrip(t *testing.T) {
	key := liveKey(t, "OPENAI_API_KEY")
	lm, err := NewOpenAILM(key)
	if err != nil {
		t.Fatal(err)
	}
	desc := "Get the current weather for a city."
	weather := FunctionTool{
		Name:        "get_weather",
		Description: &desc,
		Parameters: jmap{
			"type":       "object",
			"properties": jmap{"city": jmap{"type": "string"}},
			"required":   []any{"city"},
		},
	}
	req := Request{
		Model:    "gpt-4.1-mini",
		Messages: []Message{UserMessage("What is the weather in Montreal? Use the tool.")},
		Tools:    []Tool{weather},
		Config:   &Config{MaxTokens: i64(64)},
	}
	first, err := lm.Complete(liveCtx(t), req)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	calls := first.ToolCalls()
	if len(calls) == 0 {
		t.Fatalf("no typed tool call came back; finish=%s parts=%+v", first.FinishReason, first.Message.Parts)
	}
	call := calls[0]
	if call.Name != "get_weather" || call.ID == "" {
		t.Fatalf("tool call: %+v", call)
	}
	if _, ok := call.Input["city"]; !ok {
		t.Fatalf("tool call input missing city: %+v", call.Input)
	}
	t.Logf("tool_call id=%s name=%s input=%v finish=%s", call.ID, call.Name, call.Input, first.FinishReason)

	req.Messages = append(req.Messages,
		first.Message,
		ToolResults(ToolResult(call.ID, "Sunny, 22C")),
	)
	final, err := lm.Complete(liveCtx(t), req)
	if err != nil {
		t.Fatalf("second turn: %v", err)
	}
	if strings.TrimSpace(final.Text()) == "" {
		t.Fatalf("no final text after tool result; parts=%+v", final.Message.Parts)
	}
	if !strings.Contains(strings.ToLower(final.Text()), "22") && !strings.Contains(strings.ToLower(final.Text()), "sunny") {
		t.Logf("warning: final text doesn't echo tool result: %q", final.Text())
	}
	t.Logf("final text=%q", final.Text())
}
