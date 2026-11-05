package lm15

import (
	"errors"
	"testing"
)

// MAP-20 (lm15-contract changes/2026-11-05-stream-final-copy.md). The
// contract harness checks the SSE readers over every pinned stream and its
// variants; these pin what it cannot reach: the realtime completion path
// (WebSocket frames) and the refusal's part index.

func streamOf(t *testing.T, lm LM, req *Request, payloads ...string) *Response {
	t.Helper()
	body := ""
	for _, p := range payloads {
		body += "data: " + p + "\n\n"
	}
	events, err := ParseStreamBody(lm, req, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := MaterializeResponse(func(yield func(StreamEvent, error) bool) {
		for _, e := range events {
			if !yield(e, nil) {
				return
			}
		}
	}, req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func weatherRequest() *Request {
	return &Request{Model: "m", Messages: []Message{UserMessage("weather in Paris and Oslo")}, Tools: []Tool{weatherTool("")}}
}

func cities(t *testing.T, resp *Response) []string {
	t.Helper()
	var out []string
	for _, c := range resp.ToolCalls() {
		out = append(out, wireStr(c.Input.Get("city")))
	}
	return out
}

func TestArgumentsOnlyInTheFinalCopyReachTheResponse(t *testing.T) {
	lm, _ := NewOpenAILM(WithAPIKey("k"))
	resp := streamOf(t, lm, weatherRequest(),
		`{"type":"response.created","response":{"id":"r","model":"m"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"a","name":"get_weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"city\": \"Paris\"}"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"a","name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"b","name":"get_weather","arguments":""}}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"b","name":"get_weather","arguments":"{\"city\": \"Oslo\"}"}}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[{"type":"function_call"},{"type":"function_call"}]}}`)
	if got := cities(t, resp); len(got) != 2 || got[0] != "Paris" || got[1] != "Oslo" {
		t.Fatalf("calls: %v", got)
	}
}

func TestAContradictingFragmentIsRefused(t *testing.T) {
	lm, _ := NewOpenAILM(WithAPIKey("k"))
	body := "data: " + `{"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","call_id":"a","name":"get_weather","arguments":""}}` + "\n\n" +
		"data: " + `{"type":"response.function_call_arguments.delta","output_index":3,"delta":"{\"city\": \"Lima\"}"}` + "\n\n" +
		"data: " + `{"type":"response.output_item.done","output_index":3,"item":{"type":"function_call","call_id":"a","name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}` + "\n\n"
	_, err := ParseStreamBody(lm, weatherRequest(), []byte(body))
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindStreamAssembly || e.PartIndex == nil || *e.PartIndex != 3 {
		t.Fatalf("want StreamAssemblyError at part 3, got %v", err)
	}
}

func TestGeminiCallsInChunksOfTheirOwn(t *testing.T) {
	lm, _ := NewGeminiLM(WithAPIKey("k"))
	resp := streamOf(t, lm, weatherRequest(),
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}],"role":"model"}}]}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Oslo"}}}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}`)
	if got := cities(t, resp); len(got) != 2 || got[0] != "Paris" || got[1] != "Oslo" {
		t.Fatalf("calls: %v", got)
	}
}

func TestRealtimeCompletionCountsTheArgumentsOnce(t *testing.T) {
	lm, _ := NewOpenAILM(WithAPIKey("k"))
	frames := []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"get_weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"call_id":"c","delta":"{\"city\": "}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"call_id":"c","delta":"\"Montreal\"}"}`,
		`{"type":"response.function_call_arguments.done","output_index":0,"call_id":"c","arguments":"{\"city\": \"Montreal\"}"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"c","name":"get_weather","arguments":"{\"city\": \"Montreal\"}"}}`,
	}
	copies := newFinalCopies()
	input := ""
	for _, f := range frames {
		events, err := lm.decodeLiveCompletionEvents([]byte(f), copies)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range events {
			tc := e.(StreamDeltaEvent).Delta.(ToolCallDelta)
			if tc.Name == "tool" {
				t.Fatal("a frame without a name named the call \"tool\"")
			}
			input += tc.Input
		}
	}
	if input != `{"city": "Montreal"}` {
		t.Fatalf("input %q", input)
	}
}
