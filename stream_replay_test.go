package lm15

import (
	"strings"
	"testing"
)

func mustReplay(t *testing.T, provider, body string) ([]StreamEvent, Response) {
	t.Helper()
	req := Request{Model: "m", Config: &Config{}}
	events, resp, err := ReplayStream(provider, req, []byte(body))
	if err != nil {
		t.Fatalf("ReplayStream: %v", err)
	}
	return events, resp
}

func TestParseSSEGrammar(t *testing.T) {
	body := ": comment\nevent: foo\ndata: {\"a\":1}\ndata: more\n\ndata: tail"
	events, err := ParseSSE([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 {
		t.Fatalf("want 2 events, got %d", len(events))
	}
	if events[0].Event == nil || *events[0].Event != "foo" {
		t.Errorf("event name not parsed: %+v", events[0])
	}
	if events[0].Data != "{\"a\":1}\nmore" {
		t.Errorf("multi-line data join wrong: %q", events[0].Data)
	}
	if events[1].Data != "tail" {
		t.Errorf("trailing unflushed event lost: %q", events[1].Data)
	}
}

func TestParseSSELineLimit(t *testing.T) {
	long := "data: " + strings.Repeat("x", sseMaxLineBytes+1) + "\n\n"
	if _, err := ParseSSE([]byte(long)); err == nil {
		t.Fatal("expected line-limit TransportError")
	}
}

// MAP-3: vLLM/SGLang/Groq send finish_reason chunk, then a post-finish
// usage-only chunk, then [DONE] — the trace must end with exactly one
// merged StreamEndEvent carrying both.
func TestChatPostFinishUsageCoalesced(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"hi"},"index":0}]}`,
		``,
		`data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}`,
		``,
		`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":7,"total_tokens":10}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	events, resp := mustReplay(t, "openai_chat", body)
	endCount := 0
	for _, e := range events {
		if _, ok := e.(StreamEndEvent); ok {
			endCount++
		}
	}
	if endCount != 1 {
		t.Fatalf("MAP-3 violated: %d end events", endCount)
	}
	end, ok := events[len(events)-1].(StreamEndEvent)
	if !ok {
		t.Fatal("end event is not final")
	}
	if end.FinishReason == nil || *end.FinishReason != "stop" {
		t.Errorf("finish_reason lost: %+v", end.FinishReason)
	}
	if end.Usage == nil || end.Usage.InputTokens == nil || *end.Usage.InputTokens != 3 ||
		*end.Usage.OutputTokens != 7 {
		t.Errorf("post-finish usage chunk not absorbed: %+v", end.Usage)
	}
	if resp.Usage.TotalTokens == nil || *resp.Usage.TotalTokens != 10 {
		t.Errorf("materialized usage wrong: %+v", resp.Usage)
	}
	if got := resp.Message.Parts[0].(TextPart).Text; got != "hi" {
		t.Errorf("text wrong: %q", got)
	}
}

// Anthropic splits terminal data across message_delta + message_stop.
func TestAnthropicMessageDeltaThenStop(t *testing.T) {
	body := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":{"input_tokens":4}}}`,
		``,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":4,"output_tokens":6}}`,
		``,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	events, resp := mustReplay(t, "anthropic", body)
	end := events[len(events)-1].(StreamEndEvent)
	if end.FinishReason == nil || *end.FinishReason != "stop" {
		t.Errorf("finish: %+v", end.FinishReason)
	}
	if end.Usage == nil || *end.Usage.TotalTokens != 10 {
		t.Errorf("usage: %+v", end.Usage)
	}
	if resp.ID == nil || *resp.ID != "msg_1" || resp.Model != "claude-x" {
		t.Errorf("start fields lost: %+v", resp)
	}
	if len(resp.Message.Continuation) != 1 || resp.Message.Continuation[0].Kind != "message_id" {
		t.Errorf("message_id continuation missing: %+v", resp.Message.Continuation)
	}
}

// Tool-call deltas accumulate raw JSON across chunks; name falls back to
// the single declared function tool; finish "stop" upgrades to tool_call.
func TestToolCallAccumulationAndFinishUpgrade(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\"a\":"}}]},"index":0}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"index":0}]}`,
		``,
		`data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	req := Request{Model: "m", Config: &Config{}, Tools: []Tool{FunctionTool{Name: "get_weather", Parameters: DefaultParameters()}}}
	events, resp, err := ReplayStream("openai_chat", req, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	_ = events
	tc, ok := resp.Message.Parts[0].(ToolCallPart)
	if !ok {
		t.Fatalf("expected tool_call part, got %T", resp.Message.Parts[0])
	}
	if tc.ID != "call_1" || tc.Name != "get_weather" {
		t.Errorf("tool meta: %+v", tc)
	}
	if tc.Input["a"] == nil {
		t.Errorf("split-chunk input not parsed: %+v", tc.Input)
	}
	if resp.FinishReason != "tool_call" {
		t.Errorf("finish not upgraded: %s", resp.FinishReason)
	}
}

// MAP-2: a stream with no content deltas materializes one empty TextPart.
func TestEmptyStreamMaterializesEmptyText(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\",\"index\":0}]}\n\ndata: [DONE]\n\n"
	_, resp := mustReplay(t, "openai_chat", body)
	if len(resp.Message.Parts) != 1 {
		t.Fatalf("parts: %+v", resp.Message.Parts)
	}
	if tp, ok := resp.Message.Parts[0].(TextPart); !ok || tp.Text != "" {
		t.Errorf("MAP-2 fallback wrong: %+v", resp.Message.Parts[0])
	}
	if resp.FinishReason != "length" {
		t.Errorf("finish: %s", resp.FinishReason)
	}
}

// Gemini: per-chunk responseId continuations, usage on the finish chunk,
// reasoning token accounting.
func TestGeminiStreamFinishCarriesUsage(t *testing.T) {
	body := strings.Join([]string{
		`data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":9,"totalTokenCount":33,"thoughtsTokenCount":21},"responseId":"r1"}`,
		``,
	}, "\n")
	events, resp := mustReplay(t, "gemini", body)
	end := events[len(events)-1].(StreamEndEvent)
	if end.Usage == nil || *end.Usage.ReasoningTokens != 21 || *end.Usage.TotalTokens != 33 {
		t.Errorf("usage: %+v", end.Usage)
	}
	if got := resp.Message.Parts[0].(TextPart).Text; got != "Hello!" {
		t.Errorf("text: %q", got)
	}
	if len(resp.Message.Continuation) != 2 || resp.Message.Continuation[0].Kind != "response_id" {
		t.Errorf("continuations: %+v", resp.Message.Continuation)
	}
}

// OpenAI Responses: [DONE] re-emits finish "stop" after response.completed;
// the coalescer keeps it last-wins, the materializer restores tool_call.
func TestOpenAIResponsesStream(t *testing.T) {
	body := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-x"}}`,
		``,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"hey"}`,
		``,
		`data: {"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":2,"output_tokens":5,"total_tokens":7}}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	events, resp := mustReplay(t, "openai", body)
	if _, ok := events[0].(StreamStartEvent); !ok {
		t.Fatalf("first event not start: %T", events[0])
	}
	end := events[len(events)-1].(StreamEndEvent)
	if end.Usage == nil || *end.Usage.TotalTokens != 7 {
		t.Errorf("usage: %+v", end.Usage)
	}
	if resp.ID == nil || *resp.ID != "resp_1" || resp.Model != "gpt-x" {
		t.Errorf("identity: %+v", resp)
	}
}

// Stream error frames become StreamErrorEvent with canonical codes; no end
// event is fabricated.
func TestStreamErrorNoFabricatedEnd(t *testing.T) {
	body := "data: {\"error\":{\"code\":\"invalid_api_key\",\"message\":\"bad key\"}}\n\n"
	events, _ := mustReplay(t, "openai_chat", body)
	if len(events) != 1 {
		t.Fatalf("events: %+v", events)
	}
	ev, ok := events[0].(StreamErrorEvent)
	if !ok {
		t.Fatalf("not error event: %T", events[0])
	}
	if ev.Error.Code != "auth" || *ev.Error.ProviderCode != "invalid_api_key" {
		t.Errorf("error detail: %+v", ev.Error)
	}
}
