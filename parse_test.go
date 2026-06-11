package lm15

import (
	"encoding/base64"
	"encoding/json"
	"testing"
)

func mustParse(t *testing.T, provider string, request jmap, body string) jmap {
	t.Helper()
	out, err := ParseResponseMap(provider, request, 200, base64.StdEncoding.EncodeToString([]byte(body)))
	if err != nil {
		t.Fatalf("ParseResponseMap(%s): %v", provider, err)
	}
	return out
}

func jsonEq(t *testing.T, got any, want string) {
	t.Helper()
	gotBytes, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var g, w any
	if err := json.Unmarshal(gotBytes, &g); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatal(err)
	}
	gn, _ := json.Marshal(g)
	wn, _ := json.Marshal(w)
	if string(gn) != string(wn) {
		t.Fatalf("mismatch:\n got: %s\nwant: %s", gn, wn)
	}
}

var basicRequest = jmap{
	"model": "m",
	"messages": []any{jmap{
		"role":  "user",
		"parts": []any{jmap{"type": "text", "text": "hi"}},
	}},
}

func TestParseAnthropicBasic(t *testing.T) {
	body := `{"id":"msg_1","model":"claude-x","content":[{"type":"text","text":"Hello"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":12,` +
		`"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}`
	out := mustParse(t, "anthropic", basicRequest, body)
	jsonEq(t, out["canonical_response"], `{
		"id":"msg_1","model":"claude-x",
		"message":{"role":"assistant","parts":[{"type":"text","text":"Hello"}],
			"continuation":[{"provider":"anthropic","kind":"message_id","data":{"id":"msg_1"}}]},
		"finish_reason":"stop",
		"usage":{"input_tokens":10,"output_tokens":12,"total_tokens":22,
			"cache_read_tokens":0,"cache_write_tokens":0}}`)
	if _, ok := out["unmapped"]; ok {
		t.Fatal("unexpected unmapped entries")
	}
}

func TestParseUnmappedCanary(t *testing.T) {
	body := `{"id":"msg_1","model":"claude-x","content":[{"type":"mystery_block"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	out := mustParse(t, "anthropic", basicRequest, body)
	jsonEq(t, out["unmapped"], `[{"path":"content[0]","type":"mystery_block"}]`)
}

func TestParseMap2NeverEmpty(t *testing.T) {
	// Gemini response with no candidate parts → single empty TextPart.
	body := `{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":5}}`
	out := mustParse(t, "gemini", basicRequest, body)
	resp := out["canonical_response"].(jmap)
	jsonEq(t, resp["message"], `{"role":"assistant","parts":[{"type":"text","text":""}]}`)
	if resp["finish_reason"] != "length" {
		t.Fatalf("finish_reason = %v, want length", resp["finish_reason"])
	}
}

func TestParseChatToolCall(t *testing.T) {
	body := `{"id":"cmpl-1","model":"gpt-x","choices":[{"message":{"tool_calls":[` +
		`{"id":"call_a","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]},` +
		`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`
	out := mustParse(t, "openai_chat", basicRequest, body)
	resp := out["canonical_response"].(jmap)
	jsonEq(t, resp["message"], `{"role":"assistant","parts":[
		{"type":"tool_call","id":"call_a","name":"f","input":{"x":1}}]}`)
	if resp["finish_reason"] != "tool_call" {
		t.Fatalf("finish_reason = %v, want tool_call", resp["finish_reason"])
	}
}

func TestParseOpenAIProviderExecutedSkipped(t *testing.T) {
	// MAP-1: provider-executed tool activity never becomes parts.
	body := `{"id":"resp_1","model":"gpt-x","status":"completed","output":[` +
		`{"type":"web_search_call","id":"ws_1"},` +
		`{"type":"message","content":[{"type":"output_text","text":"ok","annotations":[]}]}],` +
		`"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`
	out := mustParse(t, "openai", basicRequest, body)
	resp := out["canonical_response"].(jmap)
	jsonEq(t, resp["message"].(jmap)["parts"], `[{"type":"text","text":"ok"}]`)
	if _, ok := out["unmapped"]; ok {
		t.Fatal("provider-executed items must not be recorded as unmapped")
	}
}
