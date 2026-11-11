package lm15

// MAP-24 (lm15-contract changes/2026-11-11-empty-parts.md, issue #3): a part
// with nothing for a wire writes nothing. The contract's vectors
// (mapping/empty-parts.json) and the history replay pin the bytes for the
// four dialects through the vet shim. These pin what they do not: nothing is
// recorded, the caller's request is not touched, the state comes back on its
// own dialect, and Gemini's cached prefix and live turns follow the rule.

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// What lm15 reads from a Responses reasoning item with no summary (MAP-7
// rules 9 and 11): hidden thinking for the OpenAI dialect.
func hiddenReasoning() ThinkingPart {
	return ThinkingPart{Continuation: []ContinuationState{{Provider: "openai", Kind: "reasoning_item",
		Data: JSONObject{{"id", "rs_1"}, {"encrypted_content", "gAAAA"}}}}}
}

func deleteCall() ToolCallPart { return ToolCall("call_1", "delete_program", JSONObject{}) }

func deleteResult(content ...Part) Message {
	return ToolMessageParts(ToolResultPart{ID: "call_1", Name: "delete_program", Content: content})
}

// The history of issue #3: two steps answered by an OpenAI Responses model
// with hidden reasoning, then a new user message.
func reportRequest(model string) *Request {
	return &Request{Model: model, Messages: []Message{
		UserMessage("Delete the test program now."),
		AssistantMessage(hiddenReasoning(), deleteCall()),
		deleteResult(Text("ok")),
		AssistantMessage(hiddenReasoning(), Text("Deleted.")),
		UserMessage("hello"),
	}}
}

type requestBuilder interface {
	buildRequest(req *Request, stream bool, scope *adaptScope) (*TransportRequest, error)
}

func builtBody(t *testing.T, lm requestBuilder, provider string, req *Request, policy string) (map[string]any, *adaptScope) {
	t.Helper()
	scope := newAdaptScope(policy, provider, false)
	treq, err := lm.buildRequest(req, false, scope)
	if err != nil {
		t.Fatalf("%s (%s): %v", provider, policy, err)
	}
	var body map[string]any
	if err := json.Unmarshal(treq.Body, &body); err != nil {
		t.Fatal(err)
	}
	return body, scope
}

func emptyPartsAdapters(t *testing.T) map[string]requestBuilder {
	t.Helper()
	anthropic, err := NewAnthropicLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	gemini, err := NewGeminiLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	responses, err := NewOpenAILM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	chat, err := NewOpenAIChatLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	return map[string]requestBuilder{"anthropic": anthropic, "gemini": gemini, "openai": responses, "openai_chat": chat}
}

var emptyPartsModels = map[string]string{"anthropic": "claude-opus-5-5", "gemini": "gemini-2.5-flash", "openai": "gpt-5.4", "openai_chat": "gpt-5.4"}

func TestTheReportBuildsABodyClaudeAccepts(t *testing.T) {
	body, _ := builtBody(t, emptyPartsAdapters(t)["anthropic"], "anthropic", reportRequest("claude-opus-5-5"), "note")
	messages := body["messages"].([]any)
	var roles []string
	for _, m := range messages {
		roles = append(roles, m.(map[string]any)["role"].(string))
	}
	if !reflect.DeepEqual(roles, []string{"user", "assistant", "user", "assistant", "user"}) {
		t.Fatalf("roles: %v", roles)
	}
	want := []any{map[string]any{"type": "tool_use", "id": "call_1", "name": "delete_program", "input": map[string]any{}}}
	if got := messages[1].(map[string]any)["content"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("the call's message: %v", got)
	}
	raw, _ := json.Marshal(messages)
	if strings.Contains(string(raw), `"text":""`) {
		t.Fatalf("an empty text block reached the wire: %s", raw)
	}
}

func TestNothingIsRecordedAndTheHistoryIsNotTouched(t *testing.T) {
	// MAP-24 rule 5: translation, not an adaptation; "refuse" would fail
	// the build on any recorded deviation.
	for name, lm := range emptyPartsAdapters(t) {
		req := reportRequest(emptyPartsModels[name])
		before, _ := json.Marshal(RequestToDict(req))
		for _, policy := range []string{"note", "silent", "refuse"} {
			_, scope := builtBody(t, lm, name, req, policy)
			for _, a := range scope.records {
				if strings.HasPrefix(a.Field, "messages") {
					t.Fatalf("%s (%s) recorded %+v", name, policy, a)
				}
			}
		}
		after, _ := json.Marshal(RequestToDict(req))
		if string(before) != string(after) {
			t.Fatalf("%s changed the caller's request", name)
		}
	}
}

func TestTheStateGoesBackNativelyOnItsOwnDialect(t *testing.T) {
	adapters := emptyPartsAdapters(t)
	req := reportRequest("gpt-5.4")
	builtBody(t, adapters["anthropic"], "anthropic", reportRequest("claude-opus-5-5"), "note")
	body, _ := builtBody(t, adapters["openai"], "openai", req, "note")
	var reasoning []any
	for _, item := range body["input"].([]any) {
		if item.(map[string]any)["type"] == "reasoning" {
			reasoning = append(reasoning, item)
		}
	}
	one := map[string]any{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA", "summary": []any{}}
	if !reflect.DeepEqual(reasoning, []any{one, one}) {
		t.Fatalf("reasoning items: %v", reasoning)
	}
}

func TestAnEmptyThinkingPartWithClaudesSignatureIsStillAThinkingBlock(t *testing.T) {
	signed := func(signature string) ThinkingPart {
		return ThinkingPart{Continuation: []ContinuationState{{Provider: "anthropic", Kind: "thinking_signature",
			Data: JSONObject{{"signature", signature}}}}}
	}
	lm := emptyPartsAdapters(t)["anthropic"]
	body, _ := builtBody(t, lm, "anthropic", &Request{Model: "claude-opus-5-5", Messages: []Message{
		UserMessage("hi"), AssistantMessage(signed("sig"), deleteCall()), deleteResult(Text("ok"))}}, "note")
	first := body["messages"].([]any)[1].(map[string]any)["content"].([]any)[0]
	if !reflect.DeepEqual(first, map[string]any{"type": "thinking", "thinking": "", "signature": "sig"}) {
		t.Fatalf("Claude's omitted thinking: %v", first)
	}
	// A server that signs nothing sends signature "": that is no state.
	body, _ = builtBody(t, lm, "anthropic", &Request{Model: "claude-opus-5-5", Messages: []Message{
		UserMessage("hi"), AssistantMessage(signed(""), Text("Hello.")), UserMessage("again")}}, "note")
	content := body["messages"].([]any)[1].(map[string]any)["content"]
	if !reflect.DeepEqual(content, []any{map[string]any{"type": "text", "text": "Hello."}}) {
		t.Fatalf("an unsigned empty thinking part: %v", content)
	}
}

func TestGeminisCachedPrefixAndLiveTurnsLeaveTheMessageOut(t *testing.T) {
	lm, err := NewGeminiLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	messages := []Message{UserMessage("Here is the manual."), AssistantMessage(hiddenReasoning()), UserMessage("And the appendix.")}
	system := SystemParts(Text(""))
	treq, err := lm.cacheCreateRequest(&Request{Model: "gemini-2.5-flash", System: system, Messages: messages}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	var created map[string]any
	if err := json.Unmarshal(treq.Body, &created); err != nil {
		t.Fatal(err)
	}
	if got := len(created["contents"].([]any)); got != 2 {
		t.Fatalf("cached contents: %d, want 2", got)
	}
	if _, has := created["systemInstruction"]; has {
		t.Fatal("an empty system prompt reached the cached prefix")
	}
	payloads, err := lm.liveClientContentFromRequest(&Request{Model: "gemini-2.5-flash", Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(payloads[0])
	var frame map[string]any
	_ = json.Unmarshal(raw, &frame)
	if got := len(frame["clientContent"].(map[string]any)["turns"].([]any)); got != 2 {
		t.Fatalf("live turns: %d, want 2", got)
	}
}

func TestACacheMarkAskedOnAMessageLeftOutGoesToTheOneWrittenBeforeIt(t *testing.T) {
	// Not pinned by the contract (stated in the change entry): the
	// reference's choice on the Messages wire.
	one := 1
	body, _ := builtBody(t, emptyPartsAdapters(t)["anthropic"], "anthropic", &Request{Model: "claude-opus-5-5",
		Config:   Config{Cache: &CacheConfig{PrefixUntilIndex: &one}},
		Messages: []Message{UserMessage("Here is the manual."), AssistantMessage(hiddenReasoning()), UserMessage("And the appendix.")}}, "note")
	marked := func(i int) bool {
		content := body["messages"].([]any)[i].(map[string]any)["content"].([]any)
		_, has := content[len(content)-1].(map[string]any)["cache_control"]
		return has
	}
	if !marked(0) || marked(1) {
		t.Fatalf("cache marks: first %v, second %v", marked(0), marked(1))
	}
}

func TestAToolResultKeepsItsEmptyOutputAndDropsAnEmptyBlockBesideAnImage(t *testing.T) {
	image := Image(WithURL("https://example.com/receipt.png"))
	for _, tc := range []struct {
		content []Part
		want    any
	}{
		{[]Part{Text("")}, ""},
		{[]Part{Text(""), image}, []any{map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": "https://example.com/receipt.png"}}}},
	} {
		body, _ := builtBody(t, emptyPartsAdapters(t)["anthropic"], "anthropic", &Request{Model: "claude-opus-5-5", Messages: []Message{
			UserMessage("hi"), AssistantMessage(deleteCall()), deleteResult(tc.content...)}}, "note")
		block := body["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
		if block["type"] != "tool_result" || !reflect.DeepEqual(block["content"], tc.want) {
			t.Fatalf("tool result: %v", block)
		}
	}
}

func TestTextJoinedHasNoLineForAnEmptyPartAndWhitespaceIsContent(t *testing.T) {
	text, err := partsToText([]Part{Text(""), Text("a"), Text(""), Text("b")}, "", "")
	if err != nil || text != "a\nb" {
		t.Fatalf("joined: %q %v", text, err)
	}
	body, _ := builtBody(t, emptyPartsAdapters(t)["anthropic"], "anthropic", &Request{Model: "claude-opus-5-5", Messages: []Message{
		UserMessage("hi"), AssistantMessage(Text("\n")), UserMessage("again")}}, "note")
	content := body["messages"].([]any)[1].(map[string]any)["content"]
	if !reflect.DeepEqual(content, []any{map[string]any{"type": "text", "text": "\n"}}) {
		t.Fatalf("whitespace is content: %v", content)
	}
}

func TestTheBlockTestReadsTheThreeTypedSpellingsOnly(t *testing.T) {
	for _, kind := range []string{"text", "input_text", "output_text"} {
		if !isEmptyTextBlock(JSONObject{{"type", kind}, {"text", ""}}) {
			t.Fatalf("%s with no characters is an empty text block", kind)
		}
	}
	for _, block := range []JSONObject{
		{{"type", "text"}, {"text", " "}},
		{{"type", "thinking"}, {"thinking", ""}, {"signature", "s"}},
		{{"type", "tool_use"}, {"id", "c"}, {"name", "f"}, {"input", JSONObject{{"text", ""}}}},
	} {
		if isEmptyTextBlock(block) {
			t.Fatalf("not an empty text block: %v", block)
		}
	}
}
