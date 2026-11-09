package lm15

// Tinker (Thinking Machines): the two doors, the reasoning switch per base
// family, a checkpoint that follows its declared base model, URI model
// names, FastAPI error bodies (MAP-21), the stop sequence returned in the
// text (MAP-22) and the catalog's fine-tune origin (MAP-23). lm15-contract
// changes/2026-11-09-tinker-live.md; the harness pins the same rules through
// the corpus, these tests keep them visible here.

import (
	"encoding/json"
	"strings"
	"testing"
)

const tinkerCK = "tinker://e68146d2-1a2e-56e8-b42f-1f5957705501:train:0/sampler_weights/lm15-untrained"

func tinkerChat(t *testing.T, base map[string]string) *OpenAIChatLM {
	t.Helper()
	def, _ := LookupProvider("tinker")
	opts := []Option{WithAPIKey("k"), WithAccess(def.Access), WithCompatPreset("tinker")}
	if base != nil {
		opts = append(opts, WithBaseModels(base))
	}
	lm, err := NewOpenAIChatLM(opts...)
	if err != nil {
		t.Fatal(err)
	}
	return lm
}

func tinkerWire(t *testing.T, lm *OpenAIChatLM, model string, cfg Config) (map[string]any, []Adaptation) {
	t.Helper()
	treq, records, err := lm.build(&Request{Model: model, Messages: []Message{UserMessage("hi")}, Config: cfg}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(treq.Body, &body); err != nil {
		t.Fatal(err)
	}
	return body, records
}

func TestTinkerDoorsAndRouting(t *testing.T) {
	c, _ := LookupProvider("tinker")
	m, _ := LookupProvider("tinker-anthropic")
	if c.Access.EnvKeys[0] != "TINKER_API_KEY" || m.Access.EnvKeys[0] != "TINKER_API_KEY" {
		t.Fatal("one key for both doors")
	}
	cfg := RouterConfig{Env: map[string]string{}}
	for _, tc := range [][3]string{
		{tinkerCK, "tinker", tinkerCK},
		{"tinker:" + tinkerCK, "tinker", tinkerCK},
		{"tinker-anthropic:" + tinkerCK, "tinker-anthropic", tinkerCK},
		{"thinkingmachines:" + tinkerCK, "tinker", tinkerCK},
		{"thinkingmachines:Qwen/Qwen3-8B", "tinker", "Qwen/Qwen3-8B"},
	} {
		r, err := Resolve(tc[0], cfg)
		if err != nil || r.Provider != tc[1] || r.Model != tc[2] {
			t.Fatalf("%s: %+v %v", tc[0], r, err)
		}
	}
	if _, err := Resolve("openai://x", cfg); err == nil {
		t.Fatal("a URI with no rule routes nowhere")
	}
	if got, _ := OpenAIChatModelString("thinkingmachines/Qwen/Qwen3-8B"); got != "tinker:Qwen/Qwen3-8B" {
		t.Fatal(got)
	}
}

func TestTinkerReasoningPerFamily(t *testing.T) {
	lm := tinkerChat(t, nil)
	off := Config{Reasoning: &Reasoning{Effort: "off"}}
	for model, want := range map[string]any{
		"thinkingmachines/Inkling-Small": "none", "Qwen/Qwen3.5-4B": false, "moonshotai/Kimi-K2.6": false,
		"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": false, "Qwen/Qwen3.8-27B": false,
		"deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": false, tinkerCK: false,
	} {
		body, records := tinkerWire(t, lm, model, off)
		if body["reasoning_effort"] != want || len(records) != 0 {
			t.Fatalf("%s: %v %v", model, body["reasoning_effort"], records)
		}
	}
	body, records := tinkerWire(t, lm, "Qwen/Qwen3.5-4B", Config{Reasoning: &Reasoning{Effort: "high"}})
	if body["reasoning_effort"] != true || len(records) != 1 || records[0].Applied != "on" || records[0].Asked != "high" {
		t.Fatalf("switch: %v %+v", body["reasoning_effort"], records)
	}
	for _, model := range []string{"thinkingmachines/Inkling", "openai/gpt-oss-20b", tinkerCK} {
		if body, _ := tinkerWire(t, lm, model, Config{Reasoning: &Reasoning{Effort: "low"}}); body["reasoning_effort"] != "low" {
			t.Fatalf("%s: %v", model, body["reasoning_effort"])
		}
	}
}

func TestTinkerCheckpointFollowsItsBase(t *testing.T) {
	off := Config{Reasoning: &Reasoning{Effort: "off"}}
	if body, _ := tinkerWire(t, tinkerChat(t, map[string]string{tinkerCK: "thinkingmachines/Inkling-Small"}), tinkerCK, off); body["reasoning_effort"] != "none" {
		t.Fatal(body["reasoning_effort"])
	}
	if body, _ := tinkerWire(t, tinkerChat(t, map[string]string{tinkerCK: "Qwen/Qwen3.5-4B"}), tinkerCK, Config{Reasoning: &Reasoning{Effort: "medium"}}); body["reasoning_effort"] != true {
		t.Fatal(body["reasoning_effort"])
	}
	router, err := NewRouterWithConfig(RouterConfig{Env: map[string]string{"TINKER_API_KEY": "k"}, BaseModels: map[string]string{tinkerCK: "thinkingmachines/Inkling"}})
	if err != nil {
		t.Fatal(err)
	}
	lm, err := router.LM(tinkerCK)
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := tinkerWire(t, lm.(*OpenAIChatLM), tinkerCK, off); body["reasoning_effort"] != "none" {
		t.Fatal(body["reasoning_effort"])
	}
	if _, err := NewRouterWithConfig(RouterConfig{BaseModels: map[string]string{tinkerCK: ""}}); err == nil {
		t.Fatal("an empty base model is refused")
	}
}

func TestTinkerResponseFormatDropped(t *testing.T) {
	for _, format := range []JSONObject{{KV("type", "json_object")}, {KV("type", "json_schema"), KV("name", "x"), KV("schema", JSONObject{KV("type", "object")})}} {
		body, records := tinkerWire(t, tinkerChat(t, nil), "thinkingmachines/Inkling", Config{ResponseFormat: format})
		if _, ok := body["response_format"]; ok || len(records) != 1 || records[0].Field != "config.response_format" {
			t.Fatalf("%v %+v", body, records)
		}
	}
}

func TestTinkerStopSequenceCut(t *testing.T) {
	lm := tinkerChat(t, nil)
	req := &Request{Model: "thinkingmachines/Inkling-Small", Messages: []Message{UserMessage("count")}, Config: Config{Stop: []string{" 7"}}}
	reply := `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"1 2 3 4 5 6 7"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}`
	resp, err := lm.ParseResponse(req, &HTTPResponse{Status: 200, Body: []byte(reply)})
	if err != nil || resp.TextOr("") != "1 2 3 4 5 6" || resp.Usage.InputTokens == nil {
		t.Fatalf("%v %v", resp, err)
	}
	sse := `data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"1 2 3 4 5 6 7"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}` + "\n\ndata: [DONE]\n\n"
	events, err := ParseStreamBody(lm, req, []byte(sse))
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, e := range events {
		if td, ok := textDeltaOf(e); ok {
			text.WriteString(td.Text)
		}
	}
	if text.String() != "1 2 3 4 5 6" {
		t.Fatal(text.String())
	}
}

func TestTinkerDetailErrorsAndForms(t *testing.T) {
	lm := tinkerChat(t, nil)
	for _, tc := range []struct {
		status int
		body   string
		kind   ErrorKind
	}{
		{401, `{"detail":"Unable to validate api-key"}`, KindAuth},
		{404, `{"detail":"Weights not found for model_path ` + tinkerCK + `"}`, KindUnsupportedModel},
		{400, `{"detail":"Sampling is not supported for x. (org)"}`, KindUnsupportedModel},
		{400, `{"detail":"model_path 'tinker://x' is not a valid tinker path. Expected"}`, KindUnsupportedModel},
		{422, `{"detail":"model returned no tool call when tool_choice required one"}`, KindInvalidRequest},
	} {
		e := lm.NormalizeError(tc.status, tc.body)
		if e.Kind != tc.kind {
			t.Fatalf("%s: %v", tc.body, e.Kind)
		}
	}
	got := DetailMessage(JSONObject{KV("detail", "Validation Error"), KV("errors", []any{JSONObject{KV("loc", "body -> messages -> 0 -> role"), KV("msg", "Input should be 'system'")}})})
	if got != "Validation Error: body -> messages -> 0 -> role: Input should be 'system'" {
		t.Fatal(got)
	}
}

func TestTinkerCatalogNamesTheBase(t *testing.T) {
	models, err := tinkerChat(t, nil).modelsFromBody(`{"object":"list","data":[{"id":"` + tinkerCK + `","object":"model","created":1,"owned_by":"x","base_model":"Qwen/Qwen3.5-4B"}]}`)
	if err != nil || len(models) != 1 || models[0].Origin.Type != "fine-tune" || models[0].Origin.BaseModel != "Qwen/Qwen3.5-4B" {
		t.Fatalf("%+v %v", models, err)
	}
}
