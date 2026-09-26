package lm15

// DeepInfra, Together AI, Fireworks AI and Parasail: the registry entries and
// the compat rules each carries (lm15-contract changes/2026-09-26-inference-
// hosts-live.md, ratified 2026-09-26; every rule has a receipt under
// receipts/2026-09-26-<host>/). The harness pins the same rules through the
// corpus; these tests keep them visible in this repository.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

var inferenceHosts = map[string][3]string{
	"deepinfra": {"https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "deepinfra"},
	"together":  {"https://api.together.ai/v1", "TOGETHER_API_KEY", "together_ai"},
	"fireworks": {"https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "fireworks_ai"},
	"parasail":  {"https://api.parasail.io/v1", "PARASAIL_API_KEY", "parasail"},
}

var hostWeather = FunctionTool{Name: "get_weather", Description: "Get weather.", Parameters: JSONObject{
	KV("type", "object"), KV("properties", JSONObject{KV("city", JSONObject{KV("type", "string")})}), KV("required", []any{"city"}),
}}

func hostBuild(t *testing.T, provider string, req *Request) (map[string]any, []Adaptation, error) {
	t.Helper()
	lm, err := NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset(provider))
	if err != nil {
		t.Fatal(err)
	}
	treq, adaptations, err := lm.build(req, false, false)
	if err != nil {
		return nil, nil, err
	}
	var body map[string]any
	if err := json.Unmarshal(treq.Body, &body); err != nil {
		t.Fatal(err)
	}
	return body, adaptations, nil
}

func hostAsk(model string, cfg Config) *Request {
	return &Request{Model: model, Messages: []Message{UserMessage("hi")}, Config: cfg}
}

func TestInferenceHostRegistryAndRouting(t *testing.T) {
	for provider, want := range inferenceHosts {
		def, ok := LookupProvider(provider)
		if !ok || def.Dialect != DialectOpenAIChat || def.Access.BaseURL != want[0] || len(def.Access.EnvKeys) != 1 || def.Access.EnvKeys[0] != want[1] || !strings.HasPrefix(def.ConsoleURL, "https://") {
			t.Fatalf("%s: registry entry %+v", provider, def)
		}
		router, err := NewRouterWithConfig(RouterConfig{Env: map[string]string{want[1]: "k"}})
		if err != nil {
			t.Fatal(err)
		}
		res, err := router.Resolve(provider + ":vendor/some-model")
		if err != nil || res.Provider != provider || res.Model != "vendor/some-model" || res.EnvKey != want[1] {
			t.Fatalf("%s: resolve %+v %v", provider, res, err)
		}
		if got, err := OpenAIChatModelString(want[2] + "/vendor/some-model"); err != nil || got != provider+":vendor/some-model" {
			t.Fatalf("%s: litellm %q %v", provider, got, err)
		}
	}
}

func TestInferenceHostDialAndReplay(t *testing.T) {
	cap := 50
	for provider := range inferenceHosts {
		body, _, err := hostBuild(t, provider, hostAsk("vendor/reasoner", Config{MaxTokens: &cap, Reasoning: &Reasoning{Effort: "low"}}))
		if err != nil || body["reasoning_effort"] != "low" || body["max_completion_tokens"] != float64(50) || body["reasoning"] != nil {
			t.Fatalf("%s: %v %v", provider, body, err)
		}
		turn := AssistantMessage(Thinking("Need the tool."), ToolCall("call_1", "get_weather", JSONObject{KV("city", "Paris")}))
		req := &Request{Model: "vendor/m", Tools: []Tool{hostWeather}, Messages: []Message{UserMessage("Weather?"), turn, ToolMessage("call_1", "Sunny")}}
		body, _, err = hostBuild(t, provider, req)
		if err != nil {
			t.Fatal(err)
		}
		assistant := body["messages"].([]any)[1].(map[string]any)
		if assistant["reasoning_content"] != "Need the tool." || assistant["content"] != nil {
			t.Fatalf("%s: replay %v", provider, assistant)
		}
	}
}

func TestDeepInfraForcedToolChoiceGoesOnlyToReceiptedModels(t *testing.T) {
	for model, sent := range map[string]bool{
		"meta-llama/Llama-3.3-70B-Instruct-Turbo": false,
		"openai/gpt-oss-120b":                     false,
		"zai-org/GLM-4.7":                         false,
		"deepseek-ai/DeepSeek-V4-Pro":             false, // untested: refused, never silently ignored
		"deepseek-ai/DeepSeek-V4.1-Flash":         true,
		"zai-org/GLM-5.3-Flash":                   true,
		"anthropic/claude-haiku-4-5":              true,
		"deepseek-ai/DeepSeek-V4-Flash-0731":      true, // a suffixed variant inherits its entry
	} {
		req := &Request{Model: model, Messages: []Message{UserMessage("hi")}, Tools: []Tool{hostWeather}, Config: Config{ToolChoice: &ToolChoice{Mode: "required"}}}
		body, _, err := hostBuild(t, "deepinfra", req)
		if sent && (err != nil || body["tool_choice"] != "required") {
			t.Fatalf("%s: want sent, got %v %v", model, body, err)
		}
		if !sent && (err == nil || AsError(err).Kind != KindUnsupportedFeature) {
			t.Fatalf("%s: want refused, got %v %v", model, body, err)
		}
	}
}

func TestTogetherGptOssRules(t *testing.T) {
	forced := func(model string) *Request {
		return &Request{Model: model, Messages: []Message{UserMessage("hi")}, Tools: []Tool{hostWeather}, Config: Config{ToolChoice: &ToolChoice{Mode: "required"}}}
	}
	if _, _, err := hostBuild(t, "together", forced("openai/gpt-oss-120b")); err == nil || AsError(err).Kind != KindUnsupportedFeature {
		t.Fatalf("gpt-oss forced tool choice: %v", err)
	}
	if body, _, err := hostBuild(t, "together", forced("meta-llama/Llama-3.3-70B-Instruct-Turbo")); err != nil || body["tool_choice"] != "required" {
		t.Fatalf("llama forced tool choice: %v %v", body, err)
	}
	for asked, applied := range map[string]string{"max": "high", "xhigh": "high", "minimal": "low"} {
		body, adaptations, err := hostBuild(t, "together", hostAsk("openai/gpt-oss-120b", Config{Reasoning: &Reasoning{Effort: asked}}))
		if err != nil || body["reasoning_effort"] != applied || len(adaptations) != 1 || adaptations[0].Action != "clamped" || adaptations[0].Applied != applied {
			t.Fatalf("clamp %s: %v %+v %v", asked, body, adaptations, err)
		}
	}
	if body, _, _ := hostBuild(t, "together", hostAsk("deepseek-ai/DeepSeek-V4.1-Flash", Config{Reasoning: &Reasoning{Effort: "max"}})); body["reasoning_effort"] != "max" {
		t.Fatalf("deepseek effort sent verbatim: %v", body)
	}
}

func TestReasoningOffLowestWhereTheServerIgnoresNone(t *testing.T) {
	off := Config{Reasoning: &Reasoning{Effort: "off"}}
	for _, c := range [][2]string{{"together", "openai/gpt-oss-120b"}, {"together", "zai-org/GLM-5.3-Flash"}, {"deepinfra", "openai/gpt-oss-120b"}} {
		body, adaptations, err := hostBuild(t, c[0], hostAsk(c[1], off))
		if err != nil || body["reasoning_effort"] != "low" || len(adaptations) != 1 ||
			adaptations[0].Field != "config.reasoning.effort" || adaptations[0].Action != "substituted" || adaptations[0].Asked != "off" || adaptations[0].Applied != "low" {
			t.Fatalf("%v: %v %+v %v", c, body, adaptations, err)
		}
	}
	for _, c := range [][2]string{{"together", "deepseek-ai/DeepSeek-V4.1-Flash"}, {"fireworks", "accounts/fireworks/models/gpt-oss-120b"}, {"parasail", "openai/gpt-oss-20b"}} {
		body, adaptations, err := hostBuild(t, c[0], hostAsk(c[1], off))
		if err != nil || body["reasoning_effort"] != "none" || len(adaptations) != 0 {
			t.Fatalf("%v: %v %+v %v", c, body, adaptations, err)
		}
	}
}

func TestChatModelsBareArrayAndMalformedShape(t *testing.T) {
	lm, _ := NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset("together"))
	models, err := lm.modelsFromBody(`[{"id": "a"}, {"id": "b"}]`)
	if err != nil || len(models) != 2 || models[1].ID != "b" {
		t.Fatalf("bare array: %v %v", models, err)
	}
	if models, err := lm.modelsFromBody(`{"object": "list", "data": [{"id": "c"}]}`); err != nil || len(models) != 1 {
		t.Fatalf("data envelope: %v %v", models, err)
	}
	fake := &recordingTransport{replies: []string{`{"models": [{"id": "x"}]}`}}
	lm, _ = NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset("together"), WithTransport(fake))
	if _, err := lm.ListModels(context.Background()); err == nil || AsError(err).Kind != KindProvider || !strings.Contains(err.Error(), "malformed provider reply") {
		t.Fatalf("unknown shape must be a ProviderError, got %v", err)
	}
}

func TestChatCachedTokensNestedThenFlat(t *testing.T) {
	flat := usageFromChat(JSONObject{KV("prompt_tokens", 9), KV("cached_tokens", 0)})
	nested := usageFromChat(JSONObject{KV("prompt_tokens_details", JSONObject{KV("cached_tokens", 3)}), KV("cached_tokens", 9)})
	none := usageFromChat(JSONObject{KV("prompt_tokens", 9)})
	if flat.CacheReadTokens == nil || *flat.CacheReadTokens != 0 || nested.CacheReadTokens == nil || *nested.CacheReadTokens != 3 || none.CacheReadTokens != nil {
		t.Fatalf("cached tokens: %v %v %v", flat.CacheReadTokens, nested.CacheReadTokens, none.CacheReadTokens)
	}
}
