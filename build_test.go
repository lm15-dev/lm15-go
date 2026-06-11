package lm15

import "testing"

func i64(v int64) *int64 { return &v }

func textReq(model string, cfg *Config) Request {
	return Request{
		Model:    model,
		Messages: []Message{{Role: "user", Parts: []Part{TextPart{Text: "hi"}}}},
		Config:   cfg,
	}
}

func TestAnthropicMaxTokensArithmetic(t *testing.T) {
	// max_tokens = thinking budget + visible budget (spec/invariants.md).
	cfg := &Config{MaxTokens: i64(400), Reasoning: &Reasoning{Effort: "medium", ThinkingBudget: i64(1024)}}
	tr, err := BuildRequest("anthropic", textReq("claude-sonnet-4-5", cfg), false, "k", "")
	if err != nil {
		t.Fatal(err)
	}
	if got := tr.Body["max_tokens"]; string(got.(interface{ String() string }).String()) != "1424" {
		t.Fatalf("max_tokens = %v, want 1424", got)
	}
	// total_budget <= thinking_budget is rejected.
	cfg = &Config{Reasoning: &Reasoning{Effort: "medium", ThinkingBudget: i64(1024), TotalBudget: i64(512)}}
	if _, err := BuildRequest("anthropic", textReq("claude-sonnet-4-5", cfg), false, "k", ""); err == nil {
		t.Fatal("expected total_budget <= thinking_budget error")
	}
}

func TestChatCompatPresets(t *testing.T) {
	for name, wantField := range map[string]string{
		"openai": "max_completion_tokens", "ollama": "max_tokens",
		"groq": "max_tokens", "openrouter": "max_tokens",
		"vllm": "max_tokens", "sglang": "max_tokens",
	} {
		c, err := ChatCompatPreset(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if c.MaxTokensField != wantField {
			t.Fatalf("%s: MaxTokensField = %s, want %s", name, c.MaxTokensField, wantField)
		}
		if _, ok := ChatPresetBaseURLs[name]; !ok {
			t.Fatalf("%s: missing preset base URL", name)
		}
	}
	if _, err := ChatCompatPreset("nope"); err == nil {
		t.Fatal("expected unknown preset error")
	}
}

func TestChatMaxTokensPolicy(t *testing.T) {
	cfg := &Config{MaxTokens: i64(64)}
	tr, err := buildOpenAIChat(textReq("m", cfg), false, "k", "https://x/v1", defaultChatCompat())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tr.Body["max_completion_tokens"]; !ok {
		t.Fatal("default policy must emit max_completion_tokens")
	}
	groq, _ := ChatCompatPreset("groq")
	tr, err = buildOpenAIChat(textReq("m", cfg), false, "k", "https://x/v1", groq)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tr.Body["max_tokens"]; !ok {
		t.Fatal("groq preset must emit max_tokens")
	}
}

func TestAuthHeaders(t *testing.T) {
	for provider, header := range map[string]string{
		"openai": "authorization", "openai_chat": "authorization",
		"anthropic": "x-api-key", "gemini": "x-goog-api-key",
	} {
		tr, err := BuildRequest(provider, textReq("m", nil), false, "test-key-123", "")
		if err != nil {
			t.Fatal(err)
		}
		want := "test-key-123"
		if header == "authorization" {
			want = "Bearer test-key-123"
		}
		if tr.Headers[header] != want {
			t.Fatalf("%s: header %s = %q, want %q", provider, header, tr.Headers[header], want)
		}
	}
}

func TestGeminiStreamParams(t *testing.T) {
	tr, err := BuildRequest("gemini", textReq("gemini-2.5-flash", nil), true, "k", "")
	if err != nil {
		t.Fatal(err)
	}
	if tr.Params["alt"] != "sse" {
		t.Fatalf("params = %v, want alt=sse", tr.Params)
	}
	if tr.URL != "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent" {
		t.Fatalf("url = %s", tr.URL)
	}
}
