package lm15

import (
	"errors"
	"strings"
	"testing"
)

// An output cap or store=true on the Codex backend is refused, never stripped (MAP-13 rule 4).
func TestCodexRefusesCapAndStore(t *testing.T) {
	lm, err := NewOpenAILM(WithAPIKey("tok"), WithAccess(OpenAICodex), WithAccountID("acct"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(config Config) error {
		_, err := lm.BuildRequest(&Request{Model: "gpt-5.5", Messages: []Message{UserMessage("hi")}, Config: config}, true)
		return err
	}
	if err := build(Config{}); err != nil {
		t.Fatalf("a request without a cap builds: %v", err)
	}
	yes := true
	for feature, config := range map[string]Config{"config.max_tokens": {MaxTokens: I(5)}, "config.store": {Store: &yes}} {
		var e *Error
		if err := build(config); !errors.As(err, &e) || e.Kind != KindUnsupportedFeature || e.Feature != feature || !strings.Contains(e.Error(), "openai-codex") {
			t.Fatalf("%s: want an UnsupportedFeatureError naming openai-codex, got %v", feature, err)
		}
	}
}
