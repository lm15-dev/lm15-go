package lm15

import "testing"

func TestUniversalLMRejectsUnknownProvider(t *testing.T) {
	lm := NewUniversalLM()
	_, err := lm.Complete(LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}}, "nope")
	if err == nil {
		t.Error("expected error for unknown provider")
	}
	if _, ok := err.(*ProviderError); !ok {
		t.Errorf("expected ProviderError, got %T", err)
	}
}

func TestUniversalLMRegister(t *testing.T) {
	lm := NewUniversalLM()
	lm.Register(&echoTestAdapter{})
	// Should be able to resolve
	a, err := lm.resolveAdapter("test", "echo")
	if err != nil {
		t.Fatal(err)
	}
	if a.ProviderName() != "echo" {
		t.Errorf("expected echo, got %s", a.ProviderName())
	}
}
