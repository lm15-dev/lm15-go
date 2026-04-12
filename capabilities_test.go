package lm15

import "testing"

func TestResolveProvider(t *testing.T) {
	tests := []struct {
		model string
		want  string
	}{
		{"claude-sonnet-4-5", "anthropic"},
		{"gpt-4.1-mini", "openai"},
		{"gemini-2.5-flash", "gemini"},
		{"o1-preview", "openai"},
		{"o3-mini", "openai"},
		{"o4-mini", "openai"},
	}
	for _, tt := range tests {
		got, err := ResolveProvider(tt.model)
		if err != nil {
			t.Errorf("ResolveProvider(%q) error: %v", tt.model, err)
			continue
		}
		if got != tt.want {
			t.Errorf("ResolveProvider(%q) = %q, want %q", tt.model, got, tt.want)
		}
	}
}

func TestResolveProviderUnknown(t *testing.T) {
	_, err := ResolveProvider("llama-3")
	if err == nil {
		t.Error("expected error for unknown model")
	}
	if _, ok := err.(*UnsupportedModelError); !ok {
		t.Errorf("expected UnsupportedModelError, got %T", err)
	}
}
