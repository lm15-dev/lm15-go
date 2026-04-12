package lm15

import (
	"fmt"
	"strings"
)

type providerPattern struct {
	pattern  string
	provider string
}

var defaultPatterns = []providerPattern{
	{"claude", "anthropic"},
	{"gemini", "gemini"},
	{"gpt", "openai"},
	{"o1", "openai"},
	{"o3", "openai"},
	{"o4", "openai"},
	{"chatgpt", "openai"},
	{"dall-e", "openai"},
	{"tts", "openai"},
	{"whisper", "openai"},
}

// ResolveProvider infers the provider name from a model name.
func ResolveProvider(model string) (string, error) {
	lower := strings.ToLower(model)
	for _, p := range defaultPatterns {
		if strings.HasPrefix(lower, p.pattern) {
			return p.provider, nil
		}
	}
	return "", &UnsupportedModelError{ProviderError{ULMError{
		fmt.Sprintf("unable to resolve provider for model '%s'\n\n"+
			"  To fix, do one of:\n"+
			"    1. Use Provider option to specify explicitly\n"+
			"    2. Verify the model name (common prefixes: gpt-, claude-, gemini-)\n", model),
	}}}
}
