package lm15

import (
	"sort"
	"strings"
)

// The one table of named providers (reference lm15/registry.py). A routable
// provider string names a dialect, an access policy and — for the chat
// dialect — a compat preset. Nothing else lists providers.

// Dialect names the wire format an adapter speaks.
const (
	DialectOpenAIResponses = "openai-responses"
	DialectOpenAIChat      = "openai-chat"
	DialectAnthropic       = "anthropic"
	DialectGemini          = "gemini"
	DialectTypeSafe        = "typesafe"
)

// ProviderDefinition is everything lm15 knows about one named provider.
type ProviderDefinition struct {
	ID             string
	Dialect        string
	Access         AccessPolicy
	Compat         string
	PlaceholderKey string
	ConsoleURL     string
	Note           string
	// AdapterOwned: the dialect's own manifest is this policy (openai,
	// anthropic, gemini, xai, claude-code, openai-codex, openai-chat).
	AdapterOwned bool
	// Aliases are extra input spellings (hyphenated) of a declared provider
	// (RouterConfig.Providers); the registry's entries use none.
	Aliases []string
	// CompatValue is a declared provider's compat as a value (Compat holds
	// a preset name); one of the two.
	CompatValue *OpenAIChatCompat
	// Declared: from RouterConfig.Providers, not the receipted registry.
	Declared bool
}

// Spellings are every input spelling that names this provider: the id
// and the aliases.
func (d ProviderDefinition) Spellings() []string {
	return append([]string{d.ID}, d.Aliases...)
}

// DeclareChatProvider declares a provider the registry does not list — a
// gateway, a service lm15 has not receipted — speaking the OpenAI Chat
// Completions wire: access names it (AccessPolicy{Provider, EnvKeys,
// BaseURL}) and compat describes the server's spellings. It routes like a
// registry entry in every router built with a RouterConfig that lists it,
// and answers Resolution.Declared: no live receipt backs it, and lm15
// says so.
func DeclareChatProvider(access AccessPolicy, compat OpenAIChatCompat, aliases []string, placeholderKey, consoleURL, note string) (ProviderDefinition, error) {
	if err := access.Validate(); err != nil {
		return ProviderDefinition{}, err
	}
	if err := compat.Validate(); err != nil {
		return ProviderDefinition{}, err
	}
	if access.Provider == "" {
		return ProviderDefinition{}, valueErrorf("a declared provider needs a non-empty id (AccessPolicy.Provider)")
	}
	seen := map[string]bool{access.Provider: true}
	for _, a := range aliases {
		if a == "" || CanonicalProvider(a) != a {
			return ProviderDefinition{}, valueErrorf("%s: aliases are non-empty and hyphenated, got %q", access.Provider, a)
		}
		if seen[a] {
			return ProviderDefinition{}, valueErrorf("%s: aliases repeat a spelling", access.Provider)
		}
		seen[a] = true
	}
	c := compat
	return ProviderDefinition{ID: access.Provider, Dialect: DialectOpenAIChat, Access: access, CompatValue: &c, Aliases: append([]string(nil), aliases...),
		PlaceholderKey: placeholderKey, ConsoleURL: consoleURL, Note: note, Declared: true}, nil
}

// Bound reports whether the router binds Access onto the dialect at construction.
func (d ProviderDefinition) Bound() bool { return !d.AdapterOwned }

// Hosted reports whether the access policy names a cloud host.
func (d ProviderDefinition) Hosted() bool { return d.Access.Host != nil }

// EnvKeys are the declared environment keys.
func (d ProviderDefinition) EnvKeys() []string { return d.Access.EnvKeys }

// CredentialPolicy is the declared AUTH-1 policy.
func (d ProviderDefinition) CredentialPolicy() string { return d.Access.EffectiveCredentialPolicy() }

// CanonicalProvider maps the permanent underscore alias to the hyphenated form.
func CanonicalProvider(name string) string { return strings.ReplaceAll(name, "_", "-") }

func owned(id, dialect string, access AccessPolicy, console, note string) ProviderDefinition {
	return ProviderDefinition{ID: id, Dialect: dialect, Access: access, ConsoleURL: console, Note: note, AdapterOwned: true}
}

func chatBound(access AccessPolicy, compat, placeholder, console, note string) ProviderDefinition {
	if compat == "" {
		compat = access.Provider
	}
	return ProviderDefinition{ID: access.Provider, Dialect: DialectOpenAIChat, Access: access, Compat: compat, PlaceholderKey: placeholder, ConsoleURL: console, Note: note}
}

func responsesBound(access AccessPolicy, compat, console, note string) ProviderDefinition {
	return ProviderDefinition{ID: access.Provider, Dialect: DialectOpenAIResponses, Access: access, Compat: compat, ConsoleURL: console, Note: note}
}

func anthropicBound(access AccessPolicy, compat, console, note string) ProviderDefinition {
	return ProviderDefinition{ID: access.Provider, Dialect: DialectAnthropic, Access: access, Compat: compat, ConsoleURL: console, Note: note}
}

func hosted(access AccessPolicy, dialect, compat, console, note string) ProviderDefinition {
	return ProviderDefinition{ID: access.Provider, Dialect: dialect, Access: access, Compat: compat, ConsoleURL: console, Note: note}
}

// providerDefinitions is the declaration order (presentation order).
var providerDefinitions = []ProviderDefinition{
	owned("openai", DialectOpenAIResponses, OpenAIAPI, "https://platform.openai.com/api-keys", "OpenAI Responses API"),
	owned("openai-chat", DialectOpenAIChat, OpenAIChatAPI, "https://platform.openai.com/api-keys", "OpenAI Chat Completions dialect (the de-facto standard other servers speak)"),
	owned("anthropic", DialectAnthropic, AnthropicAPI, "https://console.anthropic.com", "Anthropic Messages API"),
	owned("gemini", DialectGemini, GeminiAPI, "https://aistudio.google.com/apikey", "Google Gemini API"),
	owned("xai", DialectOpenAIChat, Xai, "https://console.x.ai", "xAI Grok (Chat Completions dialect; XAI_API_KEY or subscription OAuth)"),
	owned("claude-code", DialectAnthropic, ClaudeCode, "", "Claude subscription through the local `claude` CLI login"),
	owned("openai-codex", DialectOpenAIResponses, OpenAICodex, "", "ChatGPT subscription through the local `codex` CLI login"),
	owned("typesafe", DialectTypeSafe, TypeSafeAPI, "https://console.typesafe.ai/keys", "TypeSafe System One (Jev): judgments over declared keys with probabilities; no text generation"),
	chatBound(Groq, "", "", "https://console.groq.com/keys", "Groq Cloud (Chat Completions dialect)"),
	chatBound(OpenRouter, "", "", "https://openrouter.ai/keys", "OpenRouter (Chat Completions dialect)"),
	chatBound(DeepSeek, "", "", "https://platform.deepseek.com/api_keys", "DeepSeek (Chat Completions dialect; thinking mode on by default)"),
	anthropicBound(DeepSeekAnthropic, "deepseek", "https://platform.deepseek.com/api_keys", "DeepSeek over the Anthropic Messages wire (same key as `deepseek`; no model listing)"),
	chatBound(Zai, "", "", "https://z.ai/manage-apikey/apikey-list", "Z.AI GLM (Chat Completions dialect; general endpoint, not the Coding Plan)"),
	chatBound(Moonshotai, "", "", "https://platform.kimi.ai/console/api-keys", "Moonshot AI Kimi (Chat Completions dialect; kimi-k3 takes reasoning effort low|high|max, kimi-k2.6 takes effort off; Moonshot's docs call the key MOONSHOT_API_KEY — read after MOONSHOTAI_API_KEY)"),
	responsesBound(MoonshotaiResponses, "moonshotai", "https://platform.kimi.ai/console/api-keys", "Moonshot AI Kimi over the Responses wire (same key as `moonshotai`; kimi-k3 only; stateless — reasoning replays as summary text; web_search built-in)"),
	anthropicBound(MoonshotaiAnthropic, "moonshotai", "https://platform.kimi.ai/console/api-keys", "Moonshot AI Kimi over the Anthropic Messages wire (same key as `moonshotai`, bearer token; kimi-k3 only)"),
	chatBound(DeepInfra, "", "", "https://deepinfra.com/dash/api_keys", "DeepInfra open-model inference (Chat Completions dialect; models are vendor/name ids)"),
	chatBound(Together, "", "", "https://api.together.ai/settings/projects/~current/api-keys", "Together AI open-model inference (Chat Completions dialect; gpt-oss refuses a forced tool choice client-side — Together answers it with HTTP 500)"),
	chatBound(Fireworks, "", "", "https://app.fireworks.ai/settings/users/api-keys", "Fireworks AI open-model inference (Chat Completions dialect; models are accounts/fireworks/models/<name> ids)"),
	chatBound(Parasail, "", "", "https://www.saas.parasail.io/keys", "Parasail open-model inference (Chat Completions dialect; serverless models)"),
	responsesBound(Meta, "meta", "https://dev.meta.ai/", "Meta Model API — Muse Spark over the Responses wire (reasoning replay, web_search), plus Files, Images (muse-image-1.0) and Models; Meta's docs call the key MODEL_API_KEY — export it as META_API_KEY"),
	chatBound(MetaChat, "meta", "", "https://dev.meta.ai/", "Meta Model API over the Chat Completions wire (same key as `meta`; no cross-turn reasoning)"),
	anthropicBound(MetaAnthropic, "meta", "https://dev.meta.ai/", "Meta Model API over the Anthropic Messages wire (same key as `meta`; bearer token)"),
	hosted(Azure, DialectOpenAIResponses, "", "https://portal.azure.com/", "Azure OpenAI v1 Responses wire ({resource}.openai.azure.com; model = deployment name; api-key or Entra token)"),
	hosted(AzureChat, DialectOpenAIChat, "openai", "https://portal.azure.com/", "Azure OpenAI v1 Chat Completions wire (same resource; also Foundry-sold models such as DeepSeek and Grok)"),
	hosted(AzureAnthropic, DialectAnthropic, "", "https://ai.azure.com/", "Claude in Microsoft Foundry ({resource}.services.ai.azure.com/anthropic; api-key, x-api-key or Entra token)"),
	hosted(AwsAnthropic, DialectAnthropic, "", "https://console.aws.amazon.com/", "Claude Platform on AWS (Anthropic-operated; SigV4 or ANTHROPIC_AWS_API_KEY; needs AWS_REGION and ANTHROPIC_AWS_WORKSPACE_ID)"),
	hosted(BedrockAnthropic, DialectAnthropic, "", "https://console.aws.amazon.com/bedrock/", "Claude in Amazon Bedrock (bedrock-mantle, Opus 4.7 and later; SigV4 or AWS_BEARER_TOKEN_BEDROCK; needs AWS_REGION)"),
	hosted(BedrockChat, DialectOpenAIChat, "bedrock", "https://console.aws.amazon.com/bedrock/", "Amazon Bedrock over the OpenAI Chat Completions wire (bedrock-runtime /openai/v1; SigV4 or AWS_BEARER_TOKEN_BEDROCK)"),
	hosted(BedrockMantleChat, DialectOpenAIChat, "bedrock-mantle", "https://console.aws.amazon.com/bedrock/", "Amazon Bedrock Chat Completions on bedrock-mantle (un-versioned ids, GET /v1/models; SigV4 or AWS_BEARER_TOKEN_BEDROCK)"),
	hosted(Vertex, DialectGemini, "", "https://console.cloud.google.com/vertex-ai", "Gemini on Google Cloud (Agent Platform); ADC chain; needs GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION defaults to global"),
	hosted(VertexExpress, DialectGemini, "", "https://console.cloud.google.com/vertex-ai/studio", "Agent Platform express mode: GOOGLE_API_KEY as ?key=, no project or location"),
	hosted(VertexAnthropic, DialectAnthropic, "", "https://console.cloud.google.com/vertex-ai/model-garden", "Claude on Google Cloud (rawPredict; model in the path, anthropic_version in the body)"),
	chatBound(Ollama, "", "ollama", "", "local ollama server (keyless)"),
	chatBound(VLLM, "", "EMPTY", "", "local vLLM server (keyless)"),
	chatBound(SGLang, "", "EMPTY", "", "local SGLang server (keyless)"),
}

// Providers maps every provider id to its definition.
var Providers = func() map[string]ProviderDefinition {
	out := make(map[string]ProviderDefinition, len(providerDefinitions))
	for _, d := range providerDefinitions {
		out[d.ID] = d
	}
	return out
}()

// ProviderIDs lists every provider id, sorted.
func ProviderIDs() []string {
	out := make([]string, 0, len(Providers))
	for id := range Providers {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// LookupProvider returns the definition for a provider string in either spelling.
func LookupProvider(name string) (ProviderDefinition, bool) {
	d, ok := Providers[CanonicalProvider(name)]
	return d, ok
}

// providerEnvKeys returns the declared env keys for a routable provider.
func providerEnvKeys(provider string) []string {
	if d, ok := Providers[provider]; ok {
		return d.Access.EnvKeys
	}
	return nil
}

// providerCredentialPolicy returns the declared AUTH-1 policy.
func providerCredentialPolicy(provider string) string {
	if d, ok := Providers[provider]; ok {
		return d.CredentialPolicy()
	}
	return "key"
}
