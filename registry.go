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

// definitionOf is a generated table row as a definition: every row but an
// adapter-owned one has its access policy bound by the router.
func definitionOf(r tableRow, declared bool) ProviderDefinition {
	return ProviderDefinition{
		ID: r.ID, Dialect: r.Dialect, Access: r.Access, Compat: r.Compat, CompatValue: r.CompatValue,
		PlaceholderKey: r.PlaceholderKey, ConsoleURL: r.ConsoleURL, Note: r.Note,
		AdapterOwned: r.Kind == "adapter-owned", Aliases: r.Aliases, Declared: declared,
	}
}

// providerDefinitions is the declaration order (presentation order): the
// reference's rows (lm15-contract tables/providers.json, generated into
// tables_generated.go). A provider is added there, never here.
var providerDefinitions = func() []ProviderDefinition {
	out := make([]ProviderDefinition, 0, len(tableProviders))
	for _, r := range tableProviders {
		out = append(out, definitionOf(r, false))
	}
	return out
}()

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
