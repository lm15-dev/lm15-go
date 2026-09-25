package lm15

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
)

// The router: a lookup table you can read. Three rungs in fixed order —
// explicit prefix, catalog (opt-in), built-in rules — nothing else.

// RouteRule maps a model-id prefix to a provider.
type RouteRule struct {
	Prefix   string
	Provider string
	Note     string
}

// DefaultRules is the complete built-in knowledge of the router; first match wins.
var DefaultRules = []RouteRule{
	{"claude-", "anthropic", "Anthropic Claude family"},
	{"gpt-", "openai", "OpenAI GPT family (Responses API; use openai-chat: for Chat Completions)"},
	{"o1", "openai", "OpenAI o1 reasoning family"},
	{"o3", "openai", "OpenAI o3 reasoning family"},
	{"o4", "openai", "OpenAI o4 reasoning family"},
	{"gemini-", "gemini", "Google Gemini family"},
	{"gemma-", "gemini", "Google Gemma open models, served by the Gemini API (live /models listing 2026-09-01)"},
	{"nano-banana", "gemini", "Google image models on the Gemini API (live /models listing 2026-09-01)"},
	{"grok-", "xai", "xAI Grok family (XAI_API_KEY or subscription OAuth)"},
	{"sora-", "openai", "OpenAI Sora video generation"},
	{"veo-", "gemini", "Google Veo video generation"},
	{"chat-latest", "openai", "OpenAI rolling chat alias (live /models listing 2026-09-01)"},
	{"jev-", "typesafe", "TypeSafe Jev (live /v1/models listing 2026-09-17: jev-latest, jev-preview; versioned ids jev-1.13.0 accepted)"},
}

// Resolution is the complete answer to "how did you route this string".
type Resolution struct {
	Requested string
	Model     string
	Provider  string
	Adapter   string
	Source    string // "prefix" | "catalog" | "rule"
	Rule      *RouteRule
	EnvKey    string
	ModelInfo *ModelInfo
	Compat    string
	// Declared: the provider comes from RouterConfig.Providers, not the
	// receipted registry.
	Declared bool
	// CredentialPolicy is the provider's AUTH-1 policy.
	CredentialPolicy string
	// PlaceholderKey is a keyless local server's default key.
	PlaceholderKey string
}

// Describe renders a one-paragraph explanation.
func (r Resolution) Describe() string {
	parts := []string{fmt.Sprintf("%q -> provider %q (%s)", r.Requested, r.Provider, r.Adapter)}
	switch r.Source {
	case "prefix":
		parts = append(parts, "via explicit provider prefix")
	case "catalog":
		parts = append(parts, "via catalog match")
	case "rule":
		if r.Rule != nil {
			note := ""
			if r.Rule.Note != "" {
				note = " — " + r.Rule.Note
			}
			parts = append(parts, fmt.Sprintf("via built-in rule prefix=%q%s", r.Rule.Prefix, note))
		}
	}
	if r.Compat != "" {
		parts = append(parts, fmt.Sprintf("compat preset %q", r.Compat))
	}
	if r.Declared {
		parts = append(parts, "declared by RouterConfig.Providers — no lm15 receipts")
	}
	parts = append(parts, fmt.Sprintf("wire model %q", r.Model))
	def := ProviderDefinition{PlaceholderKey: r.PlaceholderKey}
	policy := r.CredentialPolicy
	if policy == "" {
		policy = "key"
	}
	switch {
	case policy == "oauth-unless-explicit":
		chain := "key from explicit api_keys, else the stored subscription OAuth credential"
		if r.EnvKey != "" {
			chain += ", else $" + r.EnvKey
		}
		parts = append(parts, chain)
	case r.EnvKey != "":
		parts = append(parts, "key from $"+r.EnvKey)
	case policy == "oauth":
		parts = append(parts, "local OAuth credential (no env key)")
	case def.PlaceholderKey != "":
		parts = append(parts, "key from explicit api_keys or the preset's local-server default")
	default:
		parts = append(parts, "key from explicit api_keys")
	}
	return strings.Join(parts, "; ") + "."
}

func (r Resolution) String() string { return r.Describe() }

// RouterConfig is everything the router consults. All explicit.
//
// BaseURLs: a provider string → the address to send to: a proxy in front
// of OpenAI, a vLLM server on another port. On a cloud door (azure,
// bedrock, vertex) the entry is the endpoint root — what the console
// shows, a private endpoint, a gateway — and the door appends its own
// path (/openai/v1, /anthropic/v1) unless the URL already ends with it;
// the door's auth scheme, error mapping and doctor stay attached. Without
// an entry the vendor's own variable is read (AZURE_OPENAI_ENDPOINT,
// ANTHROPIC_FOUNDRY_BASE_URL, AWS_ENDPOINT_URL_BEDROCK_RUNTIME /
// AWS_ENDPOINT_URL), then the URL is built from Settings. With an
// endpoint, resource is not needed; region still is on AWS (it signs).
//
// Credentials maps a cloud provider string → one named identity:
// "platform" (the machine's own), "workload" (the federated Kubernetes
// kind), "environment" (a service principal or static keys from env
// variables), "cli" (az / aws / gcloud sign-in). That rung only is used
// and the cloud's chain is not walked (AUTH-1, amended 2026-09-19). An
// APIKeys entry and a Credentials entry for one provider is refused.
//
// Providers declares providers the registry does not list
// (DeclareChatProvider). They route like registry entries in every router
// built with this config and answer Resolution.Declared.
//
// Timeouts and MaxConnections shape the one transport the router builds
// and shares across its LMs; defaults are the provider SDKs' (connect
// 10 s, read/write/pool 600 s, 100 connections). Transport replaces that
// transport with one you built; it cannot be combined with the two knobs.
//
// Adaptations is the MAP-13 policy every LM the router builds runs under:
// "note" (default), "silent", "refuse".
type RouterConfig struct {
	Registry       *ModelRegistry
	Rules          []RouteRule       // nil = DefaultRules
	Env            map[string]string // nil = the process environment
	APIKeys        map[string]CredentialLike
	BaseURLs       map[string]string
	Settings       map[string]map[string]string
	Credentials    map[string]string
	Transport      Transport
	Timeouts       *Timeouts
	MaxConnections int
	Adaptations    string
	Providers      []ProviderDefinition
	// Auth is managed authentication (AUTH-15 mode B): its saved connections
	// supply the credential when no explicit APIKeys / Credentials entry
	// does. With it, environment keys, other tools' login files and the
	// machine's cloud identity are never consulted: a missing, expired,
	// rejected or signed-out connection is a typed AuthOperationError, never
	// a silent switch to a metered key. Keyless local servers still work
	// without a connection. It also routes the connection-only providers
	// (kimi-code, github-copilot).
	Auth *Auth
}

// definitions is the provider table this config routes with: the registry
// plus the declared providers.
func (c RouterConfig) definitions() map[string]ProviderDefinition {
	if len(c.Providers) == 0 && c.Auth == nil {
		return Providers
	}
	out := make(map[string]ProviderDefinition, len(Providers)+len(c.Providers)+2)
	for k, v := range Providers {
		out[k] = v
	}
	if c.Auth != nil {
		// A managed router also routes the connection-only doors: declared
		// providers, added only here because only a managed Auth can hold
		// their credential.
		for _, d := range DeclaredLoginProviders {
			out[d.ID] = d
		}
	}
	for _, d := range c.Providers {
		out[d.ID] = d
	}
	return out
}

// providerID maps a canonical input spelling to the provider it names: a
// declared alias resolves to its provider; anything else is itself.
func (c RouterConfig) providerID(name string) string {
	for _, d := range c.Providers {
		if inVocab(name, d.Aliases) {
			return d.ID
		}
	}
	return name
}

func (c RouterConfig) lookup(provider string) (ProviderDefinition, bool) {
	d, ok := c.definitions()[provider]
	return d, ok
}

func (c RouterConfig) routable(provider string) bool {
	_, ok := c.lookup(provider)
	return ok
}

func (c RouterConfig) providerIDs() []string {
	defs := c.definitions()
	out := make([]string, 0, len(defs))
	for id := range defs {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func (c RouterConfig) envKeysOf(provider string) []string {
	if d, ok := c.lookup(provider); ok {
		return d.Access.EnvKeys
	}
	return nil
}

// checkDeclared validates RouterConfig.Providers: definitions only, each
// spelling naming one door that nothing built in already names.
func checkDeclared(providers []ProviderDefinition) error {
	taken := map[string]string{}
	for _, d := range providers {
		if !d.Declared {
			return typeErrorf("RouterConfig.Providers: %q is not a declared provider; declare one with DeclareChatProvider", d.ID)
		}
		for _, spelling := range d.Spellings() {
			builtIn := ""
			if existing, ok := Providers[spelling]; ok {
				builtIn = existing.ID
			} else if p, ok := LitellmProviderPrefixes[spelling]; ok {
				builtIn = p
			}
			if builtIn != "" {
				return NotConfiguredErrorf("", nil, "", "RouterConfig.Providers: %q already names lm15's %q door; a declared provider takes a new id and aliases", spelling, builtIn)
			}
			if other, dup := taken[spelling]; dup {
				return NotConfiguredErrorf("", nil, "", "RouterConfig.Providers: %q is spelled by both %q and %q", spelling, other, d.ID)
			}
			taken[spelling] = d.ID
		}
	}
	return nil
}

// Validate checks the config's own consistency (the provider-keyed maps
// are checked by NewRouterWithConfig).
func (c RouterConfig) Validate() error {
	if c.Adaptations != "" {
		if err := checkAdaptationPolicy(c.Adaptations); err != nil {
			return err
		}
	}
	if err := checkDeclared(c.Providers); err != nil {
		return err
	}
	for key, name := range c.Credentials {
		if !inVocab(name, NamedCredentials) {
			return NotConfiguredErrorf("", nil, "", "RouterConfig.Credentials{%q: %q}: not a named credential; one of %s. A credential VALUE (a key, a token, a provider) goes in APIKeys.", key, name, strings.Join(NamedCredentials, ", "))
		}
	}
	if c.Timeouts != nil {
		if err := c.Timeouts.Validate(); err != nil {
			return err
		}
	}
	if err := checkMaxConnections(c.MaxConnections); err != nil {
		return err
	}
	if c.Transport != nil && (c.Timeouts != nil || c.MaxConnections != 0) {
		return NotConfiguredErrorf("", nil, "", "RouterConfig.Transport cannot be combined with Timeouts or MaxConnections: they configure the transport lm15 would build, and would silently not apply to the one you passed. Configure that transport directly (NewHTTPTransportWith).")
	}
	return nil
}

func (c RouterConfig) adaptations() string {
	if c.Adaptations == "" {
		return AdaptationsNote
	}
	return c.Adaptations
}

func (c RouterConfig) rules() []RouteRule {
	if c.Rules == nil {
		return DefaultRules
	}
	return c.Rules
}

func (c RouterConfig) env() map[string]string {
	if c.Env != nil {
		return c.Env
	}
	out := map[string]string{}
	for _, kv := range os.Environ() {
		if k, v, ok := strings.Cut(kv, "="); ok {
			out[k] = v
		}
	}
	return out
}

func adapterName(dialect string) string {
	switch dialect {
	case DialectOpenAIResponses:
		return "OpenAILM"
	case DialectOpenAIChat:
		return "OpenAIChatLM"
	case DialectAnthropic:
		return "AnthropicLM"
	case DialectGemini:
		return "GeminiLM"
	case DialectTypeSafe:
		return "TypeSafeLM"
	}
	return dialect
}

func knownProviders() string { return strings.Join(ProviderIDs(), ", ") }

func unknownModel(model, message string) *Error {
	e := newError(KindUnknownModel, message)
	e.Model = model
	return e
}

func ambiguousModel(model, message string, providers []string) *Error {
	e := newError(KindAmbiguousModel, message)
	e.Model = model
	e.Providers = providers
	return e
}

// checkProviderKeyed refuses RouterConfig entries keyed by a non-provider.
func checkProviderKeyed(config RouterConfig) error {
	known := config.providerIDs()
	check := func(field string, keys []string) error {
		seen := map[string]bool{}
		for _, key := range keys {
			provider := config.providerID(CanonicalProvider(key))
			if (field == "api_keys" || field == "credentials") && seen[provider] {
				return NotConfiguredErrorf("", nil, "", "RouterConfig(%s=...): duplicate spellings for %q; use one entry", field, provider)
			}
			seen[provider] = true
			if config.routable(provider) {
				if field == "credentials" {
					if err := checkNamedCredential(config, provider, config.Credentials[key]); err != nil {
						return err
					}
				}
				continue
			}
			hint := ""
			if close := closestMatch(provider, known, 0.6); close != "" {
				hint = fmt.Sprintf(" Did you mean %q?", close)
			}
			return NotConfiguredErrorf("", nil, "", "RouterConfig(%s=...): %q is not a provider lm15 routes to.%s router.resolve(model).provider (or resolve_openai_chat) names the one a model string uses; known: %s", field, key, hint, strings.Join(known, ", "))
		}
		return nil
	}
	var apiKeys, baseURLs, settings, credentials []string
	for k := range config.APIKeys {
		apiKeys = append(apiKeys, k)
	}
	for k := range config.BaseURLs {
		baseURLs = append(baseURLs, k)
	}
	for k := range config.Settings {
		settings = append(settings, k)
	}
	for k := range config.Credentials {
		credentials = append(credentials, k)
	}
	sort.Strings(apiKeys)
	sort.Strings(baseURLs)
	sort.Strings(settings)
	sort.Strings(credentials)
	if err := check("api_keys", apiKeys); err != nil {
		return err
	}
	if err := check("base_urls", baseURLs); err != nil {
		return err
	}
	if err := check("settings", settings); err != nil {
		return err
	}
	return check("credentials", credentials)
}

// checkNamedCredential: a Credentials entry names an identity on a cloud
// door only, and never alongside an APIKeys entry for the same provider.
func checkNamedCredential(config RouterConfig, provider, name string) error {
	def, ok := config.lookup(provider)
	if !ok || !def.Access.CloudChain() {
		return NotConfiguredErrorf("", nil, "", "RouterConfig.Credentials{%q: %q}: %q is not a cloud door; named credentials (platform, workload, environment, cli) exist on the azure, bedrock and vertex doors. Pass this provider's credential in APIKeys.", provider, name, provider)
	}
	source, err := apiKeysSource(config, provider)
	if err != nil {
		return err
	}
	if source != "" {
		return NotConfiguredErrorf("", nil, "", "RouterConfig: both APIKeys and Credentials name %q; a door has one identity — pass the credential value (APIKeys) or name the identity (Credentials), not both.", provider)
	}
	return nil
}

// credentialsEntry is the named credential for a provider (AUTH-1),
// matching either spelling.
func credentialsEntry(config RouterConfig, provider string) string {
	for key, value := range config.Credentials {
		if config.providerID(CanonicalProvider(key)) == provider {
			return value
		}
	}
	return ""
}

// baseURLEntry is the BaseURLs entry for a provider, matching either spelling.
func baseURLEntry(config RouterConfig, provider string) string {
	for key, value := range config.BaseURLs {
		if config.providerID(CanonicalProvider(key)) == provider {
			return value
		}
	}
	return ""
}

// hostedEndpoint is the endpoint root for a cloud door: the explicit
// BaseURLs entry, else the vendor's own variable (AUTH-10, amended
// 2026-09-19).
func hostedEndpoint(config RouterConfig, provider string, def ProviderDefinition) string {
	if explicit := baseURLEntry(config, provider); explicit != "" {
		return explicit
	}
	return EndpointFromEnv(def.Access.Host, config.env())
}

// closestMatch is a difflib.get_close_matches stand-in (ratio ≥ cutoff).
func closestMatch(word string, candidates []string, cutoff float64) string {
	best, bestScore := "", cutoff
	for _, c := range candidates {
		if s := similarity(word, c); s >= bestScore {
			best, bestScore = c, s
		}
	}
	return best
}

func similarity(a, b string) float64 {
	if len(a)+len(b) == 0 {
		return 1
	}
	matches := lcsLength(a, b)
	return 2.0 * float64(matches) / float64(len(a)+len(b))
}

func lcsLength(a, b string) int {
	prev := make([]int, len(b)+1)
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		for j := 1; j <= len(b); j++ {
			if a[i-1] == b[j-1] {
				cur[j] = prev[j-1] + 1
			} else if prev[j] > cur[j-1] {
				cur[j] = prev[j]
			} else {
				cur[j] = cur[j-1]
			}
		}
		prev = cur
	}
	return prev[len(b)]
}

// apiKeysSource selects a config key (never its value): exact provider
// first, else the single entry whose declared env-key tuple is identical.
func apiKeysSource(config RouterConfig, provider string) (string, error) {
	if len(config.APIKeys) == 0 {
		return "", nil
	}
	var keys []string
	for k := range config.APIKeys {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var candidates []string
	for _, k := range keys {
		if config.providerID(CanonicalProvider(k)) == provider {
			candidates = append(candidates, k)
		}
	}
	if len(candidates) == 0 && config.routable(provider) {
		envKeys := config.envKeysOf(provider)
		if len(envKeys) > 0 {
			for _, k := range keys {
				canon := config.providerID(CanonicalProvider(k))
				if config.routable(canon) && sameStrings(config.envKeysOf(canon), envKeys) {
					candidates = append(candidates, k)
				}
			}
		}
	}
	if len(candidates) > 1 {
		quoted := make([]string, 0, len(candidates))
		for _, c := range candidates {
			quoted = append(quoted, fmt.Sprintf("%q", c))
		}
		return "", NotConfiguredErrorf("", nil, "", "RouterConfig(api_keys=...): ambiguous credentials for %q from %s; supply one entry under %q or keep only one shared entry", provider, strings.Join(quoted, ", "), provider)
	}
	if len(candidates) == 0 {
		return "", nil
	}
	key := candidates[0]
	value := config.APIKeys[key]
	if value == nil {
		return "", NotConfiguredErrorf("", nil, "", "RouterConfig(api_keys=...): empty credential under %q; no environment fallback", key)
	}
	if s, ok := value.(string); ok && s == "" {
		return "", NotConfiguredErrorf("", nil, "", "RouterConfig(api_keys=...): empty credential under %q; no environment fallback", key)
	}
	return key, nil
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func envKeyFor(provider string, config RouterConfig) (string, error) {
	source, err := apiKeysSource(config, provider)
	if err != nil {
		return "", err
	}
	if source != "" {
		return "", nil
	}
	envKeys := config.envKeysOf(provider)
	if len(envKeys) == 0 {
		return "", nil
	}
	env := config.env()
	for _, k := range envKeys {
		if env[k] != "" {
			return k, nil
		}
	}
	return envKeys[0], nil
}

func resolution(requested, wireModel, provider, source string, config RouterConfig, rule *RouteRule, info *ModelInfo) (Resolution, error) {
	def, _ := config.lookup(provider)
	envKey, err := envKeyFor(provider, config)
	if err != nil {
		return Resolution{}, err
	}
	res := Resolution{Requested: requested, Model: wireModel, Provider: provider, Adapter: adapterName(def.Dialect), Source: source, Rule: rule, EnvKey: envKey, ModelInfo: info,
		Declared: def.Declared, CredentialPolicy: def.CredentialPolicy(), PlaceholderKey: def.PlaceholderKey}
	if def.Bound() {
		res.Compat = def.Compat
	}
	return res, nil
}

// Resolve is the pure lookup: no network, no credential invocation.
func Resolve(model string, config RouterConfig) (Resolution, error) {
	if model == "" {
		return Resolution{}, unknownModel(model, "model must be a non-empty string")
	}
	if head, rest, ok := strings.Cut(model, ":"); ok {
		provider := config.providerID(CanonicalProvider(head))
		if config.routable(provider) && rest != "" {
			return resolution(model, rest, provider, "prefix", config, nil, nil)
		}
	}
	if config.Registry != nil {
		var matches []ModelInfo
		for _, info := range config.Registry.List("") {
			if info.ID == model || inVocab(model, info.Aliases) {
				matches = append(matches, info)
			}
		}
		var providers []string
		seen := map[string]bool{}
		for _, info := range matches {
			if !seen[info.Provider] {
				seen[info.Provider] = true
				providers = append(providers, info.Provider)
			}
		}
		if len(providers) > 1 {
			options := make([]string, 0, len(providers))
			for _, p := range providers {
				options = append(options, fmt.Sprintf("%q", p+":"+model))
			}
			return Resolution{}, ambiguousModel(model, fmt.Sprintf("model %q is offered by multiple providers: %s. Fix: use the explicit form, e.g. Request(model=%q) — options: %s.", model, strings.Join(providers, ", "), providers[0]+":"+model, strings.Join(options, " or ")), providers)
		}
		if len(matches) > 0 {
			var exact []ModelInfo
			for _, info := range matches {
				if info.ID == model {
					exact = append(exact, info)
				}
			}
			narrowed := matches
			if len(exact) > 0 {
				narrowed = exact
			}
			if len(narrowed) > 1 {
				ids := make([]string, 0, len(narrowed))
				for _, info := range narrowed {
					ids = append(ids, info.ID)
				}
				return Resolution{}, ambiguousModel(model, fmt.Sprintf("model %q matches multiple catalog entries (%s) under provider %q. Fix: request a canonical id directly.", model, strings.Join(ids, ", "), narrowed[0].Provider), providers)
			}
			info := narrowed[0]
			provider := config.providerID(CanonicalProvider(info.Provider))
			if !config.routable(provider) {
				return Resolution{}, unknownModel(model, fmt.Sprintf("model %q resolved in the catalog to provider %q, but lm15 has no adapter or compat preset for it. Known providers: %s. Construct a provider LM directly (e.g. OpenAIChatLM with a custom base_url) for OpenAI-compatible servers.", model, info.Provider, strings.Join(config.providerIDs(), ", ")))
			}
			wire := model
			if inVocab(model, info.Aliases) {
				wire = info.ID
			}
			copied := info
			return resolution(model, wire, provider, "catalog", config, nil, &copied)
		}
	}
	for i := range config.rules() {
		rule := config.rules()[i]
		if strings.HasPrefix(model, rule.Prefix) {
			provider := config.providerID(CanonicalProvider(rule.Provider))
			if !config.routable(provider) {
				return Resolution{}, unknownModel(model, fmt.Sprintf("rule %+v names provider %q, which has no adapter. Known providers: %s.", rule, rule.Provider, strings.Join(config.providerIDs(), ", ")))
			}
			return resolution(model, model, provider, "rule", config, &rule, nil)
		}
	}
	var hints []string
	if head, rest, ok := strings.Cut(model, ":"); ok {
		if close := closestMatch(CanonicalProvider(head), config.providerIDs(), 0.75); close != "" {
			hints = append(hints, fmt.Sprintf("Did you mean %q?", close+":"+rest))
		}
	}
	hints = append(hints, fmt.Sprintf("Use an explicit provider prefix — \"provider:%s\" with provider one of: %s.", model, strings.Join(config.providerIDs(), ", ")))
	catalog := "no catalog supplied"
	if config.Registry != nil {
		catalog = "no catalog match"
	} else {
		hints = append(hints, "Or pass a model catalog: LMRouter(config=RouterConfig(registry=ModelRegistry.discover())) — install a catalog package such as 'aimo' first.")
	}
	return Resolution{}, unknownModel(model, fmt.Sprintf("could not route model %q: no provider prefix, %s, and none of the %d built-in rules matched. %s", model, catalog, len(config.rules()), strings.Join(hints, " ")))
}

// buildLM constructs the provider LM for a resolution (AUTH-1 chain).
func buildLM(res Resolution, config RouterConfig, transport Transport) (LM, error) {
	def, _ := config.lookup(res.Provider)
	var opts []Option
	if transport != nil {
		opts = append(opts, WithTransport(transport))
	} else if config.Transport != nil {
		opts = append(opts, WithTransport(config.Transport))
	}
	if config.adaptations() != AdaptationsNote {
		opts = append(opts, WithAdaptations(config.adaptations()))
	}
	baseURL := baseURLEntry(config, res.Provider)
	if def.Hosted() {
		baseURL = hostedEndpoint(config, res.Provider, def)
	}
	if baseURL != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}
	policy := def.CredentialPolicy()
	if config.Auth != nil {
		return buildManagedLM(res, config, def, opts, baseURL)
	}
	if def.Bound() {
		opts = append(opts, WithAccess(def.Access))
		switch {
		case def.CompatValue != nil:
			opts = append(opts, WithOpenAIChatCompat(*def.CompatValue))
		case def.Compat != "":
			opts = append(opts, WithCompatPreset(def.Compat))
		}
	}
	if policy == "oauth" {
		return construct(def, opts)
	}
	source, err := apiKeysSource(config, res.Provider)
	if err != nil {
		return nil, err
	}
	var apiKey CredentialLike
	if source != "" {
		apiKey = config.APIKeys[source]
	}
	env := config.env()
	origin := ""
	if def.Hosted() {
		ctx := OnlineChainContext(env)
		var given map[string]string
		for key, s := range config.Settings {
			if config.providerID(CanonicalProvider(key)) == res.Provider {
				given = s
			}
		}
		settings, err := resolveSettingsWithEndpoint(def.Access.Host, given, env, res.Provider, ProfileSettings(def.Access, ctx), baseURL)
		if err != nil {
			return nil, err
		}
		ctx.Settings = settings
		named := credentialsEntry(config, res.Provider)
		if apiKey == nil && def.Access.CloudChain() {
			provider, err := NamedCredentialProviderFor(def.Access, ctx, named)
			if err != nil {
				return nil, err
			}
			apiKey = provider
		} else if apiKey == nil {
			for _, k := range def.Access.EnvKeys {
				if v := env[k]; v != "" {
					apiKey = v
					origin = "env $" + k + " (value never shown)"
					break
				}
			}
		}
		if apiKey == nil {
			return nil, missingCredential(res.Provider, def.Access.EnvKeys, "credential")
		}
		return constructWithOrigin(def, append(opts, WithAPIKey(apiKey), WithSettings(settings)), origin)
	}
	if apiKey == nil && policy == "oauth-unless-explicit" {
		// A usable stored subscription login outranks ambient env keys: it
		// spends no money per token (AUTH-1). An unusable or signed-out one
		// BLOCKS them (R3, ratified 2026-09-22): a failed subscription is
		// never silently replaced by a metered key.
		switch state := StoredCredentialState(def.Access); state {
		case "usable":
			return construct(def, opts)
		case "unusable", "logged_out":
			what := "is expired and cannot be renewed"
			if state == "logged_out" {
				what = "was signed out"
			}
			present := ""
			for _, k := range def.Access.EnvKeys {
				if env[k] != "" {
					present = "$" + k + " is set but is used only when passed explicitly: "
					break
				}
			}
			e := NotConfiguredErrorf(res.Provider, def.Access.EnvKeys, def.Access.LoginHint,
				"the %q subscription login %s. %ssign in again, or pass the key deliberately with RouterConfig(api_keys={%q: \"...\"}).",
				res.Provider, what, present, res.Provider)
			e.Kind = KindMissingCredential
			return nil, e
		}
	}
	if apiKey == nil {
		for _, k := range def.Access.EnvKeys {
			if v := env[k]; v != "" {
				apiKey = v
				origin = "env $" + k + " (value never shown)"
				break
			}
		}
	}
	if apiKey == nil && def.PlaceholderKey != "" {
		apiKey = def.PlaceholderKey
		origin = "the local server's placeholder key"
	}
	if apiKey == nil && policy == "oauth-unless-explicit" {
		return construct(def, opts) // the constructor raises the typed login-hint error
	}
	if apiKey == nil {
		return nil, missingCredential(res.Provider, def.Access.EnvKeys, "API key")
	}
	return constructWithOrigin(def, append(opts, WithAPIKey(apiKey)), origin)
}

// buildManagedLM is AUTH-15 mode B. Order: an explicit APIKeys entry; an
// explicit named cloud identity; the scope's saved connection (its credential
// resolved and renewed per request); a keyless local server's placeholder.
// Never an environment key, another tool's login file or the machine's cloud chain.
func buildManagedLM(res Resolution, config RouterConfig, def ProviderDefinition, opts []Option, baseURL string) (LM, error) {
	auth := config.Auth
	source, err := apiKeysSource(config, res.Provider)
	if err != nil {
		return nil, err
	}
	var apiKey CredentialLike
	if source != "" {
		apiKey = config.APIKeys[source]
	}
	named := credentialsEntry(config, res.Provider)
	origin := ""
	access := def.Access
	if apiKey == nil && named == "" {
		connection, shape, err := auth.Selection(res.Provider)
		switch {
		case err == nil:
			if shape.Named != "" {
				named = shape.Named
			} else {
				provider := res.Provider
				apiKey = CredentialFunc(func(ctx context.Context) (Credential, error) {
					got, err := auth.RequestAuth(ctx, provider, nil)
					if err != nil {
						return nil, err
					}
					if got.CredentialKind == "bearer" {
						return BearerToken{Value: got.Credential}, nil
					}
					return APIKey{Value: got.Credential}, nil
				})
				if shape.AccountID != "" {
					opts = append(opts, WithAccountID(shape.AccountID))
				}
				if shape.BaseURL != "" && baseURL == "" && !def.Hosted() {
					opts = append(opts, WithBaseURL(shape.BaseURL))
				}
				if !def.Hosted() {
					access.Headers = mergeHeaders(access.Headers, shape.Headers)
				}
			}
			origin = "managed connection " + connection.ID + " (" + connection.Label + ")"
		case isReason(err, "login_required") && def.PlaceholderKey != "":
			status, serr := auth.Status(res.Provider)
			if serr != nil || status.LoggedOut {
				return nil, err
			}
			apiKey = def.PlaceholderKey
			origin = "the local server's placeholder key"
		case isReason(err, "login_required") && def.Hosted():
			e := authOperation(res.Provider+": no saved connection in this scope; the machine's cloud identity is not used under a managed Auth — save a named identity (Auth.Configure with the cloud method) or pass RouterConfig.Credentials explicitly", "login_required", "resolution", "not_committed", "select_connection")
			e.Provider = res.Provider
			return nil, e
		default:
			return nil, err
		}
	}
	if def.Bound() || def.Declared {
		opts = append(opts, WithAccess(access))
		switch {
		case def.CompatValue != nil:
			opts = append(opts, WithOpenAIChatCompat(*def.CompatValue))
		case def.Compat != "":
			opts = append(opts, WithCompatPreset(def.Compat))
		}
	}
	if def.Hosted() {
		env := config.env()
		ctx := OnlineChainContext(env)
		var given map[string]string
		for key, s := range config.Settings {
			if config.providerID(CanonicalProvider(key)) == res.Provider {
				given = s
			}
		}
		settings, err := resolveSettingsWithEndpoint(def.Access.Host, given, env, res.Provider, ProfileSettings(def.Access, ctx), baseURL)
		if err != nil {
			return nil, err
		}
		ctx.Settings = settings
		if apiKey == nil && named != "" {
			provider, err := NamedCredentialProviderFor(def.Access, ctx, named)
			if err != nil {
				return nil, err
			}
			apiKey = provider
		}
		if apiKey == nil {
			e := authOperation(res.Provider+": no credential for this cloud door under a managed Auth", "login_required", "resolution", "not_committed", "select_connection")
			e.Provider = res.Provider
			return nil, e
		}
		return constructWithOrigin(def, append(opts, WithAPIKey(apiKey), WithSettings(settings)), origin)
	}
	if apiKey == nil {
		e := authOperation(res.Provider+": no saved connection in this scope; sign in with Auth.Login or Connect", "login_required", "resolution", "not_committed", "restart_login")
		e.Provider = res.Provider
		return nil, e
	}
	return constructWithOrigin(def, append(opts, WithAPIKey(apiKey)), origin)
}

func isReason(err error, reason string) bool {
	var e *Error
	return errors.As(err, &e) && e.Kind == KindAuthOperation && e.Reason == reason
}

func mergeHeaders(static [][2]string, extra map[string]string) [][2]string {
	out := append([][2]string{}, static...)
	keys := make([]string, 0, len(extra))
	for k := range extra {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if strings.EqualFold(k, "chatgpt-account-id") {
			continue // the Codex adapter sends it from the account id
		}
		taken := false
		for _, h := range out {
			if strings.EqualFold(h[0], k) {
				taken = true
			}
		}
		if !taken {
			out = append(out, [2]string{k, extra[k]})
		}
	}
	return out
}

// DeclaredLoginProviders are the routes that exist only for a managed
// connection (lm15-python lm15/login/declared.py): kimi-code (Anthropic
// Messages at api.kimi.com/coding) and github-copilot (Chat Completions at
// the account's Copilot host). No contract wire receipt, so no registry row
// (a row is a support claim, AUTH-26); a router routes them only with a
// managed Auth.
var DeclaredLoginProviders = []ProviderDefinition{
	{
		ID: "kimi-code", Dialect: DialectAnthropic, Declared: true,
		Access: AccessPolicy{Provider: "kimi-code", Supports: EndpointSupport{Complete: true, Stream: true}, AuthModes: []string{"bearer"}, AuthScheme: []string{"bearer"}, BaseURL: "https://api.kimi.com/coding"},
		Note:   "Kimi Code subscription over the Anthropic Messages wire (managed login only; no lm15 wire receipt yet)",
	},
	{
		ID: "github-copilot", Dialect: DialectOpenAIChat, Declared: true,
		Access: AccessPolicy{Provider: "github-copilot", Supports: EndpointSupport{Complete: true, Stream: true, Models: true}, AuthModes: []string{"bearer"}, AuthScheme: []string{"bearer"},
			Headers: copilotHeaders, BaseURL: copilotDefaultAPIBase},
		CompatValue: &OpenAIChatCompat{InstructionRole: "system", MaxTokensField: "max_completion_tokens", StreamUsage: "include", ThinkingFormat: "reasoning_effort"},
		Note:        "GitHub Copilot over the Chat Completions wire (managed login only; the account's host comes from the token; no lm15 wire receipt yet)",
	},
}

// PlanningKey is the placeholder credential a planning LM carries.
const PlanningKey = "lm15-planning"

// buildPlanningLM is a throwaway LM for Plan: the build's bytes are
// discarded, so it carries a placeholder credential under every policy —
// no stored login is read or refreshed, no cloud chain is walked, no
// environment key is needed — and placeholder host settings where a cloud
// door would otherwise refuse to render its URL. Never cached. Everything
// else (compat preset, access policy, base URL, adaptations policy) is the
// real route's, so the plan is the call's.
func buildPlanningLM(res Resolution, config RouterConfig, transport Transport) (LM, error) {
	def, _ := config.lookup(res.Provider)
	opts := []Option{WithAPIKey(PlanningKey), WithAdaptations(config.adaptations())}
	if transport != nil {
		opts = append(opts, WithTransport(transport))
	} else if config.Transport != nil {
		opts = append(opts, WithTransport(config.Transport))
	}
	baseURL := baseURLEntry(config, res.Provider)
	if def.Hosted() {
		baseURL = hostedEndpoint(config, res.Provider, def)
	}
	if baseURL != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}
	if def.ID == "openai-codex" {
		opts = append(opts, WithAccountID(PlanningKey))
	}
	if def.Bound() {
		opts = append(opts, WithAccess(def.Access))
		switch {
		case def.CompatValue != nil:
			opts = append(opts, WithOpenAIChatCompat(*def.CompatValue))
		case def.Compat != "":
			opts = append(opts, WithCompatPreset(def.Compat))
		}
	}
	if def.Hosted() {
		env := config.env()
		var given map[string]string
		for key, s := range config.Settings {
			if config.providerID(CanonicalProvider(key)) == res.Provider {
				given = s
			}
		}
		settings, err := resolveSettingsWithEndpoint(def.Access.Host, given, env, res.Provider, nil, baseURL)
		if err != nil {
			if !IsKind(err, KindNotConfigured) {
				return nil, err
			}
			placeholders := map[string]string{}
			for _, setting := range def.Access.Host.Settings {
				value := given[setting.Name]
				if value == "" {
					value = setting.Default
				}
				if value == "" {
					value = "planning"
				}
				placeholders[setting.Name] = value
			}
			if settings, err = resolveSettings(def.Access.Host, placeholders, nil, res.Provider, nil); err != nil {
				return nil, err
			}
		}
		opts = append(opts, WithSettings(settings))
	}
	return construct(def, opts)
}

func constructWithOrigin(def ProviderDefinition, opts []Option, origin string) (LM, error) {
	lm, err := construct(def, opts)
	if err != nil {
		return nil, err
	}
	if origin != "" {
		if stamper, ok := lm.(interface{ SetCredentialOrigin(string) }); ok {
			stamper.SetCredentialOrigin(origin)
		}
	}
	return lm, nil
}

func missingCredential(provider string, envKeys []string, what string) *Error {
	e := NotConfiguredErrorf(provider, envKeys, "", "no %s found for provider %q. Set %s in the environment, or pass RouterConfig(api_keys={%q: \"...\"}).", what, provider, strings.Join(envKeys, " or "), provider)
	e.Kind = KindMissingCredential
	return e
}

func construct(def ProviderDefinition, opts []Option) (LM, error) {
	switch def.ID {
	case "claude-code":
		return NewClaudeCodeLM(opts...)
	case "openai-codex":
		return NewOpenAICodexLM(opts...)
	case "xai":
		return NewXaiLM(opts...)
	}
	switch def.Dialect {
	case DialectOpenAIResponses:
		return NewOpenAILM(opts...)
	case DialectOpenAIChat:
		return NewOpenAIChatLM(opts...)
	case DialectAnthropic:
		return NewAnthropicLM(opts...)
	case DialectGemini:
		return NewGeminiLM(opts...)
	case DialectTypeSafe:
		return NewTypeSafeLM(opts...)
	}
	return nil, valueErrorf("unknown dialect %q", def.Dialect)
}

func routedRequest(req *Request, res Resolution) *Request {
	if req.Model == res.Model {
		return req
	}
	return req.WithModel(res.Model)
}

// LMRouter routes model strings to provider LMs; one LM per provider,
// built lazily, reused, and the one transport those LMs share (the
// router's pool is one pool; Close releases every socket).
type LMRouter struct {
	Config    RouterConfig
	mu        sync.Mutex
	lms       map[string]LM
	transport Transport
}

// NewRouter creates a router over the process environment.
func NewRouter() *LMRouter { return &LMRouter{lms: map[string]LM{}} }

// NewRouterWithConfig creates a router; the config's provider-keyed maps
// are checked for typos and the config for consistency.
func NewRouterWithConfig(config RouterConfig) (*LMRouter, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if err := checkProviderKeyed(config); err != nil {
		return nil, err
	}
	return &LMRouter{Config: config, lms: map[string]LM{}, transport: config.Transport}, nil
}

// Resolve is the offline lookup (also the explain method).
func (r *LMRouter) Resolve(model string) (Resolution, error) { return Resolve(model, r.Config) }

// sharedTransport is the transport every LM of this router uses: the
// configured one, else one HTTPTransport built from Timeouts /
// MaxConnections on first use and shared.
func (r *LMRouter) sharedTransport() Transport {
	if r.transport == nil {
		timeouts := DefaultTimeouts()
		if r.Config.Timeouts != nil {
			timeouts = *r.Config.Timeouts
		}
		r.transport = NewHTTPTransportWith(timeouts, r.Config.MaxConnections)
	}
	return r.transport
}

// Close closes every connection this router holds. Idempotent; the router
// may be used again afterwards (a fresh transport is built).
func (r *LMRouter) Close() error {
	r.mu.Lock()
	transport := r.transport
	r.transport = nil
	r.lms = map[string]LM{}
	r.mu.Unlock()
	if t, ok := transport.(*HTTPTransport); ok && t.Client != nil {
		if inner, ok := t.Client.Transport.(*http.Transport); ok {
			inner.CloseIdleConnections()
		}
		return nil
	}
	if closer, ok := transport.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// LM resolves, then constructs or reuses the provider LM.
func (r *LMRouter) LM(model string) (LM, error) {
	res, err := r.Resolve(model)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.lms == nil {
		r.lms = map[string]LM{}
	}
	if lm, ok := r.lms[res.Provider]; ok {
		return lm, nil
	}
	lm, err := buildLM(res, r.Config, r.sharedTransport())
	if err != nil {
		return nil, err
	}
	r.lms[res.Provider] = lm
	return lm, nil
}

// Plan is the MAP-13 pre-flight: what this request WOULD adapt on its
// route, no network and no credential (like Resolve); it returns what the
// call would return. A route with no key gets a throwaway planning LM
// (not cached): the build's bytes are discarded, so no key is needed.
func (r *LMRouter) Plan(req *Request) ([]Adaptation, error) {
	res, err := r.Resolve(req.Model)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	lm, cached := r.lms[res.Provider]
	transport := r.sharedTransport()
	r.mu.Unlock()
	if !cached {
		if lm, err = buildPlanningLM(res, r.Config, transport); err != nil {
			return nil, err
		}
	}
	return lm.Plan(routedRequest(req, res))
}

// Complete performs one call through the routed LM.
func (r *LMRouter) Complete(ctx context.Context, req *Request) (*Response, error) {
	res, err := r.Resolve(req.Model)
	if err != nil {
		return nil, err
	}
	lm, err := r.LM(req.Model)
	if err != nil {
		return nil, err
	}
	return lm.Complete(ctx, routedRequest(req, res))
}

// Stream streams through the routed LM.
func (r *LMRouter) Stream(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error] {
	res, err := r.Resolve(req.Model)
	if err != nil {
		return errSeq(err)
	}
	lm, err := r.LM(req.Model)
	if err != nil {
		return errSeq(err)
	}
	return lm.Stream(ctx, routedRequest(req, res))
}

// Cache is the MAP-6 door, routed by the prefix's model.
func (r *LMRouter) Cache(ctx context.Context, prefix *Request, ttlSeconds *int, label string) (CachedPrefix, error) {
	res, err := r.Resolve(prefix.Model)
	if err != nil {
		return CachedPrefix{}, err
	}
	lm, err := r.LM(prefix.Model)
	if err != nil {
		return CachedPrefix{}, err
	}
	cached, err := lm.Cache(ctx, routedRequest(prefix, res), ttlSeconds, label)
	if err != nil {
		return CachedPrefix{}, err
	}
	cached.Provider = res.Provider
	return cached, nil
}

// ─── The OpenAI-shaped door (api-family § Ingest) ────────────────────

// LitellmProviderPrefixes maps litellm's `<provider>/` prefixes to lm15 doors.
var LitellmProviderPrefixes = map[string]string{
	"openai": "openai-chat", "anthropic": "anthropic", "gemini": "gemini", "groq": "groq", "openrouter": "openrouter",
	"deepseek": "deepseek", "xai": "xai", "ollama": "ollama", "ollama_chat": "ollama", "hosted_vllm": "vllm",
	"moonshot": "moonshotai", "azure": "azure-chat",
}

var clientKeywords = map[string]string{
	"api_key":  "LMRouter(RouterConfig(api_keys={provider: key})) or the environment",
	"api_base": "LMRouter(RouterConfig(base_urls={provider: url}))", "base_url": "LMRouter(RouterConfig(base_urls={provider: url}))",
	"timeout": "RouterConfig(Timeouts: Timeouts{Read: ...})", "num_retries": "your own retry loop over lm15.RETRYABLE_ERRORS (lm15 never retries)",
	"max_retries": "your own retry loop over lm15.RETRYABLE_ERRORS (lm15 never retries)", "headers": "RouterConfig(transport=...)",
	"extra_headers": "RouterConfig(transport=...)", "extra_body": "config.extensions on the Request (build it with request_from_openai_chat and edit)",
	"extra_query": "RouterConfig(transport=...)", "cache": "your own cache keyed on the Request (lm15 has no response cache)",
	"caching": "your own cache keyed on the Request (lm15 has no response cache)", "mock_response": "lm15.testing.FakeLM",
	"drop_params":         "RouterConfig(Adaptations: \"silent\"): lm15 adapts what a wire cannot carry and records it on the response (MAP-13); 'silent' keeps no record, 'refuse' raises instead",
	"custom_llm_provider": "the model string's prefix",
}

// OpenAIChatModelString reads a model string written for the OpenAI SDK or
// litellm into lm15's form.
func OpenAIChatModelString(model string) (string, error) {
	return OpenAIChatModelStringWith(model, nil)
}

// OpenAIChatModelStringWith is OpenAIChatModelString that also reads a
// declared provider's id and aliases (RouterConfig.Providers) as a prefix.
func OpenAIChatModelStringWith(model string, providers []ProviderDefinition) (string, error) {
	if strings.Contains(model, ":") {
		return model, nil
	}
	if head, rest, ok := strings.Cut(model, "/"); ok && rest != "" {
		provider, known := LitellmProviderPrefixes[head]
		if !known {
			canonical := CanonicalProvider(head)
			for _, d := range providers {
				if inVocab(canonical, d.Spellings()) {
					provider, known = d.ID, true
					break
				}
			}
		}
		if !known {
			keys := make([]string, 0, len(LitellmProviderPrefixes))
			for k := range LitellmProviderPrefixes {
				keys = append(keys, k)
			}
			for _, d := range providers {
				keys = append(keys, d.Spellings()...)
			}
			sort.Strings(keys)
			return "", unknownModel(model, fmt.Sprintf("could not read %q as a litellm model string: %q is not a provider prefix lm15 has a door for (known: %s); write it as lm15's provider:model instead", model, head, strings.Join(keys, ", ")))
		}
		return provider + ":" + rest, nil
	}
	return model, nil
}

// ResolveOpenAIChat is Resolve for the OpenAI-shaped door: a bare OpenAI
// name takes the Chat Completions door.
func (r *LMRouter) ResolveOpenAIChat(model string) (Resolution, error) {
	lmModel, err := OpenAIChatModelStringWith(model, r.Config.Providers)
	if err != nil {
		return Resolution{}, err
	}
	res, err := r.Resolve(lmModel)
	if err != nil {
		return Resolution{}, err
	}
	if res.Source == "rule" && res.Provider == "openai" {
		return r.Resolve("openai-chat:" + res.Model)
	}
	return res, nil
}

// RequestFromOpenAIChat reads the OpenAI SDK / litellm call (model,
// messages, kwargs) into the Request it builds and the LM it routes to.
func (r *LMRouter) RequestFromOpenAIChat(model string, messages []any, kwargs JSONObject) (*Request, LM, error) {
	res, err := r.ResolveOpenAIChat(model)
	if err != nil {
		return nil, nil, err
	}
	for _, k := range sortedKeys(kwargs) {
		if where, refused := clientKeywords[k]; refused {
			return nil, nil, NotConfiguredErrorf("", nil, "", "%q configures the client, not the request; in lm15 it lives in %s", k, where)
		}
	}
	body := JSONObject{"model": res.Requested, "messages": messages}
	for k, v := range kwargs {
		body[k] = v
	}
	lm, err := r.LM(res.Requested)
	if err != nil {
		return nil, nil, err
	}
	var req *Request
	if lmDef, ok := r.Config.lookup(res.Provider); ok && lmDef.Dialect == DialectOpenAIChat {
		req, err = lm.RequestFromOpenAIChat(body)
	} else {
		req, err = RequestFromOpenAIChat(body, "")
	}
	if err != nil {
		return nil, nil, err
	}
	return routedRequest(req, res), lm, nil
}

// CompleteFromOpenAIChat answers the OpenAI SDK's / litellm's call with a
// canonical Response.
func (r *LMRouter) CompleteFromOpenAIChat(ctx context.Context, model string, messages []any, kwargs JSONObject) (*Response, error) {
	req, lm, err := r.RequestFromOpenAIChat(model, messages, kwargs)
	if err != nil {
		return nil, err
	}
	return lm.Complete(ctx, req)
}

// StreamFromOpenAIChat is the streaming twin: typed lm15 events.
func (r *LMRouter) StreamFromOpenAIChat(ctx context.Context, model string, messages []any, kwargs JSONObject) iter.Seq2[StreamEvent, error] {
	req, lm, err := r.RequestFromOpenAIChat(model, messages, kwargs)
	if err != nil {
		return errSeq(err)
	}
	return lm.Stream(ctx, req)
}

// ResponseStreamFromOpenAIChat is the lazy ResponseStream form.
func (r *LMRouter) ResponseStreamFromOpenAIChat(ctx context.Context, model string, messages []any, kwargs JSONObject) (*ResponseStream, error) {
	req, lm, err := r.RequestFromOpenAIChat(model, messages, kwargs)
	if err != nil {
		return nil, err
	}
	return NewResponseStream(lm.Stream(ctx, req), req), nil
}
