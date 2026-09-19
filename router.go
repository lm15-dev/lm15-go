package lm15

import (
	"context"
	"fmt"
	"iter"
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
	parts = append(parts, fmt.Sprintf("wire model %q", r.Model))
	def, _ := Providers[r.Provider]
	policy := providerCredentialPolicy(r.Provider)
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
type RouterConfig struct {
	Registry  *ModelRegistry
	Rules     []RouteRule       // nil = DefaultRules
	Env       map[string]string // nil = the process environment
	APIKeys   map[string]CredentialLike
	BaseURLs  map[string]string
	Settings  map[string]map[string]string
	Transport Transport
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
	}
	return dialect
}

func routable(provider string) bool {
	_, ok := Providers[provider]
	return ok
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
	known := ProviderIDs()
	check := func(field string, keys []string) error {
		seen := map[string]bool{}
		for _, key := range keys {
			provider := CanonicalProvider(key)
			if field == "api_keys" && seen[provider] {
				return NotConfiguredErrorf("", nil, "", "RouterConfig(api_keys=...): duplicate spellings for %q; use one entry", provider)
			}
			seen[provider] = true
			if routable(provider) {
				continue
			}
			hint := ""
			if close := closestMatch(provider, known, 0.6); close != "" {
				hint = fmt.Sprintf(" Did you mean %q?", close)
			}
			return NotConfiguredErrorf("", nil, "", "RouterConfig(%s=...): %q is not a provider lm15 routes to.%s router.resolve(model).provider (or resolve_openai_chat) names the one a model string uses; known: %s", field, key, hint, knownProviders())
		}
		return nil
	}
	var apiKeys, baseURLs, settings []string
	for k := range config.APIKeys {
		apiKeys = append(apiKeys, k)
	}
	for k := range config.BaseURLs {
		baseURLs = append(baseURLs, k)
	}
	for k := range config.Settings {
		settings = append(settings, k)
	}
	sort.Strings(apiKeys)
	sort.Strings(baseURLs)
	sort.Strings(settings)
	if err := check("api_keys", apiKeys); err != nil {
		return err
	}
	if err := check("base_urls", baseURLs); err != nil {
		return err
	}
	return check("settings", settings)
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
		if CanonicalProvider(k) == provider {
			candidates = append(candidates, k)
		}
	}
	if len(candidates) == 0 && routable(provider) {
		envKeys := providerEnvKeys(provider)
		if len(envKeys) > 0 {
			for _, k := range keys {
				canon := CanonicalProvider(k)
				if routable(canon) && sameStrings(providerEnvKeys(canon), envKeys) {
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
	envKeys := providerEnvKeys(provider)
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
	def := Providers[provider]
	envKey, err := envKeyFor(provider, config)
	if err != nil {
		return Resolution{}, err
	}
	res := Resolution{Requested: requested, Model: wireModel, Provider: provider, Adapter: adapterName(def.Dialect), Source: source, Rule: rule, EnvKey: envKey, ModelInfo: info}
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
		provider := CanonicalProvider(head)
		if routable(provider) && rest != "" {
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
			provider := CanonicalProvider(info.Provider)
			if !routable(provider) {
				return Resolution{}, unknownModel(model, fmt.Sprintf("model %q resolved in the catalog to provider %q, but lm15 has no adapter or compat preset for it. Known providers: %s. Construct a provider LM directly (e.g. OpenAIChatLM with a custom base_url) for OpenAI-compatible servers.", model, info.Provider, knownProviders()))
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
			provider := CanonicalProvider(rule.Provider)
			if !routable(provider) {
				return Resolution{}, unknownModel(model, fmt.Sprintf("rule %+v names provider %q, which has no adapter. Known providers: %s.", rule, rule.Provider, knownProviders()))
			}
			return resolution(model, model, provider, "rule", config, &rule, nil)
		}
	}
	var hints []string
	if head, rest, ok := strings.Cut(model, ":"); ok {
		if close := closestMatch(CanonicalProvider(head), ProviderIDs(), 0.75); close != "" {
			hints = append(hints, fmt.Sprintf("Did you mean %q?", close+":"+rest))
		}
	}
	hints = append(hints, fmt.Sprintf("Use an explicit provider prefix — \"provider:%s\" with provider one of: %s.", model, knownProviders()))
	catalog := "no catalog supplied"
	if config.Registry != nil {
		catalog = "no catalog match"
	} else {
		hints = append(hints, "Or pass a model catalog: LMRouter(config=RouterConfig(registry=ModelRegistry.discover())) — install a catalog package such as 'aimo' first.")
	}
	return Resolution{}, unknownModel(model, fmt.Sprintf("could not route model %q: no provider prefix, %s, and none of the %d built-in rules matched. %s", model, catalog, len(config.rules()), strings.Join(hints, " ")))
}

// buildLM constructs the provider LM for a resolution (AUTH-1 chain).
func buildLM(res Resolution, config RouterConfig) (LM, error) {
	def := Providers[res.Provider]
	var opts []Option
	if config.Transport != nil {
		opts = append(opts, WithTransport(config.Transport))
	}
	for key, u := range config.BaseURLs {
		if CanonicalProvider(key) == res.Provider {
			if def.Hosted() {
				return nil, NotConfiguredErrorf("", nil, "", "RouterConfig(base_urls={%q: ...}): a cloud door's URL is built from its host settings (resource, region), not given whole; set them in RouterConfig(settings={%q: {...}}) instead.", res.Provider, res.Provider)
			}
			opts = append(opts, WithBaseURL(u))
		}
	}
	policy := def.CredentialPolicy()
	if def.Bound() {
		opts = append(opts, WithAccess(def.Access))
		if def.Compat != "" {
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
	if def.Hosted() {
		ctx := OnlineChainContext(env)
		var given map[string]string
		for key, s := range config.Settings {
			if CanonicalProvider(key) == res.Provider {
				given = s
			}
		}
		settings, err := resolveSettings(def.Access.Host, given, env, res.Provider, ProfileSettings(def.Access, ctx))
		if err != nil {
			return nil, err
		}
		ctx.Settings = settings
		if apiKey == nil && def.Access.CloudChain() {
			apiKey = CredentialProviderFor(def.Access, ctx)
		} else if apiKey == nil {
			for _, k := range def.Access.EnvKeys {
				if v := env[k]; v != "" {
					apiKey = v
					break
				}
			}
		}
		if apiKey == nil {
			return nil, missingCredential(res.Provider, def.Access.EnvKeys, "credential")
		}
		return construct(def, append(opts, WithAPIKey(apiKey), WithSettings(settings)))
	}
	if apiKey == nil && policy == "oauth-unless-explicit" && HasStoredCredential(def.Access) {
		return construct(def, opts)
	}
	if apiKey == nil {
		for _, k := range def.Access.EnvKeys {
			if v := env[k]; v != "" {
				apiKey = v
				break
			}
		}
	}
	if apiKey == nil && def.PlaceholderKey != "" {
		apiKey = def.PlaceholderKey
	}
	if apiKey == nil && policy == "oauth-unless-explicit" {
		return construct(def, opts) // the constructor raises the typed login-hint error
	}
	if apiKey == nil {
		return nil, missingCredential(res.Provider, def.Access.EnvKeys, "API key")
	}
	return construct(def, append(opts, WithAPIKey(apiKey)))
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
// built lazily, reused.
type LMRouter struct {
	Config RouterConfig
	mu     sync.Mutex
	lms    map[string]LM
}

// NewRouter creates a router over the process environment.
func NewRouter() *LMRouter { return &LMRouter{lms: map[string]LM{}} }

// NewRouterWithConfig creates a router; the config's provider-keyed maps
// are checked for typos.
func NewRouterWithConfig(config RouterConfig) (*LMRouter, error) {
	if err := checkProviderKeyed(config); err != nil {
		return nil, err
	}
	return &LMRouter{Config: config, lms: map[string]LM{}}, nil
}

// Resolve is the offline lookup (also the explain method).
func (r *LMRouter) Resolve(model string) (Resolution, error) { return Resolve(model, r.Config) }

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
	lm, err := buildLM(res, r.Config)
	if err != nil {
		return nil, err
	}
	r.lms[res.Provider] = lm
	return lm, nil
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
	return lm.Cache(ctx, routedRequest(prefix, res), ttlSeconds, label)
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
	"timeout": "RouterConfig(transport=...)", "num_retries": "your own retry loop over lm15.RETRYABLE_ERRORS (lm15 never retries)",
	"max_retries": "your own retry loop over lm15.RETRYABLE_ERRORS (lm15 never retries)", "headers": "RouterConfig(transport=...)",
	"extra_headers": "RouterConfig(transport=...)", "extra_body": "config.extensions on the Request (build it with request_from_openai_chat and edit)",
	"extra_query": "RouterConfig(transport=...)", "cache": "your own cache keyed on the Request (lm15 has no response cache)",
	"caching": "your own cache keyed on the Request (lm15 has no response cache)", "mock_response": "lm15.testing.FakeLM",
	"drop_params": "nothing: lm15 refuses what it cannot carry instead of dropping it", "custom_llm_provider": "the model string's prefix",
}

// OpenAIChatModelString reads a model string written for the OpenAI SDK or
// litellm into lm15's form.
func OpenAIChatModelString(model string) (string, error) {
	if strings.Contains(model, ":") {
		return model, nil
	}
	if head, rest, ok := strings.Cut(model, "/"); ok && rest != "" {
		provider, known := LitellmProviderPrefixes[head]
		if !known {
			keys := make([]string, 0, len(LitellmProviderPrefixes))
			for k := range LitellmProviderPrefixes {
				keys = append(keys, k)
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
	lmModel, err := OpenAIChatModelString(model)
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
	if lmDef, ok := Providers[res.Provider]; ok && lmDef.Dialect == DialectOpenAIChat {
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
