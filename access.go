package lm15

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
)

// The access policies (spec/auth.md AUTH-10; reference lm15/access.py),
// generated from lm15-contract tables/providers.json into
// tables_generated.go. The named variables are the package's public names;
// a provider added to the table needs none (the registry iterates the rows).

// tableAccess is the generated table's access policy for provider, registry
// and managed-login declared rows alike; a name the table lacks is a
// build-time bug, so it panics at package initialization.
func tableAccess(provider string) AccessPolicy {
	for _, rows := range [][]tableRow{tableProviders, tableDeclaredLogin} {
		for _, r := range rows {
			if r.ID == provider {
				return r.Access
			}
		}
	}
	panic("lm15: no access policy for " + provider + " in tables_generated.go")
}

// AnthropicAPI is the Anthropic Messages API on an API key.
var AnthropicAPI = tableAccess("anthropic")

// Claude Code constants. DefaultClaudeCodeVersion is the Claude Code release
// the claude-code door says it is (user-agent: claude-cli/<version>).
// Anthropic's server reads it: a model can require a newer release
// (claude-opus-5-5 refuses anything before 2.1.280, live 2026-09-23 and
// 2026-09-30). The latest release when last receipted (lm15-contract
// changes/2026-09-30-claude-code-client-version.md); callers move it without
// a release through the client_version setting or LM15_CLAUDE_CODE_VERSION.
const (
	DefaultClaudeCodeVersion      = "2.1.285"
	ClaudeCodeVersionEnv          = "LM15_CLAUDE_CODE_VERSION"
	CodexClientVersionEnv         = "LM15_CODEX_CLIENT_VERSION"
	DefaultClaudeCodeSystemPrompt = "You are Claude Code, Anthropic's official CLI for Claude."
	ClaudeCodeLoginHint           = "Log in again: run `claude` and use /login (Claude subscription auth)"
	OpenAICodexLoginHint          = "Log in again: run `codex login` (ChatGPT subscription auth)"
	XaiLoginHint                  = "Log in again: run lm15.auth.login_xai() (SuperGrok / X Premium subscription auth)"
)

// ClaudeCode is the Anthropic dialect on a local Claude Code login.
var ClaudeCode = tableAccess("claude-code")

// OpenAIAPI is the OpenAI Responses API on an API key.
var OpenAIAPI = tableAccess("openai")

// Codex constants.
const (
	DefaultCodexBaseURL       = "https://chatgpt.com/backend-api/codex"
	DefaultCodexOriginator    = "lm15"
	DefaultCodexInstructions  = "You are a helpful assistant."
	DefaultCodexClientVersion = "0.147.0"
	CodexBackend              = "chatgpt-codex"
)

// OpenAICodex is the Responses dialect on a local Codex CLI login.
var OpenAICodex = tableAccess("openai-codex")

// OpenAIChatAPI is the OpenAI Chat Completions dialect on an API key.
var OpenAIChatAPI = tableAccess("openai-chat")

// DefaultXaiBaseURL is api.x.ai.
const DefaultXaiBaseURL = "https://api.x.ai/v1"

// Xai is xAI Grok: Chat Completions dialect with subscription OAuth fallback.
var Xai = tableAccess("xai")

// GeminiAPI is the Google Gemini API on an API key.
var GeminiAPI = tableAccess("gemini")

// Meta is the Meta Model API over the Responses wire.
var Meta = tableAccess("meta")

// ─── Open-model inference hosts (changes/2026-09-26-inference-hosts-live.md) ───
// A bearer key each, the provider's own documented variable; batch, files
// and media endpoints they also sell are not registered.

// DeepInfra is DeepInfra open-model inference (Chat Completions dialect).
var DeepInfra = tableAccess("deepinfra")

// Together is Together AI open-model inference (Chat Completions dialect).
var Together = tableAccess("together")

// Fireworks is Fireworks AI open-model inference (Chat Completions dialect).
var Fireworks = tableAccess("fireworks")

// Parasail is Parasail open-model inference (Chat Completions dialect).
var Parasail = tableAccess("parasail")

// Groq is Groq Cloud (Chat Completions dialect).
var Groq = tableAccess("groq")

// OpenRouter is OpenRouter (Chat Completions dialect).
var OpenRouter = tableAccess("openrouter")

// DeepSeek is DeepSeek (Chat Completions dialect).
var DeepSeek = tableAccess("deepseek")

// Zai is Z.AI (Chat Completions dialect).
var Zai = tableAccess("zai")

// Moonshotai is Moonshot AI Kimi (Chat Completions dialect).
var Moonshotai = tableAccess("moonshotai")

// MoonshotaiResponses is Moonshot's Responses wire.
var MoonshotaiResponses = tableAccess("moonshotai-responses")

// MetaChat is Meta's Chat Completions wire.
var MetaChat = tableAccess("meta-chat")

// DeepSeekAnthropic is DeepSeek over the Anthropic Messages wire.
var DeepSeekAnthropic = tableAccess("deepseek-anthropic")

// MetaAnthropic is Meta over the Anthropic Messages wire.
var MetaAnthropic = tableAccess("meta-anthropic")

// MoonshotaiAnthropic is Moonshot over the Anthropic Messages wire.
var MoonshotaiAnthropic = tableAccess("moonshotai-anthropic")

// AwsAnthropic is Claude Platform on AWS.
var AwsAnthropic = tableAccess("aws-anthropic")

// BedrockAnthropic is Claude in Amazon Bedrock (mantle).
var BedrockAnthropic = tableAccess("bedrock-anthropic")

// BedrockChat is Bedrock's Chat Completions door on bedrock-runtime.
var BedrockChat = tableAccess("bedrock-chat")

// BedrockMantleChat is Bedrock Chat Completions on bedrock-mantle.
var BedrockMantleChat = tableAccess("bedrock-mantle-chat")

// Azure is Azure OpenAI v1 (Responses wire).
var Azure = tableAccess("azure")

// AzureChat is Azure OpenAI v1 (Chat Completions wire).
var AzureChat = tableAccess("azure-chat")

// AzureAnthropic is Claude in Microsoft Foundry.
var AzureAnthropic = tableAccess("azure-anthropic")

// Vertex is Gemini on Google Cloud. API keys (amended 2026-09-26): a Vertex
// API key in x-goog-api-key on the project-scoped hosts; key first, a
// token-shaped string still bearer (AuthHeaderFor). No env key:
// GOOGLE_API_KEY belongs to the Gemini API and vertex-express, and reading
// it here would silently replace the ADC identity.
var Vertex = tableAccess("vertex")

// VertexExpress is Vertex express mode (API key).
var VertexExpress = tableAccess("vertex-express")

// VertexAnthropic is Claude on Google Cloud (rawPredict).
var VertexAnthropic = tableAccess("vertex-anthropic")

// Keyless local servers.
var (
	Ollama = tableAccess("ollama")
	VLLM   = tableAccess("vllm")
	SGLang = tableAccess("sglang")
)

// ─── Scheme selection (AUTH-2) ───────────────────────────────────────

var acceptedSchemes = map[string][]string{
	"api_key":      {"bearer", "x-api-key", "api-key", "query-key"},
	"bearer_token": {"bearer", "x-api-key"},
	"aws":          {"sigv4"},
}

// SelectScheme picks the scheme this credential kind travels under.
func SelectScheme(policy AccessPolicy, credential Credential) (string, error) {
	accepted := acceptedSchemes[credential.Kind()]
	schemes := policy.EffectiveAuthScheme()
	if credential.Kind() == "bearer_token" {
		for _, s := range accepted {
			if inVocab(s, schemes) {
				return s, nil
			}
		}
	} else {
		for _, s := range schemes {
			if inVocab(s, accepted) {
				return s, nil
			}
		}
	}
	return "", NotConfiguredErrorf(policy.Provider, policy.EnvKeys, "",
		"%s: a %s credential cannot travel under %s; it accepts %s",
		policy.Provider, credential.Kind(), strings.Join(schemes, "/"), strings.Join(accepted, "/"))
}

// looksLikeAccessToken is the token shape a plain string has, if any
// (AUTH-2, amended 2026-09-19 and 2026-09-26): "JWT" (JWS compact) or
// "Google access token" (ya29., what every Google token endpoint issues).
// No key any door issues has either shape.
func looksLikeAccessToken(text string) string {
	if strings.HasPrefix(text, "ya29.") {
		return "Google access token"
	}
	if looksLikeJWT(text) {
		return "JWT"
	}
	return ""
}

func looksLikeJWT(text string) bool {
	parts := strings.Split(text, ".")
	if len(parts) != 3 {
		return false
	}
	for _, p := range parts {
		if p == "" {
			return false
		}
	}
	head := parts[0]
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(head, "="))
	if err != nil {
		return false
	}
	var header JSONObject
	if err := json.Unmarshal(decoded, &header); err != nil {
		return false
	}
	_, ok := header.Lookup("alg")
	return ok
}

// AuthHeaderFor returns the (name, value) header carrying the credential, or
// ok=false when the scheme is not a header (sigv4 signs; query-key rides
// the query).
func AuthHeaderFor(policy AccessPolicy, credential Credential, apiKeyHeader string) (name, value string, ok bool, err error) {
	if apiKeyHeader == "" {
		apiKeyHeader = "x-api-key"
	}
	scheme, err := SelectScheme(policy, credential)
	if err != nil {
		return "", "", false, err
	}
	var raw string
	switch c := credential.(type) {
	case APIKey:
		raw = c.Value
	case BearerToken:
		raw = c.Value
	default:
		return "", "", false, nil
	}
	if (scheme == "api-key" || scheme == "x-api-key") && inVocab("bearer", policy.EffectiveAuthScheme()) && looksLikeAccessToken(raw) != "" {
		// AUTH-2, amended 2026-09-19: a JWS compact JWT is never an API key
		// on any door lm15 has; it is an Entra/OAuth access token a
		// token-provider callable handed over as a string. Sent as a key it
		// is a bare 401 (live 2026-09-04); it travels as bearer — the only
		// reading under which the request can succeed. The BearerToken wrap
		// stays accepted and is the form when nothing should be read from a
		// token's shape.
		if _, isKey := credential.(APIKey); isKey {
			scheme = "bearer"
		}
	}
	switch scheme {
	case "bearer":
		return "Authorization", "Bearer " + raw, true, nil
	case "x-api-key":
		return apiKeyHeader, raw, true, nil
	case "api-key":
		return "api-key", raw, true, nil
	}
	return "", "", false, nil
}

// ─── Credential loading, keyed by provider ───────────────────────────

// LoadedCredential is the credential an adapter will send, and where it came from.
type LoadedCredential struct {
	Provider  CredentialProvider
	AccountID string
	Source    string // "explicit", "stored" or "named"
	// Origin is the AUTH-1 provenance label: honest about what lm15 can
	// see (a caller's provider is not introspected). Never the value.
	Origin string
}

// originLabel is the provenance label for a credential the adapter was
// handed (AUTH-1 provenance).
func originLabel(provider CredentialProvider, source string) string {
	if provider == nil {
		return "no credential"
	}
	if source == "stored" {
		return "the stored local login (credentials file)"
	}
	if _, static := provider.(StaticCredential); static {
		return "an explicit api_key (value never shown)"
	}
	return "an application-supplied callable (identity not inspected by lm15)"
}

// LoadCredential resolves the credential an adapter sends under policy: an
// explicit credential wins (AUTH-1); a stored-login policy loads its file;
// a key policy with nothing is a typed not-configured error.
func LoadCredential(policy AccessPolicy, explicit CredentialProvider, credentialsPath string) (LoadedCredential, error) {
	return LoadCredentialNamed(policy, explicit, credentialsPath, "")
}

// LoadCredentialNamed is LoadCredential with a named credential (AUTH-1,
// 2026-09-19): one identity on a cloud door, read from this process's
// environment, never the chain. It cannot be combined with an explicit
// credential: two answers to "who am I" is a configuration error.
func LoadCredentialNamed(policy AccessPolicy, explicit CredentialProvider, credentialsPath string, named string) (LoadedCredential, error) {
	if named != "" {
		if !policy.CloudChain() {
			return LoadedCredential{}, NotConfiguredErrorf(policy.Provider, nil, "", "%s: credential=%q names a cloud identity, and this door is not a cloud door; pass api_key= instead", policy.Provider, named)
		}
		if explicit != nil {
			return LoadedCredential{}, NotConfiguredErrorf(policy.Provider, nil, "", "%s: both api_key= and credential=%q were given; a door has one identity — pass the credential value, or name the identity, not both", policy.Provider, named)
		}
		ctx := OnlineChainContext(nil)
		provider, err := NamedCredentialProviderFor(policy, ctx, named)
		if err != nil {
			return LoadedCredential{}, err
		}
		return LoadedCredential{Provider: provider, Source: "named"}, nil
	}
	loaded, err := loadCredential(policy, explicit, credentialsPath)
	if err != nil {
		return loaded, err
	}
	loaded.Origin = originLabel(loaded.Provider, loaded.Source)
	return loaded, nil
}

func loadCredential(policy AccessPolicy, explicit CredentialProvider, credentialsPath string) (LoadedCredential, error) {
	if explicit != nil {
		return LoadedCredential{Provider: explicit, Source: "explicit"}, nil
	}
	if policy.EffectiveCredentialPolicy() != "key" {
		switch policy.Provider {
		case "claude-code":
			if _, err := GetClaudeCodeAccessToken(context.Background(), credentialsPath, true); err != nil {
				return LoadedCredential{}, err
			}
			return LoadedCredential{Provider: CredentialFunc(func(ctx context.Context) (Credential, error) {
				token, err := GetClaudeCodeAccessToken(ctx, credentialsPath, true)
				if err != nil {
					return nil, err
				}
				return APIKey{Value: token}, nil
			}), Source: "stored"}, nil
		case "openai-codex":
			initial, err := GetCodexCLIAccessToken(context.Background(), credentialsPath, true)
			if err != nil {
				return LoadedCredential{}, err
			}
			account := initial.AccountID
			if account == "" {
				account = ExtractChatGPTAccountID(initial.AccessToken)
			}
			return LoadedCredential{Provider: CredentialFunc(func(ctx context.Context) (Credential, error) {
				cred, err := GetCodexCLIAccessToken(ctx, credentialsPath, true)
				if err != nil {
					return nil, err
				}
				return APIKey{Value: cred.AccessToken}, nil
			}), AccountID: account, Source: "stored"}, nil
		case "xai":
			if _, err := GetXaiAccessToken(context.Background(), credentialsPath, true); err != nil {
				return LoadedCredential{}, err
			}
			return LoadedCredential{Provider: CredentialFunc(func(ctx context.Context) (Credential, error) {
				token, err := GetXaiAccessToken(ctx, credentialsPath, true)
				if err != nil {
					return nil, err
				}
				return APIKey{Value: token}, nil
			}), Source: "stored"}, nil
		}
	}
	hint := "pass api_key="
	if len(policy.EnvKeys) > 0 {
		hint = "set " + strings.Join(policy.EnvKeys, " or ") + " or pass api_key="
	}
	return LoadedCredential{}, NotConfiguredErrorf(policy.Provider, policy.EnvKeys, policy.LoginHint, "%s: no credential given; %s", policy.Provider, hint)
}

// HasStoredCredential is the offline probe (files, never the network) for
// the router's oauth-unless-explicit chain.
func HasStoredCredential(policy AccessPolicy) bool {
	return StoredCredentialState(policy) == "usable"
}

// StoredCredentialState is "usable", "unusable", "logged_out" or "absent"
// for policy's stored login (AUTH-1 oauth-unless-explicit, R3). Files only.
func StoredCredentialState(policy AccessPolicy) string {
	if policy.Provider == "xai" {
		return XaiStoredState("")
	}
	return "absent"
}
