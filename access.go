package lm15

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
)

// The access policies (spec/auth.md AUTH-10; reference lm15/access.py).
// Ports copy the table as data and consult it at the same named points.

// AnthropicAPI is the Anthropic Messages API on an API key.
var AnthropicAPI = AccessPolicy{
	Provider:   "anthropic",
	Supports:   EndpointSupport{Complete: true, Stream: true, Files: true, Batches: true, Models: true},
	AuthModes:  []string{"x-api-key"},
	EnvKeys:    []string{"ANTHROPIC_API_KEY"},
	AuthScheme: []string{"x-api-key"},
}

// Claude Code constants.
const (
	DefaultClaudeCodeVersion      = "2.1.170"
	DefaultClaudeCodeSystemPrompt = "You are Claude Code, Anthropic's official CLI for Claude."
	ClaudeCodeLoginHint           = "Log in again: run `claude` and use /login (Claude subscription auth)"
	OpenAICodexLoginHint          = "Log in again: run `codex login` (ChatGPT subscription auth)"
	XaiLoginHint                  = "Log in again: run lm15.auth.login_xai() (SuperGrok / X Premium subscription auth)"
)

// ClaudeCode is the Anthropic dialect on a local Claude Code login.
var ClaudeCode = AccessPolicy{
	Provider:         "claude-code",
	Supports:         EndpointSupport{Complete: true, Stream: true, Models: true},
	CredentialPolicy: "oauth",
	AuthModes:        []string{"claude-code-oauth", "bearer-oauth"},
	AuthScheme:       []string{"bearer"},
	Headers: [][2]string{
		{"anthropic-dangerous-direct-browser-access", "true"},
		{"anthropic-beta", "claude-code-20250219,oauth-2025-04-20"},
		{"x-app", "cli"},
		{"user-agent", "claude-cli/" + DefaultClaudeCodeVersion},
	},
	LoginHint:    ClaudeCodeLoginHint,
	Backend:      "claude-code",
	SystemPrefix: DefaultClaudeCodeSystemPrompt,
}

// OpenAIAPI is the OpenAI Responses API on an API key.
var OpenAIAPI = AccessPolicy{
	Provider: "openai",
	Supports: EndpointSupport{Complete: true, Stream: true, Live: true, Files: true, Batches: true,
		Images: true, Speech: true, Video: true, ResponsesAPI: true, Models: true},
	AuthModes:          []string{"bearer"},
	EnvKeys:            []string{"OPENAI_API_KEY"},
	EnterpriseVariants: []string{"azure-openai"},
}

// Codex constants.
const (
	DefaultCodexBaseURL       = "https://chatgpt.com/backend-api/codex"
	DefaultCodexOriginator    = "lm15"
	DefaultCodexInstructions  = "You are a helpful assistant."
	DefaultCodexClientVersion = "0.147.0"
	CodexBackend              = "chatgpt-codex"
)

// OpenAICodex is the Responses dialect on a local Codex CLI login.
var OpenAICodex = AccessPolicy{
	Provider:         "openai-codex",
	Supports:         EndpointSupport{Complete: true, Stream: true, Models: true},
	CredentialPolicy: "oauth",
	AuthModes:        []string{"chatgpt-oauth", "bearer-oauth"},
	Headers: [][2]string{
		{"OpenAI-Beta", "responses=experimental"},
		{"originator", DefaultCodexOriginator},
	},
	LoginHint:      OpenAICodexLoginHint,
	Backend:        CodexBackend,
	BackendOptions: map[string]string{"client_version": DefaultCodexClientVersion},
	SystemPrefix:   DefaultCodexInstructions,
	BaseURL:        DefaultCodexBaseURL,
}

// OpenAIChatAPI is the OpenAI Chat Completions dialect on an API key.
var OpenAIChatAPI = AccessPolicy{
	Provider:  "openai-chat",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"OPENAI_API_KEY"},
}

// DefaultXaiBaseURL is api.x.ai.
const DefaultXaiBaseURL = "https://api.x.ai/v1"

// Xai is xAI Grok: Chat Completions dialect with subscription OAuth fallback.
var Xai = AccessPolicy{
	Provider:         "xai",
	Supports:         EndpointSupport{Complete: true, Stream: true, Models: true, Images: true, Video: true},
	CredentialPolicy: "oauth-unless-explicit",
	AuthModes:        []string{"bearer", "xai-oauth"},
	EnvKeys:          []string{"XAI_API_KEY"},
	LoginHint:        XaiLoginHint,
	BaseURL:          DefaultXaiBaseURL,
}

// GeminiAPI is the Google Gemini API on an API key.
var GeminiAPI = AccessPolicy{
	Provider: "gemini",
	Supports: EndpointSupport{Complete: true, Stream: true, Live: true, Files: true, Batches: true,
		Images: true, Speech: true, Video: true, Models: true, Caches: true},
	AuthModes:  []string{"query-api-key", "x-goog-api-key"},
	EnvKeys:    []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"},
	AuthScheme: []string{"x-api-key"}, // the dialect renders it as x-goog-api-key
}

var metaEnvKeys = []string{"META_API_KEY"}
var moonshotEnvKeys = []string{"MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY"}

// Meta is the Meta Model API over the Responses wire.
var Meta = AccessPolicy{
	Provider:  "meta",
	Supports:  EndpointSupport{Complete: true, Stream: true, Files: true, Images: true, ResponsesAPI: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   metaEnvKeys,
	BaseURL:   OpenAIResponsesPresetBaseURLs["meta"],
}

// ─── Open-model inference hosts (changes/2026-09-26-inference-hosts-live.md) ───
// A bearer key each, the provider's own documented variable; batch, files
// and media endpoints they also sell are not registered.

// DeepInfra is DeepInfra open-model inference (Chat Completions dialect).
var DeepInfra = AccessPolicy{
	Provider:  "deepinfra",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"DEEPINFRA_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["deepinfra"],
}

// Together is Together AI open-model inference (Chat Completions dialect).
var Together = AccessPolicy{
	Provider:  "together",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"TOGETHER_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["together"],
}

// Fireworks is Fireworks AI open-model inference (Chat Completions dialect).
var Fireworks = AccessPolicy{
	Provider:  "fireworks",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"FIREWORKS_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["fireworks"],
}

// Parasail is Parasail open-model inference (Chat Completions dialect).
var Parasail = AccessPolicy{
	Provider:  "parasail",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"PARASAIL_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["parasail"],
}

// Groq is Groq Cloud (Chat Completions dialect).
var Groq = AccessPolicy{
	Provider:  "groq",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"GROQ_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["groq"],
}

// OpenRouter is OpenRouter (Chat Completions dialect).
var OpenRouter = AccessPolicy{
	Provider:  "openrouter",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"OPENROUTER_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["openrouter"],
}

// DeepSeek is DeepSeek (Chat Completions dialect).
var DeepSeek = AccessPolicy{
	Provider:  "deepseek",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"DEEPSEEK_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["deepseek"],
}

// Zai is Z.AI (Chat Completions dialect).
var Zai = AccessPolicy{
	Provider:  "zai",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   []string{"ZAI_API_KEY"},
	BaseURL:   OpenAIChatPresetBaseURLs["zai"],
}

// Moonshotai is Moonshot AI Kimi (Chat Completions dialect).
var Moonshotai = AccessPolicy{
	Provider:  "moonshotai",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   moonshotEnvKeys,
	BaseURL:   OpenAIChatPresetBaseURLs["moonshotai"],
}

// MoonshotaiResponses is Moonshot's Responses wire.
var MoonshotaiResponses = AccessPolicy{
	Provider:  "moonshotai-responses",
	Supports:  EndpointSupport{Complete: true, Stream: true, ResponsesAPI: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   moonshotEnvKeys,
	BaseURL:   OpenAIResponsesPresetBaseURLs["moonshotai"],
}

// MetaChat is Meta's Chat Completions wire.
var MetaChat = AccessPolicy{
	Provider:  "meta-chat",
	Supports:  EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes: []string{"bearer"},
	EnvKeys:   metaEnvKeys,
	BaseURL:   OpenAIChatPresetBaseURLs["meta"],
}

// DeepSeekAnthropic is DeepSeek over the Anthropic Messages wire.
var DeepSeekAnthropic = AccessPolicy{
	Provider:   "deepseek-anthropic",
	Supports:   EndpointSupport{Complete: true, Stream: true},
	AuthModes:  []string{"x-api-key"},
	EnvKeys:    []string{"DEEPSEEK_API_KEY"},
	AuthScheme: []string{"x-api-key"},
	BaseURL:    AnthropicPresetBaseURLs["deepseek"],
}

// MetaAnthropic is Meta over the Anthropic Messages wire.
var MetaAnthropic = AccessPolicy{
	Provider:   "meta-anthropic",
	Supports:   EndpointSupport{Complete: true, Stream: true, Models: true},
	AuthModes:  []string{"bearer"},
	EnvKeys:    metaEnvKeys,
	AuthScheme: []string{"bearer"},
	BaseURL:    AnthropicPresetBaseURLs["meta"],
}

// MoonshotaiAnthropic is Moonshot over the Anthropic Messages wire.
var MoonshotaiAnthropic = AccessPolicy{
	Provider:   "moonshotai-anthropic",
	Supports:   EndpointSupport{Complete: true, Stream: true},
	AuthModes:  []string{"bearer"},
	EnvKeys:    moonshotEnvKeys,
	AuthScheme: []string{"bearer"},
	BaseURL:    AnthropicPresetBaseURLs["moonshotai"],
}

// Cloud host settings (AUTH-10).
var (
	awsRegionSetting      = HostSetting{Name: "region", Env: []string{"AWS_REGION", "AWS_DEFAULT_REGION"}}
	awsWorkspace          = HostSetting{Name: "workspace", Env: []string{"ANTHROPIC_AWS_WORKSPACE_ID"}}
	gcpProject            = HostSetting{Name: "project", Env: []string{"GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"}}
	gcpLocation           = HostSetting{Name: "location", Env: []string{"GOOGLE_CLOUD_LOCATION"}, Default: "global"}
	azureOpenAIRes        = HostSetting{Name: "resource", Env: []string{"AZURE_OPENAI_RESOURCE"}}
	azureFoundryRes       = HostSetting{Name: "resource", Env: []string{"ANTHROPIC_FOUNDRY_RESOURCE"}}
	azureAuthoritySetting = HostSetting{Name: "authority_host", Env: []string{"AZURE_AUTHORITY_HOST"}, Default: "https://login.microsoftonline.com"}
	azureScopeSetting     = HostSetting{Name: "scope", Default: "https://ai.azure.com/.default"}
	vertexBase            = "https://{location_host}/v1/projects/{project}/locations/{location}"

	// Endpoint overrides (AUTH-10, amended 2026-09-19): the vendor's own
	// variables naming a full URL root for a door, consulted by the router
	// after an explicit base_urls entry and before the {resource} /
	// {region} template. AWS: AWS_ENDPOINT_URL_<SERVICE_ID> then the
	// generic AWS_ENDPOINT_URL (the service id is the SigV4 service name
	// upper-cased with - → _, the SDK's rule). Azure OpenAI:
	// AZURE_OPENAI_ENDPOINT. Foundry Claude: ANTHROPIC_FOUNDRY_BASE_URL.
	// Vertex: no vendor variable is cited; base_urls only.
	awsEndpoint          = []string{"AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_ENDPOINT_URL"}
	awsMantleEndpoint    = []string{"AWS_ENDPOINT_URL_BEDROCK_MANTLE", "AWS_ENDPOINT_URL"}
	awsAnthropicEndpoint = []string{"AWS_ENDPOINT_URL_AWS_EXTERNAL_ANTHROPIC", "AWS_ENDPOINT_URL"}
	azureOpenAIEndpoint  = []string{"AZURE_OPENAI_ENDPOINT"}
	azureFoundryEndpoint = []string{"ANTHROPIC_FOUNDRY_BASE_URL"}
)

// AwsAnthropic is Claude Platform on AWS.
var AwsAnthropic = AccessPolicy{
	Provider:         "aws-anthropic",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "aws-chain",
	AuthModes:        []string{"sigv4", "x-api-key"},
	EnvKeys:          []string{"ANTHROPIC_AWS_API_KEY"},
	AuthScheme:       []string{"sigv4", "x-api-key"},
	Backend:          "aws-external-anthropic",
	Host: &HostSpec{
		BaseURL:         "https://aws-external-anthropic.{region}.api.aws/v1",
		Settings:        []HostSetting{awsRegionSetting, awsWorkspace},
		RequiredHeaders: [][2]string{{"anthropic-workspace-id", "workspace"}},
		SigV4Service:    "aws-external-anthropic",
		EndpointEnv:     awsAnthropicEndpoint,
	},
}

// BedrockAnthropic is Claude in Amazon Bedrock (mantle).
var BedrockAnthropic = AccessPolicy{
	Provider:         "bedrock-anthropic",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "aws-chain",
	AuthModes:        []string{"sigv4", "x-api-key"},
	EnvKeys:          []string{"AWS_BEARER_TOKEN_BEDROCK"},
	AuthScheme:       []string{"sigv4", "x-api-key"},
	Backend:          "bedrock-mantle",
	Host: &HostSpec{
		BaseURL:      "https://bedrock-mantle.{region}.api.aws/anthropic/v1",
		Settings:     []HostSetting{awsRegionSetting},
		SigV4Service: "bedrock-mantle",
		EndpointEnv:  awsMantleEndpoint,
	},
}

// BedrockChat is Bedrock's Chat Completions door on bedrock-runtime.
var BedrockChat = AccessPolicy{
	Provider:         "bedrock-chat",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "aws-chain",
	AuthModes:        []string{"sigv4", "bearer"},
	EnvKeys:          []string{"AWS_BEARER_TOKEN_BEDROCK"},
	AuthScheme:       []string{"sigv4", "bearer"},
	Backend:          "bedrock-runtime",
	Host: &HostSpec{
		BaseURL:      "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
		Settings:     []HostSetting{awsRegionSetting},
		SigV4Service: "bedrock",
		EndpointEnv:  awsEndpoint,
	},
}

// BedrockMantleChat is Bedrock Chat Completions on bedrock-mantle.
var BedrockMantleChat = AccessPolicy{
	Provider:         "bedrock-mantle-chat",
	Supports:         EndpointSupport{Complete: true, Stream: true, Models: true},
	CredentialPolicy: "aws-chain",
	AuthModes:        []string{"sigv4", "bearer"},
	EnvKeys:          []string{"AWS_BEARER_TOKEN_BEDROCK"},
	AuthScheme:       []string{"sigv4", "bearer"},
	Backend:          "bedrock-mantle",
	Host: &HostSpec{
		BaseURL:      "https://bedrock-mantle.{region}.api.aws/v1",
		Settings:     []HostSetting{awsRegionSetting},
		SigV4Service: "bedrock-mantle",
		EndpointEnv:  awsMantleEndpoint,
	},
}

// Azure is Azure OpenAI v1 (Responses wire).
var Azure = AccessPolicy{
	Provider: "azure",
	Supports: EndpointSupport{Complete: true, Stream: true, Live: true, Files: true, Batches: true, Speech: true,
		ResponsesAPI: true, Models: true},
	CredentialPolicy: "azure-chain",
	AuthModes:        []string{"api-key", "entra-oauth"},
	EnvKeys:          []string{"AZURE_OPENAI_API_KEY"},
	AuthScheme:       []string{"api-key", "bearer"},
	Backend:          "azure-openai",
	Host: &HostSpec{
		BaseURL:     "https://{resource}.openai.azure.com/openai/v1",
		Settings:    []HostSetting{azureOpenAIRes, azureAuthoritySetting, azureScopeSetting},
		EndpointEnv: azureOpenAIEndpoint,
	},
}

// AzureChat is Azure OpenAI v1 (Chat Completions wire).
var AzureChat = AccessPolicy{
	Provider:         "azure-chat",
	Supports:         EndpointSupport{Complete: true, Stream: true, Models: true},
	CredentialPolicy: "azure-chain",
	AuthModes:        []string{"api-key", "entra-oauth"},
	EnvKeys:          []string{"AZURE_OPENAI_API_KEY"},
	AuthScheme:       []string{"api-key", "bearer"},
	Backend:          "azure-openai",
	Host: &HostSpec{
		BaseURL:     "https://{resource}.openai.azure.com/openai/v1",
		Settings:    []HostSetting{azureOpenAIRes, azureAuthoritySetting, azureScopeSetting},
		EndpointEnv: azureOpenAIEndpoint,
	},
}

// AzureAnthropic is Claude in Microsoft Foundry.
var AzureAnthropic = AccessPolicy{
	Provider:         "azure-anthropic",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "azure-chain",
	AuthModes:        []string{"x-api-key", "entra-oauth"},
	EnvKeys:          []string{"ANTHROPIC_FOUNDRY_API_KEY"},
	AuthScheme:       []string{"x-api-key", "bearer"},
	Backend:          "azure-foundry",
	Host: &HostSpec{
		BaseURL:     "https://{resource}.services.ai.azure.com/anthropic/v1",
		Settings:    []HostSetting{azureFoundryRes, azureAuthoritySetting, azureScopeSetting},
		EndpointEnv: azureFoundryEndpoint,
	},
}

// Vertex is Gemini on Google Cloud. API keys (amended 2026-09-26): a Vertex
// API key in x-goog-api-key on the project-scoped hosts; key first, a
// token-shaped string still bearer (AuthHeaderFor). No env key:
// GOOGLE_API_KEY belongs to the Gemini API and vertex-express, and reading
// it here would silently replace the ADC identity.
var Vertex = AccessPolicy{
	Provider:         "vertex",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "gcp-chain",
	AuthModes:        []string{"x-goog-api-key", "google-oauth"},
	AuthScheme:       []string{"x-api-key", "bearer"},
	Backend:          "vertex",
	Host:             &HostSpec{BaseURL: vertexBase + "/publishers/google", Settings: []HostSetting{gcpProject, gcpLocation}},
}

// VertexExpress is Vertex express mode (API key).
var VertexExpress = AccessPolicy{
	Provider:         "vertex-express",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "key",
	AuthModes:        []string{"query-api-key"},
	EnvKeys:          []string{"GOOGLE_API_KEY"},
	AuthScheme:       []string{"query-key"},
	Backend:          "vertex-express",
	Host:             &HostSpec{BaseURL: "https://aiplatform.googleapis.com/v1/publishers/google"},
}

// VertexAnthropic is Claude on Google Cloud (rawPredict).
var VertexAnthropic = AccessPolicy{
	Provider:         "vertex-anthropic",
	Supports:         EndpointSupport{Complete: true, Stream: true},
	CredentialPolicy: "gcp-chain",
	AuthModes:        []string{"google-oauth"},
	AuthScheme:       []string{"bearer"},
	Backend:          "vertex",
	Host: &HostSpec{
		BaseURL:  vertexBase,
		Settings: []HostSetting{gcpProject, gcpLocation},
		Paths: map[string]string{
			"messages":        "/publishers/anthropic/models/{model}:rawPredict",
			"messages/stream": "/publishers/anthropic/models/{model}:streamRawPredict",
		},
		ModelIn:            "path",
		AnthropicVersionIn: "body:vertex-2023-10-16",
	},
}

// Keyless local servers.
var (
	Ollama = AccessPolicy{Provider: "ollama", Supports: EndpointSupport{Complete: true, Stream: true, Models: true}, AuthModes: []string{"bearer"}, BaseURL: OpenAIChatPresetBaseURLs["ollama"]}
	VLLM   = AccessPolicy{Provider: "vllm", Supports: EndpointSupport{Complete: true, Stream: true, Models: true}, AuthModes: []string{"bearer"}, BaseURL: OpenAIChatPresetBaseURLs["vllm"]}
	SGLang = AccessPolicy{Provider: "sglang", Supports: EndpointSupport{Complete: true, Stream: true, Models: true}, AuthModes: []string{"bearer"}, BaseURL: OpenAIChatPresetBaseURLs["sglang"]}
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
