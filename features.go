package lm15

import (
	"regexp"
	"strings"
)

// EndpointSupport declares the surfaces an access path carries. A dialect
// that implements a surface still refuses when the policy does not carry it.
type EndpointSupport struct {
	Complete     bool
	Stream       bool
	Live         bool
	Files        bool
	Batches      bool
	Images       bool
	Speech       bool
	Video        bool
	ResponsesAPI bool
	Models       bool
	Caches       bool
	Extra        []string
}

// SupportsEndpoint reports whether the named surface is carried.
func (s EndpointSupport) SupportsEndpoint(name string) bool {
	if inVocab(name, s.Extra) {
		return true
	}
	switch name {
	case "complete":
		return s.Complete
	case "stream":
		return s.Stream
	case "live":
		return s.Live
	case "files":
		return s.Files
	case "batches":
		return s.Batches
	case "images":
		return s.Images
	case "speech":
		return s.Speech
	case "video":
		return s.Video
	case "responses_api":
		return s.ResponsesAPI
	case "models":
		return s.Models
	case "caches":
		return s.Caches
	}
	return false
}

// endpointSupportFields lists the typed booleans in declaration order.
var endpointSupportFields = []string{"complete", "stream", "live", "files", "batches", "images", "speech", "video", "responses_api", "models", "caches"}

// HostSetting is one host setting: its name, the env variables consulted
// in order, and its default ("" = required).
type HostSetting struct {
	Name    string
	Env     []string
	Default string
}

// HostSpec is how a dialect reaches a cloud door (AUTH-10 host).
type HostSpec struct {
	BaseURL            string
	Settings           []HostSetting
	Paths              map[string]string
	ModelIn            string // "" reads as "body"
	AnthropicVersionIn string // "" reads as "header"
	StreamFraming      string // "" reads as "sse"
	RequiredHeaders    [][2]string
	SigV4Service       string
	// EndpointEnv: the cloud vendor's own variables, consulted in order by
	// the router, that name a full endpoint root for this door
	// (AZURE_OPENAI_ENDPOINT, AWS_ENDPOINT_URL_BEDROCK_RUNTIME). An endpoint
	// replaces the root of BaseURL (the part before the first path
	// segment); the door's path is appended unless already present
	// (AUTH-10, amended 2026-09-19).
	EndpointEnv []string
}

// RootTemplate is BaseURL up to (not including) the first path segment:
// the part an endpoint override replaces.
func (h HostSpec) RootTemplate() string {
	scheme, rest, _ := strings.Cut(h.BaseURL, "://")
	host, _, _ := strings.Cut(rest, "/")
	return scheme + "://" + host
}

// PathTemplate is the door's path under the root ("/openai/v1",
// "/v1/projects/{project}/locations/{location}/publishers/google"); ""
// when the template is a bare host.
func (h HostSpec) PathTemplate() string {
	_, rest, _ := strings.Cut(h.BaseURL, "://")
	_, path, slash := strings.Cut(rest, "/")
	if !slash {
		return ""
	}
	return "/" + path
}

var templateSettingRe = regexp.MustCompile(`\{(\w+)\}`)

func templateSettings(text string) map[string]bool {
	out := map[string]bool{}
	for _, m := range templateSettingRe.FindAllStringSubmatch(text, -1) {
		out[m[1]] = true
	}
	return out
}

// URLOnlySettings are the settings an endpoint override makes
// unnecessary: those that appear in the root of the template and nowhere
// else — not in the path, not in a required header, and not the SigV4
// signing region (AWS's own SDK requires a region even with endpoint_url;
// the signature's credential scope names it).
func (h HostSpec) URLOnlySettings() map[string]bool {
	inRoot := templateSettings(h.RootTemplate())
	if inRoot["location_host"] {
		delete(inRoot, "location_host")
		inRoot["location"] = true
	}
	inPath := templateSettings(h.PathTemplate())
	for _, rh := range h.RequiredHeaders {
		delete(inRoot, rh[1])
	}
	for name := range inPath {
		delete(inRoot, name)
	}
	if h.SigV4Service != "" {
		delete(inRoot, "region")
	}
	return inRoot
}

// EffectiveModelIn returns ModelIn or "body".
func (h HostSpec) EffectiveModelIn() string {
	if h.ModelIn == "" {
		return "body"
	}
	return h.ModelIn
}

// EffectiveAnthropicVersionIn returns AnthropicVersionIn or "header".
func (h HostSpec) EffectiveAnthropicVersionIn() string {
	if h.AnthropicVersionIn == "" {
		return "header"
	}
	return h.AnthropicVersionIn
}

// EffectiveStreamFraming returns StreamFraming or "sse".
func (h HostSpec) EffectiveStreamFraming() string {
	if h.StreamFraming == "" {
		return "sse"
	}
	return h.StreamFraming
}

// SettingNames lists the setting names.
func (h HostSpec) SettingNames() []string {
	out := make([]string, 0, len(h.Settings))
	for _, s := range h.Settings {
		out = append(out, s.Name)
	}
	return out
}

// AccessPolicy is how an adapter reaches a backend. Pure data (AUTH-10).
type AccessPolicy struct {
	Provider           string
	Supports           EndpointSupport
	AuthModes          []string
	EnterpriseVariants []string
	EnvKeys            []string
	CredentialPolicy   string   // "" reads as "key"
	AuthScheme         []string // nil reads as ["bearer"]
	Headers            [][2]string
	Host               *HostSpec
	LoginHint          string
	Backend            string // "" reads as "api"
	BackendOptions     map[string]string
	SystemPrefix       string
	BaseURL            string
}

// EffectiveCredentialPolicy returns CredentialPolicy or "key".
func (p AccessPolicy) EffectiveCredentialPolicy() string {
	if p.CredentialPolicy == "" {
		return "key"
	}
	return p.CredentialPolicy
}

// EffectiveAuthScheme returns AuthScheme or ["bearer"].
func (p AccessPolicy) EffectiveAuthScheme() []string {
	if len(p.AuthScheme) == 0 {
		return []string{"bearer"}
	}
	return p.AuthScheme
}

// EffectiveBackend returns Backend or "api".
func (p AccessPolicy) EffectiveBackend() string {
	if p.Backend == "" {
		return "api"
	}
	return p.Backend
}

// AuthHeader is the first header-carrying scheme (bearer or x-api-key).
func (p AccessPolicy) AuthHeader() string {
	for _, s := range p.EffectiveAuthScheme() {
		switch s {
		case "bearer":
			return "bearer"
		case "x-api-key", "api-key":
			return "x-api-key"
		}
	}
	return "bearer"
}

// CloudChain reports whether the credential policy is a cloud chain.
func (p AccessPolicy) CloudChain() bool {
	switch p.EffectiveCredentialPolicy() {
	case "aws-chain", "azure-chain", "gcp-chain":
		return true
	}
	return false
}

// Hosted reports whether the policy names a cloud host.
func (p AccessPolicy) Hosted() bool { return p.Host != nil }

// Validate checks the policy invariants.
func (p AccessPolicy) Validate() error {
	if p.Provider == "" {
		return valueErrorf("AccessPolicy.provider must be non-empty")
	}
	policy := p.EffectiveCredentialPolicy()
	if policy == "oauth" && len(p.EnvKeys) > 0 {
		return valueErrorf("%s: an 'oauth' access policy declares no env_keys", p.Provider)
	}
	if !inVocab(policy, CredentialPolicies) {
		return valueErrorf("%s: unknown credential_policy %q", p.Provider, p.CredentialPolicy)
	}
	for _, s := range p.EffectiveAuthScheme() {
		if !inVocab(s, AuthSchemes) {
			return valueErrorf("%s: unknown auth_scheme %q", p.Provider, s)
		}
	}
	if inVocab("sigv4", p.EffectiveAuthScheme()) && (p.Host == nil || p.Host.SigV4Service == "") {
		return valueErrorf("%s: sigv4 needs a host with sigv4_service", p.Provider)
	}
	if p.CloudChain() && p.Host == nil && p.Provider != "vertex-express" {
		return valueErrorf("%s: a cloud chain policy needs a host", p.Provider)
	}
	return nil
}

// WithHeaders returns a copy with these static headers replaced or appended
// (names compared case-insensitively).
func (p AccessPolicy) WithHeaders(headers [][2]string) AccessPolicy {
	lowered := map[string]bool{}
	for _, h := range headers {
		lowered[strings.ToLower(h[0])] = true
	}
	var kept [][2]string
	for _, h := range p.Headers {
		if !lowered[strings.ToLower(h[0])] {
			kept = append(kept, h)
		}
	}
	p.Headers = append(kept, headers...)
	return p
}

// WithBackendOptions returns a copy with the options merged.
func (p AccessPolicy) WithBackendOptions(options map[string]string) AccessPolicy {
	merged := map[string]string{}
	for k, v := range p.BackendOptions {
		merged[k] = v
	}
	for k, v := range options {
		merged[k] = v
	}
	p.BackendOptions = merged
	return p
}

// ProviderManifest is the earlier name: an adapter's manifest is its policy.
type ProviderManifest = AccessPolicy
