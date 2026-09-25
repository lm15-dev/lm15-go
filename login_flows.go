package lm15

import (
	"regexp"
	"strings"
)

// The account flows lm15 implements (R11's inventory): ports of lm15-python
// lm15/login/flows/{xai,claude,codex,copilot,kimi,meta,openrouter}.py. Same
// clients, same requests, same material shapes — an entry one SDK writes is
// another's (lm15-contract auth/managed/profiles.json). A flow describes one
// provider's protocol; the engine owns deadlines, cancellation, UI and HTTP
// bounds; the store owns persistence; the manager owns lifecycle.

type flowResult struct {
	material     JSONObject
	label        string
	renewal      string // refresh_token | remint | none | external | recipe
	settings     map[string]string
	accountLabel string
}

const deviceGrant = "urn:ietf:params:oauth:grant-type:device_code"

// oauthMaterial: the actual expiry and the numbers the renewal lead needs.
func oauthMaterial(access, refresh string, expiresInS float64, nowMs int64, extra JSONObject) JSONObject {
	m := JSONObject{{"type", "oauth"}, {"access", access}}
	if refresh != "" {
		m.Set("refresh", refresh)
	}
	m.Set("issued_at", nowMs)
	if expiresInS > 0 {
		m.Set("lifetime_s", floatLexeme(expiresInS))
		m.Set("expires", int64(float64(nowMs)+expiresInS*1000))
	}
	for k, v := range extra.All() {
		m.Set(k, v)
	}
	return m
}

// floatLexeme keeps a float a float in the file (3600.0, as lm15-python writes it).
func floatLexeme(f float64) any {
	if f == float64(int64(f)) {
		return jsonNumberOf(strings.TrimSuffix(formatFloat(f), ".0") + ".0")
	}
	return f
}

func materialStr(m JSONObject, key string) string {
	s, _ := m.Get(key).(string)
	return s
}

type accountFlow string

const (
	flowXai        accountFlow = "xai"
	flowClaude     accountFlow = "claude-code"
	flowCodex      accountFlow = "openai-codex"
	flowCopilot    accountFlow = "github-copilot"
	flowKimi       accountFlow = "kimi-code"
	flowMeta       accountFlow = "meta"
	flowOpenRouter accountFlow = "openrouter"
)

var accountFlows = []accountFlow{flowXai, flowClaude, flowCodex, flowOpenRouter, flowMeta, flowKimi, flowCopilot}

func accountFlowFor(provider string) (accountFlow, bool) {
	for _, f := range accountFlows {
		if string(f) == provider {
			return f, true
		}
	}
	return "", false
}

const (
	xaiClientID  = "b1a00492-073a-47ea-816f-4c329264a828"
	xaiDeviceURL = "https://auth.x.ai/oauth2/device/code"
	xaiTokenURL  = "https://auth.x.ai/oauth2/token"
	xaiScope     = "openid profile email offline_access grok-cli:access api:access"

	claudeClientID             = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	claudeAuthorizeURL         = "https://claude.com/cai/oauth/authorize"
	claudeLoopbackAuthorizeURL = "https://claude.ai/oauth/authorize"
	claudeTokenURL             = "https://platform.claude.com/v1/oauth/token"
	claudeRedirectURI          = "https://platform.claude.com/oauth/code/callback"
	claudeLoopbackRedirectURI  = "http://localhost:53692/callback"
	claudeScopes               = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

	codexClientID           = "app_EMoamEEZ73f0CkXaXp7hrann"
	codexAuthorizeURL       = "https://auth.openai.com/oauth/authorize"
	codexTokenURL           = "https://auth.openai.com/oauth/token"
	codexRedirectURI        = "http://localhost:1455/auth/callback"
	codexDeviceUserCodeURL  = "https://auth.openai.com/api/accounts/deviceauth/usercode"
	codexDeviceTokenURL     = "https://auth.openai.com/api/accounts/deviceauth/token"
	codexDeviceVerification = "https://auth.openai.com/codex/device"
	codexDeviceRedirectURI  = "https://auth.openai.com/deviceauth/callback"
	codexDeviceTimeoutS     = 900.0
	codexScope              = "openid profile email offline_access"

	copilotClientID       = "Iv1.b507a08c87ecfe98"
	copilotDefaultDomain  = "github.com"
	copilotDefaultAPIBase = "https://api.individual.githubcopilot.com"

	kimiClientID         = "17e5f671-d194-4dfb-9706-5516cb48c098"
	kimiDefaultOAuthHost = "https://auth.kimi.com"

	metaClientID    = "1031625952748946"
	metaDeviceURL   = "https://auth.meta.com/oidc/device/authorization/"
	metaTokenURL    = "https://auth.meta.com/oidc/device/token/"
	metaKeyMintURL  = "https://api.meta.ai/muse-code/key"
	metaKeyLifetime = 86400.0

	openrouterAuthorizeURL = "https://openrouter.ai/auth"
	openrouterKeyURL       = "https://openrouter.ai/api/v1/auth/keys"
)

var copilotHeaders = [][2]string{
	{"User-Agent", "GitHubCopilotChat/0.35.0"},
	{"Editor-Version", "vscode/1.107.0"},
	{"Editor-Plugin-Version", "copilot-chat/0.35.0"},
	{"Copilot-Integration-Id", "vscode-chat"},
}

func pkcePair() (string, string) {
	verifier := randomBase64URL(64) // 86 characters, the reference's generate_pkce
	return verifier, PKCEChallenge(verifier)
}

func accountMethod(id, label, flow, availability, reason string, delivery []string, subscription bool, note string) LoginMethod {
	return LoginMethod{ID: id, Label: label, Kind: "account", Flow: flow, Availability: availability, Reason: reason, Delivery: delivery, Subscription: subscription, BillingNote: note}
}

// descriptor is the AUTH-13 descriptor as the provider's registration defines it (native).
func (f accountFlow) descriptor() ProviderDescriptor {
	unverified := "provider permission and billing remain unverified"
	claudeNote := "Provider permission and included usage must be verified separately for your account."
	switch f {
	case flowXai:
		return ProviderDescriptor{ID: "xai", Label: "xAI", Service: "xAI", Routes: []string{"xai"}, ConsoleURL: "https://console.x.ai", Methods: []LoginMethod{
			accountMethod("device", "Sign in with SuperGrok or X Premium", "device_code", "supported", "", []string{"device"}, true, "Subscription access per xAI's own recommendation (2026-09-01); the API key path is metered."),
		}}
	case flowClaude:
		return ProviderDescriptor{ID: "claude-code", Label: "Claude (subscription)", Service: "Anthropic", Routes: []string{"claude-code"}, Methods: []LoginMethod{
			accountMethod("browser", "Sign in with Claude (paste code from hosted page)", "authorization_code", "unverified", "LM15 hosted login, inference, persistence and early renewal observed 2026-09-23; permission and billing remain unverified", []string{"manual"}, true, claudeNote),
			accountMethod("loopback", "Sign in with Claude (local browser callback)", "authorization_code", "unverified", "no live LM15 receipt for this local callback flow", []string{"loopback", "manual"}, true, claudeNote),
		}}
	case flowCodex:
		return ProviderDescriptor{ID: "openai-codex", Label: "ChatGPT (subscription)", Service: "OpenAI", Routes: []string{"openai-codex"}, Methods: []LoginMethod{
			accountMethod("browser", "Sign in with ChatGPT (browser)", "authorization_code", "unverified", "Browser login, inference, persistence and early renewal observed 2026-09-23; "+unverified, []string{"loopback", "manual"}, true, ""),
			accountMethod("device", "Sign in with ChatGPT (device code, for SSH/headless)", "device_code", "unverified", "Device login and inference observed 2026-09-23; "+unverified, []string{"device"}, true, ""),
		}}
	case flowCopilot:
		m := accountMethod("device", "Sign in with GitHub (Copilot subscription)", "device_code", "unverified", "Login, catalog, inference, persistence and early renewal observed 2026-09-23; permission review pending", []string{"device"}, true, "Some models require enabling on your account first; LM15 does not change that setting during login.")
		m.Fields = []MethodField{{ID: "enterprise_domain", Label: "GitHub Enterprise domain (blank for github.com)", Type: "text", Required: false, Help: "e.g. company.ghe.com"}}
		return ProviderDescriptor{ID: "github-copilot", Label: "GitHub Copilot", Service: "GitHub", Routes: []string{"github-copilot"}, Methods: []LoginMethod{m}}
	case flowKimi:
		return ProviderDescriptor{ID: "kimi-code", Label: "Kimi Code (subscription)", Service: "Moonshot AI", Routes: []string{"kimi-code"}, Methods: []LoginMethod{
			accountMethod("device", "Sign in with Kimi Code (subscription)", "device_code", "unverified", "no live receipt yet", []string{"device"}, true, ""),
		}}
	case flowMeta:
		return ProviderDescriptor{ID: "meta", Label: "Meta", Service: "Meta", Routes: []string{"meta", "meta-chat", "meta-anthropic"}, ConsoleURL: "https://dev.meta.ai", Methods: []LoginMethod{
			accountMethod("device", "Sign in with Meta (Muse subscription)", "device_code", "unverified", "no live receipt yet", []string{"device"}, true, "Minted Model API keys are tied to the Muse subscription; verify entitlement on your account."),
		}}
	default: // openrouter
		return ProviderDescriptor{ID: "openrouter", Label: "OpenRouter", Service: "OpenRouter", Routes: []string{"openrouter"}, ConsoleURL: "https://openrouter.ai/keys", Methods: []LoginMethod{
			accountMethod("browser", "Sign in with OpenRouter (creates an API key for this app)", "authorization_code", "unverified", "Login, key limit, inference and persistence observed 2026-09-23; broader support review pending", []string{"loopback", "manual"}, false, "The minted key spends your OpenRouter credits like any other key."),
		}}
	}
}

func (f accountFlow) login(c *loginContext, method string, settings, answers map[string]string) (flowResult, error) {
	switch f {
	case flowXai:
		return xaiLogin(c)
	case flowClaude:
		return claudeLogin(c, method == "browser")
	case flowCodex:
		if method == "device" {
			return codexDevice(c)
		}
		return codexBrowser(c)
	case flowCopilot:
		return copilotLogin(c, settings, answers)
	case flowKimi:
		return kimiLogin(c, settings)
	case flowMeta:
		return metaLogin(c)
	default:
		return openrouterLogin(c)
	}
}

func (f accountFlow) renew(c *loginContext, material JSONObject, settings map[string]string) (flowResult, error) {
	switch f {
	case flowXai:
		return xaiRenew(c, material)
	case flowClaude:
		return claudeRenew(c, material)
	case flowCodex:
		return codexRenew(c, material)
	case flowCopilot:
		github := materialStr(material, "refresh")
		if github == "" {
			return flowResult{}, denied("Copilot credential has no GitHub token to renew with")
		}
		m, err := copilotExchange(c, github, settings)
		return flowResult{material: m, label: "GitHub Copilot", renewal: "remint"}, err
	case flowKimi:
		return kimiRenew(c, material, settings)
	case flowMeta:
		identity := materialStr(material, "refresh")
		if identity == "" {
			return flowResult{}, denied("Meta credential has no identity token to re-mint with")
		}
		m, err := metaMint(c, identity)
		return flowResult{material: m, label: "Meta (Muse subscription)", renewal: "remint"}, err
	default:
		return flowResult{material: material, label: "OpenRouter (minted key)", renewal: "none"}, nil
	}
}

func (f accountFlow) requestAuth(material JSONObject, settings map[string]string) (RequestAuth, error) {
	incomplete := denied("the saved credential is incomplete")
	auth := RequestAuth{Headers: map[string]string{}}
	switch f {
	case flowMeta:
		if auth.Credential = materialStr(material, "access"); auth.Credential == "" {
			return auth, incomplete
		}
		auth.CredentialKind = "api_key"
		return auth, nil
	case flowOpenRouter:
		if auth.Credential = materialStr(material, "key"); auth.Credential == "" {
			return auth, incomplete
		}
		auth.CredentialKind = "api_key"
		return auth, nil
	}
	if auth.Credential = materialStr(material, "access"); auth.Credential == "" {
		return auth, incomplete
	}
	auth.CredentialKind = "bearer"
	switch f {
	case flowCodex:
		account := materialStr(material, "accountId")
		if account == "" {
			account = ExtractChatGPTAccountID(auth.Credential)
		}
		if account != "" {
			auth.Headers["chatgpt-account-id"] = account
		}
		auth.AccountID = account
	case flowCopilot:
		for _, h := range copilotHeaders {
			auth.Headers[h[0]] = h[1]
		}
		base, err := copilotBaseURL(material, settings)
		if err != nil {
			return auth, err
		}
		auth.BaseURL = base
	}
	return auth, nil
}

// ─── xAI ─────────────────────────────────────────────────────────────

func xaiMaterial(r *httpReply, nowMs int64, previousRefresh string) (JSONObject, error) {
	access := r.str("access_token")
	if access == "" {
		return nil, denied("xAI token response carried no access token")
	}
	refresh := r.str("refresh_token")
	if refresh == "" {
		refresh = previousRefresh // xAI may omit it when it does not rotate
	}
	lifetime := positive(r.body.Get("expires_in"))
	if lifetime == 0 {
		lifetime = 3600
	}
	return oauthMaterial(access, refresh, lifetime, nowMs, nil), nil
}

func xaiLogin(c *loginContext) (flowResult, error) {
	start, err := c.form(xaiDeviceURL, [][2]string{{"client_id", xaiClientID}, {"scope", xaiScope}, {"referrer", "lm15"}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !start.ok {
		return flowResult{}, denied("xAI refused to start a device authorization (HTTP %d)", start.status)
	}
	deviceCode, userCode := start.str("device_code"), start.str("user_code")
	if deviceCode == "" || userCode == "" {
		return flowResult{}, denied("xAI device authorization response is missing required fields")
	}
	target := httpsURL(start.body.Get("verification_uri"))
	if target == "" {
		return flowResult{}, denied("xAI returned an untrusted verification URL")
	}
	if complete, _ := start.body.Get("verification_uri_complete").(string); complete != "" {
		if target = httpsURL(complete); target == "" {
			return flowResult{}, denied("xAI returned an untrusted verification URL")
		}
	}
	interval, expires := positive(start.body.Get("interval")), positive(start.body.Get("expires_in"))
	c.notify(Notice{Type: "device_code", UserCode: userCode, VerificationURL: target, ExpiresInS: orDefault(expires, 900), IntervalS: orDefault(interval, 5)})
	value, err := runDeviceFlow(c, interval, expires, func() (deviceStep, error) {
		reply, err := c.form(xaiTokenURL, [][2]string{{"grant_type", deviceGrant}, {"client_id", xaiClientID}, {"device_code", deviceCode}}, nil)
		if err != nil {
			return deviceStep{}, err
		}
		if reply.ok {
			m, err := xaiMaterial(reply, c.nowMs(), "")
			return deviceStep{status: "complete", value: m}, err
		}
		switch reply.errorCode() {
		case "authorization_pending":
			return deviceStep{status: "pending"}, nil
		case "slow_down":
			return deviceStep{status: "slow_down", interval: positive(reply.body.Get("interval"))}, nil
		case "access_denied", "authorization_denied":
			return deviceStep{status: "denied"}, nil
		case "expired_token":
			return deviceStep{status: "expired"}, nil
		}
		return deviceStep{}, denied("xAI device token polling failed (HTTP %d)", reply.status)
	})
	if err != nil {
		return flowResult{}, err
	}
	return flowResult{material: value.(JSONObject), label: "xAI subscription", renewal: "refresh_token"}, nil
}

func xaiRenew(c *loginContext, material JSONObject) (flowResult, error) {
	refresh := materialStr(material, "refresh")
	if refresh == "" {
		return flowResult{}, denied("xAI credential has no refresh token")
	}
	reply, err := c.form(xaiTokenURL, [][2]string{{"grant_type", "refresh_token"}, {"client_id", xaiClientID}, {"refresh_token", refresh}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !reply.ok {
		if reply.status == 400 || reply.status == 401 || reply.status == 403 {
			return flowResult{}, denied("xAI rejected the refresh token (HTTP %d)", reply.status)
		}
		return flowResult{}, denied("xAI refresh failed (HTTP %d)", reply.status)
	}
	m, err := xaiMaterial(reply, c.nowMs(), refresh)
	return flowResult{material: m, label: "xAI subscription", renewal: "refresh_token"}, err
}

func orDefault(v, fallback float64) float64 {
	if v > 0 {
		return v
	}
	return fallback
}

// ─── Claude ──────────────────────────────────────────────────────────

func claudeTokens(r *httpReply, nowMs int64) (JSONObject, error) {
	access, refresh := r.str("access_token"), r.str("refresh_token")
	if access == "" || refresh == "" {
		return nil, denied("Claude token response is missing required fields")
	}
	return oauthMaterial(access, refresh, positive(r.body.Get("expires_in")), nowMs, nil), nil
}

func openListener(c *loginContext, path, state string, checkState bool, port int, redirectHost string) (*callbackListener, error) {
	if !c.listenerAvailable {
		return nil, nil
	}
	l, err := openCallbackListener(path, state, checkState, port, "127.0.0.1", redirectHost)
	if err != nil {
		if e, ok := err.(*Error); ok && e.Reason == "method_unavailable" {
			c.notify(Notice{Type: "info", Message: "Could not listen on port " + itoa(port) + "; paste the full redirect URL when the browser finishes."})
			return nil, nil
		}
		return nil, err
	}
	return l, nil
}

func claudeLogin(c *loginContext, hosted bool) (flowResult, error) {
	if err := c.check(); err != nil {
		return flowResult{}, err
	}
	var verifier, challenge string
	if hosted {
		verifier = randomBase64URL(32) // 43 characters, as in the captured native flow
		challenge = PKCEChallenge(verifier)
	} else {
		verifier, challenge = pkcePair()
	}
	state := randomBase64URL(32)
	redirect, authorize, path := claudeRedirectURI, claudeAuthorizeURL, "/oauth/code/callback"
	var listener *callbackListener
	if !hosted {
		redirect, authorize, path = claudeLoopbackRedirectURI, claudeLoopbackAuthorizeURL, "/callback"
		var err error
		if listener, err = openListener(c, "/callback", state, true, 53692, "localhost"); err != nil {
			return flowResult{}, err
		}
		if listener != nil {
			defer listener.stop()
		}
	}
	instructions := "Sign in to Claude in your browser. If the local callback cannot be reached, paste the full redirect URL (or code#state) here."
	if hosted {
		instructions = "Sign in to Claude in your browser. On the Authentication code page, copy the whole displayed code (including #state) and paste it here. The full return URL also works. Your browser may be on another machine; no localhost connection is needed."
	}
	c.notify(Notice{Type: "auth_url", URL: withQuery(authorize, [][2]string{
		{"code", "true"}, {"client_id", claudeClientID}, {"response_type", "code"}, {"redirect_uri", redirect},
		{"scope", claudeScopes}, {"code_challenge", challenge}, {"code_challenge_method", "S256"}, {"state", state},
	}), Instructions: instructions})
	prompt := Prompt{Type: "manual_code", FieldID: "return", Label: "Paste the full code#state or return URL here", Accepted: "the full return URL, or code#state (a bare code without state is not accepted)"}
	returned, err := awaitReturn(c, listener, prompt, returnContext{expectedState: state, checkState: true, registeredPath: path, registeredURI: redirect})
	if err != nil {
		return flowResult{}, err
	}
	if err := c.check(); err != nil {
		return flowResult{}, err
	}
	c.notify(Notice{Type: "progress", Stage: "exchange", Message: "Exchanging the authorization code…"})
	reply, err := c.json(claudeTokenURL, JSONObject{
		{"grant_type", "authorization_code"}, {"code", returned.code}, {"redirect_uri", redirect},
		{"client_id", claudeClientID}, {"code_verifier", verifier}, {"state", state},
	}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !reply.ok {
		return flowResult{}, deniedAt("exchange", reply, "Claude authorization-code exchange failed: %s. The authorization code will not be retried automatically.", reply.failureSummary())
	}
	if err := c.check(); err != nil {
		return flowResult{}, err
	}
	m, err := claudeTokens(reply, c.nowMs())
	return flowResult{material: m, label: "Claude subscription", renewal: "refresh_token"}, err
}

func claudeRenew(c *loginContext, material JSONObject) (flowResult, error) {
	refresh := materialStr(material, "refresh")
	if refresh == "" {
		return flowResult{}, denied("Claude credential has no refresh token")
	}
	reply, err := c.json(claudeTokenURL, JSONObject{{"grant_type", "refresh_token"}, {"client_id", claudeClientID}, {"refresh_token", refresh}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !reply.ok {
		return flowResult{}, deniedAt("renewal", reply, "Claude token renewal failed: %s", reply.failureSummary())
	}
	m, err := claudeTokens(reply, c.nowMs())
	return flowResult{material: m, label: "Claude subscription", renewal: "refresh_token"}, err
}

// ─── ChatGPT / Codex ─────────────────────────────────────────────────

func codexTokens(body JSONObject, nowMs int64) (JSONObject, error) {
	access, _ := body.Get("access_token").(string)
	refresh, _ := body.Get("refresh_token").(string)
	if access == "" || refresh == "" {
		return nil, denied("ChatGPT token response is missing required fields")
	}
	lifetime := positive(body.Get("expires_in"))
	if lifetime == 0 {
		if exp := jwtExpiresAtMs(access); exp != nil {
			if s := float64(*exp+5*60*1000-nowMs) / 1000; s > 0 {
				lifetime = s
			}
		}
	}
	account := ExtractChatGPTAccountID(access)
	if account == "" {
		return nil, denied("ChatGPT token carries no account id")
	}
	extra := JSONObject{{"accountId", account}}
	if id, _ := body.Get("id_token").(string); id != "" {
		extra.Set("id_token", id)
	}
	return oauthMaterial(access, refresh, lifetime, nowMs, extra), nil
}

func codexExchange(c *loginContext, code, verifier, redirect string) (flowResult, error) {
	c.notify(Notice{Type: "progress", Stage: "exchange", Message: "Exchanging the authorization code…"})
	reply, err := c.form(codexTokenURL, [][2]string{{"grant_type", "authorization_code"}, {"client_id", codexClientID}, {"code", code}, {"code_verifier", verifier}, {"redirect_uri", redirect}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !reply.ok {
		return flowResult{}, denied("ChatGPT rejected the authorization code (HTTP %d)", reply.status)
	}
	m, err := codexTokens(reply.body, c.nowMs())
	if err != nil {
		return flowResult{}, err
	}
	return flowResult{material: m, label: "ChatGPT subscription", renewal: "refresh_token", accountLabel: materialStr(m, "accountId")}, nil
}

func codexBrowser(c *loginContext) (flowResult, error) {
	verifier, challenge := pkcePair()
	state := secretHex(16)
	listener, err := openListener(c, "/auth/callback", state, true, 1455, "localhost")
	if err != nil {
		return flowResult{}, err
	}
	if listener != nil {
		defer listener.stop()
	}
	c.notify(Notice{Type: "auth_url", URL: withQuery(codexAuthorizeURL, [][2]string{
		{"response_type", "code"}, {"client_id", codexClientID}, {"redirect_uri", codexRedirectURI}, {"scope", codexScope},
		{"code_challenge", challenge}, {"code_challenge_method", "S256"}, {"state", state},
		{"id_token_add_organizations", "true"}, {"codex_cli_simplified_flow", "true"}, {"originator", "lm15"},
	}), Instructions: "Sign in to ChatGPT in your browser. If the browser is on another machine, paste the final redirect URL back here."})
	prompt := Prompt{Type: "manual_code", FieldID: "return", Label: "Paste the redirect URL here (or wait for the browser)", Accepted: "the full redirect URL, or the code"}
	returned, err := awaitReturn(c, listener, prompt, returnContext{expectedState: state, checkState: true, registeredPath: "/auth/callback"})
	if err != nil {
		return flowResult{}, err
	}
	return codexExchange(c, returned.code, verifier, codexRedirectURI)
}

func codexDevice(c *loginContext) (flowResult, error) {
	start, err := c.json(codexDeviceUserCodeURL, JSONObject{{"client_id", codexClientID}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !start.ok {
		if start.status == 404 {
			return flowResult{}, denied("ChatGPT device-code login is not enabled for this server; use the browser method")
		}
		return flowResult{}, denied("ChatGPT refused to start a device authorization (HTTP %d)", start.status)
	}
	deviceID, userCode := start.str("device_auth_id"), start.str("user_code")
	if deviceID == "" || userCode == "" {
		return flowResult{}, denied("ChatGPT device authorization response is missing required fields")
	}
	interval, _ := number(start.body.Get("interval"))
	if interval < 0 {
		interval = 0
	}
	c.notify(Notice{Type: "device_code", UserCode: userCode, VerificationURL: codexDeviceVerification, ExpiresInS: codexDeviceTimeoutS, IntervalS: orDefault(interval, 5)})
	value, err := runDeviceFlow(c, interval, codexDeviceTimeoutS, func() (deviceStep, error) {
		reply, err := c.json(codexDeviceTokenURL, JSONObject{{"device_auth_id", deviceID}, {"user_code", userCode}}, nil)
		if err != nil {
			return deviceStep{}, err
		}
		if reply.ok {
			code, verifier := reply.str("authorization_code"), reply.str("code_verifier")
			if code == "" || verifier == "" {
				return deviceStep{}, denied("ChatGPT device token response is missing required fields")
			}
			return deviceStep{status: "complete", value: [2]string{code, verifier}}, nil
		}
		if reply.status == 403 || reply.status == 404 {
			return deviceStep{status: "pending"}, nil
		}
		switch reply.errorCode() {
		case "deviceauth_authorization_pending":
			return deviceStep{status: "pending"}, nil
		case "slow_down":
			return deviceStep{status: "slow_down"}, nil
		}
		return deviceStep{}, denied("ChatGPT device authorization failed (HTTP %d)", reply.status)
	})
	if err != nil {
		return flowResult{}, err
	}
	pair := value.([2]string)
	return codexExchange(c, pair[0], pair[1], codexDeviceRedirectURI)
}

func codexRenew(c *loginContext, material JSONObject) (flowResult, error) {
	refresh := materialStr(material, "refresh")
	if refresh == "" {
		return flowResult{}, denied("ChatGPT credential has no refresh token")
	}
	reply, err := c.form(codexTokenURL, [][2]string{{"grant_type", "refresh_token"}, {"refresh_token", refresh}, {"client_id", codexClientID}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !reply.ok {
		return flowResult{}, denied("ChatGPT rejected the refresh token (HTTP %d)", reply.status)
	}
	body := JSONObject{}
	for k, v := range reply.body.All() {
		body.Set(k, v)
	}
	if s, _ := body.Get("refresh_token").(string); s == "" {
		body.Set("refresh_token", refresh) // OpenAI may omit it when it does not rotate
	}
	m, err := codexTokens(body, c.nowMs())
	if err != nil {
		return flowResult{}, err
	}
	return flowResult{material: m, label: "ChatGPT subscription", renewal: "refresh_token", accountLabel: materialStr(m, "accountId")}, nil
}

// ─── GitHub Copilot ──────────────────────────────────────────────────

var hostPattern = regexp.MustCompile(`^[a-z0-9.-]+$`)

func copilotDomain(settings map[string]string) (string, error) {
	raw := strings.TrimSpace(settings["enterprise_domain"])
	if raw == "" {
		return copilotDefaultDomain, nil
	}
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	host := ""
	if rest := raw[strings.Index(raw, "://")+3:]; rest != "" {
		host = strings.ToLower(strings.SplitN(strings.SplitN(rest, "/", 2)[0], ":", 2)[0])
	}
	if host == "" || !hostPattern.MatchString(host) {
		return "", denied("invalid GitHub Enterprise domain")
	}
	return host, nil
}

// copilotBaseURL is the account's API host from the Copilot token, validated
// against GitHub's domains; never an arbitrary host a token names (AUTH-20.9).
func copilotBaseURL(material JSONObject, settings map[string]string) (string, error) {
	domain, err := copilotDomain(settings)
	if err != nil {
		return "", err
	}
	token := materialStr(material, "access")
	if i := strings.Index(token, "proxy-ep="); i >= 0 {
		host := strings.ToLower(strings.TrimSpace(strings.SplitN(token[i+len("proxy-ep="):], ";", 2)[0]))
		api := host
		if strings.HasPrefix(host, "proxy.") {
			api = "api." + strings.TrimPrefix(host, "proxy.")
		}
		allowed := []string{".githubcopilot.com"}
		if domain != copilotDefaultDomain {
			allowed = []string{"." + domain, ".githubcopilot.com"}
		}
		for _, suffix := range allowed {
			if hostPattern.MatchString(api) && strings.HasSuffix(api, suffix) {
				return "https://" + api, nil
			}
		}
	}
	if domain != copilotDefaultDomain {
		return "https://copilot-api." + domain, nil
	}
	return copilotDefaultAPIBase, nil
}

func copilotExchange(c *loginContext, githubToken string, settings map[string]string) (JSONObject, error) {
	domain, err := copilotDomain(settings)
	if err != nil {
		return nil, err
	}
	headers := append([][2]string{{"Authorization", "Bearer " + githubToken}}, copilotHeaders...)
	reply, err := c.get("https://api."+domain+"/copilot_internal/v2/token", headers)
	if err != nil {
		return nil, err
	}
	if reply.status == 401 || reply.status == 403 {
		return nil, denied("GitHub rejected the token for Copilot; sign in again")
	}
	if !reply.ok {
		return nil, denied("Copilot token exchange failed (HTTP %d)", reply.status)
	}
	token := reply.str("token")
	expiresAt, ok := number(reply.body.Get("expires_at"))
	if _, isString := reply.body.Get("expires_at").(string); token == "" || !ok || isString {
		return nil, denied("Copilot token response is missing required fields")
	}
	now := c.nowMs()
	expiresMs := int64(expiresAt * 1000)
	lifetime := float64(expiresMs-now) / 1000
	if lifetime < 1 {
		lifetime = 1
	}
	return JSONObject{{"type", "oauth"}, {"access", token}, {"refresh", githubToken}, {"issued_at", now}, {"lifetime_s", floatLexeme(lifetime)}, {"expires", expiresMs}}, nil
}

func copilotLogin(c *loginContext, settings, answers map[string]string) (flowResult, error) {
	merged := map[string]string{}
	for k, v := range settings {
		merged[k] = v
	}
	if d := answers["enterprise_domain"]; d != "" {
		merged["enterprise_domain"] = d
	}
	domain, err := copilotDomain(merged)
	if err != nil {
		return flowResult{}, err
	}
	ua := [][2]string{copilotHeaders[0]}
	start, err := c.form("https://"+domain+"/login/device/code", [][2]string{{"client_id", copilotClientID}, {"scope", "read:user"}}, ua)
	if err != nil {
		return flowResult{}, err
	}
	if !start.ok {
		return flowResult{}, denied("GitHub refused to start a device authorization (HTTP %d)", start.status)
	}
	deviceCode, userCode, verification := start.str("device_code"), start.str("user_code"), start.str("verification_uri")
	if deviceCode == "" || userCode == "" || verification == "" {
		return flowResult{}, denied("GitHub device authorization response is missing required fields")
	}
	if httpURL(verification) == "" {
		return flowResult{}, denied("GitHub returned an untrusted verification URL")
	}
	interval, expires := positive(start.body.Get("interval")), positive(start.body.Get("expires_in"))
	c.notify(Notice{Type: "device_code", UserCode: userCode, VerificationURL: verification, ExpiresInS: orDefault(expires, 900), IntervalS: orDefault(interval, 5)})
	tokenURL := "https://" + domain + "/login/oauth/access_token"
	value, err := runDeviceFlow(c, interval, expires, func() (deviceStep, error) {
		reply, err := c.form(tokenURL, [][2]string{{"client_id", copilotClientID}, {"device_code", deviceCode}, {"grant_type", deviceGrant}}, ua)
		if err != nil {
			return deviceStep{}, err
		}
		if token := reply.str("access_token"); token != "" {
			return deviceStep{status: "complete", value: token}, nil
		}
		switch reply.errorCode() {
		case "authorization_pending":
			return deviceStep{status: "pending"}, nil
		case "slow_down":
			return deviceStep{status: "slow_down", interval: positive(reply.body.Get("interval"))}, nil
		case "expired_token":
			return deviceStep{status: "expired"}, nil
		case "access_denied":
			return deviceStep{status: "denied"}, nil
		}
		return deviceStep{}, denied("GitHub device authorization failed (HTTP %d)", reply.status)
	})
	if err != nil {
		return flowResult{}, err
	}
	c.notify(Notice{Type: "progress", Stage: "exchange", Message: "Exchanging the GitHub token for a Copilot token…"})
	m, err := copilotExchange(c, value.(string), merged)
	if err != nil {
		return flowResult{}, err
	}
	result := flowResult{material: m, label: "GitHub Copilot", renewal: "remint", settings: map[string]string{}}
	if domain != copilotDefaultDomain {
		result.label = "GitHub Copilot (" + domain + ")"
		result.settings["enterprise_domain"] = domain
	}
	return result, nil
}

// ─── Kimi Code ───────────────────────────────────────────────────────

func kimiHost(settings map[string]string) string {
	h := settings["oauth_host"]
	if h == "" {
		h = kimiDefaultOAuthHost
	}
	return strings.TrimRight(h, "/")
}

func kimiTokens(r *httpReply, nowMs int64) (JSONObject, error) {
	access, refresh := r.str("access_token"), r.str("refresh_token")
	if access == "" || refresh == "" {
		return nil, denied("Kimi Code token response is missing required fields")
	}
	return oauthMaterial(access, refresh, positive(r.body.Get("expires_in")), nowMs, nil), nil
}

func kimiLogin(c *loginContext, settings map[string]string) (flowResult, error) {
	host := kimiHost(settings)
	start, err := c.form(host+"/api/oauth/device_authorization", [][2]string{{"client_id", kimiClientID}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !start.ok {
		return flowResult{}, denied("Kimi Code refused to start a device authorization (HTTP %d)", start.status)
	}
	verification := httpURL(start.body.Get("verification_uri_complete"))
	if verification == "" {
		verification = httpURL(start.body.Get("verification_uri"))
	}
	deviceCode, userCode := start.str("device_code"), start.str("user_code")
	if deviceCode == "" || userCode == "" || verification == "" {
		return flowResult{}, denied("Kimi Code device authorization response is missing required fields")
	}
	interval, expires := positive(start.body.Get("interval")), orDefault(positive(start.body.Get("expires_in")), 900)
	c.notify(Notice{Type: "device_code", UserCode: userCode, VerificationURL: verification, ExpiresInS: expires, IntervalS: orDefault(interval, 5)})
	value, err := runDeviceFlow(c, interval, expires, func() (deviceStep, error) {
		reply, err := c.form(host+"/api/oauth/token", [][2]string{{"client_id", kimiClientID}, {"device_code", deviceCode}, {"grant_type", deviceGrant}}, nil)
		if err != nil {
			return deviceStep{}, err
		}
		if _, ok := reply.body.Get("access_token").(string); reply.ok && ok {
			m, err := kimiTokens(reply, c.nowMs())
			return deviceStep{status: "complete", value: m}, err
		}
		switch reply.errorCode() {
		case "authorization_pending":
			return deviceStep{status: "pending"}, nil
		case "slow_down":
			return deviceStep{status: "slow_down", interval: positive(reply.body.Get("interval"))}, nil
		case "expired_token":
			return deviceStep{status: "expired"}, nil
		case "access_denied":
			return deviceStep{status: "denied"}, nil
		}
		return deviceStep{}, denied("Kimi Code device token request failed (HTTP %d)", reply.status)
	})
	if err != nil {
		return flowResult{}, err
	}
	result := flowResult{material: value.(JSONObject), label: "Kimi Code subscription", renewal: "refresh_token", settings: map[string]string{}}
	if host != kimiDefaultOAuthHost {
		result.settings["oauth_host"] = host
	}
	return result, nil
}

func kimiRenew(c *loginContext, material JSONObject, settings map[string]string) (flowResult, error) {
	refresh := materialStr(material, "refresh")
	if refresh == "" {
		return flowResult{}, denied("Kimi Code credential has no refresh token")
	}
	reply, err := c.form(kimiHost(settings)+"/api/oauth/token", [][2]string{{"client_id", kimiClientID}, {"grant_type", "refresh_token"}, {"refresh_token", refresh}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if reply.status == 401 || reply.status == 403 || reply.errorCode() == "invalid_grant" {
		return flowResult{}, denied("Kimi Code rejected the refresh token (HTTP %d)", reply.status)
	}
	if !reply.ok {
		// A 429 is transient: this renewal fails, the credential stays.
		e := newError(KindRateLimit, "Kimi Code rate-limited the token refresh (HTTP "+itoa(reply.status)+")")
		e.Provider, e.Status = "kimi-code", reply.status
		return flowResult{}, e
	}
	m, err := kimiTokens(reply, c.nowMs())
	return flowResult{material: m, label: "Kimi Code subscription", renewal: "refresh_token"}, err
}

// ─── Meta ────────────────────────────────────────────────────────────

func metaMint(c *loginContext, identity string) (JSONObject, error) {
	c.notify(Notice{Type: "progress", Stage: "exchange", Message: "Enabling Meta Model API access…"})
	reply, err := c.json(metaKeyMintURL, JSONObject{}, [][2]string{{"Authorization", "Bearer " + identity}, {"x-api-version", "1.0.0"}})
	if err != nil {
		return nil, err
	}
	if reply.status == 401 || reply.status == 403 {
		return nil, denied("Meta session is no longer valid; sign in again")
	}
	if !reply.ok {
		return nil, denied("Meta API key mint failed (HTTP %d)", reply.status)
	}
	key := reply.str("api_key")
	if key == "" {
		if action := httpURL(reply.body.Get("action_url")); action != "" {
			return nil, denied("Meta did not issue an API key; complete setup at %s", action)
		}
		return nil, denied("Meta did not issue an API key")
	}
	now := c.nowMs()
	return JSONObject{{"type", "oauth"}, {"access", key}, {"refresh", identity}, {"issued_at", now}, {"lifetime_s", floatLexeme(metaKeyLifetime)}, {"expires", int64(float64(now) + metaKeyLifetime*1000)}}, nil
}

func metaLogin(c *loginContext) (flowResult, error) {
	start, err := c.form(metaDeviceURL, [][2]string{{"client_id", metaClientID}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !start.ok {
		return flowResult{}, denied("Meta refused to start a device authorization (HTTP %d)", start.status)
	}
	verification := httpURL(start.body.Get("verification_uri_complete"))
	if verification == "" {
		verification = httpURL(start.body.Get("verification_uri"))
	}
	deviceCode, userCode := start.str("device_code"), start.str("user_code")
	if deviceCode == "" || userCode == "" || verification == "" {
		return flowResult{}, denied("Meta device authorization response is missing required fields")
	}
	interval, expires := positive(start.body.Get("interval")), positive(start.body.Get("expires_in"))
	c.notify(Notice{Type: "device_code", UserCode: userCode, VerificationURL: verification, ExpiresInS: orDefault(expires, 900), IntervalS: orDefault(interval, 5)})
	value, err := runDeviceFlow(c, interval, expires, func() (deviceStep, error) {
		reply, err := c.form(metaTokenURL, [][2]string{{"grant_type", deviceGrant}, {"device_code", deviceCode}, {"client_id", metaClientID}}, nil)
		if err != nil {
			return deviceStep{}, err
		}
		if token := reply.str("access_token"); reply.ok && token != "" {
			return deviceStep{status: "complete", value: token}, nil
		}
		switch reply.errorCode() {
		case "authorization_pending":
			return deviceStep{status: "pending"}, nil
		case "slow_down":
			return deviceStep{status: "slow_down", interval: positive(reply.body.Get("interval"))}, nil
		case "access_denied":
			return deviceStep{status: "denied"}, nil
		case "expired_token":
			return deviceStep{status: "expired"}, nil
		}
		return deviceStep{}, denied("Meta device token request failed (HTTP %d)", reply.status)
	})
	if err != nil {
		return flowResult{}, err
	}
	m, err := metaMint(c, value.(string))
	return flowResult{material: m, label: "Meta (Muse subscription)", renewal: "remint"}, err
}

// ─── OpenRouter ──────────────────────────────────────────────────────

func openrouterLogin(c *loginContext) (flowResult, error) {
	verifier, challenge := pkcePair()
	path := "/oauth/callback/" + randomBase64URL(24)
	if !c.listenerAvailable {
		return flowResult{}, authOperation("openrouter: this sign-in needs a local callback listener, which this host does not provide", "method_unavailable", "reservation", "not_committed", "choose_method")
	}
	// No state in OpenRouter's protocol: the one-time random callback path
	// plus PKCE is the evidenced equivalent binding (AUTH-18).
	listener, err := openCallbackListener(path, "", false, 0, "127.0.0.1", "")
	if err != nil {
		return flowResult{}, err
	}
	defer listener.stop()
	c.notify(Notice{Type: "auth_url", URL: withQuery(openrouterAuthorizeURL, [][2]string{{"callback_url", listener.redirectURI}, {"code_challenge", challenge}, {"code_challenge_method", "S256"}}),
		Instructions: "Sign in to OpenRouter in your browser and approve the key. If the browser is on another machine, paste the final redirect URL back here."})
	prompt := Prompt{Type: "manual_code", FieldID: "return", Label: "Paste the redirect URL or code here (or wait for the browser)", Accepted: "the full redirect URL, or the code"}
	returned, err := awaitReturn(c, listener, prompt, returnContext{allowBareCode: true, registeredPath: path})
	if err != nil {
		return flowResult{}, err
	}
	c.notify(Notice{Type: "progress", Stage: "exchange", Message: "Exchanging the code for an API key…"})
	reply, err := c.json(openrouterKeyURL, JSONObject{{"code", returned.code}, {"code_verifier", verifier}, {"code_challenge_method", "S256"}}, nil)
	if err != nil {
		return flowResult{}, err
	}
	if !reply.ok {
		return flowResult{}, denied("OpenRouter rejected the authorization code (HTTP %d)", reply.status)
	}
	key := reply.str("key")
	if key == "" {
		return flowResult{}, denied("OpenRouter returned no key")
	}
	return flowResult{material: JSONObject{{"type", "api_key"}, {"key", key}, {"minted", true}}, label: "OpenRouter (minted key)", renewal: "none"}, nil
}
