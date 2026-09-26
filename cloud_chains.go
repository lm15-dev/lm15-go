package lm15

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/lm15-dev/lm15-go/internal/rs256"
	"github.com/lm15-dev/lm15-go/internal/sigv4"
)

// The three cloud credential chains as data over ten rung kinds
// (spec/auth.md AUTH-1 cloud chains, AUTH-11). Two entry points: Explain
// (the offline doctor walk) and CredentialProviderFor (the online AUTH-2
// provider with the AUTH-3 cache).

const (
	gcpScope            = "https://www.googleapis.com/auth/cloud-platform"
	gcpTokenURL         = "https://oauth2.googleapis.com/token"
	gcpSTSURL           = "https://sts.googleapis.com/v1/token"
	jwtBearerGrant      = "urn:ietf:params:oauth:grant-type:jwt-bearer"
	clientAssertionType = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
)

// HTTPFunc performs one HTTP call for a chain rung (never following redirects).
type HTTPFunc func(method, url string, headers map[string]string, body []byte, timeout time.Duration) (int, map[string]string, []byte, error)

// RunFunc runs a CLI and returns its stdout.
type RunFunc func(argv []string, timeout time.Duration) (string, error)

// ChainContext is everything a chain touches, injectable. HTTP / Run nil =
// offline (the doctor). Files overrides the filesystem (the harness).
type ChainContext struct {
	Env      map[string]string
	Home     string
	Files    map[string]string
	HTTP     HTTPFunc
	Run      RunFunc
	Now      func() time.Time
	Settings map[string]string
}

// OnlineChainContext builds a context over the real environment and network.
func OnlineChainContext(env map[string]string) *ChainContext {
	if env == nil {
		env = map[string]string{}
		for _, kv := range os.Environ() {
			if k, v, ok := strings.Cut(kv, "="); ok {
				env[k] = v
			}
		}
	}
	home := env["HOME"]
	if home == "" {
		home = homeDir()
	}
	return &ChainContext{Env: env, Home: home, HTTP: defaultChainHTTP, Run: func(argv []string, timeout time.Duration) (string, error) { return defaultChainRun(argv, timeout, env) }, Now: func() time.Time { return time.Now().UTC() }}
}

func defaultChainHTTP(method, rawURL string, headers map[string]string, body []byte, timeout time.Duration) (int, map[string]string, []byte, error) {
	client := &http.Client{Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	var reader io.Reader
	if body != nil {
		reader = strings.NewReader(string(body))
	}
	req, err := http.NewRequest(method, rawURL, reader)
	if err != nil {
		return 0, nil, nil, chainAuthError("credential HTTP request failed")
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, nil, chainAuthError("credential HTTP request failed")
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	out := map[string]string{}
	for k, v := range resp.Header {
		if len(v) > 0 {
			out[strings.ToLower(k)] = v[0]
		}
	}
	return resp.StatusCode, out, raw, nil
}

func defaultChainRun(argv []string, timeout time.Duration, env map[string]string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	var envList []string
	for k, v := range env {
		envList = append(envList, k+"="+v)
	}
	cmd.Env = envList
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", chainAuthError(fmt.Sprintf("credential command exited %d", exitErr.ExitCode()))
		}
		return "", chainAuthError("credential command failed")
	}
	return string(out), nil
}

func chainAuthError(msg string) *Error { return newError(KindAuth, msg) }

func (c *ChainContext) now() time.Time {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now().UTC()
}

func (c *ChainContext) offline() bool { return c.HTTP == nil }

func (c *ChainContext) path(text string) string {
	if strings.HasPrefix(text, "~") {
		return filepath.Join(c.Home, strings.TrimLeft(text[1:], "/\\"))
	}
	return text
}

func (c *ChainContext) read(text string) (string, bool) {
	if c.Files != nil {
		want := filepath.Clean(c.path(text))
		for key, content := range c.Files {
			if filepath.Clean(c.path(key)) == want {
				return content, true
			}
		}
		return "", false
	}
	raw, err := os.ReadFile(c.path(text))
	if err != nil {
		return "", false
	}
	return string(raw), true
}

func (c *ChainContext) exists(text string) bool {
	_, ok := c.read(text)
	return ok
}

func (c *ChainContext) onPath(command string) string {
	if strings.ContainsAny(command, `/\`) {
		if c.exists(command) {
			return command
		}
		return ""
	}
	pathEnv := c.Env["PATH"]
	if c.Files != nil {
		for _, dir := range strings.Split(pathEnv, string(os.PathListSeparator)) {
			if dir != "" && c.exists(strings.TrimRight(dir, "/")+"/"+command) {
				return strings.TrimRight(dir, "/") + "/" + command
			}
		}
		return ""
	}
	if pathEnv == "" {
		return ""
	}
	for _, dir := range strings.Split(pathEnv, string(os.PathListSeparator)) {
		candidate := filepath.Join(dir, command)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
		if runtime.GOOS == "windows" {
			if info, err := os.Stat(candidate + ".exe"); err == nil && !info.IsDir() {
				return candidate + ".exe"
			}
		}
	}
	return ""
}

// Rung is one step of a chain.
type Rung struct {
	Name    string // the fixture kind: "env:AWS_REGION", "assume-role", "imds", …
	Kind    string // RungKinds
	Source  string
	Needs   string                               // "" | "network" | "subprocess"
	Probe   func(*ChainContext) (string, string) // ("usable"|"configured"|"absent", detail)
	Acquire func(*ChainContext) (Credential, error)
}

// ChainStep is the doctor's view of one rung.
type ChainStep struct {
	Kind   string
	Source string
	Detail string
	State  string
}

// ─── Helpers ─────────────────────────────────────────────────────────

func jsonBody(raw []byte) JSONObject {
	data, err := DecodeJSON(raw)
	if err != nil {
		return JSONObject{}
	}
	if obj, ok := asObject(data); ok {
		return obj
	}
	return JSONObject{}
}

func formBody(pairs [][2]string) []byte {
	values := url.Values{}
	var order []string
	for _, p := range pairs {
		values.Add(p[0], p[1])
		order = append(order, p[0])
	}
	var parts []string
	for _, p := range pairs {
		parts = append(parts, url.QueryEscape(p[0])+"="+url.QueryEscape(p[1]))
	}
	_ = order
	return []byte(strings.Join(parts, "&"))
}

// oauthErrorWords: AUTH-21 (clarified 2026-09-24). From a failed
// auth-endpoint exchange only the status and, when the reply's error (or
// error.code / error.type) is one of these fixed words, that word. An error
// description can reflect the request (a refresh token, a signed
// assertion); a fixed word cannot.
var oauthErrorWords = map[string]bool{
	"invalid_request": true, "invalid_client": true, "invalid_grant": true, "unauthorized_client": true,
	"unsupported_grant_type": true, "invalid_scope": true, "access_denied": true, "server_error": true,
	"temporarily_unavailable": true, "authorization_pending": true, "slow_down": true, "expired_token": true,
}

func oauthErrorWord(data JSONObject) string {
	errValue := data.Get("error")
	candidates := []any{errValue}
	if obj := wireObj(errValue); obj != nil {
		candidates = []any{obj.Get("code"), obj.Get("type")}
	}
	for _, c := range candidates {
		if w, ok := c.(string); ok && oauthErrorWords[w] {
			return w
		}
	}
	return ""
}

func exchange(ctx *ChainContext, method, url string, headers map[string]string, body []byte, what string) (JSONObject, error) {
	return exchangeHint(ctx, method, url, headers, body, what, "")
}

// exchangeHint is one token-endpoint round trip whose refusal carries the
// status, a fixed OAuth word (AUTH-21) and, when the caller knows it, the
// one action that fixes it instead of the API-key guidance.
func exchangeHint(ctx *ChainContext, method, url string, headers map[string]string, body []byte, what, hint string) (JSONObject, error) {
	status, _, raw, err := ctx.HTTP(method, url, headers, body, 30*time.Second)
	if err != nil {
		return nil, err
	}
	if status < 200 || status >= 300 {
		data := jsonBody(raw)
		word := oauthErrorWord(data)
		msg := fmt.Sprintf("%s: HTTP %d", what, status)
		if word != "" {
			msg += " (" + word + ")"
		}
		var e *Error
		if hint != "" {
			e = AuthErrorf("", nil, hint, "%s", msg)
		} else {
			e = chainAuthError(msg)
		}
		e.ProviderCode = word
		return nil, e
	}
	return jsonBody(raw), nil
}

func expiresFrom(now time.Time, seconds any) *time.Time {
	f, err := jsonFloat64(seconds, "")
	if err != nil {
		if s, ok := seconds.(string); ok {
			if parsed, perr := strconv.ParseFloat(s, 64); perr == nil {
				t := now.Add(time.Duration(int64(parsed)) * time.Second)
				return &t
			}
		}
		return nil
	}
	t := now.Add(time.Duration(int64(f)) * time.Second)
	return &t
}

func bearerFromOAuth(data JSONObject, now time.Time, what string) (BearerToken, error) {
	token := stringOnly(data.Get("access_token"))
	if token == "" {
		return BearerToken{}, chainAuthError(what + ": no valid access_token in response")
	}
	var expires *time.Time
	if v := data.Get("expires_on"); v != nil && wireStr(v) != "" {
		if f, err := jsonFloat64(v, ""); err == nil {
			t := time.Unix(int64(f), 0).UTC()
			expires = &t
		} else if s, ok := v.(string); ok {
			if parsed, perr := strconv.ParseFloat(s, 64); perr == nil {
				t := time.Unix(int64(parsed), 0).UTC()
				expires = &t
			}
		}
	}
	if expires == nil {
		if v := data.Get("expires_in"); v != nil && wireStr(v) != "" {
			expires = expiresFrom(now, v)
		}
	}
	return BearerToken{Value: token, ExpiresAt: expires}, nil
}

// ─── INI (AWS shared files) ──────────────────────────────────────────

type iniFile map[string]map[string]string

func parseINI(text string) (iniFile, error) {
	out := iniFile{}
	section := ""
	var lastKey string
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimRight(line, "\r")
		if strings.TrimSpace(trimmed) == "" || strings.HasPrefix(strings.TrimSpace(trimmed), "#") || strings.HasPrefix(strings.TrimSpace(trimmed), ";") {
			continue
		}
		if strings.HasPrefix(trimmed, "[") {
			end := strings.Index(trimmed, "]")
			if end < 0 {
				return nil, fmt.Errorf("malformed section header")
			}
			section = strings.TrimSpace(trimmed[1:end])
			if _, ok := out[section]; !ok {
				out[section] = map[string]string{}
			}
			lastKey = ""
			continue
		}
		if section == "" {
			return nil, fmt.Errorf("key before any section")
		}
		if (strings.HasPrefix(trimmed, " ") || strings.HasPrefix(trimmed, "\t")) && lastKey != "" {
			out[section][lastKey] += "\n" + strings.TrimSpace(trimmed)
			continue
		}
		k, v, ok := strings.Cut(trimmed, "=")
		if !ok {
			return nil, fmt.Errorf("malformed key line")
		}
		lastKey = strings.ToLower(strings.TrimSpace(k))
		out[section][lastKey] = strings.TrimSpace(v)
	}
	return out, nil
}

func (f iniFile) section(name string) map[string]string {
	if s, ok := f[name]; ok {
		return s
	}
	return map[string]string{}
}

func (f iniFile) has(name string) bool { _, ok := f[name]; return ok }

// ─── AWS ─────────────────────────────────────────────────────────────

func awsConfig(ctx *ChainContext) (iniFile, iniFile, string, error) {
	profile := ctx.Env["AWS_PROFILE"]
	if profile == "" {
		profile = "default"
	}
	creds, conf := iniFile{}, iniFile{}
	credsPath := ctx.Env["AWS_SHARED_CREDENTIALS_FILE"]
	if credsPath == "" {
		credsPath = "~/.aws/credentials"
	}
	confPath := ctx.Env["AWS_CONFIG_FILE"]
	if confPath == "" {
		confPath = "~/.aws/config"
	}
	malformed := NotConfiguredErrorf("", nil, "", "malformed AWS profile configuration; check the AWS config and credentials files")
	if text, ok := ctx.read(credsPath); ok && text != "" {
		parsed, err := parseINI(text)
		if err != nil {
			return nil, nil, "", malformed
		}
		creds = parsed
	}
	if text, ok := ctx.read(confPath); ok && text != "" {
		parsed, err := parseINI(text)
		if err != nil {
			return nil, nil, "", malformed
		}
		conf = parsed
	}
	return creds, conf, profile, nil
}

func awsProfileSection(conf iniFile, profile string) map[string]string {
	name := profile
	if profile != "default" {
		name = "profile " + profile
	}
	if conf.has(name) {
		return conf.section(name)
	}
	if conf.has(profile) {
		return conf.section(profile)
	}
	return map[string]string{}
}

func awsStatic(section map[string]string) *AwsCredentials {
	key, secret := section["aws_access_key_id"], section["aws_secret_access_key"]
	if key == "" || secret == "" {
		return nil
	}
	return &AwsCredentials{AccessKeyID: key, SecretAccessKey: secret, SessionToken: section["aws_session_token"]}
}

func awsFromResponse(d JSONObject) (AwsCredentials, error) {
	var expires *time.Time
	raw := d.Get("Expiration")
	if raw == nil {
		raw = d.Get("expiration")
	}
	switch x := raw.(type) {
	case string:
		if t, err := ParseRFC3339(x); err == nil {
			expires = &t
		}
	case nil:
	default:
		if f, err := jsonFloat64(x, ""); err == nil {
			if f > 1e11 {
				f /= 1000
			}
			t := time.Unix(int64(f), 0).UTC()
			expires = &t
		}
	}
	key := firstStr(d.Get("AccessKeyId"), d.Get("accessKeyId"))
	secret := firstStr(d.Get("SecretAccessKey"), d.Get("secretAccessKey"))
	if key == "" || secret == "" {
		return AwsCredentials{}, chainAuthError("AWS credential response lacks access key id or secret access key")
	}
	return AwsCredentials{AccessKeyID: key, SecretAccessKey: secret, SessionToken: firstStr(d.Get("SessionToken"), d.Get("Token"), d.Get("sessionToken")), ExpiresAt: expires}, nil
}

type stsResponse struct {
	Credentials struct {
		AccessKeyId     string `xml:"AccessKeyId"`
		SecretAccessKey string `xml:"SecretAccessKey"`
		SessionToken    string `xml:"SessionToken"`
		Expiration      string `xml:"Expiration"`
	} `xml:"AssumeRoleResult>Credentials"`
	WebCredentials struct {
		AccessKeyId     string `xml:"AccessKeyId"`
		SecretAccessKey string `xml:"SecretAccessKey"`
		SessionToken    string `xml:"SessionToken"`
		Expiration      string `xml:"Expiration"`
	} `xml:"AssumeRoleWithWebIdentityResult>Credentials"`
}

func stsXMLCredentials(raw []byte) (AwsCredentials, error) {
	var parsed stsResponse
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return AwsCredentials{}, chainAuthError("STS: no Credentials in response")
	}
	c := parsed.Credentials
	if c.AccessKeyId == "" {
		c = parsed.WebCredentials
	}
	if c.AccessKeyId == "" {
		return AwsCredentials{}, chainAuthError("STS: no Credentials in response")
	}
	var expires *time.Time
	if c.Expiration != "" {
		if t, err := ParseRFC3339(c.Expiration); err == nil {
			expires = &t
		}
	}
	return AwsCredentials{AccessKeyID: c.AccessKeyId, SecretAccessKey: c.SecretAccessKey, SessionToken: c.SessionToken, ExpiresAt: expires}, nil
}

func envAws(ctx *ChainContext) *AwsCredentials {
	key, secret := ctx.Env["AWS_ACCESS_KEY_ID"], ctx.Env["AWS_SECRET_ACCESS_KEY"]
	if key == "" || secret == "" {
		return nil
	}
	return &AwsCredentials{AccessKeyID: key, SecretAccessKey: secret, SessionToken: ctx.Env["AWS_SESSION_TOKEN"]}
}

func awsRegion(ctx *ChainContext, section map[string]string) string {
	for _, v := range []string{ctx.Settings["region"], ctx.Env["AWS_REGION"], ctx.Env["AWS_DEFAULT_REGION"], section["region"]} {
		if v != "" {
			return v
		}
	}
	return "us-east-1"
}

func awsSourceCredentials(ctx *ChainContext, section map[string]string, depth int) (AwsCredentials, error) {
	if depth > 5 {
		return AwsCredentials{}, chainAuthError("assume-role: source_profile chain too deep")
	}
	if sourceProfile := section["source_profile"]; sourceProfile != "" {
		creds, conf, _, err := awsConfig(ctx)
		if err != nil {
			return AwsCredentials{}, err
		}
		sub := map[string]string{}
		for k, v := range awsProfileSection(conf, sourceProfile) {
			sub[k] = v
		}
		if creds.has(sourceProfile) {
			for k, v := range creds.section(sourceProfile) {
				sub[k] = v
			}
		}
		if sub["role_arn"] != "" {
			return assumeRole(ctx, sub, depth+1)
		}
		static := awsStatic(sub)
		if static == nil {
			return AwsCredentials{}, NotConfiguredErrorf("", nil, "", "assume-role: source_profile %q has no keys", sourceProfile)
		}
		return *static, nil
	}
	switch section["credential_source"] {
	case "Environment":
		if static := envAws(ctx); static != nil {
			return *static, nil
		}
		return AwsCredentials{}, NotConfiguredErrorf("", nil, "", "assume-role: credential_source=Environment but AWS_ACCESS_KEY_ID is not set")
	case "EcsContainer":
		got, err := containerAcquire(ctx)
		if err != nil {
			return AwsCredentials{}, err
		}
		if got == nil {
			return AwsCredentials{}, NotConfiguredErrorf("", nil, "", "assume-role: credential_source=EcsContainer but no container endpoint is configured")
		}
		return got.(AwsCredentials), nil
	case "Ec2InstanceMetadata":
		got, err := imdsAcquire(ctx)
		if err != nil {
			return AwsCredentials{}, err
		}
		if got == nil {
			return AwsCredentials{}, NotConfiguredErrorf("", nil, "", "assume-role: credential_source=Ec2InstanceMetadata but IMDS answered nothing")
		}
		return got.(AwsCredentials), nil
	}
	return AwsCredentials{}, NotConfiguredErrorf("", nil, "", "assume-role: profile needs source_profile or credential_source")
}

func assumeRole(ctx *ChainContext, section map[string]string, depth int) (AwsCredentials, error) {
	source, err := awsSourceCredentials(ctx, section, depth)
	if err != nil {
		return AwsCredentials{}, err
	}
	region := awsRegion(ctx, section)
	stsURL := "https://sts." + region + ".amazonaws.com/"
	sessionName := section["role_session_name"]
	if sessionName == "" {
		sessionName = "lm15-" + randomHex(6)
	}
	pairs := [][2]string{{"Action", "AssumeRole"}, {"Version", "2011-06-15"}, {"RoleArn", section["role_arn"]}, {"RoleSessionName", sessionName}}
	if v := section["external_id"]; v != "" {
		pairs = append(pairs, [2]string{"ExternalId", v})
	}
	if v := section["duration_seconds"]; v != "" {
		pairs = append(pairs, [2]string{"DurationSeconds", v})
	}
	body := formBody(pairs)
	sig := sigv4.Sign("POST", stsURL, map[string]string{"content-type": "application/x-www-form-urlencoded"}, body, sigv4.Credentials{AccessKeyID: source.AccessKeyID, SecretAccessKey: source.SecretAccessKey, SessionToken: source.SessionToken}, region, "sts", ctx.now())
	status, _, raw, err := ctx.HTTP("POST", stsURL, sig.Headers, body, 30*time.Second)
	if err != nil {
		return AwsCredentials{}, err
	}
	if status >= 400 {
		return AwsCredentials{}, chainAuthError(fmt.Sprintf("STS AssumeRole: HTTP %d", status))
	}
	return stsXMLCredentials(raw)
}

func webIdentityConfig(ctx *ChainContext) (tokenFile, role, session string, ok bool, err error) {
	tokenFile, role, session = ctx.Env["AWS_WEB_IDENTITY_TOKEN_FILE"], ctx.Env["AWS_ROLE_ARN"], ctx.Env["AWS_ROLE_SESSION_NAME"]
	if tokenFile != "" && role != "" {
		return tokenFile, role, session, true, nil
	}
	_, conf, profile, err := awsConfig(ctx)
	if err != nil {
		return "", "", "", false, err
	}
	section := awsProfileSection(conf, profile)
	tokenFile, role, session = section["web_identity_token_file"], section["role_arn"], section["role_session_name"]
	if tokenFile != "" && role != "" && section["source_profile"] == "" && section["credential_source"] == "" {
		return tokenFile, role, session, true, nil
	}
	return "", "", "", false, nil
}

func webIdentityAcquire(ctx *ChainContext) (Credential, error) {
	tokenFile, role, session, ok, err := webIdentityConfig(ctx)
	if err != nil || !ok {
		return nil, err
	}
	token, found := ctx.read(tokenFile)
	if !found {
		return nil, NotConfiguredErrorf("", nil, "", "web identity token file %s is unreadable", tokenFile)
	}
	region := awsRegion(ctx, nil)
	if session == "" {
		session = "lm15-" + randomHex(6)
	}
	body := formBody([][2]string{{"Action", "AssumeRoleWithWebIdentity"}, {"Version", "2011-06-15"}, {"RoleArn", role}, {"RoleSessionName", session}, {"WebIdentityToken", strings.TrimSpace(token)}})
	status, _, raw, err := ctx.HTTP("POST", "https://sts."+region+".amazonaws.com/", map[string]string{"content-type": "application/x-www-form-urlencoded"}, body, 30*time.Second)
	if err != nil {
		return nil, err
	}
	if status >= 400 {
		return nil, chainAuthError(fmt.Sprintf("STS AssumeRoleWithWebIdentity: HTTP %d", status))
	}
	return stsXMLCredentials(raw)
}

func ssoConfig(ctx *ChainContext) (map[string]string, error) {
	_, conf, profile, err := awsConfig(ctx)
	if err != nil {
		return nil, err
	}
	section := map[string]string{}
	for k, v := range awsProfileSection(conf, profile) {
		section[k] = v
	}
	if name := section["sso_session"]; name != "" {
		sess := conf.section("sso-session " + name)
		if sess["sso_start_url"] == "" {
			return nil, nil
		}
		out := map[string]string{}
		for k, v := range sess {
			out[k] = v
		}
		for k, v := range section {
			out[k] = v
		}
		sum := sha1.Sum([]byte(name))
		out["cache_key"] = hex.EncodeToString(sum[:])
		out["session_name"] = name
		return out, nil
	}
	if section["sso_start_url"] != "" {
		sum := sha1.Sum([]byte(section["sso_start_url"]))
		section["cache_key"] = hex.EncodeToString(sum[:])
		return section, nil
	}
	return nil, nil
}

func ssoAcquire(ctx *ChainContext) (Credential, error) {
	cfg, err := ssoConfig(ctx)
	if err != nil || cfg == nil {
		return nil, err
	}
	raw, ok := ctx.read("~/.aws/sso/cache/" + cfg["cache_key"] + ".json")
	if !ok {
		return nil, NotConfiguredErrorf("", nil, "aws sso login", "IAM Identity Center: no cached token; run `aws sso login`")
	}
	token := jsonBody([]byte(raw))
	now := ctx.now()
	var expires *time.Time
	if s := stringOnly(token.Get("expiresAt")); s != "" {
		if t, err := ParseRFC3339(s); err == nil {
			expires = &t
		}
	}
	access := stringOnly(token.Get("accessToken"))
	ssoRegion := cfg["sso_region"]
	if ssoRegion == "" {
		ssoRegion = "us-east-1"
	}
	if access == "" || (expires != nil && expires.Sub(now) <= expirySkew) {
		if stringOnly(token.Get("refreshToken")) == "" || stringOnly(token.Get("clientId")) == "" || stringOnly(token.Get("clientSecret")) == "" {
			return nil, NotConfiguredErrorf("", nil, "aws sso login", "IAM Identity Center: token expired and not refreshable; run `aws sso login`")
		}
		data, err := exchange(ctx, "POST", "https://oidc."+ssoRegion+".amazonaws.com/token", map[string]string{"content-type": "application/json"},
			mustJSON(JSONObject{{"clientId", token.Get("clientId")}, {"clientSecret", token.Get("clientSecret")}, {"grantType", "refresh_token"}, {"refreshToken", token.Get("refreshToken")}}), "sso-oidc CreateToken")
		if err != nil {
			return nil, err
		}
		access = stringOnly(data.Get("accessToken"))
		if access == "" {
			return nil, chainAuthError("sso-oidc CreateToken: no accessToken")
		}
	}
	account, role := cfg["sso_account_id"], cfg["sso_role_name"]
	if account == "" || role == "" {
		return nil, NotConfiguredErrorf("", nil, "", "IAM Identity Center: profile needs sso_account_id and sso_role_name")
	}
	query := url.Values{"role_name": {role}, "account_id": {account}}.Encode()
	status, _, rawCreds, err := ctx.HTTP("GET", "https://portal.sso."+ssoRegion+".amazonaws.com/federation/credentials?"+query, map[string]string{"x-amz-sso_bearer_token": access}, nil, 30*time.Second)
	if err != nil {
		return nil, err
	}
	if status >= 400 {
		return nil, chainAuthError(fmt.Sprintf("sso GetRoleCredentials: HTTP %d", status))
	}
	return awsFromResponse(wireObj(jsonBody(rawCreds).Get("roleCredentials")))
}

func loginConfig(ctx *ChainContext) (string, error) {
	_, conf, profile, err := awsConfig(ctx)
	if err != nil {
		return "", err
	}
	return awsProfileSection(conf, profile)["login_session"], nil
}

func loginCached(ctx *ChainContext) (*AwsCredentials, error) {
	session, err := loginConfig(ctx)
	if err != nil || session == "" {
		return nil, err
	}
	dir := ctx.Env["AWS_LOGIN_CACHE_DIRECTORY"]
	if dir == "" {
		dir = "~/.aws/login/cache"
	}
	sum := sha256.Sum256([]byte(session))
	raw, ok := ctx.read(dir + "/" + hex.EncodeToString(sum[:]) + ".json")
	if !ok {
		return nil, nil
	}
	token := wireObj(jsonBody([]byte(raw)).Get("accessToken"))
	if stringOnly(token.Get("accessKeyId")) == "" {
		return nil, nil
	}
	creds, err := awsFromResponse(JSONObject{{"AccessKeyId", token.Get("accessKeyId")}, {"SecretAccessKey", token.Get("secretAccessKey")}, {"SessionToken", token.Get("sessionToken")}, {"Expiration", token.Get("expiresAt")}})
	if err != nil {
		return nil, err
	}
	return &creds, nil
}

func loginAcquire(ctx *ChainContext) (Credential, error) {
	session, err := loginConfig(ctx)
	if err != nil || session == "" {
		return nil, err
	}
	cached, err := loginCached(ctx)
	if err != nil {
		return nil, err
	}
	if cached != nil && !cached.IsExpired(ctx.now()) {
		return *cached, nil
	}
	return nil, NotConfiguredErrorf("", nil, "aws login", "AWS login session expired; run `aws login`")
}

func processAcquire(ctx *ChainContext) (Credential, error) {
	_, conf, profile, err := awsConfig(ctx)
	if err != nil {
		return nil, err
	}
	command := awsProfileSection(conf, profile)["credential_process"]
	if command == "" || ctx.Run == nil {
		return nil, nil
	}
	out, err := ctx.Run(shellSplit(command), 60*time.Second)
	if err != nil {
		return nil, err
	}
	data := jsonBody([]byte(out))
	if wireInt(data.Get("Version"), 0) != 1 {
		return nil, chainAuthError("credential_process: output Version must be 1")
	}
	return awsFromResponse(data)
}

// shellSplit is a small POSIX-ish splitter (quotes and backslashes).
func shellSplit(command string) []string {
	var out []string
	var cur strings.Builder
	inSingle, inDouble, escaped, has := false, false, false, false
	for _, r := range command {
		switch {
		case escaped:
			cur.WriteRune(r)
			escaped = false
		case r == '\\' && !inSingle:
			escaped = true
		case r == '\'' && !inDouble:
			inSingle = !inSingle
			has = true
		case r == '"' && !inSingle:
			inDouble = !inDouble
			has = true
		case (r == ' ' || r == '\t' || r == '\n') && !inSingle && !inDouble:
			if has || cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
				has = false
			}
		default:
			cur.WriteRune(r)
			has = true
		}
	}
	if has || cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

var containerAllowed = map[string]bool{"169.254.170.2": true, "169.254.170.23": true, "fd00:ec2::23": true, "localhost": true}

func containerConfig(ctx *ChainContext) (string, error) {
	rel, full := ctx.Env["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"], ctx.Env["AWS_CONTAINER_CREDENTIALS_FULL_URI"]
	if rel != "" {
		if !strings.HasPrefix(rel, "/") || strings.HasPrefix(rel, "//") || strings.ContainsAny(rel, "\\\r\n\t#") {
			return "", NotConfiguredErrorf("", nil, "", "container credentials relative URI must be an absolute path")
		}
		return "http://169.254.170.2" + rel, nil
	}
	if full != "" {
		u, err := url.Parse(full)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.User != nil || u.Fragment != "" {
			return "", NotConfiguredErrorf("", nil, "", "container credentials URI must be HTTP(S), without userinfo or fragment")
		}
		host := u.Hostname()
		loopback := false
		if ip := net.ParseIP(host); ip != nil {
			loopback = ip.IsLoopback()
		}
		if u.Scheme != "https" && !loopback && !containerAllowed[host] {
			var allowed []string
			for k := range containerAllowed {
				allowed = append(allowed, k)
			}
			sort.Strings(allowed)
			return "", NotConfiguredErrorf("", nil, "", "Unsupported host %q. Can only retrieve metadata from a loopback address or one of these hosts: %s", host, strings.Join(allowed, ", "))
		}
		return full, nil
	}
	return "", nil
}

func containerAcquire(ctx *ChainContext) (Credential, error) {
	u, err := containerConfig(ctx)
	if err != nil || u == "" || ctx.HTTP == nil {
		return nil, err
	}
	headers := map[string]string{}
	token := ctx.Env["AWS_CONTAINER_AUTHORIZATION_TOKEN"]
	if tokenFile := ctx.Env["AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"]; tokenFile != "" && token == "" {
		if t, ok := ctx.read(tokenFile); ok {
			token = strings.TrimSpace(t)
		}
	}
	if token != "" {
		headers["Authorization"] = token
	}
	status, _, raw, err := ctx.HTTP("GET", u, headers, nil, 5*time.Second)
	if err != nil {
		return nil, err
	}
	if status >= 400 {
		return nil, chainAuthError(fmt.Sprintf("container credentials: HTTP %d", status))
	}
	return awsFromResponse(jsonBody(raw))
}

func imdsDisabled(ctx *ChainContext) bool {
	return strings.ToLower(strings.TrimSpace(ctx.Env["AWS_EC2_METADATA_DISABLED"])) == "true"
}

func imdsAcquire(ctx *ChainContext) (Credential, error) {
	if imdsDisabled(ctx) || ctx.HTTP == nil {
		return nil, nil
	}
	base := ctx.Env["AWS_EC2_METADATA_SERVICE_ENDPOINT"]
	if base == "" {
		base = "http://169.254.169.254"
		if strings.ToLower(ctx.Env["AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE"]) == "ipv6" {
			base = "http://[fd00:ec2::254]"
		}
	}
	base = strings.TrimRight(base, "/")
	status, _, tok, err := ctx.HTTP("PUT", base+"/latest/api/token", map[string]string{"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, nil, time.Second)
	if err != nil || status != 200 {
		return nil, nil
	}
	headers := map[string]string{"X-aws-ec2-metadata-token": string(tok)}
	status, _, role, err := ctx.HTTP("GET", base+"/latest/meta-data/iam/security-credentials/", headers, nil, time.Second)
	if err != nil || status != 200 || strings.TrimSpace(string(role)) == "" {
		return nil, nil
	}
	roleName := strings.Split(strings.TrimSpace(string(role)), "\n")[0]
	status, _, raw, err := ctx.HTTP("GET", base+"/latest/meta-data/iam/security-credentials/"+roleName, headers, nil, time.Second)
	if err != nil || status != 200 {
		return nil, nil
	}
	data := jsonBody(raw)
	if code := data.Get("Code"); code != nil && wireStr(code) != "Success" {
		return nil, chainAuthError("IMDS rejected the credential request")
	}
	return awsFromResponse(data)
}

func envProbe(varName string) func(*ChainContext) (string, string) {
	return func(ctx *ChainContext) (string, string) {
		if ctx.Env[varName] != "" {
			return "usable", "set (value never shown)"
		}
		return "absent", "not set"
	}
}

func awsChain(policy AccessPolicy) []Rung {
	var rungs []Rung
	if len(policy.EnvKeys) > 0 {
		doorKey := policy.EnvKeys[0]
		rungs = append(rungs, Rung{Name: "env:" + doorKey, Kind: "env", Source: "env $" + doorKey, Probe: envProbe(doorKey), Acquire: func(ctx *ChainContext) (Credential, error) {
			v := ctx.Env[doorKey]
			if v == "" {
				return nil, nil
			}
			if doorKey == "AWS_BEARER_TOKEN_BEDROCK" {
				return BearerToken{Value: v}, nil
			}
			return APIKey{Value: v}, nil
		}})
	}
	rungs = append(rungs,
		Rung{Name: "env:AWS_ACCESS_KEY_ID", Kind: "env", Source: "env $AWS_ACCESS_KEY_ID (+SECRET, +SESSION_TOKEN)", Probe: func(ctx *ChainContext) (string, string) {
			if envAws(ctx) != nil {
				return "usable", "set (values never shown)"
			}
			return "absent", "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set"
		}, Acquire: func(ctx *ChainContext) (Credential, error) {
			if c := envAws(ctx); c != nil {
				return *c, nil
			}
			return nil, nil
		}},
		Rung{Name: "assume-role", Kind: "sigv4-sts", Source: "profile assume-role via STS", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			_, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			s := awsProfileSection(conf, profile)
			if s["role_arn"] != "" && (s["source_profile"] != "" || s["credential_source"] != "") {
				return "configured", fmt.Sprintf("profile %q assumes %s (STS call at request time)", profile, s["role_arn"])
			}
			return "absent", fmt.Sprintf("profile %q has no role_arn with a source", profile)
		}, Acquire: func(ctx *ChainContext) (Credential, error) {
			_, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return nil, err
			}
			s := awsProfileSection(conf, profile)
			if s["role_arn"] != "" && (s["source_profile"] != "" || s["credential_source"] != "") {
				return assumeRole(ctx, s, 0)
			}
			return nil, nil
		}},
		Rung{Name: "web-identity", Kind: "unsigned-sts", Source: "web identity via STS", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			tokenFile, role, _, ok, err := webIdentityConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			if ok {
				return "configured", fmt.Sprintf("token file %s → %s (STS call at request time)", tokenFile, role)
			}
			return "absent", "AWS_WEB_IDENTITY_TOKEN_FILE / AWS_ROLE_ARN not set"
		}, Acquire: webIdentityAcquire},
		Rung{Name: "sso", Kind: "file-cache", Source: "IAM Identity Center (~/.aws/sso/cache)", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			cfg, err := ssoConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			if cfg == nil {
				return "absent", "no sso_session / sso_start_url in the profile"
			}
			if ctx.exists("~/.aws/sso/cache/" + cfg["cache_key"] + ".json") {
				return "configured", "cached token present (GetRoleCredentials at request time)"
			}
			return "configured", "no cached token; run `aws sso login`"
		}, Acquire: ssoAcquire},
		Rung{Name: "shared-credentials-file", Kind: "ini-profile", Source: "~/.aws/credentials", Probe: func(ctx *ChainContext) (string, string) {
			creds, _, profile, err := awsConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			if creds.has(profile) && awsStatic(creds.section(profile)) != nil {
				return "usable", fmt.Sprintf("profile %q (values never shown)", profile)
			}
			return "absent", fmt.Sprintf("no keys for profile %q", profile)
		}, Acquire: func(ctx *ChainContext) (Credential, error) {
			creds, _, profile, err := awsConfig(ctx)
			if err != nil {
				return nil, err
			}
			if creds.has(profile) {
				if c := awsStatic(creds.section(profile)); c != nil {
					return *c, nil
				}
			}
			return nil, nil
		}},
		Rung{Name: "login", Kind: "file-cache", Source: "aws login session (~/.aws/login/cache)", Probe: func(ctx *ChainContext) (string, string) {
			session, err := loginConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			if session == "" {
				return "absent", "no login_session in the profile"
			}
			cached, _ := loginCached(ctx)
			if cached != nil && !cached.IsExpired(ctx.now()) {
				return "usable", "cached short-term credentials are fresh"
			}
			return "configured", "cached credentials missing or expired; refresh needs `aws login`"
		}, Acquire: loginAcquire},
		Rung{Name: "credential_process", Kind: "subprocess", Source: "profile credential_process", Needs: "subprocess", Probe: func(ctx *ChainContext) (string, string) {
			_, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			cmd := awsProfileSection(conf, profile)["credential_process"]
			if cmd == "" {
				return "absent", "no credential_process in the profile"
			}
			argv := shellSplit(cmd)
			if len(argv) == 0 || ctx.onPath(argv[0]) == "" {
				exe := ""
				if len(argv) > 0 {
					exe = argv[0]
				}
				return "absent", fmt.Sprintf("credential_process %q is not on PATH", exe)
			}
			return "configured", "credential_process configured (run at request time)"
		}, Acquire: processAcquire},
		Rung{Name: "config-file", Kind: "ini-profile", Source: "~/.aws/config static keys", Probe: func(ctx *ChainContext) (string, string) {
			_, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			if awsStatic(awsProfileSection(conf, profile)) != nil {
				return "usable", fmt.Sprintf("static keys in config for profile %q", profile)
			}
			return "absent", "no static keys in config"
		}, Acquire: func(ctx *ChainContext) (Credential, error) {
			_, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return nil, err
			}
			if c := awsStatic(awsProfileSection(conf, profile)); c != nil {
				return *c, nil
			}
			return nil, nil
		}},
		Rung{Name: "container", Kind: "http-metadata", Source: "container credentials endpoint", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			u, err := containerConfig(ctx)
			if err != nil {
				return "absent", firstLine(err)
			}
			if u != "" {
				return "configured", "container endpoint configured (HTTP at request time)"
			}
			return "absent", "no AWS_CONTAINER_CREDENTIALS_* URI"
		}, Acquire: containerAcquire},
		Rung{Name: "imds", Kind: "http-metadata", Source: "EC2 instance metadata (IMDSv2)", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			if imdsDisabled(ctx) {
				return "absent", "AWS_EC2_METADATA_DISABLED=true"
			}
			return "configured", "instance metadata probed at request time"
		}, Acquire: imdsAcquire},
	)
	return rungs
}

func firstLine(err error) string {
	msg := err.Error()
	if e := AsError(err); e != nil {
		msg = e.Message
	}
	line, _, _ := strings.Cut(msg, "\n")
	return line
}

// ─── Azure ───────────────────────────────────────────────────────────

func azureAuthority(ctx *ChainContext) string {
	for _, v := range []string{ctx.Settings["authority_host"], ctx.Env["AZURE_AUTHORITY_HOST"]} {
		if v != "" {
			return strings.TrimRight(v, "/")
		}
	}
	return "https://login.microsoftonline.com"
}

func azureScope(ctx *ChainContext) string {
	if v := ctx.Settings["scope"]; v != "" {
		return v
	}
	return "https://ai.azure.com/.default"
}

func azureTokenURL(ctx *ChainContext, tenant string) string {
	return azureAuthority(ctx) + "/" + tenant + "/oauth2/v2.0/token"
}

// AzureCertificateAssertion is the Entra client assertion (RS256, x5t).
func AzureCertificateAssertion(ctx *ChainContext, tenant, clientID, pem, jti string, sendChain bool) (string, error) {
	key, err := rs256.LoadPrivateKey(pem)
	if err != nil {
		return "", NotConfiguredErrorf("", nil, "", "%s", err.Error())
	}
	der, err := rs256.CertificateDER(pem)
	if err != nil {
		return "", NotConfiguredErrorf("", nil, "", "%s", err.Error())
	}
	now := ctx.now().Unix()
	sum := sha1.Sum(der)
	var header rs256.OrderedObject
	header.Set("alg", "RS256")
	header.Set("typ", "JWT")
	header.Set("x5t", rs256.B64URL(sum[:]))
	if sendChain {
		header.Set("x5c", []string{base64.StdEncoding.EncodeToString(der)})
	}
	if jti == "" {
		jti = uuidV4()
	}
	var payload rs256.OrderedObject
	payload.Set("aud", azureTokenURL(ctx, tenant))
	payload.Set("iss", clientID)
	payload.Set("sub", clientID)
	payload.Set("exp", now+600)
	payload.Set("iat", now)
	payload.Set("jti", jti)
	return rs256.JWTEncode(header, payload, key)
}

func uuidV4() string {
	h := randomHex(16)
	return h[0:8] + "-" + h[8:12] + "-4" + h[13:16] + "-" + h[16:20] + "-" + h[20:32]
}

func azureEnvironmentKind(ctx *ChainContext) string {
	if ctx.Env["AZURE_TENANT_ID"] == "" || ctx.Env["AZURE_CLIENT_ID"] == "" {
		return ""
	}
	if ctx.Env["AZURE_CLIENT_SECRET"] != "" {
		return "secret"
	}
	if ctx.Env["AZURE_CLIENT_CERTIFICATE_PATH"] != "" {
		return "certificate"
	}
	return ""
}

// AzureEnvironmentRequest is the (token URL, form pairs) for the env service principal.
func AzureEnvironmentRequest(ctx *ChainContext, jti string) (string, [][2]string, error) {
	kind := azureEnvironmentKind(ctx)
	tenant, client := ctx.Env["AZURE_TENANT_ID"], ctx.Env["AZURE_CLIENT_ID"]
	tokenURL := azureTokenURL(ctx, tenant)
	scope := azureScope(ctx)
	switch kind {
	case "secret":
		return tokenURL, [][2]string{{"client_id", client}, {"scope", scope}, {"client_secret", ctx.Env["AZURE_CLIENT_SECRET"]}, {"grant_type", "client_credentials"}}, nil
	case "certificate":
		certPath := ctx.Env["AZURE_CLIENT_CERTIFICATE_PATH"]
		pem, ok := ctx.read(certPath)
		if !ok {
			return "", nil, NotConfiguredErrorf("", nil, "", "AZURE_CLIENT_CERTIFICATE_PATH %s is unreadable", certPath)
		}
		if ctx.Env["AZURE_CLIENT_CERTIFICATE_PASSWORD"] != "" {
			return "", nil, NotConfiguredErrorf("", nil, "openssl pkey -in cert.pem -out cert-plain.pem", "password-protected certificates are not supported; decrypt with `openssl pkey`")
		}
		sendChain := strings.ToLower(ctx.Env["AZURE_CLIENT_SEND_CERTIFICATE_CHAIN"]) == "1" || strings.ToLower(ctx.Env["AZURE_CLIENT_SEND_CERTIFICATE_CHAIN"]) == "true"
		assertion, err := AzureCertificateAssertion(ctx, tenant, client, pem, jti, sendChain)
		if err != nil {
			return "", nil, err
		}
		return tokenURL, [][2]string{{"client_id", client}, {"scope", scope}, {"client_assertion_type", clientAssertionType}, {"client_assertion", assertion}, {"grant_type", "client_credentials"}}, nil
	}
	return "", nil, NotConfiguredErrorf("", nil, "", "Azure environment credential needs AZURE_CLIENT_SECRET or AZURE_CLIENT_CERTIFICATE_PATH")
}

func azureEnvironmentAcquire(ctx *ChainContext) (Credential, error) {
	if azureEnvironmentKind(ctx) == "" {
		return nil, nil
	}
	tokenURL, pairs, err := AzureEnvironmentRequest(ctx, "")
	if err != nil {
		return nil, err
	}
	data, err := exchange(ctx, "POST", tokenURL, map[string]string{"content-type": "application/x-www-form-urlencoded"}, formBody(pairs), "Entra client credentials")
	if err != nil {
		return nil, err
	}
	return bearerFromOAuth(data, ctx.now(), "Entra")
}

func azureWorkloadConfig(ctx *ChainContext) bool {
	return ctx.Env["AZURE_FEDERATED_TOKEN_FILE"] != "" && ctx.Env["AZURE_CLIENT_ID"] != "" && ctx.Env["AZURE_TENANT_ID"] != ""
}

func azureWorkloadAcquire(ctx *ChainContext) (Credential, error) {
	if !azureWorkloadConfig(ctx) {
		return nil, nil
	}
	token, ok := ctx.read(ctx.Env["AZURE_FEDERATED_TOKEN_FILE"])
	if !ok {
		return nil, NotConfiguredErrorf("", nil, "", "AZURE_FEDERATED_TOKEN_FILE %s is unreadable", ctx.Env["AZURE_FEDERATED_TOKEN_FILE"])
	}
	pairs := [][2]string{{"client_id", ctx.Env["AZURE_CLIENT_ID"]}, {"scope", azureScope(ctx)}, {"client_assertion_type", clientAssertionType}, {"client_assertion", strings.TrimSpace(token)}, {"grant_type", "client_credentials"}}
	data, err := exchange(ctx, "POST", azureTokenURL(ctx, ctx.Env["AZURE_TENANT_ID"]), map[string]string{"content-type": "application/x-www-form-urlencoded"}, formBody(pairs), "Entra workload identity")
	if err != nil {
		return nil, err
	}
	return bearerFromOAuth(data, ctx.now(), "Entra")
}

func azureMSIFlavor(ctx *ChainContext) string {
	e := ctx.Env
	if e["IDENTITY_ENDPOINT"] != "" {
		if e["IDENTITY_HEADER"] != "" {
			if e["IDENTITY_SERVER_THUMBPRINT"] != "" {
				return "service-fabric"
			}
			return "app-service"
		}
		if e["IMDS_ENDPOINT"] != "" {
			return "azure-arc"
		}
	}
	if e["MSI_ENDPOINT"] != "" {
		if e["MSI_SECRET"] != "" {
			return "azure-ml"
		}
		return "cloud-shell"
	}
	return "imds"
}

func azureMSIAcquire(ctx *ChainContext) (Credential, error) {
	if ctx.HTTP == nil {
		return nil, nil
	}
	e := ctx.Env
	resource := strings.TrimSuffix(azureScope(ctx), "/.default")
	clientID := e["AZURE_CLIENT_ID"]
	now := ctx.now()
	switch azureMSIFlavor(ctx) {
	case "imds":
		q := url.Values{"api-version": {"2018-02-01"}, "resource": {resource}}
		if clientID != "" {
			q.Set("client_id", clientID)
		}
		status, _, raw, err := ctx.HTTP("GET", "http://169.254.169.254/metadata/identity/oauth2/token?"+q.Encode(), map[string]string{"Metadata": "true"}, nil, time.Second)
		if err != nil || status != 200 {
			return nil, nil
		}
		return bearerFromOAuth(jsonBody(raw), now, "managed identity")
	case "app-service":
		q := url.Values{"api-version": {"2019-08-01"}, "resource": {resource}}
		if clientID != "" {
			q.Set("client_id", clientID)
		}
		data, err := exchange(ctx, "GET", e["IDENTITY_ENDPOINT"]+"?"+q.Encode(), map[string]string{"X-IDENTITY-HEADER": e["IDENTITY_HEADER"]}, nil, "App Service managed identity")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "managed identity")
	case "cloud-shell":
		data, err := exchange(ctx, "POST", e["MSI_ENDPOINT"], map[string]string{"Metadata": "true", "content-type": "application/x-www-form-urlencoded"}, formBody([][2]string{{"resource", resource}}), "Cloud Shell managed identity")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "managed identity")
	case "azure-ml":
		q := url.Values{"api-version": {"2017-09-01"}, "resource": {resource}}
		if clientID != "" {
			q.Set("clientid", clientID)
		}
		data, err := exchange(ctx, "GET", e["MSI_ENDPOINT"]+"?"+q.Encode(), map[string]string{"secret": e["MSI_SECRET"]}, nil, "Azure ML managed identity")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "managed identity")
	case "azure-arc":
		u := e["IDENTITY_ENDPOINT"] + "?" + url.Values{"api-version": {"2019-11-01"}, "resource": {resource}}.Encode()
		status, headers, _, err := ctx.HTTP("GET", u, map[string]string{"Metadata": "true"}, nil, 5*time.Second)
		if err != nil {
			return nil, err
		}
		challenge := headers["www-authenticate"]
		if status != 401 || !strings.Contains(challenge, "realm=") {
			return nil, chainAuthError(fmt.Sprintf("Azure Arc managed identity: expected a 401 challenge, got %d", status))
		}
		keyPath := strings.Trim(strings.TrimSpace(strings.SplitN(challenge, "realm=", 2)[1]), "\"")
		dir := "/var/opt/azcmagent/tokens"
		if runtime.GOOS == "windows" {
			pd := e["PROGRAMDATA"]
			if pd == "" {
				pd = "C:/ProgramData"
			}
			dir = filepath.Join(pd, "AzureConnectedMachineAgent", "Tokens")
		}
		if filepath.Clean(filepath.Dir(keyPath)) != filepath.Clean(dir) || filepath.Ext(keyPath) != ".key" {
			return nil, chainAuthError("Azure Arc managed identity: invalid challenge file location")
		}
		secret, ok := ctx.read(keyPath)
		if !ok || len(secret) > 4096 {
			return nil, chainAuthError("Azure Arc managed identity: challenge file missing or too large")
		}
		data, err := exchange(ctx, "GET", u, map[string]string{"Metadata": "true", "Authorization": "Basic " + strings.TrimSpace(secret)}, nil, "Azure Arc managed identity")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "managed identity")
	}
	return nil, NotConfiguredErrorf("", nil, "", "Service Fabric managed identity (TLS thumbprint pinning) is not supported; use a certificate or secret")
}

func azCLIAcquire(ctx *ChainContext) (Credential, error) {
	if ctx.Run == nil || ctx.onPath("az") == "" {
		return nil, nil
	}
	argv := []string{"az", "account", "get-access-token", "--output", "json", "--scope", azureScope(ctx)}
	if tenant := ctx.Env["AZURE_TENANT_ID"]; tenant != "" {
		argv = append(argv, "--tenant", tenant)
	}
	out, err := ctx.Run(argv, 30*time.Second)
	if err != nil {
		return nil, err
	}
	data := jsonBody([]byte(out))
	if data.Get("accessToken") == nil {
		return nil, nil
	}
	parsed, err := bearerFromOAuth(JSONObject{{"access_token", data.Get("accessToken")}, {"expires_on", data.Get("expires_on")}}, ctx.now(), "Azure CLI")
	if err != nil {
		return nil, err
	}
	if parsed.ExpiresAt == nil && data.Get("expiresOn") != nil {
		if t, err := ParseRFC3339(wireStr(data.Get("expiresOn"))); err == nil {
			parsed.ExpiresAt = &t
		}
	}
	return parsed, nil
}

func pwshAcquire(ctx *ChainContext) (Credential, error) {
	if ctx.Run == nil || ctx.onPath("pwsh") == "" {
		return nil, nil
	}
	resource := strings.ReplaceAll(strings.TrimSuffix(azureScope(ctx), "/.default"), "'", "''")
	script := "Get-AzAccessToken -ResourceUrl '" + resource + "' -AsSecureString:$false | ConvertTo-Json -Compress"
	out, err := ctx.Run([]string{"pwsh", "-NoProfile", "-NonInteractive", "-Command", script}, 30*time.Second)
	if err != nil {
		return nil, err
	}
	data := jsonBody([]byte(out))
	if data.Get("Token") == nil {
		return nil, nil
	}
	return bearerFromOAuth(JSONObject{{"access_token", data.Get("Token")}}, ctx.now(), "Azure PowerShell")
}

func azdAcquire(ctx *ChainContext) (Credential, error) {
	if ctx.Run == nil || ctx.onPath("azd") == "" {
		return nil, nil
	}
	out, err := ctx.Run([]string{"azd", "auth", "token", "--output", "json", "--scope", azureScope(ctx)}, 30*time.Second)
	if err != nil {
		return nil, err
	}
	data := jsonBody([]byte(out))
	if data.Get("token") == nil {
		return nil, nil
	}
	parsed, err := bearerFromOAuth(JSONObject{{"access_token", data.Get("token")}}, ctx.now(), "Azure Developer CLI")
	if err != nil {
		return nil, err
	}
	if data.Get("expiresOn") != nil {
		if t, err := ParseRFC3339(wireStr(data.Get("expiresOn"))); err == nil {
			parsed.ExpiresAt = &t
		}
	}
	return parsed, nil
}

func azureNarrowed(ctx *ChainContext, name string, developer bool) bool {
	value := strings.ToLower(strings.TrimSpace(ctx.Env["AZURE_TOKEN_CREDENTIALS"]))
	switch value {
	case "":
		return false
	case "prod":
		return developer
	case "dev":
		return !developer
	}
	return value != strings.ToLower(name)
}

func azureChain(policy AccessPolicy) []Rung {
	guard := func(fn func(*ChainContext) (Credential, error), name string, developer bool) func(*ChainContext) (Credential, error) {
		return func(ctx *ChainContext) (Credential, error) {
			if azureNarrowed(ctx, name, developer) {
				return nil, nil
			}
			return fn(ctx)
		}
	}
	cliProbe := func(name, label, command string) func(*ChainContext) (string, string) {
		return func(ctx *ChainContext) (string, string) {
			if azureNarrowed(ctx, name, true) {
				return "absent", "excluded by AZURE_TOKEN_CREDENTIALS"
			}
			if ctx.onPath(command) == "" {
				return "absent", command + " is not on PATH"
			}
			return "configured", label + " run at request time"
		}
	}
	var rungs []Rung
	if len(policy.EnvKeys) > 0 {
		doorKey := policy.EnvKeys[0]
		rungs = append(rungs, Rung{Name: "env:" + doorKey, Kind: "env", Source: "env $" + doorKey, Probe: envProbe(doorKey), Acquire: func(ctx *ChainContext) (Credential, error) {
			if v := ctx.Env[doorKey]; v != "" {
				return APIKey{Value: v}, nil
			}
			return nil, nil
		}})
	}
	rungs = append(rungs,
		Rung{Name: "environment", Kind: "http-token-exchange", Source: "Entra service principal from AZURE_* env", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			if azureNarrowed(ctx, "EnvironmentCredential", false) {
				return "absent", "excluded by AZURE_TOKEN_CREDENTIALS"
			}
			if kind := azureEnvironmentKind(ctx); kind != "" {
				return "configured", "service principal by " + kind + " (token exchange at request time)"
			}
			return "absent", "AZURE_TENANT_ID/AZURE_CLIENT_ID + secret or certificate not set"
		}, Acquire: guard(azureEnvironmentAcquire, "EnvironmentCredential", false)},
		Rung{Name: "workload-identity", Kind: "http-token-exchange", Source: "Entra workload identity", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			if azureNarrowed(ctx, "WorkloadIdentityCredential", false) {
				return "absent", "excluded by AZURE_TOKEN_CREDENTIALS"
			}
			if azureWorkloadConfig(ctx) {
				return "configured", "federated token file present (exchange at request time)"
			}
			return "absent", "AZURE_FEDERATED_TOKEN_FILE not set"
		}, Acquire: guard(azureWorkloadAcquire, "WorkloadIdentityCredential", false)},
		Rung{Name: "managed-identity", Kind: "http-metadata", Source: "Azure managed identity", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			if azureNarrowed(ctx, "ManagedIdentityCredential", false) {
				return "absent", "excluded by AZURE_TOKEN_CREDENTIALS"
			}
			return "configured", "managed identity (" + azureMSIFlavor(ctx) + ") probed at request time"
		}, Acquire: guard(azureMSIAcquire, "ManagedIdentityCredential", false)},
		Rung{Name: "az", Kind: "subprocess", Source: "az account get-access-token", Needs: "subprocess", Probe: cliProbe("AzureCliCredential", "`az`", "az"), Acquire: guard(azCLIAcquire, "AzureCliCredential", true)},
		Rung{Name: "pwsh", Kind: "subprocess", Source: "Azure PowerShell Get-AzAccessToken", Needs: "subprocess", Probe: cliProbe("AzurePowerShellCredential", "`pwsh`", "pwsh"), Acquire: guard(pwshAcquire, "AzurePowerShellCredential", true)},
		Rung{Name: "azd", Kind: "subprocess", Source: "azd auth token", Needs: "subprocess", Probe: cliProbe("AzureDeveloperCliCredential", "`azd`", "azd"), Acquire: guard(azdAcquire, "AzureDeveloperCliCredential", true)},
	)
	return rungs
}

// ─── Google Cloud ────────────────────────────────────────────────────

// GCPServiceAccountAssertion is (token_uri, JWT) for a service_account file.
func GCPServiceAccountAssertion(ctx *ChainContext, info JSONObject, scope string) (string, string, error) {
	if scope == "" {
		scope = gcpScope
	}
	key, err := rs256.LoadPrivateKey(wireStr(info.Get("private_key")))
	if err != nil {
		return "", "", NotConfiguredErrorf("", nil, "", "%s", err.Error())
	}
	now := ctx.now().Unix()
	tokenURI := stringOnly(info.Get("token_uri"))
	if tokenURI == "" {
		tokenURI = gcpTokenURL
	}
	var header rs256.OrderedObject
	header.Set("alg", "RS256")
	header.Set("typ", "JWT")
	if kid := stringOnly(info.Get("private_key_id")); kid != "" {
		header.Set("kid", kid)
	}
	var payload rs256.OrderedObject
	payload.Set("iat", now)
	payload.Set("exp", now+3600)
	payload.Set("iss", wireStr(info.Get("client_email")))
	payload.Set("aud", tokenURI)
	payload.Set("scope", scope)
	jwt, err := rs256.JWTEncode(header, payload, key)
	return tokenURI, jwt, err
}

func gcpCredentialFile(ctx *ChainContext, path string) (JSONObject, error) {
	raw, ok := ctx.read(path)
	if !ok {
		return nil, nil
	}
	data, err := DecodeJSON([]byte(raw))
	if err != nil {
		return nil, NotConfiguredErrorf("", nil, "", "%s: not valid JSON", path)
	}
	obj, _ := asObject(data)
	return obj, nil
}

func gcpFromInfo(ctx *ChainContext, info JSONObject, where string) (Credential, error) {
	now := ctx.now()
	switch wireStr(info.Get("type")) {
	case "authorized_user":
		for _, k := range []string{"refresh_token", "client_id", "client_secret"} {
			if !truthy(info.Get(k)) {
				return nil, NotConfiguredErrorf("", nil, "", "%s: authorized_user file lacks %s", where, k)
			}
		}
		tokenURI := stringOnly(info.Get("token_uri"))
		if tokenURI == "" {
			tokenURI = gcpTokenURL
		}
		pairs := [][2]string{{"grant_type", "refresh_token"}, {"client_id", wireStr(info.Get("client_id"))}, {"client_secret", wireStr(info.Get("client_secret"))}, {"refresh_token", wireStr(info.Get("refresh_token"))}}
		data, err := exchangeHint(ctx, "POST", tokenURI, map[string]string{"content-type": "application/x-www-form-urlencoded"}, formBody(pairs),
			"Google OAuth refresh ("+where+")", gcpUserLoginHint(where))
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "Google OAuth")
	case "service_account":
		tokenURI, assertion, err := GCPServiceAccountAssertion(ctx, info, "")
		if err != nil {
			return nil, err
		}
		data, err := exchangeHint(ctx, "POST", tokenURI, map[string]string{"content-type": "application/x-www-form-urlencoded"}, formBody([][2]string{{"grant_type", jwtBearerGrant}, {"assertion", assertion}}),
			"Google service account key ("+where+")",
			"the key in "+where+" may have been deleted or disabled, or this machine's clock is off; create a new key (Cloud console: IAM & Admin > Service accounts > Keys) or use another identity")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "Google service account")
	case "external_account":
		return gcpExternalAccount(ctx, info, where)
	case "impersonated_service_account":
		source := wireObj(info.Get("source_credentials"))
		if source == nil {
			return nil, NotConfiguredErrorf("", nil, "", "%s: impersonated_service_account lacks source_credentials", where)
		}
		base, err := gcpFromInfo(ctx, source, where+".source_credentials")
		if err != nil {
			return nil, err
		}
		return gcpImpersonate(ctx, base.(BearerToken), wireStr(info.Get("service_account_impersonation_url")), wireList(info.Get("delegates")), where)
	}
	return nil, NotConfiguredErrorf("", nil, "", "%s: credential type %q is not supported by lm15 (external_account_authorized_user and gdch_service_account are stated gaps)", where, wireStr(info.Get("type")))
}

func gcpImpersonate(ctx *ChainContext, source BearerToken, impersonationURL string, delegates []any, where string) (Credential, error) {
	if delegates == nil {
		delegates = []any{}
	}
	body := mustJSON(JSONObject{{"delegates", delegates}, {"scope", []any{gcpScope}}, {"lifetime", "3600s"}})
	data, err := exchangeHint(ctx, "POST", impersonationURL, map[string]string{"content-type": "application/json", "authorization": "Bearer " + source.Value}, body,
		"service account impersonation ("+where+"; generateAccessToken)",
		"the service account named in "+where+" must exist, and the source identity needs roles/iam.serviceAccountTokenCreator on it (roles/iam.workloadIdentityUser for a workload identity pool), and the IAM Credentials API (iamcredentials.googleapis.com) enabled; a new grant can take several minutes to apply")
	if err != nil {
		return nil, err
	}
	token := stringOnly(data.Get("accessToken"))
	if token == "" {
		return nil, chainAuthError("generateAccessToken: no accessToken")
	}
	var expires *time.Time
	if s := stringOnly(data.Get("expireTime")); s != "" {
		if t, err := ParseRFC3339(s); err == nil {
			expires = &t
		}
	}
	return BearerToken{Value: token, ExpiresAt: expires}, nil
}

func gcpExternalAccount(ctx *ChainContext, info JSONObject, where string) (Credential, error) {
	source := wireObj(info.Get("credential_source"))
	if _, has := source.Lookup("environment_id"); has {
		return nil, NotConfiguredErrorf("", nil, "", "%s: external_account with an AWS credential_source is a stated gap in lm15; use a file/url/executable source or a service account", where)
	}
	subject := ""
	found := false
	format := wireObj(source.Get("format"))
	switch {
	case truthy(source.Get("file")):
		raw, ok := ctx.read(wireStr(source.Get("file")))
		if !ok {
			return nil, NotConfiguredErrorf("", nil, "", "%s: subject token file %s is unreadable", where, wireStr(source.Get("file")))
		}
		subject, found = strings.TrimSpace(raw), true
	case truthy(source.Get("url")):
		headers := map[string]string{}
		for k, v := range wireObj(source.Get("headers")).All() {
			headers[k] = wireStr(v)
		}
		status, _, raw, err := ctx.HTTP("GET", wireStr(source.Get("url")), headers, nil, 30*time.Second)
		if err != nil {
			return nil, err
		}
		if status >= 400 {
			return nil, chainAuthError(fmt.Sprintf("%s: subject token url HTTP %d", where, status))
		}
		subject, found = strings.TrimSpace(string(raw)), true
	case wireObj(source.Get("executable")) != nil:
		if ctx.Run == nil {
			return nil, NotConfiguredErrorf("", nil, "", "%s: executable credential source needs subprocess access", where)
		}
		if ctx.Env["GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES"] != "1" {
			return nil, NotConfiguredErrorf("", nil, "", "%s: set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to allow the executable source", where)
		}
		exe := wireObj(source.Get("executable"))
		timeoutMs := wireFloat(exe.Get("timeout_millis"), 30000)
		out, err := ctx.Run(shellSplit(wireStr(exe.Get("command"))), time.Duration(timeoutMs)*time.Millisecond)
		if err != nil {
			return nil, err
		}
		data := jsonBody([]byte(out))
		if v, ok := data.Get("success").(bool); ok && !v {
			return nil, chainAuthError("external account executable reported failure")
		}
		subject, found = firstStr(data.Get("id_token"), data.Get("saml_response")), true
		format = JSONObject{{"type", "text"}}
	}
	if !found {
		return nil, NotConfiguredErrorf("", nil, "", "%s: external_account credential_source is not file/url/executable", where)
	}
	if wireStr(format.Get("type")) == "json" {
		subject = wireStr(jsonBody([]byte(subject)).Get(wireStr(format.Get("subject_token_field_name"))))
	}
	body := mustJSON(JSONObject{
		{"grantType", "urn:ietf:params:oauth:grant-type:token-exchange"}, {"audience", wireStr(info.Get("audience"))}, {"scope", gcpScope},
		{"requestedTokenType", "urn:ietf:params:oauth:token-type:access_token"}, {"subjectToken", subject}, {"subjectTokenType", wireStr(info.Get("subject_token_type"))},
	})
	tokenURL := stringOnly(info.Get("token_url"))
	if tokenURL == "" {
		tokenURL = gcpSTSURL
	}
	data, err := exchangeHint(ctx, "POST", tokenURL, map[string]string{"content-type": "application/json"}, body, "Google STS exchange ("+where+")",
		"the workload identity pool refused the external token: check the provider's issuer, allowed audience and attribute condition, and that the subject token is fresh")
	if err != nil {
		return nil, err
	}
	token, err := bearerFromOAuth(data, ctx.now(), "Google STS")
	if err != nil {
		return nil, err
	}
	if u := stringOnly(info.Get("service_account_impersonation_url")); u != "" {
		return gcpImpersonate(ctx, token, u, nil, where)
	}
	return token, nil
}

func gcpMetadataAcquire(ctx *ChainContext) (Credential, error) {
	if ctx.HTTP == nil || gceCheckDisabled(ctx) {
		return nil, nil
	}
	host := ctx.Env["GCE_METADATA_HOST"]
	if host == "" {
		host = ctx.Env["GCE_METADATA_ROOT"]
	}
	if host == "" {
		host = "metadata.google.internal"
	}
	status, _, raw, err := ctx.HTTP("GET", "http://"+host+"/computeMetadata/v1/instance/service-accounts/default/token", map[string]string{"Metadata-Flavor": "Google"}, nil, time.Second)
	if err != nil || status != 200 {
		return nil, nil
	}
	return bearerFromOAuth(jsonBody(raw), ctx.now(), "GCE metadata")
}

func gceCheckDisabled(ctx *ChainContext) bool {
	v := strings.ToLower(ctx.Env["NO_GCE_CHECK"])
	return v == "1" || v == "true"
}

func gcloudAcquire(ctx *ChainContext) (Credential, error) {
	if ctx.Run == nil || ctx.onPath("gcloud") == "" {
		return nil, nil
	}
	out, err := ctx.Run([]string{"gcloud", "auth", "print-access-token"}, 30*time.Second)
	if err != nil {
		var e *Error
		if errors.As(err, &e) && e.Kind.IsA(KindAuth) {
			// gcloud's own words stay unread (AUTH-5: a command's stderr is not shown).
			base, _, _ := strings.Cut(e.Message, guidanceMarker)
			return nil, AuthErrorf("", nil, "run `gcloud auth print-access-token` yourself to see gcloud's reason; usually `gcloud auth login` fixes it (or `gcloud auth application-default login`, which lm15 reads first)",
				"`gcloud auth print-access-token` failed: %s", base)
		}
		return nil, err
	}
	if token := strings.TrimSpace(out); token != "" {
		return BearerToken{Value: token}, nil
	}
	return nil, nil
}

func gcpUserLoginHint(where string) string {
	return "the saved Google login in " + where + " has expired or was revoked; run `gcloud auth application-default login` (Google ends these sessions on its own schedule)"
}

func adcFilePath(ctx *ChainContext) string {
	base := ctx.Env["CLOUDSDK_CONFIG"]
	if base == "" {
		base = "~/.config/gcloud"
	}
	return strings.TrimRight(base, "/") + "/application_default_credentials.json"
}

func gcpChain(policy AccessPolicy) []Rung {
	fileProbe := func(label string, pathFn func(*ChainContext) string) func(*ChainContext) (string, string) {
		return func(ctx *ChainContext) (string, string) {
			path := pathFn(ctx)
			if path == "" {
				return "absent", label + " not set"
			}
			info, err := gcpCredentialFile(ctx, path)
			if err != nil {
				return "absent", firstLine(err)
			}
			if info == nil {
				return "absent", path + " missing or unreadable"
			}
			t := stringOnly(info.Get("type"))
			if t == "" {
				t = "?"
			}
			return "configured", t + " credentials in " + path + " (token exchange at request time)"
		}
	}
	fileAcquire := func(pathFn func(*ChainContext) string) func(*ChainContext) (Credential, error) {
		return func(ctx *ChainContext) (Credential, error) {
			path := pathFn(ctx)
			if path == "" {
				return nil, nil
			}
			info, err := gcpCredentialFile(ctx, path)
			if err != nil || info == nil {
				return nil, err
			}
			return gcpFromInfo(ctx, info, path)
		}
	}
	envPath := func(ctx *ChainContext) string { return ctx.Env["GOOGLE_APPLICATION_CREDENTIALS"] }
	var rungs []Rung
	if len(policy.EnvKeys) > 0 {
		doorKey := policy.EnvKeys[0]
		rungs = append(rungs, Rung{Name: "env:" + doorKey, Kind: "env", Source: "env $" + doorKey, Probe: envProbe(doorKey), Acquire: func(ctx *ChainContext) (Credential, error) {
			if v := ctx.Env[doorKey]; v != "" {
				return APIKey{Value: v}, nil
			}
			return nil, nil
		}})
	}
	rungs = append(rungs,
		Rung{Name: "adc-env", Kind: "json-file", Source: "GOOGLE_APPLICATION_CREDENTIALS file", Needs: "network", Probe: fileProbe("GOOGLE_APPLICATION_CREDENTIALS", envPath), Acquire: fileAcquire(envPath)},
		Rung{Name: "adc-file", Kind: "json-file", Source: "gcloud application default credentials file", Needs: "network", Probe: fileProbe("ADC file", adcFilePath), Acquire: fileAcquire(adcFilePath)},
		Rung{Name: "metadata", Kind: "http-metadata", Source: "GCE metadata server", Needs: "network", Probe: func(ctx *ChainContext) (string, string) {
			if gceCheckDisabled(ctx) {
				return "absent", "NO_GCE_CHECK set"
			}
			return "configured", "GCE metadata server probed at request time"
		}, Acquire: gcpMetadataAcquire},
		Rung{Name: "gcloud", Kind: "subprocess", Source: "gcloud auth print-access-token", Needs: "subprocess", Probe: func(ctx *ChainContext) (string, string) {
			if ctx.onPath("gcloud") != "" {
				return "configured", "`gcloud` run at request time"
			}
			return "absent", "gcloud is not on PATH"
		}, Acquire: gcloudAcquire},
	)
	return rungs
}

// ─── Settings from the cloud profile ─────────────────────────────────

var gcloudConfigName = regexp.MustCompile(`^[a-z][-a-z0-9]*$`) // gcloud's own rule (named_configs.py:37); keeps the name inside the directory

// gcloudConfigProject is the project `gcloud config get project` prints,
// read from the files gcloud reads (AUTH-10, amended 2026-09-26):
// CLOUDSDK_CORE_PROJECT, then [core] project in
// $CLOUDSDK_CONFIG/configurations/config_<name>, <name> from
// CLOUDSDK_ACTIVE_CONFIG_NAME, else the active_config file, else default.
func gcloudConfigProject(ctx *ChainContext) (value, from string) {
	if v := strings.TrimSpace(ctx.Env["CLOUDSDK_CORE_PROJECT"]); v != "" {
		return v, "env:CLOUDSDK_CORE_PROJECT"
	}
	base := ctx.Env["CLOUDSDK_CONFIG"]
	if base == "" {
		base = "~/.config/gcloud"
	}
	base = strings.TrimRight(base, "/")
	name := strings.TrimSpace(ctx.Env["CLOUDSDK_ACTIVE_CONFIG_NAME"])
	if name == "" {
		active, _ := ctx.read(base + "/active_config")
		name = strings.TrimSpace(active)
	}
	if name == "" {
		name = "default"
	}
	if !gcloudConfigName.MatchString(name) {
		return "", ""
	}
	raw, ok := ctx.read(base + "/configurations/config_" + name)
	if !ok || raw == "" {
		return "", ""
	}
	ini, err := parseINI(raw)
	if err != nil {
		return "", ""
	}
	if v := strings.TrimSpace(ini.section("core")["project"]); v != "" {
		return v, "gcloud-config"
	}
	return "", ""
}

// gcpMetadataProject is project/project-id from the metadata server: the
// project a Cloud Run service, GKE pod or VM runs in. Offline (the
// doctor) it is ("", "metadata"): unprobed.
func gcpMetadataProject(ctx *ChainContext) (value, from string) {
	if gceCheckDisabled(ctx) {
		return "", ""
	}
	if ctx.HTTP == nil {
		return "", "metadata"
	}
	host := ctx.Env["GCE_METADATA_HOST"]
	if host == "" {
		host = ctx.Env["GCE_METADATA_ROOT"]
	}
	if host == "" {
		host = "metadata.google.internal"
	}
	status, _, raw, err := ctx.HTTP("GET", "http://"+host+"/computeMetadata/v1/project/project-id", map[string]string{"Metadata-Flavor": "Google"}, nil, time.Second)
	if err != nil || status != 200 {
		return "", ""
	}
	v := strings.TrimSpace(string(raw))
	if v == "" || strings.ContainsAny(v, " \t\r\n/?#") {
		return "", ""
	}
	return v, "metadata"
}

// ProfileSettings returns the setting values the cloud's own configuration
// carries, as (value, from) in the AUTH-10 vocabulary. AWS region: the
// active profile. Google project (amended 2026-09-26, the order google-auth
// and gcloud give it): the GOOGLE_APPLICATION_CREDENTIALS file's project_id
// (then quota_project_id); gcloud's active configuration; the ADC file's
// quota_project_id / project_id; the metadata server (online at
// construction; offline ("", "metadata") = unprobed). Nothing for Azure.
func ProfileSettings(policy AccessPolicy, ctx *ChainContext) func(string) (string, string) {
	return func(name string) (string, string) {
		switch {
		case policy.EffectiveCredentialPolicy() == "aws-chain" && name == "region":
			creds, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return "", ""
			}
			if v := awsProfileSection(conf, profile)["region"]; v != "" {
				return v, "aws-profile"
			}
			if creds.has(profile) {
				if v := creds.section(profile)["region"]; v != "" {
					return v, "aws-profile"
				}
			}
		case policy.EffectiveCredentialPolicy() == "gcp-chain" && name == "project":
			if path := ctx.Env["GOOGLE_APPLICATION_CREDENTIALS"]; path != "" {
				info, _ := gcpCredentialFile(ctx, path)
				if v := firstStr(info.Get("project_id"), info.Get("quota_project_id")); v != "" {
					return v, "adc-env"
				}
			}
			if v, from := gcloudConfigProject(ctx); v != "" {
				return v, from
			}
			info, _ := gcpCredentialFile(ctx, adcFilePath(ctx))
			if v := firstStr(info.Get("quota_project_id"), info.Get("project_id")); v != "" {
				return v, "adc-file"
			}
			return gcpMetadataProject(ctx)
		}
		return "", ""
	}
}

// ─── Chains ──────────────────────────────────────────────────────────

// ChainFor returns the rungs of a cloud chain policy.
func ChainFor(policy AccessPolicy) ([]Rung, error) {
	switch policy.EffectiveCredentialPolicy() {
	case "aws-chain":
		return awsChain(policy), nil
	case "azure-chain":
		return azureChain(policy), nil
	case "gcp-chain":
		return gcpChain(policy), nil
	}
	return nil, valueErrorf("%s: not a cloud chain policy", policy.Provider)
}

// CredentialSource is where a resolved credential came from (AUTH-1
// provenance, 2026-09-19). Never the value: the rung's fixture kind, its
// human label, the name that selected it (platform …) when one did, and
// the expiry if known.
type CredentialSource struct {
	Rung      string
	Label     string
	Named     string
	ExpiresAt *time.Time
}

// Describe renders the source for an error or the doctor.
func (s CredentialSource) Describe(now time.Time) string {
	text := s.Label
	if s.Named != "" {
		text += " (named credential \"" + s.Named + "\")"
	}
	if s.ExpiresAt != nil {
		if now.IsZero() {
			now = time.Now().UTC()
		}
		left := int(s.ExpiresAt.Sub(now).Seconds())
		switch {
		case left <= 0:
			text += ", expired"
		case left < 3600:
			minutes := left / 60
			if minutes < 1 {
				minutes = 1
			}
			text += fmt.Sprintf(", expires in %d min", minutes)
		default:
			text += fmt.Sprintf(", expires in %d h %d min", left/3600, (left%3600)/60)
		}
	}
	return text
}

// NamedRungs is AUTH-1 (amended 2026-09-19): the rungs each named
// credential covers on each cloud. One word, the same on every cloud; the
// doctor and every auth error print the concrete mechanism, never just the
// word. platform on AWS covers two rungs (the container endpoint, then
// IMDS): both are the machine's own identity, boto3 tries them in this
// order, and the doctor says which answered. workload and environment on
// GCP are the same file rung told apart by the file's type
// (external_account vs service_account); the wrong type is refused by
// name, never read as the other.
var NamedRungs = map[string]map[string][]string{
	"aws-chain": {
		CredentialPlatform:    {"container", "imds"},
		CredentialWorkload:    {"web-identity"},
		CredentialEnvironment: {"env:AWS_ACCESS_KEY_ID"},
		CredentialCLI:         {"assume-role", "sso", "shared-credentials-file", "login", "credential_process", "config-file"},
	},
	"azure-chain": {
		CredentialPlatform:    {"managed-identity"},
		CredentialWorkload:    {"workload-identity"},
		CredentialEnvironment: {"environment"},
		CredentialCLI:         {"az", "pwsh", "azd"},
	},
	"gcp-chain": {
		CredentialPlatform:    {"metadata"},
		CredentialWorkload:    {"adc-env"},
		CredentialEnvironment: {"adc-env"},
		CredentialCLI:         {"adc-file", "gcloud"},
	},
}

var gcpNamedTypes = map[string][]string{
	CredentialWorkload:    {"external_account"},
	CredentialEnvironment: {"service_account", "impersonated_service_account"},
}

// NamedMeaning is what each name means on each cloud, for the doctor and
// for errors.
var NamedMeaning = map[string]map[string]string{
	"aws-chain": {
		CredentialPlatform:    "the ECS/EKS container endpoint, else the EC2 instance role (IMDSv2)",
		CredentialWorkload:    "web identity (AWS_WEB_IDENTITY_TOKEN_FILE + AWS_ROLE_ARN) via STS",
		CredentialEnvironment: "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY",
		CredentialCLI:         "the active `aws` profile (assume-role, SSO, shared files, `aws login`, credential_process)",
	},
	"azure-chain": {
		CredentialPlatform:    "Azure managed identity",
		CredentialWorkload:    "Entra workload identity (AZURE_FEDERATED_TOKEN_FILE)",
		CredentialEnvironment: "an Entra service principal from AZURE_TENANT_ID / AZURE_CLIENT_ID + secret or certificate",
		CredentialCLI:         "`az`, Azure PowerShell or `azd` sign-in",
	},
	"gcp-chain": {
		CredentialPlatform:    "the attached service account (GCE metadata server)",
		CredentialWorkload:    "workload identity federation (GOOGLE_APPLICATION_CREDENTIALS, type external_account)",
		CredentialEnvironment: "a service-account file (GOOGLE_APPLICATION_CREDENTIALS, type service_account)",
		CredentialCLI:         "`gcloud auth application-default login` (the ADC file) or `gcloud auth print-access-token`",
	},
}

// gcpTyped narrows the GOOGLE_APPLICATION_CREDENTIALS rung to the file
// types the name means; another type is refused naming the name it
// belongs to.
func gcpTyped(rung Rung, name string) Rung {
	allowed := gcpNamedTypes[name]
	other := CredentialWorkload
	if name == CredentialWorkload {
		other = CredentialEnvironment
	}
	kindOf := func(ctx *ChainContext) (string, string) {
		path := ctx.Env["GOOGLE_APPLICATION_CREDENTIALS"]
		if path == "" {
			return "", ""
		}
		info, _ := gcpCredentialFile(ctx, path)
		if info == nil {
			return path, ""
		}
		return path, wireStr(info.Get("type"))
	}
	mismatch := func(path, kind string) string {
		return fmt.Sprintf("%s holds %s credentials; that is the named credential %q, not %q", path, kind, other, name)
	}
	out := rung
	out.Probe = func(ctx *ChainContext) (string, string) {
		if path, kind := kindOf(ctx); path != "" && kind != "" && !inVocab(kind, allowed) {
			return "absent", mismatch(path, kind)
		}
		return rung.Probe(ctx)
	}
	out.Acquire = func(ctx *ChainContext) (Credential, error) {
		if path, kind := kindOf(ctx); path != "" && kind != "" && !inVocab(kind, allowed) {
			return nil, NotConfiguredErrorf("", nil, "credentials={\"<provider>\": \""+other+"\"}", "%s", mismatch(path, kind))
		}
		return rung.Acquire(ctx)
	}
	return out
}

// NamedRungsFor is the rungs a named credential covers on this door, in
// chain order. An unknown name fails here, at construction.
func NamedRungsFor(policy AccessPolicy, name string) ([]Rung, error) {
	if !inVocab(name, NamedCredentials) {
		quoted := make([]string, 0, len(NamedCredentials))
		for _, n := range NamedCredentials {
			quoted = append(quoted, strconv.Quote(n))
		}
		return nil, NotConfiguredErrorf(policy.Provider, nil, "", "%s: unknown named credential %q; one of %s", policy.Provider, name, strings.Join(quoted, ", "))
	}
	all, err := ChainFor(policy)
	if err != nil {
		return nil, err
	}
	wanted := NamedRungs[policy.EffectiveCredentialPolicy()][name]
	var rungs []Rung
	for _, rung := range all {
		if inVocab(rung.Name, wanted) {
			if policy.EffectiveCredentialPolicy() == "gcp-chain" && gcpNamedTypes[name] != nil {
				rung = gcpTyped(rung, name)
			}
			rungs = append(rungs, rung)
		}
	}
	return rungs, nil
}

// NamedMeaningFor is the sentence a name means on this door.
func NamedMeaningFor(policy AccessPolicy, name string) string {
	return NamedMeaning[policy.EffectiveCredentialPolicy()][name]
}

func chainWalk(policy AccessPolicy, named string) ([]Rung, error) {
	if named != "" {
		return NamedRungsFor(policy, named)
	}
	return ChainFor(policy)
}

// ExplainChain is the AUTH-7 walk over a cloud chain (offline).
func ExplainChain(policy AccessPolicy, ctx *ChainContext, explicit bool) ([]ChainStep, bool, error) {
	return ExplainChainNamed(policy, ctx, explicit, "")
}

// ExplainChainNamed is ExplainChain under a named credential: only the
// rungs the name covers are walked (the rest are not steps at all: a named
// credential never falls through).
func ExplainChainNamed(policy AccessPolicy, ctx *ChainContext, explicit bool, named string) ([]ChainStep, bool, error) {
	rungs, err := chainWalk(policy, named)
	if err != nil {
		return nil, false, err
	}
	var steps []ChainStep
	selected := false
	if explicit {
		steps = append(steps, ChainStep{"api_keys", "explicit api_keys entry", "provided (value never shown)", "selected"})
		selected = true
	} else {
		steps = append(steps, ChainStep{"api_keys", "explicit api_keys entry", "not provided", "absent"})
	}
	unprobed := false
	for _, rung := range rungs {
		verdict, detail := rung.Probe(ctx)
		state := "absent"
		switch verdict {
		case "configured":
			state = "unprobed"
			if selected {
				state = "shadowed"
			} else {
				unprobed = true
			}
		case "usable":
			state = "selected"
			if selected {
				state = "shadowed"
			}
			selected = true
		}
		steps = append(steps, ChainStep{rung.Name, rung.Source, detail, state})
	}
	return steps, selected || unprobed, nil
}

// ResolveChain walks the chain online; the first rung that yields wins.
func ResolveChain(policy AccessPolicy, ctx *ChainContext) (Credential, error) {
	cred, _, err := ResolveChainNamed(policy, ctx, "")
	return cred, err
}

func sourceOf(rung Rung, value Credential, named string) CredentialSource {
	src := CredentialSource{Rung: rung.Name, Label: rung.Source, Named: named}
	switch v := value.(type) {
	case BearerToken:
		src.ExpiresAt = v.ExpiresAt
	case AwsCredentials:
		src.ExpiresAt = v.ExpiresAt
	}
	return src
}

func probedSummary(ctx *ChainContext, rungs []Rung) string {
	var parts []string
	for _, rung := range rungs {
		_, detail := rung.Probe(ctx)
		parts = append(parts, rung.Source+": "+detail)
	}
	return strings.Join(parts, "; ")
}

// ResolveChainNamed walks the chain online; the first rung that yields
// wins, and the result names the rung (AUTH-1 provenance). A rung that is
// configured and fails raises. Azure developer commands are the AUTH-1
// exception: try all three before reporting their failure. With named,
// only that name's rungs run and nothing else is tried.
func ResolveChainNamed(policy AccessPolicy, ctx *ChainContext, named string) (Credential, CredentialSource, error) {
	rungs, err := chainWalk(policy, named)
	if err != nil {
		return nil, CredentialSource{}, err
	}
	developerFailed := false
	for _, rung := range rungs {
		got, err := rung.Acquire(ctx)
		if err != nil {
			if IsKind(err, KindAuth) && policy.EffectiveCredentialPolicy() == "azure-chain" && (rung.Name == "az" || rung.Name == "pwsh" || rung.Name == "azd") {
				developerFailed = true
				continue
			}
			return nil, CredentialSource{}, err
		}
		if got != nil {
			return got, sourceOf(rung, got, named), nil
		}
	}
	if developerFailed {
		e := chainAuthError("Azure developer credentials failed; sign in with az, Azure PowerShell, or azd")
		e.Provider = policy.Provider
		return nil, CredentialSource{}, e
	}
	if named != "" {
		e := NotConfiguredErrorf(policy.Provider, nil,
			fmt.Sprintf("credentials={%q: \"<platform|workload|environment|cli>\"} names one identity; omit it to walk the %s chain", policy.Provider, policy.EffectiveCredentialPolicy()),
			"%s: named credential %q — %s — answered nothing (%s). This door was told to use that identity only; it will not try the rest of the %s chain.",
			policy.Provider, named, NamedMeaningFor(policy, named), probedSummary(ctx, rungs), policy.EffectiveCredentialPolicy())
		return nil, CredentialSource{}, e
	}
	return nil, CredentialSource{}, NotConfiguredErrorf(policy.Provider, policy.EnvKeys, nothingFoundHint(policy),
		"%s: no credential found in the %s chain (%s)", policy.Provider, policy.EffectiveCredentialPolicy(), probedSummary(ctx, rungs))
}

// nothingFoundHints: what to do when a whole chain answers nothing — the
// command that creates a credential the chain reads, then the deployed
// alternatives.
var nothingFoundHints = map[string]string{
	"gcp-chain":   "on a laptop: `gcloud auth application-default login`; elsewhere: set GOOGLE_APPLICATION_CREDENTIALS to a service-account or workload-identity file, run on Google Cloud with an attached service account, or pass RouterConfig APIKeys{\"<provider>\": <token, key or provider>}",
	"azure-chain": "on a laptop: `az login`; elsewhere: a managed identity, AZURE_TENANT_ID + AZURE_CLIENT_ID with a secret or certificate, or RouterConfig APIKeys{\"<provider>\": <token provider>}",
	"aws-chain":   "on a laptop: `aws sso login` or `aws configure`; elsewhere: the instance or container role, AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or RouterConfig APIKeys{\"<provider>\": <credentials provider>}",
}

func nothingFoundHint(policy AccessPolicy) string {
	hint := nothingFoundHints[policy.EffectiveCredentialPolicy()]
	if hint != "" && len(policy.EnvKeys) > 0 {
		hint = "set " + policy.EnvKeys[0] + ", or " + hint
	}
	return strings.Replace(hint, "<provider>", policy.Provider, 1)
}

// wireAuthHint: how a cloud door's wire refusal (HTTP 401/403 after a
// credential was obtained) is fixed; replaces the generic API-key guidance.
// sent is what the adapter knows it sent: "key", "token", or "" (a provider
// function decides per request).
func wireAuthHint(policy AccessPolicy, status int, sent string) string {
	if policy.EffectiveCredentialPolicy() != "gcp-chain" {
		return ""
	}
	switch {
	case status == 403:
		return "give the identity named above the Vertex AI User role (roles/aiplatform.user) on the project and enable the Vertex AI API (aiplatform.googleapis.com); a new project or a new grant can take a few minutes to apply. To use another identity: `gcloud auth application-default login`, or GOOGLE_APPLICATION_CREDENTIALS=<file>"
	case status != 401:
		return ""
	case sent == "key":
		return "Google refused this API key: use a Vertex AI key (Cloud console > APIs & Services > Credentials, restricted to the Vertex AI API or bound to a service account); Claude on Vertex takes no keys. If the value is an access token that does not start with `ya29.`, pass lm15.BearerToken{Value: value}"
	}
	return "Google refused this access token: it expired (they last an hour; pass a provider, or let lm15's chain refresh it) or it is not an OAuth token. Sign in again with `gcloud auth application-default login`"
}

// cachingProvider is the AUTH-2 provider over a cloud chain (or a named
// credential) with the AUTH-3 in-memory cache. Source is where the last
// resolution came from (AUTH-1 provenance); zero until the first request.
type cachingProvider struct {
	policy AccessPolicy
	ctx    *ChainContext
	named  string
	mu     sync.Mutex
	value  Credential
	source CredentialSource
}

// Named is the named credential this provider was built for ("" = the chain).
func (p *cachingProvider) Named() string { return p.named }

// Source is where the last resolution came from; ok is false before the
// first request.
func (p *cachingProvider) Source() (CredentialSource, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.source, p.source.Rung != ""
}

func (p *cachingProvider) Credential(context.Context) (Credential, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.value != nil && !p.value.IsExpired(p.ctx.now()) {
		return p.value, nil
	}
	value, source, err := ResolveChainNamed(p.policy, p.ctx, p.named)
	if err != nil {
		return nil, err
	}
	p.source = source
	if value.IsExpired(p.ctx.now()) {
		e := chainAuthError("cloud credential is expired; renew the configured credential source (credential came from: " + source.Describe(p.ctx.now()) + ")")
		e.Provider = p.policy.Provider
		return nil, e
	}
	switch v := value.(type) {
	case APIKey, AwsCredentials:
		p.value = value
	case BearerToken:
		if v.ExpiresAt != nil {
			p.value = value
		} else {
			p.value = nil // CLI output without an expiry cannot be cached forever
		}
	}
	return value, nil
}

func (p *cachingProvider) String() string {
	return "<cloud credential provider for " + p.policy.Provider + ">"
}

// CredentialProviderFor is the AUTH-2 provider over a cloud chain with the
// AUTH-3 in-memory cache.
func CredentialProviderFor(policy AccessPolicy, ctx *ChainContext) CredentialProvider {
	return &cachingProvider{policy: policy, ctx: ctx}
}

// NamedCredentialProviderFor is CredentialProviderFor under a named
// credential (AUTH-1): an unknown name fails at construction, not on the
// first request.
func NamedCredentialProviderFor(policy AccessPolicy, ctx *ChainContext, named string) (CredentialProvider, error) {
	if named != "" {
		if _, err := NamedRungsFor(policy, named); err != nil {
			return nil, err
		}
	}
	return &cachingProvider{policy: policy, ctx: ctx, named: named}, nil
}

// ChainCredentialSource reports the provenance of a cloud credential
// provider after its first resolution (nil for any other provider).
func ChainCredentialSource(p CredentialProvider) *CredentialSource {
	cp, ok := p.(*cachingProvider)
	if !ok {
		return nil
	}
	if src, ok := cp.Source(); ok {
		return &src
	}
	return nil
}

// ChainCacheKey is provider id + the identity-selecting settings (AUTH-3).
func ChainCacheKey(policy AccessPolicy, ctx *ChainContext) string {
	parts := []string{policy.Provider, ctx.Env["AWS_PROFILE"], ctx.Env["AZURE_TENANT_ID"], ctx.Env["AZURE_CLIENT_ID"], ctx.Env["GOOGLE_APPLICATION_CREDENTIALS"], ctx.Env["CLOUDSDK_CONFIG"], ctx.Home}
	keys := make([]string, 0, len(ctx.Settings))
	for k := range ctx.Settings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		parts = append(parts, k+"="+ctx.Settings[k])
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x1f")))
	return hex.EncodeToString(sum[:])
}

// ─── Harness ops ─────────────────────────────────────────────────────

// TokenExchangeBuild is the exact token-exchange request a rung would send
// under the fixed clock.
func TokenExchangeBuild(policy AccessPolicy, rung string, inputs JSONObject, ctx *ChainContext) (JSONObject, error) {
	switch rung {
	case "adc-env", "adc-file", "service-account":
		info := wireObj(inputs.Get("credential_file"))
		scope := stringOnly(inputs.Get("scope"))
		tokenURI, assertion, err := GCPServiceAccountAssertion(ctx, info, scope)
		if err != nil {
			return nil, err
		}
		return JSONObject{{"method", "POST"}, {"url", tokenURI}, {"headers", JSONObject{{"content-type", "application/x-www-form-urlencoded"}}}, {"body_encoding", "form"}, {"body", JSONObject{{"grant_type", jwtBearerGrant}, {"assertion", assertion}}}}, nil
	case "environment":
		tokenURL, pairs, err := AzureEnvironmentRequest(ctx, stringOnly(inputs.Get("jti")))
		if err != nil {
			return nil, err
		}
		body := JSONObject{}
		for _, p := range pairs {
			body.Set(p[0], p[1])
		}
		return JSONObject{{"method", "POST"}, {"url", tokenURL}, {"headers", JSONObject{{"content-type", "application/x-www-form-urlencoded"}}}, {"body_encoding", "form"}, {"body", body}}, nil
	}
	return nil, valueErrorf("token_exchange_build: rung %q has no deterministic request", rung)
}

// TokenExchangeParse is the credential a rung produces from a pinned body.
func TokenExchangeParse(policy AccessPolicy, rung string, status int, body JSONObject, ctx *ChainContext) (Credential, error) {
	now := ctx.now()
	switch rung {
	case "adc-env", "adc-file", "service-account", "environment", "workload-identity", "managed-identity", "metadata":
		if status < 200 || status >= 300 {
			return nil, chainAuthError(fmt.Sprintf("%s: HTTP %d", rung, status))
		}
		return bearerFromOAuth(body, now, rung)
	case "credential_process":
		if status != 0 || wireInt(body.Get("Version"), 0) != 1 {
			return nil, chainAuthError("credential_process failed or returned an unsupported Version")
		}
		return awsFromResponse(body)
	case "imds", "container":
		if status < 200 || status >= 300 {
			return nil, chainAuthError(fmt.Sprintf("%s: HTTP %d", rung, status))
		}
		return awsFromResponse(body)
	}
	return nil, valueErrorf("token_exchange_parse: rung %q is not a parse vector", rung)
}
