package lm15

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
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
	if obj, ok := data.(map[string]any); ok {
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

func exchange(ctx *ChainContext, method, url string, headers map[string]string, body []byte, what string) (JSONObject, error) {
	status, _, raw, err := ctx.HTTP(method, url, headers, body, 30*time.Second)
	if err != nil {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, chainAuthError(fmt.Sprintf("%s: HTTP %d", what, status))
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
	token := stringOnly(data["access_token"])
	if token == "" {
		return BearerToken{}, chainAuthError(what + ": no valid access_token in response")
	}
	var expires *time.Time
	if v := data["expires_on"]; v != nil && wireStr(v) != "" {
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
		if v := data["expires_in"]; v != nil && wireStr(v) != "" {
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
	raw := d["Expiration"]
	if raw == nil {
		raw = d["expiration"]
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
	key := firstStr(d["AccessKeyId"], d["accessKeyId"])
	secret := firstStr(d["SecretAccessKey"], d["secretAccessKey"])
	if key == "" || secret == "" {
		return AwsCredentials{}, chainAuthError("AWS credential response lacks access key id or secret access key")
	}
	return AwsCredentials{AccessKeyID: key, SecretAccessKey: secret, SessionToken: firstStr(d["SessionToken"], d["Token"], d["sessionToken"]), ExpiresAt: expires}, nil
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
	if s := stringOnly(token["expiresAt"]); s != "" {
		if t, err := ParseRFC3339(s); err == nil {
			expires = &t
		}
	}
	access := stringOnly(token["accessToken"])
	ssoRegion := cfg["sso_region"]
	if ssoRegion == "" {
		ssoRegion = "us-east-1"
	}
	if access == "" || (expires != nil && expires.Sub(now) <= expirySkew) {
		if stringOnly(token["refreshToken"]) == "" || stringOnly(token["clientId"]) == "" || stringOnly(token["clientSecret"]) == "" {
			return nil, NotConfiguredErrorf("", nil, "aws sso login", "IAM Identity Center: token expired and not refreshable; run `aws sso login`")
		}
		data, err := exchange(ctx, "POST", "https://oidc."+ssoRegion+".amazonaws.com/token", map[string]string{"content-type": "application/json"},
			mustJSON(JSONObject{"clientId": token["clientId"], "clientSecret": token["clientSecret"], "grantType": "refresh_token", "refreshToken": token["refreshToken"]}), "sso-oidc CreateToken")
		if err != nil {
			return nil, err
		}
		access = stringOnly(data["accessToken"])
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
	return awsFromResponse(wireObj(jsonBody(rawCreds)["roleCredentials"]))
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
	token := wireObj(jsonBody([]byte(raw))["accessToken"])
	if stringOnly(token["accessKeyId"]) == "" {
		return nil, nil
	}
	creds, err := awsFromResponse(JSONObject{"AccessKeyId": token["accessKeyId"], "SecretAccessKey": token["secretAccessKey"], "SessionToken": token["sessionToken"], "Expiration": token["expiresAt"]})
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
	if wireInt(data["Version"], 0) != 1 {
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
	if code := data["Code"]; code != nil && wireStr(code) != "Success" {
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
	if data["accessToken"] == nil {
		return nil, nil
	}
	parsed, err := bearerFromOAuth(JSONObject{"access_token": data["accessToken"], "expires_on": data["expires_on"]}, ctx.now(), "Azure CLI")
	if err != nil {
		return nil, err
	}
	if parsed.ExpiresAt == nil && data["expiresOn"] != nil {
		if t, err := ParseRFC3339(wireStr(data["expiresOn"])); err == nil {
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
	if data["Token"] == nil {
		return nil, nil
	}
	return bearerFromOAuth(JSONObject{"access_token": data["Token"]}, ctx.now(), "Azure PowerShell")
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
	if data["token"] == nil {
		return nil, nil
	}
	parsed, err := bearerFromOAuth(JSONObject{"access_token": data["token"]}, ctx.now(), "Azure Developer CLI")
	if err != nil {
		return nil, err
	}
	if data["expiresOn"] != nil {
		if t, err := ParseRFC3339(wireStr(data["expiresOn"])); err == nil {
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
	key, err := rs256.LoadPrivateKey(wireStr(info["private_key"]))
	if err != nil {
		return "", "", NotConfiguredErrorf("", nil, "", "%s", err.Error())
	}
	now := ctx.now().Unix()
	tokenURI := stringOnly(info["token_uri"])
	if tokenURI == "" {
		tokenURI = gcpTokenURL
	}
	var header rs256.OrderedObject
	header.Set("alg", "RS256")
	header.Set("typ", "JWT")
	if kid := stringOnly(info["private_key_id"]); kid != "" {
		header.Set("kid", kid)
	}
	var payload rs256.OrderedObject
	payload.Set("iat", now)
	payload.Set("exp", now+3600)
	payload.Set("iss", wireStr(info["client_email"]))
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
	obj, _ := data.(map[string]any)
	return obj, nil
}

func gcpFromInfo(ctx *ChainContext, info JSONObject, where string) (Credential, error) {
	now := ctx.now()
	switch wireStr(info["type"]) {
	case "authorized_user":
		for _, k := range []string{"refresh_token", "client_id", "client_secret"} {
			if !truthy(info[k]) {
				return nil, NotConfiguredErrorf("", nil, "", "%s: authorized_user file lacks %s", where, k)
			}
		}
		tokenURI := stringOnly(info["token_uri"])
		if tokenURI == "" {
			tokenURI = gcpTokenURL
		}
		pairs := [][2]string{{"grant_type", "refresh_token"}, {"client_id", wireStr(info["client_id"])}, {"client_secret", wireStr(info["client_secret"])}, {"refresh_token", wireStr(info["refresh_token"])}}
		data, err := exchange(ctx, "POST", tokenURI, map[string]string{"content-type": "application/x-www-form-urlencoded"}, formBody(pairs), "Google OAuth refresh")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "Google OAuth")
	case "service_account":
		tokenURI, assertion, err := GCPServiceAccountAssertion(ctx, info, "")
		if err != nil {
			return nil, err
		}
		data, err := exchange(ctx, "POST", tokenURI, map[string]string{"content-type": "application/x-www-form-urlencoded"}, formBody([][2]string{{"grant_type", jwtBearerGrant}, {"assertion", assertion}}), "Google service account")
		if err != nil {
			return nil, err
		}
		return bearerFromOAuth(data, now, "Google service account")
	case "external_account":
		return gcpExternalAccount(ctx, info, where)
	case "impersonated_service_account":
		source := wireObj(info["source_credentials"])
		if source == nil {
			return nil, NotConfiguredErrorf("", nil, "", "%s: impersonated_service_account lacks source_credentials", where)
		}
		base, err := gcpFromInfo(ctx, source, where+".source_credentials")
		if err != nil {
			return nil, err
		}
		return gcpImpersonate(ctx, base.(BearerToken), wireStr(info["service_account_impersonation_url"]), wireList(info["delegates"]))
	}
	return nil, NotConfiguredErrorf("", nil, "", "%s: credential type %q is not supported by lm15 (external_account_authorized_user and gdch_service_account are stated gaps)", where, wireStr(info["type"]))
}

func gcpImpersonate(ctx *ChainContext, source BearerToken, impersonationURL string, delegates []any) (Credential, error) {
	if delegates == nil {
		delegates = []any{}
	}
	body := mustJSON(JSONObject{"delegates": delegates, "scope": []any{gcpScope}, "lifetime": "3600s"})
	data, err := exchange(ctx, "POST", impersonationURL, map[string]string{"content-type": "application/json", "authorization": "Bearer " + source.Value}, body, "generateAccessToken")
	if err != nil {
		return nil, err
	}
	token := stringOnly(data["accessToken"])
	if token == "" {
		return nil, chainAuthError("generateAccessToken: no accessToken")
	}
	var expires *time.Time
	if s := stringOnly(data["expireTime"]); s != "" {
		if t, err := ParseRFC3339(s); err == nil {
			expires = &t
		}
	}
	return BearerToken{Value: token, ExpiresAt: expires}, nil
}

func gcpExternalAccount(ctx *ChainContext, info JSONObject, where string) (Credential, error) {
	source := wireObj(info["credential_source"])
	if _, has := source["environment_id"]; has {
		return nil, NotConfiguredErrorf("", nil, "", "%s: external_account with an AWS credential_source is a stated gap in lm15; use a file/url/executable source or a service account", where)
	}
	subject := ""
	found := false
	format := wireObj(source["format"])
	switch {
	case truthy(source["file"]):
		raw, ok := ctx.read(wireStr(source["file"]))
		if !ok {
			return nil, NotConfiguredErrorf("", nil, "", "%s: subject token file %s is unreadable", where, wireStr(source["file"]))
		}
		subject, found = strings.TrimSpace(raw), true
	case truthy(source["url"]):
		headers := map[string]string{}
		for k, v := range wireObj(source["headers"]) {
			headers[k] = wireStr(v)
		}
		status, _, raw, err := ctx.HTTP("GET", wireStr(source["url"]), headers, nil, 30*time.Second)
		if err != nil {
			return nil, err
		}
		if status >= 400 {
			return nil, chainAuthError(fmt.Sprintf("%s: subject token url HTTP %d", where, status))
		}
		subject, found = strings.TrimSpace(string(raw)), true
	case wireObj(source["executable"]) != nil:
		if ctx.Run == nil {
			return nil, NotConfiguredErrorf("", nil, "", "%s: executable credential source needs subprocess access", where)
		}
		if ctx.Env["GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES"] != "1" {
			return nil, NotConfiguredErrorf("", nil, "", "%s: set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to allow the executable source", where)
		}
		exe := wireObj(source["executable"])
		timeoutMs := wireFloat(exe["timeout_millis"], 30000)
		out, err := ctx.Run(shellSplit(wireStr(exe["command"])), time.Duration(timeoutMs)*time.Millisecond)
		if err != nil {
			return nil, err
		}
		data := jsonBody([]byte(out))
		if v, ok := data["success"].(bool); ok && !v {
			return nil, chainAuthError("external account executable reported failure")
		}
		subject, found = firstStr(data["id_token"], data["saml_response"]), true
		format = JSONObject{"type": "text"}
	}
	if !found {
		return nil, NotConfiguredErrorf("", nil, "", "%s: external_account credential_source is not file/url/executable", where)
	}
	if wireStr(format["type"]) == "json" {
		subject = wireStr(jsonBody([]byte(subject))[wireStr(format["subject_token_field_name"])])
	}
	body := mustJSON(JSONObject{
		"grantType": "urn:ietf:params:oauth:grant-type:token-exchange", "audience": wireStr(info["audience"]), "scope": gcpScope,
		"requestedTokenType": "urn:ietf:params:oauth:token-type:access_token", "subjectToken": subject, "subjectTokenType": wireStr(info["subject_token_type"]),
	})
	tokenURL := stringOnly(info["token_url"])
	if tokenURL == "" {
		tokenURL = gcpSTSURL
	}
	data, err := exchange(ctx, "POST", tokenURL, map[string]string{"content-type": "application/json"}, body, "Google STS exchange")
	if err != nil {
		return nil, err
	}
	token, err := bearerFromOAuth(data, ctx.now(), "Google STS")
	if err != nil {
		return nil, err
	}
	if u := stringOnly(info["service_account_impersonation_url"]); u != "" {
		return gcpImpersonate(ctx, token, u, nil)
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
		return nil, err
	}
	if token := strings.TrimSpace(out); token != "" {
		return BearerToken{Value: token}, nil
	}
	return nil, nil
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
			t := stringOnly(info["type"])
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

// ProfileSettings returns the setting values the cloud's own config files
// carry (AWS region, GCP project).
func ProfileSettings(policy AccessPolicy, ctx *ChainContext) func(string) string {
	return func(name string) string {
		switch {
		case policy.EffectiveCredentialPolicy() == "aws-chain" && name == "region":
			creds, conf, profile, err := awsConfig(ctx)
			if err != nil {
				return ""
			}
			if v := awsProfileSection(conf, profile)["region"]; v != "" {
				return v
			}
			if creds.has(profile) {
				return creds.section(profile)["region"]
			}
		case policy.EffectiveCredentialPolicy() == "gcp-chain" && name == "project":
			for _, path := range []string{ctx.Env["GOOGLE_APPLICATION_CREDENTIALS"], adcFilePath(ctx)} {
				if path == "" {
					continue
				}
				info, _ := gcpCredentialFile(ctx, path)
				if v := firstStr(info["quota_project_id"], info["project_id"]); v != "" {
					return v
				}
			}
		}
		return ""
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

// ExplainChain is the AUTH-7 walk over a cloud chain (offline).
func ExplainChain(policy AccessPolicy, ctx *ChainContext, explicit bool) ([]ChainStep, bool, error) {
	rungs, err := ChainFor(policy)
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
	rungs, err := ChainFor(policy)
	if err != nil {
		return nil, err
	}
	developerFailed := false
	for _, rung := range rungs {
		got, err := rung.Acquire(ctx)
		if err != nil {
			if IsKind(err, KindAuth) && policy.EffectiveCredentialPolicy() == "azure-chain" && (rung.Name == "az" || rung.Name == "pwsh" || rung.Name == "azd") {
				developerFailed = true
				continue
			}
			return nil, err
		}
		if got != nil {
			return got, nil
		}
	}
	if developerFailed {
		e := chainAuthError("Azure developer credentials failed; sign in with az, Azure PowerShell, or azd")
		e.Provider = policy.Provider
		return nil, e
	}
	hint := "configure the cloud SDK"
	if len(policy.EnvKeys) > 0 {
		hint = "set " + policy.EnvKeys[0] + " or configure the cloud SDK"
	}
	return nil, NotConfiguredErrorf(policy.Provider, policy.EnvKeys, "", "%s: no credential found in the %s chain; %s", policy.Provider, policy.EffectiveCredentialPolicy(), hint)
}

type cachingProvider struct {
	policy AccessPolicy
	ctx    *ChainContext
	mu     sync.Mutex
	value  Credential
}

func (p *cachingProvider) Credential(context.Context) (Credential, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.value != nil && !p.value.IsExpired(p.ctx.now()) {
		return p.value, nil
	}
	value, err := ResolveChain(p.policy, p.ctx)
	if err != nil {
		return nil, err
	}
	if value.IsExpired(p.ctx.now()) {
		e := chainAuthError("cloud credential is expired; renew the configured credential source")
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
		info := wireObj(inputs["credential_file"])
		scope := stringOnly(inputs["scope"])
		tokenURI, assertion, err := GCPServiceAccountAssertion(ctx, info, scope)
		if err != nil {
			return nil, err
		}
		return JSONObject{"method": "POST", "url": tokenURI, "headers": JSONObject{"content-type": "application/x-www-form-urlencoded"}, "body_encoding": "form", "body": JSONObject{"grant_type": jwtBearerGrant, "assertion": assertion}}, nil
	case "environment":
		tokenURL, pairs, err := AzureEnvironmentRequest(ctx, stringOnly(inputs["jti"]))
		if err != nil {
			return nil, err
		}
		body := JSONObject{}
		for _, p := range pairs {
			body[p[0]] = p[1]
		}
		return JSONObject{"method": "POST", "url": tokenURL, "headers": JSONObject{"content-type": "application/x-www-form-urlencoded"}, "body_encoding": "form", "body": body}, nil
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
		if status != 0 || wireInt(body["Version"], 0) != 1 {
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
