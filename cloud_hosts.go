package lm15

import (
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/lm15-dev/lm15-go/internal/sigv4"
)

// A dialect reaches a cloud door through a host (AUTH-10). Three pure
// functions, in the order an adapter calls them: resolveSettings,
// renderBaseURL, finishRequest (then signRequest after serialization).

// profileFunc is the cloud's own configuration for a setting, as (value,
// from) in the AUTH-10 vocabulary; ("", "metadata") means only the
// metadata server could answer and this context is offline.
type profileFunc func(string) (string, string)

// settingsOptions are resolveSettingsFull's optional parts.
type settingsOptions struct {
	endpoint string
	// sources receives each setting's origin: explicit, env:<VAR>, adc-env,
	// gcloud-config, adc-file, metadata, aws-profile, default, missing,
	// unprobed:<from>.
	sources map[string]string
	// problems (the doctor) receives missing-setting errors instead of a
	// return error; the settings that did resolve are returned.
	problems *[]*Error
	// unprobedOK (the doctor): a setting only a network source could supply
	// is left out and recorded unprobed.
	unprobedOK bool
}

// resolveSettings: explicit values, then env (when given), then the
// cloud's own configuration, then defaults (AUTH-10).
func resolveSettings(host *HostSpec, given map[string]string, env map[string]string, provider string, profile profileFunc) (map[string]string, error) {
	return resolveSettingsFull(host, given, env, provider, profile, settingsOptions{})
}

// resolveSettingsWithEndpoint is resolveSettings when the caller named an
// endpoint root: the settings only the URL root needed are optional
// (HostSpec.URLOnlySettings; AUTH-10, amended 2026-09-19).
func resolveSettingsWithEndpoint(host *HostSpec, given map[string]string, env map[string]string, provider string, profile profileFunc, endpoint string) (map[string]string, error) {
	return resolveSettingsFull(host, given, env, provider, profile, settingsOptions{endpoint: endpoint})
}

func resolveSettingsFull(host *HostSpec, given map[string]string, env map[string]string, provider string, profile profileFunc, opts settingsOptions) (map[string]string, error) {
	out := map[string]string{}
	if host == nil {
		for k, v := range given {
			out[k] = v
		}
		return out, nil
	}
	relaxed := map[string]bool{}
	if opts.endpoint != "" {
		relaxed = host.URLOnlySettings()
	}
	remaining := map[string]string{}
	for k, v := range given {
		remaining[k] = v
	}
	record := opts.sources
	if record == nil {
		record = map[string]string{}
	}
	name := provider
	if name == "" {
		name = "host"
	}
	var missing *Error
	for _, setting := range host.Settings {
		value := remaining[setting.Name]
		delete(remaining, setting.Name)
		origin := ""
		if value != "" {
			origin = "explicit"
		}
		if value == "" && env != nil {
			for _, v := range setting.Env {
				if candidate := env[v]; candidate != "" {
					value, origin = candidate, "env:"+v
					break
				}
			}
		}
		unprobed := ""
		if value == "" && profile != nil {
			v, from := profile(setting.Name)
			if v != "" {
				value, origin = v, from
			} else {
				unprobed = from
			}
		}
		if value == "" && setting.Default != "" {
			value, origin = setting.Default, "default"
		}
		if value == "" {
			if relaxed[setting.Name] {
				continue
			}
			if unprobed != "" && opts.unprobedOK {
				record[setting.Name] = "unprobed:" + unprobed
				continue
			}
			hint := "pass settings={'" + setting.Name + "': ...}"
			if len(setting.Env) > 0 {
				hint = "set " + strings.Join(setting.Env, " or ")
			}
			if setting.Name == "project" {
				// The Google project also comes from gcloud, the credential
				// files and the metadata server; those said nothing (AUTH-10).
				hint += ", run `gcloud config set project <id>`, or pass settings={'project': ...}"
			}
			if host.URLOnlySettings()[setting.Name] && len(host.EndpointEnv) > 0 {
				hint += ", or the endpoint: " + strings.Join(host.EndpointEnv, " or ")
			}
			record[setting.Name] = "missing"
			if missing == nil {
				missing = NotConfiguredErrorf(provider, nil, hint, "%s: setting %q is required and has no default; %s", name, setting.Name, hint)
			}
			continue
		}
		out[setting.Name] = value
		record[setting.Name] = origin
	}
	if len(remaining) > 0 {
		var unknown []string
		for k := range remaining {
			unknown = append(unknown, k)
		}
		return nil, valueErrorf("%s: unknown host setting(s) %v; known: %v", name, sortStrings(unknown), host.SettingNames())
	}
	if missing != nil {
		if opts.problems == nil {
			return nil, missing
		}
		*opts.problems = append(*opts.problems, missing)
	}
	return out, nil
}

// locationHost is the Vertex host for a location.
func locationHost(location string) string {
	switch location {
	case "global":
		return "aiplatform.googleapis.com"
	case "us", "eu":
		return "aiplatform." + location + ".rep.googleapis.com"
	}
	return location + "-aiplatform.googleapis.com"
}

var dnsLabelRe = regexp.MustCompile(`^[A-Za-z0-9-]+$`)

// EndpointFromEnv is the first non-empty vendor endpoint variable this
// door honours (AUTH-10, amended 2026-09-19).
func EndpointFromEnv(host *HostSpec, env map[string]string) string {
	if host == nil || env == nil {
		return ""
	}
	for _, name := range host.EndpointEnv {
		if v := strings.TrimSpace(env[name]); v != "" {
			return v
		}
	}
	return ""
}

// JoinEndpoint is an endpoint root (the caller's or the vendor variable's)
// joined with the door's path. The door's path is appended unless the
// endpoint already ends with it, or with a leading part of it: the console
// shows an account root (https://acct.services.ai.azure.com), Microsoft's
// own examples show …/anthropic and …/openai/v1, and all three must mean
// the same door. Stated trade-off: a gateway whose own path happens to end
// with a leading part of the door's path cannot be spelled; none is known.
func JoinEndpoint(endpoint, path, provider string) (string, error) {
	who := provider
	if who == "" {
		who = "host"
	}
	u, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", NotConfiguredErrorf(provider, nil, "", "%s: endpoint must be an http(s) URL with a host, got %q", who, endpoint)
	}
	if u.RawQuery != "" || u.Fragment != "" || u.User != nil {
		return "", NotConfiguredErrorf(provider, nil, "", "%s: endpoint must not carry a query, fragment or userinfo", who)
	}
	split := func(p string) []string {
		var out []string
		for _, seg := range strings.Split(p, "/") {
			if seg != "" {
				out = append(out, seg)
			}
		}
		return out
	}
	given, door := split(u.Path), split(path)
	base := given
	limit := len(given)
	if len(door) < limit {
		limit = len(door)
	}
	for k := limit; k > 0; k-- {
		match := true
		for i := 0; i < k; i++ {
			if given[len(given)-k+i] != door[i] {
				match = false
				break
			}
		}
		if match {
			base = given[:len(given)-k]
			break
		}
	}
	joined := strings.Join(append(append([]string{}, base...), door...), "/")
	out := u.Scheme + "://" + u.Host
	if joined != "" {
		out += "/" + joined
	}
	return out, nil
}

// renderBaseURL renders the host template over the settings.
func renderBaseURL(host HostSpec, settings map[string]string) (string, error) {
	return renderBaseURLAt(host, settings, "", "")
}

// renderBaseURLAt renders the base URL for the settings; with an endpoint
// (a full URL root) the template's root is replaced and the door's path
// rendered and appended (JoinEndpoint).
func renderBaseURLAt(host HostSpec, settings map[string]string, endpoint, provider string) (string, error) {
	values := map[string]string{}
	for k, v := range settings {
		values[k] = v
	}
	for _, name := range []string{"region", "resource", "location"} {
		if v, ok := values[name]; ok && !dnsLabelRe.MatchString(v) {
			return "", NotConfiguredErrorf("", nil, "", "host setting %q must be a DNS label", name)
		}
	}
	if p, ok := values["project"]; ok {
		values["project"] = url.PathEscape(p)
	}
	if loc, ok := values["location"]; ok {
		if _, has := values["location_host"]; !has {
			values["location_host"] = locationHost(loc)
		}
	}
	out := host.BaseURL
	if endpoint != "" {
		out = host.PathTemplate()
	}
	for {
		start := strings.Index(out, "{")
		if start < 0 {
			break
		}
		end := strings.Index(out[start:], "}")
		if end < 0 {
			break
		}
		name := out[start+1 : start+end]
		v, ok := values[name]
		if !ok {
			return "", NotConfiguredErrorf("", nil, "", "host base URL needs setting %q", name)
		}
		out = out[:start] + v + out[start+end+1:]
	}
	if endpoint != "" {
		return JoinEndpoint(endpoint, out, provider)
	}
	return out, nil
}

type finishedRequest struct {
	url     string
	headers [][2]string
	payload any
	params  map[string]string
}

// finishRequest applies the host's closed set of rewrites (endpoint path
// override, model into the path, anthropic_version into the body, required
// headers, query-key) before serialization.
func finishRequest(policy AccessPolicy, settings map[string]string, baseURL string, spec emitSpec, headers [][2]string, credential Credential) (finishedRequest, error) {
	params := map[string]string{}
	for k, v := range spec.params {
		params[k] = v
	}
	out := finishedRequest{url: spec.url, headers: headers, payload: spec.payload, params: params}
	host := policy.Host
	if host == nil {
		return out, nil
	}
	if host.EffectiveStreamFraming() != "sse" && spec.stream {
		return out, UnsupportedFeatureErrorf(policy.Provider, "%s: %s stream framing is not implemented yet (phase 2)", policy.Provider, host.EffectiveStreamFraming())
	}
	key := spec.endpoint
	if spec.endpoint != "" && spec.stream {
		if _, ok := host.Paths[spec.endpoint+"/stream"]; ok {
			key = spec.endpoint + "/stream"
		}
	}
	if path, ok := host.Paths[key]; ok && key != "" {
		if strings.Contains(path, "{model}") && spec.model == "" {
			return out, valueErrorf("%s: endpoint %q needs the model in the path", policy.Provider, spec.endpoint)
		}
		model := spec.model
		if spec.endpoint == "generateContent" {
			model = strings.TrimPrefix(model, "models/")
		}
		out.url = strings.TrimRight(baseURL, "/") + strings.ReplaceAll(path, "{model}", quoteSafe(model, ":@"))
	}
	if payload, ok := asObject(spec.payload); ok {
		copied := copyObject(payload)
		if host.EffectiveModelIn() == "path" {
			copied.Delete("model")
		}
		if v := host.EffectiveAnthropicVersionIn(); strings.HasPrefix(v, "body:") {
			copied.Set("anthropic_version", strings.TrimPrefix(v, "body:"))
			var kept [][2]string
			for _, h := range out.headers {
				if !strings.EqualFold(h[0], "anthropic-version") {
					kept = append(kept, h)
				}
			}
			out.headers = kept
		}
		out.payload = copied
	}
	for _, rh := range host.RequiredHeaders {
		value := settings[rh[1]]
		if value == "" {
			return out, NotConfiguredErrorf(policy.Provider, nil, "", "%s: header %s needs setting %q", policy.Provider, rh[0], rh[1])
		}
		out.headers = append(out.headers, [2]string{rh[0], value})
	}
	if key, ok := credential.(APIKey); ok && inVocab("query-key", policy.EffectiveAuthScheme()) {
		if scheme, err := SelectScheme(policy, key); err == nil && scheme == "query-key" {
			out.params["key"] = key.Value
		}
	}
	return out, nil
}

// signRequest returns the headers to send under sigv4.
func signRequest(policy AccessPolicy, settings map[string]string, req *TransportRequest, credential AwsCredentials, now time.Time) ([][2]string, error) {
	host := policy.Host
	if host == nil || host.SigV4Service == "" {
		return nil, NotConfiguredErrorf(policy.Provider, nil, "", "%s: AWS credentials need a sigv4 host", policy.Provider)
	}
	region := settings["region"]
	if region == "" {
		return nil, NotConfiguredErrorf(policy.Provider, nil, "", "%s: sigv4 needs the region setting", policy.Provider)
	}
	headers := map[string]string{}
	for _, h := range req.Headers {
		lk := strings.ToLower(h[0])
		if lk == "authorization" || lk == "x-api-key" {
			continue
		}
		headers[h[0]] = h[1]
	}
	sig := sigv4.Sign(req.Method, req.URL, headers, req.Body, sigv4.Credentials{
		AccessKeyID: credential.AccessKeyID, SecretAccessKey: credential.SecretAccessKey, SessionToken: credential.SessionToken,
	}, region, host.SigV4Service, now)
	names := make([]string, 0, len(sig.Headers))
	for k := range sig.Headers {
		names = append(names, k)
	}
	names = sortStrings(names)
	out := make([][2]string, 0, len(names))
	for _, k := range names {
		out = append(out, [2]string{k, sig.Headers[k]})
	}
	return out, nil
}

// SigV4Sign exposes the signer for the vet protocol and tests.
func SigV4Sign(method, rawURL string, headers map[string]string, payload []byte, creds AwsCredentials, region, service string, now time.Time) sigv4.Signature {
	return sigv4.Sign(method, rawURL, headers, payload, sigv4.Credentials{
		AccessKeyID: creds.AccessKeyID, SecretAccessKey: creds.SecretAccessKey, SessionToken: creds.SessionToken,
	}, region, service, now)
}
