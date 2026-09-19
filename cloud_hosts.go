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

// resolveSettings fills the host's settings from explicit values, then env
// (when given), then a profile lookup, then defaults; a required setting
// with no value raises NotConfiguredError naming the variable.
func resolveSettings(host *HostSpec, given map[string]string, env map[string]string, provider string, profile func(string) string) (map[string]string, error) {
	out := map[string]string{}
	if host == nil {
		for k, v := range given {
			out[k] = v
		}
		return out, nil
	}
	remaining := map[string]string{}
	for k, v := range given {
		remaining[k] = v
	}
	for _, setting := range host.Settings {
		value := remaining[setting.Name]
		delete(remaining, setting.Name)
		if value == "" && env != nil {
			for _, v := range setting.Env {
				if candidate := env[v]; candidate != "" {
					value = candidate
					break
				}
			}
		}
		if value == "" && profile != nil {
			value = profile(setting.Name)
		}
		if value == "" {
			value = setting.Default
		}
		if value == "" {
			name := provider
			if name == "" {
				name = "host"
			}
			hint := "pass settings={'" + setting.Name + "': ...}"
			if len(setting.Env) > 0 {
				hint = "set " + strings.Join(setting.Env, " or ")
			}
			return nil, NotConfiguredErrorf(provider, nil, hint, "%s: setting %q is required and has no default; %s", name, setting.Name, hint)
		}
		out[setting.Name] = value
	}
	if len(remaining) > 0 {
		var unknown []string
		for k := range remaining {
			unknown = append(unknown, k)
		}
		name := provider
		if name == "" {
			name = "host"
		}
		return nil, valueErrorf("%s: unknown host setting(s) %v; known: %v", name, sortStrings(unknown), host.SettingNames())
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

// renderBaseURL renders the host template over the settings.
func renderBaseURL(host HostSpec, settings map[string]string) (string, error) {
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
	if payload, ok := spec.payload.(JSONObject); ok {
		copied := copyObject(payload)
		if host.EffectiveModelIn() == "path" {
			delete(copied, "model")
		}
		if v := host.EffectiveAnthropicVersionIn(); strings.HasPrefix(v, "body:") {
			copied["anthropic_version"] = strings.TrimPrefix(v, "body:")
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
