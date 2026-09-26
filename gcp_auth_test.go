package lm15

// Google Cloud (Vertex): API keys on the project door, where the project
// comes from, and guidance that names the fix (lm15-contract spec/auth.md
// AUTH-2, AUTH-10, amended 2026-09-26; changes/2026-09-26-vertex-live.md).

import (
	"encoding/base64"
	"strings"
	"testing"
	"time"
)

func vertexAuthHeaders(t *testing.T, credential CredentialLike) map[string]string {
	t.Helper()
	lm, err := NewGeminiLM(WithAPIKey(credential), WithAccess(Vertex), WithSettings(map[string]string{"project": "p", "location": "global"}))
	if err != nil {
		t.Fatal(err)
	}
	req, err := lm.BuildRequest(&Request{Model: "gemini-2.5-flash", Messages: []Message{UserMessage("hi")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	for _, h := range req.Headers {
		if k := strings.ToLower(h[0]); k == "authorization" || k == "x-goog-api-key" {
			out[k] = h[1]
		}
	}
	return out
}

func TestVertexStringIsAKey(t *testing.T) {
	for _, key := range []string{"AQ.Ab8RN6-test", "AIzaSyTestKey", "test-key-123"} {
		got := vertexAuthHeaders(t, key)
		if len(got) != 1 || got["x-goog-api-key"] != key {
			t.Fatalf("%s: %v", key, got)
		}
	}
	if len(Vertex.EnvKeys) != 0 {
		t.Fatal("vertex must not read an ambient key")
	}
}

func TestVertexTokenShapedStringIsBearer(t *testing.T) {
	jwt := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256"}`)) + ".e30.c2ln"
	for _, token := range []string{"ya29.a0-test", jwt} {
		got := vertexAuthHeaders(t, token)
		if len(got) != 1 || got["authorization"] != "Bearer "+token {
			t.Fatalf("%s: %v", token, got)
		}
	}
	if got := vertexAuthHeaders(t, BearerToken{Value: "opaque"}); got["authorization"] != "Bearer opaque" {
		t.Fatal(got)
	}
}

func vertexProject(t *testing.T, env, files map[string]string, http HTTPFunc) (string, string) {
	t.Helper()
	full := map[string]string{"HOME": "/h"}
	for k, v := range env {
		full[k] = v
	}
	ctx := &ChainContext{Env: full, Home: "/h", Files: files, HTTP: http, Now: time.Now}
	sources := map[string]string{}
	var problems []*Error
	out, err := resolveSettingsFull(Vertex.Host, nil, full, "vertex", ProfileSettings(Vertex, ctx),
		settingsOptions{sources: sources, problems: &problems, unprobedOK: http == nil})
	if err != nil {
		t.Fatal(err)
	}
	return out["project"], sources["project"]
}

func TestVertexProjectSources(t *testing.T) {
	cfg := "[core]\naccount = a@example.com\nproject = from-gcloud\n"
	adc := `{"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r", "quota_project_id": "from-adc-quota"}`
	both := map[string]string{"~/.config/gcloud/application_default_credentials.json": adc, "~/.config/gcloud/configurations/config_default": cfg}
	check := func(name string, gotV, gotFrom, wantV, wantFrom string) {
		t.Helper()
		if gotV != wantV || gotFrom != wantFrom {
			t.Fatalf("%s: got (%q, %q), want (%q, %q)", name, gotV, gotFrom, wantV, wantFrom)
		}
	}
	v, f := vertexProject(t, map[string]string{"GOOGLE_CLOUD_PROJECT": "from-env"}, both, nil)
	check("env", v, f, "from-env", "env:GOOGLE_CLOUD_PROJECT")
	v, f = vertexProject(t, map[string]string{"NO_GCE_CHECK": "1"}, both, nil)
	check("gcloud before adc quota", v, f, "from-gcloud", "gcloud-config")
	v, f = vertexProject(t, map[string]string{"CLOUDSDK_CORE_PROJECT": "core"}, both, nil)
	check("core env", v, f, "core", "env:CLOUDSDK_CORE_PROJECT")
	v, f = vertexProject(t, nil, map[string]string{"~/.config/gcloud/application_default_credentials.json": adc}, nil)
	check("adc quota", v, f, "from-adc-quota", "adc-file")
	named := map[string]string{"~/.config/gcloud/active_config": "work\n", "~/.config/gcloud/configurations/config_work": "[core]\nproject = from-work\n"}
	v, f = vertexProject(t, nil, named, nil)
	check("named configuration", v, f, "from-work", "gcloud-config")
	v, f = vertexProject(t, map[string]string{"CLOUDSDK_ACTIVE_CONFIG_NAME": "../../x", "NO_GCE_CHECK": "1"}, map[string]string{"~/x": cfg}, nil)
	check("config name outside gcloud's rule", v, f, "", "missing")
	v, f = vertexProject(t, nil, map[string]string{}, nil)
	check("offline metadata", v, f, "", "unprobed:metadata")
	var asked []string
	online := func(method, url string, headers map[string]string, body []byte, timeout time.Duration) (int, map[string]string, []byte, error) {
		asked = append(asked, url+" "+headers["Metadata-Flavor"])
		return 200, nil, []byte("from-metadata"), nil
	}
	v, f = vertexProject(t, nil, map[string]string{}, online)
	check("online metadata", v, f, "from-metadata", "metadata")
	if len(asked) != 1 || asked[0] != "http://metadata.google.internal/computeMetadata/v1/project/project-id Google" {
		t.Fatal(asked)
	}
}

func TestStaleADCLoginNamesTheWordAndTheCommand(t *testing.T) {
	const secret = "SECRET-SENTINEL-DO-NOT-PRINT"
	ctx := &ChainContext{
		Env:   map[string]string{"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json", "NO_GCE_CHECK": "1"},
		Files: map[string]string{"/creds.json": `{"type": "authorized_user", "client_id": "c", "client_secret": "` + secret + `", "refresh_token": "` + secret + `"}`},
		HTTP: func(method, url string, headers map[string]string, body []byte, timeout time.Duration) (int, map[string]string, []byte, error) {
			return 400, nil, []byte(`{"error": "invalid_grant", "error_description": "reauth ` + secret + `"}`), nil
		},
		Now: time.Now,
	}
	_, err := ResolveChain(Vertex, ctx)
	if err == nil {
		t.Fatal("expected an error")
	}
	e := err.(*Error)
	text := e.Error()
	if !strings.Contains(text, "HTTP 400 (invalid_grant)") || !strings.Contains(text, "gcloud auth application-default login") || strings.Contains(text, secret) {
		t.Fatal(text)
	}
	if e.ProviderCode != "invalid_grant" {
		t.Fatal(e.ProviderCode)
	}
}

func TestVertexWireRefusalGuidance(t *testing.T) {
	body := `{"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "denied"}}`
	token, _ := NewGeminiLM(WithAPIKey(BearerToken{Value: "t"}), WithAccess(Vertex), WithSettings(map[string]string{"project": "p"}))
	if msg := token.NormalizeError(403, body).Error(); !strings.Contains(msg, "roles/aiplatform.user") || strings.Contains(msg, "Check that your API key") {
		t.Fatal(msg)
	}
	if msg := token.NormalizeError(401, body).Error(); !strings.Contains(msg, "access token") {
		t.Fatal(msg)
	}
	key, _ := NewGeminiLM(WithAPIKey("AQ.not-a-real-key"), WithAccess(Vertex), WithSettings(map[string]string{"project": "p"}))
	if msg := key.NormalizeError(401, body).Error(); !strings.Contains(msg, "Vertex AI key") || strings.Contains(msg, "not-a-real-key") {
		t.Fatal(msg)
	}
}
