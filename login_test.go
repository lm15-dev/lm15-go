package lm15

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Managed authentication beyond what the contract's managed direction runs:
// the managed router (AUTH-15 mode B), bound clients (AUTH-20.1, R4),
// Connect (AUTH-23), the loopback listener with real sockets (AUTH-18) and
// the file store (AUTH-25). No network beyond 127.0.0.1.

type recordingTransport struct {
	mu       sync.Mutex
	replies  []string
	requests []*TransportRequest
}

func (t *recordingTransport) Do(_ context.Context, req *TransportRequest) (*TransportResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.requests = append(t.requests, req)
	if len(t.replies) == 0 {
		return nil, errors.New("no scripted reply")
	}
	body := t.replies[0]
	t.replies = t.replies[1:]
	return &TransportResponse{Status: 200, Headers: [][2]string{{"content-type", "application/json"}}, Body: io.NopCloser(strings.NewReader(body))}, nil
}

func (t *recordingTransport) header(name string) string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.requests[len(t.requests)-1].Header(name)
}

const responsesReply = `{"id":"r","object":"response","status":"completed","model":"m","output":[{"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
const chatReply = `{"id":"r","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func request(t *testing.T, model string) *Request {
	t.Helper()
	req, err := NewRequest(model, []Message{UserMessage("hi")})
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func reason(err error) string {
	var e *Error
	if errors.As(err, &e) {
		return e.Reason
	}
	return ""
}

func TestManagedRouterSendsTheSavedKeyNeverTheEnvironments(t *testing.T) {
	ctx := context.Background()
	auth := MemoryAuth()
	if _, err := auth.SetAPIKey(ctx, "openai", "saved-key", ""); err != nil {
		t.Fatal(err)
	}
	transport := &recordingTransport{replies: []string{responsesReply}}
	router, err := NewRouterWithConfig(RouterConfig{Auth: auth, Env: map[string]string{"OPENAI_API_KEY": "ambient-key"}, Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := router.Complete(ctx, request(t, "openai:gpt-test")); err != nil {
		t.Fatal(err)
	}
	if got := transport.header("Authorization"); got != "Bearer saved-key" {
		t.Fatalf("authorization = %q", got)
	}
	report, err := ExplainAuth("openai", ExplainOptions{Env: map[string]string{"OPENAI_API_KEY": "ambient-key"}, Auth: auth})
	if err != nil {
		t.Fatal(err)
	}
	var states []string
	for _, s := range report.Steps {
		states = append(states, s.Kind+"="+s.State)
	}
	if strings.Join(states, " ") != "api_keys=absent connection=selected env:OPENAI_API_KEY=shadowed" {
		t.Fatalf("doctor walk = %v", states)
	}
	if _, err := auth.Logout(ctx, "openai"); err != nil {
		t.Fatal(err)
	}
	fresh, _ := NewRouterWithConfig(RouterConfig{Auth: auth, Env: map[string]string{"OPENAI_API_KEY": "ambient-key"}})
	if _, err := fresh.LM("openai:gpt-test"); reason(err) != "login_required" {
		t.Fatalf("signed out: want login_required, got %v", err)
	}
}

func TestExplicitKeyOutranksTheSavedConnection(t *testing.T) {
	ctx := context.Background()
	auth := MemoryAuth()
	_, _ = auth.SetAPIKey(ctx, "openai", "saved-key", "")
	transport := &recordingTransport{replies: []string{responsesReply}}
	router, _ := NewRouterWithConfig(RouterConfig{Auth: auth, APIKeys: map[string]CredentialLike{"openai": "explicit-key"}, Transport: transport})
	if _, err := router.Complete(ctx, request(t, "openai:gpt-test")); err != nil {
		t.Fatal(err)
	}
	if got := transport.header("Authorization"); got != "Bearer explicit-key" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestManagedRouterRoutesTheConnectionOnlyProviders(t *testing.T) {
	ctx := context.Background()
	auth := MemoryAuth()
	now := auth.now()
	doc := JSONObject{
		{"github-copilot", JSONObject{{"type", "oauth"}, {"access", "tid=1;proxy-ep=proxy.business.githubcopilot.com;tok"}, {"refresh", "gh"}, {"expires", now + 3_600_000}, {"issued_at", now}, {"lifetime_s", 3600.0}}},
		{"_lm15", JSONObject{{"version", 1}, {"slots", JSONObject{{"github-copilot", JSONObject{{"generation", "1"}, {"connection_id", "cn_copilotcopilot01"}, {"revision", "1"}, {"kind", "account"}, {"method_id", "device"}, {"instance_id", "public"}, {"label", "GitHub Copilot"}, {"created_at", "2026-09-25T00:00:00Z"}, {"routes", []any{"github-copilot"}}, {"settings", JSONObject{}}, {"state", "ready"}, {"renewal", "remint"}}}}}}},
	}
	if _, err := mutateStore(ctx, auth.Store(), func(JSONObject) (JSONObject, error) { return doc, nil }); err != nil {
		t.Fatal(err)
	}
	transport := &recordingTransport{replies: []string{chatReply}}
	router, _ := NewRouterWithConfig(RouterConfig{Auth: auth, Transport: transport})
	if _, err := router.Complete(ctx, request(t, "github-copilot:gpt-4.1")); err != nil {
		t.Fatal(err)
	}
	if url := transport.requests[0].URL; !strings.HasPrefix(url, "https://api.business.githubcopilot.com/") {
		t.Fatalf("url = %s", url)
	}
	if got := transport.header("Editor-Version"); got != "vscode/1.107.0" {
		t.Fatalf("editor-version = %q", got)
	}
	if _, err := NewRouter().LM("github-copilot:gpt-4.1"); err == nil {
		t.Fatal("github-copilot must not route without a managed Auth")
	}
}

func TestBoundClientFollowsRenewalsOnly(t *testing.T) {
	ctx := context.Background()
	auth := MemoryAuth()
	first, _ := auth.SetAPIKey(ctx, "openai", "key-1", "")
	transport := &recordingTransport{replies: []string{responsesReply}}
	client, err := NewBoundClient(auth, ModelSelection{Provider: "openai", Model: "gpt-test", ConnectionID: first.ID, IdentityGeneration: first.IdentityGeneration}, RouterConfig{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Ask(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	if got := transport.header("Authorization"); got != "Bearer key-1" {
		t.Fatalf("authorization = %q", got)
	}
	if _, err := client.Complete(ctx, request(t, "anthropic:claude")); reason(err) != "selection_mismatch" {
		t.Fatalf("want selection_mismatch, got %v", err)
	}
	_, _ = auth.SetAPIKey(ctx, "openai", "key-2", first.ID)
	if _, err := client.Ask(ctx, "hi"); reason(err) != "connection_changed" {
		t.Fatalf("want connection_changed, got %v", err)
	}
	_, _ = auth.Logout(ctx, "openai")
	if _, err := client.Ask(ctx, "hi"); reason(err) != "login_required" {
		t.Fatalf("want login_required, got %v", err)
	}
}

type scriptUI struct {
	answers []string
	asked   []string
	notices []string
}

func (u *scriptUI) Prompt(_ context.Context, p Prompt) (string, error) {
	u.asked = append(u.asked, p.FieldID)
	if len(u.answers) == 0 {
		return "", ErrPromptCancelled
	}
	a := u.answers[0]
	u.answers = u.answers[1:]
	return a, nil
}

func (u *scriptUI) Notify(n Notice) {
	if n.Type == "info" {
		u.notices = append(u.notices, n.Message)
	}
}

func TestConnectWalksProviderKeyAndModel(t *testing.T) {
	ctx := context.Background()
	auth := NewAuthWithSeams(NewMemoryStore(), AuthSeams{Env: func(string) string { return "" }})
	ui := &scriptUI{answers: []string{"openai", "typed-key", "__manual__", "gpt-test"}}
	client, err := Connect(ctx, ConnectOptions{Auth: auth, UI: ui, RouterConfig: RouterConfig{Transport: &recordingTransport{}}})
	if err != nil {
		t.Fatal(err)
	}
	if client.Routed() != "openai:gpt-test" {
		t.Fatalf("routed = %s", client.Routed())
	}
	if strings.Join(ui.asked, ",") != "provider,key,model,model" {
		t.Fatalf("asked = %v", ui.asked)
	}
	got, err := auth.RequestAuth(ctx, "openai", nil)
	if err != nil || got.Credential != "typed-key" {
		t.Fatalf("saved key = %q, %v", got.Credential, err)
	}
	if len(ui.notices) == 0 || !strings.Contains(strings.Join(ui.notices, "\n"), "Could not list models") {
		t.Fatalf("notices = %v", ui.notices)
	}
	if !isTerminal(os.Stdin) {
		if _, err := Connect(ctx, ConnectOptions{Auth: MemoryAuth()}); reason(err) != "interaction_required" {
			t.Fatalf("no terminal: want interaction_required, got %v", err)
		}
	}
}

func get(t *testing.T, url string) int {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode
}

func TestLoopbackListenerChecksPathAndStateAndIsOneUse(t *testing.T) {
	l, err := openCallbackListener("/cb", "S", true, 0, "127.0.0.1", "")
	if err != nil {
		t.Fatal(err)
	}
	base := l.redirectURI
	if !strings.HasPrefix(base, "http://127.0.0.1:") || !strings.HasSuffix(base, "/cb") {
		t.Fatalf("redirect uri = %s", base)
	}
	for url, want := range map[string]int{
		strings.TrimSuffix(base, "/cb") + "/other?code=c&state=S": 404,
		base + "?code=c&state=wrong":                              400,
		base + "?error=access_denied&state=wrong":                 400,
		base + "?code=c&state=S&state=S":                          400,
	} {
		if got := get(t, url); got != want {
			t.Errorf("%s: %d, want %d", url, got, want)
		}
	}
	if got := get(t, base+"?code=the-code&state=S"); got != 200 {
		t.Fatalf("valid return: %d", got)
	}
	result := <-l.results
	if result.err != nil || result.value.code != "the-code" {
		t.Fatalf("return = %+v", result)
	}
	busy, _ := openCallbackListener("/cb", "", false, 0, "127.0.0.1", "")
	defer busy.stop()
	port, _ := strconv.Atoi(busy.redirectURI[strings.LastIndex(busy.redirectURI, ":")+1 : strings.LastIndex(busy.redirectURI, "/")])
	if _, err := openCallbackListener("/cb", "", false, port, "127.0.0.1", ""); reason(err) != "method_unavailable" {
		t.Fatalf("busy port: want method_unavailable, got %v", err)
	}
	if _, err := openCallbackListener("/cb", "", false, 0, "0.0.0.0", ""); err == nil {
		t.Fatal("a wildcard bind must be refused")
	}
}

func TestFileStoreSharesTheLayoutAndNeverOverwritesAnUnreadableFile(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	path := filepath.Join(dir, "credentials.json")
	// As lm15-python writes it (a float lifetime, insertion order).
	python := `{"openai": {"type": "api_key", "key": "py-key"}, "_lm15": {"version": 1, "slots": {"openai": {"generation": "1", "connection_id": "cn_pythonwrote0001", "revision": "1", "kind": "api_key", "method_id": "api_key", "instance_id": "public", "label": "openai API key", "created_at": "2026-09-25T00:00:00Z", "routes": ["openai"], "settings": {}, "state": "ready", "renewal": "none"}}}}`
	if err := os.WriteFile(path, []byte(python), 0o600); err != nil {
		t.Fatal(err)
	}
	store, _ := NewFileStore(path)
	auth := NewAuth(store)
	got, err := auth.RequestAuth(ctx, "openai", nil)
	if err != nil || got.Credential != "py-key" {
		t.Fatalf("read the Python-written key: %q, %v", got.Credential, err)
	}
	if _, err := auth.Logout(ctx, "openai"); err != nil {
		t.Fatal(err)
	}
	var after JSONObject
	data, _ := os.ReadFile(path)
	_ = json.NewDecoder(bytes.NewReader(data)).Decode(&after)
	slot := after.Get("_lm15").(JSONObject).Get("slots").(JSONObject).Get("openai").(JSONObject)
	if slot.Get("logged_out") != true || after.Get("openai") != nil {
		t.Fatalf("after logout: %s", data)
	}
	_ = os.WriteFile(path, []byte("{broken"), 0o600)
	if _, err := auth.SetAPIKey(ctx, "openai", "k", ""); reason(err) != "storage_unavailable" {
		t.Fatalf("broken store: want storage_unavailable, got %v", err)
	}
	if data, _ := os.ReadFile(path); string(data) != "{broken" {
		t.Fatal("an unreadable store was overwritten")
	}
}
