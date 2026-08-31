package auth

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const sentinel = "SECRET-SENTINEL-DO-NOT-PRINT"

func TestCredentialFuncIsInvokedPerCall(t *testing.T) {
	calls := 0
	provider := CredentialFunc(func() (string, error) {
		calls++
		return fmt.Sprintf("token-%d", calls), nil
	})
	first, _ := provider.Token()
	second, _ := provider.Token()
	if first == second {
		t.Fatal("credential provider must be consulted per call (AUTH-2)")
	}
}

func TestStaticCredentialRedactsEveryRendering(t *testing.T) {
	credential := Static(sentinel)
	for _, rendering := range []string{
		fmt.Sprintf("%v", credential),
		fmt.Sprintf("%+v", credential),
		fmt.Sprintf("%#v", credential),
		fmt.Sprint(credential),
	} {
		if strings.Contains(rendering, sentinel) {
			t.Errorf("sentinel leaked: %s", rendering)
		}
	}
}

func TestUnderscoreAliasIsAccepted(t *testing.T) {
	report, err := ExplainAuth("openai_chat", ExplainOptions{Env: map[string]string{}})
	if err != nil {
		t.Fatal(err)
	}
	if report.Provider != "openai-chat" {
		t.Fatalf("provider = %q", report.Provider)
	}
}

func TestUnknownProviderNamesKnownOnes(t *testing.T) {
	_, err := ExplainAuth("not-a-provider", ExplainOptions{})
	if err == nil || !strings.Contains(err.Error(), "anthropic") {
		t.Fatalf("error should list known providers, got: %v", err)
	}
}

func TestGeminiEnvKeyOrderFirstWins(t *testing.T) {
	report, err := ExplainAuth("gemini", ExplainOptions{Env: map[string]string{
		"GEMINI_API_KEY": "a",
		"GOOGLE_API_KEY": "b",
	}})
	if err != nil {
		t.Fatal(err)
	}
	selected, ok := report.Selected()
	if !ok || selected.Kind != "env:GEMINI_API_KEY" {
		t.Fatalf("selected = %+v", selected)
	}
	if report.Steps[2].Kind != "env:GOOGLE_API_KEY" || report.Steps[2].State != Shadowed {
		t.Fatalf("GOOGLE_API_KEY should be shadowed: %+v", report.Steps[2])
	}
}

func fakeCodexJWT(t *testing.T, accountID string, expiresAt time.Time) string {
	t.Helper()
	encode := func(value map[string]any) string {
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return base64.RawURLEncoding.EncodeToString(raw)
	}
	header := encode(map[string]any{"alg": "none", "typ": "JWT"})
	payload := encode(map[string]any{
		"exp": expiresAt.Unix(),
		"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": accountID},
	})
	return header + "." + payload + ".signature"
}

func TestReadCodexCredentialDecodesJWTExpiryAndAccount(t *testing.T) {
	token := fakeCodexJWT(t, "acct_test", time.Now().Add(time.Hour))
	path := filepath.Join(t.TempDir(), "auth.json")
	raw, _ := json.Marshal(map[string]any{
		"auth_mode": "chatgpt",
		"tokens":    map[string]any{"access_token": token, "refresh_token": "rt"},
	})
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	credential, err := ReadCodexCLICredential(path)
	if err != nil {
		t.Fatal(err)
	}
	if credential.Expired() {
		t.Error("credential should be fresh")
	}
	if credential.AccountID() != "acct_test" {
		t.Errorf("accountID = %q", credential.AccountID())
	}
	if !credential.HasRefreshToken() {
		t.Error("refresh token should be present")
	}
}

func TestCodexExpiredJWTIsExpiredWithSkew(t *testing.T) {
	token := fakeCodexJWT(t, "acct", time.Now().Add(2*time.Minute)) // inside 5min skew
	path := filepath.Join(t.TempDir(), "auth.json")
	raw, _ := json.Marshal(map[string]any{"tokens": map[string]any{"access_token": token}})
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	credential, err := ReadCodexCLICredential(path)
	if err != nil {
		t.Fatal(err)
	}
	if !credential.Expired() {
		t.Error("token inside the skew window must count as expired (AUTH-3)")
	}
}

func TestMissingFilesAreTypedNotConfigured(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "nope.json")
	if _, err := ReadClaudeCodeCredential(missing); !IsNotConfigured(err) {
		t.Errorf("claude: want not-configured, got %v", err)
	}
	if _, err := ReadCodexCLICredential(missing); !IsNotConfigured(err) {
		t.Errorf("codex: want not-configured, got %v", err)
	}
}

func TestCredentialRenderingsRedactTokens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "credentials.json")
	raw, _ := json.Marshal(map[string]any{"claudeAiOauth": map[string]any{
		"accessToken":  sentinel,
		"refreshToken": sentinel,
		"expiresAt":    time.Now().UnixMilli() + 60_000,
	}})
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	credential, err := ReadClaudeCodeCredential(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, rendering := range []string{
		fmt.Sprintf("%v", credential),
		fmt.Sprintf("%+v", credential),
		fmt.Sprintf("%#v", credential),
		fmt.Sprint(credential),
	} {
		if strings.Contains(rendering, sentinel) {
			t.Errorf("sentinel leaked: %s", rendering)
		}
	}
	if credential.AccessToken() != sentinel {
		t.Error("accessor must return the real token")
	}
}
