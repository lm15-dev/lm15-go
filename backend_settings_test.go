package lm15

// AUTH-10 backend settings (amended 2026-09-30) and MAP-7 rule 6's default
// max_tokens: lm15-contract changes/2026-09-30-claude-code-client-version.md.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const refusal = "Claude Code 2.1.170 does not support this model; version 2.1.280 or newer is required. Run 'claude update', or update the Claude desktop app, then try again."

func userAgent(t *testing.T, lm LM) string {
	t.Helper()
	req, err := lm.BuildRequest(&Request{Model: "claude-opus-5-5", Messages: []Message{UserMessage("hi")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	return req.Header("user-agent")
}

// scratchHome writes the two CLI login files a routed subscription door reads.
func scratchHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	expires := time.Now().Add(time.Hour).UnixMilli()
	claude := filepath.Join(home, ".claude", ".credentials.json")
	codex := filepath.Join(home, ".codex", "auth.json")
	for _, dir := range []string{filepath.Dir(claude), filepath.Dir(codex)} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	body, _ := json.Marshal(map[string]any{"claudeAiOauth": map[string]any{"accessToken": "tok", "refreshToken": "r", "expiresAt": expires}})
	if err := os.WriteFile(claude, body, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codex, []byte(`{"tokens": {"access_token": "tok", "refresh_token": "r", "account_id": "acct"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	// The stored-login reader uses the process's home (os.UserHomeDir), never the
	// machine's real login: CI has none, and a developer's must not decide the test.
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	return home
}

func TestBackendSettingsTableDefault(t *testing.T) {
	if DefaultClaudeCodeVersion != "2.1.285" || ClaudeCode.BackendOptions["client_version"] != DefaultClaudeCodeVersion {
		t.Fatalf("table default %q / %v", DefaultClaudeCodeVersion, ClaudeCode.BackendOptions)
	}
	if got := ClaudeCode.BackendSettings; len(got) != 1 || got[0].Env[0] != "LM15_CLAUDE_CODE_VERSION" {
		t.Fatalf("claude-code backend settings %v", got)
	}
	if got := OpenAICodex.BackendSettings; len(got) != 1 || got[0].Env[0] != "LM15_CODEX_CLIENT_VERSION" {
		t.Fatalf("codex backend settings %v", got)
	}
	lm, err := NewClaudeCodeLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	if got := userAgent(t, lm); got != "claude-cli/2.1.285" {
		t.Fatalf("default user-agent %q", got)
	}
}

func TestBackendSettingsMoveTheHeader(t *testing.T) {
	bySetting, err := NewClaudeCodeLM(WithAPIKey("k"), WithSettings(map[string]string{"client_version": "2.1.280"}))
	if err != nil {
		t.Fatal(err)
	}
	byOption, err := NewClaudeCodeLM(WithAPIKey("k"), WithClaudeCodeVersion("2.1.280"))
	if err != nil {
		t.Fatal(err)
	}
	byPolicy, err := NewAnthropicLM(WithAPIKey("k"), WithAccess(ClaudeCode), WithSettings(map[string]string{"client_version": "2.1.280"}))
	if err != nil {
		t.Fatal(err)
	}
	for _, lm := range []LM{bySetting, byOption, byPolicy} {
		if got := userAgent(t, lm); got != "claude-cli/2.1.280" {
			t.Fatalf("user-agent %q", got)
		}
	}
	if _, err := NewClaudeCodeLM(WithAPIKey("k"), WithClaudeCodeVersion("1"), WithSettings(map[string]string{"client_version": "2"})); err == nil {
		t.Fatal("two different versions were accepted")
	}
	t.Setenv("LM15_CLAUDE_CODE_VERSION", "9.9.9")
	byHand, err := NewClaudeCodeLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	if got := userAgent(t, byHand); got != "claude-cli/2.1.285" {
		t.Fatalf("an adapter built by hand read the environment: %q", got)
	}
}

func TestBackendSettingsThroughTheRouter(t *testing.T) {
	home := scratchHome(t)
	route := func(config RouterConfig, model string) LM {
		t.Helper()
		router, err := NewRouterWithConfig(config)
		if err != nil {
			t.Fatal(err)
		}
		lm, err := router.LM(model)
		if err != nil {
			t.Fatal(err)
		}
		return lm
	}
	explicit := route(RouterConfig{Env: map[string]string{"HOME": home, "LM15_CLAUDE_CODE_VERSION": "2.1.282"},
		Settings: map[string]map[string]string{"claude_code": {"client_version": "2.1.281"}}}, "claude-code:claude-opus-5-5")
	if got := userAgent(t, explicit); got != "claude-cli/2.1.281" {
		t.Fatalf("explicit: %q", got)
	}
	fromEnv := route(RouterConfig{Env: map[string]string{"HOME": home, "LM15_CLAUDE_CODE_VERSION": "2.1.282"}}, "claude-code:claude-opus-5-5")
	if got := userAgent(t, fromEnv); got != "claude-cli/2.1.282" {
		t.Fatalf("env: %q", got)
	}
	byDefault := route(RouterConfig{Env: map[string]string{"HOME": home}}, "claude-code:claude-opus-5-5")
	if got := userAgent(t, byDefault); got != "claude-cli/2.1.285" {
		t.Fatalf("default: %q", got)
	}
	codex := route(RouterConfig{Env: map[string]string{"HOME": home, "LM15_CODEX_CLIENT_VERSION": "0.151.0"}}, "openai-codex:gpt-5.4-mini")
	if got := codex.Access().BackendOptions["client_version"]; got != "0.151.0" {
		t.Fatalf("codex: %q", got)
	}
	if lm, err := NewOpenAICodexLM(WithAPIKey("k"), WithAccountID("a"), WithCodexClientVersion("0.150.0")); err != nil || lm.Access().BackendOptions["client_version"] != "0.150.0" {
		t.Fatalf("codex option: %v", err)
	}
}

func TestBackendSettingNothingReadsIsRefused(t *testing.T) {
	if _, err := NewClaudeCodeLM(WithAPIKey("k"), WithSettings(map[string]string{"version": "2.1.280"})); err == nil || !strings.Contains(err.Error(), "known: client_version") {
		t.Fatalf("unknown setting: %v", err)
	}
	router, err := NewRouterWithConfig(RouterConfig{APIKeys: map[string]CredentialLike{"anthropic": "k"},
		Settings: map[string]map[string]string{"anthropic": {"client_version": "1"}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := router.LM("anthropic:claude-opus-5-5"); err == nil || !strings.Contains(err.Error(), "this door takes no settings") {
		t.Fatalf("settings on a door that reads none: %v", err)
	}
}

func TestBackendSettingsInTheDoctor(t *testing.T) {
	report, err := ExplainAuth("claude-code", ExplainOptions{Env: map[string]string{"LM15_CLAUDE_CODE_VERSION": "2.1.290"}, ClaudeCredentialsPath: "/nonexistent"})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Settings) != 1 || report.Settings[0] != [2]string{"client_version", "2.1.290"} ||
		report.SettingSources[0] != [2]string{"client_version", "env:LM15_CLAUDE_CODE_VERSION"} {
		t.Fatalf("report %v %v", report.Settings, report.SettingSources)
	}
	if !strings.Contains(report.Describe(), "setting client_version: 2.1.290 (from env $LM15_CLAUDE_CODE_VERSION)") {
		t.Fatalf("describe: %s", report.Describe())
	}
}

func TestClaudeCodeMinimumVersionRefusal(t *testing.T) {
	body := `{"type": "error", "error": {"type": "invalid_request_error", "message": "` + refusal + `"}, "request_id": "req_1"}`
	lm, err := NewClaudeCodeLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	e := lm.NormalizeError(400, body)
	want := refusal + "\n\n  To fix:\n    - lm15 sends this version itself; updating Claude Code does not change it\n    - Set the claude-code setting client_version to 2.1.280 or newer (or LM15_CLAUDE_CODE_VERSION=2.1.280)\n"
	if e.Kind != KindInvalidRequest || e.Message != want {
		t.Fatalf("got %s %q", e.Kind, e.Message)
	}
	api, _ := NewAnthropicLM(WithAPIKey("k"))
	if got := api.NormalizeError(400, body).Message; got != refusal {
		t.Fatalf("api door: %q", got)
	}
}

func TestAnthropicDefaultMaxTokensIsTheModelsCeiling(t *testing.T) {
	cases := []struct {
		model          string
		budget         int
		wire, recorded int
	}{
		{"claude-opus-5-5", 0, 128000, 128000},
		{"claude-haiku-4-5", 0, 64000, 64000},
		{"claude-sonnet-4-5", 32768, 64000, 64000 - 32768},
		{"claude-haiku-4-5", 64000, 64000 + 16384, 16384},
		{"anthropic.claude-haiku-4-5-20251001-v1:0", 0, 64000, 64000},
		{"claude-3-5-haiku-20241022", 0, 8192, 8192},
		{"deepseek-v4-flash", 0, 16384, 16384},
	}
	lm, err := NewAnthropicLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		req := &Request{Model: c.model, Messages: []Message{UserMessage("hi")}}
		if c.budget > 0 {
			budget := c.budget
			req.Config.Reasoning = &Reasoning{Effort: "high", ThinkingBudget: &budget}
		}
		built, err := lm.BuildRequest(req, false)
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(built.Body, &body); err != nil {
			t.Fatal(err)
		}
		if body["max_tokens"] != float64(c.wire) {
			t.Fatalf("%s: wire max_tokens %v, want %d", c.model, body["max_tokens"], c.wire)
		}
		plan, err := lm.Plan(req)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, a := range plan {
			if a.Field == "config.max_tokens" {
				found = true
				if a.Action != "defaulted" || a.Applied != c.recorded {
					t.Fatalf("%s: recorded %v %v, want %d", c.model, a.Action, a.Applied, c.recorded)
				}
			}
		}
		if !found {
			t.Fatalf("%s: no defaulted record", c.model)
		}
	}
}
