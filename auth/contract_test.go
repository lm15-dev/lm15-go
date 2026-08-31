package auth

// Runs the lm15-contract auth-resolution fixtures (auth/resolution.json,
// spec/auth.md AUTH-1/AUTH-7, ratified 2026-08-31). Divergence between this
// port and the fixtures is a port bug, never a reason to edit the fixture
// (AUTHORITY.md).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fixtureFile struct {
	Sentinel string        `json:"sentinel"`
	Cases    []fixtureCase `json:"cases"`
}

type fixtureCase struct {
	ID               string            `json:"id"`
	Provider         string            `json:"provider"`
	Env              map[string]string `json:"env"`
	APIKeysProviders []string          `json:"api_keys_providers"`
	BorrowedFile     *struct {
		State string `json:"state"`
	} `json:"borrowed_file"`
	Expect struct {
		Configured bool `json:"configured"`
		Steps      []struct {
			Kind  string `json:"kind"`
			State string `json:"state"`
		} `json:"steps"`
	} `json:"expect"`
}

func loadFixture(t *testing.T) fixtureFile {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "conformance", "auth_resolution.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture fixtureFile
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return fixture
}

func materializeBorrowedFile(t *testing.T, state, sentinel string) string {
	t.Helper()
	dir := t.TempDir()
	if state == "missing" {
		return filepath.Join(dir, "does-not-exist.json")
	}
	nowMS := time.Now().UnixMilli()
	oauth := map[string]any{"accessToken": sentinel}
	switch state {
	case "fresh":
		oauth["expiresAt"] = nowMS + 3_600_000
		oauth["refreshToken"] = sentinel
	case "expired-with-refresh":
		oauth["expiresAt"] = 1
		oauth["refreshToken"] = sentinel
	case "expired-no-refresh":
		oauth["expiresAt"] = 1
	default:
		t.Fatalf("unknown borrowed_file state %q", state)
	}
	raw, err := json.Marshal(map[string]any{"claudeAiOauth": oauth})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "credentials.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestAuthResolutionContract(t *testing.T) {
	fixture := loadFixture(t)
	if len(fixture.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}
	for _, testCase := range fixture.Cases {
		t.Run(testCase.ID, func(t *testing.T) {
			opts := ExplainOptions{Env: map[string]string{}}
			for key, value := range testCase.Env {
				opts.Env[key] = value
			}
			if len(testCase.APIKeysProviders) > 0 {
				opts.Credentials = map[string]CredentialProvider{}
				for _, provider := range testCase.APIKeysProviders {
					opts.Credentials[provider] = Static(fixture.Sentinel)
				}
			}
			if testCase.BorrowedFile != nil {
				if testCase.Provider != "claude-code" {
					t.Fatal("fixture uses claude-code for oauth cases")
				}
				opts.ClaudeCredentialsPath = materializeBorrowedFile(
					t, testCase.BorrowedFile.State, fixture.Sentinel)
			}

			report, err := ExplainAuth(testCase.Provider, opts)
			if err != nil {
				t.Fatalf("ExplainAuth: %v", err)
			}
			if report.Configured != testCase.Expect.Configured {
				t.Errorf("configured = %v, want %v", report.Configured, testCase.Expect.Configured)
			}
			if len(report.Steps) != len(testCase.Expect.Steps) {
				t.Fatalf("got %d steps, want %d: %v", len(report.Steps), len(testCase.Expect.Steps), report.Steps)
			}
			for i, expected := range testCase.Expect.Steps {
				actual := report.Steps[i]
				if actual.Kind != expected.Kind || string(actual.State) != expected.State {
					t.Errorf("step %d = {%s %s}, want {%s %s}",
						i, actual.Kind, actual.State, expected.Kind, expected.State)
				}
			}

			// AUTH-5: no rendering may carry the planted sentinel.
			for _, rendering := range []string{
				report.Describe(),
				fmt.Sprintf("%v", report),
				fmt.Sprintf("%+v", report),
			} {
				if strings.Contains(rendering, fixture.Sentinel) {
					t.Errorf("sentinel leaked into rendering: %s", rendering)
				}
			}
		})
	}
}
