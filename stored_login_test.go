package lm15

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// R3 (ratified 2026-09-22): an expired-without-refresh or signed-out xAI login
// BLOCKS $XAI_API_KEY; an explicit key still works; a usable login wins; with
// nothing stored the env key applies.
func TestStoredXaiLoginBlocksTheEnvKey(t *testing.T) {
	const secret = "SECRET-SENTINEL-DO-NOT-PRINT"
	dir := t.TempDir()
	file := filepath.Join(dir, "credentials.json")
	t.Setenv("LM15_CREDENTIALS_PATH", file)
	t.Setenv("HOME", dir) // no Pi agent store either
	write := func(body string) {
		if err := os.WriteFile(file, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	lm := func(config RouterConfig) error {
		router, err := NewRouterWithConfig(config)
		if err != nil {
			t.Fatal(err)
		}
		_, err = router.LM("xai:grok-4")
		return err
	}
	env := map[string]string{"XAI_API_KEY": secret}

	write(`{"xai":{"type":"oauth","access":"a","expires":1}}`)
	err := lm(RouterConfig{Env: env})
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindMissingCredential {
		t.Fatalf("expired login: want a missing-credential error, got %v", err)
	}
	for _, want := range []string{"expired and cannot be renewed", "$XAI_API_KEY is set but is used only when passed explicitly"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("message lacks %q: %v", want, err)
		}
	}
	if strings.Contains(err.Error(), secret) {
		t.Error("the key's value printed")
	}
	if err := lm(RouterConfig{Env: env, APIKeys: map[string]CredentialLike{"xai": "explicit"}}); err != nil {
		t.Fatalf("an explicit key is deliberate authority: %v", err)
	}

	write(`{"_lm15":{"slots":{"xai":{"logged_out":true}}}}`)
	if err := lm(RouterConfig{Env: env}); err == nil || !strings.Contains(err.Error(), "was signed out") {
		t.Fatalf("signed out: want the block, got %v", err)
	}

	// Expired with a refresh token is usable (the adapter renews it; checked
	// offline here so the test never reaches the token endpoint).
	write(`{"xai":{"type":"oauth","access":"a","refresh":"r","expires":1}}`)
	if state := XaiStoredState(""); state != "usable" {
		t.Fatalf("expired with a refresh token: want usable, got %s", state)
	}
	write(`{"xai":{"type":"oauth","access":"a","expires":99999999999999}}`)
	if err := lm(RouterConfig{Env: env}); err != nil {
		t.Fatalf("a fresh login wins: %v", err)
	}

	write(`{}`)
	if err := lm(RouterConfig{Env: env}); err != nil {
		t.Fatalf("nothing stored: the env key applies: %v", err)
	}
}
