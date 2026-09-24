package lm15

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/lm15-dev/lm15-go/internal/fslock"
)

// Local subscription credentials (spec/auth.md AUTH-3, AUTH-4, AUTH-8,
// AUTH-9): Claude Code (~/.claude/.credentials.json), the Codex CLI
// (~/.codex/auth.json) and xAI (lm15's own store, then the Pi agent store).
// Refresh is double-checked under the cross-process lock; writes are atomic
// and private; token material never appears in errors or String().

const (
	ClaudeCodeClientID  = "9d1c250a-e61b-44d5-88ed-5944d1962f5e"
	ClaudeCodeTokenURL  = "https://platform.claude.com/v1/oauth/token"
	OpenAICodexClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
	OpenAICodexTokenURL = "https://auth.openai.com/oauth/token"
	openaiCodexJWTClaim = "https://api.openai.com/auth"
	XaiClientID         = "b1a00492-073a-47ea-816f-4c329264a828"
	XaiDeviceCodeURL    = "https://auth.x.ai/oauth2/device/code"
	XaiTokenURL         = "https://auth.x.ai/oauth2/token"
	XaiOAuthScope       = "openid profile email offline_access grok-cli:access api:access"
	refreshSkewMs       = 5 * 60 * 1000
	xaiDefaultLifetimeS = 3600
	lockTimeout         = 60 * time.Second
)

// LocalOAuthCredential is a locally stored OAuth credential. Token fields
// never print.
type LocalOAuthCredential struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    *int64 // epoch milliseconds
	AccountID    string
}

// Expired reports whether the recorded expiry has passed.
func (c LocalOAuthCredential) Expired() bool {
	return c.ExpiresAt != nil && time.Now().UnixMilli() >= *c.ExpiresAt
}

// String redacts (AUTH-5).
func (c LocalOAuthCredential) String() string {
	return fmt.Sprintf("LocalOAuthCredential(<redacted>, expires_at=%v, account_id=%q)", c.ExpiresAt != nil, c.AccountID)
}

func homeDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return os.Getenv("HOME")
}

func expandHome(p string) string {
	if strings.HasPrefix(p, "~") {
		return filepath.Join(homeDir(), strings.TrimLeft(p[1:], "/\\"))
	}
	return p
}

// Well-known paths (AUTH-8).
func ClaudeCodeCredentialsPath() string {
	return filepath.Join(homeDir(), ".claude", ".credentials.json")
}
func CodexCLIAuthPath() string { return filepath.Join(homeDir(), ".codex", "auth.json") }
func PiAgentAuthPath() string  { return filepath.Join(homeDir(), ".pi", "agent", "auth.json") }

// DefaultCredentialsPath is $LM15_CREDENTIALS_PATH, else
// $XDG_CONFIG_HOME/lm15/credentials.json, else ~/.config/lm15/credentials.json.
func DefaultCredentialsPath() string {
	if v := os.Getenv("LM15_CREDENTIALS_PATH"); v != "" {
		return expandHome(v)
	}
	if v := os.Getenv("XDG_CONFIG_HOME"); v != "" {
		return filepath.Join(expandHome(v), "lm15", "credentials.json")
	}
	return filepath.Join(homeDir(), ".config", "lm15", "credentials.json")
}

func coercePath(path, fallback string) string {
	if path == "" {
		return fallback
	}
	return expandHome(path)
}

func notConfigured(provider, message, hint string) *Error {
	return NotConfiguredErrorf(provider, nil, hint, "%s", message)
}

func readJSONFile(path, provider, hint string) (JSONObject, error) {
	text, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, notConfigured(provider, "No credentials file at "+path+".", hint)
		}
		return nil, notConfigured(provider, "Could not read credentials file at "+path+": "+err.Error(), hint)
	}
	data, err := DecodeJSON(text)
	if err != nil {
		return nil, notConfigured(provider, "Credentials file at "+path+" is not valid JSON.", hint)
	}
	obj, ok := data.(map[string]any)
	if !ok {
		return nil, notConfigured(provider, "Credentials file at "+path+" has an unexpected shape.", hint)
	}
	return obj, nil
}

func readJSONFileOrNil(path string) JSONObject {
	text, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	data, err := DecodeJSON(text)
	if err != nil {
		return nil
	}
	obj, _ := data.(map[string]any)
	return obj
}

func writePrivateJSON(path string, data JSONObject) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(data); err != nil {
		return err
	}
	return fslock.WritePrivateAtomic(path, buf.Bytes())
}

// holdFileLock takes the AUTH-4 lock for path.
func holdFileLock(ctx context.Context, path string) (*fslock.Lock, error) {
	lockPath := fslock.LockPathFor(path, os.Getenv, homeDir())
	lock, err := fslock.Acquire(ctx, lockPath, lockTimeout)
	if err != nil {
		if err == fslock.ErrTimeout {
			e := newError(KindCredentialLockWait, fmt.Sprintf("Could not lock credential file %s within %.0fs (lock file: %s). Another process may be refreshing the same credential; retry, or remove a stale lock only if you are certain no other process holds it.", path, lockTimeout.Seconds(), lockPath))
			e.Path = path
			e.LockPath = lockPath
			return nil, e
		}
		return nil, err
	}
	return lock, nil
}

// ─── JWT helpers ─────────────────────────────────────────────────────

func base64URLJSON(segment string) JSONObject {
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(segment, "="))
	if err != nil {
		return nil
	}
	data, err := DecodeJSON(decoded)
	if err != nil {
		return nil
	}
	obj, _ := data.(map[string]any)
	return obj
}

// DecodeJWTPayload returns a JWT's claims (nil when not a JWT).
func DecodeJWTPayload(token string) JSONObject {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil
	}
	return base64URLJSON(parts[1])
}

func jwtExpiresAtMs(token string) *int64 {
	claims := DecodeJWTPayload(token)
	if claims == nil {
		return nil
	}
	exp, err := jsonFloat64(claims["exp"], "")
	if err != nil || claims["exp"] == nil {
		return nil
	}
	ms := int64(exp*1000) - refreshSkewMs
	return &ms
}

// ExtractChatGPTAccountID reads the chatgpt_account_id claim from a Codex token.
func ExtractChatGPTAccountID(token string) string {
	claims := DecodeJWTPayload(token)
	if claims == nil {
		return ""
	}
	return stringOnly(wireObj(claims[openaiCodexJWTClaim])["chatgpt_account_id"])
}

// ─── Token endpoints ─────────────────────────────────────────────────

var authHTTPClient = &http.Client{Timeout: 30 * time.Second}

func postJSON(ctx context.Context, url string, payload JSONObject) (JSONObject, int, error) {
	body := mustJSON(payload)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	return doAuthRequest(req)
}

func postForm(ctx context.Context, rawURL string, form url.Values) (JSONObject, int, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", rawURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	return doAuthRequest(req)
}

func doAuthRequest(req *http.Request) (JSONObject, int, error) {
	resp, err := authHTTPClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	data, _ := DecodeJSON(raw)
	obj, _ := data.(map[string]any)
	if obj == nil {
		obj = JSONObject{}
	}
	return obj, resp.StatusCode, nil
}

// ─── Claude Code ─────────────────────────────────────────────────────

// LoadClaudeCodeCredential reads the Claude Code credential, typed errors on failure.
func LoadClaudeCodeCredential(path string) (LocalOAuthCredential, error) {
	path = coercePath(path, ClaudeCodeCredentialsPath())
	data, err := readJSONFile(path, "claude-code", ClaudeCodeLoginHint)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	raw := wireObj(data["claudeAiOauth"])
	if raw == nil {
		return LocalOAuthCredential{}, notConfigured("claude-code", "Credentials file at "+path+" has no claudeAiOauth section.", ClaudeCodeLoginHint)
	}
	access := stringOnly(raw["accessToken"])
	if access == "" {
		return LocalOAuthCredential{}, notConfigured("claude-code", "Credentials file at "+path+" has no access token.", ClaudeCodeLoginHint)
	}
	cred := LocalOAuthCredential{AccessToken: access, RefreshToken: stringOnly(raw["refreshToken"])}
	if _, isBool := raw["expiresAt"].(bool); !isBool {
		if f, err := jsonFloat64(raw["expiresAt"], ""); err == nil && raw["expiresAt"] != nil {
			ms := int64(f)
			cred.ExpiresAt = &ms
		}
	}
	return cred, nil
}

// ReadClaudeCodeCredential is the optional-style loader.
func ReadClaudeCodeCredential(path string) *LocalOAuthCredential {
	cred, err := LoadClaudeCodeCredential(path)
	if err != nil {
		return nil
	}
	return &cred
}

// RefreshClaudeCodeCredential exchanges a refresh token.
func RefreshClaudeCodeCredential(ctx context.Context, refreshToken string) (LocalOAuthCredential, error) {
	payload, status, err := postJSON(ctx, ClaudeCodeTokenURL, JSONObject{"grant_type": "refresh_token", "client_id": ClaudeCodeClientID, "refresh_token": refreshToken})
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	if status >= 400 {
		return LocalOAuthCredential{}, fmt.Errorf("Claude Code token refresh failed: HTTP %d", status)
	}
	access := stringOnly(payload["access_token"])
	refresh := stringOnly(payload["refresh_token"])
	expiresIn, err := jsonFloat64(payload["expires_in"], "")
	if access == "" || refresh == "" || err != nil {
		return LocalOAuthCredential{}, fmt.Errorf("Claude Code token refresh response is missing required fields")
	}
	ms := time.Now().UnixMilli() + int64(expiresIn*1000) - refreshSkewMs
	return LocalOAuthCredential{AccessToken: access, RefreshToken: refresh, ExpiresAt: &ms}, nil
}

func writeClaudeCodeUnlocked(cred LocalOAuthCredential, path string) error {
	data := readJSONFileOrNil(path)
	if data == nil {
		data = JSONObject{}
	}
	current := wireObj(data["claudeAiOauth"])
	if current == nil {
		current = JSONObject{}
	}
	current["accessToken"] = cred.AccessToken
	if cred.RefreshToken != "" {
		current["refreshToken"] = cred.RefreshToken
	}
	if cred.ExpiresAt != nil {
		current["expiresAt"] = *cred.ExpiresAt
	}
	data["claudeAiOauth"] = current
	return writePrivateJSON(path, data)
}

// WriteClaudeCodeCredential writes the credential under the file lock.
func WriteClaudeCodeCredential(ctx context.Context, cred LocalOAuthCredential, path string) error {
	path = coercePath(path, ClaudeCodeCredentialsPath())
	lock, err := holdFileLock(ctx, path)
	if err != nil {
		return err
	}
	defer lock.Release()
	return writeClaudeCodeUnlocked(cred, path)
}

// GetClaudeCodeAccessToken returns a usable token, refreshing under the lock
// when expired (double-checked).
func GetClaudeCodeAccessToken(ctx context.Context, path string, refresh bool) (string, error) {
	cred, err := LoadClaudeCodeCredential(path)
	if err != nil {
		return "", err
	}
	if !cred.Expired() {
		return cred.AccessToken, nil
	}
	if !refresh || cred.RefreshToken == "" {
		return "", AuthErrorf("claude-code", nil, ClaudeCodeLoginHint, "Claude Code OAuth token is expired and no refresh token is available.")
	}
	path = coercePath(path, ClaudeCodeCredentialsPath())
	lock, err := holdFileLock(ctx, path)
	if err != nil {
		return "", err
	}
	defer lock.Release()
	if cred, err = LoadClaudeCodeCredential(path); err != nil {
		return "", err
	}
	if !cred.Expired() {
		return cred.AccessToken, nil
	}
	if cred.RefreshToken == "" {
		return "", AuthErrorf("claude-code", nil, ClaudeCodeLoginHint, "Claude Code OAuth token is expired and no refresh token is available.")
	}
	refreshed, err := RefreshClaudeCodeCredential(ctx, cred.RefreshToken)
	if err != nil {
		return "", AuthErrorf("claude-code", nil, ClaudeCodeLoginHint, "Claude Code OAuth token is expired and the refresh attempt failed.").WithCause(err)
	}
	if err := writeClaudeCodeUnlocked(refreshed, path); err != nil {
		return "", err
	}
	return refreshed.AccessToken, nil
}

// ─── OpenAI Codex CLI ────────────────────────────────────────────────

// LoadCodexCLICredential reads the Codex CLI credential.
func LoadCodexCLICredential(path string) (LocalOAuthCredential, error) {
	path = coercePath(path, CodexCLIAuthPath())
	data, err := readJSONFile(path, "openai-codex", OpenAICodexLoginHint)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	tokens := wireObj(data["tokens"])
	if tokens == nil {
		return LocalOAuthCredential{}, notConfigured("openai-codex", "Credentials file at "+path+" has no tokens section.", OpenAICodexLoginHint)
	}
	access := stringOnly(tokens["access_token"])
	if access == "" {
		return LocalOAuthCredential{}, notConfigured("openai-codex", "Credentials file at "+path+" has no access token.", OpenAICodexLoginHint)
	}
	account := stringOnly(tokens["account_id"])
	if account == "" {
		account = ExtractChatGPTAccountID(access)
	}
	return LocalOAuthCredential{AccessToken: access, RefreshToken: stringOnly(tokens["refresh_token"]), ExpiresAt: jwtExpiresAtMs(access), AccountID: account}, nil
}

// ReadCodexCLICredential is the optional-style loader.
func ReadCodexCLICredential(path string) *LocalOAuthCredential {
	cred, err := LoadCodexCLICredential(path)
	if err != nil {
		return nil
	}
	return &cred
}

// RefreshCodexCLICredential exchanges a refresh token.
func RefreshCodexCLICredential(ctx context.Context, refreshToken string) (LocalOAuthCredential, error) {
	payload, status, err := postForm(ctx, OpenAICodexTokenURL, url.Values{"grant_type": {"refresh_token"}, "refresh_token": {refreshToken}, "client_id": {OpenAICodexClientID}})
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	if status >= 400 {
		return LocalOAuthCredential{}, fmt.Errorf("Codex token refresh failed: HTTP %d", status)
	}
	access := stringOnly(payload["access_token"])
	refresh := stringOnly(payload["refresh_token"])
	if refresh == "" {
		refresh = refreshToken
	}
	if access == "" {
		return LocalOAuthCredential{}, fmt.Errorf("Codex token refresh response is missing required fields")
	}
	return LocalOAuthCredential{AccessToken: access, RefreshToken: refresh, ExpiresAt: jwtExpiresAtMs(access), AccountID: ExtractChatGPTAccountID(access)}, nil
}

func writeCodexUnlocked(cred LocalOAuthCredential, path, idToken string) error {
	data := readJSONFileOrNil(path)
	if data == nil {
		data = JSONObject{}
	}
	tokens := wireObj(data["tokens"])
	if tokens == nil {
		tokens = JSONObject{}
	}
	tokens["access_token"] = cred.AccessToken
	if cred.RefreshToken != "" {
		tokens["refresh_token"] = cred.RefreshToken
	}
	if cred.AccountID != "" {
		tokens["account_id"] = cred.AccountID
	}
	if idToken != "" {
		tokens["id_token"] = idToken
	}
	data["tokens"] = tokens
	if _, has := data["auth_mode"]; !has {
		data["auth_mode"] = "chatgpt"
	}
	data["last_refresh"] = time.Now().UTC().Format("2006-01-02T15:04:05.000000Z")
	return writePrivateJSON(path, data)
}

// WriteCodexCLICredential writes the credential under the file lock.
func WriteCodexCLICredential(ctx context.Context, cred LocalOAuthCredential, path, idToken string) error {
	path = coercePath(path, CodexCLIAuthPath())
	lock, err := holdFileLock(ctx, path)
	if err != nil {
		return err
	}
	defer lock.Release()
	return writeCodexUnlocked(cred, path, idToken)
}

// GetCodexCLIAccessToken returns a usable credential, refreshing under the lock.
func GetCodexCLIAccessToken(ctx context.Context, path string, refresh bool) (LocalOAuthCredential, error) {
	cred, err := LoadCodexCLICredential(path)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	if !cred.Expired() {
		return cred, nil
	}
	if !refresh || cred.RefreshToken == "" {
		return LocalOAuthCredential{}, AuthErrorf("openai-codex", nil, OpenAICodexLoginHint, "Codex CLI OAuth token is expired and no refresh token is available.")
	}
	path = coercePath(path, CodexCLIAuthPath())
	lock, err := holdFileLock(ctx, path)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	defer lock.Release()
	if cred, err = LoadCodexCLICredential(path); err != nil {
		return LocalOAuthCredential{}, err
	}
	if !cred.Expired() {
		return cred, nil
	}
	if cred.RefreshToken == "" {
		return LocalOAuthCredential{}, AuthErrorf("openai-codex", nil, OpenAICodexLoginHint, "Codex CLI OAuth token is expired and no refresh token is available.")
	}
	refreshed, err := RefreshCodexCLICredential(ctx, cred.RefreshToken)
	if err != nil {
		return LocalOAuthCredential{}, AuthErrorf("openai-codex", nil, OpenAICodexLoginHint, "Codex CLI OAuth token is expired and the refresh attempt failed.").WithCause(err)
	}
	idToken := stringOnly(wireObj(readJSONFileOrNil(path)["tokens"])["id_token"])
	if err := writeCodexUnlocked(refreshed, path, idToken); err != nil {
		return LocalOAuthCredential{}, err
	}
	return refreshed, nil
}

// ─── xAI (lm15 store or Pi agent store) ──────────────────────────────

func xaiStorePaths() []string { return []string{DefaultCredentialsPath(), PiAgentAuthPath()} }

func xaiEntryToCredential(entry any) *LocalOAuthCredential {
	obj := wireObj(entry)
	if obj == nil {
		return nil
	}
	access := stringOnly(obj["access"])
	if access == "" {
		return nil
	}
	cred := &LocalOAuthCredential{AccessToken: access, RefreshToken: stringOnly(obj["refresh"])}
	if _, isBool := obj["expires"].(bool); !isBool && obj["expires"] != nil {
		if i, err := jsonInt(obj["expires"], ""); err == nil {
			ms := int64(i)
			cred.ExpiresAt = &ms
		}
	}
	return cred
}

func xaiCredentialToEntry(cred LocalOAuthCredential, current JSONObject) JSONObject {
	entry := copyObject(current)
	if entry == nil {
		entry = JSONObject{}
	}
	entry["type"] = "oauth"
	entry["access"] = cred.AccessToken
	if cred.RefreshToken != "" {
		entry["refresh"] = cred.RefreshToken
	}
	if cred.ExpiresAt != nil {
		entry["expires"] = *cred.ExpiresAt
	}
	return entry
}

func loadXaiWithSource(path string) (LocalOAuthCredential, string, error) {
	paths := xaiStorePaths()
	if path != "" {
		paths = []string{expandHome(path)}
	}
	for _, p := range paths {
		if data := readJSONFileOrNil(p); data != nil {
			if cred := xaiEntryToCredential(data["xai"]); cred != nil {
				return *cred, p, nil
			}
		}
	}
	return LocalOAuthCredential{}, "", notConfigured("xai", "No xAI OAuth credential found (checked: "+strings.Join(paths, ", ")+").", XaiLoginHint)
}

// LoadXaiCredential reads the stored xAI credential.
func LoadXaiCredential(path string) (LocalOAuthCredential, error) {
	cred, _, err := loadXaiWithSource(path)
	return cred, err
}

// ReadXaiCredential is the optional-style loader.
func ReadXaiCredential(path string) *LocalOAuthCredential {
	cred, err := LoadXaiCredential(path)
	if err != nil {
		return nil
	}
	return &cred
}

// UsableXaiCredential reports a stored login that is fresh or refreshable.
// Files only; never the network (the oauth-unless-explicit probe).
func UsableXaiCredential(path string) bool {
	return XaiStoredState(path) == "usable"
}

// XaiStoredState is the stored xAI subscription's state, offline
// (spec/auth.md AUTH-1, ratified R2/R3 2026-09-22): "usable" (fresh, or
// expired with a refresh token); "unusable" (expired, no refresh token);
// "logged_out" (lm15's non-secret sign-out marker); "absent" (nothing
// stored). "unusable" and "logged_out" BLOCK the environment key: a failed
// subscription is never silently replaced by a metered key.
func XaiStoredState(path string) string { return xaiStoredStateAt(path, time.Now()) }

func xaiStoredStateAt(path string, now time.Time) string {
	paths := xaiStorePaths()
	if path != "" {
		paths = []string{expandHome(path)}
	}
	for _, p := range paths {
		data := readJSONFileOrNil(p)
		if data == nil {
			continue
		}
		if cred := xaiEntryToCredential(data["xai"]); cred != nil {
			expired := cred.ExpiresAt != nil && now.UnixMilli() >= *cred.ExpiresAt
			if !expired || cred.RefreshToken != "" {
				return "usable"
			}
			return "unusable"
		}
		if own := wireObj(data["_lm15"]); own != nil {
			if slots := wireObj(own["slots"]); slots != nil {
				if slot := wireObj(slots["xai"]); slot != nil {
					marker := slot["logged_out"]
					if on, isBool := marker.(bool); marker != nil && (!isBool || on) {
						return "logged_out"
					}
				}
			}
		}
	}
	return "absent"
}

func xaiCredentialFromToken(payload JSONObject, previousRefresh string) (LocalOAuthCredential, error) {
	access := stringOnly(payload["access_token"])
	if access == "" {
		return LocalOAuthCredential{}, fmt.Errorf("xAI token response is missing access_token")
	}
	refresh := stringOnly(payload["refresh_token"])
	if refresh == "" {
		refresh = previousRefresh
	}
	lifetime := float64(xaiDefaultLifetimeS)
	if _, isBool := payload["expires_in"].(bool); !isBool {
		if f, err := jsonFloat64(payload["expires_in"], ""); err == nil && f > 0 {
			lifetime = f
		}
	}
	ms := time.Now().UnixMilli() + int64(lifetime*1000) - refreshSkewMs
	return LocalOAuthCredential{AccessToken: access, RefreshToken: refresh, ExpiresAt: &ms}, nil
}

// RefreshXaiCredential exchanges a refresh token.
func RefreshXaiCredential(ctx context.Context, refreshToken string) (LocalOAuthCredential, error) {
	payload, status, err := postForm(ctx, XaiTokenURL, url.Values{"grant_type": {"refresh_token"}, "client_id": {XaiClientID}, "refresh_token": {refreshToken}})
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	if status >= 400 {
		return LocalOAuthCredential{}, fmt.Errorf("xAI token refresh failed: HTTP %d", status)
	}
	return xaiCredentialFromToken(payload, refreshToken)
}

// WriteXaiCredential writes the credential into a store (default: lm15's own).
func WriteXaiCredential(ctx context.Context, cred LocalOAuthCredential, path string) error {
	store := NewCredentialFileStore(coercePath(path, DefaultCredentialsPath()))
	_, err := store.Mutate(ctx, "xai", func(current JSONObject) (JSONObject, error) {
		return xaiCredentialToEntry(cred, current), nil
	})
	return err
}

// GetXaiAccessToken returns a usable xAI token, refreshing and persisting to
// the file the credential came from (xAI rotates refresh tokens).
func GetXaiAccessToken(ctx context.Context, path string, refresh bool) (string, error) {
	cred, source, err := loadXaiWithSource(path)
	if err != nil {
		return "", err
	}
	if !cred.Expired() {
		return cred.AccessToken, nil
	}
	if !refresh || cred.RefreshToken == "" {
		return "", AuthErrorf("xai", nil, XaiLoginHint, "xAI OAuth token is expired and no refresh token is available.")
	}
	result := ""
	_, err = NewCredentialFileStore(source).Mutate(ctx, "xai", func(current JSONObject) (JSONObject, error) {
		fresh := xaiEntryToCredential(current)
		if fresh != nil && !fresh.Expired() {
			result = fresh.AccessToken
			return nil, nil
		}
		refreshToken := cred.RefreshToken
		if fresh != nil && fresh.RefreshToken != "" {
			refreshToken = fresh.RefreshToken
		}
		if refreshToken == "" {
			return nil, AuthErrorf("xai", nil, XaiLoginHint, "xAI OAuth token is expired and no refresh token is available.")
		}
		refreshed, err := RefreshXaiCredential(ctx, refreshToken)
		if err != nil {
			if IsKind(err, KindAuth) {
				return nil, err
			}
			return nil, AuthErrorf("xai", nil, XaiLoginHint, "xAI OAuth token is expired and the refresh attempt failed.").WithCause(err)
		}
		result = refreshed.AccessToken
		return xaiCredentialToEntry(refreshed, current), nil
	})
	if err != nil {
		return "", err
	}
	return result, nil
}

// ─── lm15-owned credential store ─────────────────────────────────────

// CredentialFileStore is a locked, atomic, private credential file keyed by
// provider id: {"<provider>": {...}}.
type CredentialFileStore struct{ Path string }

// NewCredentialFileStore opens a store ("" = the default path).
func NewCredentialFileStore(path string) *CredentialFileStore {
	return &CredentialFileStore{Path: coercePath(path, DefaultCredentialsPath())}
}

// String never shows contents.
func (s *CredentialFileStore) String() string {
	return fmt.Sprintf("CredentialFileStore(path=%q)", s.Path)
}

func (s *CredentialFileStore) readAll() (JSONObject, error) {
	text, err := os.ReadFile(s.Path)
	if err != nil {
		if os.IsNotExist(err) {
			return JSONObject{}, nil
		}
		return nil, err
	}
	data, err := DecodeJSON(text)
	if err != nil {
		return nil, valueErrorf("Credential store at %s is not valid JSON.", s.Path)
	}
	obj, ok := data.(map[string]any)
	if !ok {
		return nil, valueErrorf("Credential store at %s must be a JSON object.", s.Path)
	}
	return obj, nil
}

// Read returns the stored credential for provider, or nil.
func (s *CredentialFileStore) Read(provider string) (JSONObject, error) {
	all, err := s.readAll()
	if err != nil {
		return nil, err
	}
	return copyObject(wireObj(all[provider])), nil
}

// List returns the provider ids with stored credentials.
func (s *CredentialFileStore) List() ([]string, error) {
	all, err := s.readAll()
	if err != nil {
		return nil, err
	}
	return sortedKeys(all), nil
}

// Write replaces one provider's credential.
func (s *CredentialFileStore) Write(ctx context.Context, provider string, credential JSONObject) error {
	_, err := s.Mutate(ctx, provider, func(JSONObject) (JSONObject, error) { return credential, nil })
	return err
}

// Delete removes one provider's credential.
func (s *CredentialFileStore) Delete(ctx context.Context, provider string) error {
	lock, err := holdFileLock(ctx, s.Path)
	if err != nil {
		return err
	}
	defer lock.Release()
	all, err := s.readAll()
	if err != nil {
		return err
	}
	if _, has := all[provider]; has {
		delete(all, provider)
		return writePrivateJSON(s.Path, all)
	}
	return nil
}

// Mutate is the serialized read-modify-write; fn returns nil to leave the
// entry unchanged. Returns the post-write credential.
func (s *CredentialFileStore) Mutate(ctx context.Context, provider string, fn func(current JSONObject) (JSONObject, error)) (JSONObject, error) {
	lock, err := holdFileLock(ctx, s.Path)
	if err != nil {
		return nil, err
	}
	defer lock.Release()
	all, err := s.readAll()
	if err != nil {
		return nil, err
	}
	current := copyObject(wireObj(all[provider]))
	if wireObj(all[provider]) == nil {
		current = nil
	}
	replacement, err := fn(current)
	if err != nil {
		return nil, err
	}
	if replacement == nil {
		return current, nil
	}
	all[provider] = replacement
	if err := writePrivateJSON(s.Path, all); err != nil {
		return nil, err
	}
	return copyObject(replacement), nil
}

// ─── PKCE (RFC 7636, S256) ───────────────────────────────────────────

// PKCEPair is a verifier (secret) and its S256 challenge.
type PKCEPair struct {
	Verifier  string
	Challenge string
	Method    string
}

// String hides the verifier.
func (p PKCEPair) String() string { return "PKCEPair(<redacted>, challenge=" + p.Challenge + ")" }

// PKCEChallenge is the S256 challenge for a verifier.
func PKCEChallenge(verifier string) string {
	sum := sha256.Sum256([]byte(verifier))
	return base64.RawURLEncoding.EncodeToString(sum[:])
}

// GeneratePKCE creates a fresh pair: 64 random bytes → 86-char verifier.
func GeneratePKCE() (PKCEPair, error) {
	buf := make([]byte, 64)
	if _, err := rand.Read(buf); err != nil {
		return PKCEPair{}, err
	}
	verifier := base64.RawURLEncoding.EncodeToString(buf)
	return PKCEPair{Verifier: verifier, Challenge: PKCEChallenge(verifier), Method: "S256"}, nil
}

// ─── Device-code polling (RFC 8628 §3.5) ─────────────────────────────

// DevicePollResult is one poll outcome.
type DevicePollResult struct {
	Pending  bool
	SlowDown bool
	Interval *float64 // seconds, when the server names one
	Complete bool
	Value    any
	Failed   bool
	Message  string
}

// PollDeviceCode runs the polling loop; sleep and clock are injectable.
func PollDeviceCode(ctx context.Context, poll func() (DevicePollResult, error), intervalS, expiresInS float64, provider string, waitBeforeFirst bool, sleep func(time.Duration), clock func() time.Time) (any, error) {
	if sleep == nil {
		sleep = time.Sleep
	}
	if clock == nil {
		clock = time.Now
	}
	if intervalS < 0 {
		intervalS = 0
	}
	deadline := clock().Add(time.Duration(expiresInS * float64(time.Second)))
	first := true
	expiredErr := func() *Error {
		e := newError(KindDeviceCodeExpired, "Device authorization expired before it was approved. Start the login again.")
		e.Provider = provider
		return e
	}
	for {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if !first || waitBeforeFirst {
			if clock().Add(time.Duration(intervalS * float64(time.Second))).After(deadline) {
				return nil, expiredErr()
			}
			sleep(time.Duration(intervalS * float64(time.Second)))
		}
		first = false
		if clock().After(deadline) {
			return nil, expiredErr()
		}
		result, err := poll()
		if err != nil {
			return nil, err
		}
		switch {
		case result.Complete:
			return result.Value, nil
		case result.Failed:
			e := newError(KindAuth, result.Message)
			e.Provider = provider
			return nil, e
		case result.SlowDown:
			if result.Interval != nil && *result.Interval > 0 {
				intervalS = *result.Interval
			} else {
				intervalS += 5
			}
		}
	}
}

// XaiDeviceAuthorization is one pending device authorization.
type XaiDeviceAuthorization struct {
	UserCode                string
	VerificationURI         string
	VerificationURIComplete string
	IntervalS               float64
	ExpiresInS              float64
	deviceCode              string
}

// String hides the device code.
func (d XaiDeviceAuthorization) String() string {
	return fmt.Sprintf("XaiDeviceAuthorization(user_code=%q, verification_uri=%q)", d.UserCode, d.VerificationURI)
}

func httpsOrRaise(raw any) (string, error) {
	s, ok := raw.(string)
	if ok {
		if u, err := url.Parse(s); err == nil && u.Scheme == "https" && u.Host != "" {
			return s, nil
		}
	}
	e := newError(KindAuth, "xAI device authorization returned an untrusted verification URI.")
	e.Provider = "xai"
	return "", e
}

// StartXaiDeviceLogin requests a device authorization.
func StartXaiDeviceLogin(ctx context.Context) (XaiDeviceAuthorization, error) {
	payload, status, err := postForm(ctx, XaiDeviceCodeURL, url.Values{"client_id": {XaiClientID}, "scope": {XaiOAuthScope}, "referrer": {"lm15"}})
	if err != nil {
		return XaiDeviceAuthorization{}, err
	}
	if status >= 400 {
		detail := firstStr(payload["error_description"], payload["error"])
		if detail == "" {
			detail = "request failed"
		}
		e := newError(KindAuth, "xAI device authorization failed: "+detail)
		e.Provider = "xai"
		return XaiDeviceAuthorization{}, e
	}
	deviceCode := stringOnly(payload["device_code"])
	userCode := stringOnly(payload["user_code"])
	if deviceCode == "" || userCode == "" {
		e := newError(KindAuth, "xAI device authorization response is missing required fields.")
		e.Provider = "xai"
		return XaiDeviceAuthorization{}, e
	}
	expiresIn, err := jsonFloat64(payload["expires_in"], "")
	if err != nil || expiresIn <= 0 {
		e := newError(KindAuth, "xAI device authorization response is missing expires_in.")
		e.Provider = "xai"
		return XaiDeviceAuthorization{}, e
	}
	interval := 5.0
	if f, err := jsonFloat64(payload["interval"], ""); err == nil && f > 0 {
		interval = f
	}
	uri, err := httpsOrRaise(payload["verification_uri"])
	if err != nil {
		return XaiDeviceAuthorization{}, err
	}
	complete := ""
	if s := stringOnly(payload["verification_uri_complete"]); s != "" {
		if complete, err = httpsOrRaise(s); err != nil {
			return XaiDeviceAuthorization{}, err
		}
	}
	return XaiDeviceAuthorization{UserCode: userCode, VerificationURI: uri, VerificationURIComplete: complete, IntervalS: interval, ExpiresInS: expiresIn, deviceCode: deviceCode}, nil
}

// PollXaiDeviceLogin polls until the user approves.
func PollXaiDeviceLogin(ctx context.Context, device XaiDeviceAuthorization, sleep func(time.Duration)) (LocalOAuthCredential, error) {
	poll := func() (DevicePollResult, error) {
		payload, status, err := postForm(ctx, XaiTokenURL, url.Values{"grant_type": {"urn:ietf:params:oauth:grant-type:device_code"}, "client_id": {XaiClientID}, "device_code": {device.deviceCode}})
		if err != nil {
			return DevicePollResult{}, err
		}
		if status < 400 {
			cred, err := xaiCredentialFromToken(payload, "")
			if err != nil {
				return DevicePollResult{}, err
			}
			return DevicePollResult{Complete: true, Value: cred}, nil
		}
		switch stringOnly(payload["error"]) {
		case "authorization_pending":
			return DevicePollResult{Pending: true}, nil
		case "slow_down":
			r := DevicePollResult{SlowDown: true}
			if f, err := jsonFloat64(payload["interval"], ""); err == nil && f > 0 {
				r.Interval = &f
			}
			return r, nil
		case "access_denied", "authorization_denied":
			return DevicePollResult{Failed: true, Message: "xAI device authorization was denied."}, nil
		case "expired_token":
			return DevicePollResult{Failed: true, Message: "xAI device code expired before it was approved."}, nil
		}
		detail := firstStr(payload["error_description"], payload["error"])
		if detail == "" {
			detail = "request failed"
		}
		return DevicePollResult{Failed: true, Message: "xAI device token polling failed: " + detail}, nil
	}
	value, err := PollDeviceCode(ctx, poll, device.IntervalS, device.ExpiresInS, "xai", true, sleep, nil)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	return value.(LocalOAuthCredential), nil
}

// LoginXai runs the device-code login, persists and returns the credential.
func LoginXai(ctx context.Context, path string, echo func(string)) (LocalOAuthCredential, error) {
	if echo == nil {
		echo = func(s string) { fmt.Println(s) }
	}
	device, err := StartXaiDeviceLogin(ctx)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	target := device.VerificationURIComplete
	if target == "" {
		target = device.VerificationURI
	}
	echo("Open " + target + " and enter code: " + device.UserCode)
	cred, err := PollXaiDeviceLogin(ctx, device, nil)
	if err != nil {
		return LocalOAuthCredential{}, err
	}
	if err := WriteXaiCredential(ctx, cred, path); err != nil {
		return LocalOAuthCredential{}, err
	}
	return cred, nil
}

// ─── Uniform login door (AUTH-9) ─────────────────────────────────────

var keyConsoleURLs = map[string]string{
	"openai": "https://platform.openai.com/api-keys", "openai-chat": "https://platform.openai.com/api-keys",
	"anthropic": "https://console.anthropic.com", "gemini": "https://aistudio.google.com/apikey",
	"groq": "https://console.groq.com/keys", "openrouter": "https://openrouter.ai/keys", "xai": "https://console.x.ai",
}

var cliLoginHints = map[string]string{"claude-code": ClaudeCodeLoginHint, "openai-codex": OpenAICodexLoginHint}

// Login runs the flow lm15 owns for provider (today: xai); every other
// provider fails typed, naming the real path.
func Login(ctx context.Context, provider, credentialsPath string, echo func(string)) (LocalOAuthCredential, error) {
	canonical := CanonicalProvider(provider)
	switch {
	case canonical == "xai":
		return LoginXai(ctx, credentialsPath, echo)
	case cliLoginHints[canonical] != "":
		return LocalOAuthCredential{}, UnsupportedFeatureErrorf(canonical, "lm15 does not own the %q login flow — the provider CLI does. %s", canonical, cliLoginHints[canonical])
	case canonical == "ollama" || canonical == "vllm" || canonical == "sglang":
		return LocalOAuthCredential{}, UnsupportedFeatureErrorf(canonical, "%q is a keyless local server — there is nothing to log into. The router sends the placeholder key the server expects.", canonical)
	case keyConsoleURLs[canonical] != "":
		return LocalOAuthCredential{}, UnsupportedFeatureErrorf(canonical, "%q offers no OAuth login flow — only manually created API keys. Create one at %s and set it in the environment or RouterConfig(api_keys={%q: \"...\"}).", canonical, keyConsoleURLs[canonical], canonical)
	}
	return LocalOAuthCredential{}, UnsupportedFeatureErrorf(canonical, "lm15 has no login flow for %q. Supply an API key via the environment or RouterConfig(api_keys=...).", provider)
}
