package auth

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
)

// refreshSkewMS: a token inside this window counts as expired (AUTH-3).
const refreshSkewMS = 5 * 60 * 1000

// LocalOAuthCredential is a locally stored OAuth credential read from a
// borrowed CLI file (AUTH-8). Token fields are unexported and every fmt
// rendering is redacted (AUTH-5); use the accessors.
type LocalOAuthCredential struct {
	accessToken  string
	refreshToken string
	expiresAtMS  int64 // 0 = no recorded expiry
	accountID    string
}

// AccessToken returns the secret access token. Callers own its hygiene.
func (c *LocalOAuthCredential) AccessToken() string { return c.accessToken }

// HasRefreshToken reports refresh capability without exposing the token.
func (c *LocalOAuthCredential) HasRefreshToken() bool { return c.refreshToken != "" }

// AccountID returns the non-secret account id, when known.
func (c *LocalOAuthCredential) AccountID() string { return c.accountID }

// Expired applies the AUTH-3 skew-free file semantics: a recorded expiry in
// the past. (Codex expiry already carries the skew from the JWT decode.)
func (c *LocalOAuthCredential) Expired() bool {
	return c.expiresAtMS != 0 && time.Now().UnixMilli() >= c.expiresAtMS
}

// String and GoString redact token material (%v, %s, %+v, %#v).
func (c *LocalOAuthCredential) String() string {
	return fmt.Sprintf("LocalOAuthCredential(expiresAtMS=%d, refresh=%t, redacted)",
		c.expiresAtMS, c.HasRefreshToken())
}

func (c *LocalOAuthCredential) GoString() string { return c.String() }

var errNotConfigured = errors.New("credential file missing or unusable")

// IsNotConfigured reports whether err means "no usable credential file".
func IsNotConfigured(err error) bool { return errors.Is(err, errNotConfigured) }

func readJSONObject(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", errNotConfigured, path)
	}
	var data map[string]any
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("%w: %s is not valid JSON", errNotConfigured, path)
	}
	return data, nil
}

func stringField(object map[string]any, key string) string {
	value, _ := object[key].(string)
	return value
}

// ReadClaudeCodeCredential reads the Claude Code CLI credential file
// (claudeAiOauth section). Read-only; never writes or refreshes.
func ReadClaudeCodeCredential(path string) (*LocalOAuthCredential, error) {
	data, err := readJSONObject(path)
	if err != nil {
		return nil, err
	}
	oauth, _ := data["claudeAiOauth"].(map[string]any)
	if oauth == nil {
		return nil, fmt.Errorf("%w: %s has no claudeAiOauth section", errNotConfigured, path)
	}
	access := stringField(oauth, "accessToken")
	if access == "" {
		return nil, fmt.Errorf("%w: %s has no access token", errNotConfigured, path)
	}
	credential := &LocalOAuthCredential{
		accessToken:  access,
		refreshToken: stringField(oauth, "refreshToken"),
	}
	if expires, ok := oauth["expiresAt"].(float64); ok {
		credential.expiresAtMS = int64(expires)
	}
	return credential, nil
}

// ReadCodexCLICredential reads the OpenAI Codex CLI auth file (tokens
// section); expiry comes from the access token's JWT exp claim minus the
// AUTH-3 skew. Read-only; never writes or refreshes.
func ReadCodexCLICredential(path string) (*LocalOAuthCredential, error) {
	data, err := readJSONObject(path)
	if err != nil {
		return nil, err
	}
	tokens, _ := data["tokens"].(map[string]any)
	if tokens == nil {
		return nil, fmt.Errorf("%w: %s has no tokens section", errNotConfigured, path)
	}
	access := stringField(tokens, "access_token")
	if access == "" {
		return nil, fmt.Errorf("%w: %s has no access token", errNotConfigured, path)
	}
	credential := &LocalOAuthCredential{
		accessToken:  access,
		refreshToken: stringField(tokens, "refresh_token"),
		accountID:    stringField(tokens, "account_id"),
	}
	if expires, ok := jwtExpiresAtMS(access); ok {
		credential.expiresAtMS = expires
	}
	if credential.accountID == "" {
		credential.accountID = chatGPTAccountID(access)
	}
	return credential, nil
}

func jwtPayload(token string) (map[string]any, bool) {
	parts := splitJWT(token)
	if parts == nil {
		return nil, false
	}
	decoded, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, false
	}
	var payload map[string]any
	if err := json.Unmarshal(decoded, &payload); err != nil {
		return nil, false
	}
	return payload, true
}

func splitJWT(token string) []string {
	var parts []string
	start := 0
	for i := 0; i <= len(token); i++ {
		if i == len(token) || token[i] == '.' {
			parts = append(parts, token[start:i])
			start = i + 1
		}
	}
	if len(parts) != 3 {
		return nil
	}
	return parts
}

func jwtExpiresAtMS(token string) (int64, bool) {
	payload, ok := jwtPayload(token)
	if !ok {
		return 0, false
	}
	exp, ok := payload["exp"].(float64)
	if !ok {
		return 0, false
	}
	return int64(exp)*1000 - refreshSkewMS, true
}

func chatGPTAccountID(token string) string {
	payload, ok := jwtPayload(token)
	if !ok {
		return ""
	}
	claim, _ := payload["https://api.openai.com/auth"].(map[string]any)
	if claim == nil {
		return ""
	}
	return stringField(claim, "chatgpt_account_id")
}
