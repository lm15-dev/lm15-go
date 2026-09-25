package lm15

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// A credential is a closed sum, not a string (spec/auth.md AUTH-2):
//
//	APIKey          {"kind": "api_key", "value"}
//	BearerToken     {"kind": "bearer_token", "value", "expires_at"?}
//	AwsCredentials  {"kind": "aws", "access_key_id", "secret_access_key", "session_token"?, "expires_at"?}
//
// A plain string anywhere a credential is accepted reads as an APIKey.
// Secrecy (AUTH-5): String() never shows values.

const expirySkew = 300 * time.Second // AUTH-3: inside the skew window counts as expired.

// Credential is one of APIKey, BearerToken, AwsCredentials.
type Credential interface {
	Kind() string
	IsExpired(now time.Time) bool
	credentialDict() JSONObject
	sealedCredential()
}

// APIKey is an ordinary provider API key.
type APIKey struct{ Value string }

// BearerToken is an OAuth / Entra access token with an optional expiry.
type BearerToken struct {
	Value     string
	ExpiresAt *time.Time
}

// AwsCredentials sign requests with SigV4.
type AwsCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	ExpiresAt       *time.Time
}

func (APIKey) Kind() string              { return "api_key" }
func (BearerToken) Kind() string         { return "bearer_token" }
func (AwsCredentials) Kind() string      { return "aws" }
func (APIKey) sealedCredential()         {}
func (BearerToken) sealedCredential()    {}
func (AwsCredentials) sealedCredential() {}

func (APIKey) IsExpired(time.Time) bool { return false }

func expired(at *time.Time, now time.Time) bool {
	if at == nil {
		return false
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	return at.Sub(now) <= expirySkew
}

func (b BearerToken) IsExpired(now time.Time) bool    { return expired(b.ExpiresAt, now) }
func (a AwsCredentials) IsExpired(now time.Time) bool { return expired(a.ExpiresAt, now) }

// String redacts (AUTH-5).
func (APIKey) String() string { return "APIKey(<redacted>)" }
func (b BearerToken) String() string {
	if b.ExpiresAt != nil {
		return "BearerToken(<redacted>, expires_at=" + FormatRFC3339(*b.ExpiresAt) + ")"
	}
	return "BearerToken(<redacted>)"
}
func (a AwsCredentials) String() string {
	tail := ""
	if a.ExpiresAt != nil {
		tail = ", expires_at=" + FormatRFC3339(*a.ExpiresAt)
	}
	return fmt.Sprintf("AwsCredentials(access_key_id=%q, <redacted>%s)", a.AccessKeyID, tail)
}
func (APIKey) GoString() string           { return "APIKey(<redacted>)" }
func (b BearerToken) GoString() string    { return b.String() }
func (a AwsCredentials) GoString() string { return a.String() }

// Validate checks the value shape.
func (k APIKey) Validate() error {
	if k.Value == "" {
		return valueErrorf("ApiKey.value must be a non-empty string")
	}
	return nil
}

func (b BearerToken) Validate() error {
	if b.Value == "" {
		return valueErrorf("BearerToken.value must be a non-empty string")
	}
	return nil
}

func (a AwsCredentials) Validate() error {
	if a.AccessKeyID == "" || a.SecretAccessKey == "" {
		return valueErrorf("AwsCredentials needs non-empty string access_key_id and secret_access_key")
	}
	return nil
}

func (k APIKey) credentialDict() JSONObject {
	return JSONObject{{"kind", "api_key"}, {"value", k.Value}}
}

func (b BearerToken) credentialDict() JSONObject {
	out := JSONObject{{"kind", "bearer_token"}, {"value", b.Value}}
	if b.ExpiresAt != nil {
		out.Set("expires_at", FormatRFC3339(*b.ExpiresAt))
	}
	return out
}

func (a AwsCredentials) credentialDict() JSONObject {
	out := JSONObject{{"kind", "aws"}, {"access_key_id", a.AccessKeyID}, {"secret_access_key", a.SecretAccessKey}}
	if a.SessionToken != "" {
		out.Set("session_token", a.SessionToken)
	}
	if a.ExpiresAt != nil {
		out.Set("expires_at", FormatRFC3339(*a.ExpiresAt))
	}
	return out
}

// CredentialToDict is the canonical JSON of a credential (AUTH-2). Absent
// fields are omitted, never null. This is the only way a value leaves the
// type: callers that serialize a credential are sending it on purpose.
func CredentialToDict(c Credential) JSONObject { return c.credentialDict() }

// CredentialFromDict reads the canonical JSON form.
func CredentialFromDict(d JSONObject) (Credential, error) {
	kind, _ := d.Get("kind").(string)
	var expires *time.Time
	if raw, ok := d.Lookup("expires_at"); ok && raw != nil {
		t, err := ParseRFC3339(wireStr(raw))
		if err != nil {
			return nil, err
		}
		expires = &t
	}
	switch kind {
	case "api_key":
		v, err := reqString(d, "value")
		if err != nil {
			return nil, err
		}
		c := APIKey{Value: v}
		return c, c.Validate()
	case "bearer_token":
		v, err := reqString(d, "value")
		if err != nil {
			return nil, err
		}
		c := BearerToken{Value: v, ExpiresAt: expires}
		return c, c.Validate()
	case "aws":
		id, err := reqString(d, "access_key_id")
		if err != nil {
			return nil, err
		}
		secret, err := reqString(d, "secret_access_key")
		if err != nil {
			return nil, err
		}
		token, err := optString(d, "session_token")
		if err != nil {
			return nil, err
		}
		c := AwsCredentials{AccessKeyID: id, SecretAccessKey: secret, SessionToken: token, ExpiresAt: expires}
		return c, c.Validate()
	}
	return nil, valueErrorf("unknown credential kind")
}

// CredentialProvider supplies a credential per request (AUTH-2). The adapter
// invokes it at request-build time and never caches the result; caching and
// refreshing belong to the provider itself.
type CredentialProvider interface {
	Credential(ctx context.Context) (Credential, error)
}

// CredentialFunc adapts a function to CredentialProvider.
type CredentialFunc func(ctx context.Context) (Credential, error)

// Credential implements CredentialProvider.
func (f CredentialFunc) Credential(ctx context.Context) (Credential, error) { return f(ctx) }

// StaticCredential wraps a fixed credential as a provider.
type StaticCredential struct{ Value Credential }

// Credential implements CredentialProvider.
func (s StaticCredential) Credential(context.Context) (Credential, error) { return s.Value, nil }

// String redacts.
func (s StaticCredential) String() string { return "StaticCredential(<redacted>)" }

// CredentialLike is what every api_key= argument accepts: a string (the
// APIKey shorthand), a Credential value, or a CredentialProvider.
type CredentialLike = any

// coerceCredentialLike turns a CredentialLike into a provider.
func coerceCredentialLike(v CredentialLike) (CredentialProvider, error) {
	switch x := v.(type) {
	case nil:
		return nil, nil
	case string:
		if x == "" {
			return nil, nil
		}
		return StaticCredential{APIKey{Value: x}}, nil
	case Credential:
		if err := validateCredential(x); err != nil {
			return nil, err
		}
		return StaticCredential{x}, nil
	case CredentialProvider:
		return x, nil
	case func(context.Context) (Credential, error):
		return CredentialFunc(x), nil
	case func() (Credential, error):
		return CredentialFunc(func(context.Context) (Credential, error) { return x() }), nil
	case func() (string, error):
		return CredentialFunc(func(context.Context) (Credential, error) {
			s, err := x()
			if err != nil {
				return nil, err
			}
			return APIKey{Value: s}, nil
		}), nil
	}
	return nil, typeErrorf("not a credential: %T", v)
}

func validateCredential(c Credential) error {
	switch x := c.(type) {
	case APIKey:
		return x.Validate()
	case BearerToken:
		return x.Validate()
	case AwsCredentials:
		return x.Validate()
	}
	return typeErrorf("not a credential: %T", c)
}

// ─── RFC 3339 ────────────────────────────────────────────────────────

// ParseRFC3339 reads "2026-09-03T12:00:00Z" (or an offset) into UTC.
func ParseRFC3339(value string) (time.Time, error) {
	text := strings.TrimSpace(value)
	layouts := []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05", "2006-01-02T15:04:05.999999999", "2006-01-02 15:04:05Z07:00", "2006-01-02 15:04:05"}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, text); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, valueErrorf("invalid RFC 3339 timestamp: %q", value)
}

// FormatRFC3339 renders whole-second UTC: YYYY-MM-DDTHH:MM:SSZ.
func FormatRFC3339(t time.Time) string {
	return t.UTC().Truncate(time.Second).Format("2006-01-02T15:04:05Z")
}
