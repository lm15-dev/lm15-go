package lm15

import (
	"context"
	"errors"
	"sort"
)

// Managed authentication — sign in once, use everywhere
// (spec/auth-managed.md AUTH-12–26). The same component as lm15-python's
// lm15.login, lm15-ts's Auth and lm15-rs's lm15::login: same rules, same
// store file (lm15-contract auth/managed/store-layout.md), graded by the same
// runs (the contract harness's `managed` direction), so a login saved by one
// SDK is used, renewed and signed out by another.
//
// This file is the public, secret-free vocabulary (AUTH-12, AUTH-13, AUTH-16,
// AUTH-23, AUTH-24): a Connection is metadata about a saved credential, never
// the credential.

// SelectOption is one choice of a select field or prompt.
type SelectOption struct {
	ID          string
	Label       string
	Description string
}

// MethodField is one input a login method needs before it can start.
type MethodField struct {
	ID       string
	Label    string
	Type     string // text | secret | select
	Required bool
	Options  []SelectOption
	Help     string
}

// LoginMethod is a named way to establish a connection (AUTH-13.3).
// Availability: "supported" has a recorded receipt; "unverified" exists
// without one (explicit opt-in only); "unavailable" cannot run here.
type LoginMethod struct {
	ID           string
	Label        string
	Kind         string // account | api_key | cloud_identity | local_server
	Flow         string // authorization_code | device_code | form | source_recipe
	Availability string
	Reason       string
	Fields       []MethodField
	Delivery     []string // loopback | manual | device
	Subscription bool     // backed by a provider subscription, per provider docs; never an entitlement promise
	BillingNote  string
	Guidance     string
}

// ProviderDescriptor is a provider a manager can connect (AUTH-13.1). ID is
// the lm15 route; Service is a presentation group only.
type ProviderDescriptor struct {
	ID         string
	Label      string
	Service    string
	Routes     []string
	Methods    []LoginMethod
	ConsoleURL string
}

// Method returns the method with this id.
func (d ProviderDescriptor) Method(id string) (LoginMethod, bool) {
	for _, m := range d.Methods {
		if m.ID == id {
			return m, true
		}
	}
	return LoginMethod{}, false
}

// Connection is secret-free metadata for one saved credential in a scope.
type Connection struct {
	ID                 string
	Provider           string
	InstanceID         string
	Kind               string
	MethodID           string
	Routes             []string
	Label              string
	CreatedAt          string
	IdentityGeneration string
	CredentialRevision string
	Settings           map[string]string
	AccountLabel       string // untrusted display text (AUTH-12); never proof of identity
}

// Verification is the last explicit check of a connection.
type Verification struct {
	Result    string // valid | rejected | unverified
	CheckedAt string
	Check     string
	Detail    string
}

// ConnectionStatus (AUTH-24): presence, usability and last verification are separate.
type ConnectionStatus struct {
	Provider     string
	Presence     string // saved | absent
	Usability    string // ready | renewal_due | needs_login | indeterminate | unknown
	Connection   *Connection
	ExpiresAt    string // RFC 3339, "never", "unknown", or "" when nothing expires
	LoggedOut    bool
	Verification *Verification
	Detail       string
}

// Ready reports usability ready or renewal_due.
func (s ConnectionStatus) Ready() bool { return s.Usability == "ready" || s.Usability == "renewal_due" }

// ForgetResult is what a logout did.
type ForgetResult struct {
	Provider           string
	Forgot             bool
	Routes             []string
	IdentityGeneration string
}

// RequestAuth is what a request sends for a saved connection. Credential is secret.
type RequestAuth struct {
	CredentialKind string // bearer | api_key | "" for a named cloud identity
	Credential     string
	Headers        map[string]string
	BaseURL        string
	AccountID      string
	Named          string // a saved cloud recipe: the identity the router's chain runs (AUTH-15)
}

// String never renders the credential.
func (r RequestAuth) String() string {
	keys := make([]string, 0, len(r.Headers))
	for k := range r.Headers {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return "RequestAuth(kind=" + r.CredentialKind + ", headers=" + joinComma(keys) + ", named=" + r.Named + ")"
}

func joinComma(items []string) string {
	out := ""
	for i, s := range items {
		if i > 0 {
			out += ","
		}
		out += s
	}
	return out
}

// ─── The UI boundary (AUTH-16) ────────────────────────────────────────

// Prompt asks the person for one answer. Type: text | secret | select | manual_code.
type Prompt struct {
	Type        string
	FieldID     string
	Label       string
	Placeholder string
	Options     []SelectOption // select
	Accepted    string         // manual_code: what may be pasted
}

// Notice tells the person something. Type: auth_url | device_code | progress | info.
type Notice struct {
	Type            string
	URL             string // auth_url: session-sensitive (state, PKCE challenge)
	Instructions    string
	UserCode        string // device_code: show it, never log it (AUTH-21)
	VerificationURL string
	ExpiresInS      float64
	IntervalS       float64
	Stage           string // progress
	Message         string // progress, info
	Links           [][2]string
}

// ErrPromptCancelled is what a UI returns when the person cancels a prompt.
var ErrPromptCancelled = errors.New("prompt cancelled")

// AuthUI is what an application supplies so a login can talk to a person.
// Prompt returns the answer (a select answers the option id) and must return
// promptly with ctx.Err() or ErrPromptCancelled once ctx is done. A UI never
// opens anything unless that is the application's own choice.
type AuthUI interface {
	Prompt(ctx context.Context, p Prompt) (string, error)
	Notify(n Notice)
}

// Dismisser is an optional AuthUI method: a displayed prompt became stale
// (the loopback return won the race).
type Dismisser interface {
	Dismiss(p Prompt)
}

// ModelSelection is an exact route + model bound to one connection id and generation.
type ModelSelection struct {
	Provider           string
	Model              string
	ConnectionID       string
	IdentityGeneration string
}

// Routed is "provider:model".
func (s ModelSelection) Routed() string { return s.Provider + ":" + s.Model }

// ModelChoice is a model the saved connection can select, and where that came from.
type ModelChoice struct {
	Provider     string
	Model        string
	ConnectionID string
	Source       string // provider | manual
	FetchedAt    string
	Capability   string // the requested capability, if any
	State        string // supported | unsupported | unknown
}
