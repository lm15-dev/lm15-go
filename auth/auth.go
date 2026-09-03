// Package auth implements the lm15-contract auth surface
// (lm15-contract/spec/auth.md, ratified 2026-08-31): credential providers
// (AUTH-2), the resolution chain (AUTH-1), the explain report (AUTH-7), and
// read-side support for borrowed local CLI credentials (AUTH-8).
//
// Secrecy invariant (AUTH-5): no secret value is ever stored on a Report,
// rendered by Describe, or printed by any String/GoString in this package.
//
// Not yet implemented in this port (stated, not absorbed): the write side of
// AUTH-3/AUTH-4 (locked double-checked refresh, atomic 0600 writes) and the
// AUTH-9 login primitives. This port currently reads credentials only.
package auth

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// CredentialProvider supplies a credential value per request (AUTH-2).
// Adapters must invoke it at request-build time and never cache the value;
// caching belongs to the provider implementation itself.
type CredentialProvider interface {
	Token() (string, error)
}

// Static is a fixed credential value.
type Static string

func (s Static) Token() (string, error) { return string(s), nil }

// String redacts: a Static credential must never print its value.
func (s Static) String() string   { return "auth.Static(redacted)" }
func (s Static) GoString() string { return "auth.Static(redacted)" }

// CredentialFunc adapts a function to CredentialProvider.
type CredentialFunc func() (string, error)

func (f CredentialFunc) Token() (string, error) { return f() }

// providerSpec is one row of the built-in provider table. EnvKeys order is
// normative (AUTH-1: first non-empty wins).
type providerSpec struct {
	envKeys    []string
	defaultKey string // local-server placeholder ("" = none)
	oauthFile  string // "claude-code" | "openai-codex" ("" = not OAuth)
}

// The table mirrors the reference implementation's router knowledge.
var providers = map[string]providerSpec{
	"openai":       {envKeys: []string{"OPENAI_API_KEY"}},
	"openai-chat":  {envKeys: []string{"OPENAI_API_KEY"}},
	"anthropic":    {envKeys: []string{"ANTHROPIC_API_KEY"}},
	"gemini":       {envKeys: []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}},
	"groq":         {envKeys: []string{"GROQ_API_KEY"}},
	"openrouter":   {envKeys: []string{"OPENROUTER_API_KEY"}},
	"deepseek":     {envKeys: []string{"DEEPSEEK_API_KEY"}},
	"zai":          {envKeys: []string{"ZAI_API_KEY"}},
	"ollama":       {defaultKey: "ollama"},
	"vllm":         {defaultKey: "EMPTY"},
	"sglang":       {defaultKey: "EMPTY"},
	"claude-code":  {oauthFile: "claude-code"},
	"openai-codex": {oauthFile: "openai-codex"},
}

// StepState classifies one rung of the chain (AUTH-7).
type StepState string

const (
	Selected StepState = "selected" // this rung supplies the credential
	Shadowed StepState = "shadowed" // usable, but an earlier rung wins
	Absent   StepState = "absent"   // nothing here
)

// Step is one rung of the resolution chain. Kind uses the contract's
// language-neutral vocabulary: "api_keys", "env:<KEY>", "placeholder",
// "oauth-file". Detail is human text and carries no secret material.
type Step struct {
	Kind   string
	Detail string
	State  StepState
}

func (s Step) describe() string {
	marker := map[StepState]string{Selected: "=> ", Shadowed: " ~ ", Absent: " - "}[s.State]
	return fmt.Sprintf("%s%s: %s", marker, s.Kind, s.Detail)
}

// Report is the full answer to "how does this provider's credential
// resolve" (AUTH-7). It never contains secret values.
type Report struct {
	Provider   string
	Steps      []Step
	Configured bool
}

// Selected returns the winning step, if any.
func (r Report) Selected() (Step, bool) {
	for _, step := range r.Steps {
		if step.State == Selected {
			return step, true
		}
	}
	return Step{}, false
}

// Describe renders the rung-by-rung report.
func (r Report) Describe() string {
	var b strings.Builder
	fmt.Fprintf(&b, "auth for provider %q:\n", r.Provider)
	for _, step := range r.Steps {
		fmt.Fprintf(&b, "  %s\n", step.describe())
	}
	if selected, ok := r.Selected(); ok {
		fmt.Fprintf(&b, "  configured: yes — %s", selected.Kind)
	} else {
		b.WriteString("  configured: no")
	}
	return b.String()
}

func (r Report) String() string { return r.Describe() }

// ExplainOptions parameterizes ExplainAuth. Env nil means the process
// environment. Credentials holds explicit per-provider credential providers
// (the api_keys rung); only presence is consulted, never the value.
type ExplainOptions struct {
	Env                   map[string]string
	Credentials           map[string]CredentialProvider
	ClaudeCredentialsPath string // override for ~/.claude/.credentials.json
	CodexAuthPath         string // override for ~/.codex/auth.json
}

// CanonicalProvider maps the permanent underscore alias to the hyphenated
// provider string.
func CanonicalProvider(name string) string { return strings.ReplaceAll(name, "_", "-") }

// KnownProviders lists every provider in the built-in table, sorted.
func KnownProviders() []string {
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func envValue(opts ExplainOptions, key string) string {
	if opts.Env != nil {
		return opts.Env[key]
	}
	return os.Getenv(key)
}

// ExplainAuth walks the AUTH-1 chain for one provider and reports every
// rung (AUTH-7). No network I/O. Env values are tested for presence only;
// they are never retained on the report — that presence check is the one
// stated purity trade-off.
func ExplainAuth(provider string, opts ExplainOptions) (Report, error) {
	canonical := CanonicalProvider(provider)
	spec, ok := providers[canonical]
	if !ok {
		return Report{}, fmt.Errorf(
			"unknown provider %q; known providers: %s",
			provider, strings.Join(KnownProviders(), ", "),
		)
	}

	if spec.oauthFile != "" {
		step := oauthFileStep(spec.oauthFile, opts)
		return Report{
			Provider:   canonical,
			Steps:      []Step{step},
			Configured: step.State == Selected,
		}, nil
	}

	steps := make([]Step, 0, 2+len(spec.envKeys))
	selected := false

	if _, has := opts.Credentials[canonical]; has {
		steps = append(steps, Step{Kind: "api_keys", Detail: "provided (value never shown)", State: Selected})
		selected = true
	} else {
		steps = append(steps, Step{Kind: "api_keys", Detail: "not provided", State: Absent})
	}

	for _, key := range spec.envKeys {
		kind := "env:" + key
		if envValue(opts, key) != "" {
			state := Selected
			if selected {
				state = Shadowed
			}
			steps = append(steps, Step{Kind: kind, Detail: "set (value never shown)", State: state})
			selected = true
		} else {
			steps = append(steps, Step{Kind: kind, Detail: "not set", State: Absent})
		}
	}

	if spec.defaultKey != "" {
		state := Selected
		if selected {
			state = Shadowed
		}
		steps = append(steps, Step{
			Kind:   "placeholder",
			Detail: fmt.Sprintf("preset default for keyless %s servers", canonical),
			State:  state,
		})
		selected = true
	}

	return Report{Provider: canonical, Steps: steps, Configured: selected}, nil
}

func oauthFileStep(provider string, opts ExplainOptions) Step {
	var path string
	var credential *LocalOAuthCredential
	var err error
	switch provider {
	case "claude-code":
		path = opts.ClaudeCredentialsPath
		if path == "" {
			path = DefaultClaudeCredentialsPath()
		}
		credential, err = ReadClaudeCodeCredential(path)
	default: // openai-codex
		path = opts.CodexAuthPath
		if path == "" {
			path = DefaultCodexAuthPath()
		}
		credential, err = ReadCodexCLICredential(path)
	}
	if err != nil || credential == nil {
		return Step{Kind: "oauth-file", Detail: "missing or unreadable", State: Absent}
	}
	if credential.Expired() {
		if credential.HasRefreshToken() {
			return Step{Kind: "oauth-file", Detail: "expired, refresh token present", State: Selected}
		}
		return Step{Kind: "oauth-file", Detail: "expired, NO refresh token", State: Absent}
	}
	return Step{Kind: "oauth-file", Detail: "fresh", State: Selected}
}

// DefaultClaudeCredentialsPath is ~/.claude/.credentials.json (AUTH-8).
func DefaultClaudeCredentialsPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude", ".credentials.json")
}

// DefaultCodexAuthPath is ~/.codex/auth.json (AUTH-8).
func DefaultCodexAuthPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".codex", "auth.json")
}
