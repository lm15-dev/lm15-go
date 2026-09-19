package lm15

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ExplainAuth answers "why is my key (not) being used?" (AUTH-7): the exact
// AUTH-1 chain, rung by rung, no network, no secrets.

// AuthStep is one rung of the credential chain.
type AuthStep struct {
	Kind   string // api_keys | env:<VAR> | placeholder | oauth-file | the cloud rung names
	Source string
	Detail string
	State  string // selected | shadowed | absent | unprobed
}

// Describe renders the rung.
func (s AuthStep) Describe() string {
	marker := map[string]string{"selected": "=> ", "shadowed": " ~ ", "absent": " - ", "unprobed": " ? "}[s.State]
	return marker + s.Source + ": " + s.Detail
}

// AuthReport is the full answer for one provider.
type AuthReport struct {
	Provider   string
	Steps      []AuthStep
	Configured bool
	Settings   [][2]string
}

// Selected returns the winning step.
func (r AuthReport) Selected() (AuthStep, bool) {
	for _, s := range r.Steps {
		if s.State == "selected" {
			return s, true
		}
	}
	return AuthStep{}, false
}

// Describe renders the report (no secrets by construction).
func (r AuthReport) Describe() string {
	lines := []string{fmt.Sprintf("auth for provider %q:", r.Provider)}
	for _, s := range r.Steps {
		lines = append(lines, "  "+s.Describe())
	}
	var unprobed []string
	for _, s := range r.Steps {
		if s.State == "unprobed" {
			unprobed = append(unprobed, s.Source)
		}
	}
	selected, ok := r.Selected()
	switch {
	case r.Configured && ok:
		lines = append(lines, "  configured: yes — "+selected.Source)
		if len(unprobed) > 0 {
			lines = append(lines, "  note: "+strings.Join(unprobed, ", ")+" run first at request time and may win")
		}
	case r.Configured:
		lines = append(lines, "  configured: probably — "+strings.Join(unprobed, ", ")+" (unprobed offline)")
	default:
		lines = append(lines, "  configured: no")
	}
	for _, s := range r.Settings {
		lines = append(lines, "  setting "+s[0]+": "+s[1])
	}
	return strings.Join(lines, "\n")
}

func (r AuthReport) String() string { return r.Describe() }

// ExplainOptions parameterizes ExplainAuth. Env nil = the process env.
type ExplainOptions struct {
	Env                   map[string]string
	APIKeys               map[string]CredentialLike
	ClaudeCredentialsPath string
	CodexAuthPath         string
	XaiCredentialsPath    string
	Files                 map[string]string
	Home                  string
	Settings              map[string]string
	// Now overrides the clock used for expiry details.
	Now func() time.Time
}

func expiryDetail(cred LocalOAuthCredential, now time.Time) string {
	if cred.ExpiresAt == nil {
		return "no recorded expiry"
	}
	remaining := *cred.ExpiresAt - now.UnixMilli()
	if remaining <= 0 {
		if cred.RefreshToken != "" {
			return "expired, refresh token present"
		}
		return "expired, NO refresh token"
	}
	minutes := remaining / 60000
	hours, mins := minutes/60, minutes%60
	if hours > 0 {
		return fmt.Sprintf("fresh, expires in %dh %02dm", hours, mins)
	}
	return fmt.Sprintf("fresh, expires in %dm", mins)
}

func usableState(cred LocalOAuthCredential, detail string, shadowed bool) string {
	if strings.Contains(detail, "expired") && cred.RefreshToken == "" {
		return "absent"
	}
	if shadowed {
		return "shadowed"
	}
	return "selected"
}

func oauthStep(provider, override string, now time.Time) AuthStep {
	var path string
	var cred *LocalOAuthCredential
	if provider == "claude-code" {
		path = coercePath(override, ClaudeCodeCredentialsPath())
		cred = ReadClaudeCodeCredential(path)
	} else {
		path = coercePath(override, CodexCLIAuthPath())
		cred = ReadCodexCLICredential(path)
	}
	source := "local OAuth credential " + path
	if cred == nil {
		return AuthStep{Kind: "oauth-file", Source: source, Detail: "missing or unreadable", State: "absent"}
	}
	detail := expiryDetail(*cred, now)
	return AuthStep{Kind: "oauth-file", Source: source, Detail: detail, State: usableState(*cred, detail, false)}
}

func xaiOAuthStep(override string, shadowed bool, now time.Time) AuthStep {
	paths := xaiStorePaths()
	if override != "" {
		paths = []string{expandHome(override)}
	}
	cred, path, err := loadXaiWithSource(override)
	if err != nil {
		return AuthStep{Kind: "oauth-file", Source: "local OAuth credential " + strings.Join(paths, " or "), Detail: "missing or unreadable", State: "absent"}
	}
	detail := expiryDetail(cred, now)
	return AuthStep{Kind: "oauth-file", Source: "local OAuth credential " + path, Detail: detail, State: usableState(cred, detail, shadowed)}
}

func entrySource(provider, entry string) string {
	source := "explicit api_keys entry"
	if entry != "" && CanonicalProvider(entry) != provider {
		source += fmt.Sprintf(" (via %q, shared env-key declarations)", entry)
	}
	return source
}

// ExplainAuth walks the AUTH-1 chain for a provider and reports every rung.
func ExplainAuth(provider string, opts ExplainOptions) (AuthReport, error) {
	canonical := CanonicalProvider(provider)
	def, ok := Providers[canonical]
	if !ok {
		return AuthReport{}, valueErrorf("Unknown provider %q. Known providers: %s", provider, knownProviders())
	}
	now := time.Now().UTC()
	if opts.Now != nil {
		now = opts.Now()
	}
	config := RouterConfig{Env: opts.Env, APIKeys: opts.APIKeys}
	if def.Access.CloudChain() || def.Hosted() {
		return explainCloud(canonical, def, config, opts)
	}
	policy := def.CredentialPolicy()
	if policy == "oauth" {
		override := opts.CodexAuthPath
		if canonical == "claude-code" {
			override = opts.ClaudeCredentialsPath
		}
		step := oauthStep(canonical, override, now)
		return AuthReport{Provider: canonical, Steps: []AuthStep{step}, Configured: step.State == "selected"}, nil
	}
	env := config.env()
	var steps []AuthStep
	selected := false
	entry, err := apiKeysSource(config, canonical)
	if err != nil {
		return AuthReport{}, err
	}
	if entry != "" {
		steps = append(steps, AuthStep{Kind: "api_keys", Source: entrySource(canonical, entry), Detail: "provided (value never shown)", State: "selected"})
		selected = true
	} else {
		steps = append(steps, AuthStep{Kind: "api_keys", Source: "explicit api_keys entry", Detail: "not provided", State: "absent"})
	}
	if policy == "oauth-unless-explicit" {
		step := xaiOAuthStep(opts.XaiCredentialsPath, selected, now)
		steps = append(steps, step)
		selected = selected || step.State == "selected"
	}
	for _, key := range def.Access.EnvKeys {
		if env[key] != "" {
			state := "selected"
			if selected {
				state = "shadowed"
			}
			steps = append(steps, AuthStep{Kind: "env:" + key, Source: "env $" + key, Detail: "set (value never shown)", State: state})
			selected = true
		} else {
			steps = append(steps, AuthStep{Kind: "env:" + key, Source: "env $" + key, Detail: "not set", State: "absent"})
		}
	}
	if def.PlaceholderKey != "" {
		state := "selected"
		if selected {
			state = "shadowed"
		}
		steps = append(steps, AuthStep{Kind: "placeholder", Source: "local-server placeholder key", Detail: "preset default for keyless " + canonical + " servers", State: state})
		selected = true
	}
	return AuthReport{Provider: canonical, Steps: steps, Configured: selected}, nil
}

func explainCloud(canonical string, def ProviderDefinition, config RouterConfig, opts ExplainOptions) (AuthReport, error) {
	env := config.env()
	entry, err := apiKeysSource(config, canonical)
	if err != nil {
		return AuthReport{}, err
	}
	home := opts.Home
	if home == "" {
		home = env["HOME"]
	}
	if home == "" {
		home = homeDir()
	}
	ctx := &ChainContext{Env: env, Home: expandHome(home), Files: opts.Files, Now: opts.Now}
	settingError := ""
	resolved, err := resolveSettings(def.Access.Host, opts.Settings, env, canonical, ProfileSettings(def.Access, ctx))
	if err != nil {
		if IsKind(err, KindNotConfigured) {
			settingError = firstLine(err)
			resolved = map[string]string{}
		} else {
			return AuthReport{}, err
		}
	}
	ctx.Settings = resolved
	var steps []AuthStep
	configured := false
	if def.Access.CloudChain() {
		chainSteps, isConfigured, err := ExplainChain(def.Access, ctx, entry != "")
		if err != nil {
			return AuthReport{}, err
		}
		configured = isConfigured
		for _, s := range chainSteps {
			source := s.Source
			if s.Kind == "api_keys" {
				source = entrySource(canonical, entry)
			}
			steps = append(steps, AuthStep{Kind: s.Kind, Source: source, Detail: s.Detail, State: s.State})
		}
	} else {
		state, detail := "absent", "not provided"
		if entry != "" {
			state, detail, configured = "selected", "provided (value never shown)", true
		}
		steps = append(steps, AuthStep{Kind: "api_keys", Source: entrySource(canonical, entry), Detail: detail, State: state})
		for _, key := range def.Access.EnvKeys {
			if env[key] != "" {
				st := "selected"
				if configured {
					st = "shadowed"
				}
				steps = append(steps, AuthStep{Kind: "env:" + key, Source: "env $" + key, Detail: "set (value never shown)", State: st})
				configured = true
			} else {
				steps = append(steps, AuthStep{Kind: "env:" + key, Source: "env $" + key, Detail: "not set", State: "absent"})
			}
		}
	}
	names := make([]string, 0, len(resolved))
	for k := range resolved {
		names = append(names, k)
	}
	sort.Strings(names)
	var shown [][2]string
	for _, k := range names {
		shown = append(shown, [2]string{k, resolved[k]})
	}
	if settingError != "" {
		shown = append(shown, [2]string{"error", settingError})
	}
	return AuthReport{Provider: canonical, Steps: steps, Configured: configured, Settings: shown}, nil
}
