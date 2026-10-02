package lm15

import "testing"

// TestTableConstants pins the package's exported constants to the generated
// table (tables_generated.go, from lm15-contract tables/providers.json). Go
// constants cannot be computed from variables, so they stay literal; this is
// what keeps each one a copy of the table rather than a second source.
func TestTableConstants(t *testing.T) {
	setting := func(p AccessPolicy, name string) HostSetting {
		for _, s := range p.BackendSettings {
			if s.Name == name {
				return s
			}
		}
		t.Fatalf("%s: no backend setting %q in the table", p.Provider, name)
		return HostSetting{}
	}
	header := func(p AccessPolicy, name string) string {
		for _, h := range p.Headers {
			if h[0] == name {
				return h[1]
			}
		}
		t.Fatalf("%s: no %q header in the table", p.Provider, name)
		return ""
	}
	claude, codex, xai, copilot := tableAccess("claude-code"), tableAccess("openai-codex"), tableAccess("xai"), tableAccess("github-copilot")
	for _, c := range []struct{ name, constant, table string }{
		{"DefaultClaudeCodeVersion", DefaultClaudeCodeVersion, claude.BackendOptions["client_version"]},
		{"ClaudeCodeVersionEnv", ClaudeCodeVersionEnv, setting(claude, "client_version").Env[0]},
		{"DefaultClaudeCodeSystemPrompt", DefaultClaudeCodeSystemPrompt, claude.SystemPrefix},
		{"ClaudeCodeLoginHint", ClaudeCodeLoginHint, claude.LoginHint},
		{"claude-code user-agent", "claude-cli/" + DefaultClaudeCodeVersion, header(claude, "user-agent")},
		{"CodexClientVersionEnv", CodexClientVersionEnv, setting(codex, "client_version").Env[0]},
		{"DefaultCodexBaseURL", DefaultCodexBaseURL, codex.BaseURL},
		{"DefaultCodexOriginator", DefaultCodexOriginator, header(codex, "originator")},
		{"DefaultCodexInstructions", DefaultCodexInstructions, codex.SystemPrefix},
		{"DefaultCodexClientVersion", DefaultCodexClientVersion, codex.BackendOptions["client_version"]},
		{"CodexBackend", CodexBackend, codex.Backend},
		{"OpenAICodexLoginHint", OpenAICodexLoginHint, codex.LoginHint},
		{"DefaultXaiBaseURL", DefaultXaiBaseURL, xai.BaseURL},
		{"XaiLoginHint", XaiLoginHint, xai.LoginHint},
		{"copilotDefaultAPIBase", copilotDefaultAPIBase, copilot.BaseURL},
	} {
		if c.constant != c.table {
			t.Errorf("%s = %q, the generated table says %q: change the constant (the table is the reference's)", c.name, c.constant, c.table)
		}
	}
}

// TestTableRowsAreTheRegistry checks the registry and the routing tables are
// read from the generated rows, in their order.
func TestTableRowsAreTheRegistry(t *testing.T) {
	if len(providerDefinitions) != len(tableProviders) {
		t.Fatalf("%d definitions for %d table rows", len(providerDefinitions), len(tableProviders))
	}
	for i, r := range tableProviders {
		d := providerDefinitions[i]
		if d.ID != r.ID || d.Dialect != r.Dialect || d.Compat != r.Compat || d.Note != r.Note || d.AdapterOwned != (r.Kind == "adapter-owned") {
			t.Errorf("row %d (%s): definition %+v does not read the table row", i, r.ID, d)
		}
	}
	if len(DefaultRules) != len(tableDefaultRules) || len(LitellmProviderPrefixes) != len(tableLitellmPrefixes) {
		t.Errorf("router tables are not the generated ones")
	}
}
