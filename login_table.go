package lm15

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// The providers a manager can connect (AUTH-13), from definitions only, and
// the recipe connections (AUTH-15): ports of lm15-python
// lm15/login/flows/__init__.py and flows/recipes.py. Account flows are
// hand-written per provider; key, env, external, cloud and local recipes are
// generated from the provider registry. Recipes are recipes, not tokens: an
// env connection saves the variable's name, an external one names another
// tool's login (read and renewed in place, never copied — R1), local a
// keyless server's URL, cloud a named cloud identity.

const radiusID = "radius"

// ExternalSources: source id → provider route and human label.
var ExternalSources = [][3]string{
	{"claude-code-cli", "claude-code", "your Claude Code login (~/.claude/.credentials.json)"},
	{"codex-cli", "openai-codex", "your Codex CLI login (~/.codex/auth.json)"},
	{"pi-xai", "xai", "your Pi agent xAI login (~/.pi/agent/auth.json)"},
}

var serviceLabels = map[string]string{
	"anthropic": "Anthropic", "claude-code": "Anthropic", "openai": "OpenAI", "openai-chat": "OpenAI",
	"openai-codex": "OpenAI", "gemini": "Google", "vertex": "Google Cloud", "vertex-anthropic": "Google Cloud",
	"vertex-express": "Google Cloud", "azure": "Microsoft Azure", "azure-chat": "Microsoft Azure",
	"azure-anthropic": "Microsoft Azure", "aws-anthropic": "AWS", "bedrock-anthropic": "AWS",
	"bedrock-chat": "AWS", "bedrock-mantle-chat": "AWS", "meta": "Meta", "meta-chat": "Meta",
	"meta-anthropic": "Meta", "moonshotai": "Moonshot AI", "moonshotai-anthropic": "Moonshot AI",
	"moonshotai-responses": "Moonshot AI", "kimi-code": "Moonshot AI", "deepseek": "DeepSeek",
	"deepseek-anthropic": "DeepSeek", "groq": "Groq", "openrouter": "OpenRouter", "xai": "xAI",
	"zai": "Z.AI", "typesafe": "TypeSafe", "ollama": "Local", "vllm": "Local", "sglang": "Local",
	"github-copilot": "GitHub",
}

func recipeMethod(id, label, kind, flow string, fields []MethodField, note string) LoginMethod {
	return LoginMethod{ID: id, Label: label, Kind: kind, Flow: flow, Availability: "supported", Fields: fields, Delivery: []string{}, BillingNote: note}
}

func recipeMethods(provider string) []LoginMethod {
	var methods []LoginMethod
	for _, source := range ExternalSources {
		if source[1] == provider {
			m := recipeMethod("external:"+source[0], "Use "+source[2], "account", "source_recipe", nil, "Whatever that tool's login is entitled to; LM15 reads and renews it in place and copies nothing.")
			m.Subscription = true
			m.Guidance = "Sign in with that tool first if it says no credential is present."
			methods = append(methods, m)
		}
	}
	def, ok := Providers[provider]
	if !ok {
		return methods
	}
	if def.Access.CloudChain() {
		options := make([]SelectOption, len(NamedCredentials))
		for i, n := range NamedCredentials {
			options[i] = SelectOption{ID: n, Label: n}
		}
		methods = append(methods, recipeMethod("cloud", "Use a named cloud identity", "cloud_identity", "source_recipe",
			[]MethodField{{ID: "named", Label: "Identity", Type: "select", Required: true, Options: options}}, "Billed to that cloud account."))
	}
	if def.PlaceholderKey != "" {
		return append(methods, recipeMethod("local", "Local server (no key needed)", "local_server", "source_recipe",
			[]MethodField{{ID: "base_url", Label: "Server URL", Type: "text", Required: false}}, ""))
	}
	if def.CredentialPolicy() != "oauth" {
		methods = append(methods, recipeMethod("api_key", "Paste an API key", "api_key", "form",
			[]MethodField{{ID: "key", Label: "API key", Type: "secret", Required: true}}, "Metered per token by the provider."))
		if keys := def.Access.EnvKeys; len(keys) > 0 {
			options := make([]SelectOption, len(keys))
			for i, k := range keys {
				options[i] = SelectOption{ID: k, Label: "$" + k}
			}
			methods = append(methods, recipeMethod("env", "Use the key in $"+keys[0]+" from the environment", "api_key", "source_recipe",
				[]MethodField{{ID: "name", Label: "Environment variable", Type: "select", Required: true, Options: options}},
				"Metered per token by the provider; the variable's value is read at request time, never saved."))
		}
	}
	return methods
}

func loginProviderIDs() []string {
	seen := map[string]bool{radiusID: true}
	for id := range Providers {
		seen[id] = true
	}
	for _, f := range accountFlows {
		seen[string(f)] = true
	}
	out := make([]string, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// loginDescriptor is the AUTH-13 descriptor on this host; listener: a
// loopback return listener exists (native), so loopback delivery is kept.
func loginDescriptor(provider string, listener bool) (ProviderDescriptor, bool) {
	id := CanonicalProvider(provider)
	if id == radiusID {
		m := recipeMethod("browser", "Sign in with Radius", "account", "authorization_code", nil, "")
		m.Availability = "unavailable"
		m.Reason = "Radius's model protocol is not implemented in lm15; login without inference would be a false 'supported' claim"
		return ProviderDescriptor{ID: id, Label: "Radius", Service: "Radius", Routes: []string{}, Methods: []LoginMethod{m}}, true
	}
	recipes := recipeMethods(id)
	if flow, ok := accountFlowFor(id); ok {
		d := flow.descriptor()
		var own []LoginMethod
		for _, m := range d.Methods {
			if m.Kind != "account" || strings.HasPrefix(m.ID, "external:") {
				continue
			}
			var delivery []string
			for _, x := range m.Delivery {
				if x != "loopback" || listener {
					delivery = append(delivery, x)
				}
			}
			if delivery == nil {
				delivery = []string{}
			}
			m.Delivery = delivery
			if len(delivery) == 0 && m.Availability != "unavailable" {
				m.Availability = "unavailable"
				m.Reason = "needs a local callback listener, which this host does not provide"
			}
			own = append(own, m)
		}
		d.Methods = append(own, recipes...)
		return d, true
	}
	def, ok := Providers[id]
	if !ok {
		return ProviderDescriptor{}, false
	}
	service := serviceLabels[id]
	if service == "" {
		service = id
	}
	if recipes == nil {
		recipes = []LoginMethod{}
	}
	return ProviderDescriptor{ID: id, Label: id, Service: service, Routes: []string{id}, Methods: recipes, ConsoleURL: def.ConsoleURL}, true
}

// ─── Recipes ─────────────────────────────────────────────────────────

func recipeLogin(provider, method string, answers, settings map[string]string, home string) (flowResult, error) {
	switch {
	case method == "api_key":
		key := strings.TrimSpace(answers["key"])
		if key == "" {
			return flowResult{}, denied("no API key was entered")
		}
		return flowResult{material: JSONObject{"type": "api_key", "key": key}, label: provider + " API key", renewal: "none"}, nil
	case method == "env":
		name := answers["name"]
		if name == "" {
			return flowResult{}, denied("no environment variable was chosen")
		}
		return flowResult{material: JSONObject{"type": "env", "name": name}, label: provider + " key from $" + name, renewal: "recipe"}, nil
	case method == "cloud":
		named := answers["named"]
		if !inVocab(named, NamedCredentials) {
			return flowResult{}, denied("choose one of %s", strings.Join(NamedCredentials, ", "))
		}
		return flowResult{material: JSONObject{"type": "cloud", "named": named}, label: provider + " via " + named + " identity", renewal: "recipe"}, nil
	case method == "local":
		base := answers["base_url"]
		if base == "" {
			base = settings["base_url"]
		}
		key := answers["key"]
		if key == "" {
			key = "local"
		}
		result := flowResult{material: JSONObject{"type": "local", "base_url": base, "key": key}, label: provider + " local server", renewal: "none", settings: map[string]string{}}
		if base != "" {
			result.settings["base_url"] = base
		}
		return result, nil
	case strings.HasPrefix(method, "external:"):
		source := strings.TrimPrefix(method, "external:")
		label := ""
		for _, s := range ExternalSources {
			if s[0] == source {
				label = s[2]
			}
		}
		if label == "" {
			return flowResult{}, denied("unknown external source %q", source)
		}
		if err := probeExternal(source, home); err != nil {
			return flowResult{}, err // fail now, typed, if that tool has no login here
		}
		return flowResult{material: JSONObject{"type": "external", "source": source}, label: provider + " via " + label, renewal: "external"}, nil
	}
	return flowResult{}, denied("unknown recipe method %q", method)
}

func externalPath(source, home string) string {
	if home == "" {
		home = homeDir()
	}
	switch source {
	case "claude-code-cli":
		return filepath.Join(home, ".claude", ".credentials.json")
	case "codex-cli":
		return filepath.Join(home, ".codex", "auth.json")
	}
	return filepath.Join(home, ".pi", "agent", "auth.json")
}

func probeExternal(source, home string) error {
	path := externalPath(source, home)
	var err error
	switch source {
	case "claude-code-cli":
		_, err = LoadClaudeCodeCredential(path)
	case "codex-cli":
		_, err = LoadCodexCLICredential(path)
	default:
		_, err = LoadXaiCredential(path)
	}
	return err
}

// externalRequestAuth reads (and renews in place, under that tool's lock) another tool's login.
func externalRequestAuth(ctx context.Context, source, home string) (RequestAuth, error) {
	path := externalPath(source, home)
	auth := RequestAuth{CredentialKind: "bearer", Headers: map[string]string{}}
	switch source {
	case "claude-code-cli":
		token, err := GetClaudeCodeAccessToken(ctx, path, true)
		if err != nil {
			return auth, err
		}
		auth.Credential = token
	case "codex-cli":
		cred, err := GetCodexCLIAccessToken(ctx, path, true)
		if err != nil {
			return auth, err
		}
		auth.Credential = cred.AccessToken
		account := cred.AccountID
		if account == "" {
			account = ExtractChatGPTAccountID(cred.AccessToken)
		}
		if account != "" {
			auth.Headers["chatgpt-account-id"] = account
		}
		auth.AccountID = account
	default:
		token, err := GetXaiAccessToken(ctx, path, true)
		if err != nil {
			return auth, err
		}
		auth.Credential = token
	}
	return auth, nil
}

func externalPeek(source, home string) RequestAuth {
	auth := RequestAuth{Headers: map[string]string{}}
	if source != "codex-cli" {
		return auth
	}
	if cred, err := LoadCodexCLICredential(externalPath(source, home)); err == nil {
		account := cred.AccountID
		if account == "" {
			account = ExtractChatGPTAccountID(cred.AccessToken)
		}
		if account != "" {
			auth.Headers["chatgpt-account-id"] = account
		}
		auth.AccountID = account
	}
	return auth
}

func recipeRequestAuth(material JSONObject, env func(string) string) (RequestAuth, error) {
	auth := RequestAuth{Headers: map[string]string{}}
	switch materialStr(material, "type") {
	case "api_key":
		auth.CredentialKind, auth.Credential = "api_key", materialStr(material, "key")
	case "env":
		name := materialStr(material, "name")
		value := env(name)
		if value == "" {
			return auth, denied("$%s is not set in this process's environment", name)
		}
		auth.CredentialKind, auth.Credential = "api_key", value
	case "local":
		key := materialStr(material, "key")
		if key == "" {
			key = "local"
		}
		auth.CredentialKind, auth.Credential, auth.BaseURL = "api_key", key, materialStr(material, "base_url")
	case "cloud":
		auth.Named = materialStr(material, "named")
	default:
		return auth, denied("unknown connection material %q", fmt.Sprint(material["type"]))
	}
	return auth, nil
}

func isRecipe(material JSONObject) bool {
	switch materialStr(material, "type") {
	case "api_key", "env", "external", "local", "cloud":
		minted, _ := material["minted"].(bool)
		return !minted
	}
	return false
}
