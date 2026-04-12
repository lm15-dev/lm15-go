package lm15

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// ProviderEnvKeys returns {provider: [env_var_names]} for all core adapters.
func ProviderEnvKeys() map[string][]string {
	return map[string][]string{
		"openai":    {"OPENAI_API_KEY"},
		"anthropic": {"ANTHROPIC_API_KEY"},
		"gemini":    {"GEMINI_API_KEY", "GOOGLE_API_KEY"},
	}
}

// BuildOpts configures BuildDefault.
type BuildOpts struct {
	APIKey       interface{} // string or map[string]string
	ProviderHint string
	Env          string      // path to .env file
	Policy       *TransportPolicy
}

// AdapterFactory creates an adapter from an API key and transport.
type AdapterFactory struct {
	Provider string
	EnvKeys  []string
	Create   func(apiKey string, transport Transport) Adapter
}

// BuildDefaultWithFactories builds a UniversalLM using the given adapter factories.
func BuildDefaultWithFactories(factories []AdapterFactory, opts *BuildOpts) *UniversalLM {
	policy := DefaultPolicy()
	if opts != nil && opts.Policy != nil {
		policy = *opts.Policy
	}
	transport := NewStdTransport(policy)
	client := NewUniversalLM()

	// Build env key map
	envKeyMap := make(map[string]string) // ENV_VAR → provider
	for _, f := range factories {
		for _, k := range f.EnvKeys {
			envKeyMap[k] = f.Provider
		}
	}

	// Resolve explicit keys
	explicit := resolveExplicitKeys(opts, factories)

	// Parse env file
	fileKeys := make(map[string]string)
	if opts != nil && opts.Env != "" {
		fileKeys = parseEnvFile(opts.Env, envKeyMap)
	}

	// Register adapters
	for _, factory := range factories {
		key := explicit[factory.Provider]
		if key == "" {
			key = fileKeys[factory.Provider]
		}
		if key == "" {
			for _, envVar := range factory.EnvKeys {
				if v := os.Getenv(envVar); v != "" {
					key = v
					break
				}
			}
		}
		if key != "" {
			client.Register(factory.Create(key, transport))
		}
	}

	return client
}

func resolveExplicitKeys(opts *BuildOpts, factories []AdapterFactory) map[string]string {
	if opts == nil || opts.APIKey == nil {
		return nil
	}
	result := make(map[string]string)
	switch v := opts.APIKey.(type) {
	case string:
		if opts.ProviderHint != "" {
			result[opts.ProviderHint] = v
		} else {
			for _, f := range factories {
				result[f.Provider] = v
			}
		}
	case map[string]string:
		for k, val := range v {
			result[k] = val
		}
	}
	return result
}

func parseEnvFile(path string, envKeyMap map[string]string) map[string]string {
	result := make(map[string]string)
	expandedPath := path
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			expandedPath = filepath.Join(home, path[2:])
		}
	}

	f, err := os.Open(expandedPath)
	if err != nil {
		return result
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = line[7:]
		}
		eq := strings.Index(line, "=")
		if eq < 0 {
			continue
		}
		key := strings.TrimSpace(line[:eq])
		value := strings.TrimSpace(line[eq+1:])
		// Strip quotes
		if len(value) >= 2 && ((value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}

		if provider, ok := envKeyMap[key]; ok && value != "" {
			result[provider] = value
			// Also set in env for any plugins
			if os.Getenv(key) == "" {
				os.Setenv(key, value)
			}
		}
	}
	return result
}
