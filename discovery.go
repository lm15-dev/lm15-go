package lm15

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"time"
)

// Models discovers available models from live provider APIs and models.dev.
func Models(opts *ModelsOpts) ([]ModelSpec, error) {
	if opts == nil {
		opts = &ModelsOpts{}
	}
	envMap := ProviderEnvKeys()
	allProviders := make([]string, 0, len(envMap))
	for p := range envMap {
		allProviders = append(allProviders, p)
	}
	selected := allProviders
	if opts.Provider != "" {
		selected = []string{opts.Provider}
	}
	keys := resolveAPIKeys(opts.APIKey, opts.Provider)
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 5 * time.Second
	}

	var liveSpecs []ModelSpec
	if !opts.SkipLive {
		for _, p := range selected {
			key := keys[p]
			if key == "" {
				continue
			}
			fetched, err := fetchLiveModels(p, key, timeout)
			if err != nil {
				continue
			}
			liveSpecs = append(liveSpecs, fetched...)
		}
	}

	var fallback []ModelSpec
	all, err := FetchModelsDev(timeout)
	if err == nil {
		selectedSet := make(map[string]bool)
		for _, s := range selected {
			selectedSet[s] = true
		}
		for _, s := range all {
			if selectedSet[s.Provider] {
				fallback = append(fallback, s)
			}
		}
	}

	merged := mergeSpecs(liveSpecs, fallback)
	return filterSpecs(merged, opts), nil
}

// ProvidersInfo returns status info for each provider.
func ProvidersInfo(opts *ModelsOpts) (map[string]ProviderInfo, error) {
	envMap := ProviderEnvKeys()
	keys := resolveAPIKeys(nil, "")
	if opts != nil {
		keys = resolveAPIKeys(opts.APIKey, "")
	}

	specs, _ := Models(opts)
	counts := make(map[string]int)
	for _, s := range specs {
		counts[s.Provider]++
	}

	out := make(map[string]ProviderInfo)
	for p, envKeys := range envMap {
		out[p] = ProviderInfo{
			EnvKeys:    envKeys,
			Configured: keys[p] != "",
			ModelCount: counts[p],
		}
	}
	return out, nil
}

// ProviderInfo describes a provider's status.
type ProviderInfo struct {
	EnvKeys    []string
	Configured bool
	ModelCount int
}

// ModelsOpts configures Models.
type ModelsOpts struct {
	Provider         string
	SkipLive         bool
	Timeout          time.Duration
	APIKey           interface{} // string or map[string]string
	Supports         map[string]bool
	InputModalities  map[string]bool
	OutputModalities map[string]bool
}

func resolveAPIKeys(apiKey interface{}, provider string) map[string]string {
	envMap := ProviderEnvKeys()
	resolved := make(map[string]string)

	if apiKey != nil {
		switch v := apiKey.(type) {
		case string:
			if provider != "" {
				resolved[provider] = v
			} else {
				for p := range envMap {
					resolved[p] = v
				}
			}
		case map[string]string:
			for k, val := range v {
				resolved[k] = val
			}
		}
	}

	for p, vars := range envMap {
		if resolved[p] != "" {
			continue
		}
		for _, v := range vars {
			if val := os.Getenv(v); val != "" {
				resolved[p] = val
				break
			}
		}
	}
	return resolved
}

func fetchLiveModels(provider, apiKey string, timeout time.Duration) ([]ModelSpec, error) {
	client := &http.Client{Timeout: timeout}

	switch provider {
	case "openai":
		return fetchOpenAIModelList(client, apiKey)
	case "anthropic":
		return fetchAnthropicModelList(client, apiKey)
	case "gemini":
		return fetchGeminiModelList(client, apiKey)
	}
	return nil, nil
}

func fetchOpenAIModelList(client *http.Client, apiKey string) ([]ModelSpec, error) {
	req, _ := http.NewRequest("GET", "https://api.openai.com/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var data struct {
		Data []map[string]any `json:"data"`
	}
	json.Unmarshal(body, &data)
	var out []ModelSpec
	for _, item := range data.Data {
		id, _ := item["id"].(string)
		if id != "" {
			out = append(out, ModelSpec{ID: id, Provider: "openai", Raw: item})
		}
	}
	return out, nil
}

func fetchAnthropicModelList(client *http.Client, apiKey string) ([]ModelSpec, error) {
	req, _ := http.NewRequest("GET", "https://api.anthropic.com/v1/models", nil)
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var data struct {
		Data []map[string]any `json:"data"`
	}
	json.Unmarshal(body, &data)
	var out []ModelSpec
	for _, item := range data.Data {
		id, _ := item["id"].(string)
		if id != "" {
			out = append(out, ModelSpec{ID: id, Provider: "anthropic", Raw: item})
		}
	}
	return out, nil
}

func fetchGeminiModelList(client *http.Client, apiKey string) ([]ModelSpec, error) {
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models?key=%s", apiKey)
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var data struct {
		Models []map[string]any `json:"models"`
	}
	json.Unmarshal(body, &data)
	var out []ModelSpec
	for _, item := range data.Models {
		name, _ := item["name"].(string)
		if name == "" {
			continue
		}
		id := name
		if len(id) > 7 && id[:7] == "models/" {
			id = id[7:]
		}
		spec := ModelSpec{ID: id, Provider: "gemini", Raw: item}
		if v, ok := item["inputTokenLimit"].(float64); ok {
			i := int(v)
			spec.ContextWindow = &i
		}
		if v, ok := item["outputTokenLimit"].(float64); ok {
			i := int(v)
			spec.MaxOutput = &i
		}
		out = append(out, spec)
	}
	return out, nil
}

func mergeSpecs(primary, fallback []ModelSpec) []ModelSpec {
	merged := make(map[string]ModelSpec)
	for _, s := range primary {
		merged[s.Provider+":"+s.ID] = s
	}
	for _, f := range fallback {
		key := f.Provider + ":" + f.ID
		if existing, ok := merged[key]; ok {
			if existing.ContextWindow == nil && f.ContextWindow != nil {
				existing.ContextWindow = f.ContextWindow
			}
			if existing.MaxOutput == nil && f.MaxOutput != nil {
				existing.MaxOutput = f.MaxOutput
			}
			if len(existing.InputModalities) == 0 {
				existing.InputModalities = f.InputModalities
			}
			if len(existing.OutputModalities) == 0 {
				existing.OutputModalities = f.OutputModalities
			}
			existing.ToolCall = existing.ToolCall || f.ToolCall
			existing.StructuredOutput = existing.StructuredOutput || f.StructuredOutput
			existing.Reasoning = existing.Reasoning || f.Reasoning
			merged[key] = existing
		} else {
			merged[key] = f
		}
	}

	out := make([]ModelSpec, 0, len(merged))
	for _, s := range merged {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func filterSpecs(specs []ModelSpec, opts *ModelsOpts) []ModelSpec {
	if opts == nil {
		return specs
	}
	if opts.Supports == nil && opts.InputModalities == nil && opts.OutputModalities == nil {
		return specs
	}

	var out []ModelSpec
	for _, s := range specs {
		if opts.Supports != nil {
			features := make(map[string]bool)
			if s.ToolCall {
				features["tools"] = true
			}
			if s.StructuredOutput {
				features["json_output"] = true
			}
			if s.Reasoning {
				features["reasoning"] = true
			}
			ok := true
			for f := range opts.Supports {
				if !features[f] {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
		}
		if opts.InputModalities != nil {
			mods := make(map[string]bool)
			for _, m := range s.InputModalities {
				mods[m] = true
			}
			ok := true
			for m := range opts.InputModalities {
				if !mods[m] {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
		}
		if opts.OutputModalities != nil {
			mods := make(map[string]bool)
			for _, m := range s.OutputModalities {
				mods[m] = true
			}
			ok := true
			for m := range opts.OutputModalities {
				if !mods[m] {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
		}
		out = append(out, s)
	}
	return out
}
