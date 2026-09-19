package lm15

import "sort"

// InferencePricing is per-million-token pricing (advisory metadata).
type InferencePricing struct {
	InputPerMillion      *float64
	OutputPerMillion     *float64
	CacheReadPerMillion  *float64
	CacheWritePerMillion *float64
	Currency             string // "" reads as USD
	Dimensions           JSONObject
}

// EffectiveCurrency returns Currency or USD.
func (p InferencePricing) EffectiveCurrency() string {
	if p.Currency == "" {
		return "USD"
	}
	return p.Currency
}

// Validate checks the rates.
func (p InferencePricing) Validate() error {
	for name, v := range map[string]*float64{
		"input_per_million": p.InputPerMillion, "output_per_million": p.OutputPerMillion,
		"cache_read_per_million": p.CacheReadPerMillion, "cache_write_per_million": p.CacheWritePerMillion,
	} {
		if v != nil && *v < 0 {
			return valueErrorf("%s must be >= 0", name)
		}
	}
	return checkJSONObject(p.Dimensions, "dimensions", false)
}

// Estimate computes a cost lower bound; nil counts are unknown and skipped.
func (p InferencePricing) Estimate(u Usage) float64 {
	total := 0.0
	add := func(rate *float64, count *int) {
		if rate != nil && count != nil {
			total += float64(*count) * *rate / 1_000_000
		}
	}
	add(p.InputPerMillion, u.InputTokens)
	add(p.OutputPerMillion, u.OutputTokens)
	add(p.CacheReadPerMillion, u.CacheReadTokens)
	add(p.CacheWritePerMillion, u.CacheWriteTokens)
	return total
}

// InferenceModelInfo describes a model's inference capabilities.
type InferenceModelInfo struct {
	InputModalities   []string // nil reads as ["text"]
	OutputModalities  []string // nil reads as ["text"]
	ContextWindow     *int
	MaxOutputTokens   *int
	SupportsReasoning bool
	ReasoningEfforts  []string
	Pricing           *InferencePricing
	Extensions        JSONObject
}

// Validate checks the info.
func (i InferenceModelInfo) Validate() error {
	for _, list := range [][]string{i.InputModalities, i.OutputModalities, i.ReasoningEfforts} {
		for _, v := range list {
			if v == "" {
				return valueErrorf("modalities and reasoning_efforts must contain non-empty strings")
			}
		}
	}
	if i.ContextWindow != nil && *i.ContextWindow <= 0 {
		return valueErrorf("context_window must be a positive integer or None")
	}
	if i.MaxOutputTokens != nil && *i.MaxOutputTokens <= 0 {
		return valueErrorf("max_output_tokens must be a positive integer or None")
	}
	if i.Pricing != nil {
		if err := i.Pricing.Validate(); err != nil {
			return err
		}
	}
	return checkJSONObject(i.Extensions, "extensions", false)
}

// ModelOrigin describes where a model comes from. Type "" reads as "provider".
type ModelOrigin struct {
	Type         string
	ID           string
	BaseModel    string
	ProviderData JSONObject
}

// EffectiveType returns Type or "provider".
func (o ModelOrigin) EffectiveType() string {
	if o.Type == "" {
		return "provider"
	}
	return o.Type
}

// ModelInfo is optional model metadata (listing, catalogs, routing).
type ModelInfo struct {
	ID         string
	Provider   string
	APIFamily  string
	Aliases    []string
	Origin     ModelOrigin
	Inference  *InferenceModelInfo
	Extensions JSONObject
}

// Validate checks the info.
func (m ModelInfo) Validate() error {
	if m.ID == "" {
		return valueErrorf("ModelInfo.id must be a non-empty string")
	}
	if m.Provider == "" {
		return valueErrorf("ModelInfo.provider must be a non-empty string")
	}
	if m.APIFamily == "" {
		return valueErrorf("ModelInfo.api_family must be a non-empty string")
	}
	for _, a := range m.Aliases {
		if a == "" {
			return valueErrorf("ModelInfo.aliases must contain non-empty strings")
		}
	}
	if m.Inference != nil {
		if err := m.Inference.Validate(); err != nil {
			return err
		}
	}
	if err := checkJSONObject(m.Origin.ProviderData, "provider_data", false); err != nil {
		return err
	}
	return checkJSONObject(m.Extensions, "extensions", false)
}

// ModelRegistry is an in-memory catalog keyed by (provider, id).
type ModelRegistry struct {
	order   []modelKey
	models  map[modelKey]ModelInfo
	aliases map[modelKey]modelKey
}

type modelKey struct{ provider, id string }

// NewModelRegistry creates an empty registry.
func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{models: map[modelKey]ModelInfo{}, aliases: map[modelKey]modelKey{}}
}

// Add registers a model; replace=false refuses duplicates.
func (r *ModelRegistry) Add(m ModelInfo, replace bool) error {
	if err := m.Validate(); err != nil {
		return err
	}
	key := modelKey{m.Provider, m.ID}
	if _, exists := r.models[key]; exists {
		if !replace {
			return valueErrorf("model already registered: %s/%s", m.Provider, m.ID)
		}
	} else {
		r.order = append(r.order, key)
	}
	r.models[key] = m
	for _, alias := range m.Aliases {
		r.aliases[modelKey{m.Provider, alias}] = key
	}
	return nil
}

// Get finds a model by provider and id or alias.
func (r *ModelRegistry) Get(provider, model string) (ModelInfo, bool) {
	key := modelKey{provider, model}
	if m, ok := r.models[key]; ok {
		return m, true
	}
	if target, ok := r.aliases[key]; ok {
		m, found := r.models[target]
		return m, found
	}
	return ModelInfo{}, false
}

// Resolve finds a model by id or alias across providers (unique match only).
func (r *ModelRegistry) Resolve(model, provider string) (ModelInfo, bool) {
	if provider != "" {
		return r.Get(provider, model)
	}
	var matches []ModelInfo
	for _, key := range r.order {
		info := r.models[key]
		if info.ID == model || inVocab(model, info.Aliases) {
			matches = append(matches, info)
		}
	}
	if len(matches) == 1 {
		return matches[0], true
	}
	return ModelInfo{}, false
}

// List returns every model (optionally for one provider), in insertion order.
func (r *ModelRegistry) List(provider string) []ModelInfo {
	var out []ModelInfo
	for _, key := range r.order {
		if provider == "" || key.provider == provider {
			out = append(out, r.models[key])
		}
	}
	return out
}

// Providers lists the distinct providers, sorted.
func (r *ModelRegistry) Providers() []string {
	seen := map[string]bool{}
	var out []string
	for _, key := range r.order {
		if !seen[key.provider] {
			seen[key.provider] = true
			out = append(out, key.provider)
		}
	}
	sort.Strings(out)
	return out
}

// ModelRegistryFromDicts builds a registry from canonical ModelInfo dicts.
func ModelRegistryFromDicts(dicts []JSONObject) (*ModelRegistry, error) {
	r := NewModelRegistry()
	for _, d := range dicts {
		m, err := ModelInfoFromDict(d)
		if err != nil {
			return nil, err
		}
		if err := r.Add(m, true); err != nil {
			return nil, err
		}
	}
	return r, nil
}
