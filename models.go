package lm15

import "sort"

// PricingTier holds the rates of a call whose prompt is longer than
// AboveInputTokens (spec/types.md § PricingTier). A tier reprices the WHOLE
// call; a rate the tier leaves out is unknown at that tier, never taken from
// the base rates.
type PricingTier struct {
	AboveInputTokens     int
	InputPerMillion      *float64
	OutputPerMillion     *float64
	CacheReadPerMillion  *float64
	CacheWritePerMillion *float64
}

func checkRates(rates map[string]*float64) error {
	for name, v := range rates {
		if v != nil && *v < 0 {
			return valueErrorf("%s must be >= 0", name)
		}
	}
	return nil
}

// Validate checks the threshold and the rates.
func (t PricingTier) Validate() error {
	if t.AboveInputTokens <= 0 {
		return valueErrorf("PricingTier.above_input_tokens must be a positive integer")
	}
	return checkRates(map[string]*float64{
		"input_per_million": t.InputPerMillion, "output_per_million": t.OutputPerMillion,
		"cache_read_per_million": t.CacheReadPerMillion, "cache_write_per_million": t.CacheWritePerMillion,
	})
}

// InferencePricing holds list prices per million tokens of each billing
// bucket (uncached input, cache reads, cache writes, output; advisory
// metadata). An absent rate is unknown, never zero. Tiers (strictly
// ascending, INV-057) reprice long prompts; Dimensions is opaque catalog data
// Estimate does not read.
type InferencePricing struct {
	InputPerMillion      *float64
	OutputPerMillion     *float64
	CacheReadPerMillion  *float64
	CacheWritePerMillion *float64
	Currency             string // "" reads as USD
	Dimensions           JSONObject
	Tiers                []PricingTier
}

// EffectiveCurrency returns Currency or USD.
func (p InferencePricing) EffectiveCurrency() string {
	if p.Currency == "" {
		return "USD"
	}
	return p.Currency
}

// Validate checks the rates and the tiers' order.
func (p InferencePricing) Validate() error {
	if err := checkRates(map[string]*float64{
		"input_per_million": p.InputPerMillion, "output_per_million": p.OutputPerMillion,
		"cache_read_per_million": p.CacheReadPerMillion, "cache_write_per_million": p.CacheWritePerMillion,
	}); err != nil {
		return err
	}
	previous := 0
	for _, tier := range p.Tiers {
		if err := tier.Validate(); err != nil {
			return err
		}
		if tier.AboveInputTokens <= previous {
			return valueErrorf("InferencePricing.tiers must be in strictly ascending above_input_tokens order")
		}
		previous = tier.AboveInputTokens
	}
	return checkJSONObject(p.Dimensions, "dimensions", false)
}

// Estimate is the list-price cost of one call; ok is false when it is not
// known (spec/types.md § InferencePricing, "estimate"). Exact or nothing: the
// sum, in this order, of uncached input, cache reads, cache writes and
// output, each tokens * rate / 1_000_000, at the base rates or the highest
// tier whose threshold InputTokens exceeds. Unknown when input or output is
// unknown; when an unreported cache count's rate differs from the input
// rate; when a bucket with tokens has no rate; when audio tokens were used.
// Service tiers, regional multipliers, longer cache lifetimes and tool fees
// are outside it.
func (p InferencePricing) Estimate(u Usage) (cost float64, ok bool) {
	if u.InputTokens == nil || u.OutputTokens == nil {
		return 0, false
	}
	if intOr0(u.InputAudioTokens) > 0 || intOr0(u.OutputAudioTokens) > 0 {
		return 0, false
	}
	input, output := *u.InputTokens, *u.OutputTokens
	rates := PricingTier{InputPerMillion: p.InputPerMillion, OutputPerMillion: p.OutputPerMillion,
		CacheReadPerMillion: p.CacheReadPerMillion, CacheWritePerMillion: p.CacheWritePerMillion}
	for _, tier := range p.Tiers {
		if input > tier.AboveInputTokens {
			rates = tier
		}
	}
	// An unknown cache count is harmless only when its tokens cost the same
	// as plain input: then the split cannot change the price.
	known := func(count *int, rate *float64) (int, bool) {
		if count != nil {
			return *count, true
		}
		if rate != nil && rates.InputPerMillion != nil && *rate == *rates.InputPerMillion {
			return 0, true
		}
		return 0, false
	}
	read, okRead := known(u.CacheReadTokens, rates.CacheReadPerMillion)
	write, okWrite := known(u.CacheWriteTokens, rates.CacheWritePerMillion)
	if !okRead || !okWrite {
		return 0, false
	}
	buckets := []struct {
		tokens int
		rate   *float64
	}{
		{input - read - write, rates.InputPerMillion},
		{read, rates.CacheReadPerMillion},
		{write, rates.CacheWritePerMillion},
		{output, rates.OutputPerMillion},
	}
	total := 0.0
	for _, b := range buckets {
		if b.tokens == 0 {
			continue
		}
		if b.rate == nil {
			return 0, false
		}
		total += float64(b.tokens) * *b.rate / 1_000_000
	}
	return total, true
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
