package lm15

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// CostBreakdown is an itemized cost estimate in US dollars.
type CostBreakdown struct {
	Input       float64
	Output      float64
	CacheRead   float64
	CacheWrite  float64
	Reasoning   float64
	InputAudio  float64
	OutputAudio float64
	Total       float64
}

// additiveCacheProviders: input_tokens excludes cached (cache counts are additive).
var additiveCacheProviders = map[string]bool{"anthropic": true}

// separateReasoningProviders: reasoning_tokens is separate from output_tokens.
var separateReasoningProviders = map[string]bool{"gemini": true, "google": true}

var (
	costIndexMu      sync.RWMutex
	costIndex        map[string]ModelSpec
	fetchModelsDevFn = FetchModelsDev
)

func perToken(ratePerMillion float64) float64 {
	return ratePerMillion / 1_000_000
}

func intOrZero(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func zeroCostBreakdown() CostBreakdown {
	return CostBreakdown{}
}

func addCostBreakdown(total *CostBreakdown, cost CostBreakdown) {
	total.Input += cost.Input
	total.Output += cost.Output
	total.CacheRead += cost.CacheRead
	total.CacheWrite += cost.CacheWrite
	total.Reasoning += cost.Reasoning
	total.InputAudio += cost.InputAudio
	total.OutputAudio += cost.OutputAudio
	total.Total += cost.Total
}

// EstimateCost estimates the cost of a request from Usage and pricing rates.
// Rates are in $/million tokens. Provider is required to interpret token semantics correctly.
func EstimateCost(usage Usage, rates map[string]float64, provider string) CostBreakdown {
	rInput := perToken(rates["input"])
	rOutput := perToken(rates["output"])
	rCacheRead := perToken(rates["cache_read"])
	rCacheWrite := perToken(rates["cache_write"])
	rReasoning := perToken(rates["reasoning"])
	rInputAudio := perToken(rates["input_audio"])
	rOutputAudio := perToken(rates["output_audio"])

	cacheRead := intOrZero(usage.CacheReadTokens)
	cacheWrite := intOrZero(usage.CacheWriteTokens)
	reasoning := intOrZero(usage.ReasoningTokens)
	inputAudio := intOrZero(usage.InputAudioTokens)
	outputAudio := intOrZero(usage.OutputAudioTokens)

	var textInput int
	if additiveCacheProviders[provider] {
		textInput = usage.InputTokens - inputAudio
	} else {
		textInput = usage.InputTokens - cacheRead - cacheWrite - inputAudio
	}
	textInput = int(math.Max(float64(textInput), 0))

	var textOutput int
	if separateReasoningProviders[provider] {
		textOutput = usage.OutputTokens - outputAudio
	} else {
		textOutput = usage.OutputTokens - reasoning - outputAudio
	}
	textOutput = int(math.Max(float64(textOutput), 0))

	cInput := float64(textInput) * rInput
	cOutput := float64(textOutput) * rOutput
	cCacheRead := float64(cacheRead) * rCacheRead
	cCacheWrite := float64(cacheWrite) * rCacheWrite
	cReasoning := float64(reasoning) * rReasoning
	cInputAudio := float64(inputAudio) * rInputAudio
	cOutputAudio := float64(outputAudio) * rOutputAudio

	return CostBreakdown{
		Input: cInput, Output: cOutput,
		CacheRead: cCacheRead, CacheWrite: cCacheWrite,
		Reasoning:  cReasoning,
		InputAudio: cInputAudio, OutputAudio: cOutputAudio,
		Total: cInput + cOutput + cCacheRead + cCacheWrite + cReasoning + cInputAudio + cOutputAudio,
	}
}

// EstimateCostFromSpec estimates cost using a models.dev ModelSpec.
func EstimateCostFromSpec(usage Usage, spec ModelSpec) CostBreakdown {
	rates := make(map[string]float64)
	if spec.Raw != nil {
		if rawCost, ok := spec.Raw["cost"].(map[string]any); ok {
			for k, v := range rawCost {
				switch n := v.(type) {
				case float64:
					rates[k] = n
				case int:
					rates[k] = float64(n)
				case int64:
					rates[k] = float64(n)
				}
			}
		}
	}
	return EstimateCost(usage, rates, spec.Provider)
}

// EnableCostTracking fetches pricing from models.dev and enables automatic cost lookup.
func EnableCostTracking(timeout time.Duration) error {
	if timeout == 0 {
		timeout = 20 * time.Second
	}
	specs, err := fetchModelsDevFn(timeout)
	if err != nil {
		return err
	}
	index := make(map[string]ModelSpec)
	for _, spec := range specs {
		if spec.Raw == nil || spec.Raw["cost"] == nil {
			continue
		}
		index[spec.ID] = spec
	}
	costIndexMu.Lock()
	costIndex = index
	costIndexMu.Unlock()
	return nil
}

// DisableCostTracking clears the pricing index.
func DisableCostTracking() {
	costIndexMu.Lock()
	costIndex = nil
	costIndexMu.Unlock()
}

// GetCostIndex returns a copy of the current global pricing index.
func GetCostIndex() map[string]ModelSpec {
	costIndexMu.RLock()
	defer costIndexMu.RUnlock()
	if costIndex == nil {
		return nil
	}
	out := make(map[string]ModelSpec, len(costIndex))
	for k, v := range costIndex {
		out[k] = v
	}
	return out
}

// SetCostIndex installs a pricing index manually.
func SetCostIndex(index map[string]ModelSpec) {
	costIndexMu.Lock()
	defer costIndexMu.Unlock()
	if index == nil {
		costIndex = nil
		return
	}
	costIndex = make(map[string]ModelSpec, len(index))
	for k, v := range index {
		costIndex[k] = v
	}
}

// LookupCost estimates cost using the global pricing index.
func LookupCost(model string, usage Usage) (*CostBreakdown, bool) {
	costIndexMu.RLock()
	spec, ok := costIndex[model]
	costIndexMu.RUnlock()
	if !ok {
		return nil, false
	}
	cost := EstimateCostFromSpec(usage, spec)
	return &cost, true
}

// SumCosts adds multiple cost breakdowns together.
func SumCosts(costs []CostBreakdown) CostBreakdown {
	total := zeroCostBreakdown()
	for _, cost := range costs {
		addCostBreakdown(&total, cost)
	}
	return total
}

// ConfigureOpts configures module-level defaults and optional cost tracking.
type ConfigureOpts struct {
	Env         string
	APIKey      interface{}
	TrackCosts  bool
	CostTimeout time.Duration
}

func configureCostTracking(opts ConfigureOpts) error {
	if opts.TrackCosts {
		if err := EnableCostTracking(opts.CostTimeout); err != nil {
			return fmt.Errorf("enable cost tracking: %w", err)
		}
		return nil
	}
	DisableCostTracking()
	return nil
}
