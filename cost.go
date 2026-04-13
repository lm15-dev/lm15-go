package lm15

import "math"

// CostBreakdown is an itemized cost estimate in US dollars.
type CostBreakdown struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
	Reasoning  float64
	InputAudio float64
	OutputAudio float64
	Total      float64
}

// additiveCacheProviders: input_tokens excludes cached (cache counts are additive).
var additiveCacheProviders = map[string]bool{"anthropic": true}

// separateReasoningProviders: reasoning_tokens is separate from output_tokens.
var separateReasoningProviders = map[string]bool{"gemini": true, "google": true}

func perToken(ratePerMillion float64) float64 {
	return ratePerMillion / 1_000_000
}

func intOrZero(p *int) int {
	if p == nil {
		return 0
	}
	return *p
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
		Reasoning: cReasoning,
		InputAudio: cInputAudio, OutputAudio: cOutputAudio,
		Total: cInput + cOutput + cCacheRead + cCacheWrite + cReasoning + cInputAudio + cOutputAudio,
	}
}
