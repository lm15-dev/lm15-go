package lm15

import (
	"math"
	"testing"
)

func assertClose(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestEstimateCostBasic(t *testing.T) {
	usage := Usage{InputTokens: 1000, OutputTokens: 500, TotalTokens: 1500}
	cost := EstimateCost(usage, map[string]float64{"input": 3.0, "output": 15.0}, "openai")
	if cost.Total <= 0 {
		t.Error("expected positive total")
	}
	assertClose(t, cost.Input, 1000*3.0/1_000_000)
	assertClose(t, cost.Output, 500*15.0/1_000_000)
}

func TestEstimateCostAnthropicCache(t *testing.T) {
	cr := 300
	cw := 100
	usage := Usage{InputTokens: 500, OutputTokens: 200, TotalTokens: 700, CacheReadTokens: &cr, CacheWriteTokens: &cw}
	cost := EstimateCost(usage, map[string]float64{"input": 3.0, "output": 15.0, "cache_read": 1.5, "cache_write": 3.75}, "anthropic")
	assertClose(t, cost.Input, 500*3.0/1_000_000)
	assertClose(t, cost.CacheRead, 300*1.5/1_000_000)
	assertClose(t, cost.CacheWrite, 100*3.75/1_000_000)
}

func TestEstimateCostOpenAIReasoning(t *testing.T) {
	reasoning := 200
	usage := Usage{InputTokens: 100, OutputTokens: 500, TotalTokens: 600, ReasoningTokens: &reasoning}
	cost := EstimateCost(usage, map[string]float64{"input": 3.0, "output": 15.0, "reasoning": 15.0}, "openai")
	// text_output = 500 - 200 = 300
	assertClose(t, cost.Output, 300*15.0/1_000_000)
	assertClose(t, cost.Reasoning, 200*15.0/1_000_000)
}

func TestEstimateCostGeminiReasoning(t *testing.T) {
	reasoning := 200
	usage := Usage{InputTokens: 100, OutputTokens: 300, TotalTokens: 600, ReasoningTokens: &reasoning}
	cost := EstimateCost(usage, map[string]float64{"input": 3.0, "output": 15.0, "reasoning": 15.0}, "gemini")
	// text_output = 300 (not subtracted for gemini)
	assertClose(t, cost.Output, 300*15.0/1_000_000)
	assertClose(t, cost.Reasoning, 200*15.0/1_000_000)
}
