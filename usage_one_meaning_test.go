package lm15

// MAP-19 (one meaning per usage counter) and InferencePricing.Estimate: the
// Go surface around the contract's mapping/usage.json vectors.

import (
	"strings"
	"testing"
)

func ip(n int) *int         { return &n }
func fp(f float64) *float64 { return &f }

func TestUsagePartsNeverExceedTheirWholes(t *testing.T) {
	for _, u := range []Usage{
		{InputTokens: ip(3), CacheWriteTokens: ip(3275)},
		{InputTokens: ip(10), CacheReadTokens: ip(6), CacheWriteTokens: ip(6)},
		{InputTokens: ip(5), InputAudioTokens: ip(6)},
		{OutputTokens: ip(48), ReasoningTokens: ip(155)},
		{OutputTokens: ip(1), OutputAudioTokens: ip(2)},
		{InputTokens: ip(1), OutputTokens: ip(2), TotalTokens: ip(10)},
	} {
		err := u.Validate()
		if err == nil || !strings.Contains(err.Error(), "MAP-19") {
			t.Fatalf("%+v: want an INV-029 error naming MAP-19, got %v", u, err)
		}
	}
	if err := (Usage{CacheReadTokens: ip(4), ReasoningTokens: ip(7)}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestSettleWithholdsWhatTheNumbersDoNotDetermine(t *testing.T) {
	u := Usage{InputTokens: ip(10), OutputTokens: ip(5), TotalTokens: ip(20)}.Settle()
	if u.InputTokens != nil || u.OutputTokens != nil || *u.TotalTokens != 20 {
		t.Fatalf("%+v", u)
	}
	u = Usage{InputTokens: ip(10), OutputTokens: ip(5), CacheReadTokens: ip(12)}.Settle()
	if u.InputTokens != nil || *u.OutputTokens != 5 {
		t.Fatalf("%+v", u)
	}
}

func TestReasoningUsageSeparate(t *testing.T) {
	body := JSONObject{KV("prompt_tokens", 254), KV("completion_tokens", 9), KV("total_tokens", 374),
		KV("completion_tokens_details", JSONObject{KV("reasoning_tokens", 111)})}
	if u := usageFromChat(body, "separate"); u.OutputTokens == nil || *u.OutputTokens != 120 {
		t.Fatalf("separate: %+v", u)
	}
	if u := usageFromChat(body, "in_completion"); u.OutputTokens != nil || *u.TotalTokens != 374 {
		t.Fatalf("in_completion withholds: %+v", u)
	}
	if got := ResolveOpenAIChatCompat(openaiChatPresets["xai"]).ReasoningUsage; got != "separate" {
		t.Fatalf("xai preset: %q", got)
	}
}

func TestEstimateExactOrUnknown(t *testing.T) {
	sol := InferencePricing{InputPerMillion: fp(2), OutputPerMillion: fp(10), CacheReadPerMillion: fp(0.1), CacheWritePerMillion: fp(2.5),
		Tiers: []PricingTier{{AboveInputTokens: 272000, InputPerMillion: fp(4), OutputPerMillion: fp(15), CacheReadPerMillion: fp(0.2), CacheWritePerMillion: fp(5)}}}
	if cost, ok := sol.Estimate(Usage{InputTokens: ip(100000), OutputTokens: ip(1000), CacheReadTokens: ip(90000), CacheWriteTokens: ip(0)}); !ok || cost != 0.039 {
		t.Fatalf("cached: %v %v", cost, ok)
	}
	if cost, ok := sol.Estimate(Usage{InputTokens: ip(300000), OutputTokens: ip(1000), CacheReadTokens: ip(290000), CacheWriteTokens: ip(0)}); !ok || cost != 0.113 {
		t.Fatalf("tier: %v %v", cost, ok)
	}
	if _, ok := sol.Estimate(Usage{InputTokens: ip(10), OutputTokens: ip(1)}); ok {
		t.Fatal("an unknown cache split must be unknown")
	}
	bad := InferencePricing{Tiers: []PricingTier{{AboveInputTokens: 2}, {AboveInputTokens: 1}}}
	if bad.Validate() == nil {
		t.Fatal("tiers must be ascending")
	}
}
