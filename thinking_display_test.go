package lm15

// MAP-7 rule 7 (amended 2026-11-07, lm15-contract changes/2026-11-07-
// anthropic-thinking-display.md): a summary asked for on Claude is
// thinking.display "summarized"; Opus 4.7, 4.8 and every Claude 5 hide the
// thinking without it. The contract pins the adaptive and budget wires
// (anthropic.reasoning_summary, anthropic.reasoning_summary_budget); these
// pin what its cases do not: no summary, no knob; other servers, no knob.

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func displayThinking(t *testing.T, preset, model string, r Reasoning) any {
	t.Helper()
	opts := []Option{WithAPIKey("k")}
	if preset != "" {
		opts = append(opts, WithCompatPreset(preset))
	}
	lm, err := NewAnthropicLM(opts...)
	if err != nil {
		t.Fatal(err)
	}
	req := &Request{Model: model, Messages: []Message{UserMessage("q")}, Config: Config{Reasoning: &r}}
	treq, err := lm.buildRequest(req, false, newAdaptScope("note", "anthropic", false))
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(treq.Body, &body); err != nil {
		t.Fatal(err)
	}
	return body["thinking"]
}

func TestThinkingDisplayWhenASummaryIsAsked(t *testing.T) {
	cases := []struct {
		model string
		r     Reasoning
		want  map[string]any
	}{
		{"claude-opus-5-5", Reasoning{Effort: "high", Summary: "auto"}, map[string]any{"type": "adaptive", "display": "summarized"}},
		{"claude-opus-5-5", Reasoning{Effort: "high", Summary: "detailed"}, map[string]any{"type": "adaptive", "display": "summarized"}},
		{"claude-haiku-4-5-20251001", Reasoning{Effort: "low", Summary: "auto"}, map[string]any{"type": "enabled", "budget_tokens": float64(2048), "display": "summarized"}},
		{"claude-opus-5-5", Reasoning{Effort: "high"}, map[string]any{"type": "adaptive"}},
	}
	for _, c := range cases {
		if got := displayThinking(t, "", c.model, c.r); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("%s %+v: thinking %v, want %v", c.model, c.r, got, c.want)
		}
	}
}

func TestThinkingDisplayIsAnthropicsOwnKnob(t *testing.T) {
	for preset, model := range map[string]string{"deepseek": "deepseek-v4-flash", "meta": "muse-spark-1.3", "moonshotai": "kimi-k3"} {
		got, _ := json.Marshal(displayThinking(t, preset, model, Reasoning{Effort: "high", Summary: "auto"}))
		if strings.Contains(string(got), "display") {
			t.Fatalf("%s: %s", preset, got)
		}
	}
}
