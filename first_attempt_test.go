package lm15

import (
	"encoding/json"
	"testing"
)

// 2026-10-10: a budget alone fills Effort (MAP-7 rule 3 read the other way);
// a DataPart answer reads through Text (types.md §Response convenience).
func TestBudgetAloneFillsEffort(t *testing.T) {
	want := map[int]string{512: "minimal", 1024: "minimal", 2047: "minimal", 2048: "low", 8192: "medium", 16384: "high", 24576: "xhigh", 32768: "max", 1000000: "max"}
	for b, e := range want {
		if got := EffortForBudget(b); got != e {
			t.Fatalf("budget %d: %s, want %s", b, got, e)
		}
	}
	n := 2000
	cfg := Config{Reasoning: &Reasoning{ThinkingBudget: &n}}
	if err := cfg.Validate(); err != nil || cfg.Reasoning.Effort != "minimal" {
		t.Fatalf("filled %q, err %v", cfg.Reasoning.Effort, err)
	}
	if err := (Config{Reasoning: &Reasoning{Effort: "none"}}).Validate(); err == nil {
		t.Fatal("effort none accepted")
	}
}

func TestDataPartAnswerReadsThroughText(t *testing.T) {
	var v any
	_ = json.Unmarshal([]byte(`{"ok":true}`), &v)
	r := &Response{Message: Message{Role: RoleAssistant, Parts: []Part{DataPart{Value: &v}}}}
	if got := r.Text(); got == nil || *got != `{"ok":true}` {
		t.Fatalf("text %v", got)
	}
}
