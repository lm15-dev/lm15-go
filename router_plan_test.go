package lm15

import (
	"testing"
)

func TestRouterPlanOffline(t *testing.T) {
	router, err := NewRouterWithConfig(RouterConfig{Env: map[string]string{}, Adaptations: "note"})
	if err != nil {
		t.Fatal(err)
	}
	quality, _ := Score("How good?", ScoreLevel{Name: "poor"}, ScoreLevel{Name: "great"})
	format, _ := Judgments("j", true, JudgmentProperty{Name: "quality", Schema: quality}, JudgmentProperty{Name: "ok", Schema: YesNo("Is it ok?")})
	req := &Request{Model: "gemini-2.5-flash", Messages: []Message{UserMessage("note")}, Config: Config{ResponseFormat: format, Probabilities: "if_available", UserID: "u1"}}
	plan, err := router.Plan(req)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, a := range plan {
		got[a.Field] = a.Action
	}
	if got["config.probabilities"] != "dropped" || got["config.user_id"] != "dropped" {
		t.Fatalf("plan %+v", plan)
	}
	// typesafe: the state is the one user part, verbatim
	req2 := &Request{Model: "jev-latest", Messages: []Message{UserMessage("note")}, Config: Config{ResponseFormat: format, Probabilities: "required"}}
	if plan, err := router.Plan(req2); err != nil || len(plan) != 0 {
		t.Fatalf("typesafe plan %+v %v", plan, err)
	}
	// A system prompt on typesafe is refused with the native place named.
	req3 := &Request{Model: "jev-latest", System: System("be brief"), Messages: req2.Messages, Config: req2.Config}
	if _, err := router.Plan(req3); err == nil || AsError(err).Feature != "system" {
		t.Fatalf("expected a refusal naming system, got %v", err)
	}
	// azure: a planning LM renders its URL with placeholders and needs no key.
	req4 := &Request{Model: "azure:gpt-5-mini", Messages: req2.Messages, Config: Config{Seed: I(1)}}
	if _, err := router.Plan(req4); err != nil {
		t.Fatalf("azure plan: %v", err)
	}
}
