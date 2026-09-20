package lm15

import (
	"context"
	"testing"
)

func lp(token string, score float64) TokenLogprob {
	return TokenLogprob{Token: token, Logprob: score}
}

// MAP-13 client-side stop: the cut lands on the text, whole tokens before
// it keep their scores, a token the cut splits loses its score and marks
// coverage incomplete; unmatched stops change nothing.
func TestApplyClientSideStop(t *testing.T) {
	resp := &Response{Model: "m", FinishReason: FinishLength,
		Message:  Message{Role: RoleAssistant, Parts: []Part{TextPart{Text: "hi there"}, TextPart{Text: "END tail"}}},
		Logprobs: []TokenLogprob{lp("hi", -0.1), lp(" there", -0.2), lp("END", -0.3), lp(" tail", -0.4)},
	}
	cut := ApplyClientSideStop(resp, []string{"END"})
	if got := cut.Message.Parts; len(got) != 2 || got[0].(TextPart).Text != "hi there" || got[1].(TextPart).Text != "" {
		t.Fatalf("parts %+v", got)
	}
	if cut.FinishReason != FinishStop || len(cut.Logprobs) != 2 || cut.LogprobsIncomplete {
		t.Fatalf("scores after an aligned cut: %+v incomplete=%v", cut.Logprobs, cut.LogprobsIncomplete)
	}
	// A cut inside a token: the retained text keeps its prefix, the split
	// token's score is dropped and the flag is set.
	inside := ApplyClientSideStop(resp, []string{"ere"})
	if got := inside.Message.Parts; len(got) != 1 || got[0].(TextPart).Text != "hi th" {
		t.Fatalf("parts %+v", got)
	}
	if len(inside.Logprobs) != 1 || !inside.LogprobsIncomplete {
		t.Fatalf("scores after a split cut: %+v incomplete=%v", inside.Logprobs, inside.LogprobsIncomplete)
	}
	if same := ApplyClientSideStop(resp, []string{"nope"}); same.FinishReason != FinishLength || len(same.Logprobs) != 4 {
		t.Fatal("an unmatched stop must change nothing")
	}
}

func TestTruncateStreamAtStop(t *testing.T) {
	events := []StreamEvent{
		StreamStartEvent{Model: "m"},
		StreamDeltaEvent{Delta: TextDelta{Text: "ab", Logprobs: []TokenLogprob{lp("ab", -1)}}},
		StreamDeltaEvent{Delta: ThinkingDelta{Text: "t", PartIndex: 1}},
		StreamDeltaEvent{Delta: TextDelta{Text: "cSTOPxyz", Logprobs: []TokenLogprob{lp("c", -1), lp("STOP", -1), lp("xyz", -1)}}},
		StreamDeltaEvent{Delta: TextDelta{Text: "never"}},
		StreamEndEvent{FinishReason: FinishStop, Usage: &Usage{InputTokens: I(1)}},
	}
	var out []StreamEvent
	for e, err := range TruncateStreamAtStop(SliceSeq(events), []string{"STOP"}) {
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, e)
	}
	// start, "ab" (held then released whole), thinking, "c" (cut), synthesized end
	if len(out) != 5 {
		t.Fatalf("got %d events: %+v", len(out), out)
	}
	if td := out[3].(StreamDeltaEvent).Delta.(TextDelta); td.Text != "c" || len(td.Logprobs) != 1 || td.LogprobsIncomplete {
		t.Fatalf("cut delta %+v", td)
	}
	if end := out[4].(StreamEndEvent); end.FinishReason != FinishStop || end.Usage != nil {
		t.Fatalf("the synthesized end carries no usage: %+v", end)
	}
	resp, err := MaterializeResponse(TruncateStreamAtStop(SliceSeq(events), []string{"STOP"}), &Request{Model: "m"})
	if err != nil || resp.TextOr("") != "abc" || len(resp.Logprobs) != 2 {
		t.Fatalf("materialized %+v %v", resp, err)
	}
}

// Plan invokes no credential and refuses under "refuse" what a call would.
func TestPlanAndRefuse(t *testing.T) {
	lm, err := NewAnthropicLM(WithAPIKey(CredentialFunc(func(ctx context.Context) (Credential, error) {
		t.Fatal("Plan must not invoke the credential provider")
		return nil, nil
	})))
	if err != nil {
		t.Fatal(err)
	}
	req := &Request{Model: "claude-haiku-4-5", Messages: []Message{UserMessage("hi")}, Config: Config{Seed: I(7), Temperature: F(1.5)}}
	plan, err := lm.Plan(req)
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]string{}
	for _, a := range plan {
		fields[a.Field] = a.Action
	}
	if fields["config.seed"] != AdaptDropped || fields["config.temperature"] != AdaptClamped || fields["config.max_tokens"] != AdaptDefaulted {
		t.Fatalf("plan %+v", plan)
	}
	strict, _ := NewAnthropicLM(WithAPIKey("k"), WithAdaptations(AdaptationsRefuse))
	if _, err := strict.Plan(req); err == nil || AsError(err).Feature != "config.seed" {
		t.Fatalf("refuse policy: %v", err)
	}
	silent, _ := NewAnthropicLM(WithAPIKey("k"), WithAdaptations(AdaptationsSilent))
	if plan, _ := silent.Plan(req); len(plan) != 3 {
		t.Fatalf("plan under silent returns the full record, got %d", len(plan))
	}
}
