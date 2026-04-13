package lm15

import (
	"testing"
	"time"
)

func testModelSpec(id, provider string, rates map[string]any) ModelSpec {
	return ModelSpec{
		ID:       id,
		Provider: provider,
		Raw:      map[string]any{"cost": rates},
	}
}

func TestLookupCostDisabled(t *testing.T) {
	DisableCostTracking()
	if cost, ok := LookupCost("gpt-4.1-mini", Usage{InputTokens: 100, OutputTokens: 50, TotalTokens: 150}); ok || cost != nil {
		t.Fatalf("expected no cost, got %#v, %v", cost, ok)
	}
}

func TestResultCost(t *testing.T) {
	SetCostIndex(map[string]ModelSpec{
		"gpt-4.1-mini": testModelSpec("gpt-4.1-mini", "openai", map[string]any{"input": 3.0, "output": 15.0}),
	})
	defer DisableCostTracking()

	r := NewResult(ResultOpts{
		Request: LMRequest{Model: "gpt-4.1-mini", Messages: []Message{UserMessage("hi")}},
		StartStream: func(req LMRequest) (<-chan StreamEvent, error) {
			ch := make(chan StreamEvent, 8)
			go func() {
				defer close(ch)
				idx := 0
				ch <- StreamEvent{Type: "start", ID: "r1", Model: req.Model}
				ch <- StreamEvent{Type: "delta", PartIndex: &idx, Delta: &PartDelta{Type: "text", Text: "ok"}}
				usage := Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8}
				ch <- StreamEvent{Type: "end", FinishReason: FinishStop, Usage: &usage}
			}()
			return ch, nil
		},
	})
	cost := r.Cost()
	if cost == nil {
		t.Fatal("expected cost")
	}
	assertClose(t, cost.Input, 5*3.0/1_000_000)
	assertClose(t, cost.Output, 3*15.0/1_000_000)
}

func TestModelTotalCost(t *testing.T) {
	SetCostIndex(map[string]ModelSpec{
		"gpt-4.1-mini": testModelSpec("gpt-4.1-mini", "openai", map[string]any{"input": 3.0, "output": 15.0}),
	})
	defer DisableCostTracking()

	m := NewModel(ModelOpts{LM: echoLM(), Model: "gpt-4.1-mini", Provider: "echo"})
	m.history = append(m.history,
		HistoryEntry{Response: LMResponse{Model: "gpt-4.1-mini", Usage: Usage{InputTokens: 1000, OutputTokens: 500, TotalTokens: 1500}}},
		HistoryEntry{Response: LMResponse{Model: "gpt-4.1-mini", Usage: Usage{InputTokens: 1000, OutputTokens: 500, TotalTokens: 1500}}},
	)

	total := m.TotalCost()
	if total == nil {
		t.Fatal("expected total cost")
	}
	assertClose(t, total.Total, 2*((1000*3.0/1_000_000)+(500*15.0/1_000_000)))
}

func TestModelTotalCostEmptyHistory(t *testing.T) {
	SetCostIndex(map[string]ModelSpec{
		"gpt-4.1-mini": testModelSpec("gpt-4.1-mini", "openai", map[string]any{"input": 3.0, "output": 15.0}),
	})
	defer DisableCostTracking()

	m := NewModel(ModelOpts{LM: echoLM(), Model: "gpt-4.1-mini", Provider: "echo"})
	total := m.TotalCost()
	if total == nil {
		t.Fatal("expected zero cost breakdown")
	}
	if total.Total != 0 {
		t.Fatalf("expected zero total, got %v", total.Total)
	}
}

func TestConfigureWithOptionsTrackCosts(t *testing.T) {
	orig := fetchModelsDevFn
	defer func() { fetchModelsDevFn = orig }()
	defer DisableCostTracking()

	fetchModelsDevFn = func(timeout time.Duration) ([]ModelSpec, error) {
		return []ModelSpec{
			testModelSpec("gpt-4.1-mini", "openai", map[string]any{"input": 3.0, "output": 15.0}),
		}, nil
	}

	if err := ConfigureWithOptions(ConfigureOpts{TrackCosts: true}); err != nil {
		t.Fatalf("ConfigureWithOptions error: %v", err)
	}
	index := GetCostIndex()
	if index == nil || index["gpt-4.1-mini"].ID != "gpt-4.1-mini" {
		t.Fatalf("expected hydrated cost index, got %#v", index)
	}

	if err := ConfigureWithOptions(ConfigureOpts{}); err != nil {
		t.Fatalf("ConfigureWithOptions disable error: %v", err)
	}
	if GetCostIndex() != nil {
		t.Fatal("expected cost tracking disabled")
	}
}
