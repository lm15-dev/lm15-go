package lm15

import (
	"testing"
	"time"
)

func makeTestResponse() LMResponse {
	return LMResponse{
		ID: "r1", Model: "test",
		Message:      Message{Role: RoleAssistant, Parts: []Part{TextPart("ok")}},
		FinishReason: FinishStop,
		Usage:        Usage{InputTokens: 5, OutputTokens: 2, TotalTokens: 7},
	}
}

func TestMiddlewarePipeline(t *testing.T) {
	pipeline := MiddlewarePipeline{}
	var log []string

	pipeline.Add(func(req LMRequest, next CompleteFn) (LMResponse, error) {
		log = append(log, "before")
		resp, err := next(req)
		log = append(log, "after")
		return resp, err
	})

	fn := pipeline.WrapComplete(func(req LMRequest) (LMResponse, error) {
		log = append(log, "inner")
		return makeTestResponse(), nil
	})

	fn(LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}})
	if len(log) != 3 || log[0] != "before" || log[1] != "inner" || log[2] != "after" {
		t.Errorf("unexpected log: %v", log)
	}
}

func TestWithCache(t *testing.T) {
	cache := make(map[string]LMResponse)
	mw := WithCache(cache)
	callCount := 0

	fn := func(req LMRequest) (LMResponse, error) {
		callCount++
		return makeTestResponse(), nil
	}

	req := LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}}
	mw(req, fn)
	mw(req, fn)
	if callCount != 1 {
		t.Errorf("expected 1 call, got %d", callCount)
	}
}

func TestWithHistory(t *testing.T) {
	var history []MiddlewareHistoryEntry
	mw := WithHistory(&history)

	fn := func(req LMRequest) (LMResponse, error) {
		return makeTestResponse(), nil
	}

	req := LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}}
	mw(req, fn)
	if len(history) != 1 {
		t.Errorf("expected 1 history entry, got %d", len(history))
	}
	if history[0].Model != "test" {
		t.Errorf("expected test, got %s", history[0].Model)
	}
}

func TestWithRetries(t *testing.T) {
	mw := WithRetries(2, 1*time.Millisecond)
	attempts := 0

	resp, err := mw(LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}}, func(req LMRequest) (LMResponse, error) {
		attempts++
		if attempts < 3 {
			return LMResponse{}, &RateLimitError{ProviderError{ULMError{"429"}}}
		}
		return makeTestResponse(), nil
	})

	if err != nil {
		t.Fatal(err)
	}
	if attempts != 3 {
		t.Errorf("expected 3 attempts, got %d", attempts)
	}
	if resp.ID != "r1" {
		t.Errorf("unexpected response: %+v", resp)
	}
}
