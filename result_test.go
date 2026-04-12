package lm15

import "testing"

func makeTestStream(text string) StartStreamFn {
	return func(req LMRequest) (<-chan StreamEvent, error) {
		ch := make(chan StreamEvent, 8)
		go func() {
			defer close(ch)
			idx := 0
			ch <- StreamEvent{Type: "start", ID: "r1", Model: "test"}
			ch <- StreamEvent{Type: "delta", PartIndex: &idx, Delta: &PartDelta{Type: "text", Text: text}}
			usage := Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8}
			ch <- StreamEvent{Type: "end", FinishReason: FinishStop, Usage: &usage}
		}()
		return ch, nil
	}
}

func TestResultText(t *testing.T) {
	r := NewResult(ResultOpts{
		Request:     LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}},
		StartStream: makeTestStream("Hello!"),
	})
	text := r.Text()
	if text != "Hello!" {
		t.Errorf("expected Hello!, got %s", text)
	}
}

func TestResultStream(t *testing.T) {
	r := NewResult(ResultOpts{
		Request:     LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}},
		StartStream: makeTestStream("Hi"),
	})
	var chunks []StreamChunk
	for chunk := range r.Stream() {
		chunks = append(chunks, chunk)
	}
	hasText := false
	hasFinished := false
	for _, c := range chunks {
		if c.Type == "text" {
			hasText = true
		}
		if c.Type == "finished" {
			hasFinished = true
		}
	}
	if !hasText {
		t.Error("expected text chunk")
	}
	if !hasFinished {
		t.Error("expected finished chunk")
	}
}

func TestResultResponse(t *testing.T) {
	r := NewResult(ResultOpts{
		Request:     LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}},
		StartStream: makeTestStream("ok"),
	})
	resp, err := r.Response()
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != FinishStop {
		t.Errorf("expected stop, got %s", resp.FinishReason)
	}
	if resp.Usage.InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", resp.Usage.InputTokens)
	}
}

func TestResultToolAutoExec(t *testing.T) {
	callCount := 0
	toolStream := func(req LMRequest) (<-chan StreamEvent, error) {
		ch := make(chan StreamEvent, 8)
		go func() {
			defer close(ch)
			callCount++
			if callCount == 1 {
				// First call: return a tool call
				idx := 0
				ch <- StreamEvent{Type: "start", ID: "r1", Model: "test"}
				ch <- StreamEvent{
					Type: "delta", PartIndex: &idx,
					DeltaRaw: map[string]any{"type": "tool_call", "id": "c1", "name": "greet", "input": "{}"},
				}
				ch <- StreamEvent{Type: "end", FinishReason: FinishToolCall, Usage: &Usage{}}
			} else {
				// Second call: return text
				idx := 0
				ch <- StreamEvent{Type: "start", ID: "r2", Model: "test"}
				ch <- StreamEvent{Type: "delta", PartIndex: &idx, Delta: &PartDelta{Type: "text", Text: "done"}}
				ch <- StreamEvent{Type: "end", FinishReason: FinishStop, Usage: &Usage{}}
			}
		}()
		return ch, nil
	}

	r := NewResult(ResultOpts{
		Request: LMRequest{
			Model:    "test",
			Messages: []Message{UserMessage("hi")},
			Tools:    []Tool{{Type: "function", Name: "greet"}},
		},
		StartStream: toolStream,
		CallableRegistry: map[string]func(map[string]any) (any, error){
			"greet": func(args map[string]any) (any, error) { return "Hello!", nil },
		},
		MaxToolRounds: 2,
	})

	text := r.Text()
	if text != "done" {
		t.Errorf("expected done, got %s", text)
	}
	if callCount != 2 {
		t.Errorf("expected 2 stream calls, got %d", callCount)
	}
}

func TestResultJSON(t *testing.T) {
	r := NewResult(ResultOpts{
		Request:     LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}},
		StartStream: makeTestStream(`{"name": "Alice", "age": 30}`),
	})
	resp, _ := r.Response()
	var data struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	if err := resp.JSON(&data); err != nil {
		t.Fatal(err)
	}
	if data.Name != "Alice" || data.Age != 30 {
		t.Errorf("unexpected: %+v", data)
	}
}

func TestResultOnFinished(t *testing.T) {
	called := false
	r := NewResult(ResultOpts{
		Request:     LMRequest{Model: "test", Messages: []Message{UserMessage("hi")}},
		StartStream: makeTestStream("ok"),
		OnFinished:  func(req LMRequest, resp LMResponse) { called = true },
	})
	r.Text()
	if !called {
		t.Error("OnFinished not called")
	}
}
