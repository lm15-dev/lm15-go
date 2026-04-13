package lm15

import "testing"

func echoStream(req LMRequest) (<-chan StreamEvent, error) {
	text := ""
	for _, m := range req.Messages {
		for _, p := range m.Parts {
			if p.Type == PartText {
				if text != "" {
					text += " | "
				}
				text += p.Text
			}
		}
	}
	ch := make(chan StreamEvent, 8)
	go func() {
		defer close(ch)
		idx := 0
		ch <- StreamEvent{Type: "start", ID: "echo-1", Model: req.Model}
		ch <- StreamEvent{Type: "delta", PartIndex: &idx, Delta: &PartDelta{Type: "text", Text: "Echo: " + text}}
		usage := Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}
		ch <- StreamEvent{Type: "end", FinishReason: FinishStop, Usage: &usage}
	}()
	return ch, nil
}

// echoLM creates a UniversalLM with a mock echo adapter for tests.
func echoLM() *UniversalLM {
	lm := NewUniversalLM()
	lm.Register(&echoTestAdapter{})
	return lm
}

type echoTestAdapter struct{ BaseAdapter }

func (a *echoTestAdapter) ProviderName() string { return "echo" }
func (a *echoTestAdapter) Manifest() ProviderManifest {
	return ProviderManifest{Provider: "echo", Supports: EndpointSupport{Complete: true, Stream: true}, EnvKeys: []string{}}
}
func (a *echoTestAdapter) BuildRequest(req LMRequest, stream bool) HTTPRequest   { return HTTPRequest{} }
func (a *echoTestAdapter) ParseResponse(req LMRequest, resp HTTPResponse) (LMResponse, error) {
	return LMResponse{}, nil
}
func (a *echoTestAdapter) ParseStreamEvent(req LMRequest, raw SSEEvent) (*StreamEvent, error) {
	return nil, nil
}
func (a *echoTestAdapter) NormalizeError(status int, body string) error {
	return MapHTTPError(status, body)
}

func TestModelCall(t *testing.T) {
	_ = NewModel(ModelOpts{LM: echoLM(), Model: "echo-1", Provider: "echo"})
	// Use the direct Result approach to test without a real transport
	r := NewResult(ResultOpts{
		Request:     LMRequest{Model: "echo-1", Messages: []Message{UserMessage("hello")}},
		StartStream: echoStream,
	})
	text := r.Text()
	if text != "Echo: hello" {
		t.Errorf("expected 'Echo: hello', got %q", text)
	}
}

func TestModelHistory(t *testing.T) {
	m := NewModel(ModelOpts{LM: echoLM(), Model: "echo-1", Provider: "echo"})
	// Simulate history
	m.history = append(m.history, HistoryEntry{
		Request:  LMRequest{Model: "echo-1", Messages: []Message{UserMessage("first")}},
		Response: LMResponse{Message: Message{Role: RoleAssistant, Parts: []Part{TextPart("ok")}}},
	})
	m.history = append(m.history, HistoryEntry{
		Request:  LMRequest{Model: "echo-1", Messages: []Message{UserMessage("second")}},
		Response: LMResponse{Message: Message{Role: RoleAssistant, Parts: []Part{TextPart("ok")}}},
	})
	if len(m.History()) != 2 {
		t.Errorf("expected 2 history entries, got %d", len(m.History()))
	}
}

func TestModelClearHistory(t *testing.T) {
	m := NewModel(ModelOpts{LM: echoLM(), Model: "echo-1", Provider: "echo"})
	m.history = append(m.history, HistoryEntry{})
	m.ClearHistory()
	if len(m.History()) != 0 {
		t.Error("expected empty history after clear")
	}
}

func TestModelCopy(t *testing.T) {
	m := NewModel(ModelOpts{LM: echoLM(), Model: "echo-1", System: "original", Provider: "echo"})
	copy := m.Copy(&ModelOpts{System: "override"})
	if copy.opts.System != "override" {
		t.Errorf("expected override, got %s", copy.opts.System)
	}
	if m.opts.System != "original" {
		t.Errorf("original changed: %s", m.opts.System)
	}
}

func TestModelPrepare(t *testing.T) {
	m := NewModel(ModelOpts{LM: echoLM(), Model: "echo-1", System: "test system", Provider: "echo"})
	req := m.Prepare("hello", nil)
	if req.Model != "echo-1" {
		t.Errorf("expected echo-1, got %s", req.Model)
	}
	if req.System != "test system" {
		t.Errorf("expected test system, got %s", req.System)
	}
	if len(req.Messages) != 1 || req.Messages[0].Parts[0].Text != "hello" {
		t.Errorf("unexpected messages: %+v", req.Messages)
	}
}
