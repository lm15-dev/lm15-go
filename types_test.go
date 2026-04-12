package lm15

import (
	"testing"
)

func TestTextPart(t *testing.T) {
	p := TextPart("hello")
	if p.Type != PartText {
		t.Errorf("expected text, got %s", p.Type)
	}
	if p.Text != "hello" {
		t.Errorf("expected hello, got %s", p.Text)
	}
}

func TestThinkingPart(t *testing.T) {
	p := ThinkingPart("hmm")
	if p.Type != PartThinking || p.Text != "hmm" {
		t.Errorf("unexpected: %+v", p)
	}
}

func TestToolCallPart(t *testing.T) {
	p := ToolCallPart("c1", "weather", map[string]any{"city": "Paris"})
	if p.Type != PartToolCall || p.ID != "c1" || p.Name != "weather" {
		t.Errorf("unexpected: %+v", p)
	}
	if p.Input["city"] != "Paris" {
		t.Errorf("expected Paris, got %v", p.Input["city"])
	}
}

func TestImageURL(t *testing.T) {
	p := ImageURL("https://example.com/img.png")
	if p.Type != PartImage || p.Source == nil || p.Source.URL != "https://example.com/img.png" {
		t.Errorf("unexpected: %+v", p)
	}
}

func TestDataSourceBytes(t *testing.T) {
	ds := DataSource{Type: "base64", Data: "AQID", MediaType: "application/octet-stream"}
	b, err := ds.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 3 || b[0] != 1 || b[1] != 2 || b[2] != 3 {
		t.Errorf("unexpected bytes: %v", b)
	}
}

func TestDataSourceBytesError(t *testing.T) {
	ds := DataSource{Type: "url", URL: "https://example.com"}
	_, err := ds.Bytes()
	if err == nil {
		t.Error("expected error for URL data source")
	}
}

func TestUserMessage(t *testing.T) {
	m := UserMessage("hello")
	if m.Role != RoleUser || len(m.Parts) != 1 || m.Parts[0].Text != "hello" {
		t.Errorf("unexpected: %+v", m)
	}
}

func TestLMResponseText(t *testing.T) {
	resp := LMResponse{
		Message: Message{Role: RoleAssistant, Parts: []Part{
			TextPart("Hello"),
			TextPart("World"),
		}},
	}
	if resp.Text() != "Hello\nWorld" {
		t.Errorf("expected Hello\\nWorld, got %s", resp.Text())
	}
}

func TestLMResponseToolCalls(t *testing.T) {
	resp := LMResponse{
		Message: Message{Role: RoleAssistant, Parts: []Part{
			TextPart("thinking..."),
			ToolCallPart("c1", "weather", map[string]any{"city": "Paris"}),
		}},
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "weather" {
		t.Errorf("unexpected tool calls: %+v", calls)
	}
}

func TestLMResponseJSON(t *testing.T) {
	resp := LMResponse{
		Message: Message{Role: RoleAssistant, Parts: []Part{
			TextPart(`{"name": "Alice", "age": 30}`),
		}},
	}
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
