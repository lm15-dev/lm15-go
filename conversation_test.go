package lm15

import "testing"

func TestConversation(t *testing.T) {
	conv := NewConversation("test system")
	conv.User("hello")
	conv.User("world")

	if len(conv.Messages()) != 2 {
		t.Errorf("expected 2 messages, got %d", len(conv.Messages()))
	}
	if conv.System != "test system" {
		t.Errorf("expected test system, got %s", conv.System)
	}
}

func TestConversationAssistant(t *testing.T) {
	conv := NewConversation("")
	conv.User("hi")
	conv.Assistant(LMResponse{
		Message: Message{Role: RoleAssistant, Parts: []Part{TextPart("hello")}},
	})

	msgs := conv.Messages()
	if len(msgs) != 2 || msgs[1].Role != RoleAssistant {
		t.Errorf("unexpected: %+v", msgs)
	}
}

func TestConversationClear(t *testing.T) {
	conv := NewConversation("")
	conv.User("hi")
	conv.Clear()
	if len(conv.Messages()) != 0 {
		t.Error("expected empty after clear")
	}
}

func TestConversationPrefill(t *testing.T) {
	conv := NewConversation("")
	conv.User("Output JSON.")
	conv.Prefill("{")
	msgs := conv.Messages()
	if len(msgs) != 2 || msgs[1].Role != RoleAssistant {
		t.Errorf("unexpected: %+v", msgs)
	}
}
