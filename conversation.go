package lm15

// Conversation accumulates messages for multi-turn interactions.
type Conversation struct {
	System   string
	messages []Message
}

// NewConversation creates a new conversation.
func NewConversation(system string) *Conversation {
	return &Conversation{System: system}
}

// User adds a user message.
func (c *Conversation) User(content string) {
	c.messages = append(c.messages, UserMessage(content))
}

// UserParts adds a user message with mixed parts.
func (c *Conversation) UserParts(parts ...Part) {
	c.messages = append(c.messages, Message{Role: RoleUser, Parts: parts})
}

// Assistant adds an assistant response.
func (c *Conversation) Assistant(resp LMResponse) {
	c.messages = append(c.messages, resp.Message)
}

// ToolResults adds tool results.
func (c *Conversation) ToolResults(results map[string]string) {
	c.messages = append(c.messages, ToolResultMessage(results))
}

// Prefill adds an assistant prefill message.
func (c *Conversation) Prefill(text string) {
	c.messages = append(c.messages, AssistantMessage(text))
}

// Messages returns a copy of all messages.
func (c *Conversation) Messages() []Message {
	out := make([]Message, len(c.messages))
	copy(out, c.messages)
	return out
}

// Clear removes all messages.
func (c *Conversation) Clear() {
	c.messages = nil
}
