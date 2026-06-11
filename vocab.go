package lm15

// Closed vocabularies per lm15-contract/spec/vocabularies.md. A canonical
// value not listed here does not exist; constructors and deserializers
// reject unknown values.

var RoleValues = []string{"user", "assistant", "tool", "developer"}

var PartTypes = []string{
	"text", "image", "audio", "video", "document", "binary",
	"tool_call", "tool_result", "thinking", "refusal", "citation",
}

var DeltaTypes = []string{
	"text", "thinking", "audio", "image", "tool_call", "citation", "continuation",
}

var FinishReasons = []string{"stop", "length", "tool_call", "content_filter", "error"}

var ReasoningEfforts = []string{"off", "adaptive", "minimal", "low", "medium", "high", "xhigh"}

var ReasoningSummaries = []string{"auto", "concise", "detailed"}

var ErrorCodes = []string{
	"auth", "billing", "rate_limit", "invalid_request", "context_length",
	"timeout", "server", "unsupported_model", "unsupported_feature",
	"not_configured", "transport", "provider",
}

var StreamEventTypes = []string{"start", "delta", "end", "error"}

var BatchStatuses = []string{"submitted", "queued", "running", "completed", "failed", "cancelled"}

var AudioEncodings = []string{"pcm16", "opus", "mp3", "aac"}

var ToolChoiceModes = []string{"auto", "required", "none"}

var CacheModes = []string{"auto", "off"}

var CacheRetentions = []string{"short", "long"}

var LiveClientEventTypes = []string{"turn", "audio", "image", "text", "tool_result", "interrupt", "end_audio"}

var LiveServerEventTypes = []string{"audio", "text", "tool_call", "tool_call_delta", "interrupted", "turn_end", "error"}

func inVocab(vocab []string, v string) bool {
	for _, x := range vocab {
		if x == v {
			return true
		}
	}
	return false
}
