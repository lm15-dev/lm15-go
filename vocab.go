package lm15

// Closed string vocabularies (lm15-contract/spec/vocabularies.md).
//
// Every value listed here exists; a value not listed does not. Constructors
// and Validate reject unknown values; adapters map unknown provider tokens to
// the closest canonical value and never invent a new one.

// Role of a Message.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
	RoleDeveloper = "developer"
)

// Roles is the Role vocabulary in declaration order.
var Roles = []string{RoleUser, RoleAssistant, RoleTool, RoleDeveloper}

// Part type discriminators.
const (
	PartTypeText       = "text"
	PartTypeImage      = "image"
	PartTypeAudio      = "audio"
	PartTypeVideo      = "video"
	PartTypeDocument   = "document"
	PartTypeBinary     = "binary"
	PartTypeToolCall   = "tool_call"
	PartTypeToolResult = "tool_result"
	PartTypeThinking   = "thinking"
	PartTypeRefusal    = "refusal"
	PartTypeCitation   = "citation"
)

// PartTypes is the PartType vocabulary in declaration order.
var PartTypes = []string{
	PartTypeText, PartTypeImage, PartTypeAudio, PartTypeVideo, PartTypeDocument, PartTypeBinary,
	PartTypeToolCall, PartTypeToolResult, PartTypeThinking, PartTypeRefusal, PartTypeCitation,
}

// StreamablePartTypes / NonStreamablePartTypes partition PartTypes (INV-035).
var (
	StreamablePartTypes    = []string{PartTypeText, PartTypeThinking, PartTypeImage, PartTypeAudio, PartTypeToolCall, PartTypeCitation}
	NonStreamablePartTypes = []string{PartTypeVideo, PartTypeDocument, PartTypeBinary, PartTypeToolResult, PartTypeRefusal}
)

// Delta type discriminators.
const (
	DeltaTypeText         = "text"
	DeltaTypeThinking     = "thinking"
	DeltaTypeAudio        = "audio"
	DeltaTypeImage        = "image"
	DeltaTypeToolCall     = "tool_call"
	DeltaTypeCitation     = "citation"
	DeltaTypeContinuation = "continuation"
)

// DeltaTypes is the DeltaType vocabulary in declaration order.
var DeltaTypes = []string{DeltaTypeText, DeltaTypeThinking, DeltaTypeAudio, DeltaTypeImage, DeltaTypeToolCall, DeltaTypeCitation, DeltaTypeContinuation}

// FinishReason values.
const (
	FinishStop          = "stop"
	FinishLength        = "length"
	FinishToolCall      = "tool_call"
	FinishContentFilter = "content_filter"
	FinishError         = "error"
)

// FinishReasons is the FinishReason vocabulary.
var FinishReasons = []string{FinishStop, FinishLength, FinishToolCall, FinishContentFilter, FinishError}

// ReasoningEffort values (MAP-7).
const (
	EffortOff     = "off"
	EffortMinimal = "minimal"
	EffortLow     = "low"
	EffortMedium  = "medium"
	EffortHigh    = "high"
	EffortXHigh   = "xhigh"
	EffortMax     = "max"
)

// ReasoningEfforts is the ReasoningEffort vocabulary.
var ReasoningEfforts = []string{EffortOff, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}

// ReasoningSummaries is the ReasoningSummary vocabulary.
var ReasoningSummaries = []string{"auto", "concise", "detailed"}

// ErrorCode values (bidirectional with the error class hierarchy).
const (
	CodeAuth               = "auth"
	CodeBilling            = "billing"
	CodeRateLimit          = "rate_limit"
	CodeInvalidRequest     = "invalid_request"
	CodeContextLength      = "context_length"
	CodeTimeout            = "timeout"
	CodeServer             = "server"
	CodeUnsupportedModel   = "unsupported_model"
	CodeUnsupportedFeature = "unsupported_feature"
	CodeNotConfigured      = "not_configured"
	CodeUnknownModel       = "unknown_model"
	CodeAmbiguousModel     = "ambiguous_model"
	CodeTransport          = "transport"
	CodeLockTimeout        = "lock_timeout"
	CodeStreamAssembly     = "stream_assembly"
	CodeProvider           = "provider"
)

// ErrorCodes is the ErrorCode vocabulary in declaration order.
var ErrorCodes = []string{
	CodeAuth, CodeBilling, CodeRateLimit, CodeInvalidRequest, CodeContextLength, CodeTimeout, CodeServer,
	CodeUnsupportedModel, CodeUnsupportedFeature, CodeNotConfigured, CodeUnknownModel, CodeAmbiguousModel,
	CodeTransport, CodeLockTimeout, CodeStreamAssembly, CodeProvider,
}

// StreamEventTypes is the StreamEventType vocabulary.
var StreamEventTypes = []string{"start", "delta", "end", "error"}

// Batch statuses.
const (
	BatchQueued     = "queued"
	BatchRunning    = "running"
	BatchCancelling = "cancelling"
	BatchCompleted  = "completed"
	BatchFailed     = "failed"
	BatchCancelled  = "cancelled"
	BatchExpired    = "expired"
)

// BatchStatuses is the BatchStatus vocabulary; BatchTerminalStatuses its terminal subset.
var (
	BatchStatuses         = []string{BatchQueued, BatchRunning, BatchCancelling, BatchCompleted, BatchFailed, BatchCancelled, BatchExpired}
	BatchTerminalStatuses = []string{BatchCompleted, BatchFailed, BatchCancelled, BatchExpired}
)

// BatchOutcomes is the BatchOutcome vocabulary.
var BatchOutcomes = []string{"succeeded", "errored", "cancelled", "expired"}

// VideoStatuses is the VideoStatus vocabulary; VideoTerminalStatuses its terminal subset.
var (
	VideoStatuses         = []string{"queued", "running", "completed", "failed", "cancelled"}
	VideoTerminalStatuses = []string{"completed", "failed", "cancelled"}
)

// FileReadinessValues is the FileReadiness vocabulary.
var FileReadinessValues = []string{"pending", "ready", "failed"}

// AudioEncodings is the AudioEncoding vocabulary.
var AudioEncodings = []string{"pcm16", "opus", "mp3", "aac"}

// ToolChoiceModes is the ToolChoiceMode vocabulary.
var ToolChoiceModes = []string{"auto", "required", "none"}

// Cache vocabularies (MAP-6).
var (
	CacheModes      = []string{"auto", "off"}
	CacheRetentions = []string{"short", "long"}
	CachePrefixes   = []string{"stable", "history"}
)

// Live event vocabularies.
var (
	LiveClientEventTypes = []string{"turn", "audio", "image", "text", "tool_result", "interrupt", "end_audio"}
	LiveServerEventTypes = []string{"audio", "text", "tool_call", "tool_call_delta", "interrupted", "turn_end", "usage", "error"}
)

// Auth vocabularies (spec/auth.md, spec/vocabularies.md, 2026-09-03).
var (
	AuthSchemes        = []string{"bearer", "x-api-key", "api-key", "query-key", "sigv4"}
	CredentialKinds    = []string{"api_key", "bearer_token", "aws"}
	CredentialPolicies = []string{"key", "oauth", "oauth-unless-explicit", "aws-chain", "azure-chain", "gcp-chain"}
	RungKinds          = []string{"env", "ini-profile", "json-file", "subprocess", "http-metadata", "http-token-exchange", "sigv4-sts", "unsigned-sts", "jwt-rs256", "file-cache"}
	AuthStepStates     = []string{"selected", "shadowed", "absent", "unprobed"}
	StreamFramings     = []string{"sse", "aws-event-stream"}
	ModelPlacements    = []string{"body", "path"}
)

// Vocabularies maps every vocabulary name to its values (the runtime mirror
// surface_dump reflects). Names follow spec/vocabularies.md.
var Vocabularies = map[string][]string{
	"Role":                    Roles,
	"PartType":                PartTypes,
	"DeltaType":               DeltaTypes,
	"FinishReason":            FinishReasons,
	"ReasoningEffort":         ReasoningEfforts,
	"ReasoningSummary":        ReasoningSummaries,
	"ErrorCode":               ErrorCodes,
	"StreamEventType":         StreamEventTypes,
	"BatchStatus":             BatchStatuses,
	"BATCH_TERMINAL_STATUSES": BatchTerminalStatuses,
	"BatchOutcome":            BatchOutcomes,
	"VideoStatus":             VideoStatuses,
	"VIDEO_TERMINAL_STATUSES": VideoTerminalStatuses,
	"FileReadiness":           FileReadinessValues,
	"AudioEncoding":           AudioEncodings,
	"ToolChoiceMode":          ToolChoiceModes,
	"CacheMode":               CacheModes,
	"CacheRetention":          CacheRetentions,
	"CachePrefix":             CachePrefixes,
	"LiveClientEventType":     LiveClientEventTypes,
	"LiveServerEventType":     LiveServerEventTypes,
	"AuthScheme":              AuthSchemes,
	"CredentialKind":          CredentialKinds,
	"CredentialPolicy":        CredentialPolicies,
	"RungKind":                RungKinds,
	"AuthStepState":           AuthStepStates,
	"StreamFraming":           StreamFramings,
	"ModelPlacement":          ModelPlacements,
}

func inVocab(value string, vocab []string) bool {
	for _, v := range vocab {
		if v == value {
			return true
		}
	}
	return false
}
