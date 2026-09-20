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
	PartTypeData       = "data"
)

// PartTypes is the PartType vocabulary in declaration order.
var PartTypes = []string{
	PartTypeText, PartTypeImage, PartTypeAudio, PartTypeVideo, PartTypeDocument, PartTypeBinary,
	PartTypeToolCall, PartTypeToolResult, PartTypeThinking, PartTypeRefusal, PartTypeCitation, PartTypeData,
}

// StreamablePartTypes / NonStreamablePartTypes partition PartTypes (INV-035).
var (
	StreamablePartTypes    = []string{PartTypeText, PartTypeThinking, PartTypeImage, PartTypeAudio, PartTypeToolCall, PartTypeCitation}
	NonStreamablePartTypes = []string{PartTypeVideo, PartTypeDocument, PartTypeBinary, PartTypeToolResult, PartTypeRefusal, PartTypeData}
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
	CodeCollectionLimit    = "collection_limit"
	CodeProvider           = "provider"
)

// ErrorCodes is the ErrorCode vocabulary in declaration order.
var ErrorCodes = []string{
	CodeAuth, CodeBilling, CodeRateLimit, CodeInvalidRequest, CodeContextLength, CodeTimeout, CodeServer,
	CodeUnsupportedModel, CodeUnsupportedFeature, CodeNotConfigured, CodeUnknownModel, CodeAmbiguousModel,
	CodeTransport, CodeLockTimeout, CodeStreamAssembly, CodeCollectionLimit, CodeProvider,
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

// AdaptationAction values (MAP-13, 2026-09-14): what the wire got that
// differs from what was asked.
const (
	AdaptDropped     = "dropped"
	AdaptClamped     = "clamped"
	AdaptSubstituted = "substituted"
	AdaptClientSide  = "client_side"
	AdaptSatisfied   = "satisfied"
	AdaptDefaulted   = "defaulted"
)

// AdaptationActions is the AdaptationAction vocabulary.
var AdaptationActions = []string{AdaptDropped, AdaptClamped, AdaptSubstituted, AdaptClientSide, AdaptSatisfied, AdaptDefaulted}

// AdaptationPolicy values: the value of WithAdaptations / RouterConfig.Adaptations.
const (
	AdaptationsNote   = "note"
	AdaptationsSilent = "silent"
	AdaptationsRefuse = "refuse"
)

// AdaptationPolicies is the AdaptationPolicy vocabulary.
var AdaptationPolicies = []string{AdaptationsNote, AdaptationsSilent, AdaptationsRefuse}

// ProbabilityPolicy values (Config.Probabilities; MAP-14, 2026-09-17).
const (
	ProbabilitiesOff         = "off"
	ProbabilitiesIfAvailable = "if_available"
	ProbabilitiesRequired    = "required"
)

// ProbabilityPolicies is the ProbabilityPolicy vocabulary.
var ProbabilityPolicies = []string{ProbabilitiesOff, ProbabilitiesIfAvailable, ProbabilitiesRequired}

// JudgmentMethod values: how a DataPart's distribution was measured.
const (
	MethodProviderClassification      = "provider_classification"
	MethodCandidateSequenceLikelihood = "candidate_sequence_likelihood"
)

// JudgmentMethods is the JudgmentMethod vocabulary.
var JudgmentMethods = []string{MethodProviderClassification, MethodCandidateSequenceLikelihood}

// NamedCredential values (AUTH-1, 2026-09-19): one identity on a cloud
// door instead of its chain; the same four words on every cloud.
const (
	CredentialPlatform    = "platform"
	CredentialWorkload    = "workload"
	CredentialEnvironment = "environment"
	CredentialCLI         = "cli"
)

// NamedCredentials is the NamedCredential vocabulary.
var NamedCredentials = []string{CredentialPlatform, CredentialWorkload, CredentialEnvironment, CredentialCLI}

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
	"AdaptationAction":        AdaptationActions,
	"AdaptationPolicy":        AdaptationPolicies,
	"ProbabilityPolicy":       ProbabilityPolicies,
	"JudgmentMethod":          JudgmentMethods,
	"NamedCredential":         NamedCredentials,
}

func inVocab(value string, vocab []string) bool {
	for _, v := range vocab {
		if v == value {
			return true
		}
	}
	return false
}
