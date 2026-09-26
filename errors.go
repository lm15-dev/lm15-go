package lm15

import (
	"errors"
	"fmt"
	"strings"
)

// ErrorKind is the canonical error class (spec/vocabularies.md ErrorCode).
// The class hierarchy is replicated as data: IsA walks it.
//
//	LM15Error
//	├── TransportError
//	├── LockTimeoutError
//	├── StreamAssemblyError
//	├── CollectionLimitError
//	├── ConfigurationError
//	│   ├── NotConfiguredError
//	│   ├── UnknownModelError
//	│   └── AmbiguousModelError
//	├── CapabilityError
//	│   └── UnsupportedFeatureError
//	└── ProviderError
//	    ├── AuthError
//	    ├── BillingError
//	    ├── RateLimitError
//	    ├── InvalidRequestError
//	    │   ├── ContextLengthError
//	    │   └── UnsupportedModelError
//	    ├── TimeoutError
//	    └── ServerError
type ErrorKind string

const (
	KindLM15Error          ErrorKind = "LM15Error"
	KindTransport          ErrorKind = "TransportError"
	KindLockTimeout        ErrorKind = "LockTimeoutError"
	KindStreamAssembly     ErrorKind = "StreamAssemblyError"
	KindCollectionLimit    ErrorKind = "CollectionLimitError"
	KindConfiguration      ErrorKind = "ConfigurationError"
	KindNotConfigured      ErrorKind = "NotConfiguredError"
	KindUnknownModel       ErrorKind = "UnknownModelError"
	KindAmbiguousModel     ErrorKind = "AmbiguousModelError"
	KindCapability         ErrorKind = "CapabilityError"
	KindUnsupportedFeature ErrorKind = "UnsupportedFeatureError"
	KindProvider           ErrorKind = "ProviderError"
	KindAuth               ErrorKind = "AuthError"
	KindBilling            ErrorKind = "BillingError"
	KindRateLimit          ErrorKind = "RateLimitError"
	KindInvalidRequest     ErrorKind = "InvalidRequestError"
	KindContextLength      ErrorKind = "ContextLengthError"
	KindUnsupportedModel   ErrorKind = "UnsupportedModelError"
	KindTimeout            ErrorKind = "TimeoutError"
	KindServer             ErrorKind = "ServerError"
	KindMissingCredential  ErrorKind = "MissingCredentialError" // router subclass of NotConfiguredError
	KindCredentialLockWait ErrorKind = "CredentialLockTimeout"  // auth subclass of LockTimeoutError
	KindDeviceCodeExpired  ErrorKind = "DeviceCodeExpiredError" // authkit subclass of AuthError
	// KindAuthOperation: a managed-auth lifecycle operation failed locally
	// (spec/auth-managed.md AUTH-24). Root-level, never retried.
	KindAuthOperation ErrorKind = "AuthOperationError"
)

var errorParent = map[ErrorKind]ErrorKind{
	KindTransport:          KindLM15Error,
	KindLockTimeout:        KindLM15Error,
	KindStreamAssembly:     KindLM15Error,
	KindCollectionLimit:    KindLM15Error,
	KindConfiguration:      KindLM15Error,
	KindNotConfigured:      KindConfiguration,
	KindUnknownModel:       KindConfiguration,
	KindAmbiguousModel:     KindConfiguration,
	KindCapability:         KindLM15Error,
	KindUnsupportedFeature: KindCapability,
	KindProvider:           KindLM15Error,
	KindAuth:               KindProvider,
	KindBilling:            KindProvider,
	KindRateLimit:          KindProvider,
	KindInvalidRequest:     KindProvider,
	KindContextLength:      KindInvalidRequest,
	KindUnsupportedModel:   KindInvalidRequest,
	KindTimeout:            KindProvider,
	KindServer:             KindProvider,
	KindMissingCredential:  KindNotConfigured,
	KindCredentialLockWait: KindLockTimeout,
	KindDeviceCodeExpired:  KindAuth,
	KindAuthOperation:      KindLM15Error,
}

// most-specific-class-first, like the reference's _CLASS_TO_CODE.
var kindDefaultCode = map[ErrorKind]string{
	KindContextLength:      CodeContextLength,
	KindUnsupportedModel:   CodeUnsupportedModel,
	KindAuth:               CodeAuth,
	KindBilling:            CodeBilling,
	KindRateLimit:          CodeRateLimit,
	KindInvalidRequest:     CodeInvalidRequest,
	KindTimeout:            CodeTimeout,
	KindServer:             CodeServer,
	KindUnsupportedFeature: CodeUnsupportedFeature,
	KindCapability:         CodeUnsupportedFeature,
	KindNotConfigured:      CodeNotConfigured,
	KindConfiguration:      CodeNotConfigured,
	KindMissingCredential:  CodeNotConfigured,
	KindUnknownModel:       CodeUnknownModel,
	KindAmbiguousModel:     CodeAmbiguousModel,
	KindTransport:          CodeTransport,
	KindLockTimeout:        CodeLockTimeout,
	KindCredentialLockWait: CodeLockTimeout,
	KindStreamAssembly:     CodeStreamAssembly,
	KindCollectionLimit:    CodeCollectionLimit,
	KindProvider:           CodeProvider,
	KindDeviceCodeExpired:  CodeAuth,
	KindAuthOperation:      CodeAuthOperation,
}

var codeToKind = map[string]ErrorKind{
	CodeContextLength:      KindContextLength,
	CodeUnsupportedModel:   KindUnsupportedModel,
	CodeAuth:               KindAuth,
	CodeBilling:            KindBilling,
	CodeRateLimit:          KindRateLimit,
	CodeInvalidRequest:     KindInvalidRequest,
	CodeTimeout:            KindTimeout,
	CodeServer:             KindServer,
	CodeUnsupportedFeature: KindUnsupportedFeature,
	CodeNotConfigured:      KindNotConfigured,
	CodeUnknownModel:       KindUnknownModel,
	CodeAmbiguousModel:     KindAmbiguousModel,
	CodeTransport:          KindTransport,
	CodeLockTimeout:        KindLockTimeout,
	CodeStreamAssembly:     KindStreamAssembly,
	CodeCollectionLimit:    KindCollectionLimit,
	CodeProvider:           KindProvider,
	CodeAuthOperation:      KindAuthOperation,
}

// IsA reports whether kind is parent or a descendant of it.
func (k ErrorKind) IsA(parent ErrorKind) bool {
	for cur := k; cur != ""; cur = errorParent[cur] {
		if cur == parent {
			return true
		}
	}
	return false
}

// Retryable reports whether the class is in the retryable set
// (RateLimitError, TimeoutError, ServerError, TransportError, LockTimeoutError).
func (k ErrorKind) Retryable() bool {
	return k.IsA(KindRateLimit) || k.IsA(KindTimeout) || k.IsA(KindServer) || k.IsA(KindTransport) || k.IsA(KindLockTimeout)
}

// CanonicalCode is the ErrorCode literal for a class (most-specific first).
func (k ErrorKind) CanonicalCode() string {
	for cur := k; cur != ""; cur = errorParent[cur] {
		if code, ok := kindDefaultCode[cur]; ok {
			return code
		}
	}
	return CodeProvider
}

// ErrorKindForCode returns the class for a canonical code (unknown → ProviderError).
func ErrorKindForCode(code string) ErrorKind {
	if k, ok := codeToKind[code]; ok {
		return k
	}
	return KindProvider
}

// Error is the one lm15 error type. Kind is the canonical class; Code the
// ErrorCode literal. Use errors.As(err, &e) and e.Kind.IsA(...).
type Error struct {
	Kind         ErrorKind
	Code         string
	Message      string
	Provider     string
	ProviderCode string
	Status       int // 0 = none
	RequestID    string
	RetryAfter   *float64

	// RateLimitHeaders is the bounded, immutable snapshot of provider
	// rate-limit evidence (docs/error-diagnostics.md, 2026-09-19): lowercase
	// header name → the values in arrival order, closed allowlist, at most
	// four values of 1–256 printable ASCII characters each. Empty means no
	// retained evidence, not unlimited quota. Never a credential.
	RateLimitHeaders RateLimitHeaders

	// Feature (CapabilityError, MAP-13): the config path the refusal is
	// about — "config.top_k", "messages[0].parts[1]", "tools[name]" — so a
	// policy layer can act without parsing prose. Empty when the refusal
	// is not about one addressable field.
	Feature string

	// Guidance metadata (AuthError / NotConfiguredError).
	EnvKeys        []string
	CredentialHint string
	// CredentialOrigin (AuthError, AUTH-1 provenance): where the rejected
	// credential came from — a label, never the value.
	CredentialOrigin string

	// CollectionLimitError: the breached budget and what was kept.
	Limit         string            // "max_bytes" | "max_events"
	Maximum       int               // the configured budget
	RetainedBytes int               // bytes charged for the accepted events
	PartialEvents []LiveServerEvent // every accepted event, in order
	RejectedEvent LiveServerEvent   // a byte overflow's received-but-not-kept event; nil on a count overflow

	// StreamAssemblyError: what assembled, and the first offending part.
	Partial   *Response
	PartIndex *int

	// UnknownModelError / AmbiguousModelError.
	Model     string
	Providers []string

	// LockTimeoutError.
	Path     string
	LockPath string

	// AuthOperationError (AUTH-24): programs match on Reason; CommitState
	// says whether the store changed; Recovery is guidance for a person,
	// never an instruction to retry. The ids are safe references.
	Reason       string
	Stage        string
	CommitState  string
	Recovery     string
	Operation    string
	ConnectionID string
	AttemptID    string
	MethodID     string

	cause error
}

// Error renders the message with provider / HTTP status / request id context
// for provider errors (the message FIELD is what the contract pins).
func (e *Error) Error() string {
	base := e.Message
	if base == "" {
		base = e.Code
	}
	if !e.Kind.IsA(KindProvider) {
		return base
	}
	var ctx []string
	if e.Provider != "" {
		ctx = append(ctx, e.Provider)
	}
	if e.Status != 0 {
		ctx = append(ctx, fmt.Sprintf("HTTP %d", e.Status))
	}
	if e.RequestID != "" {
		ctx = append(ctx, "request "+e.RequestID)
	}
	suffix := ""
	if len(ctx) > 0 {
		suffix = " (" + strings.Join(ctx, ", ") + ")"
	}
	details := diagnosticsText(e.RateLimitHeaders, e.RetryAfter)
	if head, tail, ok := strings.Cut(base, "\n\n"); ok {
		return head + suffix + details + "\n\n" + tail
	}
	return base + suffix + details
}

// Unwrap exposes the cause (a transport failure, a JSON error).
func (e *Error) Unwrap() error { return e.cause }

// Retryable is the caller's retry-policy datum; lm15 never retries.
func (e *Error) Retryable() bool { return e.Kind.Retryable() }

// Is supports errors.Is(err, &lm15.Error{Kind: KindRateLimit}) by class.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok {
		return false
	}
	if t.Kind != "" && !e.Kind.IsA(t.Kind) {
		return false
	}
	if t.Code != "" && e.Code != t.Code {
		return false
	}
	return true
}

// WithCause attaches a cause.
func (e *Error) WithCause(cause error) *Error {
	e.cause = cause
	return e
}

// newError builds an Error of kind with the class's default code.
func newError(kind ErrorKind, message string) *Error {
	return &Error{Kind: kind, Code: kind.CanonicalCode(), Message: message}
}

// Errorf builds an Error of kind with a formatted message.
func Errorf(kind ErrorKind, format string, args ...any) *Error {
	return newError(kind, fmt.Sprintf(format, args...))
}

// AsError extracts an *Error from err, or nil.
func AsError(err error) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	return nil
}

// IsKind reports whether err is an lm15 Error of class kind (or a subclass).
func IsKind(err error, kind ErrorKind) bool {
	e := AsError(err)
	return e != nil && e.Kind.IsA(kind)
}

const guidanceMarker = "\n\n  To fix:"

func appendGuidance(message, guidance string) string {
	if strings.Contains(message, strings.TrimSpace(guidance)) {
		return message
	}
	return strings.TrimRight(message, " \t\r\n") + guidance
}

// authGuidance is the AuthError constructor's appended guidance.
func authGuidance(provider string, envKeys []string, credentialHint string) string {
	if credentialHint != "" {
		return "\n\n  To fix:\n    - " + credentialHint + "\n"
	}
	g := "\n\n  To fix:\n    - Check that your API key is correct and not expired\n"
	if len(envKeys) > 0 {
		keys := make([]string, 0, len(envKeys))
		for _, k := range envKeys {
			keys = append(keys, k+"=...")
		}
		g += "    - Pass the key explicitly (api_key, or RouterConfig api_keys), or on a host with an environment set " + strings.Join(keys, " or ") + "\n"
	} else {
		g += "    - Pass the key explicitly (api_key, or RouterConfig api_keys)\n"
	}
	if provider != "" {
		g += "    - Verify your " + provider + " account/project has access\n"
	}
	return g
}

func notConfiguredGuidance(provider string, envKeys []string, credentialHint string) string {
	if credentialHint != "" {
		return "\n\n  To fix:\n    - " + credentialHint + "\n"
	}
	if len(envKeys) == 0 && provider == "" {
		return ""
	}
	g := "\n\n  To fix:\n"
	if len(envKeys) > 0 {
		keys := make([]string, 0, len(envKeys))
		for _, k := range envKeys {
			keys = append(keys, k+"=...")
		}
		g += "    - Pass the key explicitly (api_key, or RouterConfig api_keys), or on a host with an environment set " + strings.Join(keys, " or ") + "\n"
	}
	if provider != "" {
		g += "    - Configure credentials for " + provider + "\n"
	}
	return g
}

// AuthErrorf builds an AuthError with the class's guidance appended.
func AuthErrorf(provider string, envKeys []string, credentialHint string, format string, args ...any) *Error {
	e := newError(KindAuth, appendGuidance(fmt.Sprintf(format, args...), authGuidance(provider, envKeys, credentialHint)))
	e.Provider = provider
	e.EnvKeys = envKeys
	e.CredentialHint = credentialHint
	return e
}

// NotConfiguredErrorf builds a NotConfiguredError with guidance appended.
func NotConfiguredErrorf(provider string, envKeys []string, credentialHint string, format string, args ...any) *Error {
	msg := fmt.Sprintf(format, args...)
	if g := notConfiguredGuidance(provider, envKeys, credentialHint); g != "" {
		msg = appendGuidance(msg, g)
	}
	e := newError(KindNotConfigured, msg)
	e.Provider = provider
	e.EnvKeys = envKeys
	e.CredentialHint = credentialHint
	return e
}

// UnsupportedFeatureErrorf builds an UnsupportedFeatureError.
func UnsupportedFeatureErrorf(provider string, format string, args ...any) *Error {
	e := newError(KindUnsupportedFeature, fmt.Sprintf(format, args...))
	e.Provider = provider
	return e
}

// UnsupportedFeature builds an UnsupportedFeatureError about one
// addressable config path (MAP-13 rule 4): feature is "config.top_k",
// "messages[0].parts[1]", "tools[name]", ...
func UnsupportedFeature(provider, feature, format string, args ...any) *Error {
	e := UnsupportedFeatureErrorf(provider, format, args...)
	e.Feature = feature
	return e
}

// WithFeature names the config path a capability refusal is about.
func (e *Error) WithFeature(feature string) *Error {
	e.Feature = feature
	return e
}

const originMarker = "\n\n  credential came from: "

// WithCredentialOrigin names where an AuthError's credential came from
// (AUTH-1 provenance, 2026-09-19): its own line under the provider's
// message, before the guidance, added once. Non-auth errors pass through.
func WithCredentialOrigin(err *Error, origin string) *Error {
	if err == nil || !err.Kind.IsA(KindAuth) || origin == "" {
		return err
	}
	base, _, _ := strings.Cut(err.Message, guidanceMarker)
	if strings.Contains(base, originMarker) {
		return err
	}
	out := AuthErrorf(err.Provider, err.EnvKeys, err.CredentialHint, "%s", strings.TrimRight(base, " \t\r\n")+originMarker+origin)
	out.ProviderCode = err.ProviderCode
	out.Status = err.Status
	out.RequestID = err.RequestID
	out.RetryAfter = err.RetryAfter
	out.RateLimitHeaders = err.RateLimitHeaders
	out.CredentialOrigin = origin
	out.cause = err.cause
	return out
}

// providerErrorf builds an error of a provider class with guidance where the
// class prepends one (RateLimitError, ContextLengthError, AuthError).
func providerErrorf(kind ErrorKind, provider string, envKeys []string, message string) *Error {
	switch {
	case kind.IsA(KindAuth):
		return AuthErrorf(provider, envKeys, "", "%s", message)
	case kind.IsA(KindRateLimit):
		message = appendGuidance(message, "\n\n  To fix:\n    - Wait a moment and retry\n    - Retry with backoff in your application layer (lm15 never retries for you)\n    - Check the reported limits and deployment capacity; a 429 does not prove the endpoint is unsupported\n")
	case kind.IsA(KindContextLength):
		message = appendGuidance(message, "\n\n  To fix:\n    - Reduce the prompt or system prompt length\n    - Clear conversation history\n    - Use a model with a larger context window\n    - Lower max_tokens to leave more room for input\n")
	}
	e := newError(kind, message)
	e.Provider = provider
	return e
}

// WithCredentialHint rewrites an AuthError's guidance for subscription
// (OAuth) adapters; every other error passes through unchanged.
func WithCredentialHint(err *Error, hint string) *Error {
	if err == nil || !err.Kind.IsA(KindAuth) {
		return err
	}
	base, _, _ := strings.Cut(err.Message, guidanceMarker)
	out := AuthErrorf(err.Provider, nil, hint, "%s", base)
	out.ProviderCode = err.ProviderCode
	out.Status = err.Status
	out.RequestID = err.RequestID
	out.RetryAfter = err.RetryAfter
	out.RateLimitHeaders = err.RateLimitHeaders
	out.CredentialOrigin = err.CredentialOrigin
	out.cause = err.cause
	return out
}

// MapHTTPError maps an HTTP status + message to a typed ProviderError
// (the dialects add their own envelope parsing before falling back here).
func MapHTTPError(status int, message, provider string, envKeys []string, providerCode, requestID string, retryAfter *float64) *Error {
	var kind ErrorKind
	switch {
	case status == 401 || status == 403:
		kind = KindAuth
	case status == 402:
		kind = KindBilling
	case status == 408 || status == 504:
		kind = KindTimeout
	case status == 429:
		kind = KindRateLimit
	case status == 400 || status == 404 || status == 409 || status == 413 || status == 422:
		kind = KindInvalidRequest
	case status >= 500 && status <= 599:
		kind = KindServer
	default:
		kind = KindProvider
	}
	e := providerErrorf(kind, provider, envKeys, message)
	e.ProviderCode = providerCode
	e.Status = status
	e.RequestID = requestID
	e.RetryAfter = retryAfter
	return e
}

// ClassName returns the canonical class name the vet protocol reports.
func (e *Error) ClassName() string { return string(e.Kind) }

// ModelNotFoundForm is one pinned MAP-15 form of a provider's "no such model"
// answer: an exact provider code, and the text tests the message must pass.
type ModelNotFoundForm struct {
	Code, Prefix, Contains, Suffix string
}

// ModelNotFoundForms are the pinned forms that carry no model-specific code and
// no not-found class (lm15-contract spec/model-not-found.json, carried verbatim;
// each form has a live receipt).
var ModelNotFoundForms = []ModelNotFoundForm{
	{Code: "not_found_error", Prefix: "model: "},                                    // Anthropic, Claude Code
	{Code: "invalid_request_error", Contains: "The supported API model names are "}, // DeepSeek
	{Code: "1211"},                                                                                        // Z.AI: Unknown Model
	{Code: "1214", Prefix: "modelCode: "},                                                                 // Z.AI: the model field is invalid
	{Code: "400", Suffix: " is not a valid model ID"},                                                     // OpenRouter
	{Code: "invalid-argument", Prefix: "Model not found: "},                                               // xAI (2026-09-01)
	{Code: "validation_error", Contains: "The provided model identifier is invalid"},                      // Bedrock Chat
	{Code: "invalid_request_error", Prefix: "Deployment ", Suffix: " doesn't exist or isn't accessible."}, // Parasail
}

// IsPinnedModelNotFound reports whether an error is one of the pinned MAP-15
// forms: the code matches exactly and the message passes every test given.
func IsPinnedModelNotFound(providerCode, message string) bool {
	if providerCode == "" {
		return false
	}
	for _, f := range ModelNotFoundForms {
		if f.Code == providerCode &&
			strings.HasPrefix(message, f.Prefix) &&
			strings.Contains(message, f.Contains) &&
			strings.HasSuffix(message, f.Suffix) {
			return true
		}
	}
	return false
}
