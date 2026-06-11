package lm15

// Canonical error hierarchy per spec/vocabularies.md "ErrorCode". Each
// concrete error is a sentinel type satisfying errors.As, carrying the
// canonical Code() string and the standard metadata fields. RetryAfter is
// float-typed per the Number rule.

// ErrorMeta is the shared metadata of every canonical lm15 error.
type ErrorMeta struct {
	Message      string
	Provider     string
	ProviderCode string
	Status       int
	RequestID    string
	RetryAfter   *float64
}

func (m ErrorMeta) Error() string { return m.Message }

// LM15Error is implemented by every canonical lm15 error type.
type LM15Error interface {
	error
	Code() string
	ClassName() string
	Retryable() bool
}

type TransportError struct{ ErrorMeta }

func (TransportError) Code() string      { return "transport" }
func (TransportError) ClassName() string { return "TransportError" }
func (TransportError) Retryable() bool   { return true }

type ConfigurationError struct{ ErrorMeta }

func (ConfigurationError) Code() string      { return "not_configured" }
func (ConfigurationError) ClassName() string { return "ConfigurationError" }
func (ConfigurationError) Retryable() bool   { return false }

type NotConfiguredError struct{ ConfigurationError }

func (NotConfiguredError) ClassName() string { return "NotConfiguredError" }

type CapabilityError struct{ ErrorMeta }

func (CapabilityError) Code() string      { return "unsupported_feature" }
func (CapabilityError) ClassName() string { return "CapabilityError" }
func (CapabilityError) Retryable() bool   { return false }

type UnsupportedFeatureError struct{ CapabilityError }

func (UnsupportedFeatureError) ClassName() string { return "UnsupportedFeatureError" }

type ProviderError struct{ ErrorMeta }

func (ProviderError) Code() string      { return "provider" }
func (ProviderError) ClassName() string { return "ProviderError" }
func (ProviderError) Retryable() bool   { return false }

type AuthError struct{ ProviderError }

func (AuthError) Code() string      { return "auth" }
func (AuthError) ClassName() string { return "AuthError" }

type BillingError struct{ ProviderError }

func (BillingError) Code() string      { return "billing" }
func (BillingError) ClassName() string { return "BillingError" }

type RateLimitError struct{ ProviderError }

func (RateLimitError) Code() string      { return "rate_limit" }
func (RateLimitError) ClassName() string { return "RateLimitError" }
func (RateLimitError) Retryable() bool   { return true }

type InvalidRequestError struct{ ProviderError }

func (InvalidRequestError) Code() string      { return "invalid_request" }
func (InvalidRequestError) ClassName() string { return "InvalidRequestError" }

type ContextLengthError struct{ InvalidRequestError }

func (ContextLengthError) Code() string      { return "context_length" }
func (ContextLengthError) ClassName() string { return "ContextLengthError" }

type UnsupportedModelError struct{ InvalidRequestError }

func (UnsupportedModelError) Code() string      { return "unsupported_model" }
func (UnsupportedModelError) ClassName() string { return "UnsupportedModelError" }

type TimeoutError struct{ ProviderError }

func (TimeoutError) Code() string      { return "timeout" }
func (TimeoutError) ClassName() string { return "TimeoutError" }
func (TimeoutError) Retryable() bool   { return true }

type ServerError struct{ ProviderError }

func (ServerError) Code() string      { return "server" }
func (ServerError) ClassName() string { return "ServerError" }
func (ServerError) Retryable() bool   { return true }

// ErrorForCode maps a canonical ErrorCode to a fresh error value of the
// most specific canonical class; unknown codes map to ProviderError.
func ErrorForCode(code, message string) LM15Error {
	meta := ErrorMeta{Message: message}
	switch code {
	case "auth":
		return AuthError{ProviderError{meta}}
	case "billing":
		return BillingError{ProviderError{meta}}
	case "rate_limit":
		return RateLimitError{ProviderError{meta}}
	case "invalid_request":
		return InvalidRequestError{ProviderError{meta}}
	case "context_length":
		return ContextLengthError{InvalidRequestError{ProviderError{meta}}}
	case "unsupported_model":
		return UnsupportedModelError{InvalidRequestError{ProviderError{meta}}}
	case "timeout":
		return TimeoutError{ProviderError{meta}}
	case "server":
		return ServerError{ProviderError{meta}}
	case "unsupported_feature":
		return UnsupportedFeatureError{CapabilityError{meta}}
	case "not_configured":
		return NotConfiguredError{ConfigurationError{meta}}
	case "transport":
		return TransportError{meta}
	default:
		return ProviderError{meta}
	}
}

// Unwrap methods expose the class hierarchy to errors.As/errors.Is: each
// subclass unwraps to its embedded parent class.
func (e NotConfiguredError) Unwrap() error      { return e.ConfigurationError }
func (e UnsupportedFeatureError) Unwrap() error { return e.CapabilityError }
func (e AuthError) Unwrap() error               { return e.ProviderError }
func (e BillingError) Unwrap() error            { return e.ProviderError }
func (e RateLimitError) Unwrap() error          { return e.ProviderError }
func (e InvalidRequestError) Unwrap() error     { return e.ProviderError }
func (e ContextLengthError) Unwrap() error      { return e.InvalidRequestError }
func (e UnsupportedModelError) Unwrap() error   { return e.InvalidRequestError }
func (e TimeoutError) Unwrap() error            { return e.ProviderError }
func (e ServerError) Unwrap() error             { return e.ProviderError }
