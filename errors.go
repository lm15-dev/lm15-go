package lm15

import "fmt"

// ULMError is the base error type for all lm15 errors.
type ULMError struct {
	Msg string
}

func (e *ULMError) Error() string { return e.Msg }

// TransportError indicates an HTTP/network transport failure.
type TransportError struct{ ULMError }

// ProviderError is the base for all provider-specific errors.
type ProviderError struct{ ULMError }

// AuthError indicates authentication failure (401/403).
type AuthError struct{ ProviderError }

// BillingError indicates a billing/payment issue (402).
type BillingError struct{ ProviderError }

// RateLimitError indicates rate limiting (429).
type RateLimitError struct{ ProviderError }

// InvalidRequestError indicates a bad request (400/404/422).
type InvalidRequestError struct{ ProviderError }

// ContextLengthError indicates the input exceeds the context window.
type ContextLengthError struct{ InvalidRequestError }

// TimeoutError indicates a request timeout (408/504).
type TimeoutError struct{ ProviderError }

// ServerError indicates a provider server error (5xx).
type ServerError struct{ ProviderError }

// UnsupportedModelError indicates the model is not recognized.
type UnsupportedModelError struct{ ProviderError }

// UnsupportedFeatureError indicates a feature is not supported.
type UnsupportedFeatureError struct{ ProviderError }

// MapHTTPError maps an HTTP status code to a typed error.
func MapHTTPError(status int, message string) error {
	msg := message
	if msg == "" {
		msg = fmt.Sprintf("HTTP %d", status)
	}
	switch {
	case status == 401 || status == 403:
		return &AuthError{ProviderError{ULMError{msg}}}
	case status == 402:
		return &BillingError{ProviderError{ULMError{msg}}}
	case status == 429:
		return &RateLimitError{ProviderError{ULMError{msg}}}
	case status == 408 || status == 504:
		return &TimeoutError{ProviderError{ULMError{msg}}}
	case status == 400 || status == 404 || status == 409 || status == 413 || status == 422:
		return &InvalidRequestError{ProviderError{ULMError{msg}}}
	case status >= 500 && status <= 599:
		return &ServerError{ProviderError{ULMError{msg}}}
	default:
		return &ProviderError{ULMError{msg}}
	}
}

// CanonicalErrorCode returns the lm15 error code for a given error.
func CanonicalErrorCode(err error) string {
	switch err.(type) {
	case *ContextLengthError:
		return "context_length"
	case *AuthError:
		return "auth"
	case *BillingError:
		return "billing"
	case *RateLimitError:
		return "rate_limit"
	case *InvalidRequestError:
		return "invalid_request"
	case *TimeoutError:
		return "timeout"
	case *ServerError:
		return "server"
	default:
		return "provider"
	}
}

// ErrorForCode constructs a typed error from a canonical code string.
func ErrorForCode(code, message string) error {
	switch code {
	case "auth":
		return &AuthError{ProviderError{ULMError{message}}}
	case "billing":
		return &BillingError{ProviderError{ULMError{message}}}
	case "rate_limit":
		return &RateLimitError{ProviderError{ULMError{message}}}
	case "invalid_request":
		return &InvalidRequestError{ProviderError{ULMError{message}}}
	case "context_length":
		return &ContextLengthError{InvalidRequestError{ProviderError{ULMError{message}}}}
	case "timeout":
		return &TimeoutError{ProviderError{ULMError{message}}}
	case "server":
		return &ServerError{ProviderError{ULMError{message}}}
	default:
		return &ProviderError{ULMError{message}}
	}
}

// IsTransient returns true for errors that are worth retrying.
func IsTransient(err error) bool {
	switch err.(type) {
	case *RateLimitError, *TimeoutError, *ServerError, *TransportError:
		return true
	default:
		return false
	}
}
