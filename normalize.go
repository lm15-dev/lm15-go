package lm15

// Provider error-body normalization (the normalize_error vet op), per
// spec/vocabularies.md "ErrorCode" and lm15-contract/errors/cases/*.json.
// The Python reference (lm15-python2/lm15/providers/*.py normalize_error)
// was consulted only for body-shape heuristics the spec does not pin.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// errorForCodeMeta builds the most specific canonical error class for a
// canonical ErrorCode, carrying full metadata.
func errorForCodeMeta(code string, meta ErrorMeta) LM15Error {
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

// MapHTTPError maps an HTTP status + message to a typed canonical error,
// per the spec's HTTP mapping table (unmatched status -> ProviderError).
func MapHTTPError(status int, message, provider, providerCode, requestID string) LM15Error {
	meta := ErrorMeta{
		Message:      message,
		Provider:     provider,
		ProviderCode: providerCode,
		Status:       status,
		RequestID:    requestID,
	}
	switch {
	case status == 401 || status == 403:
		return errorForCodeMeta("auth", meta)
	case status == 402:
		return errorForCodeMeta("billing", meta)
	case status == 408 || status == 504:
		return errorForCodeMeta("timeout", meta)
	case status == 429:
		return errorForCodeMeta("rate_limit", meta)
	case status == 400 || status == 404 || status == 409 || status == 413 || status == 422:
		return errorForCodeMeta("invalid_request", meta)
	case status >= 500 && status <= 599:
		return errorForCodeMeta("server", meta)
	default:
		return errorForCodeMeta("provider", meta)
	}
}

func containsAny(lowered string, markers ...string) bool {
	for _, m := range markers {
		if strings.Contains(lowered, m) {
			return true
		}
	}
	return false
}

func isModelErrorMessage(text string) bool {
	lowered := strings.ToLower(text)
	return strings.Contains(lowered, "model") && containsAny(lowered,
		"not found", "does not exist", "not exist", "not supported",
		"unsupported", "not available", "unknown")
}

func anthropicContextLength(msg string) bool {
	lowered := strings.ToLower(msg)
	return containsAny(lowered, "prompt is too long", "too many tokens",
		"context window", "context length") ||
		(strings.Contains(lowered, "token") &&
			containsAny(lowered, "limit", "exceed"))
}

func geminiContextLength(msg string) bool {
	lowered := strings.ToLower(msg)
	return (strings.Contains(lowered, "token") &&
		containsAny(lowered, "limit", "exceed")) ||
		containsAny(lowered, "too long", "context is too long", "context length")
}

// parsedErrorBody is the loosely-decoded provider error envelope.
type parsedErrorBody struct {
	message   string
	code      string // openai "code"
	errType   string // openai/anthropic "type"
	errStatus string // gemini "status"
	requestID string // anthropic top-level "request_id"
	ok        bool   // body parsed as JSON
}

func parseErrorBody(body string) parsedErrorBody {
	var data any
	if err := json.Unmarshal([]byte(body), &data); err != nil {
		return parsedErrorBody{}
	}
	out := parsedErrorBody{ok: true}
	obj, isObj := data.(map[string]any)
	if !isObj {
		return out
	}
	if rid, ok := obj["request_id"].(string); ok {
		out.requestID = rid
	}
	switch errVal := obj["error"].(type) {
	case map[string]any:
		out.message, _ = errVal["message"].(string)
		out.code = stringish(errVal["code"])
		out.errType = stringish(errVal["type"])
		out.errStatus = stringish(errVal["status"])
	case nil:
	default:
		out.message = fmt.Sprint(errVal)
	}
	return out
}

func stringish(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func fallbackMessage(body string, status int) string {
	msg := strings.TrimSpace(body)
	if len(msg) > 500 {
		msg = msg[:500]
	}
	if msg == "" {
		msg = fmt.Sprintf("HTTP %d", status)
	}
	return msg
}

var openaiModelErrorCodes = map[string]bool{
	"model_not_found":     true,
	"model_not_available": true,
	"unsupported_model":   true,
}

// NormalizeError maps a provider HTTP error response to the canonical
// error for the given provider ("openai", "openai_chat", "anthropic",
// "gemini").
func NormalizeError(provider string, status int, body string) (LM15Error, error) {
	switch provider {
	case "openai", "openai_chat":
		return normalizeOpenAIError(provider, status, body), nil
	case "anthropic":
		return normalizeAnthropicError(status, body), nil
	case "gemini":
		return normalizeGeminiError(status, body), nil
	}
	return nil, valueErr("unknown provider: %s", provider)
}

func normalizeOpenAIError(provider string, status int, body string) LM15Error {
	p := parseErrorBody(body)
	if !p.ok {
		return MapHTTPError(status, fallbackMessage(body, status), provider, "", "")
	}
	msg := p.message
	providerCode := p.code
	if providerCode == "" {
		providerCode = p.errType
	}
	meta := ErrorMeta{Message: msg, Provider: provider, ProviderCode: providerCode, Status: status}
	switch {
	case p.code == "context_length_exceeded":
		return errorForCodeMeta("context_length", meta)
	case openaiModelErrorCodes[p.code] ||
		(status == 404 && isModelErrorMessage(msg+" "+p.code+" "+p.errType)):
		return errorForCodeMeta("unsupported_model", meta)
	case p.code == "insufficient_quota" || p.errType == "insufficient_quota":
		return errorForCodeMeta("billing", meta)
	case p.code == "invalid_api_key" || p.errType == "authentication_error":
		return errorForCodeMeta("auth", meta)
	case p.code == "rate_limit_exceeded" || p.errType == "rate_limit_error":
		return errorForCodeMeta("rate_limit", meta)
	}
	if p.code != "" && !strings.Contains(msg, p.code) {
		msg = fmt.Sprintf("%s (%s)", msg, p.code)
	}
	return MapHTTPError(status, msg, provider, providerCode, "")
}

var anthropicErrorTypeCodes = map[string]string{
	"authentication_error":  "auth",
	"permission_error":      "auth",
	"billing_error":         "billing",
	"rate_limit_error":      "rate_limit",
	"request_too_large":     "invalid_request",
	"not_found_error":       "invalid_request",
	"invalid_request_error": "invalid_request",
	"api_error":             "server",
	"overloaded_error":      "server",
	"timeout_error":         "timeout",
}

func normalizeAnthropicError(status int, body string) LM15Error {
	p := parseErrorBody(body)
	if !p.ok {
		return MapHTTPError(status, fallbackMessage(body, status), "anthropic", "", "")
	}
	msg := p.message
	meta := ErrorMeta{Message: msg, Provider: "anthropic", ProviderCode: p.errType, Status: status, RequestID: p.requestID}
	switch {
	case anthropicContextLength(msg):
		return errorForCodeMeta("context_length", meta)
	case p.errType == "not_found_error" && isModelErrorMessage(msg):
		return errorForCodeMeta("unsupported_model", meta)
	}
	if code, ok := anthropicErrorTypeCodes[p.errType]; ok {
		return errorForCodeMeta(code, meta)
	}
	if p.errType != "" && !strings.Contains(msg, p.errType) {
		msg = fmt.Sprintf("%s (%s)", msg, p.errType)
	}
	return MapHTTPError(status, msg, "anthropic", p.errType, p.requestID)
}

var geminiErrorStatusCodes = map[string]string{
	"INVALID_ARGUMENT":    "invalid_request",
	"FAILED_PRECONDITION": "billing",
	"PERMISSION_DENIED":   "auth",
	"UNAUTHENTICATED":     "auth",
	"NOT_FOUND":           "invalid_request",
	"RESOURCE_EXHAUSTED":  "rate_limit",
	"INTERNAL":            "server",
	"UNAVAILABLE":         "server",
	"DEADLINE_EXCEEDED":   "timeout",
}

func normalizeGeminiError(status int, body string) LM15Error {
	p := parseErrorBody(body)
	if !p.ok {
		return MapHTTPError(status, fallbackMessage(body, status), "gemini", "", "")
	}
	msg := p.message
	meta := ErrorMeta{Message: msg, Provider: "gemini", ProviderCode: p.errStatus, Status: status}
	switch {
	case geminiContextLength(msg):
		return errorForCodeMeta("context_length", meta)
	case p.errStatus == "NOT_FOUND" && isModelErrorMessage(msg):
		return errorForCodeMeta("unsupported_model", meta)
	}
	if code, ok := geminiErrorStatusCodes[p.errStatus]; ok {
		return errorForCodeMeta(code, meta)
	}
	if p.errStatus != "" && !strings.Contains(msg, p.errStatus) {
		msg = fmt.Sprintf("%s (%s)", msg, p.errStatus)
	}
	return MapHTTPError(status, msg, "gemini", p.errStatus, "")
}

// NormalizedErrorMap renders an LM15Error in the normalize_error vet
// output shape.
func NormalizedErrorMap(err LM15Error) jmap {
	var providerCode any
	meta := metaOf(err)
	if meta.ProviderCode != "" {
		providerCode = meta.ProviderCode
	}
	return jmap{
		"class":         err.ClassName(),
		"code":          err.Code(),
		"provider_code": providerCode,
		"message":       meta.Message,
	}
}

func metaOf(err LM15Error) ErrorMeta {
	switch e := err.(type) {
	case TransportError:
		return e.ErrorMeta
	case ConfigurationError:
		return e.ErrorMeta
	case NotConfiguredError:
		return e.ErrorMeta
	case CapabilityError:
		return e.ErrorMeta
	case UnsupportedFeatureError:
		return e.ErrorMeta
	case ProviderError:
		return e.ErrorMeta
	case AuthError:
		return e.ErrorMeta
	case BillingError:
		return e.ErrorMeta
	case RateLimitError:
		return e.ErrorMeta
	case InvalidRequestError:
		return e.ErrorMeta
	case ContextLengthError:
		return e.ErrorMeta
	case UnsupportedModelError:
		return e.ErrorMeta
	case TimeoutError:
		return e.ErrorMeta
	case ServerError:
		return e.ErrorMeta
	}
	return ErrorMeta{}
}

// Meta exposes the metadata of any canonical error (used by the shim and
// callers via errors.As targets).
func Meta(err LM15Error) ErrorMeta { return metaOf(err) }
