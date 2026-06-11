package lm15

import (
	"errors"
	"testing"
)

func TestNormalizeErrorOpenAI(t *testing.T) {
	body := `{"error":{"message":"Incorrect API key provided","type":"authentication_error","code":"invalid_api_key"}}`
	err, opErr := NormalizeError("openai", 401, body)
	if opErr != nil {
		t.Fatal(opErr)
	}
	var auth AuthError
	if !errors.As(err, &auth) {
		t.Fatalf("want AuthError, got %T", err)
	}
	if err.Code() != "auth" || err.ClassName() != "AuthError" {
		t.Fatalf("bad code/class: %s/%s", err.Code(), err.ClassName())
	}
	if Meta(err).ProviderCode != "invalid_api_key" {
		t.Fatalf("bad provider_code: %q", Meta(err).ProviderCode)
	}
}

func TestNormalizeErrorAnthropicContextLength(t *testing.T) {
	body := `{"error":{"type":"invalid_request_error","message":"prompt is too long: token limit exceeded"},"request_id":"req_ctx"}`
	err, _ := NormalizeError("anthropic", 400, body)
	var ctx ContextLengthError
	if !errors.As(err, &ctx) {
		t.Fatalf("want ContextLengthError, got %T", err)
	}
	if Meta(err).RequestID != "req_ctx" {
		t.Fatalf("bad request_id: %q", Meta(err).RequestID)
	}
	// ContextLengthError must also satisfy errors.As(InvalidRequestError).
	var inv InvalidRequestError
	if !errors.As(err, &inv) {
		t.Fatal("ContextLengthError must unwrap as InvalidRequestError")
	}
}

func TestNormalizeErrorGeminiUnknownStatusFallsToHTTP(t *testing.T) {
	err, _ := NormalizeError("gemini", 503, `{"error":{"status":"UNAVAILABLE","message":"service unavailable"}}`)
	if err.ClassName() != "ServerError" || err.Code() != "server" {
		t.Fatalf("got %s/%s", err.ClassName(), err.Code())
	}
	if !err.Retryable() {
		t.Fatal("ServerError must be retryable")
	}
}

func TestNormalizeErrorNonJSONBody(t *testing.T) {
	err, _ := NormalizeError("openai_chat", 500, "Bad gateway")
	if err.ClassName() != "ServerError" {
		t.Fatalf("got %T", err)
	}
	if Meta(err).Message != "Bad gateway" {
		t.Fatalf("bad message: %q", Meta(err).Message)
	}
	m := NormalizedErrorMap(err)
	if m["provider_code"] != nil {
		t.Fatalf("provider_code should be null, got %v", m["provider_code"])
	}
}
