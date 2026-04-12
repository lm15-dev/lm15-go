package lm15

import (
	"errors"
	"testing"
)

func TestMapHTTPError(t *testing.T) {
	if _, ok := MapHTTPError(401, "test").(*AuthError); !ok {
		t.Error("401 should be AuthError")
	}
	if _, ok := MapHTTPError(402, "test").(*BillingError); !ok {
		t.Error("402 should be BillingError")
	}
	if _, ok := MapHTTPError(429, "test").(*RateLimitError); !ok {
		t.Error("429 should be RateLimitError")
	}
	if _, ok := MapHTTPError(500, "test").(*ServerError); !ok {
		t.Error("500 should be ServerError")
	}
	if _, ok := MapHTTPError(400, "test").(*InvalidRequestError); !ok {
		t.Error("400 should be InvalidRequestError")
	}
	if _, ok := MapHTTPError(408, "test").(*TimeoutError); !ok {
		t.Error("408 should be TimeoutError")
	}
}

func TestCanonicalErrorCode(t *testing.T) {
	tests := []struct {
		err  error
		want string
	}{
		{&AuthError{}, "auth"},
		{&BillingError{}, "billing"},
		{&RateLimitError{}, "rate_limit"},
		{&ContextLengthError{}, "context_length"},
		{&InvalidRequestError{}, "invalid_request"},
		{&TimeoutError{}, "timeout"},
		{&ServerError{}, "server"},
		{&ProviderError{}, "provider"},
	}
	for _, tt := range tests {
		if got := CanonicalErrorCode(tt.err); got != tt.want {
			t.Errorf("got %s, want %s", got, tt.want)
		}
	}
}

func TestIsTransient(t *testing.T) {
	if !IsTransient(&RateLimitError{}) {
		t.Error("RateLimitError should be transient")
	}
	if !IsTransient(&ServerError{}) {
		t.Error("ServerError should be transient")
	}
	if IsTransient(&AuthError{}) {
		t.Error("AuthError should not be transient")
	}
}

func TestErrorForCode(t *testing.T) {
	err := ErrorForCode("auth", "bad key")
	var authErr *AuthError
	if !errors.As(err, &authErr) {
		t.Errorf("expected AuthError, got %T", err)
	}
}
