package lm15

import (
	"strings"
	"testing"
)

func TestHTTPRequestToDict(t *testing.T) {
	req := HTTPRequest{
		Method: "POST",
		URL:    "https://api.openai.com/v1/responses",
		Headers: map[string]string{
			"Authorization": "Bearer sk-test",
			"Content-Type":  "application/json",
		},
		Body: []byte(`{"model":"gpt-4.1-mini","input":[{"role":"user","content":[{"type":"input_text","text":"Hello."}]}]}`),
	}

	dict := HTTPRequestToDict(req)

	if dict.Method != "POST" {
		t.Errorf("expected POST, got %s", dict.Method)
	}
	if dict.Headers["Authorization"] != "REDACTED" {
		t.Error("auth should be redacted")
	}
	if dict.Headers["Content-Type"] != "application/json" {
		t.Error("Content-Type should be preserved")
	}
	if dict.Body == nil {
		t.Error("body should be parsed")
	}
}

func TestHTTPRequestToCurl(t *testing.T) {
	req := HTTPRequest{
		Method: "POST",
		URL:    "https://api.openai.com/v1/responses",
		Headers: map[string]string{
			"Authorization": "Bearer sk-test",
			"Content-Type":  "application/json",
		},
		Body: []byte(`{"model":"gpt-4.1-mini"}`),
	}

	curl := HTTPRequestToCurl(req, true)
	if !strings.Contains(curl, "curl") {
		t.Error("should start with curl")
	}
	if !strings.Contains(curl, "REDACTED") {
		t.Error("auth should be redacted")
	}
	if strings.Contains(curl, "sk-test") {
		t.Error("real key should not appear when redacted")
	}
	if !strings.Contains(curl, "api.openai.com") {
		t.Error("should contain the URL")
	}

	// Test without redaction
	curlNoRedact := HTTPRequestToCurl(req, false)
	if !strings.Contains(curlNoRedact, "sk-test") {
		t.Error("real key should appear when not redacted")
	}
}

func TestHTTPRequestToCurlWithParams(t *testing.T) {
	req := HTTPRequest{
		Method:  "POST",
		URL:     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
		Params:  map[string]string{"alt": "sse"},
		Headers: map[string]string{"x-goog-api-key": "fake"},
		Body:    []byte(`{"contents":[]}`),
	}

	curl := HTTPRequestToCurl(req, true)
	if !strings.Contains(curl, "alt=sse") {
		t.Error("should include query params")
	}
}
