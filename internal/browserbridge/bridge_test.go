package browserbridge

import (
	"context"
	"encoding/base64"
	"io"
	"strings"
	"testing"

	lm15 "github.com/lm15-dev/lm15-go"
)

type transportFunc func(context.Context, *lm15.TransportRequest) (*lm15.TransportResponse, error)

func (f transportFunc) Do(ctx context.Context, req *lm15.TransportRequest) (*lm15.TransportResponse, error) {
	return f(ctx, req)
}
func input(provider string) lm15.JSONObject {
	msg, err := lm15.DecodeJSONObject([]byte(`{"provider": "", "api_key": "offline-placeholder", "canonical_request": {"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "Hello"}]}]}}`))
	if err != nil {
		panic(err)
	}
	msg.Set("provider", provider)
	return msg
}
func decode(t *testing.T, raw string) lm15.JSONObject {
	t.Helper()
	d, err := lm15.DecodeJSONObject([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return d
}
func TestBuildBytesMatchActualComplete(t *testing.T) {
	msg := input("openai-chat")
	built := decode(t, Call(context.Background(), "build_request", JSON(msg), nil, nil))
	body, err := base64.StdEncoding.DecodeString(built.Get("body_b64").(string))
	if err != nil {
		t.Fatal(err)
	}
	called := false
	transport := transportFunc(func(_ context.Context, req *lm15.TransportRequest) (*lm15.TransportResponse, error) {
		called = true
		if string(body) != string(req.Body) {
			t.Fatalf("build bytes differ from wire: %s / %s", body, req.Body)
		}
		return &lm15.TransportResponse{Status: 200, Body: io.NopCloser(strings.NewReader(`{"id":"r","model":"m","choices":[{"message":{"role":"assistant","content":"Hello back"},"finish_reason":"stop"}]}`))}, nil
	})
	reply := decode(t, Call(context.Background(), "complete", JSON(msg), transport, nil))
	if !called || reply.Get("canonical_response") == nil {
		t.Fatalf("not an actual complete: %+v", reply)
	}
}
func TestStreamUsesSDKStopAndAdaptations(t *testing.T) {
	msg := input("openai")
	canonical := msg.Get("canonical_request").(lm15.JSONObject)
	canonical.Set("config", lm15.JSONObject{{Key: "stop", Value: []any{"STOP"}}})
	msg.Set("canonical_request", canonical)
	closed := false
	transport := transportFunc(func(_ context.Context, req *lm15.TransportRequest) (*lm15.TransportResponse, error) {
		if !strings.Contains(string(req.Body), `"stream":true`) {
			t.Fatalf("not streaming: %s", req.Body)
		}
		sse := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello STOP unseen\",\"output_index\":0,\"content_index\":0}\n\n"
		return &lm15.TransportResponse{Status: 200, Body: &closeReader{Reader: strings.NewReader(sse), closed: &closed}}, nil
	})
	var events []lm15.JSONObject
	reply := decode(t, Call(context.Background(), "stream", JSON(msg), transport, func(s string) error { events = append(events, decode(t, s)); return nil }))
	response, ok := reply.Get("canonical_response").(lm15.JSONObject)
	if !ok {
		t.Fatalf("stream failed: %+v", reply)
	}
	resp, err := lm15.ResponseFromDict(response)
	if err != nil {
		t.Fatal(err)
	}
	if resp.TextOr("") != "hello " || len(resp.Adaptations) == 0 || len(events) < 3 || !closed {
		t.Fatalf("stop/assembly/adaptations/close missing: %+v events=%+v closed=%v", resp, events, closed)
	}
}

type closeReader struct {
	*strings.Reader
	closed *bool
}

func (r *closeReader) Close() error { *r.closed = true; return nil }
func TestTypedDiagnosticsAndInvalidInput(t *testing.T) {
	transport := transportFunc(func(context.Context, *lm15.TransportRequest) (*lm15.TransportResponse, error) {
		return &lm15.TransportResponse{Status: 429, Headers: [][2]string{{"x-request-id", "req-test"}, {"retry-after", "3"}, {"x-ratelimit-remaining-requests", "0"}, {"authorization", "secret"}}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"quota","code":"rate_limit"}}`))}, nil
	})
	reply := decode(t, Call(context.Background(), "complete", JSON(input("openai-chat")), transport, nil))
	e := reply.Get("error").(lm15.JSONObject)
	if e.Get("name") != "RateLimitError" || e.Get("http_response") == nil || !strings.Contains(e.Get("message").(string), "req-test") || strings.Contains(JSON(reply), "secret") {
		t.Fatalf("diagnostics: %+v", reply)
	}
	for _, raw := range []string{"null", "[]", "{", `{"provider":"openai-chat"}`, `{"provider":true}`} {
		if r := decode(t, Call(context.Background(), "complete", raw, nil, nil)); r.Get("error") == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if r := decode(t, Call(ctx, "complete", JSON(input("openai-chat")), nil, nil)); r.Get("error").(lm15.JSONObject).Get("name") != "AbortError" {
		t.Fatal(r)
	}
}
