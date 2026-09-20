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
	return lm15.JSONObject{"provider": provider, "api_key": "offline-placeholder", "canonical_request": lm15.JSONObject{"model": "m", "messages": []any{lm15.JSONObject{"role": "user", "parts": []any{lm15.JSONObject{"type": "text", "text": "Hello"}}}}}}
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
	body, err := base64.StdEncoding.DecodeString(built["body_b64"].(string))
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
	if !called || reply["canonical_response"] == nil {
		t.Fatalf("not an actual complete: %+v", reply)
	}
}
func TestStreamUsesSDKStopAndAdaptations(t *testing.T) {
	msg := input("openai")
	msg["canonical_request"].(lm15.JSONObject)["config"] = lm15.JSONObject{"stop": []any{"STOP"}}
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
	canonical, ok := reply["canonical_response"].(map[string]any)
	if !ok {
		t.Fatalf("stream failed: %+v", reply)
	}
	resp, err := lm15.ResponseFromDict(canonical)
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
	e := reply["error"].(map[string]any)
	if e["name"] != "RateLimitError" || e["http_response"] == nil || !strings.Contains(e["message"].(string), "req-test") || strings.Contains(JSON(reply), "secret") {
		t.Fatalf("diagnostics: %+v", reply)
	}
	for _, raw := range []string{"null", "[]", "{", `{"provider":"openai-chat"}`, `{"provider":true}`} {
		if r := decode(t, Call(context.Background(), "complete", raw, nil, nil)); r["error"] == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if r := decode(t, Call(ctx, "complete", JSON(input("openai-chat")), nil, nil)); r["error"].(map[string]any)["name"] != "AbortError" {
		t.Fatal(r)
	}
}
