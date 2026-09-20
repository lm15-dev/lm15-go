package lm15

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// The shared diagnostic-header vectors (lm15-contract/errors/diagnostic-headers.json):
// closed allowlist, precedence of retry hints, request-id fallbacks, secrecy.
func TestDiagnosticHeaderVectors(t *testing.T) {
	raw, err := os.ReadFile("../lm15-contract/errors/diagnostic-headers.json")
	if err != nil {
		t.Skip("contract corpus not present:", err)
	}
	var doc struct {
		Sentinel string `json:"sentinel"`
		Cases    []struct {
			ID             string      `json:"id"`
			Status         int         `json:"status"`
			BodyRetryAfter *float64    `json:"body_retry_after"`
			BodyRequestID  string      `json:"body_request_id"`
			Headers        [][2]string `json:"headers"`
			Expect         struct {
				RetryAfter *float64            `json:"retry_after"`
				RequestID  *string             `json:"request_id"`
				Headers    map[string][]string `json:"rate_limit_headers"`
			} `json:"expect"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	for _, c := range doc.Cases {
		e := MapHTTPError(c.Status, "x", "azure", nil, "", c.BodyRequestID, c.BodyRetryAfter)
		attachErrorMetadata(e, c.Headers)
		if (e.RetryAfter == nil) != (c.Expect.RetryAfter == nil) || (e.RetryAfter != nil && *e.RetryAfter != *c.Expect.RetryAfter) {
			t.Errorf("%s: retry_after got %v want %v", c.ID, e.RetryAfter, c.Expect.RetryAfter)
		}
		wantID := ""
		if c.Expect.RequestID != nil {
			wantID = *c.Expect.RequestID
		}
		if e.RequestID != wantID {
			t.Errorf("%s: request_id got %q want %q", c.ID, e.RequestID, wantID)
		}
		got, _ := json.Marshal(e.RateLimitHeaders.toJSON())
		want, _ := json.Marshal(c.Expect.Headers)
		if len(c.Expect.Headers) == 0 {
			want = []byte("null")
		}
		if string(got) != string(want) {
			t.Errorf("%s: rate_limit_headers got %s want %s", c.ID, got, want)
		}
		if strings.Contains(e.Error(), doc.Sentinel) {
			t.Errorf("%s: the sentinel leaked into the error text", c.ID)
		}
	}
}
