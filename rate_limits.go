package lm15

// Bounded provider evidence, not a quota model or a retry policy.
//
// Contract: docs/error-diagnostics.md (2026-09-19). Values keep their vendor
// units and duplicates. No credential, cookie, or header outside the closed
// set is ever copied.

import (
	"encoding/json"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// RateLimitHeaders is the diagnostic snapshot: lowercase name → values in
// arrival order. Treat it as immutable; constructors copy it.
type RateLimitHeaders map[string][]string

// rateLimitHeaderNames is the CLOSED allowlist (brace notation in the
// contract enumerates a set, not a wildcard).
var rateLimitHeaderNames = func() map[string]bool {
	names := map[string]bool{
		"retry-after": true, "retry-after-ms": true, "x-ms-retry-after-ms": true,
		"x-ratelimit-type": true, "x-ratelimit-abusepenalty-active": true,
	}
	for _, field := range []string{"limit", "remaining", "reset", "renewalperiod"} {
		for _, unit := range []string{"requests", "tokens"} {
			names["x-ratelimit-"+field+"-"+unit] = true
		}
	}
	for _, unit := range []string{"requests", "tokens", "input-tokens", "output-tokens"} {
		for _, field := range []string{"limit", "remaining", "reset"} {
			names["anthropic-ratelimit-"+unit+"-"+field] = true
		}
	}
	return names
}()

func printableASCII(value string) bool {
	for i := 0; i < len(value); i++ {
		if value[i] < 0x20 || value[i] > 0x7e {
			return false
		}
	}
	return true
}

// CaptureRateLimits builds the snapshot from response headers: closed
// allowlist, at most four values per name, each 1–256 printable ASCII
// characters, arrival order preserved. Names are lowercased; values are
// never normalized.
func CaptureRateLimits(headers [][2]string) RateLimitHeaders {
	out := RateLimitHeaders{}
	for _, h := range headers {
		name := strings.ToLower(h[0])
		value := h[1]
		if !rateLimitHeaderNames[name] || len(value) < 1 || len(value) > 256 || !printableASCII(value) {
			continue
		}
		if len(out[name]) < 4 {
			out[name] = append(out[name], value)
		}
	}
	return out
}

// freezeRateLimits re-validates a snapshot (a deserialized one, or a
// caller-supplied one) under the same bounds and copies it.
func freezeRateLimits(snapshot map[string][]string) RateLimitHeaders {
	var pairs [][2]string
	names := make([]string, 0, len(snapshot))
	for name := range snapshot {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		for _, v := range snapshot[name] {
			pairs = append(pairs, [2]string{name, v})
		}
	}
	return CaptureRateLimits(pairs)
}

// Clone returns an independent copy.
func (r RateLimitHeaders) Clone() RateLimitHeaders {
	if r == nil {
		return nil
	}
	out := make(RateLimitHeaders, len(r))
	for k, v := range r {
		out[k] = append([]string(nil), v...)
	}
	return out
}

// toJSON renders the snapshot as canonical JSON (sorted names, string arrays).
func (r RateLimitHeaders) toJSON() JSONObject {
	if len(r) == 0 {
		return nil
	}
	out := JSONObject{}
	for name, values := range r {
		out[name] = toAnyList(values, func(s string) any { return s })
	}
	return out
}

var millisecondsPattern = regexp.MustCompile(`^\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$`)

// millisecondsSeconds reads a millisecond header (retry-after-ms,
// x-ms-retry-after-ms): a finite nonnegative decimal number, never a date.
func millisecondsSeconds(value string) *float64 {
	if len(value) > 256 {
		return nil
	}
	value = strings.TrimSpace(value)
	if !millisecondsPattern.MatchString(value) {
		return nil
	}
	number, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsInf(number, 0) || math.IsNaN(number) || number < 0 {
		return nil
	}
	seconds := number / 1000
	return &seconds
}

// retryAfterSeconds reads a Retry-After header: delta-seconds or an HTTP
// date (a date in the past is zero). Anything else is no hint.
func retryAfterSeconds(value string) *float64 {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	if n, err := strconv.ParseFloat(value, 64); err == nil {
		if math.IsInf(n, 0) || math.IsNaN(n) || n < 0 {
			return nil
		}
		return &n
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return nil
	}
	secs := math.Max(0, time.Until(when).Seconds())
	return &secs
}

// firstHeader returns the first value of a header (case-insensitive).
func firstHeader(headers [][2]string, name string) (string, bool) {
	for _, h := range headers {
		if strings.EqualFold(h[0], name) {
			return h[1], true
		}
	}
	return "", false
}

// requestIDHeaderOrder is the request-id fallback order
// (docs/error-diagnostics.md): a body request id wins over all of them.
var requestIDHeaderOrder = []string{
	"x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id", "x-ms-request-id",
	"apim-request-id", "x-typesafe-request-id",
}

// httpDiagnostics is the handshake evidence an HTTP reply carries: the
// three things an error (or an in-stream ErrorDetail) may learn from headers.
type httpDiagnostics struct {
	RequestID  string
	RetryAfter *float64
	RateLimits RateLimitHeaders
}

// diagnosticsFromHeaders reads the request id, the retry hint (Retry-After
// first, then the millisecond headers) and the rate-limit snapshot.
func diagnosticsFromHeaders(headers [][2]string) httpDiagnostics {
	d := httpDiagnostics{RateLimits: CaptureRateLimits(headers)}
	if v, ok := firstHeader(headers, "retry-after"); ok {
		d.RetryAfter = retryAfterSeconds(v)
	}
	if d.RetryAfter == nil {
		for _, name := range []string{"retry-after-ms", "x-ms-retry-after-ms"} {
			if v, ok := firstHeader(headers, name); ok {
				if d.RetryAfter = millisecondsSeconds(v); d.RetryAfter != nil {
					break
				}
			}
		}
	}
	for _, name := range requestIDHeaderOrder {
		if v, ok := firstHeader(headers, name); ok && v != "" {
			d.RequestID = v
			break
		}
	}
	return d
}

// attachErrorMetadata fills what the body did not say from the headers:
// the retry hint, the request id, and the rate-limit snapshot (always
// attached, whatever the status or class).
func attachErrorMetadata(e *Error, headers [][2]string) {
	if e == nil {
		return
	}
	if e.RetryAfter != nil && (math.IsInf(*e.RetryAfter, 0) || math.IsNaN(*e.RetryAfter) || *e.RetryAfter < 0) {
		e.RetryAfter = nil
	}
	d := diagnosticsFromHeaders(headers)
	if e.RetryAfter == nil {
		e.RetryAfter = d.RetryAfter
	}
	if e.RequestID == "" {
		e.RequestID = d.RequestID
	}
	if len(e.RateLimitHeaders) == 0 && len(d.RateLimits) > 0 {
		e.RateLimitHeaders = d.RateLimits
	}
}

// diagnosticsText is the compact raw/advisory paragraph a ProviderError's
// rendering appends when it has evidence: bounded preview, values escaped
// for display, the message field itself untouched.
func diagnosticsText(snapshot RateLimitHeaders, retryAfter *float64) string {
	var pieces []string
	if retryAfter != nil && !math.IsInf(*retryAfter, 0) && !math.IsNaN(*retryAfter) && *retryAfter >= 0 {
		pieces = append(pieces, "Retry advice: "+strconv.FormatFloat(*retryAfter, 'g', -1, 64)+" seconds (not a guarantee).")
	}
	if len(snapshot) > 0 {
		raw, _ := json.Marshal(snapshot) // sorted names, ASCII-escaped
		text := string(raw)
		if len(text) > 2048 {
			text = text[:2048] + "... [full retained values in RateLimitHeaders]"
		}
		pieces = append(pieces, "Provider rate-limit headers (raw; advisory): "+text)
	}
	if len(pieces) == 0 {
		return ""
	}
	return "\n\n  " + strings.Join(pieces, "\n  ")
}

// ─── ErrorDetail.http_response ───────────────────────────────────────

// HTTPResponseDetail is the optional handshake block of an in-stream
// ErrorDetail (spec/types.md, 2026-09-19): request id, retry hint and the
// rate-limit snapshot of the HTTP reply the stream rode on. Empty means no
// evidence, never success.
type HTTPResponseDetail struct {
	RequestID        string
	RetryAfter       *float64
	RateLimitHeaders RateLimitHeaders
}

// IsEmpty reports whether the block carries nothing (omitted on the wire).
func (h HTTPResponseDetail) IsEmpty() bool {
	return h.RequestID == "" && h.RetryAfter == nil && len(h.RateLimitHeaders) == 0
}

// Validate checks the field constraints.
func (h HTTPResponseDetail) Validate() error {
	if h.RetryAfter != nil && (math.IsInf(*h.RetryAfter, 0) || math.IsNaN(*h.RetryAfter) || *h.RetryAfter < 0) {
		return valueErrorf("http_response.retry_after must be finite nonnegative seconds")
	}
	return nil
}

func (h HTTPResponseDetail) toDict() JSONObject {
	if h.IsEmpty() {
		return nil
	}
	d := dict{}
	if h.RequestID != "" {
		d["request_id"] = h.RequestID
	}
	if h.RetryAfter != nil {
		d["retry_after"] = jsonFloat(*h.RetryAfter)
	}
	if rl := h.RateLimitHeaders.toJSON(); rl != nil {
		d["rate_limit_headers"] = rl
	}
	return JSONObject(d)
}

// httpResponseDetailFromJSON reads the block: only the three keys, null is
// not an object.
func httpResponseDetailFromJSON(v any) (HTTPResponseDetail, error) {
	var out HTTPResponseDetail
	if v == nil {
		return out, nil
	}
	d, ok := v.(map[string]any)
	if !ok {
		return out, typeErrorf("ErrorDetail.http_response must be an object")
	}
	for key := range d {
		if key != "request_id" && key != "retry_after" && key != "rate_limit_headers" {
			return out, valueErrorf("unknown ErrorDetail.http_response field: %s", key)
		}
	}
	if raw, ok := d["request_id"]; ok && raw != nil {
		s, ok := raw.(string)
		if !ok || s == "" {
			return out, valueErrorf("http_response.request_id must be a non-empty string")
		}
		out.RequestID = s
	}
	if raw, ok := d["retry_after"]; ok && raw != nil {
		f, err := jsonFloat64(raw, "http_response.retry_after")
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) || f < 0 {
			return out, valueErrorf("http_response.retry_after must be finite nonnegative seconds")
		}
		out.RetryAfter = &f
	}
	if raw, ok := d["rate_limit_headers"]; ok {
		m, ok := raw.(map[string]any)
		if !ok {
			return out, typeErrorf("http_response.rate_limit_headers must map names to string arrays")
		}
		snapshot := map[string][]string{}
		for name, values := range m {
			list, ok := values.([]any)
			if !ok {
				return out, typeErrorf("http_response.rate_limit_headers must map names to string arrays")
			}
			for _, item := range list {
				s, ok := item.(string)
				if !ok {
					return out, typeErrorf("http_response.rate_limit_headers must map names to string arrays")
				}
				snapshot[name] = append(snapshot[name], s)
			}
		}
		if frozen := freezeRateLimits(snapshot); len(frozen) > 0 {
			out.RateLimitHeaders = frozen
		}
	}
	return out, out.Validate()
}

// httpResponseDetailOf is the in-stream block an HTTP driver attaches.
func httpResponseDetailOf(headers [][2]string) HTTPResponseDetail {
	d := diagnosticsFromHeaders(headers)
	return HTTPResponseDetail{RequestID: d.RequestID, RetryAfter: d.RetryAfter, RateLimitHeaders: d.RateLimits}
}
