package lm15

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/url"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// The machinery every sign-in flow runs on (AUTH-18, AUTH-20, AUTH-21):
// port of lm15-python lm15/login/engine.py. One attempt's deadline,
// cancellation and UI; bounded (30 s, 1 MiB), TLS-only exchanges whose
// failures never carry a token or reflected provider text; RFC 8628 pacing;
// pasted and loopback returns checked against the attempt's registered
// return context, an invalid paste rejected and asked for again.

const (
	attemptLifetimeMs      = 15 * 60 * 1000.0 // AUTH-18, R9
	exchangeTimeoutMs      = 30_000.0         // AUTH-20.5, R9
	deviceDefaultIntervalS = 5.0              // RFC 8628 §3.2
	deviceSlowDownStepS    = 5.0              // RFC 8628 §3.5
	authResponseLimit      = 1024 * 1024      // AUTH-18
	callbackTargetLimit    = 8 * 1024
)

var oauthErrorCodes = map[string]bool{
	"invalid_request": true, "invalid_client": true, "invalid_grant": true, "unauthorized_client": true,
	"unsupported_grant_type": true, "invalid_scope": true, "access_denied": true, "server_error": true,
	"temporarily_unavailable": true, "authorization_pending": true, "slow_down": true, "expired_token": true,
}

// ─── What flows return; the manager maps it (AUTH-24) ─────────────────

var (
	errLoginCancelled = errors.New("login cancelled")
	errLoginExpired   = errors.New("login attempt deadline reached")
)

// loginDenied is a validated provider denial; the message is ours.
type loginDenied struct {
	message      string
	status       int
	providerCode string
	stage        string
}

func (e *loginDenied) Error() string { return e.message }

func denied(format string, args ...any) error {
	return &loginDenied{message: fmt.Sprintf(format, args...), stage: "authorization"}
}

func deniedAt(stage string, reply *httpReply, format string, args ...any) error {
	e := &loginDenied{message: fmt.Sprintf(format, args...), stage: stage}
	if reply != nil {
		e.status = reply.status
		e.providerCode = reply.oauthError
	}
	return e
}

// networkFailure: the network failed during an exchange. uncertain: it may
// have reached the provider (a timeout, a dropped connection) — AUTH-20.6.
type networkFailure struct {
	err       *Error
	uncertain bool
}

func (e *networkFailure) Error() string { return e.err.Error() }
func (e *networkFailure) Unwrap() error { return e.err }

func authOperation(message, reason, stage, commit, recovery string) *Error {
	e := newError(KindAuthOperation, message)
	e.Reason, e.Stage, e.CommitState, e.Recovery = reason, stage, commit, recovery
	return e
}

// ─── The attempt context ─────────────────────────────────────────────

type loginContext struct {
	ctx               context.Context
	ui                AuthUI
	provider          string
	deadline          float64 // monotonic ms
	monotonic         func() float64
	wallClock         func() int64
	sleep             func(context.Context, time.Duration) error
	transport         Transport
	listenerAvailable bool
}

func (c *loginContext) remainingMs() float64 { return c.deadline - c.monotonic() }

func (c *loginContext) check() error {
	if c.ctx.Err() != nil {
		return errLoginCancelled
	}
	if c.remainingMs() <= 0 {
		return errLoginExpired
	}
	return nil
}

func (c *loginContext) wait(ms float64) error {
	if err := c.check(); err != nil {
		return err
	}
	bounded := math.Min(math.Max(ms, 0), math.Max(c.remainingMs(), 0))
	if bounded > 0 {
		d := time.Duration(bounded * float64(time.Millisecond))
		if c.sleep != nil {
			if err := c.sleep(c.ctx, d); err != nil {
				return errLoginCancelled
			}
		} else {
			t := time.NewTimer(d)
			select {
			case <-t.C:
			case <-c.ctx.Done():
				t.Stop()
				return errLoginCancelled
			}
		}
	}
	return c.check()
}

func (c *loginContext) notify(n Notice) { c.ui.Notify(n) }

// promptWith asks the person; abandoned when the attempt ends or stop fires.
func (c *loginContext) promptWith(p Prompt, stop context.Context) (string, error) {
	if err := c.check(); err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(c.ctx, time.Duration(math.Max(c.remainingMs(), 0)*float64(time.Millisecond)))
	defer cancel()
	if stop != nil {
		go func() {
			select {
			case <-stop.Done():
				cancel()
			case <-ctx.Done():
			}
		}()
	}
	answer, err := c.ui.Prompt(ctx, p)
	if err != nil {
		if cerr := c.check(); cerr != nil {
			return "", cerr
		}
		return "", errLoginCancelled
	}
	if err := c.check(); err != nil {
		return "", err
	}
	return answer, nil
}

func (c *loginContext) prompt(p Prompt) (string, error) { return c.promptWith(p, nil) }

func (c *loginContext) budget() time.Duration {
	ms := math.Min(math.Max(c.remainingMs(), 100), exchangeTimeoutMs)
	return time.Duration(ms * float64(time.Millisecond))
}

func (c *loginContext) nowMs() int64 { return c.wallClock() }

// ─── Bounded HTTP ────────────────────────────────────────────────────

// httpReply: body may hold tokens — never rendered, never attached to an error.
type httpReply struct {
	status            int
	body              JSONObject
	ok                bool
	responseFormat    string
	oauthError        string
	securityChallenge bool
}

func (r *httpReply) failureSummary() string {
	parts := []string{fmt.Sprintf("HTTP %d", r.status), "response=" + r.responseFormat}
	if r.oauthError != "" {
		parts = append(parts, "OAuth error="+r.oauthError)
	} else {
		parts = append(parts, "no recognized OAuth error code; cause not established")
	}
	if r.securityChallenge {
		parts = append(parts, "response explicitly marked as a security challenge")
	} else if r.responseFormat == "html" {
		parts = append(parts, "HTML alone does not establish a security block")
	}
	return strings.Join(parts, "; ")
}

func (r *httpReply) str(key string) string {
	s, _ := r.body[key].(string)
	return s
}

func (r *httpReply) errorCode() string {
	switch v := r.body["error"].(type) {
	case string:
		return v
	case map[string]any:
		s, _ := v["code"].(string)
		return s
	}
	return ""
}

// formEncode is application/x-www-form-urlencoded in the given order (the reference's urlencode).
func formEncode(pairs [][2]string) string {
	parts := make([]string, len(pairs))
	for i, p := range pairs {
		parts[i] = url.QueryEscape(p[0]) + "=" + url.QueryEscape(p[1])
	}
	return strings.Join(parts, "&")
}

func withQuery(base string, pairs [][2]string) string { return base + "?" + formEncode(pairs) }

// notSent reports a failure that proves nothing reached the server.
func notSent(err error) bool {
	var dns *net.DNSError
	if errors.As(err, &dns) || errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ENETUNREACH) || errors.Is(err, syscall.EHOSTUNREACH) {
		return true
	}
	var op *net.OpError
	return errors.As(err, &op) && op.Op == "dial" && !op.Timeout()
}

func (c *loginContext) form(rawURL string, pairs [][2]string, headers [][2]string) (*httpReply, error) {
	return c.exchange("POST", rawURL, []byte(formEncode(pairs)), "application/x-www-form-urlencoded", headers)
}

func (c *loginContext) json(rawURL string, body any, headers [][2]string) (*httpReply, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(body); err != nil {
		return nil, err
	}
	return c.exchange("POST", rawURL, bytes.TrimRight(buf.Bytes(), "\n"), "application/json", headers)
}

func (c *loginContext) get(rawURL string, headers [][2]string) (*httpReply, error) {
	return c.exchange("GET", rawURL, nil, "", headers)
}

// exchange is one auth request: TLS only, no redirect followed, a failure
// names only its class and the URL.
func (c *loginContext) exchange(method, rawURL string, body []byte, contentType string, headers [][2]string) (*httpReply, error) {
	if err := c.check(); err != nil {
		return nil, err
	}
	if !strings.HasPrefix(rawURL, "https://") || len(rawURL) <= len("https://") {
		return nil, authOperation("refusing a credential-bearing exchange over a non-HTTPS URL", "method_unavailable", "exchange", "not_committed", "operator_action")
	}
	var h [][2]string
	if contentType != "" {
		h = append(h, [2]string{"Content-Type", contentType})
	}
	h = append(h, [2]string{"Accept", "application/json"})
	for _, extra := range headers {
		kept := h[:0]
		for _, existing := range h {
			if !strings.EqualFold(existing[0], extra[0]) {
				kept = append(kept, existing)
			}
		}
		h = append(kept, extra)
	}
	hasUA := false
	for _, x := range h {
		if strings.EqualFold(x[0], "User-Agent") {
			hasUA = true
		}
	}
	if !hasUA {
		h = append(h, [2]string{"User-Agent", "lm15/" + Version}) // AUTH-18: identify this SDK
	}
	where := strings.SplitN(rawURL, "?", 2)[0]
	ctx, cancel := context.WithTimeout(c.ctx, c.budget())
	defer cancel()
	resp, err := c.transport.Do(ctx, &TransportRequest{Method: method, URL: rawURL, Headers: h, Body: body, ReadTimeout: c.budget()})
	if err != nil {
		if c.ctx.Err() != nil {
			return nil, errLoginCancelled
		}
		kind := "TransportError"
		uncertain := true
		switch {
		case ctx.Err() == context.DeadlineExceeded:
			kind = "TimeoutError"
		case notSent(err):
			kind, uncertain = "ConnectError", false
		}
		e := newError(KindTransport, fmt.Sprintf("%s: network failure during an authentication exchange (%s) to %s", c.provider, kind, where))
		e.Provider = c.provider
		return nil, &networkFailure{err: e, uncertain: uncertain}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, authResponseLimit+1))
	if err != nil {
		e := newError(KindTransport, fmt.Sprintf("%s: the reply from %s could not be read in full", c.provider, where))
		return nil, &networkFailure{err: e, uncertain: true}
	}
	if len(raw) > authResponseLimit {
		e := newError(KindAuth, fmt.Sprintf("%s: authentication response exceeded %d bytes; refused", c.provider, authResponseLimit))
		e.Provider = c.provider
		return nil, e
	}
	reply := &httpReply{status: resp.Status, body: JSONObject{}, ok: resp.Status >= 200 && resp.Status < 300, responseFormat: "empty"}
	reply.securityChallenge = strings.EqualFold(strings.TrimSpace(resp.Header("cf-mitigated")), "challenge")
	if len(raw) > 0 {
		ct := strings.ToLower(strings.TrimSpace(strings.SplitN(resp.Header("content-type"), ";", 2)[0]))
		switch {
		case ct == "text/html" || ct == "application/xhtml+xml":
			reply.responseFormat = "html"
		case ct == "application/json" || strings.HasSuffix(ct, "+json"):
			reply.responseFormat = "invalid_json"
		default:
			reply.responseFormat = "text_or_binary"
		}
		if parsed, err := decodeStrictJSON(raw); err == nil {
			reply.responseFormat = "json"
			if obj, ok := parsed.(map[string]any); ok {
				reply.body = obj
			}
		}
		code := ""
		switch v := reply.body["error"].(type) {
		case string:
			code = v
		case map[string]any:
			if s, ok := v["code"].(string); ok {
				code = s
			} else if s, ok := v["type"].(string); ok {
				code = s
			}
		}
		if oauthErrorCodes[code] {
			reply.oauthError = code
		}
	}
	if resp.Status >= 500 {
		e := newError(KindServer, fmt.Sprintf("%s: the authentication server answered HTTP %d", c.provider, resp.Status))
		e.Provider, e.Status = c.provider, resp.Status
		return nil, e
	}
	return reply, nil
}

// ─── Device flow (RFC 8628) ──────────────────────────────────────────

type deviceStep struct {
	status   string // pending | slow_down | complete | denied | expired
	interval float64
	value    any
}

// runDeviceFlow polls until complete: the provider's interval or 5 s;
// slow_down never shortens it and adds at least 5 s; the provider's expiry
// bounds the attempt but never extends it.
func runDeviceFlow(c *loginContext, intervalS, expiresInS float64, poll func() (deviceStep, error)) (any, error) {
	interval := intervalS
	if interval <= 0 {
		interval = deviceDefaultIntervalS
	}
	interval = math.Max(interval, 1)
	if expiresInS > 0 {
		c.deadline = math.Min(c.deadline, c.monotonic()+expiresInS*1000)
	}
	if err := c.wait(interval * 1000); err != nil {
		return nil, err
	}
	for {
		if err := c.check(); err != nil {
			return nil, err
		}
		step, err := poll()
		if err != nil {
			return nil, err
		}
		switch step.status {
		case "complete":
			return step.value, nil
		case "denied":
			return nil, denied("the provider reported that authorization was denied")
		case "expired":
			return nil, errLoginExpired
		case "slow_down":
			interval = math.Max(interval+deviceSlowDownStepS, step.interval)
		}
		if err := c.wait(interval * 1000); err != nil {
			return nil, err
		}
	}
}

// ─── Returns (AUTH-18) ───────────────────────────────────────────────

type callbackReturn struct {
	code  string
	state string
	has   bool // state present
}

type returnContext struct {
	expectedState  string
	checkState     bool
	allowBareCode  bool
	registeredPath string
	registeredURI  string
}

func invalidReturn(message string) error {
	return authOperation(message, "invalid_login_state", "interaction", "not_committed", "provide_input")
}

func parseQueryPairs(query string) [][2]string {
	var out [][2]string
	for _, part := range strings.Split(query, "&") {
		if part == "" {
			continue
		}
		k, v, _ := strings.Cut(part, "=")
		k, _ = url.QueryUnescape(k)
		v, _ = url.QueryUnescape(v)
		out = append(out, [2]string{k, v})
	}
	return out
}

func effectivePort(u *url.URL) string {
	if p := u.Port(); p != "" {
		return p
	}
	switch u.Scheme {
	case "https":
		return "443"
	case "http":
		return "80"
	}
	return ""
}

// parseManualReturn reads a pasted return: a URL, code=…&state=…, code#state,
// or (only where the profile allows it) a bare code. Nothing pasted is ever
// quoted in an error; a wrong-state error return is invalid, never a denial.
func parseManualReturn(text string, rc returnContext) (callbackReturn, error) {
	value := strings.TrimSpace(text)
	if value == "" {
		return callbackReturn{}, invalidReturn("nothing was pasted")
	}
	if len(value) > callbackTargetLimit {
		return callbackReturn{}, invalidReturn("pasted return is too long")
	}
	var params [][2]string
	var code, state string
	hasState, bare, parsed := false, false, false
	switch {
	case strings.Contains(value, "://"):
		wrong := invalidReturn("the pasted URL is not this sign-in's registered return URL")
		u, err := url.Parse(value)
		if err != nil || u.User != nil || u.Fragment != "" || strings.Contains(value, "#") {
			return callbackReturn{}, wrong
		}
		if rc.registeredPath != "" && u.Path != rc.registeredPath {
			return callbackReturn{}, wrong
		}
		if rc.registeredURI != "" {
			exp, err := url.Parse(rc.registeredURI)
			if err != nil || u.Scheme != exp.Scheme || !strings.EqualFold(u.Hostname(), exp.Hostname()) || effectivePort(u) != effectivePort(exp) || u.Path != exp.Path {
				return callbackReturn{}, wrong
			}
		}
		params, parsed = parseQueryPairs(u.RawQuery), true
	case strings.HasPrefix(value, "code=") || strings.HasPrefix(value, "state=") || strings.HasPrefix(value, "error="):
		params, parsed = parseQueryPairs(value), true
	case strings.Contains(value, "#"):
		code, state, _ = strings.Cut(value, "#")
		hasState = true
	default:
		code, bare = value, true
	}
	isDenied := false
	if parsed {
		seen := map[string]bool{}
		get := map[string]string{}
		for _, p := range params {
			if seen[p[0]] {
				return callbackReturn{}, invalidReturn("the pasted return repeats a parameter")
			}
			seen[p[0]] = true
			get[p[0]] = p[1]
		}
		if seen["code"] && seen["error"] {
			return callbackReturn{}, invalidReturn("the pasted return contains both a code and an error")
		}
		isDenied = seen["error"]
		code = get["code"]
		state, hasState = get["state"], seen["state"]
	}
	if bare && !rc.allowBareCode {
		return callbackReturn{}, invalidReturn("paste the complete code#state or return URL, not the code alone")
	}
	if rc.checkState {
		if !hasState {
			if !(bare && rc.allowBareCode) {
				return callbackReturn{}, invalidReturn("this provider's return must carry its state value")
			}
		} else if subtle.ConstantTimeCompare([]byte(state), []byte(rc.expectedState)) != 1 {
			return callbackReturn{}, invalidReturn("the pasted return does not belong to this sign-in attempt")
		}
	}
	if isDenied {
		return callbackReturn{}, denied("the validated pasted return carries a provider error")
	}
	if code == "" {
		return callbackReturn{}, invalidReturn("no authorization code in the pasted text")
	}
	return callbackReturn{code: code, state: state, has: hasState}, nil
}

// awaitReturn is the authorization return, from the loopback listener or a
// paste, validated. An invalid paste is rejected with a notice and asked for
// again while the listener keeps listening (AUTH-18).
func awaitReturn(c *loginContext, listener *callbackListener, p Prompt, rc returnContext) (callbackReturn, error) {
	for {
		var answer string
		if listener != nil && !listener.isDone() {
			stop, cancelPrompt := context.WithCancel(context.Background())
			type result struct {
				text string
				err  error
			}
			person := make(chan result, 1)
			go func() {
				text, err := c.promptWith(p, stop)
				person <- result{text, err}
			}()
			deadline := time.NewTimer(time.Duration(math.Max(c.remainingMs(), 0) * float64(time.Millisecond)))
			select {
			case got := <-listener.results:
				deadline.Stop()
				cancelPrompt()
				listener.markDone()
				if d, ok := c.ui.(Dismisser); ok {
					d.Dismiss(p)
				}
				if got.err != nil {
					return callbackReturn{}, got.err
				}
				return got.value, nil
			case got := <-person:
				deadline.Stop()
				cancelPrompt()
				if got.err != nil {
					return callbackReturn{}, got.err
				}
				answer = got.text
			case <-deadline.C:
				cancelPrompt()
				return callbackReturn{}, errLoginExpired
			case <-c.ctx.Done():
				deadline.Stop()
				cancelPrompt()
				return callbackReturn{}, errLoginCancelled
			}
		} else {
			text, err := c.prompt(p)
			if err != nil {
				return callbackReturn{}, err
			}
			answer = text
		}
		found, err := parseManualReturn(answer, rc)
		if err == nil {
			return found, nil
		}
		var e *Error
		if errors.As(err, &e) && e.Reason == "invalid_login_state" {
			c.notify(Notice{Type: "info", Message: e.Message + ". Try again."})
			continue
		}
		return callbackReturn{}, err
	}
}

// ─── Randomness and small checks ─────────────────────────────────────

func randomBytes(n int) []byte {
	out := make([]byte, n)
	if _, err := rand.Read(out); err != nil {
		panic("lm15: the OS random generator is unavailable: " + err.Error())
	}
	return out
}

func randomBase64URL(n int) string { return base64.RawURLEncoding.EncodeToString(randomBytes(n)) }

// secretHex is n OS-random bytes as hex; it never degrades to a predictable value.
func secretHex(n int) string { return hex.EncodeToString(randomBytes(n)) }

// number reads a JSON number (json.Number, float64, int) as float64.
func number(v any) (float64, bool) {
	switch x := v.(type) {
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		return f, err == nil
	}
	return 0, false
}

func positive(v any) float64 {
	if _, isString := v.(string); isString {
		return 0
	}
	if f, ok := number(v); ok && f > 0 && !math.IsInf(f, 0) {
		return f
	}
	return 0
}

func httpsURL(v any) string {
	s, _ := v.(string)
	u, err := url.Parse(s)
	if err != nil || u.Scheme != "https" || u.Host == "" {
		return ""
	}
	return s
}

func httpURL(v any) string {
	s, _ := v.(string)
	u, err := url.Parse(s)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return ""
	}
	return s
}

// decodeStrictJSON parses JSON rejecting duplicate member names (AUTH-25),
// numbers kept as json.Number.
func decodeStrictJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := strictValue(dec)
	if err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func strictValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			out := map[string]any{}
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, _ := keyTok.(string)
				if _, dup := out[key]; dup {
					return nil, fmt.Errorf("duplicate member name")
				}
				value, err := strictValue(dec)
				if err != nil {
					return nil, err
				}
				out[key] = value
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return out, nil
		case '[':
			out := []any{}
			for dec.More() {
				value, err := strictValue(dec)
				if err != nil {
					return nil, err
				}
				out = append(out, value)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return out, nil
		}
		return nil, fmt.Errorf("unexpected delimiter")
	default:
		return t, nil
	}
}
