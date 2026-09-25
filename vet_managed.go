package lm15

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/url"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"
)

// vetManagedRun is the vet shim's managed_run op (lm15-contract
// harness/PROTOCOL.md § managed_run): one scripted program against the
// public managed-auth API with every seam injected — the store file the
// harness created, a fake wall and monotonic clock (waits advance them), a
// scripted auth server behind the transport, a scripted UI. It reports one
// outcome per step, the ordered trace and the store file afterwards; the
// harness compares.
func vetManagedRun(msg JSONObject) (JSONObject, error) {
	var mu sync.Mutex
	events := []any{}
	record := func(e JSONObject) {
		mu.Lock()
		events = append(events, e)
		mu.Unlock()
	}
	startF, _ := number(msg.Get("clock_ms"))
	start := int64(startF)
	var elapsed float64 // ms
	sentinel := wireStr(msg.Get("sentinel"))
	env := map[string]string{}
	for k, v := range wireObj(msg.Get("env")).All() {
		env[k] = wireStr(v)
	}
	storePath := wireStr(msg.Get("store_path"))
	script := wireList(msg.Get("http"))
	answers := wireList(msg.Get("ui"))

	server := &vetAuthServer{script: script, record: record}
	ui := &vetScriptUI{answers: answers, record: record}

	saved := os.Environ()
	os.Clearenv()
	for k, v := range env {
		_ = os.Setenv(k, v)
	}
	defer func() {
		os.Clearenv()
		for _, kv := range saved {
			if k, v, ok := strings.Cut(kv, "="); ok {
				_ = os.Setenv(k, v)
			}
		}
	}()

	store, err := NewFileStore(storePath)
	if err != nil {
		return nil, err
	}
	auth := NewAuthWithSeams(store, AuthSeams{
		WallClock: func() int64 { return start + int64(elapsed) },
		Monotonic: func() float64 { return elapsed },
		Sleep: func(_ context.Context, d time.Duration) error {
			ms := float64(d) / float64(time.Millisecond)
			elapsed += ms
			record(JSONObject{{"sleep_ms", int64(ms + 0.5)}})
			return nil
		},
		Transport: server,
		Env:       func(k string) string { return env[k] },
		Home:      env["HOME"],
	})
	ctx := context.Background()
	var outcomes []any
	for i, raw := range wireList(msg.Get("steps")) {
		record(JSONObject{{"step", i}})
		step, err := vetResolveRefs(wireObj(raw), outcomes)
		if err != nil {
			outcomes = append(outcomes, JSONObject{{"ok", false}, {"error", JSONObject{{"type", "ValueError"}}}})
			continue
		}
		provider := wireStr(step.Get("provider"))
		value, err := vetManagedStep(ctx, auth, ui, step, provider, env, sentinel, &elapsed)
		if err != nil {
			outcomes = append(outcomes, JSONObject{{"ok", false}, {"error", vetManagedError(err)}})
		} else {
			outcomes = append(outcomes, JSONObject{{"ok", true}, {"value", value}})
		}
	}
	var storeOut any
	if data, err := os.ReadFile(storePath); err == nil {
		if doc, err := DecodeJSON(data); err == nil {
			storeOut = JSONObject{{"document", doc}}
		} else {
			storeOut = JSONObject{{"raw", string(data)}}
		}
	}
	if outcomes == nil {
		outcomes = []any{}
	}
	return JSONObject{{"steps", outcomes}, {"events", events}, {"store", storeOut}}, nil
}

func vetResolveRefs(step JSONObject, outcomes []any) (JSONObject, error) {
	target := func(n any) (JSONObject, error) {
		f, _ := number(n)
		i := int(f)
		if i < 0 || i >= len(outcomes) {
			return nil, errors.New("step reference out of range")
		}
		o, _ := asObject(outcomes[i])
		v, ok := asObject(o.Get("value"))
		if o.Get("ok") != true || !ok {
			return nil, errors.New("step returned no connection to refer to")
		}
		return v, nil
	}
	out := JSONObject{}
	for k, v := range step.All() {
		if m, ok := asObject(v); ok {
			if n, ok := m.Lookup("id_of_step"); ok {
				c, err := target(n)
				if err != nil {
					return nil, err
				}
				out.Set(k, c.Get("id"))
				continue
			}
			if n, ok := m.Lookup("of_step"); ok {
				c, err := target(n)
				if err != nil {
					return nil, err
				}
				out.Set(k, []any{c.Get("id"), c.Get("identity_generation")})
				continue
			}
		}
		out.Set(k, v)
	}
	return out, nil
}

func vetStrings(v any) map[string]string {
	out := map[string]string{}
	for k, x := range wireObj(v).All() {
		out[k] = wireStr(x)
	}
	return out
}

func vetConnection(c *Connection) any {
	if c == nil {
		return nil
	}
	routes := make([]any, len(c.Routes))
	for i, r := range c.Routes {
		routes[i] = r
	}
	settings := JSONObject{}
	for _, k := range sortedMapKeys(c.Settings) {
		settings = append(settings, Member{k, c.Settings[k]})
	}
	out := JSONObject{{"id", c.ID}, {"provider", c.Provider}, {"instance_id", c.InstanceID}, {"kind", c.Kind}, {"method_id", c.MethodID},
		{"routes", routes}, {"label", c.Label}, {"created_at", c.CreatedAt}, {"identity_generation", c.IdentityGeneration},
		{"credential_revision", c.CredentialRevision}, {"settings", settings}}
	if c.AccountLabel != "" {
		out.Set("account_label", c.AccountLabel)
	}
	return out
}

func vetNullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func vetManagedStep(ctx context.Context, auth *Auth, ui AuthUI, step JSONObject, provider string, env map[string]string, sentinel string, elapsed *float64) (any, error) {
	switch wireStr(step.Get("do")) {
	case "advance":
		ms, _ := number(step.Get("ms"))
		*elapsed += ms
		return nil, nil
	case "login":
		allow, _ := step.Get("allow_unverified").(bool)
		c, err := auth.Login(ctx, provider, LoginOptions{Method: wireStr(step.Get("method")), UI: ui, Answers: vetStrings(step.Get("answers")), Settings: vetStrings(step.Get("settings")), Replace: wireStr(step.Get("replace")), AllowUnverified: allow})
		if err != nil {
			return nil, err
		}
		return vetConnection(&c), nil
	case "configure":
		c, err := auth.Configure(ctx, provider, wireStr(step.Get("method")), vetStrings(step.Get("answers")), vetStrings(step.Get("settings")), wireStr(step.Get("replace")))
		if err != nil {
			return nil, err
		}
		return vetConnection(&c), nil
	case "set_api_key":
		c, err := auth.SetAPIKey(ctx, provider, wireStr(step.Get("key")), wireStr(step.Get("replace")))
		if err != nil {
			return nil, err
		}
		return vetConnection(&c), nil
	case "status":
		s, err := auth.Status(provider)
		if err != nil {
			return nil, err
		}
		var verification any
		if s.Verification != nil {
			verification = JSONObject{{"result", s.Verification.Result}, {"check", vetNullable(s.Verification.Check)}}
		}
		return JSONObject{{"provider", s.Provider}, {"presence", s.Presence}, {"usability", s.Usability}, {"connection", vetConnection(s.Connection)},
			{"expires_at", vetNullable(s.ExpiresAt)}, {"logged_out", s.LoggedOut}, {"verification", verification}}, nil
	case "connections":
		list, err := auth.Connections()
		if err != nil {
			return nil, err
		}
		out := []any{}
		for i := range list {
			out = append(out, vetConnection(&list[i]))
		}
		return out, nil
	case "logout":
		r, err := auth.Logout(ctx, wireStr(step.Get("target")))
		if err != nil {
			return nil, err
		}
		routes := make([]any, len(r.Routes))
		for i, x := range r.Routes {
			routes[i] = x
		}
		return JSONObject{{"provider", r.Provider}, {"forgot", r.Forgot}, {"routes", routes}, {"identity_generation", r.IdentityGeneration}}, nil
	case "cancel_login":
		return auth.CancelLogin(ctx, provider)
	case "request_auth":
		var pinned *[2]string
		if p := wireList(step.Get("pinned")); len(p) == 2 {
			pinned = &[2]string{wireStr(p[0]), wireStr(p[1])}
		}
		r, err := auth.RequestAuth(ctx, provider, pinned)
		if err != nil {
			return nil, err
		}
		var credential any
		if r.CredentialKind != "" {
			credential = JSONObject{{"kind", r.CredentialKind}, {"value", r.Credential}}
		}
		headers := JSONObject{}
		for _, k := range sortedMapKeys(r.Headers) {
			headers = append(headers, Member{k, r.Headers[k]})
		}
		return JSONObject{{"credential", credential}, {"headers", headers}, {"base_url", vetNullable(r.BaseURL)}, {"account_id", vetNullable(r.AccountID)}, {"named", vetNullable(r.Named)}}, nil
	case "methods":
		methods, err := auth.Methods(provider)
		if err != nil {
			return nil, err
		}
		out := []any{}
		for _, m := range methods {
			fields := []any{}
			for _, f := range m.Fields {
				options := []any{}
				for _, o := range f.Options {
					options = append(options, o.ID)
				}
				fields = append(fields, JSONObject{{"id", f.ID}, {"type", f.Type}, {"required", f.Required}, {"options", options}})
			}
			delivery := []any{}
			for _, d := range m.Delivery {
				delivery = append(delivery, d)
			}
			out = append(out, JSONObject{{"id", m.ID}, {"kind", m.Kind}, {"flow", m.Flow}, {"availability", m.Availability}, {"subscription", m.Subscription}, {"delivery", delivery}, {"fields", fields}})
		}
		return out, nil
	case "providers":
		out := []any{}
		for _, d := range auth.Providers() {
			out = append(out, d.ID)
		}
		return out, nil
	case "explain":
		opts := ExplainOptions{Env: env, Auth: auth}
		if keys := wireList(step.Get("api_keys")); len(keys) > 0 {
			opts.APIKeys = map[string]CredentialLike{}
			for _, k := range keys {
				opts.APIKeys[wireStr(k)] = sentinel + "-explicit"
			}
		}
		report, err := ExplainAuth(provider, opts)
		if err != nil {
			return nil, err
		}
		steps := []any{}
		for _, s := range report.Steps {
			steps = append(steps, JSONObject{{"kind", s.Kind}, {"state", s.State}})
		}
		return JSONObject{{"configured", report.Configured}, {"steps", steps}}, nil
	}
	return nil, valueErrorf("unknown managed step %q", wireStr(step.Get("do")))
}

func vetManagedError(err error) JSONObject {
	if errors.Is(err, context.Canceled) {
		return JSONObject{{"type", "cancelled"}}
	}
	var e *Error
	if errors.As(err, &e) {
		if e.Kind == KindAuthOperation {
			return JSONObject{{"type", "AuthOperationError"}, {"code", e.Code}, {"reason", e.Reason}, {"stage", e.Stage}, {"commit_state", e.CommitState}, {"recovery", e.Recovery}}
		}
		kind := e.Kind
		if kind == KindMissingCredential {
			kind = KindNotConfigured
		}
		if kind == KindCredentialLockWait {
			kind = KindLockTimeout
		}
		return JSONObject{{"type", string(kind)}, {"code", e.Code}}
	}
	return JSONObject{{"type", "ValueError"}}
}

// ─── The scripted auth server and UI ─────────────────────────────────

type vetAuthServer struct {
	mu     sync.Mutex
	script []any
	record func(JSONObject)
}

var vetTransportHeaders = map[string]bool{"accept": true, "accept-encoding": true, "connection": true, "content-length": true, "content-type": true, "host": true}

func (s *vetAuthServer) Do(ctx context.Context, req *TransportRequest) (*TransportResponse, error) {
	contentType := strings.TrimSpace(strings.SplitN(req.Header("Content-Type"), ";", 2)[0])
	headers := JSONObject{}
	for _, h := range req.Headers {
		key := strings.ToLower(h[0])
		if vetTransportHeaders[key] {
			continue
		}
		value := h[1]
		if key == "user-agent" && strings.HasPrefix(value, "lm15/") {
			value = "lm15"
		}
		headers.Set(key, value)
	}
	var body any
	if len(req.Body) > 0 {
		if contentType == "application/x-www-form-urlencoded" {
			form := JSONObject{}
			values, _ := url.ParseQuery(string(req.Body))
			for k, v := range values {
				form.Set(k, v[0])
			}
			body = form
		} else if parsed, err := DecodeJSON(req.Body); err == nil {
			body = parsed
		} else {
			body = string(req.Body)
		}
	}
	var ct any
	if contentType != "" {
		ct = contentType
	}
	s.record(JSONObject{{"http", JSONObject{{"method", req.Method}, {"url", req.URL}, {"content_type", ct}, {"headers", headers}, {"body", body}}}})
	s.mu.Lock()
	var reply JSONObject
	if len(s.script) > 0 {
		reply, _ = asObject(s.script[0])
		s.script = s.script[1:]
	}
	s.mu.Unlock()
	refused := &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}
	if reply == nil {
		return nil, refused
	}
	if delay, ok := number(reply.Get("delay_ms")); ok && delay > 0 {
		time.Sleep(time.Duration(delay) * time.Millisecond) // real time: another process may race this exchange
	}
	switch wireStr(reply.Get("network")) {
	case "timeout":
		return nil, &net.OpError{Op: "read", Net: "tcp", Err: errors.New("i/o timeout")}
	case "refused":
		return nil, refused
	}
	status := 200
	if f, ok := number(reply.Get("status")); ok {
		status = int(f)
	}
	if j, ok := reply.Lookup("json"); ok {
		data, _ := EncodeJSON(j)
		return &TransportResponse{Status: status, Headers: [][2]string{{"content-type", "application/json"}}, Body: io.NopCloser(bytes.NewReader(data))}, nil
	}
	ctype := wireStr(reply.Get("content_type"))
	if ctype == "" {
		ctype = "text/plain"
	}
	return &TransportResponse{Status: status, Headers: [][2]string{{"content-type", ctype}}, Body: io.NopCloser(strings.NewReader(wireStr(reply.Get("text"))))}, nil
}

type vetScriptUI struct {
	mu          sync.Mutex
	answers     []any
	record      func(JSONObject)
	lastAuthURL string
}

func (u *vetScriptUI) Notify(n Notice) {
	event := JSONObject{{"type", n.Type}}
	switch n.Type {
	case "auth_url":
		u.mu.Lock()
		u.lastAuthURL = n.URL
		u.mu.Unlock()
		event.Set("url", n.URL)
	case "device_code":
		event.Set("user_code", n.UserCode)
		event.Set("verification_url", n.VerificationURL)
		event.Set("expires_in_s", n.ExpiresInS)
		event.Set("interval_s", n.IntervalS)
	case "progress":
		event.Set("stage", n.Stage)
	}
	u.record(JSONObject{{"notice", event}})
}

func (u *vetScriptUI) Prompt(_ context.Context, p Prompt) (string, error) {
	event := JSONObject{{"type", p.Type}, {"field_id", p.FieldID}}
	if p.Type == "select" {
		options := []any{}
		for _, o := range p.Options {
			options = append(options, o.ID)
		}
		event.Set("options", options)
	}
	u.record(JSONObject{{"prompt", event}})
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.answers) == 0 {
		return "", ErrPromptCancelled
	}
	answer := u.answers[0]
	u.answers = u.answers[1:]
	if s, ok := answer.(string); ok {
		return s, nil
	}
	m, _ := asObject(answer)
	if m == nil || m.Get("cancel") != nil {
		return "", ErrPromptCancelled
	}
	parsed, _ := url.Parse(u.lastAuthURL)
	query := url.Values{}
	if parsed != nil {
		query = parsed.Query()
	}
	state := query.Get("state")
	switch {
	case m.Get("paste") != nil:
		return wireStr(m.Get("paste")) + "#" + state, nil
	case m.Get("paste_wrong_state") != nil:
		return wireStr(m.Get("paste_wrong_state")) + "#not-the-state-of-this-attempt", nil
	case m.Get("paste_url") != nil:
		return query.Get("redirect_uri") + "?" + formEncode([][2]string{{"code", wireStr(m.Get("paste_url"))}, {"state", state}}), nil
	}
	return "", ErrPromptCancelled
}
