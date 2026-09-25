package lm15

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"
)

// Auth is one scope's connections and their lifecycle (spec/auth-managed.md
// AUTH-14 construction is inert, AUTH-17 what each operation may touch,
// AUTH-19 generations, replacement, logout and cancellation ordered against
// commit, AUTH-20 renewal under the lock with a durable in-flight marker and
// uncertainty never retried blind, AUTH-24 typed outcomes).
//
// Port of lm15-python lm15/login/manager.py, graded by the same runs
// (lm15-contract harness/managed.py). A slot per provider route carries an
// identity generation (bumped on every new connection and on logout, never
// reused) and a credential revision (bumped on every renewal). A bound
// client pins (connection id, generation); a managed router reads the slot
// per request. A legacy entry (no slot record) reads as generation 1 with id
// legacy-<provider>, rewritten only by a managed commit.
type Auth struct {
	store Store
	core  *authCore
	pin   *[2]string
}

type authCore struct {
	wallClock func() int64
	monotonic func() float64
	sleep     func(context.Context, time.Duration) error
	transport Transport
	env       func(string) string
	home      string
	mu        sync.Mutex
	active    map[string]context.CancelFunc
	closed    bool
}

// AuthSeams are what tests and the vet shim inject; production leaves them zero.
type AuthSeams struct {
	WallClock func() int64   // epoch milliseconds
	Monotonic func() float64 // milliseconds
	Sleep     func(context.Context, time.Duration) error
	Transport Transport
	Env       func(string) string // the environment $VAR recipes read; default os.Getenv
	Home      string              // where other tools' logins live; default $HOME
}

// RenewalLeadMs is AUTH-20.3's lead: min(300 s, lifetime / 10).
const RenewalLeadMs = 300_000.0

// NewAuth is a manager over store. Construction reads nothing (AUTH-14).
func NewAuth(store Store) *Auth { return NewAuthWithSeams(store, AuthSeams{}) }

// NewAuthWithSeams is NewAuth with injected clocks, transport and environment.
func NewAuthWithSeams(store Store, seams AuthSeams) *Auth {
	started := time.Now()
	core := &authCore{active: map[string]context.CancelFunc{}, home: seams.Home, transport: seams.Transport, sleep: seams.Sleep}
	core.wallClock = seams.WallClock
	if core.wallClock == nil {
		core.wallClock = func() int64 { return time.Now().UnixMilli() }
	}
	core.monotonic = seams.Monotonic
	if core.monotonic == nil {
		core.monotonic = func() float64 { return float64(time.Since(started).Microseconds()) / 1000 }
	}
	core.env = seams.Env
	if core.env == nil {
		core.env = os.Getenv
	}
	return &Auth{store: store, core: core}
}

// LocalAuth is the private file ($LM15_CREDENTIALS_PATH or ~/.config/lm15/credentials.json, or path).
func LocalAuth(path string) (*Auth, error) {
	store, err := NewFileStore(path)
	if err != nil {
		return nil, err
	}
	return NewAuth(store), nil
}

// MemoryAuth keeps everything in this process.
func MemoryAuth() *Auth { return NewAuth(NewMemoryStore()) }

// Store is where this scope's connections are saved.
func (a *Auth) Store() Store { return a.store }

// String never renders contents.
func (a *Auth) String() string { return "Auth(" + a.store.Description() + ")" }

// WithPin is the same scope with every request-time resolution checked
// against one (connection id, identity generation) — a bound client's (AUTH-20.1).
func (a *Auth) WithPin(connectionID, generation string) *Auth {
	return &Auth{store: a.store, core: a.core, pin: &[2]string{connectionID, generation}}
}

func (a *Auth) now() int64 { return a.core.wallClock() }

func (a *Auth) transport() Transport {
	a.core.mu.Lock()
	defer a.core.mu.Unlock()
	if a.core.transport == nil {
		t := NewHTTPTransport()
		// AUTH-20.9: a credential-bearing exchange never follows a redirect.
		t.Client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		a.core.transport = t
	}
	return a.core.transport
}

// ─── Slots ───────────────────────────────────────────────────────────

type slot struct {
	provider        string
	generation      int64
	connectionID    string
	revision        int64
	kind            string
	methodID        string
	instanceID      string
	label           string
	accountLabel    string
	createdAt       string
	routes          []string
	settings        map[string]string
	state           string
	renewal         string
	loggedOut       bool
	renewalInFlight JSONObject
	attempt         JSONObject
	verification    JSONObject
	previousIDs     []string
}

func emptySlot(provider string) slot {
	return slot{provider: provider, kind: "account", instanceID: "public", state: "ready", renewal: "refresh_token", settings: map[string]string{}}
}

func textOf(v any, fallback string) string {
	switch x := v.(type) {
	case nil:
		return fallback
	case string:
		return x
	default:
		return fmt.Sprint(x)
	}
}

func intOf(v any) int64 {
	n, _ := strconv.ParseInt(textOf(v, "0"), 10, 64)
	return n
}

func stringsOf(v any) []string {
	list, _ := v.([]any)
	out := make([]string, 0, len(list))
	for _, item := range list {
		out = append(out, textOf(item, ""))
	}
	return out
}

func slotFromRecord(provider string, r JSONObject) slot {
	s := emptySlot(provider)
	s.generation = intOf(r["generation"])
	s.connectionID, _ = r["connection_id"].(string)
	s.revision = intOf(r["revision"])
	s.kind = textOf(r["kind"], "account")
	s.methodID = textOf(r["method_id"], "")
	s.instanceID = textOf(r["instance_id"], "public")
	s.label = textOf(r["label"], "")
	s.accountLabel, _ = r["account_label"].(string)
	s.createdAt = textOf(r["created_at"], "")
	s.routes = stringsOf(r["routes"])
	if m, ok := r["settings"].(map[string]any); ok {
		for k, v := range m {
			s.settings[k] = textOf(v, "")
		}
	}
	s.state = textOf(r["state"], "ready")
	s.renewal = textOf(r["renewal"], "refresh_token")
	s.loggedOut, _ = r["logged_out"].(bool)
	s.renewalInFlight, _ = r["renewal_in_flight"].(map[string]any)
	s.attempt, _ = r["attempt"].(map[string]any)
	s.verification, _ = r["verification"].(map[string]any)
	s.previousIDs = stringsOf(r["previous_ids"])
	return s
}

func (s slot) record() JSONObject {
	var id any
	if s.connectionID != "" {
		id = s.connectionID
	}
	routes := make([]any, len(s.routes))
	for i, r := range s.routes {
		routes[i] = r
	}
	settings := JSONObject{}
	for k, v := range s.settings {
		settings[k] = v
	}
	r := JSONObject{
		"generation": strconv.FormatInt(s.generation, 10), "connection_id": id, "revision": strconv.FormatInt(s.revision, 10),
		"kind": s.kind, "method_id": s.methodID, "instance_id": s.instanceID, "label": s.label, "created_at": s.createdAt,
		"routes": routes, "settings": settings, "state": s.state, "renewal": s.renewal,
	}
	if s.accountLabel != "" {
		r["account_label"] = s.accountLabel
	}
	if s.loggedOut {
		r["logged_out"] = true
	}
	if s.renewalInFlight != nil {
		r["renewal_in_flight"] = s.renewalInFlight
	}
	if s.attempt != nil {
		r["attempt"] = s.attempt
	}
	if s.verification != nil {
		r["verification"] = s.verification
	}
	if len(s.previousIDs) > 0 {
		tail := s.previousIDs
		if len(tail) > 8 {
			tail = tail[len(tail)-8:]
		}
		ids := make([]any, len(tail))
		for i, x := range tail {
			ids[i] = x
		}
		r["previous_ids"] = ids
	}
	return r
}

func (s slot) connection() *Connection {
	if s.connectionID == "" {
		return nil
	}
	routes := s.routes
	if len(routes) == 0 {
		routes = []string{s.provider}
	}
	label := s.label
	if label == "" {
		label = s.provider
	}
	settings := map[string]string{}
	for k, v := range s.settings {
		settings[k] = v
	}
	return &Connection{
		ID: s.connectionID, Provider: s.provider, InstanceID: s.instanceID, Kind: s.kind, MethodID: s.methodID,
		Routes: append([]string{}, routes...), Label: label, CreatedAt: s.createdAt,
		IdentityGeneration: strconv.FormatInt(s.generation, 10), CredentialRevision: strconv.FormatInt(s.revision, 10),
		Settings: settings, AccountLabel: s.accountLabel,
	}
}

var legacyMethods = map[string]string{"xai": "device", "claude-code": "external:claude-code-cli", "openai-codex": "external:codex-cli"}

func slotsOf(document JSONObject) JSONObject {
	meta, _ := document[storeMetaKey].(map[string]any)
	slots, _ := meta["slots"].(map[string]any)
	return slots
}

func viewSlot(document JSONObject, provider string) (slot, JSONObject) {
	material, _ := document[provider].(map[string]any)
	if record, ok := slotsOf(document)[provider].(map[string]any); ok {
		return slotFromRecord(provider, record), material
	}
	if material != nil {
		oauth := material["type"] == "oauth"
		s := emptySlot(provider)
		s.generation, s.connectionID, s.revision = 1, "legacy-"+provider, 1
		s.kind, s.renewal = "api_key", "none"
		if oauth {
			s.kind, s.renewal = "account", "refresh_token"
		}
		s.methodID = legacyMethods[provider]
		if s.methodID == "" {
			s.methodID = "api_key"
		}
		s.label = provider + " (existing login)"
		s.routes = []string{provider}
		return s, material
	}
	return emptySlot(provider), nil
}

func putSlot(document JSONObject, s slot, material JSONObject) JSONObject {
	meta, ok := document[storeMetaKey].(map[string]any)
	if !ok {
		meta = JSONObject{"version": storeVersion, "slots": JSONObject{}}
		document[storeMetaKey] = meta
	}
	if _, ok := meta["version"]; !ok {
		meta["version"] = storeVersion
	}
	slots, ok := meta["slots"].(map[string]any)
	if !ok {
		slots = JSONObject{}
		meta["slots"] = slots
	}
	slots[s.provider] = s.record()
	if material == nil {
		delete(document, s.provider)
	} else {
		document[s.provider] = material
	}
	return document
}

// ─── Time and expiry ─────────────────────────────────────────────────

func isoMs(ms int64) string {
	return time.UnixMilli(ms - ms%1000).UTC().Format("2006-01-02T15:04:05Z")
}

// expiryOf: ms, or never (-1), or unknown (0, false).
func expiryOf(provider string, material JSONObject) (int64, string) {
	if isRecipe(material) {
		if materialStr(material, "type") == "external" {
			return 0, "unknown"
		}
		return 0, "never"
	}
	if _, ok := accountFlowFor(provider); !ok {
		return 0, "unknown"
	}
	if materialStr(material, "type") == "api_key" {
		return 0, "never" // a minted key (OpenRouter) is permanent
	}
	if _, isBool := material["expires"].(bool); isBool {
		return 0, "unknown"
	}
	f, ok := number(material["expires"])
	if !ok {
		return 0, "unknown"
	}
	return int64(f), "at"
}

func leadMs(material JSONObject) float64 {
	lifetime := -1.0
	if l, ok := number(material["lifetime_s"]); ok && l > 0 {
		lifetime = l * 1000
	} else if issued, ok := number(material["issued_at"]); ok {
		if expires, ok := number(material["expires"]); ok && expires != 0 {
			lifetime = expires - float64(int64(issued))
			if lifetime < 0 {
				lifetime = 0
			}
		}
	}
	if lifetime < 0 {
		return RenewalLeadMs
	}
	if lifetime/10 < RenewalLeadMs {
		return float64(int64(lifetime / 10))
	}
	return RenewalLeadMs
}

func renewable(s slot, material JSONObject) bool {
	if s.renewal == "none" || s.renewal == "recipe" {
		return false
	}
	return materialStr(material, "refresh") != ""
}

// ─── Errors ──────────────────────────────────────────────────────────

type opFields struct {
	reason, stage, recovery, commit, provider, connectionID, attemptID, methodID, providerCode string
	status                                                                                     int
}

func (f opFields) err(format string, args ...any) *Error {
	commit := f.commit
	if commit == "" {
		commit = "not_committed"
	}
	e := authOperation(fmt.Sprintf(format, args...), f.reason, f.stage, commit, f.recovery)
	e.Provider, e.ConnectionID, e.AttemptID, e.MethodID = f.provider, f.connectionID, f.attemptID, f.methodID
	e.Status, e.ProviderCode = f.status, f.providerCode
	return e
}

// ErrLoginCancelled is the cancellation outcome (AUTH-24 "cancelled"): a
// login stopped by its context, a closed prompt, CancelLogin, a logout or
// Close. It wraps context.Canceled — Go's own cancellation — and is never
// dressed up as an lm15 error.
var ErrLoginCancelled = fmt.Errorf("lm15: login cancelled: %w", context.Canceled)

type noUI struct{}

func (noUI) Prompt(context.Context, Prompt) (string, error) { return "", ErrPromptCancelled }
func (noUI) Notify(Notice)                                  {}

// ─── Discovery (AUTH-13): definitions only ───────────────────────────

// Providers is every provider this manager can connect.
func (a *Auth) Providers() []ProviderDescriptor {
	var out []ProviderDescriptor
	for _, id := range loginProviderIDs() {
		if d, ok := loginDescriptor(id, true); ok {
			out = append(out, d)
		}
	}
	return out
}

// Descriptor is one provider's AUTH-13 descriptor.
func (a *Auth) Descriptor(provider string) (ProviderDescriptor, error) {
	d, ok := loginDescriptor(provider, true)
	if !ok {
		return d, opFields{reason: "method_unavailable", stage: "discovery", recovery: "choose_method", provider: provider}.err("%q is not a provider lm15 can connect; see Auth.Providers()", provider)
	}
	return d, nil
}

// Methods lists how a provider can be connected.
func (a *Auth) Methods(provider string) ([]LoginMethod, error) {
	d, err := a.Descriptor(provider)
	return d.Methods, err
}

// ─── Inspection (AUTH-17: store reads only) ──────────────────────────

// Connections lists the saved connections.
func (a *Auth) Connections() ([]Connection, error) {
	document, err := a.store.Read()
	if err != nil {
		return nil, err
	}
	known := map[string]bool{}
	for _, id := range loginProviderIDs() {
		known[id] = true
	}
	keys := make([]string, 0, len(document))
	for k := range document {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []Connection
	for _, key := range keys {
		if key == storeMetaKey || !known[key] {
			continue
		}
		s, _ := viewSlot(document, key)
		if c := s.connection(); c != nil {
			out = append(out, *c)
		}
	}
	slots := slotsOf(document)
	names := make([]string, 0, len(slots))
	for k := range slots {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, key := range names {
		if _, present := document[key]; present {
			continue
		}
		if record, ok := slots[key].(map[string]any); ok {
			if c := slotFromRecord(key, record).connection(); c != nil {
				out = append(out, *c)
			}
		}
	}
	return out, nil
}

// Status is presence, usability and last verification of a provider's slot (AUTH-24).
func (a *Auth) Status(provider string) (ConnectionStatus, error) {
	d, err := a.Descriptor(provider)
	if err != nil {
		return ConnectionStatus{}, err
	}
	document, err := a.store.Read()
	if err != nil {
		return ConnectionStatus{}, err
	}
	s, material := viewSlot(document, d.ID)
	var verification *Verification
	if s.verification != nil {
		verification = &Verification{Result: textOf(s.verification["result"], ""), CheckedAt: textOf(s.verification["checked_at"], ""), Check: textOf(s.verification["check"], ""), Detail: textOf(s.verification["detail"], "")}
	}
	c := s.connection()
	if c == nil {
		st := ConnectionStatus{Provider: d.ID, Presence: "absent", Usability: "unknown", LoggedOut: s.loggedOut}
		if s.loggedOut {
			st.Detail = "signed out; sign in again or pass a key explicitly"
		}
		return st, nil
	}
	usability, expires, detail := a.usability(s, material)
	return ConnectionStatus{Provider: d.ID, Presence: "saved", Usability: usability, Connection: c, ExpiresAt: expires, Verification: verification, Detail: detail}, nil
}

func (a *Auth) usability(s slot, material JSONObject) (string, string, string) {
	if s.state == "needs_login" {
		return "needs_login", "", "the provider rejected the saved credential; sign in again"
	}
	if s.state == "indeterminate" || s.renewalInFlight != nil {
		return "indeterminate", "", "a renewal was interrupted; sign in again to be safe"
	}
	if material == nil {
		return "needs_login", "", "credential material is missing"
	}
	expiry, kind := expiryOf(s.provider, material)
	switch kind {
	case "never":
		return "ready", "never", ""
	case "unknown":
		if materialStr(material, "type") == "external" {
			return "ready", "unknown", ""
		}
		return "unknown", "unknown", ""
	}
	if float64(a.now()) >= float64(expiry)-leadMs(material) {
		if !renewable(s, material) {
			return "needs_login", isoMs(expiry), "expired and not renewable"
		}
		return "renewal_due", isoMs(expiry), ""
	}
	return "ready", isoMs(expiry), ""
}

// ─── Login (AUTH-16/17/18/19) ────────────────────────────────────────

// LoginOptions configures one login.
type LoginOptions struct {
	Method          string // "" = ask the UI when more than one selectable method remains (AUTH-13.6)
	UI              AuthUI
	Settings        map[string]string
	Answers         map[string]string
	Replace         string // the connection id being replaced; without it an occupied slot is connection_exists
	Lifetime        time.Duration
	AllowUnverified bool // unverified methods run only with this (AUTH-13.5)
}

// Login runs one login to completion and saves the connection; it is saved
// before Login returns. Cancelling ctx (or CancelLogin) returns ErrLoginCancelled.
func (a *Auth) Login(ctx context.Context, provider string, opts LoginOptions) (Connection, error) {
	if err := a.checkOpen(); err != nil {
		return Connection{}, err
	}
	d, err := a.Descriptor(provider)
	if err != nil {
		return Connection{}, err
	}
	provider = d.ID
	if opts.UI == nil {
		opts.UI = noUI{}
	}
	chosen, err := a.chooseMethod(ctx, d, opts.Method, opts.UI, opts.AllowUnverified)
	if err != nil {
		return Connection{}, err
	}
	answers := map[string]string{}
	for k, v := range opts.Answers {
		answers[k] = v
	}
	settings := map[string]string{}
	for k, v := range opts.Settings {
		settings[k] = v
	}
	lifetime := float64(opts.Lifetime.Milliseconds())
	if lifetime <= 0 {
		lifetime = attemptLifetimeMs
	}
	attemptID := "at_" + randomBase64URL(16)
	if err := a.store.Reserve(ctx); err != nil {
		return Connection{}, err
	}
	expected, err := a.reserve(ctx, provider, attemptID, opts.Replace, lifetime)
	if err != nil {
		return Connection{}, err
	}
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	a.core.mu.Lock()
	a.core.active[provider] = cancel
	a.core.mu.Unlock()
	defer func() {
		a.core.mu.Lock()
		delete(a.core.active, provider)
		a.core.mu.Unlock()
	}()
	lc := &loginContext{
		ctx: attemptCtx, ui: opts.UI, provider: provider, deadline: a.core.monotonic() + lifetime,
		monotonic: a.core.monotonic, wallClock: a.core.wallClock, sleep: a.core.sleep, transport: a.transport(),
		listenerAvailable: runtime.GOOS != "js", // a page is the redirect target itself
	}
	result, err := a.runLogin(lc, provider, chosen, answers, settings, attemptID, lifetime)
	if err == nil {
		var c Connection
		if c, err = a.commit(ctx, provider, attemptID, expected, chosen, result, settings); err == nil {
			return c, nil
		}
	}
	// Any exit without a saved connection ends the attempt: its reservation
	// must not outlive it (a no-op when it is no longer ours).
	a.release(context.Background(), provider, attemptID)
	return Connection{}, err
}

func (a *Auth) runLogin(c *loginContext, provider string, chosen LoginMethod, answers, settings map[string]string, attemptID string, lifetime float64) (flowResult, error) {
	for _, field := range chosen.Fields {
		if _, ok := answers[field.ID]; ok {
			continue
		}
		p := Prompt{Type: "text", FieldID: field.ID, Label: field.Label}
		switch {
		case !field.Required && field.Type != "select":
		case field.Type == "secret":
			p.Type = "secret"
		case field.Type == "select":
			p.Type, p.Options = "select", field.Options
		}
		answer, err := c.prompt(p)
		if err != nil {
			return flowResult{}, a.loginFailure(err, provider, attemptID, chosen.ID, lifetime, false)
		}
		answers[field.ID] = answer
	}
	var result flowResult
	var err error
	if flow, ok := accountFlowFor(provider); ok && chosen.Kind == "account" && !isExternalMethod(chosen.ID) {
		result, err = flow.login(c, chosen.ID, settings, answers)
	} else {
		result, err = recipeLogin(provider, chosen.ID, answers, settings, a.core.home)
	}
	if err == nil {
		err = c.check() // AUTH-18: never kept past the deadline
	}
	if err != nil {
		return flowResult{}, a.loginFailure(err, provider, attemptID, chosen.ID, lifetime, true)
	}
	return result, nil
}

func isExternalMethod(id string) bool { return len(id) > 9 && id[:9] == "external:" }

func (a *Auth) loginFailure(err error, provider, attemptID, methodID string, lifetime float64, inFlow bool) error {
	var den *loginDenied
	var net *networkFailure
	switch {
	case errors.Is(err, errLoginCancelled):
		return ErrLoginCancelled
	case errors.Is(err, errLoginExpired):
		return opFields{reason: "login_expired", stage: "polling", recovery: "restart_login", provider: provider, attemptID: attemptID, methodID: methodID}.err("%s: the sign-in was not completed within %d minutes; start again", provider, int(lifetime/60000))
	case errors.As(err, &den):
		return opFields{reason: "login_denied", stage: den.stage, recovery: "restart_login", provider: provider, attemptID: attemptID, methodID: methodID, status: den.status, providerCode: den.providerCode}.err("%s: %s", provider, den.message)
	case errors.As(err, &net):
		if net.uncertain && inFlow {
			return opFields{reason: "indeterminate", stage: "exchange", recovery: "restart_login", provider: provider, attemptID: attemptID, methodID: methodID}.err("%s: the network failed after the authorization code may have been sent; the code is one-use, so sign in again rather than retry", provider)
		}
		return net.err
	}
	return err
}

func (a *Auth) chooseMethod(ctx context.Context, d ProviderDescriptor, method string, ui AuthUI, allowUnverified bool) (LoginMethod, error) {
	unavailable := func(methodID, format string, args ...any) error {
		return opFields{reason: "method_unavailable", stage: "discovery", recovery: "choose_method", provider: d.ID, methodID: methodID}.err(format, args...)
	}
	if method != "" {
		chosen, ok := d.Method(method)
		if !ok {
			return chosen, unavailable("", "%s: no login method %q; see Auth.Methods(%q)", d.ID, method, d.ID)
		}
		if chosen.Availability == "unavailable" {
			return chosen, unavailable(method, "%s: method %q is unavailable: %s", d.ID, method, chosen.Reason)
		}
		if chosen.Availability == "unverified" && !allowUnverified {
			return chosen, unavailable(method, "%s: method %q has no live receipt yet (%s); set AllowUnverified to try it knowing that", d.ID, method, chosen.Reason)
		}
		return chosen, nil
	}
	var candidates []LoginMethod
	for _, m := range d.Methods {
		if m.Availability == "supported" || (allowUnverified && m.Availability == "unverified") {
			candidates = append(candidates, m)
		}
	}
	switch len(candidates) {
	case 0:
		return LoginMethod{}, unavailable("", "%s: no selectable login method here", d.ID)
	case 1:
		return candidates[0], nil
	}
	options := make([]SelectOption, len(candidates))
	for i, m := range candidates {
		note := m.BillingNote
		if note == "" {
			note = m.Reason
		}
		options[i] = SelectOption{ID: m.ID, Label: m.Label, Description: note}
	}
	answer, err := ui.Prompt(ctx, Prompt{Type: "select", FieldID: "method", Label: "How do you want to connect to " + d.Label + "?", Options: options})
	if err != nil {
		return LoginMethod{}, ErrLoginCancelled
	}
	for _, m := range candidates {
		if m.ID == answer {
			return m, nil
		}
	}
	return LoginMethod{}, opFields{reason: "invalid_login_state", stage: "interaction", recovery: "choose_method", provider: d.ID}.err("%s: the UI answered %q, which is not one of the offered method ids", d.ID, answer)
}

func (a *Auth) reserve(ctx context.Context, provider, attemptID, replace string, lifetimeMs float64) (int64, error) {
	now := float64(a.now()) / 1000
	document, err := mutateStore(ctx, a.store, func(doc JSONObject) (JSONObject, error) {
		s, material := viewSlot(doc, provider)
		if s.attempt != nil && s.attempt["id"] != attemptID {
			started, _ := number(s.attempt["started_at_s"])
			budget, ok := number(s.attempt["lifetime_s"])
			if !ok {
				budget = attemptLifetimeMs / 1000
			}
			if now-started < budget {
				return nil, opFields{reason: "login_in_progress", stage: "reservation", recovery: "inspect_attempt", provider: provider, attemptID: textOf(s.attempt["id"], "")}.err("%s: another sign-in is already in progress in this scope; finish it or cancel it (Auth.CancelLogin)", provider)
			}
		}
		if s.connectionID != "" && replace == "" {
			return nil, opFields{reason: "connection_exists", stage: "reservation", recovery: "select_connection", provider: provider, connectionID: s.connectionID}.err("%s: a connection is already saved (%s); pass Replace with that id to replace it, or logout first", provider, s.connectionID)
		}
		if replace != "" && s.connectionID != replace {
			return nil, opFields{reason: "connection_changed", stage: "reservation", recovery: "select_connection", provider: provider, connectionID: s.connectionID}.err("%s: replace=%q does not name the current connection; select again", provider, replace)
		}
		s.attempt = JSONObject{"id": attemptID, "expected_generation": strconv.FormatInt(s.generation, 10), "started_at_s": now, "lifetime_s": floatLexeme(lifetimeMs / 1000)}
		return putSlot(doc, s, material), nil
	})
	if err != nil {
		return 0, err
	}
	s, _ := viewSlot(document, provider)
	return s.generation, nil
}

func (a *Auth) release(ctx context.Context, provider, attemptID string) {
	_, _ = mutateStore(ctx, a.store, func(doc JSONObject) (JSONObject, error) {
		s, material := viewSlot(doc, provider)
		if s.attempt == nil || s.attempt["id"] != attemptID {
			return nil, nil
		}
		s.attempt = nil
		return putSlot(doc, s, material), nil
	}) // releasing a reservation must not mask the real failure
}

func (a *Auth) commit(ctx context.Context, provider, attemptID string, expected int64, method LoginMethod, result flowResult, settings map[string]string) (Connection, error) {
	created := isoMs(a.now())
	connectionID := "cn_" + randomBase64URL(12)
	d, err := a.Descriptor(provider)
	if err != nil {
		return Connection{}, err
	}
	document, err := mutateStore(ctx, a.store, func(doc JSONObject) (JSONObject, error) {
		s, _ := viewSlot(doc, provider)
		if s.attempt == nil || s.attempt["id"] != attemptID {
			return nil, opFields{reason: "invalid_login_state", stage: "persistence", recovery: "restart_login", provider: provider, attemptID: attemptID}.err("%s: this sign-in was cancelled before it could be saved", provider)
		}
		if s.generation != expected {
			return nil, opFields{reason: "connection_changed", stage: "persistence", recovery: "select_connection", provider: provider, attemptID: attemptID}.err("%s: the saved connection changed while you were signing in; select again", provider)
		}
		merged := map[string]string{}
		if s.connectionID != "" {
			for k, v := range s.settings {
				merged[k] = v
			}
		}
		for k, v := range settings {
			merged[k] = v
		}
		for k, v := range result.settings {
			merged[k] = v
		}
		previous := append([]string{}, s.previousIDs...)
		if s.connectionID != "" {
			previous = append(previous, s.connectionID)
		}
		routes := d.Routes
		if len(routes) == 0 {
			routes = []string{provider}
		}
		next := emptySlot(provider)
		next.generation, next.connectionID, next.revision = s.generation+1, connectionID, 1
		next.kind, next.methodID, next.label, next.accountLabel = method.Kind, method.ID, result.label, result.accountLabel
		next.createdAt, next.routes, next.settings, next.renewal, next.previousIDs = created, routes, merged, result.renewal, previous
		return putSlot(doc, next, copyDocument(result.material).(map[string]any)), nil
	})
	if err != nil {
		var e *Error
		if errors.As(err, &e) && e.Kind == KindAuthOperation {
			return Connection{}, err
		}
		// A grant may exist at the provider; nothing is revoked as compensation (AUTH-19).
		return Connection{}, opFields{reason: "storage_unavailable", stage: "persistence", recovery: "repair_storage", provider: provider, attemptID: attemptID}.err("%s: signed in, but the credential could not be saved; repair the store and sign in again", provider)
	}
	s, _ := viewSlot(document, provider)
	return *s.connection(), nil
}

// CancelLogin durably cancels the slot's active attempt: "cancelled",
// "complete" when a commit already won (undo is Logout), or "none".
func (a *Auth) CancelLogin(ctx context.Context, provider string) (string, error) {
	d, err := a.Descriptor(provider)
	if err != nil {
		return "", err
	}
	outcome := "none"
	if _, err := mutateStore(ctx, a.store, func(doc JSONObject) (JSONObject, error) {
		s, material := viewSlot(doc, d.ID)
		if s.attempt == nil {
			if s.connectionID != "" {
				outcome = "complete"
			}
			return nil, nil
		}
		s.attempt = nil
		outcome = "cancelled"
		return putSlot(doc, s, material), nil
	}); err != nil {
		return "", err
	}
	// The durable record first (AUTH-19), then the running attempt here.
	a.core.mu.Lock()
	if cancel := a.core.active[d.ID]; cancel != nil {
		cancel()
	}
	a.core.mu.Unlock()
	return outcome, nil
}

// ─── Setup without a provider round-trip (AUTH-17) ───────────────────

// SetAPIKey saves a literal key (no interpolation, no verification).
func (a *Auth) SetAPIKey(ctx context.Context, provider, key, replace string) (Connection, error) {
	if len(trimSpace(key)) == 0 {
		return Connection{}, opFields{reason: "interaction_required", stage: "interaction", recovery: "provide_input", provider: provider}.err("SetAPIKey: the key is empty")
	}
	return a.Configure(ctx, provider, "api_key", map[string]string{"key": key}, nil, replace)
}

func trimSpace(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t' || s[0] == '\n' || s[0] == '\r') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t' || s[len(s)-1] == '\n' || s[len(s)-1] == '\r') {
		s = s[:len(s)-1]
	}
	return s
}

// Configure saves a recipe connection: env, external:<source>, cloud, local
// or api_key. No credential is acquired and nothing is verified.
func (a *Auth) Configure(ctx context.Context, provider, method string, answers, settings map[string]string, replace string) (Connection, error) {
	if err := a.checkOpen(); err != nil {
		return Connection{}, err
	}
	d, err := a.Descriptor(provider)
	if err != nil {
		return Connection{}, err
	}
	provider = d.ID
	chosen, ok := d.Method(method)
	if !ok {
		return Connection{}, opFields{reason: "method_unavailable", stage: "discovery", recovery: "choose_method", provider: provider}.err("%s: no setup method %q; see Auth.Methods(%q)", provider, method, provider)
	}
	if chosen.Flow != "form" && chosen.Flow != "source_recipe" {
		return Connection{}, opFields{reason: "method_unavailable", stage: "discovery", recovery: "choose_method", provider: provider}.err("%s: %q is an interactive login; use Auth.Login", provider, method)
	}
	if answers == nil {
		answers = map[string]string{}
	}
	if settings == nil {
		settings = map[string]string{}
	}
	for _, field := range chosen.Fields {
		if field.Required && answers[field.ID] == "" {
			return Connection{}, opFields{reason: "interaction_required", stage: "interaction", recovery: "provide_input", provider: provider}.err("%s: %q needs %q", provider, method, field.ID)
		}
	}
	attemptID := "at_" + randomBase64URL(16)
	if err := a.store.Reserve(ctx); err != nil {
		return Connection{}, err
	}
	expected, err := a.reserve(ctx, provider, attemptID, replace, 60_000)
	if err != nil {
		return Connection{}, err
	}
	result, err := recipeLogin(provider, method, answers, settings, a.core.home)
	if err != nil {
		a.release(ctx, provider, attemptID)
		var den *loginDenied
		if errors.As(err, &den) {
			return Connection{}, opFields{reason: "login_denied", stage: "interaction", recovery: "provide_input", provider: provider}.err("%s: %s", provider, den.message)
		}
		return Connection{}, err
	}
	return a.commit(ctx, provider, attemptID, expected, chosen, result, settings)
}

// ─── Logout (AUTH-19) ────────────────────────────────────────────────

// Logout forgets the connection locally: material removed, generation
// bumped, pending attempt cancelled, a marker kept so a restart cannot fall
// back to an ambient key (R3). Never calls a provider's revoke endpoint;
// never touches another tool's file.
func (a *Auth) Logout(ctx context.Context, providerOrConnection string) (ForgetResult, error) {
	if err := a.checkOpen(); err != nil {
		return ForgetResult{}, err
	}
	provider, target, err := a.resolveTarget(providerOrConnection)
	if err != nil {
		return ForgetResult{}, err
	}
	forgot, generation, routes := false, int64(0), []string(nil)
	if _, err := mutateStore(ctx, a.store, func(doc JSONObject) (JSONObject, error) {
		s, _ := viewSlot(doc, provider)
		if target != "" && s.connectionID != target {
			generation, routes = s.generation, s.routes
			return nil, nil // idempotent: a newer id occupying the slot is untouched
		}
		if s.connectionID == "" && s.attempt == nil {
			generation, routes = s.generation, s.routes
			return nil, nil
		}
		previous := append([]string{}, s.previousIDs...)
		if s.connectionID != "" {
			previous = append(previous, s.connectionID)
		}
		next := emptySlot(provider)
		next.generation, next.kind, next.methodID = s.generation+1, s.kind, s.methodID
		next.routes = s.routes
		if len(next.routes) == 0 {
			next.routes = []string{provider}
		}
		next.renewal, next.loggedOut, next.previousIDs = "none", true, previous
		forgot, generation, routes = true, next.generation, next.routes
		a.core.mu.Lock()
		if cancel := a.core.active[provider]; cancel != nil {
			cancel()
		}
		a.core.mu.Unlock()
		return putSlot(doc, next, nil), nil
	}); err != nil {
		return ForgetResult{}, err
	}
	if len(routes) == 0 {
		routes = []string{provider}
	}
	return ForgetResult{Provider: provider, Forgot: forgot, Routes: routes, IdentityGeneration: strconv.FormatInt(generation, 10)}, nil
}

func (a *Auth) resolveTarget(target string) (string, string, error) {
	if len(target) > 3 && (target[:3] == "cn_" || (len(target) > 7 && target[:7] == "legacy-")) {
		connections, err := a.Connections()
		if err != nil {
			return "", "", err
		}
		for _, c := range connections {
			if c.ID == target {
				return c.Provider, c.ID, nil
			}
		}
		document, err := a.store.Read()
		if err != nil {
			return "", "", err
		}
		for key, record := range slotsOf(document) {
			r, _ := record.(map[string]any)
			for _, id := range stringsOf(r["previous_ids"]) {
				if id == target {
					return key, target, nil
				}
			}
		}
		return "", "", opFields{reason: "attempt_unavailable", stage: "resolution", recovery: "select_connection"}.err("no saved connection has that id")
	}
	d, err := a.Descriptor(target)
	return d.ID, "", err
}

// ─── Verification (AUTH-17) ──────────────────────────────────────────

// Verify is an explicit, non-inference check: resolve (renewing if due) and
// list models on the route. Not universal; possibly metered by the provider.
func (a *Auth) Verify(ctx context.Context, provider string) (Verification, error) {
	if err := a.checkOpen(); err != nil {
		return Verification{}, err
	}
	d, err := a.Descriptor(provider)
	if err != nil {
		return Verification{}, err
	}
	def, ok := Providers[d.ID]
	if !ok || !def.Access.Supports.Models {
		return Verification{Result: "unverified", Check: "models", Detail: "this route has no safe non-inference check"}, nil
	}
	router, err := NewRouterWithConfig(RouterConfig{Auth: a})
	if err != nil {
		return Verification{}, err
	}
	defer router.Close()
	checked := isoMs(a.now())
	lm, err := router.LM(d.ID + ":verify")
	if err != nil {
		return Verification{}, err
	}
	lister, ok := lm.(interface {
		ListModels(context.Context) ([]ModelInfo, error)
	})
	if !ok {
		return Verification{Result: "unverified", Check: "models", Detail: "this route has no safe non-inference check"}, nil
	}
	result := Verification{Result: "valid", CheckedAt: checked, Check: "models"}
	if _, err := lister.ListModels(ctx); err != nil {
		if !IsKind(err, KindAuth) {
			return Verification{}, err
		}
		result = Verification{Result: "rejected", CheckedAt: checked, Check: "models", Detail: CodeAuth}
	}
	_, _ = mutateStore(ctx, a.store, func(doc JSONObject) (JSONObject, error) {
		s, material := viewSlot(doc, d.ID)
		if s.connectionID == "" {
			return nil, nil
		}
		s.verification = JSONObject{"result": result.Result, "checked_at": result.CheckedAt, "check": result.Check, "detail": nilIfEmpty(result.Detail)}
		return putSlot(doc, s, material), nil
	})
	return result, nil
}

// ─── Request-time resolution (AUTH-15/20) ────────────────────────────

// RequestAuth is what a request on provider sends now: the saved
// connection's credential, renewed under the lock if due. A pinned view
// (WithPin) or pinned checks the selection: a mismatch is connection_changed,
// never a silent rebind (AUTH-20.1).
func (a *Auth) RequestAuth(ctx context.Context, provider string, pinned *[2]string) (RequestAuth, error) {
	d, err := a.Descriptor(provider)
	if err != nil {
		return RequestAuth{}, err
	}
	provider = d.ID
	if a.pin != nil {
		pinned = a.pin
	}
	document, err := a.store.Read()
	if err != nil {
		return RequestAuth{}, err
	}
	s, material := viewSlot(document, provider)
	// A sibling may be renewing right now (it holds the lock), or may have
	// died mid-exchange. Only the lock can tell: wait for it, re-read, reuse
	// its result; a marker still there once we hold the lock is an
	// interrupted renewal (AUTH-20.4).
	if s.renewalInFlight != nil && s.state != "indeterminate" {
		return a.renew(ctx, provider, pinned)
	}
	if err := a.checkSelected(provider, s, material, pinned); err != nil {
		return RequestAuth{}, err
	}
	if expiry, kind := expiryOf(provider, material); kind == "at" && float64(a.now()) >= float64(expiry)-leadMs(material) {
		return a.renew(ctx, provider, pinned)
	}
	return a.authFrom(ctx, provider, material, s)
}

// Selection is the saved connection's non-secret request shape, read without
// renewal (a router's LM): the selection checks, then the base URL, headers,
// account id or named identity.
func (a *Auth) Selection(provider string) (Connection, RequestAuth, error) {
	d, err := a.Descriptor(provider)
	if err != nil {
		return Connection{}, RequestAuth{}, err
	}
	document, err := a.store.Read()
	if err != nil {
		return Connection{}, RequestAuth{}, err
	}
	s, material := viewSlot(document, d.ID)
	if err := a.checkSelected(d.ID, s, material, a.pin); err != nil {
		return Connection{}, RequestAuth{}, err
	}
	var shape RequestAuth
	switch {
	case materialStr(material, "type") == "external":
		shape = externalPeek(materialStr(material, "source"), a.core.home)
	case isRecipe(material):
		shape, err = recipeRequestAuth(material, func(string) string { return "unused" })
	default:
		if flow, ok := accountFlowFor(d.ID); ok {
			shape, err = flow.requestAuth(material, s.settings)
		}
	}
	if err != nil {
		return Connection{}, RequestAuth{}, a.selectionFailure(err, d.ID, s)
	}
	shape.Credential, shape.CredentialKind = "", ""
	return *s.connection(), shape, nil
}

func (a *Auth) selectionFailure(err error, provider string, s slot) error {
	var den *loginDenied
	if errors.As(err, &den) {
		return opFields{reason: "login_required", stage: "resolution", recovery: "restart_login", provider: provider, connectionID: s.connectionID}.err("%s: %s", provider, den.message)
	}
	var net *networkFailure
	if errors.As(err, &net) {
		return net.err
	}
	return err
}

func (a *Auth) checkSelected(provider string, s slot, material JSONObject, pinned *[2]string) error {
	if pinned != nil && (s.connectionID != pinned[0] || strconv.FormatInt(s.generation, 10) != pinned[1]) {
		if s.connectionID == "" {
			return opFields{reason: "login_required", stage: "resolution", recovery: "restart_login", provider: provider, connectionID: pinned[0]}.err("%s: the connection this client was bound to was signed out; connect again", provider)
		}
		return opFields{reason: "connection_changed", stage: "resolution", recovery: "select_connection", provider: provider, connectionID: pinned[0]}.err("%s: the saved connection was replaced after this client was bound; connect again", provider)
	}
	if s.connectionID == "" || material == nil {
		if s.loggedOut {
			return opFields{reason: "login_required", stage: "resolution", recovery: "restart_login", provider: provider}.err("%s: signed out; sign in again (Auth.Login) or pass a key explicitly (APIKeys)", provider)
		}
		return opFields{reason: "login_required", stage: "resolution", recovery: "restart_login", provider: provider}.err("%s: no saved connection in this scope; sign in with Auth.Login or Connect", provider)
	}
	if s.state == "needs_login" {
		return opFields{reason: "login_required", stage: "resolution", recovery: "restart_login", provider: provider, connectionID: s.connectionID}.err("%s: the saved credential was rejected by the provider; sign in again", provider)
	}
	if s.state == "indeterminate" || s.renewalInFlight != nil {
		return opFields{reason: "indeterminate", stage: "resolution", recovery: "restart_login", commit: "unknown", provider: provider, connectionID: s.connectionID}.err("%s: a credential renewal was interrupted and its outcome is unknown; sign in again rather than reuse a possibly consumed token", provider)
	}
	return nil
}

func (a *Auth) authFrom(ctx context.Context, provider string, material JSONObject, s slot) (RequestAuth, error) {
	var auth RequestAuth
	var err error
	switch {
	case materialStr(material, "type") == "external":
		auth, err = externalRequestAuth(ctx, materialStr(material, "source"), a.core.home)
	case isRecipe(material):
		auth, err = recipeRequestAuth(material, a.core.env)
	default:
		flow, ok := accountFlowFor(provider)
		if !ok {
			return RequestAuth{}, denied("no flow owns this connection")
		}
		auth, err = flow.requestAuth(material, s.settings)
	}
	if err != nil {
		return RequestAuth{}, a.selectionFailure(err, provider, s)
	}
	return auth, nil
}

// renew (AUTH-20.4): lock, re-read, reuse a sibling's fresh result, else
// mark in flight, exchange, write — all under the lock.
func (a *Auth) renew(ctx context.Context, provider string, pinned *[2]string) (RequestAuth, error) {
	guard, err := a.store.Lock(ctx)
	if err != nil {
		return RequestAuth{}, err
	}
	defer guard.Unlock()
	document, err := guard.Read()
	if err != nil {
		return RequestAuth{}, err
	}
	s, material := viewSlot(document, provider)
	if err := a.checkSelected(provider, s, material, pinned); err != nil {
		return RequestAuth{}, err
	}
	expiry, kind := expiryOf(provider, material)
	if kind != "at" || float64(a.now()) < float64(expiry)-leadMs(material) {
		return a.authFrom(ctx, provider, material, s) // a sibling renewed while we waited
	}
	connectionID := s.connectionID
	mark := func(state string, dropMaterial, keepMarker bool) error {
		s.state = state
		if !keepMarker {
			s.renewalInFlight = nil
		}
		keep := material
		if dropMaterial {
			keep = nil
		}
		return guard.Write(putSlot(copyDocument(document).(map[string]any), s, keep))
	}
	if !renewable(s, material) {
		if err := mark("needs_login", true, false); err != nil {
			return RequestAuth{}, err
		}
		return RequestAuth{}, opFields{reason: "credential_rejected", stage: "renewal", recovery: "restart_login", commit: "committed", provider: provider, connectionID: connectionID}.err("%s: the saved credential expired and cannot be renewed; sign in again", provider)
	}
	// Durable in-flight marker before the possibly rotating exchange.
	s.renewalInFlight = JSONObject{"started_at": isoMs(a.now()), "revision": strconv.FormatInt(s.revision, 10)}
	if err := guard.Write(putSlot(copyDocument(document).(map[string]any), s, material)); err != nil {
		return RequestAuth{}, err
	}
	flow, _ := accountFlowFor(provider)
	lc := &loginContext{
		ctx: ctx, ui: noUI{}, provider: provider, deadline: a.core.monotonic() + 60_000,
		monotonic: a.core.monotonic, wallClock: a.core.wallClock, sleep: a.core.sleep, transport: a.transport(),
	}
	result, err := flow.renew(lc, material, s.settings)
	if err != nil {
		var den *loginDenied
		var net *networkFailure
		var e *Error
		switch {
		case errors.As(err, &den):
			if werr := mark("needs_login", true, false); werr != nil {
				return RequestAuth{}, werr
			}
			return RequestAuth{}, opFields{reason: "credential_rejected", stage: "renewal", recovery: "restart_login", commit: "committed", provider: provider, connectionID: connectionID, status: den.status, providerCode: den.providerCode}.err("%s: renewal failed (%s); sign in again", provider, den.message)
		case errors.As(err, &e) && (e.Kind == KindRateLimit || e.Kind == KindServer):
			if werr := mark("ready", false, false); werr != nil { // known safe: keep the credential
				return RequestAuth{}, werr
			}
			return RequestAuth{}, err
		case errors.As(err, &net) && net.uncertain:
			if werr := mark("indeterminate", false, true); werr != nil {
				return RequestAuth{}, werr
			}
			return RequestAuth{}, opFields{reason: "indeterminate", stage: "renewal", recovery: "restart_login", commit: "unknown", provider: provider, connectionID: connectionID}.err("%s: the renewal exchange timed out after it may have reached the provider; a rotated token cannot be spent twice, so sign in again", provider)
		case errors.As(err, &net):
			if werr := mark("ready", false, false); werr != nil {
				return RequestAuth{}, werr
			}
			return RequestAuth{}, net.err
		}
		if werr := mark("indeterminate", false, true); werr != nil {
			return RequestAuth{}, werr
		}
		return RequestAuth{}, err
	}
	s.renewalInFlight, s.state = nil, "ready"
	s.revision++
	if result.accountLabel != "" {
		s.accountLabel = result.accountLabel
	}
	if err := guard.Write(putSlot(copyDocument(document).(map[string]any), s, result.material)); err != nil {
		return RequestAuth{}, err
	}
	return a.authFrom(ctx, provider, result.material, s)
}

// Close cancels logins this manager is running; never a logout.
func (a *Auth) Close() {
	a.core.mu.Lock()
	defer a.core.mu.Unlock()
	a.core.closed = true
	for _, cancel := range a.core.active {
		cancel()
	}
}

func (a *Auth) checkOpen() error {
	a.core.mu.Lock()
	defer a.core.mu.Unlock()
	if a.core.closed {
		return opFields{reason: "storage_unavailable", stage: "resolution", recovery: "operator_action"}.err("this Auth was closed")
	}
	return nil
}
