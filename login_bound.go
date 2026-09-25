package lm15

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"iter"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ─── Model choices and the bound client (AUTH-23) ────────────────────
//
// Port of lm15-python lm15/login/bound.py. A BoundClient is what Connect
// returns: one connection id, one generation, one route, one model. It
// follows that connection's renewals and nothing else — a replacement or a
// logout makes it fail connection_changed / login_required instead of
// quietly switching who pays (R4). It keeps no conversation, retries nothing.

func capabilityOf(info ModelInfo, name string) string {
	if info.Inference == nil {
		return "unknown"
	}
	switch name {
	case "reasoning":
		if info.Inference.SupportsReasoning {
			return "supported"
		}
		return "unsupported"
	case "vision":
		for _, m := range info.Inference.InputModalities {
			if m == "image" {
				return "supported"
			}
		}
		return "unsupported"
	}
	return "unknown" // structured output is not recorded in ModelInfo; say so
}

// ModelChoices lists the models the saved connection on provider can select,
// fetched now with the saved credential (renewing if due; no inference).
// With capability, only supported ones unless includeUnknown.
func ModelChoices(ctx context.Context, auth *Auth, provider, capability string, includeUnknown bool, config RouterConfig) ([]ModelChoice, error) {
	if capability != "" && capability != "reasoning" && capability != "vision" && capability != "structured-output" {
		return nil, valueErrorf("capability must be reasoning, vision or structured-output, not %q", capability)
	}
	status, err := auth.Status(provider)
	if err != nil {
		return nil, err
	}
	if status.Connection == nil {
		return nil, opFields{reason: "login_required", stage: "catalog", recovery: "restart_login", provider: provider}.err("%s: no saved connection to list models for", provider)
	}
	config.Auth = auth
	router, err := NewRouterWithConfig(config)
	if err != nil {
		return nil, err
	}
	defer router.Close()
	lm, err := router.LM(status.Connection.Provider + ":catalog")
	if err != nil {
		return nil, err
	}
	infos, err := lm.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	fetched := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	var out []ModelChoice
	for _, info := range infos {
		choice := ModelChoice{Provider: status.Connection.Provider, Model: info.ID, ConnectionID: status.Connection.ID, Source: "provider", FetchedAt: fetched}
		if capability != "" {
			choice.Capability, choice.State = capability, capabilityOf(info, capability)
			if choice.State == "unsupported" || (choice.State == "unknown" && !includeUnknown) {
				continue
			}
		}
		out = append(out, choice)
	}
	return out, nil
}

// BoundClient is one connection, one model; canonical requests and responses.
type BoundClient struct {
	Auth      *Auth
	Selection ModelSelection
	router    *LMRouter
}

// NewBoundClient pins auth to selection (config.Auth must be nil or auth).
func NewBoundClient(auth *Auth, selection ModelSelection, config RouterConfig) (*BoundClient, error) {
	if config.Auth != nil && config.Auth != auth {
		return nil, valueErrorf("RouterConfig.Auth must be this client's Auth or nil")
	}
	config.Auth = auth.WithPin(selection.ConnectionID, selection.IdentityGeneration)
	router, err := NewRouterWithConfig(config)
	if err != nil {
		return nil, err
	}
	return &BoundClient{Auth: auth, Selection: selection, router: router}, nil
}

// Routed is "provider:model".
func (b *BoundClient) Routed() string { return b.Selection.Routed() }

// String never renders a secret.
func (b *BoundClient) String() string {
	return "BoundClient(" + b.Routed() + ", connection=" + b.Selection.ConnectionID + ")"
}

func (b *BoundClient) coerce(req *Request) (*Request, error) {
	if req.Model != b.Routed() && req.Model != b.Selection.Model {
		e := opFields{reason: "selection_mismatch", stage: "dispatch", recovery: "none", provider: b.Selection.Provider}.err("this client is bound to %q; the Request names %q", b.Routed(), req.Model)
		return nil, e
	}
	return req.WithModel(b.Routed()), nil
}

// Complete sends an ordinary canonical Request (its model must be this client's).
func (b *BoundClient) Complete(ctx context.Context, req *Request) (*Response, error) {
	req, err := b.coerce(req)
	if err != nil {
		return nil, err
	}
	return b.router.Complete(ctx, req)
}

// Ask completes one user message.
func (b *BoundClient) Ask(ctx context.Context, text string) (*Response, error) {
	req, err := NewRequest(b.Routed(), []Message{UserMessage(text)})
	if err != nil {
		return nil, err
	}
	return b.Complete(ctx, req)
}

// Stream streams an ordinary canonical Request.
func (b *BoundClient) Stream(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error] {
	coerced, err := b.coerce(req)
	if err != nil {
		return func(yield func(StreamEvent, error) bool) { yield(nil, err) }
	}
	return b.router.Stream(ctx, coerced)
}

// Close releases this client's transport. Not a logout.
func (b *BoundClient) Close() error { return b.router.Close() }

// ─── Connect (AUTH-23) ───────────────────────────────────────────────
//
// Port of lm15-python lm15/interactive.py. In order: says where connections
// are saved; offers the saved connections and "connect another"
// (subscriptions first; an ambient environment key is offered only as an
// explicit choice, never taken silently: R2/R3); runs the chosen login or
// setup; lists the account's models (or takes Model) and asks; returns a
// BoundClient pinned to that connection and model. A completed login is
// saved before the model picker runs (R6). Never: send a prompt, set a
// process-wide default, fall back to a different account. Without a UI and
// without a terminal it fails before reading any secret.

// ConnectOptions configures Connect.
type ConnectOptions struct {
	Provider        string // skip the provider picker
	Model           string // skip the model picker
	Auth            *Auth  // default LocalAuth("")
	UI              AuthUI // default a terminal UI when stdin and stderr are terminals
	Capability      string // reasoning | vision | structured-output
	OpenBrowser     bool   // terminal UI only
	RouterConfig    RouterConfig
	AllowUnverified bool
}

const (
	connectNew    = "__new__"
	connectManual = "__manual__"
)

func interactionRequired(message string) error {
	return authOperation(message, "interaction_required", "interaction", "not_committed", "provide_input")
}

func ask(ctx context.Context, ui AuthUI, p Prompt) (string, error) {
	answer, err := ui.Prompt(ctx, p)
	if err != nil {
		return "", ErrLoginCancelled
	}
	return answer, nil
}

// Connect gets ready to make model requests: it returns a client pinned to a
// connection and a model the person chose.
func Connect(ctx context.Context, opts ConnectOptions) (*BoundClient, error) {
	ui := opts.UI
	if ui == nil {
		terminal := InteractiveTerminalUI(opts.OpenBrowser)
		if terminal == nil {
			return nil, interactionRequired("Connect needs a person: no interactive terminal here and no UI was supplied. On a server, attach an Auth with saved connections (RouterConfig.Auth) instead of calling Connect.")
		}
		ui = terminal
	}
	auth := opts.Auth
	if auth == nil {
		var err error
		if auth, err = LocalAuth(""); err != nil {
			return nil, err
		}
	}
	ui.Notify(Notice{Type: "info", Message: "Connections are saved privately in " + auth.Store().Description() + "."})
	connection, err := chooseConnection(ctx, auth, ui, opts)
	if err != nil {
		return nil, err
	}
	selection, err := chooseModel(ctx, auth, ui, connection, opts)
	if err != nil {
		return nil, err
	}
	ui.Notify(Notice{Type: "info", Message: "Ready: " + selection.Routed() + " through " + connection.Label + "."})
	return NewBoundClient(auth, selection, opts.RouterConfig)
}

func chooseConnection(ctx context.Context, auth *Auth, ui AuthUI, opts ConnectOptions) (Connection, error) {
	wanted := ""
	if opts.Provider != "" {
		d, err := auth.Descriptor(opts.Provider)
		if err != nil {
			return Connection{}, err
		}
		wanted = d.ID
	}
	all, err := auth.Connections()
	if err != nil {
		return Connection{}, err
	}
	var saved []Connection
	for _, c := range all {
		if wanted == "" || c.Provider == wanted {
			saved = append(saved, c)
		}
	}
	// Subscriptions first (R2): account connections before keys.
	sort.SliceStable(saved, func(i, j int) bool {
		ai, aj := saved[i].Kind != "account", saved[j].Kind != "account"
		if ai != aj {
			return !ai
		}
		return saved[i].Provider < saved[j].Provider
	})
	var usable []Connection
	for _, c := range saved {
		status, err := auth.Status(c.Provider)
		if err != nil {
			return Connection{}, err
		}
		if status.Usability == "ready" || status.Usability == "renewal_due" || status.Usability == "unknown" {
			usable = append(usable, c)
		}
	}
	if wanted != "" && len(usable) == 1 {
		return usable[0], nil
	}
	var options []SelectOption
	for _, c := range usable {
		options = append(options, SelectOption{ID: c.ID, Label: c.Label, Description: c.Provider + " · saved"})
	}
	options = append(options, SelectOption{ID: connectNew, Label: "Connect another account or API key"})
	if len(options) == 1 {
		return newConnection(ctx, auth, ui, wanted, opts)
	}
	answer, err := ask(ctx, ui, Prompt{Type: "select", FieldID: "connection", Label: "Use a saved connection, or connect another?", Options: options})
	if err != nil {
		return Connection{}, err
	}
	if answer == connectNew {
		return newConnection(ctx, auth, ui, wanted, opts)
	}
	for _, c := range usable {
		if c.ID == answer {
			return c, nil
		}
	}
	return Connection{}, authOperation("the UI answered with an unknown connection id", "invalid_login_state", "interaction", "not_committed", "select_connection")
}

func newConnection(ctx context.Context, auth *Auth, ui AuthUI, provider string, opts ConnectOptions) (Connection, error) {
	if provider == "" {
		var descriptors []ProviderDescriptor
		for _, d := range auth.Providers() {
			for _, m := range d.Methods {
				if m.Availability != "unavailable" {
					descriptors = append(descriptors, d)
					break
				}
			}
		}
		subscription := func(d ProviderDescriptor) bool {
			for _, m := range d.Methods {
				if m.Subscription && m.Availability == "supported" {
					return true
				}
			}
			return false
		}
		sort.SliceStable(descriptors, func(i, j int) bool {
			si, sj := subscription(descriptors[i]), subscription(descriptors[j])
			if si != sj {
				return si
			}
			return strings.ToLower(descriptors[i].Label) < strings.ToLower(descriptors[j].Label)
		})
		var options []SelectOption
		for _, d := range descriptors {
			o := SelectOption{ID: d.ID, Label: d.Label}
			if d.Service != d.Label {
				o.Description = d.Service
			}
			options = append(options, o)
		}
		answer, err := ask(ctx, ui, Prompt{Type: "select", FieldID: "provider", Label: "Which provider?", Options: options})
		if err != nil {
			return Connection{}, err
		}
		provider = answer
	}
	d, err := auth.Descriptor(provider)
	if err != nil {
		return Connection{}, err
	}
	status, err := auth.Status(d.ID)
	if err != nil {
		return Connection{}, err
	}
	method, err := connectMethod(ctx, auth, ui, d, opts)
	if err != nil {
		return Connection{}, err
	}
	replace := ""
	if existing := status.Connection; existing != nil {
		answer, err := ask(ctx, ui, Prompt{Type: "select", FieldID: "replace", Label: d.Label + " already has a saved connection (" + existing.Label + ").",
			Options: []SelectOption{{ID: "keep", Label: "Keep it"}, {ID: "replace", Label: "Replace it"}}})
		if err != nil {
			return Connection{}, err
		}
		if answer == "keep" {
			return *existing, nil
		}
		replace = existing.ID
	}
	if method.Flow == "form" || method.Flow == "source_recipe" {
		answers := map[string]string{}
		for _, field := range method.Fields {
			var value string
			switch {
			case field.Type == "select" && len(field.Options) == 1:
				value = field.Options[0].ID
			case field.Type == "select":
				value, err = ask(ctx, ui, Prompt{Type: "select", FieldID: field.ID, Label: field.Label, Options: field.Options})
			case field.Type == "secret":
				value, err = ask(ctx, ui, Prompt{Type: "secret", FieldID: field.ID, Label: field.Label})
			default:
				value, err = ask(ctx, ui, Prompt{Type: "text", FieldID: field.ID, Label: field.Label})
			}
			if err != nil {
				return Connection{}, err
			}
			answers[field.ID] = value
		}
		return auth.Configure(ctx, d.ID, method.ID, answers, nil, replace)
	}
	return auth.Login(ctx, d.ID, LoginOptions{Method: method.ID, UI: ui, Replace: replace, AllowUnverified: opts.AllowUnverified})
}

func connectMethod(ctx context.Context, auth *Auth, ui AuthUI, d ProviderDescriptor, opts ConnectOptions) (LoginMethod, error) {
	var methods []LoginMethod
	for _, m := range d.Methods {
		if m.Availability == "supported" || (opts.AllowUnverified && m.Availability == "unverified") {
			methods = append(methods, m)
		}
	}
	if len(methods) == 0 {
		return LoginMethod{}, authOperation(d.ID+": no login method is available here", "method_unavailable", "discovery", "not_committed", "choose_method")
	}
	// Subscriptions first; an ambient key is offered, never assumed (R2).
	sort.SliceStable(methods, func(i, j int) bool {
		rank := func(m LoginMethod) int {
			r := 0
			if !m.Subscription {
				r += 2
			}
			if m.Kind != "account" {
				r++
			}
			return r
		}
		return rank(methods[i]) < rank(methods[j])
	})
	var options []SelectOption
	for _, m := range methods {
		note := m.BillingNote
		if m.Availability == "unverified" {
			note = "UNVERIFIED — " + m.Reason
		}
		if m.ID == "env" {
			set := ""
			if len(m.Fields) > 0 {
				for _, o := range m.Fields[0].Options {
					if auth.core.env(o.ID) != "" {
						set = o.ID
						break
					}
				}
			}
			if set == "" {
				continue // nothing to offer
			}
			note = "$" + set + " is set in this environment; using it is your explicit choice"
		}
		options = append(options, SelectOption{ID: m.ID, Label: m.Label, Description: note})
	}
	chosen := ""
	if len(options) == 1 {
		chosen = options[0].ID
	} else {
		answer, err := ask(ctx, ui, Prompt{Type: "select", FieldID: "method", Label: "How do you want to connect to " + d.Label + "?", Options: options})
		if err != nil {
			return LoginMethod{}, err
		}
		chosen = answer
	}
	if m, ok := d.Method(chosen); ok {
		return m, nil
	}
	return LoginMethod{}, authOperation("the UI answered with an unknown method id", "invalid_login_state", "interaction", "not_committed", "choose_method")
}

func chooseModel(ctx context.Context, auth *Auth, ui AuthUI, c Connection, opts ConnectOptions) (ModelSelection, error) {
	selection := func(model string) ModelSelection {
		return ModelSelection{Provider: c.Provider, Model: model, ConnectionID: c.ID, IdentityGeneration: c.IdentityGeneration}
	}
	if opts.Model != "" {
		return selection(opts.Model), nil
	}
	choices, err := ModelChoices(ctx, auth, c.Provider, opts.Capability, false, opts.RouterConfig)
	note := "listed by your account just now"
	if err != nil {
		if IsKind(err, KindAuthOperation) {
			return ModelSelection{}, err
		}
		// The catalog is a convenience: say why it is missing, do not pretend.
		kind := "error"
		var e *Error
		if errors.As(err, &e) {
			kind = string(e.Kind)
		}
		ui.Notify(Notice{Type: "info", Message: fmt.Sprintf("Could not list models for %s (%s); type a model id.", c.Provider, kind)})
		choices, note = nil, ""
	}
	var options []SelectOption
	for _, choice := range choices {
		options = append(options, SelectOption{ID: choice.Model, Label: choice.Model, Description: note})
	}
	options = append(options, SelectOption{ID: connectManual, Label: "Type a model id (not verified against your account)"})
	if opts.Capability != "" && len(choices) == 0 {
		ui.Notify(Notice{Type: "info", Message: fmt.Sprintf("No model in the list is known to support %q; you can still type one.", opts.Capability)})
	}
	answer, err := ask(ctx, ui, Prompt{Type: "select", FieldID: "model", Label: "Which " + c.Provider + " model?", Options: options})
	if err != nil {
		return ModelSelection{}, err
	}
	if answer == connectManual {
		typed, err := ask(ctx, ui, Prompt{Type: "text", FieldID: "model", Label: "Model id"})
		if err != nil {
			return ModelSelection{}, err
		}
		if answer = strings.TrimSpace(typed); answer == "" {
			return ModelSelection{}, interactionRequired("no model id given")
		}
	}
	return selection(answer), nil
}

// ─── The terminal adapter (AUTH-16) ──────────────────────────────────
//
// Port of lm15-python lm15/login/terminal.py. Notices to stderr, answers from
// stdin (a secret without echo where the terminal allows it). A browser opens
// only when the application says so; otherwise the URL is printed, which is
// what a machine reached over SSH needs. A closed input cancels.

// TerminalUI prompts on the process's terminal.
type TerminalUI struct {
	OpenBrowser bool
	lines       chan string
}

// InteractiveTerminalUI is a TerminalUI when stdin and stderr are both
// terminals, else nil (AUTH-23: never prompt a server).
func InteractiveTerminalUI(openBrowser bool) *TerminalUI {
	if !isTerminal(os.Stdin) || !isTerminal(os.Stderr) {
		return nil
	}
	return &TerminalUI{OpenBrowser: openBrowser}
}

func isTerminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func (t *TerminalUI) say(text string) { fmt.Fprintln(os.Stderr, text) }

// Notify prints the notice.
func (t *TerminalUI) Notify(n Notice) {
	switch n.Type {
	case "auth_url":
		t.say("\nOpen this link to sign in:\n  " + n.URL + "\n" + n.Instructions)
		if t.OpenBrowser {
			openBrowser(n.URL)
		}
	case "device_code":
		t.say(fmt.Sprintf("\nOpen %s\nand enter this code:  %s\n(the code is valid for about %d minutes)", n.VerificationURL, n.UserCode, int(n.ExpiresInS/60)))
		if t.OpenBrowser {
			openBrowser(n.VerificationURL)
		}
	case "progress":
		t.say("… " + n.Message)
	default:
		text := n.Message
		for _, link := range n.Links {
			text += "\n  " + link[0] + ": " + link[1]
		}
		t.say(text)
	}
}

func openBrowser(url string) {
	if !strings.HasPrefix(url, "https://") {
		return // AUTH-18: never launch a scheme a provider chose
	}
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}

func (t *TerminalUI) readLine(ctx context.Context, question string, secret bool) (string, error) {
	fmt.Fprint(os.Stderr, question)
	if t.lines == nil {
		t.lines = make(chan string)
		go func(out chan<- string) {
			reader := bufio.NewReader(os.Stdin)
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					close(out)
					return
				}
				out <- strings.TrimRight(line, "\r\n")
			}
		}(t.lines)
	}
	restore := false
	if secret && runtime.GOOS != "windows" {
		c := exec.Command("stty", "-echo")
		c.Stdin = os.Stdin
		restore = c.Run() == nil
	}
	defer func() {
		if restore {
			c := exec.Command("stty", "echo")
			c.Stdin = os.Stdin
			_ = c.Run()
			fmt.Fprintln(os.Stderr)
		}
	}()
	select {
	case line, ok := <-t.lines:
		if !ok {
			return "", ErrPromptCancelled
		}
		return line, nil
	case <-ctx.Done():
		return "", ErrPromptCancelled
	}
}

// Prompt asks on the terminal.
func (t *TerminalUI) Prompt(ctx context.Context, p Prompt) (string, error) {
	switch p.Type {
	case "select":
		t.say("\n" + p.Label)
		for i, o := range p.Options {
			note := ""
			if o.Description != "" {
				note = "  — " + o.Description
			}
			t.say(fmt.Sprintf("  %d. %s%s", i+1, o.Label, note))
		}
		for {
			raw, err := t.readLine(ctx, "Choose a number: ", false)
			if err != nil {
				return "", err
			}
			raw = strings.TrimSpace(raw)
			if n, err := strconv.Atoi(raw); err == nil && n >= 1 && n <= len(p.Options) {
				return p.Options[n-1].ID, nil
			}
			for _, o := range p.Options {
				if o.ID == raw {
					return o.ID, nil
				}
			}
			t.say("Not one of the choices.")
		}
	case "secret":
		return t.readLine(ctx, p.Label+": ", true)
	case "manual_code":
		return t.readLine(ctx, p.Label+"\n> ", false)
	}
	hint := ""
	if p.Placeholder != "" {
		hint = " [" + p.Placeholder + "]"
	}
	return t.readLine(ctx, p.Label+hint+": ", false)
}
