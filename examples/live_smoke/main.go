// Command live_smoke sends real requests through the Go port and records
// receipts. It needs keys (or saved sign-ins) and costs a little money, so
// it is not a test and no CI runs it; the contract harness is the gate.
//
//	go run ./examples/live_smoke -out receipts/DATE-live-smoke            # API keys from the environment
//	go run ./examples/live_smoke -out ... -managed                        # plus the saved sign-ins (the lm15 store)
//	go run ./examples/live_smoke -only openai,groq                        # a subset
//
// For each binding with a key, through LMRouter the way a program uses it:
//
//   - hello: one complete and one stream of the same request; the
//     assembled stream must equal the complete response in text and finish
//     reason, and its text chunks must concatenate to that text.
//   - order: a json_schema whose properties are "reasoning" then "answer"
//     ("answer" sorts first). The body sent must list them in that order;
//     the model's JSON is reported with its key order (constrained decoders
//     follow the schema: that is the point of keeping it).
//   - tools: a function tool whose parameters list "location" before
//     "date"; the model calls it, the result goes back, the model answers.
//   - models: the key's catalog lists something.
//   - order-sorted-control (-control): the same schema sent with sorted
//     keys, as Go sent it before 2026-09-25, to see what the order changes.
//
// typesafe answers judgments only; it gets one judgments call instead.
//
// Receipts: <binding>-<check>.json holding every exchange (the request as
// sent, with credential headers and query keys redacted; status; response
// headers; the body or SSE text) and what lm15 made of it. Run
// lm15-contract/tools/check_secrecy.py --root on the directory before
// keeping it.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	lm15 "github.com/lm15-dev/lm15-go"
)

// ─── Recording transport ─────────────────────────────────────────────

type exchange struct {
	Sent     map[string]any    `json:"sent"`
	Status   int               `json:"status"`
	Headers  map[string]string `json:"response_headers"`
	Body     any               `json:"body"`
	Error    string            `json:"transport_error,omitempty"`
	rawBody  *bytes.Buffer
	rawSent  []byte
	complete bool
}

type recorder struct {
	inner lm15.Transport
	mu    sync.Mutex
	log   []*exchange
}

// Only these request headers keep their value; every other value (API keys,
// bearer tokens, account ids, cookies) is written as "<redacted>".
var keepHeader = map[string]bool{
	"content-type": true, "accept": true, "anthropic-version": true, "anthropic-beta": true,
	"openai-beta": true, "user-agent": true, "accept-encoding": true, "x-goog-api-client": true,
	"editor-version": true, "copilot-integration-id": true, "openai-intent": true,
}

func redactURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return "<unparseable url>"
	}
	q := u.Query()
	for k := range q {
		if lk := strings.ToLower(k); lk == "key" || strings.Contains(lk, "token") || strings.Contains(lk, "secret") {
			q.Set(k, "<redacted>")
		}
	}
	u.RawQuery = q.Encode()
	return u.String()
}

func (r *recorder) Do(ctx context.Context, req *lm15.TransportRequest) (*lm15.TransportResponse, error) {
	headers := map[string]string{}
	for _, h := range req.Headers {
		name := strings.ToLower(h[0])
		if keepHeader[name] {
			headers[name] = h[1]
		} else {
			headers[name] = "<redacted>"
		}
	}
	ex := &exchange{
		Sent:    map[string]any{"method": req.Method, "url": redactURL(req.URL), "headers": headers},
		rawSent: append([]byte(nil), req.Body...),
		rawBody: &bytes.Buffer{},
	}
	if len(req.Body) > 0 {
		if v, err := lm15.DecodeJSON(req.Body); err == nil {
			ex.Sent["body"] = v
		} else {
			ex.Sent["body_text"] = string(req.Body)
		}
	}
	r.mu.Lock()
	r.log = append(r.log, ex)
	r.mu.Unlock()
	resp, err := r.inner.Do(ctx, req)
	if err != nil {
		ex.Error = err.Error()
		return nil, err
	}
	ex.Status = resp.Status
	ex.Headers = map[string]string{}
	for _, h := range resp.Headers {
		name := strings.ToLower(h[0])
		if name == "set-cookie" {
			ex.Headers[name] = "<redacted>"
			continue
		}
		if prev, ok := ex.Headers[name]; ok {
			ex.Headers[name] = prev + ", " + h[1]
		} else {
			ex.Headers[name] = h[1]
		}
	}
	resp.Body = &teeCloser{Reader: io.TeeReader(resp.Body, ex.rawBody), c: resp.Body}
	return resp, nil
}

type teeCloser struct {
	io.Reader
	c io.Closer
}

func (t *teeCloser) Close() error { return t.c.Close() }

func (r *recorder) take() []*exchange {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.log
	r.log = nil
	for _, ex := range out {
		b := ex.rawBody.Bytes()
		if v, err := lm15.DecodeJSON(b); err == nil {
			ex.Body = v
		} else {
			ex.Body = string(b)
		}
	}
	return out
}

// ─── Bindings ────────────────────────────────────────────────────────

type binding struct {
	name     string
	envKey   string // "" for a saved sign-in
	model    string
	managed  bool
	judgment bool // typesafe: judgments only
	external bool // another tool's login file, read by the plain router (AUTH-1)
	tokens   int  // max_tokens for the reasoning-heavy checks
	// providerRefuses: checks the provider is receipted to answer 400 to,
	// with the reason; lm15 sends the request as the reference does and the
	// loud 400 is the contract (a pass when it happens, a failure when not).
	providerRefuses map[string]string
}

var apiKeyBindings = []binding{
	{name: "openai", envKey: "OPENAI_API_KEY", model: "gpt-5-mini", tokens: 4000},
	{name: "anthropic", envKey: "ANTHROPIC_API_KEY", model: "anthropic:claude-haiku-4-5", tokens: 1000},
	{name: "gemini", envKey: "GEMINI_API_KEY", model: "gemini:gemini-2.5-flash", tokens: 4000},
	{name: "groq", envKey: "GROQ_API_KEY", model: "groq:openai/gpt-oss-20b", tokens: 2000},
	{name: "openrouter", envKey: "OPENROUTER_API_KEY", model: "openrouter:openai/gpt-4.1-nano", tokens: 1000},
	{name: "deepseek", envKey: "DEEPSEEK_API_KEY", model: "deepseek:deepseek-v4-flash", tokens: 2000,
		providerRefuses: map[string]string{
			"order":                "DeepSeek has no json_schema mode and answers 400 (lm15-python compat.py, OpenAIChatJsonSchema)",
			"order-sorted-control": "DeepSeek has no json_schema mode and answers 400",
		}},
	{name: "zai", envKey: "ZAI_API_KEY", model: "zai:glm-5.3-flash", tokens: 2000},
	{name: "meta", envKey: "META_API_KEY", model: "meta:muse-spark-1.3", tokens: 2000},
	{name: "moonshotai", envKey: "MOONSHOTAI_API_KEY", model: "moonshotai:kimi-k3", tokens: 2000},
	// Open-model inference hosts (lm15-contract changes/2026-09-26-inference-hosts-live.md);
	// each model honours json_schema and tool calls on its host (live 2026-09-26).
	{name: "deepinfra", envKey: "DEEPINFRA_API_KEY", model: "deepinfra:deepseek-ai/DeepSeek-V4.1-Flash", tokens: 2000},
	{name: "together", envKey: "TOGETHER_API_KEY", model: "together:meta-llama/Llama-3.3-70B-Instruct-Turbo", tokens: 1000},
	{name: "fireworks", envKey: "FIREWORKS_API_KEY", model: "fireworks:accounts/fireworks/models/deepseek-v4p1-flash", tokens: 2000},
	{name: "parasail", envKey: "PARASAIL_API_KEY", model: "parasail:meta-llama/Llama-3.3-70B-Instruct", tokens: 1000},
	{name: "typesafe", envKey: "TYPESAFE_API_KEY", model: "typesafe:jev-latest", judgment: true},
}

var externalBindings = []binding{
	// No max_tokens: the backend has no output cap, and lm15 refuses to drop one.
	{name: "openai-codex-cli", model: "openai-codex:gpt-5.5", external: true},
}

var managedBindings = []binding{
	{name: "xai-account", model: "xai:grok-4.7", managed: true, tokens: 2000},
	{name: "claude-code-account", model: "claude-code:claude-haiku-4-5", managed: true, tokens: 1000},
	{name: "openai-codex-account", model: "openai-codex:gpt-5.5", managed: true},
	{name: "openrouter-account", model: "openrouter:openai/gpt-4.1-nano", managed: true, tokens: 1000},
	{name: "github-copilot-account", model: "github-copilot:gpt-4.1", managed: true, tokens: 1000},
}

// ─── Checks ──────────────────────────────────────────────────────────

type result struct {
	Binding  string   `json:"binding"`
	Model    string   `json:"model"`
	Check    string   `json:"check"`
	Verdict  string   `json:"verdict"` // ok | fail | refused (a typed refusal before the wire)
	Notes    []string `json:"notes,omitempty"`
	Problems []string `json:"problems,omitempty"`
	Millis   int64    `json:"ms"`
}

func errText(err error) string {
	if e := lm15.AsError(err); e != nil {
		return fmt.Sprintf("%s (%s): %s", e.ClassName(), e.Code, e.Message)
	}
	return err.Error()
}

func refusedBeforeWire(err error) bool {
	e := lm15.AsError(err)
	return e != nil && (e.Kind.IsA(lm15.KindUnsupportedFeature) || e.Code == "unsupported_feature")
}

func hello(ctx context.Context, router *lm15.LMRouter, b binding) (result, []any) {
	r := result{Binding: b.name, Model: b.model, Check: "hello"}
	req := &lm15.Request{Model: b.model, Messages: []lm15.Message{lm15.UserMessage("Reply with exactly the two words: hello world")},
		Config: lm15.Config{MaxTokens: maxTokens(b)}}
	complete, cerr := router.Complete(ctx, req)
	rs := lm15.NewResponseStream(router.Stream(ctx, req), req)
	var chunks []string
	var serr error
	for text, err := range rs.Text() {
		if err != nil {
			serr = err
			break
		}
		chunks = append(chunks, text)
	}
	var streamed *lm15.Response
	if serr == nil {
		streamed, serr = rs.Response()
	}
	switch {
	case cerr != nil:
		r.Problems = append(r.Problems, "complete: "+errText(cerr))
	case serr != nil:
		r.Problems = append(r.Problems, "stream: "+errText(serr))
	default:
		ct, st := complete.TextOr(""), streamed.TextOr("")
		r.Notes = append(r.Notes, fmt.Sprintf("complete %q finish=%s; stream %q finish=%s in %d chunks", ct, complete.FinishReason, st, streamed.FinishReason, len(chunks)))
		if !strings.Contains(strings.ToLower(ct), "hello world") || !strings.Contains(strings.ToLower(st), "hello world") {
			r.Problems = append(r.Problems, "the reply is not the two words asked for")
		}
		if complete.FinishReason != streamed.FinishReason {
			r.Problems = append(r.Problems, "finish_reason differs between complete and stream")
		}
		if strings.Join(chunks, "") != st {
			r.Problems = append(r.Problems, "the text chunks do not concatenate to the assembled text")
		}
		if streamed.Usage.OutputTokens == nil {
			r.Notes = append(r.Notes, "the stream reported no output token count")
		}
	}
	return r, []any{responseValue(complete, cerr), responseValue(streamed, serr)}
}

var orderSchema = mustObject(`{
  "type": "object",
  "properties": {
    "reasoning": {"type": "string", "description": "Work the problem out step by step before answering."},
    "answer": {"type": "string", "description": "The final answer only."}
  },
  "required": ["reasoning", "answer"],
  "additionalProperties": false
}`)

// sortedSchema is orderSchema as a port that sorts keys sends it (Go before
// 2026-09-25): the control for the order check.
var sortedSchema = lm15.ObjectFromMap(map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"reasoning": map[string]any{"type": "string", "description": "Work the problem out step by step before answering."}, "answer": map[string]any{"type": "string", "description": "The final answer only."}},
	"required":             []any{"reasoning", "answer"}, // lists kept their order; only object keys were sorted,
	"additionalProperties": false,
})

func order(ctx context.Context, router *lm15.LMRouter, b binding, rec *recorder) (result, []any) {
	return orderWith(ctx, router, b, rec, "order", orderSchema, "reasoning", "answer")
}

func orderControl(ctx context.Context, router *lm15.LMRouter, b binding, rec *recorder) (result, []any) {
	return orderWith(ctx, router, b, rec, "order-sorted-control", sortedSchema, "answer", "reasoning")
}

func orderWith(ctx context.Context, router *lm15.LMRouter, b binding, rec *recorder, name string, schema lm15.JSONObject, firstKey, secondKey string) (result, []any) {
	r := result{Binding: b.name, Model: b.model, Check: name}
	format := lm15.JSONObject{lm15.KV("type", "json_schema"), lm15.KV("name", "worked_answer"), lm15.KV("strict", true), lm15.KV("schema", schema)}
	req := &lm15.Request{Model: b.model, Messages: []lm15.Message{lm15.UserMessage(
		"A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer in JSON.")},
		Config: lm15.Config{MaxTokens: maxTokens(b), ResponseFormat: format}}
	resp, err := router.Complete(ctx, req)
	sent := rec.peekBodies()
	if err != nil {
		if refusedBeforeWire(err) && len(sent) == 0 {
			r.Verdict = "refused"
			r.Notes = append(r.Notes, "refused before the wire: "+errText(err))
			return r, []any{responseValue(nil, err)}
		}
		r.Problems = append(r.Problems, errText(err))
	}
	// What was sent: the schema's first property must precede its second
	// wherever the schema went.
	orderedOnWire := false
	for _, body := range sent {
		i, j := bytes.Index(body, []byte(`"`+firstKey+`":{`)), bytes.Index(body, []byte(`"`+secondKey+`":{`))
		if i >= 0 && j >= 0 {
			orderedOnWire = true
			if i > j {
				r.Problems = append(r.Problems, "the body sent lists "+secondKey+" before "+firstKey)
			}
		}
	}
	if !orderedOnWire && len(sent) > 0 {
		r.Notes = append(r.Notes, "the schema's properties are not in the body sent (the wire has no schema slot)")
	}
	if resp != nil {
		for _, a := range resp.Adaptations {
			r.Notes = append(r.Notes, fmt.Sprintf("adapted %s: %s", a.Field, a.Action))
			if a.Field == "config.response_format" && a.Action == "dropped" {
				// The server accepts json_schema and ignores it (receipted);
				// the adapter drops it and says so (MAP-13 note policy), as
				// the reference does. Nothing about order to check.
				r.Verdict = "adapted"
				r.Notes = append(r.Notes, fmt.Sprintf("the reply, free-form: %.80q", resp.TextOr("")))
				return r, []any{responseValue(resp, err)}
			}
		}
		text := strings.TrimSpace(resp.TextOr(""))
		text = strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(text, "```json"), "```"), "```")
		obj, derr := lm15.DecodeJSONObject([]byte(strings.TrimSpace(text)))
		if derr != nil {
			r.Problems = append(r.Problems, fmt.Sprintf("the reply is not a JSON object (finish=%s): %.120q", resp.FinishReason, resp.TextOr("")))
		} else {
			r.Notes = append(r.Notes, fmt.Sprintf("the model's JSON keys, in the order written: %v; answer %q", obj.Keys(), obj.Get("answer")))
			if !strings.Contains(fmt.Sprint(obj.Get("answer")), "0.05") && !strings.Contains(fmt.Sprint(obj.Get("answer")), "5 cents") {
				r.Notes = append(r.Notes, "the answer is not $0.05")
			}
		}
	}
	return r, []any{responseValue(resp, err)}
}

func tools(ctx context.Context, router *lm15.LMRouter, b binding) (result, []any) {
	r := result{Binding: b.name, Model: b.model, Check: "tools"}
	params := lm15.JSONObject{
		lm15.KV("type", "object"),
		lm15.KV("properties", lm15.JSONObject{
			lm15.KV("location", lm15.JSONObject{lm15.KV("type", "string"), lm15.KV("description", "City name")}),
			lm15.KV("date", lm15.JSONObject{lm15.KV("type", "string"), lm15.KV("description", "YYYY-MM-DD")}),
		}),
		lm15.KV("required", []any{"location", "date"}),
		// OpenAI strict mode wants it; Gemini's OpenAPI parameters field
		// refuses it, so on Gemini this schema goes as parametersJsonSchema
		// (MAP-16, 2026-09-26).
		lm15.KV("additionalProperties", false),
	}
	tool := lm15.FunctionTool{Name: "get_forecast", Description: "Weather forecast for a city on a date.", Parameters: params}
	req := &lm15.Request{Model: b.model, Tools: []lm15.Tool{tool},
		Messages: []lm15.Message{lm15.UserMessage("Use the tool: what is the forecast for Montreal on 2026-10-01? Then answer in one sentence.")},
		Config:   lm15.Config{MaxTokens: maxTokens(b)}}
	first, err := router.Complete(ctx, req)
	if err != nil {
		r.Problems = append(r.Problems, "first call: "+errText(err))
		return r, []any{responseValue(nil, err)}
	}
	calls := first.ToolCalls()
	if len(calls) == 0 {
		r.Problems = append(r.Problems, fmt.Sprintf("no tool call (finish=%s, text %.80q)", first.FinishReason, first.TextOr("")))
		return r, []any{responseValue(first, nil)}
	}
	call := calls[0]
	r.Notes = append(r.Notes, fmt.Sprintf("call %s(%s)", call.Name, string(mustEncode(call.Input))))
	if call.Name != "get_forecast" || !strings.Contains(strings.ToLower(fmt.Sprint(call.Input.Get("location"))), "montr") {
		r.Problems = append(r.Problems, "the call is not get_forecast for Montreal")
	}
	req.Messages = append(req.Messages, first.Message, lm15.ToolMessage(call.ID, `{"forecast": "sunny", "high_c": 14}`))
	second, err := router.Complete(ctx, req)
	if err != nil {
		r.Problems = append(r.Problems, "second call: "+errText(err))
		return r, []any{responseValue(first, nil), responseValue(nil, err)}
	}
	text := second.TextOr("")
	r.Notes = append(r.Notes, fmt.Sprintf("final %.100q", text))
	if !strings.Contains(strings.ToLower(text), "sunny") && !strings.Contains(text, "14") {
		r.Problems = append(r.Problems, "the final answer does not use the tool result")
	}
	return r, []any{responseValue(first, nil), responseValue(second, nil)}
}

func models(ctx context.Context, router *lm15.LMRouter, b binding) (result, []any) {
	r := result{Binding: b.name, Model: b.model, Check: "models"}
	lm, err := router.LM(b.model)
	if err != nil {
		r.Problems = append(r.Problems, errText(err))
		return r, nil
	}
	list, err := lm.ListModels(ctx)
	if err != nil {
		if refusedBeforeWire(err) {
			r.Verdict = "refused"
			r.Notes = append(r.Notes, errText(err))
			return r, nil
		}
		r.Problems = append(r.Problems, errText(err))
		return r, nil
	}
	ids := make([]any, 0, len(list))
	for _, m := range list {
		ids = append(ids, m.ID)
	}
	r.Notes = append(r.Notes, fmt.Sprintf("%d models", len(list)))
	if len(list) == 0 {
		r.Problems = append(r.Problems, "an empty catalog")
	}
	return r, []any{map[string]any{"ids": ids}}
}

func judgments(ctx context.Context, router *lm15.LMRouter, b binding) (result, []any) {
	r := result{Binding: b.name, Model: b.model, Check: "judgments"}
	quality, _ := lm15.Score("How good is this wine?", lm15.ScoreLevel{Name: "poor"}, lm15.ScoreLevel{Name: "fair"}, lm15.ScoreLevel{Name: "great"})
	style, _ := lm15.Choice("Dominant style?", lm15.Options("fruit", "oak", "mineral")...)
	format, err := lm15.Judgments("wine", true,
		lm15.JudgmentProperty{Name: "style", Schema: style},
		lm15.JudgmentProperty{Name: "quality", Schema: quality},
		lm15.JudgmentProperty{Name: "ageing", Schema: lm15.YesNo("Will it improve with age?")})
	if err != nil {
		r.Problems = append(r.Problems, err.Error())
		return r, nil
	}
	req := &lm15.Request{Model: b.model, Messages: []lm15.Message{lm15.UserMessage(
		"Tasting note: deep ruby, blackcurrant and cedar, firm tannins, long finish, 2019 Pauillac.")},
		Config: lm15.Config{ResponseFormat: format, Probabilities: "required"}}
	resp, err := router.Complete(ctx, req)
	if err != nil {
		r.Problems = append(r.Problems, errText(err))
		return r, []any{responseValue(nil, err)}
	}
	probs := resp.Probabilities()
	for _, name := range []string{"style", "quality", "ageing"} {
		dist := probs[name]
		sum := 0.0
		for _, p := range dist {
			sum += p
		}
		if len(dist) == 0 || sum < 0.98 || sum > 1.02 {
			r.Problems = append(r.Problems, fmt.Sprintf("%s: distribution %v", name, dist))
		}
	}
	if part, ok := resp.DataPart(); ok {
		if obj, ok := part.Value.(lm15.JSONObject); ok {
			r.Notes = append(r.Notes, fmt.Sprintf("answer keys in order %v", obj.Keys()))
		}
	}
	exp, _ := resp.Expected("quality")
	r.Notes = append(r.Notes, fmt.Sprintf("data %s; expected quality %.2f; style %v", mustEncode(resp.Data()), exp, probs["style"]))
	return r, []any{responseValue(resp, nil)}
}

// ─── Plumbing ────────────────────────────────────────────────────────

func (r *recorder) peekBodies() [][]byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out [][]byte
	for _, ex := range r.log {
		if len(ex.rawSent) > 0 {
			out = append(out, ex.rawSent)
		}
	}
	return out
}

func responseValue(resp *lm15.Response, err error) any {
	if err != nil {
		out := map[string]any{"message": err.Error()}
		if e := lm15.AsError(err); e != nil {
			out = map[string]any{"class": e.ClassName(), "code": e.Code, "message": e.Message}
		}
		return map[string]any{"error": out}
	}
	if resp == nil {
		return nil
	}
	return map[string]any{"response": lm15.ResponseToDict(resp, false)}
}

func maxTokens(b binding) *int {
	if b.tokens == 0 {
		return nil
	}
	return lm15.I(b.tokens)
}

func mustObject(s string) lm15.JSONObject {
	o, err := lm15.DecodeJSONObject([]byte(s))
	if err != nil {
		panic(err)
	}
	return o
}

func mustEncode(v any) []byte {
	b, err := lm15.EncodeJSON(v)
	if err != nil {
		return []byte(fmt.Sprintf("<%v>", err))
	}
	return b
}

func main() {
	out := flag.String("out", "", "receipt directory (none written when empty)")
	only := flag.String("only", "", "comma-separated binding names")
	managed := flag.Bool("managed", false, "also run the saved sign-ins in the lm15 store")
	control := flag.Bool("control", false, "also send the order schema with sorted keys, as a sorting port would (one more call per binding)")
	flag.Parse()
	if *out != "" {
		if err := os.MkdirAll(*out, 0o755); err != nil {
			panic(err)
		}
	}
	want := map[string]bool{}
	for _, n := range strings.Split(*only, ",") {
		if n != "" {
			want[n] = true
		}
	}
	rec := &recorder{inner: lm15.NewHTTPTransport()}
	keyRouter, err := lm15.NewRouterWithConfig(lm15.RouterConfig{Transport: rec})
	if err != nil {
		panic(err)
	}
	var managedRouter *lm15.LMRouter
	if *managed {
		auth, err := lm15.LocalAuth("")
		if err != nil {
			panic(err)
		}
		if managedRouter, err = lm15.NewRouterWithConfig(lm15.RouterConfig{Transport: rec, Auth: auth, Env: map[string]string{}}); err != nil {
			panic(err)
		}
	}
	bindings := append(append([]binding{}, apiKeyBindings...), externalBindings...)
	if *managed {
		bindings = append(bindings, managedBindings...)
	}
	var results []result
	for _, b := range bindings {
		if len(want) > 0 && !want[b.name] {
			continue
		}
		router := keyRouter
		if b.managed {
			router = managedRouter
		} else if !b.external && os.Getenv(b.envKey) == "" {
			fmt.Printf("%-24s skipped (%s unset)\n", b.name, b.envKey)
			continue
		}
		type check struct {
			name string
			run  func(context.Context) (result, []any)
		}
		var checks []check
		if b.judgment {
			checks = []check{{"judgments", func(c context.Context) (result, []any) { return judgments(c, router, b) }}}
		} else {
			checks = []check{
				{"hello", func(c context.Context) (result, []any) { return hello(c, router, b) }},
				{"order", func(c context.Context) (result, []any) { return order(c, router, b, rec) }},
				{"tools", func(c context.Context) (result, []any) { return tools(c, router, b) }},
			}
			if *control {
				checks = append(checks, check{"order-sorted-control", func(c context.Context) (result, []any) { return orderControl(c, router, b, rec) }})
			}
		}
		checks = append(checks, check{"models", func(c context.Context) (result, []any) { return models(c, router, b) }})
		for _, c := range checks {
			ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
			started := time.Now()
			r, lmValues := c.run(ctx)
			cancel()
			r.Millis = time.Since(started).Milliseconds()
			if why, ok := b.providerRefuses[c.name]; ok {
				if len(r.Problems) == 1 && strings.HasPrefix(r.Problems[0], "InvalidRequestError") {
					r.Verdict = "provider-refused"
					r.Notes = append(r.Notes, why+": "+r.Problems[0])
					r.Problems = nil
				} else {
					r.Problems = append(r.Problems, "expected the provider to refuse: "+why)
				}
			}
			if b.managed && len(r.Problems) > 0 && strings.Contains(r.Problems[len(r.Problems)-1], "AuthOperationError (auth_operation)") &&
				strings.Contains(r.Problems[len(r.Problems)-1], "signed out") {
				// A connection the user signed out of: the typed refusal is
				// the contract (AUTH-15 B: never a silent switch to a key).
				r.Verdict = "signed-out"
				r.Notes = append(r.Notes, r.Problems...)
				r.Problems = nil
			}
			if r.Verdict == "" {
				r.Verdict = "ok"
				if len(r.Problems) > 0 {
					r.Verdict = "fail"
				}
			}
			exchanges := rec.take()
			results = append(results, r)
			fmt.Printf("%-24s %-9s %-7s %6dms  %s\n", b.name, c.name, strings.ToUpper(r.Verdict), r.Millis, strings.Join(r.Notes, " | "))
			for _, p := range r.Problems {
				fmt.Printf("%-24s           - %s\n", "", p)
			}
			if *out != "" {
				if c.name == "models" {
					for _, ex := range exchanges {
						ex.Body = map[string]any{"entries_omitted": true}
					}
				}
				receipt := map[string]any{
					"port": "lm15-go " + lm15.Version, "binding": b.name, "model": b.model, "check": c.name,
					"managed": b.managed, "exchanges": exchanges, "lm15": lmValues, "verdict": r.Verdict,
					"notes": r.Notes, "problems": r.Problems, "timestamp": time.Now().UTC().Format(time.RFC3339),
				}
				raw, err := json.MarshalIndent(receipt, "", "  ")
				if err != nil {
					panic(err)
				}
				if err := os.WriteFile(filepath.Join(*out, b.name+"-"+c.name+".json"), raw, 0o644); err != nil {
					panic(err)
				}
			}
		}
	}
	failed := 0
	for _, r := range results {
		if r.Verdict == "fail" {
			failed++
		}
	}
	if *out != "" {
		sort.SliceStable(results, func(i, j int) bool { return results[i].Binding < results[j].Binding })
		raw, _ := json.MarshalIndent(results, "", "  ")
		_ = os.WriteFile(filepath.Join(*out, "results.json"), raw, 0o644)
	}
	fmt.Printf("\n%d checks, %d failed\n", len(results), failed)
	if failed > 0 {
		os.Exit(1)
	}

}
