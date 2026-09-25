package lm15

// TypeSafe System One (Jev) — provider "typesafe".
//
// changes/2026-09-17-judgments.md. One POST /v1/systemone per Request: the
// one user part becomes Jev's state verbatim (2026-09-19 D1), the judgment
// properties of the json_schema become its questions (MAP-14 §2), and the
// answers come back as one DataPart with the distribution per judgment and
// method "provider_classification" (§3). Jev generates no text: a request
// without judgments, with tools, with media, with a system prompt or a
// conversation is refused before the wire, the native place named (D2/D8).
// Wire facts: receipts/2026-09-17-judgments/, receipts/2026-09-19-typesafe/.

import (
	"context"
	"fmt"
	"iter"
	"math"
	"strconv"
	"strings"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

const typesafeDefaultBaseURL = "https://api.typesafe.ai"

// TypeSafeAPI is the typesafe access policy: bearer key, complete and
// models only (D11: no stream).
var TypeSafeAPI = AccessPolicy{
	Provider:   "typesafe",
	Supports:   EndpointSupport{Complete: true, Models: true},
	AuthModes:  []string{"bearer"},
	EnvKeys:    []string{"TYPESAFE_API_KEY"},
	AuthScheme: []string{"bearer"},
}

// TypeSafeLM is the TypeSafe System One dialect (POST /v1/systemone).
type TypeSafeLM struct {
	lmCore
}

// NewTypeSafeLM constructs the typesafe adapter.
func NewTypeSafeLM(opts ...Option) (*TypeSafeLM, error) {
	o, err := applyOptions(opts)
	if err != nil {
		return nil, err
	}
	lm := &TypeSafeLM{}
	if err := lm.bindAccess(lm, TypeSafeAPI, o, typesafeDefaultBaseURL); err != nil {
		return nil, err
	}
	return lm, nil
}

func (l *TypeSafeLM) refuse(feature, format string, args ...any) *Error {
	return UnsupportedFeature(l.provider, feature, "%s: %s", l.provider, fmt.Sprintf(format, args...))
}

// state is changes/2026-09-19-jev-state.md D1/D2: the state is the one
// user part, verbatim — a text's string or a data part's value. Jev has
// no system prompt and no conversation; anything else is refused with the
// native place named, never merged into a shape of ours. A media part is
// refused first, as the specific fault it is (MAP-10).
func (l *TypeSafeLM) state(req *Request) (any, error) {
	for m, msg := range req.Messages {
		for p, part := range msg.Parts {
			switch part.(type) {
			case TextPart, DataPart:
			default:
				return nil, l.refuse(fmt.Sprintf("messages[%d].parts[%d]", m, p), "a %s part has no slot on the systemone wire (MAP-10); Jev reads text or data", part.Type())
			}
		}
	}
	if req.System != nil {
		return nil, l.refuse("system", "Jev has no system prompt; put context in the state as a named key (UserParts(Data(map{\"policy\": ..., \"note\": ...}))), or the framing in each question's description (changes/2026-09-19-jev-state.md D2)")
	}
	if len(req.Messages) != 1 {
		return nil, l.refuse("messages", "Jev judges one state, got %d messages; put a transcript in the state as an array or object (UserParts(Data(map{\"messages\": [...]}))), where a question can point at a turn with a backtick path (changes/2026-09-19-jev-state.md D2)", len(req.Messages))
	}
	msg := req.Messages[0]
	if msg.Role != RoleUser {
		return nil, l.refuse("messages[0].role", "Jev's state is a user message, got role %q", msg.Role)
	}
	if len(msg.Parts) != 1 {
		return nil, l.refuse("messages[0].parts", "Jev's state is one text or data part, got %d parts; put several pieces in one data part as named keys", len(msg.Parts))
	}
	switch only := msg.Parts[0].(type) {
	case TextPart:
		return only.Text, nil
	case DataPart:
		return deref(only.Value), nil
	}
	return nil, l.refuse("messages[0].parts[0]", "Jev reads text or data")
}

func (l *TypeSafeLM) questions(req *Request, scope *adaptScope) (JSONObject, error) {
	f := req.Config.ResponseFormat
	if f == nil || f.Get("type") != "json_schema" {
		return nil, l.refuse("config.response_format", "Jev answers declared judgments only; give a json_schema response_format whose properties are enums / booleans / ordered levels (MAP-14), e.g. lm15.Judgments(...)")
	}
	found := RequestJudgments(req)
	extra := nonJudgmentProperties(f.Get("schema"), found)
	if len(found) == 0 || len(extra) > 0 {
		what := "no property declares a judgment"
		if len(extra) > 0 {
			what = fmt.Sprintf("properties %v are free-form", extra)
		}
		return nil, l.refuse("config.response_format", "%s; Jev cannot generate values, only pick among declared keys (MAP-14 §1)", what)
	}
	questions := JSONObject{}
	for _, j := range found {
		instruction := j.Instruction
		if instruction == "" {
			if err := scope.defaulted("config.response_format.schema.properties."+j.Name+".description", "a judgment without a description: the property name goes as the instruction (Jev never sees property names)", j.Name); err != nil {
				return nil, err
			}
			instruction = j.Name
		}
		switch j.Kind {
		case JudgmentBoolean:
			questions.Set(j.Name, JSONObject{{"type", "noul"}, {"instructions", instruction}})
		case JudgmentChoice:
			if len(j.Keys) > MaxChoiceKeys {
				return nil, l.refuse("config.response_format.schema.properties."+j.Name, "a Jev choice takes at most %d keys, got %d", MaxChoiceKeys, len(j.Keys))
			}
			criteria := JSONObject{}
			for _, k := range j.Keys {
				if d, ok := j.Descriptions[k]; ok && d != "" {
					criteria.Set(k, d)
				} else {
					criteria.Set(k, nil)
				}
			}
			questions.Set(j.Name, JSONObject{{"type", "choice"}, {"instructions", instruction}, {"criteria", criteria}})
		default:
			if len(j.Keys) > MaxOrderedLevels {
				return nil, l.refuse("config.response_format.schema.properties."+j.Name, "a Jev score takes at most %d levels, got %d", MaxOrderedLevels, len(j.Keys))
			}
			criteria := make([]any, 0, len(j.Keys))
			for _, k := range j.Keys {
				if d := j.Descriptions[k]; d != "" {
					criteria = append(criteria, d)
				} else {
					criteria = append(criteria, k)
				}
			}
			questions.Set(j.Name, JSONObject{{"type", "score"}, {"instructions", instruction}, {"criteria", criteria}})
		}
	}
	return questions, nil
}

func (l *TypeSafeLM) payload(req *Request, scope *adaptScope) (JSONObject, error) {
	if len(req.Tools) > 0 {
		return nil, l.refuse("tools", "tools have no slot on the systemone wire")
	}
	cfg := req.Config
	if cfg.ToolChoice != nil {
		return nil, l.refuse("config.tool_choice", "tool_choice has no slot on the systemone wire")
	}
	// Config knobs with no home on the systemone wire (D8): dropped with a record.
	const why = "no such control on the systemone wire (Jev returns decisions, not samples)"
	knobs := []struct {
		name string
		set  bool
		val  any
	}{
		{"max_tokens", cfg.MaxTokens != nil, deref(cfg.MaxTokens)},
		{"temperature", cfg.Temperature != nil, deref(cfg.Temperature)},
		{"top_p", cfg.TopP != nil, deref(cfg.TopP)},
		{"top_k", cfg.TopK != nil, deref(cfg.TopK)},
		{"stop", len(cfg.Stop) > 0, jsonRaw(cfg.Stop)},
		{"seed", cfg.Seed != nil, deref(cfg.Seed)},
		{"frequency_penalty", cfg.FrequencyPenalty != nil, deref(cfg.FrequencyPenalty)},
		{"presence_penalty", cfg.PresencePenalty != nil, deref(cfg.PresencePenalty)},
		{"reasoning", cfg.Reasoning != nil, reasoningAsked(cfg.Reasoning)},
		{"logprobs", cfg.Logprobs != nil, deref(cfg.Logprobs)},
		{"store", cfg.Store != nil, deref(cfg.Store)},
		{"user_id", cfg.UserID != "", cfg.UserID},
		{"service_tier", cfg.ServiceTier != "", cfg.ServiceTier},
		{"cache", cfg.Cache != nil, cacheAsked(cfg.Cache)},
	}
	for _, knob := range knobs {
		if knob.set {
			if err := scope.dropped("config."+knob.name, why, knob.val); err != nil {
				return nil, err
			}
		}
	}
	questions, err := l.questions(req, scope)
	if err != nil {
		return nil, err
	}
	state, err := l.state(req)
	if err != nil {
		return nil, err
	}
	payload := JSONObject{{"model", req.Model}, {"state", state}, {"questions", questions}}
	for key, value := range cfg.Extensions.All() {
		if key == "n" {
			if n, err := jsonFloat64(value, "n"); err == nil && n > 1 {
				return nil, l.refuse("config.extensions.n", "n > 1 has no canonical multiple-response representation")
			}
		}
		payload.Set(key, value)
	}
	return payload, nil
}

func reasoningAsked(r *Reasoning) any {
	if r == nil {
		return nil
	}
	return jsonRaw(ReasoningToDict(*r))
}

func cacheAsked(c *CacheConfig) any {
	if c == nil {
		return nil
	}
	return jsonRaw(CacheConfigToDict(*c))
}

func (l *TypeSafeLM) buildRequest(req *Request, stream bool, scope *adaptScope) (*TransportRequest, error) {
	if stream {
		return nil, l.refuse("stream", "systemone answers in one piece; there is no stream to wrap")
	}
	payload, err := l.payload(req, scope)
	if err != nil {
		return nil, err
	}
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/v1/systemone", endpoint: "systemone", model: req.Model, headers: [][2]string{{"Content-Type", "application/json"}}, payload: payload, scope: scope})
}

// ─── Response parsing ────────────────────────────────────────────────

func (l *TypeSafeLM) requestID(resp *HTTPResponse) string {
	return resp.Header("x-typesafe-request-id")
}

func (l *TypeSafeLM) parseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	data, err := l.jsonBody(resp)
	if err != nil {
		return nil, err
	}
	found := RequestJudgments(req)
	invalid := func(path, detail string) *Error {
		e := l.providerError(KindProvider, fmt.Sprintf("malformed systemone reply at %s: %s", path, detail), resp.Status, "", l.requestID(resp))
		attachErrorMetadata(e, resp.Headers)
		return e
	}
	probability := func(raw any, path string) (float64, *Error) {
		if _, isBool := raw.(bool); isBool {
			return 0, invalid(path, "expected a finite number in [0, 1]")
		}
		p, err := jsonFloat64(raw, path)
		if err != nil || math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return 0, invalid(path, "expected a finite number in [0, 1]")
		}
		return p, nil
	}
	answers, ok := asObject(data.Get("answers"))
	if !ok {
		return nil, invalid("answers", "expected an object containing every declared judgment")
	}
	if len(answers) != len(found) {
		return nil, invalid("answers", "keys must match the declared judgments exactly")
	}
	value := JSONObject{}
	probabilities := map[string]map[string]float64{}
	for _, j := range found {
		answer, ok := asObject(answers.Get(j.Name))
		path := "answers." + j.Name
		if !ok {
			if _, present := answers.Lookup(j.Name); !present {
				return nil, invalid("answers", "keys must match the declared judgments exactly")
			}
			return nil, invalid(path, "expected an answer object")
		}
		expected := map[JudgmentKind]string{JudgmentBoolean: "noul", JudgmentChoice: "choice", JudgmentOrdered: "score"}[j.Kind]
		if answer.Get("type") != expected {
			return nil, invalid(path+".type", "expected "+strconv.Quote(expected))
		}
		if j.Kind == JudgmentBoolean {
			p, perr := probability(answer.Get("noul"), path+".noul")
			if perr != nil {
				return nil, perr
			}
			value.Set(j.Name, p >= 0.5)
			probabilities[j.Name] = map[string]float64{"true": p, "false": 1.0 - p}
			continue
		}
		dist, ok := asObject(answer.Get("probabilities"))
		if !ok || len(dist) != len(j.Keys) {
			return nil, invalid(path+".probabilities", "expected one probability for every declared key, and no other keys")
		}
		probs := make(map[string]float64, len(j.Keys))
		for _, k := range j.Keys {
			raw, present := dist.Lookup(k)
			if !present {
				return nil, invalid(path+".probabilities", "expected one probability for every declared key, and no other keys")
			}
			// INV-052: validate measurements individually, NEVER their total.
			p, perr := probability(raw, path+".probabilities."+k)
			if perr != nil {
				return nil, perr
			}
			probs[k] = p
		}
		probabilities[j.Name] = probs
		if j.Kind == JudgmentChoice {
			pick, ok := answer.Get("choice").(string)
			if !ok || !inVocab(pick, j.Keys) {
				return nil, invalid(path+".choice", "expected a declared choice key")
			}
			value.Set(j.Name, pick)
		} else {
			best, bestP := "", math.Inf(-1)
			for _, k := range j.Keys { // declared order breaks ties, like the reference's max
				if probs[k] > bestP {
					best, bestP = k, probs[k]
				}
			}
			n, _ := strconv.Atoi(best)
			value.Set(j.Name, n)
		}
	}
	part := DataPart{Value: value}
	if len(probabilities) > 0 {
		part.Probabilities = probabilities
		part.Method = MethodProviderClassification
	}
	if err := part.Validate(); err != nil {
		return nil, invalid("answers", err.Error())
	}
	usage := Usage{}
	if raw, present := data.Lookup("usage"); present && raw != nil {
		u, ok := asObject(raw)
		if !ok {
			return nil, invalid("usage", "expected an object or null")
		}
		var err error
		if usage.InputTokens, err = optInt(u, "input_tokens"); err != nil {
			return nil, invalid("usage", err.Error())
		}
		if usage.OutputTokens, err = optInt(u, "output_tokens"); err != nil {
			return nil, invalid("usage", err.Error())
		}
	}
	model := req.Model
	if raw, present := data.Lookup("model"); present && raw != nil {
		s, ok := raw.(string)
		if !ok || s == "" {
			return nil, invalid("model", "expected a non-empty string")
		}
		model = s
	}
	out := &Response{
		ID:           l.requestID(resp),
		Model:        model,
		Message:      Message{Role: RoleAssistant, Parts: []Part{part}},
		FinishReason: FinishStop,
		Usage:        usage.Normalize(),
		ProviderData: JSONObject{{"typesafe", JSONObject{{"answers", answers}}}},
	}
	if err := out.Validate(); err != nil {
		return nil, invalid("$", err.Error())
	}
	return out, nil
}

func (l *TypeSafeLM) parseStreamEvents(*Request, sse.Event) ([]StreamEvent, error) {
	return nil, l.refuse("stream", "systemone has no stream")
}

func (l *TypeSafeLM) streamOverride(context.Context, *Request) (iter.Seq2[StreamEvent, error], bool) {
	return errSeq(l.refuse("stream", "systemone answers in one piece; there is no stream to wrap")), true
}

// ─── Errors (D9) ─────────────────────────────────────────────────────

func (l *TypeSafeLM) normalizeError(status int, body string) *Error {
	message := strings.TrimSpace(body)
	if len(message) > 500 {
		message = message[:500]
	}
	if message == "" {
		message = "HTTP " + itoa(status)
	}
	code := ""
	if raw, err := DecodeJSON([]byte(body)); err == nil {
		if payload := wireObj(raw); payload != nil {
			switch detail := jsonView(payload.Get("detail")).(type) {
			case JSONObject:
				if s, ok := detail.Get("error_type").(string); ok {
					code = s
				}
				if s, ok := detail.Get("message").(string); ok {
					message = s
				}
			case []any:
				// pydantic validation: [{type, loc, msg, input}]
				if len(detail) > 0 {
					first := wireObj(detail[0])
					var loc []string
					for _, x := range wireList(first.Get("loc")) {
						if s := wireStr(x); s != "body" {
							loc = append(loc, s)
						}
					}
					msg := wireStr(first.Get("msg"))
					if msg == "" {
						msg = "validation error"
					}
					if len(loc) > 0 {
						message = strings.Join(loc, ".") + ": " + msg
					} else {
						message = msg
					}
				}
			}
		}
	}
	var e *Error
	switch {
	case status == 401 || code == "authentication_error":
		e = AuthErrorf(l.provider, l.access.EnvKeys, "", "%s", message)
	case status == 429:
		e = providerErrorf(KindRateLimit, l.provider, l.access.EnvKeys, message)
	case status == 400 && strings.Contains(strings.ToLower(message), "unknown model"):
		e = providerErrorf(KindUnsupportedModel, l.provider, l.access.EnvKeys, message)
	case status == 400 || status == 422:
		e = providerErrorf(KindInvalidRequest, l.provider, l.access.EnvKeys, message)
	case status >= 500:
		e = providerErrorf(KindServer, l.provider, l.access.EnvKeys, message)
	default:
		return l.withLoginHint(MapHTTPError(status, message, l.provider, l.access.EnvKeys, code, "", nil))
	}
	e.Status = status
	e.ProviderCode = code
	return e
}

// ─── Models (D10) ────────────────────────────────────────────────────

func (l *TypeSafeLM) modelsRequest() (*TransportRequest, error) {
	return l.emit(emitSpec{method: "GET", url: strings.TrimRight(l.baseURL, "/") + "/v1/models", headers: [][2]string{{"Content-Type", "application/json"}}})
}

func (l *TypeSafeLM) modelsFromBody(body string) ([]ModelInfo, error) {
	data, err := DecodeJSONObject([]byte(body))
	if err != nil {
		return nil, err
	}
	return modelInfosFromEntries(data.Get("models"), l.provider, "typesafe_systemone", func(e JSONObject) string { return stringOnly(e.Get("name")) }), nil
}
