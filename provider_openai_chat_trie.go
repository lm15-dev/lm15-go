package lm15

// Judgments by candidate-sequence likelihood (MAP-14 §4).
//
// A server that scores named tokens (compat token_scoring =
// "logprob_token_ids": vLLM ≥ 0.29, receipted 2026-09-17) can deliver a
// distribution over the declared keys of every judgment. Pure hooks
// below; the driver sequences them like files and batch:
//
//  1. one /tokenize per (judgment, key) and per judgment prefill, so the
//     server's chat template is honoured and the key path is read in
//     context (terminator included: prefix-free paths);
//  2. ONE /v1/completions call carrying every trie node as a prompt
//     (token ids), max_tokens 1, logprob_token_ids = union of child
//     tokens; raw log-probs sum along each path, one normalisation.

import (
	"context"
	"math"
	"sort"
	"strconv"
	"strings"
)

const judgmentPrefill = "Answer:"

func (l *OpenAIChatLM) judgmentAsk(j Judgment) string {
	var keys []string
	for _, k := range j.Keys {
		label, desc := j.Titles[k], j.Descriptions[k]
		tail := ""
		switch {
		case label != "" && desc != "":
			tail = ": " + label + " - " + desc
		case label != "":
			tail = ": " + label
		case desc != "":
			tail = ": " + desc
		}
		keys = append(keys, "- "+k+tail)
	}
	instruction := j.Instruction
	if instruction == "" {
		instruction = j.Name
	}
	return instruction + "\nOptions:\n" + strings.Join(keys, "\n") + "\nAnswer with the option only, spelled exactly as listed."
}

// judgmentMessages is the conversation, the judgment's question as a
// final user turn, and the assistant's answer so far (prefill + key).
func (l *OpenAIChatLM) judgmentMessages(req *Request, j Judgment, answer string) ([]any, error) {
	compat := l.compatFor(req.Model)
	messages, err := l.buildMessages(req, compat, nil)
	if err != nil {
		return nil, err
	}
	messages = append(messages, JSONObject{"role": "user", "content": l.judgmentAsk(j)})
	messages = append(messages, JSONObject{"role": "assistant", "content": answer})
	return messages, nil
}

func (l *OpenAIChatLM) judgmentTokenizeRequest(model string, messages []any, continueFinal bool) (*TransportRequest, error) {
	root := strings.TrimSuffix(strings.TrimRight(l.baseURL, "/"), "/v1")
	return l.emit(emitSpec{method: "POST", url: root + "/tokenize", endpoint: "tokenize", model: model, headers: l.headers(),
		payload: JSONObject{"model": model, "messages": messages, "add_generation_prompt": false, "continue_final_message": continueFinal}})
}

func (l *OpenAIChatLM) judgmentReplyError(resp *HTTPResponse, detail string) *Error {
	e := l.providerError(KindProvider, "malformed judgment reply: "+detail, resp.Status, "", "")
	attachErrorMetadata(e, resp.Headers)
	return e
}

func (l *OpenAIChatLM) judgmentTokensFromBody(resp *HTTPResponse) ([]int, error) {
	data, err := resp.JSON()
	if err != nil {
		return nil, l.replyError(resp, err)
	}
	list, ok := data["tokens"].([]any)
	if !ok {
		return nil, l.judgmentReplyError(resp, "tokenize reply carries no non-negative integer token list")
	}
	out := make([]int, 0, len(list))
	for _, t := range list {
		n, ok := jsonInteger(t)
		if !ok || n < 0 {
			return nil, l.judgmentReplyError(resp, "tokenize reply carries no non-negative integer token list")
		}
		out = append(out, n)
	}
	return out, nil
}

func (l *OpenAIChatLM) judgmentScoreRequest(model string, prompts [][]int, tokenIDs []int) (*TransportRequest, error) {
	sorted := append([]int(nil), tokenIDs...)
	sort.Ints(sorted)
	return l.emit(emitSpec{method: "POST", url: strings.TrimRight(l.baseURL, "/") + "/completions", endpoint: "completions", model: model, headers: l.headers(),
		payload: JSONObject{"model": model, "prompt": toAnyList(prompts, func(p []int) any { return toAnyList(p, func(t int) any { return t }) }),
			"max_tokens": 1, "temperature": jsonFloat(1.0), "logprobs": 0, "return_tokens_as_token_ids": true,
			"logprob_token_ids": toAnyList(sorted, func(t int) any { return t })}})
}

// judgmentScoresFromBody: per prompt, the measured token log-probs;
// missing ids follow MAP-14 §4.
func (l *OpenAIChatLM) judgmentScoresFromBody(resp *HTTPResponse, nPrompts int) ([]map[int]float64, Usage, string, error) {
	data, err := resp.JSON()
	if err != nil {
		return nil, Usage{}, "", l.replyError(resp, err)
	}
	choices, ok := data["choices"].([]any)
	if !ok || len(choices) != nPrompts {
		return nil, Usage{}, "", l.judgmentReplyError(resp, "choices must contain every prompt index exactly once")
	}
	byIndex := make([]map[string]any, nPrompts)
	for _, c := range choices {
		obj, ok := c.(map[string]any)
		if !ok {
			return nil, Usage{}, "", l.judgmentReplyError(resp, "choices must contain every prompt index exactly once")
		}
		index, ok := jsonInteger(obj["index"])
		if !ok || index < 0 || index >= nPrompts || byIndex[index] != nil {
			return nil, Usage{}, "", l.judgmentReplyError(resp, "choices must contain every prompt index exactly once")
		}
		byIndex[index] = obj
	}
	out := make([]map[int]float64, 0, nPrompts)
	for _, choice := range byIndex {
		var top map[string]any
		switch lp := choice["logprobs"].(type) {
		case nil:
		case map[string]any:
			switch tl := lp["top_logprobs"].(type) {
			case nil:
			case []any:
				if len(tl) == 0 {
					break
				}
				if len(tl) != 1 {
					return nil, Usage{}, "", l.judgmentReplyError(resp, "expected one top_logprobs object")
				}
				switch first := tl[0].(type) {
				case nil:
				case map[string]any:
					top = first
				default:
					return nil, Usage{}, "", l.judgmentReplyError(resp, "expected one top_logprobs object")
				}
			default:
				return nil, Usage{}, "", l.judgmentReplyError(resp, "expected one top_logprobs object")
			}
		default:
			return nil, Usage{}, "", l.judgmentReplyError(resp, "logprobs must be an object or null")
		}
		scores := map[int]float64{}
		for token, value := range top {
			if !strings.HasPrefix(token, "token_id:") {
				continue
			}
			id, err := strconv.Atoi(token[len("token_id:"):])
			if err != nil || id < 0 {
				continue
			}
			// -inf is a zero-likelihood token; NaN/+inf, positive log
			// probabilities, booleans and string coercions are not measurements.
			if _, isBool := value.(bool); isBool {
				return nil, Usage{}, "", l.judgmentReplyError(resp, "invalid log probability for "+token)
			}
			f, ferr := jsonFloat64(value, token)
			if ferr != nil || math.IsNaN(f) || f > 0 || math.IsInf(f, 1) {
				return nil, Usage{}, "", l.judgmentReplyError(resp, "invalid log probability for "+token)
			}
			scores[id] = f
		}
		out = append(out, scores)
	}
	usage := Usage{}
	if raw, present := data["usage"]; present && raw != nil {
		u, ok := raw.(map[string]any)
		if !ok {
			return nil, Usage{}, "", l.judgmentReplyError(resp, "usage must be an object or null")
		}
		var uerr error
		if usage.InputTokens, uerr = optInt(u, "prompt_tokens"); uerr != nil {
			return nil, Usage{}, "", l.judgmentReplyError(resp, "invalid usage: "+uerr.Error())
		}
		if usage.OutputTokens, uerr = optInt(u, "completion_tokens"); uerr != nil {
			return nil, Usage{}, "", l.judgmentReplyError(resp, "invalid usage: "+uerr.Error())
		}
	}
	model := ""
	if raw, present := data["model"]; present && raw != nil {
		s, ok := raw.(string)
		if !ok || s == "" {
			return nil, Usage{}, "", l.judgmentReplyError(resp, "model must be a non-empty string")
		}
		model = s
	}
	return out, usage, model, nil
}

type judgmentKeyRequests struct {
	open, closed *TransportRequest
}

type judgmentPlanEntry struct {
	judgment Judgment
	prefill  *TransportRequest
	keys     map[string]judgmentKeyRequests
}

// judgmentPlan: one entry per judgment, the tokenize requests it needs. Pure.
func (l *OpenAIChatLM) judgmentPlan(req *Request) ([]judgmentPlanEntry, error) {
	var plan []judgmentPlanEntry
	for _, j := range RequestJudgments(req) {
		msgs, err := l.judgmentMessages(req, j, judgmentPrefill)
		if err != nil {
			return nil, err
		}
		prefill, err := l.judgmentTokenizeRequest(req.Model, msgs, true)
		if err != nil {
			return nil, err
		}
		entry := judgmentPlanEntry{judgment: j, prefill: prefill, keys: map[string]judgmentKeyRequests{}}
		for _, k := range j.Keys {
			answer := judgmentPrefill + " " + k
			openMsgs, err := l.judgmentMessages(req, j, answer)
			if err != nil {
				return nil, err
			}
			open, err := l.judgmentTokenizeRequest(req.Model, openMsgs, true)
			if err != nil {
				return nil, err
			}
			closedMsgs, err := l.judgmentMessages(req, j, answer)
			if err != nil {
				return nil, err
			}
			closed, err := l.judgmentTokenizeRequest(req.Model, closedMsgs, false)
			if err != nil {
				return nil, err
			}
			entry.keys[k] = judgmentKeyRequests{open: open, closed: closed}
		}
		plan = append(plan, entry)
	}
	return plan, nil
}

type tokenPath []int

func (p tokenPath) key() string {
	parts := make([]string, len(p))
	for i, t := range p {
		parts[i] = strconv.Itoa(t)
	}
	return strings.Join(parts, ",")
}

func prefixOf(a, b []int) bool {
	if len(a) > len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// judgmentPaths: each key's token path after the prefill, terminator included.
func judgmentPaths(prefix []int, keys map[string][2][]int, order []string) (map[string]tokenPath, error) {
	paths := map[string]tokenPath{}
	for _, key := range order {
		open, closed := keys[key][0], keys[key][1]
		if !prefixOf(prefix, open) || !prefixOf(open, closed) {
			return nil, UnsupportedFeature("openai-chat", "config.response_format", "openai-chat: key %q does not tokenize as an extension of the prefill in this chat template; candidate-sequence scoring cannot place it (rename the key or use a provider that classifies natively)", key)
		}
		body := open[len(prefix):]
		var terminator []int
		if len(closed) > len(open) {
			terminator = closed[len(open) : len(open)+1]
		}
		if len(body) == 0 || len(terminator) == 0 {
			return nil, UnsupportedFeature("openai-chat", "config.response_format", "openai-chat: key %q yields no scorable tokens (empty key or no end-of-turn token in the template)", key)
		}
		paths[key] = append(append(tokenPath(nil), body...), terminator...)
	}
	return paths, nil
}

// judgmentNodes: every trie node (a path prefix) → the child tokens that
// follow it in some key path.
func judgmentNodes(paths map[string]tokenPath, order []string) (map[string]map[int]bool, []tokenPath) {
	nodes := map[string]map[int]bool{}
	var list []tokenPath
	for _, key := range order {
		seq := paths[key]
		for i := range seq {
			node := seq[:i]
			k := tokenPath(node).key()
			if nodes[k] == nil {
				nodes[k] = map[int]bool{}
				list = append(list, append(tokenPath(nil), node...))
			}
			nodes[k][seq[i]] = true
		}
	}
	return nodes, list
}

type tokenizedJudgment struct {
	judgment Judgment
	paths    map[string]tokenPath
	prefix   []int
}

func (l *OpenAIChatLM) judgmentFold(req *Request, per []tokenizedJudgment, tables []map[string]map[int]float64, usage Usage, model string, nNodes, tokenizeCalls int) (*Response, error) {
	value := JSONObject{}
	probabilities := map[string]map[string]float64{}
	coverage := JSONObject{}
	for i, tj := range per {
		raw := map[string]float64{}
		anyFinite := false
		for _, key := range tj.judgment.Keys {
			seq := tj.paths[key]
			sum := 0.0
			for k := range seq {
				sum += tables[i][tokenPath(seq[:k]).key()][seq[k]]
			}
			raw[key] = sum
			if !math.IsInf(sum, 0) && !math.IsNaN(sum) {
				anyFinite = true
			}
		}
		if !anyFinite {
			return nil, l.providerError(KindProvider, "judgment "+strconv.Quote(tj.judgment.Name)+" has zero likelihood for every declared key; cannot normalize", 0, "", "")
		}
		mass := 0.0
		for _, v := range raw {
			mass += math.Exp(v)
		}
		coverage[tj.judgment.Name] = jsonFloat(mass)
		dist := normalizeLogprobs(raw)
		probabilities[tj.judgment.Name] = dist
		best, bestP := "", math.Inf(-1)
		for _, key := range tj.judgment.Keys {
			if dist[key] > bestP {
				best, bestP = key, dist[key]
			}
		}
		switch tj.judgment.Kind {
		case JudgmentBoolean:
			value[tj.judgment.Name] = best == "true"
		case JudgmentOrdered:
			n, _ := strconv.Atoi(best)
			value[tj.judgment.Name] = n
		default:
			value[tj.judgment.Name] = best
		}
	}
	part := DataPart{Value: value, Probabilities: probabilities, Method: MethodCandidateSequenceLikelihood}
	if model == "" {
		model = req.Model
	}
	return &Response{
		Model:        model,
		Message:      Message{Role: RoleAssistant, Parts: []Part{part}},
		FinishReason: FinishStop,
		Usage:        usage.Normalize(),
		ProviderData: JSONObject{"coverage": coverage, "judgments": JSONObject{"nodes": nNodes, "tokenize_calls": tokenizeCalls, "method": MethodCandidateSequenceLikelihood}},
	}, nil
}

func (l *OpenAIChatLM) judgmentsViaTokenScoring(req *Request) bool {
	p := req.Config.Probabilities
	return l.scoresNamedTokens() && (p == ProbabilitiesIfAvailable || p == ProbabilitiesRequired) && len(RequestJudgments(req)) > 0
}

func (l *OpenAIChatLM) judgmentAdaptations() ([]Adaptation, error) {
	scope := newAdaptScope(l.adaptations, l.provider, false)
	if err := scope.clientSide("config.response_format", "each judgment is asked as a final user turn and every declared key is scored as a token path (candidate-sequence likelihood, MAP-14 §4) instead of a generated JSON object; questions are scored independently", nil, nil); err != nil {
		return nil, err
	}
	return scope.records, nil
}

func (l *OpenAIChatLM) completeOverride(ctx context.Context, req *Request) (*Response, bool, error) {
	if !l.judgmentsViaTokenScoring(req) {
		return nil, false, nil
	}
	resp, err := l.completeByTokenScoring(ctx, req)
	return resp, true, err
}

func (l *OpenAIChatLM) completeByTokenScoring(ctx context.Context, req *Request) (*Response, error) {
	adaptations, err := l.judgmentAdaptations()
	if err != nil {
		return nil, err
	}
	plan, err := l.judgmentPlan(req)
	if err != nil {
		return nil, err
	}
	var tokenized []tokenizedJudgment
	calls := 0
	for _, entry := range plan {
		resp, err := l.sendOK(ctx, entry.prefill, nil)
		if err != nil {
			return nil, err
		}
		prefix, err := l.judgmentTokensFromBody(resp)
		if err != nil {
			return nil, err
		}
		calls++
		keys := map[string][2][]int{}
		for _, k := range entry.judgment.Keys {
			pair := entry.keys[k]
			openResp, err := l.sendOK(ctx, pair.open, nil)
			if err != nil {
				return nil, err
			}
			open, err := l.judgmentTokensFromBody(openResp)
			if err != nil {
				return nil, err
			}
			closedResp, err := l.sendOK(ctx, pair.closed, nil)
			if err != nil {
				return nil, err
			}
			closed, err := l.judgmentTokensFromBody(closedResp)
			if err != nil {
				return nil, err
			}
			calls += 2
			keys[k] = [2][]int{open, closed}
		}
		paths, err := judgmentPaths(prefix, keys, entry.judgment.Keys)
		if err != nil {
			return nil, err
		}
		tokenized = append(tokenized, tokenizedJudgment{judgment: entry.judgment, paths: paths, prefix: prefix})
	}
	return l.judgmentScoreAndFold(ctx, req, tokenized, calls, adaptations)
}

func (l *OpenAIChatLM) judgmentScoreAndFold(ctx context.Context, req *Request, tokenized []tokenizedJudgment, calls int, adaptations []Adaptation) (*Response, error) {
	var prompts [][]int
	type meta struct {
		index int
		node  tokenPath
	}
	var metas []meta
	union := map[int]bool{}
	nodesPer := make([]map[string]map[int]bool, len(tokenized))
	for i, tj := range tokenized {
		nodes, list := judgmentNodes(tj.paths, tj.judgment.Keys)
		nodesPer[i] = nodes
		for _, node := range list {
			prompts = append(prompts, append(append([]int(nil), tj.prefix...), node...))
			metas = append(metas, meta{i, node})
			for child := range nodes[node.key()] {
				union[child] = true
			}
		}
	}
	ids := make([]int, 0, len(union))
	for t := range union {
		ids = append(ids, t)
	}
	scoreReq, err := l.judgmentScoreRequest(req.Model, prompts, ids)
	if err != nil {
		return nil, err
	}
	resp, err := l.sendOK(ctx, scoreReq, nil)
	if err != nil {
		return nil, err
	}
	scores, usage, model, err := l.judgmentScoresFromBody(resp, len(prompts))
	if err != nil {
		return nil, err
	}
	tables := make([]map[string]map[int]float64, len(tokenized))
	for i := range tables {
		tables[i] = map[string]map[int]float64{}
	}
	for n, m := range metas {
		got := scores[n]
		children := nodesPer[m.index][m.node.key()]
		row := map[int]float64{}
		for child := range children {
			v, present := got[child]
			if !present {
				return l.judgmentUnmeasured(ctx, req, adaptations)
			}
			row[child] = v
		}
		tables[m.index][m.node.key()] = row
	}
	out, err := l.judgmentFold(req, tokenized, tables, usage, model, len(prompts), calls)
	if err != nil {
		if e := AsError(err); e != nil {
			e.Status = resp.Status
			attachErrorMetadata(e, resp.Headers)
		}
		return nil, err
	}
	return l.finishResponse(req, out, adaptations), nil
}

// judgmentUnmeasured: the server answered 200 without the requested token
// ids: it dropped logprob_token_ids (receipted on vLLM 0.25.1). required
// refuses; if_available answers by structured output instead and records it.
func (l *OpenAIChatLM) judgmentUnmeasured(ctx context.Context, req *Request, adaptations []Adaptation) (*Response, error) {
	if req.Config.Probabilities == ProbabilitiesRequired {
		return nil, UnsupportedFeature(l.provider, "config.probabilities", "%s: config.probabilities='required' but this server ignored logprob_token_ids (vLLM < 0.29?); no distribution can be measured here", l.provider)
	}
	scope := newAdaptScope(l.adaptations, l.provider, false)
	if err := scope.dropped("config.probabilities", "the server accepted the request and returned no log-probs for the requested token ids (logprob_token_ids ignored); answered by structured output instead", req.Config.Probabilities); err != nil {
		return nil, err
	}
	wire, built, err := l.build(req, false, false)
	if err != nil {
		return nil, err
	}
	resp, err := l.sendOK(ctx, wire, nil)
	if err != nil {
		return nil, err
	}
	parsed, err := l.parseResponse(req, resp)
	if err != nil {
		return nil, err
	}
	records := scope.records
	for _, a := range built {
		if a.Field != "config.probabilities" {
			records = append(records, a)
		}
	}
	return l.finishResponse(req, parsed, records), nil
}
