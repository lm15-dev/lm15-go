package lm15

// Bounding-box judgments (changes/2026-10-15-bounding-box.md), offline: the
// schema convention, the helper, Jev's Box on the typesafe wire, the
// probability policy, and the token-trie driver that generates a box.

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"strings"
	"sync"
	"testing"
)

func boxFormats(t *testing.T) (find, only JSONObject) {
	t.Helper()
	find, err := Judgments("judgments", true,
		JudgmentProperty{Name: "mug", Schema: BoundingBox("the coffee mug")},
		JudgmentProperty{Name: "mug_visible", Schema: YesNo("Is there a coffee mug in the image?")})
	if err != nil {
		t.Fatal(err)
	}
	only, err = Judgments("judgments", true, JudgmentProperty{Name: "mug", Schema: BoundingBox("the coffee mug")})
	if err != nil {
		t.Fatal(err)
	}
	return find, only
}

func photo() []Message {
	return []Message{UserParts(Image(WithBase64("iVBORw0KGgo="), WithMediaType("image/png")))}
}

func TestBoundingBoxHelperEmitsFourDescribedNumbersAndNoBounds(t *testing.T) {
	prop := BoundingBox("the coffee mug")
	if prop.Get("type") != "object" || prop.Get("description") != "the coffee mug" || prop.Get("additionalProperties") != false {
		t.Fatalf("box = %v", prop)
	}
	edges := prop.Get("properties").(JSONObject)
	if strings.Join(edges.Keys(), ",") != "top,bottom,left,right" {
		t.Fatalf("edge order %v", edges.Keys())
	}
	if req := prop.Get("required").([]any); len(req) != 4 || req[0] != "top" || req[3] != "right" {
		t.Fatalf("required %v", req)
	}
	for _, name := range edges.Keys() {
		edge := edges.Get(name).(JSONObject)
		if edge.Get("type") != "number" || !strings.Contains(edge.Get("description").(string), "fraction") {
			t.Fatalf("%s = %v", name, edge)
		}
		// Anthropic answers 400 on minimum/maximum (receipted 2026-10-15).
		if edge.Has("minimum") || edge.Has("maximum") {
			t.Fatalf("%s carries bounds", name)
		}
	}
}

func TestABoxIsReadByShapeAndDeclaresNoKeys(t *testing.T) {
	find, _ := boxFormats(t)
	found := JudgmentsInSchema(find.Get("schema"))
	if len(found) != 2 || found[0].Kind != JudgmentBox || len(found[0].Keys) != 0 || found[0].Instruction != "the coffee mug" || found[1].Kind != JudgmentBoolean {
		t.Fatalf("found %+v", found)
	}
	if m := MeasurableJudgments(found); len(m) != 1 || m[0].Name != "mug_visible" {
		t.Fatalf("measurable %+v", m)
	}
	edge := JSONObject{{"type", "number"}}
	all := func(s any) JSONObject {
		return JSONObject{{"top", s}, {"bottom", s}, {"left", s}, {"right", s}}
	}
	schema := JSONObject{{"type", "object"}, {"properties", JSONObject{
		{"hand_written", JSONObject{{"properties", JSONObject{{"left", edge}, {"top", edge}, {"right", edge}, {"bottom", edge}}}}},
		{"bounded", JSONObject{{"type", "object"}, {"properties", all(JSONObject{{"type", "number"}, {"minimum", 0}, {"maximum", 1}})}}},
		{"extra_edge", JSONObject{{"type", "object"}, {"properties", append(all(edge), Member{"label", JSONObject{{"type", "string"}}})}}},
		{"integer_edges", JSONObject{{"type", "object"}, {"properties", all(JSONObject{{"type", "integer"}})}}},
		{"three_edges", JSONObject{{"type", "object"}, {"properties", JSONObject{{"top", edge}, {"bottom", edge}, {"left", edge}}}}},
		{"array", JSONObject{{"type", "array"}, {"properties", all(edge)}}},
		{"pixels", JSONObject{{"type", "object"}, {"properties", JSONObject{{"x", edge}, {"y", edge}, {"w", edge}, {"h", edge}}}}},
	}}}
	var kinds []string
	for _, j := range JudgmentsInSchema(schema) {
		kinds = append(kinds, j.Name+"="+string(j.Kind))
	}
	if strings.Join(kinds, ",") != "hand_written=box,bounded=box" {
		t.Fatalf("read %v", kinds)
	}
	// The rewriting wires leave a box untouched.
	s := find.Get("schema").(JSONObject)
	want := string(mustJSON(s.Get("properties").(JSONObject).Get("mug")))
	for _, got := range []JSONObject{anthropicSchema(s, found), geminiSchema(s, found)} {
		if g := string(mustJSON(got.Get("properties").(JSONObject).Get("mug"))); g != want {
			t.Fatalf("box rewritten: %s", g)
		}
	}
}

func TestTypeSafeSendsABoxQuestionWithTheOptInHeader(t *testing.T) {
	lm, err := NewTypeSafeLM(WithAPIKey("offline-placeholder"))
	if err != nil {
		t.Fatal(err)
	}
	find, only := boxFormats(t)
	wire, err := lm.BuildRequest(&Request{Model: "jev-preview", Messages: photo(), Config: Config{ResponseFormat: find}}, false)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := DecodeJSONObject(wire.Body)
	if got := string(mustJSON(body.Get("questions"))); got != `{"mug":{"type":"box","instructions":"the coffee mug"},"mug_visible":{"type":"noul","instructions":"Is there a coffee mug in the image?"}}` {
		t.Fatalf("questions %s", got)
	}
	if wire.Header("TypeSafe-Danger-Zone-Images") != "true" {
		t.Fatal("header missing with a box")
	}
	// A box with no image still needs the header (live 400 without it, 2026-10-15).
	wire, err = lm.BuildRequest(&Request{Model: "jev-preview", Messages: []Message{UserParts(Text("a kitchen"))}, Config: Config{ResponseFormat: only}}, false)
	if err != nil {
		t.Fatal(err)
	}
	body, _ = DecodeJSONObject(wire.Body)
	if wire.Header("TypeSafe-Danger-Zone-Images") != "true" || body.Has("assets") {
		t.Fatalf("text-only box: header=%q assets=%v", wire.Header("TypeSafe-Danger-Zone-Images"), body.Has("assets"))
	}
}

func TestTypeSafeAnswersTheBoxVerbatimWithoutADistribution(t *testing.T) {
	lm, err := NewTypeSafeLM(WithAPIKey("offline-placeholder"))
	if err != nil {
		t.Fatal(err)
	}
	find, only := boxFormats(t)
	req := &Request{Model: "jev-preview", Messages: photo(), Config: Config{ResponseFormat: find, Probabilities: ProbabilitiesRequired}}
	body := `{"model":"jev-1.14.0","answers":{"mug":{"type":"box","box":{"top":0.25,"bottom":0.82,"left":0.44,"right":1},"confidence":0.92},"mug_visible":{"type":"noul","noul":0.97}}}`
	resp, err := lm.ParseResponse(req, &HTTPResponse{Status: 200, Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(mustJSON(resp.Data())); got != `{"mug":{"top":0.25,"bottom":0.82,"left":0.44,"right":1},"mug_visible":true}` {
		t.Fatalf("data %s", got)
	}
	part, _ := resp.DataPart()
	if len(part.Probabilities) != 1 || part.Probabilities["mug_visible"] == nil || part.Method != MethodProviderClassification {
		t.Fatalf("probabilities %+v", part)
	}
	if !strings.Contains(string(mustJSON(resp.ProviderData)), `"confidence":0.92`) {
		t.Fatal("confidence not kept in provider_data")
	}
	onlyReq := &Request{Model: "jev-preview", Messages: photo(), Config: Config{ResponseFormat: only}}
	resp, err = lm.ParseResponse(onlyReq, &HTTPResponse{Status: 200, Body: []byte(`{"answers":{"mug":{"type":"box","box":{"top":0.25,"bottom":0.82,"left":0.44,"right":1}}}}`)})
	if err != nil {
		t.Fatal(err)
	}
	if part, _ := resp.DataPart(); part.Probabilities != nil || part.Method != "" {
		t.Fatalf("a box carries no distribution: %+v", part)
	}
	for _, c := range []struct{ box, where string }{
		{`{"top":0.1,"bottom":0.2,"left":0.3}`, "answers.mug.box"},
		{`{"top":0.1,"bottom":0.2,"left":0.3,"right":0.4,"x":0}`, "answers.mug.box"},
		{`{"top":0.1,"bottom":1.2,"left":0.3,"right":0.4}`, "answers.mug.box.bottom"},
		{`{"top":"0.1","bottom":0.2,"left":0.3,"right":0.4}`, "answers.mug.box.top"},
		{`[0.1,0.2,0.3,0.4]`, "answers.mug.box"},
	} {
		_, err := lm.ParseResponse(onlyReq, &HTTPResponse{Status: 200, Body: []byte(`{"answers":{"mug":{"type":"box","box":` + c.box + `}}}`)})
		if !IsKind(err, KindProvider) || !strings.Contains(err.Error(), c.where) {
			t.Fatalf("%s: %v", c.box, err)
		}
	}
}

func TestProbabilityPolicyConcernsOnlyKeyedJudgments(t *testing.T) {
	find, only := boxFormats(t)
	for _, policy := range []string{ProbabilitiesIfAvailable, ProbabilitiesRequired} {
		scope := newAdaptScope("note", "anthropic", false)
		if err := noteUnmeasurableProbabilities(scope, &Request{Config: Config{ResponseFormat: only, Probabilities: policy}}, "anthropic"); err != nil || len(scope.records) != 0 {
			t.Fatalf("%s: err=%v records=%v", policy, err, scope.records)
		}
	}
	scope := newAdaptScope("note", "anthropic", false)
	if err := noteUnmeasurableProbabilities(scope, &Request{Config: Config{ResponseFormat: find, Probabilities: ProbabilitiesRequired}}, "anthropic"); err == nil {
		t.Fatal("required with a yes/no on a pick-only wire must refuse")
	}
}

// scoringServer: /tokenize, one scoring /v1/completions, and the generated /chat/completions.
type scoringServer struct {
	mu        sync.Mutex
	urls      []string
	generated string
}

func (s *scoringServer) Do(_ context.Context, req *TransportRequest) (*TransportResponse, error) {
	s.mu.Lock()
	s.urls = append(s.urls, req.URL)
	s.mu.Unlock()
	var payload map[string]any
	_ = json.Unmarshal(req.Body, &payload)
	reply := func(v any) (*TransportResponse, error) {
		b, _ := json.Marshal(v)
		return &TransportResponse{Status: 200, Headers: [][2]string{{"content-type", "application/json"}}, Body: io.NopCloser(strings.NewReader(string(b)))}, nil
	}
	switch {
	case strings.HasSuffix(req.URL, "/tokenize"):
		msgs := payload["messages"].([]any)
		answer := msgs[len(msgs)-1].(map[string]any)["content"].(string)
		tokens := []int{100}
		for _, w := range strings.Fields(answer) {
			if w == "Answer:" {
				tokens = append(tokens, 1)
			} else {
				tokens = append(tokens, int(w[0]))
			}
		}
		if cont, _ := payload["continue_final_message"].(bool); !cont {
			tokens = append(tokens, 999)
		}
		return reply(map[string]any{"tokens": tokens})
	case strings.HasSuffix(req.URL, "/v1/completions"):
		prompts := payload["prompt"].([]any)
		top := map[string]float64{}
		for _, id := range payload["logprob_token_ids"].([]any) {
			n := int(id.(float64))
			p := 0.05
			if n == int('t') {
				p = 0.9
			}
			top["token_id:"+itoa(n)] = math.Log(p)
		}
		var choices []any
		for i := range prompts {
			choices = append(choices, map[string]any{"index": i, "logprobs": map[string]any{"top_logprobs": []any{top}}})
		}
		return reply(map[string]any{"model": "lfm", "choices": choices, "usage": map[string]any{"prompt_tokens": 40, "completion_tokens": len(prompts)}})
	}
	return reply(map[string]any{"id": "g", "model": "lfm", "choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": s.generated}, "finish_reason": "stop"}}, "usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 2}})
}

func TestTrieMeasuresTheYesNoAndGenerationAnswersTheBox(t *testing.T) {
	find, only := boxFormats(t)
	box := `{"top":0.2,"bottom":0.6,"left":0.55,"right":0.85}`
	server := &scoringServer{generated: `{"mug":` + box + `,"mug_visible":false}`}
	lm, err := NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset("vllm"), WithBaseURL("http://scoring/v1"), WithTransport(server))
	if err != nil {
		t.Fatal(err)
	}
	req := &Request{Model: "lfm", Messages: []Message{UserParts(Text("a kitchen"))}, Config: Config{ResponseFormat: find, Probabilities: ProbabilitiesRequired}}
	resp, err := lm.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(mustJSON(resp.Data())); got != `{"mug":`+box+`,"mug_visible":true}` { // measured, overriding the generated pick
		t.Fatalf("data %s", got)
	}
	part, _ := resp.DataPart()
	if len(part.Probabilities) != 1 || part.Method != MethodCandidateSequenceLikelihood {
		t.Fatalf("part %+v", part)
	}
	completions := 0
	for _, u := range server.urls {
		if strings.HasSuffix(u, "completions") {
			completions++
		}
	}
	if completions != 2 || !strings.HasSuffix(server.urls[len(server.urls)-1], "/chat/completions") {
		t.Fatalf("calls %v", server.urls)
	}
	// A box missing from the generated answer is a malformed reply, never a guess.
	missing := &scoringServer{generated: `{"mug_visible":false}`}
	lm, _ = NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset("vllm"), WithBaseURL("http://scoring/v1"), WithTransport(missing))
	if _, err := lm.Complete(context.Background(), req); !IsKind(err, KindProvider) || !strings.Contains(err.Error(), "mug") {
		t.Fatalf("wanted a provider fault naming the box: %v", err)
	}
	// Boxes only: nothing to measure, one ordinary generated call.
	plain := &scoringServer{generated: `{"mug":` + box + `}`}
	lm, _ = NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset("vllm"), WithBaseURL("http://scoring/v1"), WithTransport(plain))
	resp, err = lm.Complete(context.Background(), &Request{Model: "lfm", Messages: []Message{UserParts(Text("a kitchen"))}, Config: Config{ResponseFormat: only, Probabilities: ProbabilitiesRequired}})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(mustJSON(resp.Data())); got != `{"mug":`+box+`}` || len(plain.urls) != 1 {
		t.Fatalf("data %s calls %v", got, plain.urls)
	}
}

// The same driver answers ordinary structured output beside measured
// judgments; before 2026-10-15 Go's trie dropped such properties silently.
func TestTrieAnswersOrdinaryPropertiesInsteadOfDroppingThem(t *testing.T) {
	format, err := Judgments("judgments", true,
		JudgmentProperty{Name: "ok", Schema: YesNo("Fine?")},
		JudgmentProperty{Name: "why", Schema: JSONObject{{"type", "string"}}})
	if err != nil {
		t.Fatal(err)
	}
	server := &scoringServer{generated: `{"ok":false,"why":"ordinary text"}`}
	lm, _ := NewOpenAIChatLM(WithAPIKey("k"), WithCompatPreset("vllm"), WithBaseURL("http://scoring/v1"), WithTransport(server))
	resp, err := lm.Complete(context.Background(), &Request{Model: "lfm", Messages: []Message{UserParts(Text("fine"))}, Config: Config{ResponseFormat: format, Probabilities: ProbabilitiesRequired}})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(mustJSON(resp.Data())); got != `{"ok":true,"why":"ordinary text"}` {
		t.Fatalf("data %s", got)
	}
	if !strings.Contains(string(mustJSON(resp.ProviderData)), `"generated_response"`) || resp.Usage.InputTokens == nil || *resp.Usage.InputTokens != 43 {
		t.Fatalf("usage %+v provider_data %s", resp.Usage, mustJSON(resp.ProviderData))
	}
}
