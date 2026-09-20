package lm15

import (
	"strings"
	"testing"
)

func typesafeRequest(t *testing.T) *Request {
	t.Helper()
	choice, err := Choice("Pick", Options("a", "b")...)
	if err != nil {
		t.Fatal(err)
	}
	format, err := Judgments("test", true, JudgmentProperty{Name: "choice", Schema: choice}, JudgmentProperty{Name: "yes", Schema: YesNo("Is it?")})
	if err != nil {
		t.Fatal(err)
	}
	return &Request{Model: "jev", Messages: []Message{UserParts(Data(JSONObject{"empty": nil, "list": []any{1, "two"}}))}, Config: Config{ResponseFormat: format}}
}

func TestTypeSafeStateAndMeasurements(t *testing.T) {
	lm, err := NewTypeSafeLM(WithAPIKey("offline-placeholder"))
	if err != nil {
		t.Fatal(err)
	}
	req := typesafeRequest(t)
	wire, err := lm.BuildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	body, err := DecodeJSONObject(wire.Body)
	if err != nil {
		t.Fatal(err)
	}
	state := body["state"].(map[string]any)
	if _, ok := state["empty"]; !ok || len(state["list"].([]any)) != 2 {
		t.Fatalf("state was altered: %s", wire.Body)
	}
	valid := `{"answers":{"choice":{"type":"choice","choice":"a","probabilities":{"a":0.7,"b":0.4}},"yes":{"type":"noul","noul":0.8}}}`
	// Rounded/non-normalized measurements are accepted without renormalizing.
	resp, err := lm.ParseResponse(req, &HTTPResponse{Status: 200, Body: []byte(valid)})
	if err != nil {
		t.Fatal(err)
	}
	part, ok := resp.DataPart()
	if !ok || part.Probabilities["choice"]["a"] != 0.7 || part.Probabilities["choice"]["b"] != 0.4 {
		t.Fatalf("measurements changed: %+v", part)
	}
	for _, bad := range []string{"null", "true", `"0.8"`, "-0.1", "1.1", "1e999", "[]", "{}"} {
		t.Run(bad, func(t *testing.T) {
			raw := strings.Replace(valid, `"noul":0.8`, `"noul":`+bad, 1)
			_, err := lm.ParseResponse(req, &HTTPResponse{Status: 200, Body: []byte(raw), Headers: [][2]string{{"x-typesafe-request-id", "req-test"}}})
			e := AsError(err)
			if e == nil || e.Kind != KindProvider || e.Status != 200 || e.RequestID != "req-test" {
				t.Fatalf("wanted typed malformed reply with diagnostics: %v", err)
			}
		})
	}
	for _, raw := range []string{
		strings.Replace(valid, `"a":0.7,`, "", 1),
		strings.Replace(valid, `"b":0.4`, `"c":0.4`, 1),
		strings.Replace(valid, `"choice":"a"`, `"choice":"c"`, 1),
		strings.Replace(valid, `"type":"noul"`, `"type":"score"`, 1),
		`{"answers":[]}`,
		strings.TrimSuffix(valid, "}") + `,"usage":{"input_tokens":-1}}`,
	} {
		if _, err := lm.ParseResponse(req, &HTTPResponse{Status: 200, Body: []byte(raw)}); !IsKind(err, KindProvider) {
			t.Fatalf("accepted malformed reply %s: %v", raw, err)
		}
	}
}

func TestDataMeasurementsOnlyOnAssistant(t *testing.T) {
	measured := DataWithProbabilities(true, map[string]map[string]float64{"yes": {"true": 0.8, "false": 0.2}}, MethodProviderClassification)
	if err := (Message{Role: RoleAssistant, Parts: []Part{measured}}).Validate(); err != nil {
		t.Fatal(err)
	}
	checks := []struct {
		name     string
		validate func() error
	}{
		{"user", UserParts(measured).Validate},
		{"system", SystemParts(measured).Validate},
		{"tool", ToolResultParts("call", measured).Validate},
	}
	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.validate(); err == nil {
				t.Fatal("accepted assistant measurement as input")
			}
		})
	}
	if err := SystemParts(Data(true)).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := ToolResultParts("call", Data(true)).Validate(); err != nil {
		t.Fatal(err)
	}
}
