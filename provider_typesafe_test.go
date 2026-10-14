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
	return &Request{Model: "jev", Messages: []Message{UserParts(Data(JSONObject{{"empty", nil}, {"list", []any{1, "two"}}}))}, Config: Config{ResponseFormat: format}}
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
	state := body.Get("state").(JSONObject)
	if _, ok := state.Lookup("empty"); !ok || len(state.Get("list").([]any)) != 2 {
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

// changes/2026-10-14-jev-images.md D1-D4: an image is its reference in the
// state, its bytes an asset, and the request carries the opt-in header.
func TestTypeSafeImagesAreAssetsNamedInTheState(t *testing.T) {
	lm, err := NewTypeSafeLM(WithAPIKey("offline-placeholder"))
	if err != nil {
		t.Fatal(err)
	}
	const png, jpg = "iVBORw0KGgo=", "/9j/4AAQ"
	build := func(parts ...Part) (*TransportRequest, JSONObject, error) {
		req := typesafeRequest(t)
		req.Messages = []Message{UserParts(parts...)}
		wire, err := lm.BuildRequest(req, false)
		if err != nil {
			return nil, nil, err
		}
		body, err := DecodeJSONObject(wire.Body)
		if err != nil {
			t.Fatal(err)
		}
		return wire, body, nil
	}
	wire, body, err := build(Image(WithBase64(png), WithMediaType("image/png")))
	if err != nil {
		t.Fatal(err)
	}
	if got := wire.Header("TypeSafe-Danger-Zone-Images"); got != "true" {
		t.Fatalf("images header = %q", got)
	}
	if keys := body.Keys(); strings.Join(keys, ",") != "model,state,assets,questions" {
		t.Fatalf("payload keys %v", keys)
	}
	if body.Get("state") != "<lm15-image-1>" {
		t.Fatalf("state = %v", body.Get("state"))
	}
	asset := body.Get("assets").(JSONObject).Get("<lm15-image-1>").(JSONObject)
	if asset.Get("type") != "image" || asset.Get("format") != "png" || asset.Get("data_base64") != png {
		t.Fatalf("asset = %v", asset)
	}
	_, body, err = build(Text("A red ball."), Image(WithBase64(png), WithMediaType("image/png")), Data(JSONObject{{"k", 1}}), Image(WithBase64(jpg), WithMediaType("image/jpeg")))
	if err != nil {
		t.Fatal(err)
	}
	state := body.Get("state").([]any)
	if len(state) != 4 || state[0] != "A red ball." || state[1] != "<lm15-image-1>" || state[3] != "<lm15-image-2>" {
		t.Fatalf("state = %v", state)
	}
	if f := body.Get("assets").(JSONObject).Get("<lm15-image-2>").(JSONObject).Get("format"); f != "jpg" {
		t.Fatalf("second format = %v", f)
	}
	wire, body, err = build(Text("no picture"))
	if err != nil {
		t.Fatal(err)
	}
	if wire.Header("TypeSafe-Danger-Zone-Images") != "" || body.Has("assets") {
		t.Fatal("a request without an image must not carry the header or assets")
	}
	for _, c := range []struct {
		part Part
		why  string
	}{
		{Image(WithURL("https://example.com/a.png")), "not a URL"},
		{Image(WithFileID("file_1")), "not a file id"},
		{Image(WithBase64("UklGRg=="), WithMediaType("image/webp")), "PNG or JPEG"},
		{Audio(WithBase64("UklGRg==")), "no slot on the systemone wire"},
	} {
		_, _, err := build(Text("x"), c.part)
		e := AsError(err)
		if e == nil || e.Feature != "messages[0].parts[1]" || !strings.Contains(e.Error(), c.why) {
			t.Fatalf("%s: got %v", c.why, err)
		}
	}
}
