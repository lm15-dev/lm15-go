package lm15

import (
	"os"
	"testing"
)

// MAP-16 over the contract's vectors, on every Gemini surface. The
// harness's mapping direction grades generateContent and cached prefixes
// through the vet shim; the Live setup frame, and Go values a caller wrote
// by hand, are covered here.
func TestGeminiSchemaFieldVectors(t *testing.T) {
	raw, err := os.ReadFile("../lm15-contract/mapping/gemini-schema-field.json")
	if err != nil {
		t.Skip("contract mapping vectors not present:", err)
	}
	doc, err := DecodeJSONObject(raw)
	if err != nil {
		t.Fatal(err)
	}
	lm, err := NewGeminiLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range doc.Get("cases").([]any) {
		v := c.(JSONObject)
		id, schema, openapi := v.Get("id").(string), v.Get("schema").(JSONObject), v.Get("openapi").(bool)
		if geminiOpenAPISchema(schema) != openapi {
			t.Fatalf("%s: rule says %v", id, !openapi)
		}
		want, other := "parameters", "parametersJsonSchema"
		if !openapi {
			want, other = other, want
		}
		tool := FunctionTool{Name: "f", Parameters: schema}
		wire, err := lm.BuildRequest(&Request{Model: "gemini-2.5-flash", Messages: []Message{UserMessage("x")}, Tools: []Tool{tool}}, false)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		body, _ := DecodeJSONObject(wire.Body)
		decl := wireObj(wireList(wireObj(wireList(body.Get("tools"))[0]).Get("functionDeclarations"))[0])
		if decl.Has(other) || string(mustJSON(decl.Get(want))) != string(mustJSON(schema)) {
			t.Fatalf("%s: generateContent declaration %s", id, mustJSON(decl))
		}
		frames, err := lm.liveSetupFrames(&LiveConfig{Model: "gemini-3.1-flash-live-preview", Tools: []Tool{tool}})
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		setup := wireObj(frames[0].Get("setup"))
		decl = wireObj(wireList(wireObj(wireList(setup.Get("tools"))[0]).Get("functionDeclarations"))[0])
		if decl.Has(other) || string(mustJSON(decl.Get(want))) != string(mustJSON(schema)) {
			t.Fatalf("%s: Live declaration %s", id, mustJSON(decl))
		}
	}
}

func TestGeminiSchemaFieldReadsGoValues(t *testing.T) {
	obj := func(p any) JSONObject {
		return JSONObject{{"type", "object"}, {"properties", JSONObject{{"v", p}}}}
	}
	for name, c := range map[string]struct {
		schema  any
		openapi bool
	}{
		"[]int enum":          {obj(JSONObject{{"type", "integer"}, {"enum", []int{1, 2}}}), false},
		"[]string enum":       {obj(JSONObject{{"type", "string"}, {"enum", []string{"a"}}}), true},
		"[]string type":       {obj(JSONObject{{"type", []string{"string", "null"}}}), false},
		"nested Go map":       {obj(map[string]any{"type": "string", "const": "x"}), false},
		"nested Go map plain": {obj(map[string]any{"type": "string"}), true},
		"bool sub-schema":     {obj(true), false},
	} {
		if got := geminiOpenAPISchema(c.schema); got != c.openapi {
			t.Errorf("%s: %v", name, got)
		}
	}
}
