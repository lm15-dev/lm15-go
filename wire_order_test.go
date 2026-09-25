package lm15

import (
	"strings"
	"testing"
	"time"
)

// The behavior the ordered JSONObject exists for: a model fills structured
// output in the order the schema lists its properties, so a schema that
// asks for "reasoning" before "answer" must reach every provider in that
// order. "answer" sorts first, so a port that writes objects sorted (Go
// before 2026-09-25) fails here on every dialect. Tool parameters are
// checked the same way ("zeta" before "alpha").
func TestEveryDialectSendsSchemaAndToolPropertiesInTheirOrder(t *testing.T) {
	schema, err := DecodeJSONObject([]byte(`{"type":"object","properties":{"reasoning":{"type":"string","description":"Think first."},"answer":{"type":"string"}},"required":["reasoning","answer"],"additionalProperties":false}`))
	if err != nil {
		t.Fatal(err)
	}
	params := JSONObject{
		KV("type", "object"),
		KV("properties", JSONObject{KV("zeta", JSONObject{KV("type", "string")}), KV("alpha", JSONObject{KV("type", "integer")})}),
		KV("required", []any{"zeta", "alpha"}),
	}
	format := JSONObject{KV("type", "json_schema"), KV("name", "verdict"), KV("schema", schema)}
	clock := func() time.Time { return time.Date(2026, 9, 25, 0, 0, 0, 0, time.UTC) }
	settings := map[string]string{"region": "us-east-1", "project": "p", "location": "us-central1", "resource": "r", "endpoint": "https://e.example"}

	checked := map[string]bool{}
	for _, provider := range ProviderIDs() {
		lm, err := AdapterForProvider(provider, "test-key", "", settings, clock)
		if err != nil {
			continue // a door that needs a credential of another kind
		}
		for _, withTool := range []bool{false, true} {
			req := &Request{Model: "m", Messages: []Message{UserMessage("hi")}}
			if withTool {
				req.Tools = []Tool{FunctionTool{Name: "f", Parameters: params}}
			} else {
				req.Config.ResponseFormat = format
			}
			wire, _, err := lm.Build(req, false)
			if err != nil || wire == nil {
				continue // this dialect refuses the feature, loudly: not an order question
			}
			body := string(wire.Body)
			first, second := `"reasoning":{`, `"answer":{`
			if withTool {
				first, second = `"zeta":{`, `"alpha":{`
			}
			i, j := strings.Index(body, first), strings.Index(body, second)
			if i < 0 || j < 0 {
				continue // the wire carries the schema elsewhere (not in this body)
			}
			if i > j {
				t.Errorf("%s: %s is sent after %s:\n%s", provider, first, second, body)
			}
			checked[provider] = true
		}
		lm.Close()
	}
	for _, must := range []string{"openai", "anthropic", "gemini", "openai-chat", "xai", "groq"} {
		if !checked[must] {
			t.Errorf("%s was not checked: its body carried no schema", must)
		}
	}
	if len(checked) < 15 {
		t.Errorf("only %d providers checked", len(checked))
	}
}
