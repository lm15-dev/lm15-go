package lm15

import "testing"

// MAP-17: a function tool with no description reaches every wire with no
// description key, never "description": null. The contract's
// tool_no_description cases pin the absent wire through the vet shim; these
// tests add the paths no case pins (a Gemini cached prefix, a batch body,
// a present description's slot). In Go an absent and an empty description
// are the same value (""), so one test covers both.

func toolDeclarations(v any, out *[]JSONObject) {
	switch x := v.(type) {
	case JSONObject:
		if x.Get("name") == "get_weather" && (x.Has("parameters") || x.Has("input_schema") || x.Has("parametersJsonSchema")) {
			*out = append(*out, x)
		}
		for _, m := range x {
			toolDeclarations(m.Value, out)
		}
	case []any:
		for _, e := range x {
			toolDeclarations(e, out)
		}
	case []JSONObject:
		for _, e := range x {
			toolDeclarations(e, out)
		}
	}
}

func onlyToolDeclaration(t *testing.T, label string, v any) JSONObject {
	t.Helper()
	var found []JSONObject
	toolDeclarations(v, &found)
	if len(found) != 1 {
		t.Fatalf("%s: %d declarations in %s", label, len(found), mustJSON(v))
	}
	return found[0]
}

func weatherTool(description string) FunctionTool {
	return FunctionTool{Name: "get_weather", Description: description,
		Parameters: JSONObject{{"type", "object"}, {"properties", JSONObject{{"city", JSONObject{{"type", "string"}}}}}}}
}

func wireBody(t *testing.T, label string, wire *TransportRequest, err error) JSONObject {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	body, err := DecodeJSONObject(wire.Body)
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	return body
}

func TestToolWithoutDescriptionCarriesNoDescriptionKey(t *testing.T) {
	for _, provider := range []string{"anthropic", "openai", "openai-chat", "gemini", "groq", "deepseek"} {
		lm, err := AdapterForProvider(provider, "k", "", nil, nil)
		if err != nil {
			t.Fatal(provider, err)
		}
		req := &Request{Model: "deepseek-v4-flash", Messages: []Message{UserMessage("hi")}, Tools: []Tool{weatherTool("")}}
		wire, err := lm.BuildRequest(req, false)
		decl := onlyToolDeclaration(t, provider, wireBody(t, provider, wire, err))
		if decl.Has("description") {
			t.Fatalf("%s: %s", provider, mustJSON(decl))
		}
		keys := decl.Keys()
		if keys[0] == "type" {
			keys = keys[1:]
		}
		if keys[0] != "name" {
			t.Fatalf("%s: key order %v", provider, decl.Keys())
		}
	}
}

func TestToolDescriptionKeepsItsSlotAfterTheName(t *testing.T) {
	for _, provider := range []string{"anthropic", "openai", "openai-chat", "gemini"} {
		lm, err := AdapterForProvider(provider, "k", "", nil, nil)
		if err != nil {
			t.Fatal(provider, err)
		}
		req := &Request{Model: "m-1", Messages: []Message{UserMessage("hi")}, Tools: []Tool{weatherTool("Weather for a city")}}
		wire, err := lm.BuildRequest(req, false)
		decl := onlyToolDeclaration(t, provider, wireBody(t, provider, wire, err))
		keys := decl.Keys()
		for i, k := range keys {
			if k == "name" && (i+1 >= len(keys) || keys[i+1] != "description") {
				t.Fatalf("%s: key order %v", provider, keys)
			}
		}
		if decl.Get("description") != "Weather for a city" {
			t.Fatalf("%s: %s", provider, mustJSON(decl))
		}
	}
}

func TestToolWithoutDescriptionOnLiveCacheAndBatch(t *testing.T) {
	openai, _ := NewOpenAILM(WithAPIKey("k"))
	gemini, _ := NewGeminiLM(WithAPIKey("k"))
	anthropic, _ := NewAnthropicLM(WithAPIKey("k"))
	for label, lm := range map[string]interface {
		LiveSetupFrames(*LiveConfig) ([]JSONObject, error)
	}{"gpt-realtime-mini": openai, "gemini-3.1-flash-live-preview": gemini} {
		frames, err := lm.LiveSetupFrames(&LiveConfig{Model: label, Tools: []Tool{weatherTool("")}})
		if err != nil {
			t.Fatal(label, err)
		}
		if decl := onlyToolDeclaration(t, label, frames); decl.Has("description") {
			t.Fatalf("%s live: %s", label, mustJSON(decl))
		}
	}
	prefix := &Request{Model: "gemini-2.5-flash", Messages: []Message{UserMessage("a long stable prefix")}, Tools: []Tool{weatherTool("")}}
	ttl := 300
	wire, err := gemini.cacheCreateRequest(prefix, &ttl, "")
	if decl := onlyToolDeclaration(t, "gemini cache", wireBody(t, "gemini cache", wire, err)); decl.Has("description") {
		t.Fatalf("gemini cache: %s", mustJSON(decl))
	}
	nested := &Request{Model: "claude-haiku-4-5", Messages: []Message{UserMessage("hi")}, Tools: []Tool{weatherTool("")}}
	wire, err = anthropic.batchSubmitRequest(&BatchRequest{Requests: []*Request{nested}}, nil, newAdaptScope("", "anthropic", false))
	if decl := onlyToolDeclaration(t, "anthropic batch", wireBody(t, "anthropic batch", wire, err)); decl.Has("description") {
		t.Fatalf("anthropic batch: %s", mustJSON(decl))
	}
}
