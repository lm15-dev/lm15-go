package lm15_test

import (
	"fmt"

	lm15 "github.com/lm15-dev/lm15-go"
)

// A schema lists its properties in the order the model should fill them:
// here the reasoning, then the answer it leads to.
func ExampleJSONObject() {
	schema := lm15.JSONObject{
		lm15.KV("type", "object"),
		lm15.KV("properties", lm15.JSONObject{
			lm15.KV("reasoning", lm15.JSONObject{lm15.KV("type", "string")}),
			lm15.KV("answer", lm15.JSONObject{lm15.KV("type", "string")}),
		}),
		lm15.KV("required", []any{"reasoning", "answer"}),
	}
	schema.Set("additionalProperties", false)
	out, _ := lm15.EncodeJSON(schema)
	fmt.Println(string(out))
	// Output: {"type":"object","properties":{"reasoning":{"type":"string"},"answer":{"type":"string"}},"required":["reasoning","answer"],"additionalProperties":false}
}

// JSON text keeps its order, and so does everything lm15 decodes.
func ExampleDecodeJSONObject() {
	obj, _ := lm15.DecodeJSONObject([]byte(`{"zeta": 1, "alpha": 2.0}`))
	for key, value := range obj.All() {
		fmt.Println(key, value)
	}
	// Output:
	// zeta 1
	// alpha 2.0
}

// A Go map has no order: converted, its keys come out sorted.
func ExampleObjectFromMap() {
	obj := lm15.ObjectFromMap(map[string]any{"zeta": 1, "alpha": 2})
	fmt.Println(obj.Keys())
	// Output: [alpha zeta]
}
