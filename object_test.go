package lm15

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func mustEncode(t *testing.T, v any) string {
	t.Helper()
	b, err := EncodeJSON(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestDecodeEncodeKeepsKeyOrderAndNumberSpelling(t *testing.T) {
	in := `{"reasoning":{"type":"string"},"answer":{"type":"string","enum":[1,1.0,2e3]},"b":null,"a":[{"z":{},"y":[]}],"":""}`
	v, err := DecodeJSON([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if got := mustEncode(t, v); got != in {
		t.Fatalf("round trip changed the bytes:\n got %s\nwant %s", got, in)
	}
	obj := v.(JSONObject)
	if keys := strings.Join(obj.Keys(), ","); keys != "reasoning,answer,b,a," {
		t.Fatalf("keys %q", keys)
	}
}

func TestDecodeRepeatedKeyKeepsFirstPositionLastValue(t *testing.T) {
	// JavaScript's JSON.parse and Python's json.loads read a repeated key
	// this way; both code paths (short and indexed objects) must agree.
	short := `{"a":1,"b":2,"a":3}`
	long := `{"k0":0,"k1":1,"k2":2,"k3":3,"k4":4,"k5":5,"k6":6,"k7":7,"k8":8,"k1":"again","k9":9}`
	for in, want := range map[string]string{
		short: `{"a":3,"b":2}`,
		long:  `{"k0":0,"k1":"again","k2":2,"k3":3,"k4":4,"k5":5,"k6":6,"k7":7,"k8":8,"k9":9}`,
	} {
		v, err := DecodeJSON([]byte(in))
		if err != nil {
			t.Fatal(err)
		}
		if got := mustEncode(t, v); got != want {
			t.Fatalf("%s: got %s want %s", in, got, want)
		}
	}
}

func TestDecodeRejectsWhatIsNotOneJSONDocument(t *testing.T) {
	for _, in := range []string{``, `{`, `{"a":1}x`, `{"a":1}{}`, `[1,]`, `{"a" 1}`, `{1:2}`} {
		if _, err := DecodeJSON([]byte(in)); err == nil {
			t.Fatalf("accepted %q", in)
		}
	}
	for _, in := range []string{`{}`, ` {"a":1} `, "[]\n", `"s"`, `null`} {
		if _, err := DecodeJSON([]byte(in)); err != nil {
			t.Fatalf("refused %q: %v", in, err)
		}
	}
	v, _ := DecodeJSON([]byte(`{}`))
	if obj := v.(JSONObject); obj == nil {
		t.Fatal("{} decoded to a nil (absent) object")
	}
}

func TestRepeatedKeyIsNeitherValidNorSent(t *testing.T) {
	dup := JSONObject{KV("a", 1), KV("a", 2)}
	if err := ValidateJSONValue(dup); err == nil {
		t.Fatal("ValidateJSONValue accepted a repeated key")
	}
	if _, err := EncodeJSON(JSONObject{KV("x", dup)}); err == nil {
		t.Fatal("EncodeJSON wrote a repeated key")
	}
	if err := (FunctionTool{Name: "f", Parameters: JSONObject{KV("type", "object"), KV("type", "string")}}).Validate(); err == nil {
		t.Fatal("a tool's parameters accepted a repeated key")
	}
}

func TestSetAndDeleteKeepOrderAndNeverWriteIntoAnotherCopy(t *testing.T) {
	a := make(JSONObject, 0, 8) // spare capacity: the append-aliasing trap
	a.Set("one", 1)
	a.Set("two", 2)
	b := a
	b.Set("three", 3)
	a.Set("four", 4)
	if got := mustEncode(t, b); got != `{"one":1,"two":2,"three":3}` {
		t.Fatalf("b = %s", got)
	}
	if got := mustEncode(t, a); got != `{"one":1,"two":2,"four":4}` {
		t.Fatalf("a = %s", got)
	}
	c := a
	c.Set("one", "replaced") // replace keeps the position ...
	if got := mustEncode(t, c); got != `{"one":"replaced","two":2,"four":4}` {
		t.Fatalf("c = %s", got)
	}
	if a.Get("one") != 1 { // ... and the original is untouched
		t.Fatalf("a changed through a copy: %v", a)
	}
	d := a
	d.Delete("two")
	d.Delete("absent")
	if got := mustEncode(t, d); got != `{"one":1,"four":4}` {
		t.Fatalf("d = %s", got)
	}
	if got := mustEncode(t, a); got != `{"one":1,"two":2,"four":4}` {
		t.Fatalf("a changed through Delete on a copy: %s", got)
	}
}

func TestReadersAndIteration(t *testing.T) {
	o := JSONObject{KV("n", nil), KV("x", 1)}
	if v, ok := o.Lookup("n"); !ok || v != nil {
		t.Fatal("Lookup lost a present null")
	}
	if _, ok := o.Lookup("missing"); ok || o.Has("missing") || o.Get("missing") != nil {
		t.Fatal("an absent key read as present")
	}
	var keys []string
	for k := range o.All() {
		keys = append(keys, k)
		break // early exit must be honored
	}
	if len(keys) != 1 || keys[0] != "n" {
		t.Fatalf("All: %v", keys)
	}
	if c := JSONObject(nil).Clone(); c != nil {
		t.Fatal("Clone of nil must stay nil (absent)")
	}
	if got := mustEncode(t, JSONObject(nil)); got != "null" {
		t.Fatalf("nil object encodes as %s", got)
	}
	if got := mustEncode(t, JSONObject{}); got != "{}" {
		t.Fatalf("empty object encodes as %s", got)
	}
}

func TestGoMapsAreAcceptedAndReadSorted(t *testing.T) {
	m := map[string]any{"b": map[string]any{"y": 1, "x": 2}, "a": []any{map[string]any{"d": 1, "c": 2}}}
	if got := mustEncode(t, ObjectFromMap(m)); got != `{"a":[{"c":2,"d":1}],"b":{"x":2,"y":1}}` {
		t.Fatalf("ObjectFromMap: %s", got)
	}
	// A map nested inside an object: accepted, written sorted, read as an object.
	payload := JSONObject{KV("type", "object"), KV("properties", map[string]any{"z": 1, "a": 2})}
	if err := ValidateJSONValue(payload); err != nil {
		t.Fatal(err)
	}
	if got := mustEncode(t, payload); got != `{"type":"object","properties":{"a":2,"z":1}}` {
		t.Fatalf("nested map: %s", got)
	}
	props, ok := asObject(payload.Get("properties"))
	if !ok || strings.Join(props.Keys(), ",") != "a,z" {
		t.Fatalf("asObject: %v %v", props, ok)
	}
	if _, ok := jsonView(map[string]string{"k": "v"}).(JSONObject); !ok {
		t.Fatal("jsonView did not present a map[string]string as an object")
	}
	if ObjectFromMap(nil) != nil {
		t.Fatal("ObjectFromMap(nil) must be nil")
	}
}

func TestStringsEncodeExactlyAsEncodingJSON(t *testing.T) {
	for _, s := range []string{"", "plain", `q"uote`, `back\slash`, "\n\r\t\b\f\x00\x1f", "<a&b>", "é中🙂", "\u2028\u2029", "\x7f"} {
		var want bytes.Buffer
		enc := json.NewEncoder(&want)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(s)
		got := mustEncode(t, JSONObject{KV(s, s)})
		w := strings.TrimRight(want.String(), "\n")
		if got != "{"+w+":"+w+"}" {
			t.Fatalf("%q: got %s want {%s:%s}", s, got, w, w)
		}
	}
}

func TestEncodingJSONUsesTheOrderedForms(t *testing.T) {
	type holder struct {
		Schema JSONObject `json:"schema"`
	}
	h := holder{Schema: JSONObject{KV("reasoning", 1), KV("answer", 2)}}
	b, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"schema":{"reasoning":1,"answer":2}}` {
		t.Fatalf("json.Marshal: %s", b)
	}
	var back holder
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if strings.Join(back.Schema.Keys(), ",") != "reasoning,answer" {
		t.Fatalf("json.Unmarshal: %v", back.Schema)
	}
	var obj JSONObject
	if err := json.Unmarshal([]byte(`[1]`), &obj); err == nil {
		t.Fatal("a list unmarshalled into a JSONObject")
	}
}

func TestSetInCreatesAndWritesBackEveryLevel(t *testing.T) {
	o := JSONObject{KV("setup", JSONObject{KV("model", "m")})}
	setIn(&o, true, "setup", "realtimeInputConfig", "automaticActivityDetection", "disabled")
	setIn(&o, []any{"AUDIO"}, "setup", "generationConfig", "responseModalities")
	want := `{"setup":{"model":"m","realtimeInputConfig":{"automaticActivityDetection":{"disabled":true}},"generationConfig":{"responseModalities":["AUDIO"]}}}`
	if got := mustEncode(t, o); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}
