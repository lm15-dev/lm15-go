package lm15

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// numIsInt mirrors the harness's typed comparison: a JSON number is an
// integer iff its text has no fraction/exponent.
func numIsInt(n json.Number) bool {
	return !strings.ContainsAny(n.String(), ".eE")
}

func jsonEqual(t *testing.T, path string, expected, actual any) bool {
	switch e := expected.(type) {
	case jmap:
		a, ok := actual.(jmap)
		if !ok || len(a) != len(e) {
			t.Errorf("%s: object mismatch: expected %v, got %v", path, expected, actual)
			return false
		}
		for k, ev := range e {
			av, ok := a[k]
			if !ok {
				t.Errorf("%s.%s: missing key", path, k)
				return false
			}
			if !jsonEqual(t, path+"."+k, ev, av) {
				return false
			}
		}
		return true
	case []any:
		a, ok := actual.([]any)
		if !ok || len(a) != len(e) {
			t.Errorf("%s: array mismatch: expected %v, got %v", path, expected, actual)
			return false
		}
		for i := range e {
			if !jsonEqual(t, path, e[i], a[i]) {
				return false
			}
		}
		return true
	case json.Number:
		a, ok := actual.(json.Number)
		if !ok {
			t.Errorf("%s: expected number %v, got %T %v", path, e, actual, actual)
			return false
		}
		if numIsInt(e) != numIsInt(a) {
			t.Errorf("%s: number type mismatch: expected %v, got %v", path, e, a)
			return false
		}
		if numIsInt(e) {
			ei, _ := e.Int64()
			ai, _ := a.Int64()
			if ei != ai {
				t.Errorf("%s: expected %v, got %v", path, e, a)
				return false
			}
			return true
		}
		ef, _ := e.Float64()
		af, _ := a.Float64()
		if ef != af {
			t.Errorf("%s: expected %v, got %v", path, e, a)
			return false
		}
		return true
	default:
		if expected != actual {
			t.Errorf("%s: expected %#v, got %#v", path, expected, actual)
			return false
		}
		return true
	}
}

// reparse round-trips a jmap through encoding/json with UseNumber so the
// emitted textual number forms are what gets compared.
func reparse(t *testing.T, m jmap) jmap {
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	out, err := decodeMap(data)
	if err != nil {
		t.Fatalf("reparse: %v", err)
	}
	return out
}

func TestCanonicalSerdeVectors(t *testing.T) {
	data, err := os.ReadFile("../lm15-contract/serde/canonical.json")
	if err != nil {
		t.Skipf("contract repo unavailable: %v", err)
	}
	doc, err := decodeMap(data)
	if err != nil {
		t.Fatal(err)
	}
	cases := doc["cases"].([]any)
	if len(cases) == 0 {
		t.Fatal("no serde cases")
	}
	for _, c := range cases {
		cm := c.(jmap)
		id := cm["id"].(string)
		t.Run(id, func(t *testing.T) {
			out, err := SerdeRoundtrip(cm["kind"].(string), cm["value"].(jmap))
			if err != nil {
				t.Fatalf("roundtrip failed: %v", err)
			}
			jsonEqual(t, "$", cm["value"], reparse(t, out))
		})
	}
}

func TestFloatTextEmitsDecimal(t *testing.T) {
	for in, want := range map[float64]string{1: "1.0", 0.2: "0.2", 15: "15.0", 0.125: "0.125"} {
		if got := floatText(in); got != want {
			t.Errorf("floatText(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestSurfaceDump(t *testing.T) {
	dump := SurfaceDump()
	types := dump["types"].(jmap)
	if _, ok := types["ToolCallPart"]; !ok {
		t.Error("ToolCallPart missing from surface dump")
	}
	enums := dump["enums"].(jmap)
	if _, ok := enums["FinishReason"]; !ok {
		t.Error("FinishReason missing from surface dump")
	}
}
