package lm15

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// jmap is decoded canonical JSON (json.Decoder with UseNumber, so opaque
// payload numbers round-trip verbatim as json.Number).
type jmap = map[string]any

// serdeError carries a canonical-ish error type name for the vet protocol
// ("ValueError", "TypeError", "KeyError"), mirroring the reference shim's
// native exception names.
type serdeError struct {
	Type string
	Msg  string
}

func (e *serdeError) Error() string { return e.Msg }

// ErrTypeName reports the vet-protocol error type name for an error: the
// recorded native-exception-style name for serde errors, the canonical class
// name for lm15-typed errors, else "ValueError".
func ErrTypeName(err error) string {
	if se, ok := err.(*serdeError); ok {
		return se.Type
	}
	if le, ok := err.(LM15Error); ok {
		return le.ClassName()
	}
	return "ValueError"
}

func valueErr(format string, a ...any) error {
	return &serdeError{Type: "ValueError", Msg: fmt.Sprintf(format, a...)}
}

func typeErr(format string, a ...any) error {
	return &serdeError{Type: "TypeError", Msg: fmt.Sprintf(format, a...)}
}

func keyErr(key string) error {
	return &serdeError{Type: "KeyError", Msg: fmt.Sprintf("'%s'", key)}
}

// ─── Emission helpers ────────────────────────────────────────────────

// floatText renders a declared-float field with at least one decimal so the
// canonical wire form is a JSON float (1.0, never 1) — the Number rule.
func floatText(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func jsonFloat(f float64) json.Number { return json.Number(floatText(f)) }

func jsonInt(i int64) json.Number { return json.Number(strconv.FormatInt(i, 10)) }

// isEmptyJSON implements the one omission rule's notion of empty:
// null, "", [], {} — at the top level of one typed object only.
func isEmptyJSON(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	case jmap:
		return len(x) == 0
	}
	return false
}

// putOpt adds k:v unless v is empty (omission rule for optional fields).
func putOpt(m jmap, k string, v any) {
	if !isEmptyJSON(v) {
		m[k] = v
	}
}

// ─── Parsing helpers ─────────────────────────────────────────────────

func reqString(m jmap, k string) (string, error) {
	v, ok := m[k]
	if !ok {
		return "", keyErr(k)
	}
	s, ok := v.(string)
	if !ok {
		return "", typeErr("%s must be a string", k)
	}
	return s, nil
}

func optString(m jmap, k, def string) (string, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return def, nil
	}
	s, ok := v.(string)
	if !ok {
		return "", typeErr("%s must be a string", k)
	}
	return s, nil
}

func optStringPtr(m jmap, k string) (*string, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return nil, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, typeErr("%s must be a string", k)
	}
	return &s, nil
}

func optBool(m jmap, k string, def bool) (bool, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return def, nil
	}
	b, ok := v.(bool)
	if !ok {
		return false, typeErr("%s must be a boolean", k)
	}
	return b, nil
}

func optBoolPtr(m jmap, k string) (*bool, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return nil, nil
	}
	b, ok := v.(bool)
	if !ok {
		return nil, typeErr("%s must be a boolean", k)
	}
	return &b, nil
}

// optInt applies the Number rule for int-declared fields: same-valued floats
// coerce; non-integral floats are rejected; bool never coerces.
func optInt(m jmap, k string) (*int64, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return nil, nil
	}
	n, ok := v.(json.Number)
	if !ok {
		return nil, typeErr("%s must be an integer", k)
	}
	if i, err := n.Int64(); err == nil {
		return &i, nil
	}
	f, err := n.Float64()
	if err != nil {
		return nil, valueErr("%s must be an integer", k)
	}
	if f != float64(int64(f)) {
		return nil, valueErr("%s must be an integer", k)
	}
	i := int64(f)
	return &i, nil
}

// optFloat applies the Number rule for float-declared fields: int coerces.
func optFloat(m jmap, k string) (*float64, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return nil, nil
	}
	n, ok := v.(json.Number)
	if !ok {
		return nil, typeErr("%s must be a number", k)
	}
	f, err := n.Float64()
	if err != nil {
		return nil, valueErr("%s must be a number", k)
	}
	return &f, nil
}

func optObj(m jmap, k string) (jmap, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return nil, nil
	}
	o, ok := v.(jmap)
	if !ok {
		return nil, typeErr("%s must be a JSON object", k)
	}
	return o, nil
}

func optList(m jmap, k string) ([]any, error) {
	v, ok := m[k]
	if !ok || v == nil {
		return nil, nil
	}
	l, ok := v.([]any)
	if !ok {
		return nil, typeErr("%s must be an array", k)
	}
	return l, nil
}

func stringList(m jmap, k string) ([]string, error) {
	raw, err := optList(m, k)
	if err != nil || raw == nil {
		return nil, err
	}
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		s, ok := v.(string)
		if !ok {
			return nil, typeErr("%s entries must be strings", k)
		}
		out = append(out, s)
	}
	return out, nil
}

// decodeMap decodes JSON bytes into a jmap with UseNumber.
func decodeMap(data []byte) (jmap, error) {
	var v any
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	m, ok := v.(jmap)
	if !ok {
		return nil, typeErr("expected a JSON object")
	}
	return m, nil
}
