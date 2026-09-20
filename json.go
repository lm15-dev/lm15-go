package lm15

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

// JSONObject is an opaque JSON object (tool input, extensions, provider_data,
// continuation data, ...). Contents are user or provider data and round-trip
// verbatim: lm15 validates them (INV-001) and never rewrites them (INV-002).
type JSONObject = map[string]any

// DecodeJSON parses JSON into Go values with numbers kept as json.Number, so
// an opaque payload re-encodes exactly as it arrived (1 stays 1, 1.0 stays
// 1.0). Every wire body and every canonical JSON document lm15 reads goes
// through here.
func DecodeJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	// Trailing garbage after the first value is not a JSON document.
	if dec.More() {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return out, nil
}

// DecodeJSONObject parses JSON and requires an object.
func DecodeJSONObject(data []byte) (JSONObject, error) {
	v, err := DecodeJSON(data)
	if err != nil {
		return nil, err
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("expected a JSON object, got %s", jsonTypeName(v))
	}
	return obj, nil
}

// EncodeJSON serializes a canonical dict compactly (no HTML escaping, as
// every other port does). Text that is not valid Unicode — a lone
// surrogate U+D800..U+DFFF, or any other invalid UTF-8 — has no UTF-8
// form and can reach no provider; it is refused here, before the wire, as
// the input error it is (INV-055), never silently replaced by U+FFFD.
func EncodeJSON(v any) ([]byte, error) {
	if err := checkUnicode(reflect.ValueOf(v), 0); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	out := buf.Bytes()
	if n := len(out); n > 0 && out[n-1] == '\n' {
		out = out[:n-1]
	}
	return out, nil
}

// checkUnicode walks strings inside a JSON-shaped value (INV-055).
func checkUnicode(rv reflect.Value, depth int) error {
	if !rv.IsValid() || depth > 512 {
		return nil
	}
	switch rv.Kind() {
	case reflect.String:
		return checkUnicodeString(rv.String())
	case reflect.Interface, reflect.Pointer:
		if rv.IsNil() {
			return nil
		}
		return checkUnicode(rv.Elem(), depth+1)
	case reflect.Slice, reflect.Array:
		if rv.Kind() == reflect.Slice && rv.Type().Elem().Kind() == reflect.Uint8 {
			return nil // raw bytes (json.RawMessage) are the caller's
		}
		for i := 0; i < rv.Len(); i++ {
			if err := checkUnicode(rv.Index(i), depth+1); err != nil {
				return err
			}
		}
	case reflect.Map:
		iter := rv.MapRange()
		for iter.Next() {
			if err := checkUnicode(iter.Key(), depth+1); err != nil {
				return err
			}
			if err := checkUnicode(iter.Value(), depth+1); err != nil {
				return err
			}
		}
	case reflect.Struct:
		// Typed values marshal through their own MarshalJSON (parts,
		// messages, ...), which route back through EncodeJSON.
	}
	return nil
}

func checkUnicodeString(s string) error {
	if utf8.ValidString(s) {
		return nil
	}
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			// A WTF-8 surrogate (ED A0..BF xx) names its code point.
			if i+2 < len(s) && s[i] == 0xED && s[i+1] >= 0xA0 && s[i+1] <= 0xBF {
				cp := 0xD000 | int(s[i+1]&0x3F)<<6 | int(s[i+2]&0x3F)
				return valueErrorf("request contains text that is not valid Unicode (lone surrogate U+%04X), which no provider can receive; repair the text first", cp)
			}
			return valueErrorf("request contains text that is not valid UTF-8 (byte 0x%02X at offset %d), which no provider can receive; repair the text first", s[i], i)
		}
		i += size
	}
	return nil
}

func mustJSON(v any) []byte {
	out, err := EncodeJSON(v)
	if err != nil {
		panic(err)
	}
	return out
}

// jsonFloat is a typed float field on the wire (Number rule): it always
// serializes with a fractional part or exponent, never as an integer literal.
type jsonFloat float64

func (f jsonFloat) MarshalJSON() ([]byte, error) {
	v := float64(f)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil, fmt.Errorf("non-finite float cannot be serialized")
	}
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return []byte(s), nil
}

// jsonTypeName names a decoded JSON value's type the way error messages do.
func jsonTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "bool"
	case string:
		return "str"
	case json.Number, int, int64, float64:
		return "number"
	case []any:
		return "list"
	case map[string]any:
		return "dict"
	}
	return reflect.TypeOf(v).String()
}

// ─── Strict JSON values (INV-001) ────────────────────────────────────

// ValidateJSONValue reports whether v is made only of JSON containers and
// scalars: nil, bool, string, json.Number, Go integers, finite floats, and
// slices / string-keyed maps of those (reflectively, so []string and
// map[string]int are accepted). Structs and other types are rejected.
func ValidateJSONValue(v any) error {
	return validateJSONReflect(reflect.ValueOf(v), 0)
}

func validateJSONReflect(rv reflect.Value, depth int) error {
	if depth > 512 {
		return fmt.Errorf("JSON value nests too deeply")
	}
	if !rv.IsValid() {
		return nil
	}
	switch rv.Kind() {
	case reflect.Interface, reflect.Pointer:
		if rv.IsNil() {
			return nil
		}
		return validateJSONReflect(rv.Elem(), depth)
	case reflect.Bool, reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if rv.Type() == reflect.TypeOf(json.Number("")) {
			if _, err := rv.Interface().(json.Number).Float64(); err != nil {
				return fmt.Errorf("invalid JSON number %q", rv.String())
			}
		}
		return nil
	case reflect.Float32, reflect.Float64:
		f := rv.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("non-finite float is not a JSON value")
		}
		return nil
	case reflect.Slice, reflect.Array:
		if rv.Kind() == reflect.Slice && rv.Type().Elem().Kind() == reflect.Uint8 {
			return fmt.Errorf("byte slices are not JSON values (encode them as base64 strings)")
		}
		for i := 0; i < rv.Len(); i++ {
			if err := validateJSONReflect(rv.Index(i), depth+1); err != nil {
				return err
			}
		}
		return nil
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return fmt.Errorf("JSON object keys must be strings")
		}
		iter := rv.MapRange()
		for iter.Next() {
			if err := validateJSONReflect(iter.Value(), depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("%s is not a JSON value", rv.Type())
}

func checkJSONObject(value JSONObject, field string, required bool) error {
	if value == nil {
		if required {
			return typeErrorf("%s must be a JSON object", field)
		}
		return nil
	}
	if err := ValidateJSONValue(value); err != nil {
		return typeErrorf("%s must contain only JSON-compatible values: %v", field, err)
	}
	return nil
}

// normalizeExtensions applies INV-004: an empty extensions map is absent.
func normalizeExtensions(ext JSONObject) (JSONObject, error) {
	if ext != nil && len(ext) == 0 {
		return nil, nil
	}
	return ext, checkJSONObject(ext, "extensions", false)
}

// ─── Omission rule (docs/serde-rules.md) ─────────────────────────────

// isEmptyJSON is the omission test for a typed object's own field: null,
// "", [], {} are omitted at that object's top level only.
func isEmptyJSON(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	case []string:
		return len(x) == 0
	case []JSONObject:
		return len(x) == 0
	case map[string]any:
		return len(x) == 0
	case *int, *float64, *bool, *string:
		return reflect.ValueOf(x).IsNil()
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Slice, reflect.Map:
		return rv.Len() == 0
	case reflect.Pointer, reflect.Interface:
		return rv.IsNil()
	}
	return false
}

// dict builds a canonical JSON object from key/value pairs, applying the
// omission rule to every pair (the keys marked always-emitted are added
// afterwards by the caller with put).
type dict map[string]any

func (d dict) put(key string, value any) dict {
	d[key] = value
	return d
}

func (d dict) omit(key string, value any) dict {
	if !isEmptyJSON(value) {
		d[key] = deref(value)
	}
	return d
}

// omitNull drops only nil (delta serializers: empty strings are emitted).
func (d dict) omitNull(key string, value any) dict {
	if value == nil {
		return d
	}
	if rv := reflect.ValueOf(value); rv.Kind() == reflect.Pointer && rv.IsNil() {
		return d
	}
	d[key] = deref(value)
	return d
}

func deref(v any) any {
	switch x := v.(type) {
	case *int:
		if x == nil {
			return nil
		}
		return *x
	case *float64:
		if x == nil {
			return nil
		}
		return jsonFloat(*x)
	case *bool:
		if x == nil {
			return nil
		}
		return *x
	case *string:
		if x == nil {
			return nil
		}
		return *x
	case float64:
		return jsonFloat(x)
	}
	return v
}

// ─── Reading canonical / wire JSON ───────────────────────────────────

// jsonInt reads a JSON number as an int under the Number rule (INV-007):
// same-valued floats coerce, non-integral floats and bools are rejected.
func jsonInt(v any, field string) (int, error) {
	switch x := v.(type) {
	case bool:
		return 0, typeErrorf("%s must be an int", field)
	case int:
		return x, nil
	case int64:
		return int(x), nil
	case int32:
		return int(x), nil
	case float64:
		if x != math.Trunc(x) || math.IsInf(x, 0) || math.IsNaN(x) {
			return 0, typeErrorf("%s must be an int", field)
		}
		return int(x), nil
	case float32:
		return jsonInt(float64(x), field)
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return int(i), nil
		}
		f, err := x.Float64()
		if err != nil {
			return 0, typeErrorf("%s must be an int", field)
		}
		return jsonInt(f, field)
	}
	return 0, typeErrorf("%s must be an int", field)
}

// jsonFloat64 reads a JSON number as a float under the Number rule (INV-008).
func jsonFloat64(v any, field string) (float64, error) {
	switch x := v.(type) {
	case bool:
		return 0, typeErrorf("%s must be numeric", field)
	case int:
		return float64(x), nil
	case int64:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case float64:
		if math.IsInf(x, 0) || math.IsNaN(x) {
			return 0, typeErrorf("%s must be finite", field)
		}
		return x, nil
	case float32:
		return float64(x), nil
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return 0, typeErrorf("%s must be numeric", field)
		}
		return f, nil
	}
	return 0, typeErrorf("%s must be numeric", field)
}

func optInt(d map[string]any, key string) (*int, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	i, err := jsonInt(v, key)
	if err != nil {
		return nil, err
	}
	return &i, nil
}

func optFloat(d map[string]any, key string) (*float64, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	f, err := jsonFloat64(v, key)
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func optBool(d map[string]any, key string) (*bool, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	b, ok := v.(bool)
	if !ok {
		return nil, typeErrorf("%s must be a bool", key)
	}
	return &b, nil
}

// optString reads an optional string; null or absent → "", a non-string → error.
func optString(d map[string]any, key string) (string, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", typeErrorf("%s must be a string", key)
	}
	return s, nil
}

// optStringPtr reads an optional string keeping "" distinct from absent.
func optStringPtr(d map[string]any, key string) (*string, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, typeErrorf("%s must be a string", key)
	}
	return &s, nil
}

// reqString reads a required string key.
func reqString(d map[string]any, key string) (string, error) {
	v, ok := d[key]
	if !ok {
		return "", keyError(key)
	}
	s, ok := v.(string)
	if !ok {
		return "", typeErrorf("%s must be a string", key)
	}
	return s, nil
}

func optObject(d map[string]any, key string) (JSONObject, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, typeErrorf("%s must be a JSON object", key)
	}
	return m, nil
}

func optList(d map[string]any, key string) ([]any, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	l, ok := v.([]any)
	if !ok {
		return nil, typeErrorf("%s must be a list", key)
	}
	return l, nil
}

func stringList(v any, field string) ([]string, error) {
	switch x := v.(type) {
	case nil:
		return nil, nil
	case string:
		return []string{x}, nil // INV-020: a bare string is one element
	case []string:
		return x, nil
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			s, ok := item.(string)
			if !ok {
				return nil, typeErrorf("%s must contain strings", field)
			}
			out = append(out, s)
		}
		return out, nil
	}
	return nil, typeErrorf("%s must be a list of strings", field)
}

// ─── Lenient wire accessors (provider payloads) ──────────────────────
//
// Provider bodies are read leniently: a missing or mistyped key is "absent",
// the way the reference's dict.get chains behave. These never error.

func wireObj(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func wireList(v any) []any {
	l, _ := v.([]any)
	return l
}

func wireStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return ""
	case json.Number:
		return x.String()
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case int:
		return strconv.Itoa(x)
	case int64:
		return strconv.FormatInt(x, 10)
	}
	return fmt.Sprint(v)
}

// wireIntPtr reads a provider counter: absent / null / non-numeric → nil.
func wireIntPtr(v any) *int {
	switch v.(type) {
	case bool, nil:
		return nil
	}
	i, err := jsonInt(v, "")
	if err != nil {
		// A provider may report a float count; truncate like int(value).
		f, ferr := jsonFloat64(v, "")
		if ferr != nil {
			return nil
		}
		i = int(f)
	}
	return &i
}

func wireInt(v any, fallback int) int {
	if p := wireIntPtr(v); p != nil {
		return *p
	}
	return fallback
}

func wireFloat(v any, fallback float64) float64 {
	f, err := jsonFloat64(v, "")
	if err != nil {
		return fallback
	}
	return f
}

// truthy mirrors Python truthiness for wire values.
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case json.Number:
		f, _ := x.Float64()
		return f != 0
	case float64:
		return x != 0
	case int:
		return x != 0
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

// firstStr returns the first non-empty string among the values.
func firstStr(values ...any) string {
	for _, v := range values {
		if s := wireStr(v); s != "" && v != nil {
			if _, isBool := v.(bool); isBool {
				continue
			}
			return s
		}
	}
	return ""
}

// copyObject shallow-copies a JSON object.
func copyObject(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func toAnyList[T any](items []T, f func(T) any) []any {
	out := make([]any, 0, len(items))
	for _, it := range items {
		out = append(out, f(it))
	}
	return out
}

// jsonEqual compares two JSON values structurally (numbers by value).
func jsonEqual(a, b any) bool {
	return bytes.Equal(canonicalBytes(a), canonicalBytes(b))
}

func canonicalBytes(v any) []byte {
	out, err := EncodeJSON(normalizeNumbers(v))
	if err != nil {
		return nil
	}
	return out
}

func normalizeNumbers(v any) any {
	switch x := v.(type) {
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return i
		}
		f, _ := x.Float64()
		return f
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = normalizeNumbers(item)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, item := range x {
			out[k] = normalizeNumbers(item)
		}
		return out
	}
	return v
}

// typeErrorf / valueErrorf / keyError are the native-error spellings the vet
// protocol reports (TypeError / ValueError / KeyError) for malformed input.
type nativeError struct {
	kind string
	msg  string
}

func (e *nativeError) Error() string { return e.msg }

// Kind is the native exception name (TypeError, ValueError, KeyError).
func (e *nativeError) Kind() string { return e.kind }

func typeErrorf(format string, args ...any) error {
	return &nativeError{kind: "TypeError", msg: fmt.Sprintf(format, args...)}
}

func valueErrorf(format string, args ...any) error {
	return &nativeError{kind: "ValueError", msg: fmt.Sprintf(format, args...)}
}

func keyError(key string) error {
	return &nativeError{kind: "KeyError", msg: fmt.Sprintf("'%s'", key)}
}

// NativeErrorKind returns the native exception name for malformed-input
// errors (TypeError, ValueError, KeyError), or "" for other errors.
func NativeErrorKind(err error) string {
	if ne, ok := err.(*nativeError); ok {
		return ne.kind
	}
	return ""
}
