package lm15

import (
	"bytes"
	"encoding/json"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"unicode/utf8"
)

// Member is one key/value pair of a JSONObject.
type Member struct {
	Key   string
	Value any
}

// KV builds a Member. It is the short, vet-clean way to write an object
// literal outside this package:
//
//	schema := lm15.JSONObject{
//		lm15.KV("type", "object"),
//		lm15.KV("properties", lm15.JSONObject{
//			lm15.KV("reasoning", lm15.JSONObject{lm15.KV("type", "string")}),
//			lm15.KV("answer", lm15.JSONObject{lm15.KV("type", "string")}),
//		}),
//	}
func KV(key string, value any) Member { return Member{Key: key, Value: value} }

// JSONObject is a JSON object that keeps its keys in the order they were
// written or received. Order is data: a model fills structured output in
// the order the schema lists its properties, and a signature or an upload
// covers the exact bytes. Every object lm15 decodes is a JSONObject in the
// order it arrived, and every object it encodes is written in member order.
//
// A JSONObject is a slice, so a literal lists its members in order
// (lm15.KV keeps go vet quiet about unkeyed fields). Read with Get, Lookup
// and All; write with Set and Delete. Set and Delete never write into
// memory another copy of the object can see: they copy first, so an
// object you handed to lm15, or received from it, never changes behind
// your back. That costs one copy per call; objects here are small.
//
// A nil JSONObject is "absent"; JSONObject{} is the empty object. Keys are
// unique: an object with a repeated key is not a JSON value lm15 accepts
// or sends.
type JSONObject []Member

func (o JSONObject) index(key string) int {
	for i := range o {
		if o[i].Key == key {
			return i
		}
	}
	return -1
}

// Get returns the value stored under key, or nil when the key is absent.
// Use Lookup to tell an absent key from a JSON null.
func (o JSONObject) Get(key string) any {
	if i := o.index(key); i >= 0 {
		return o[i].Value
	}
	return nil
}

// Lookup returns the value stored under key and whether the key is present.
func (o JSONObject) Lookup(key string) (any, bool) {
	if i := o.index(key); i >= 0 {
		return o[i].Value, true
	}
	return nil, false
}

// Has reports whether key is present.
func (o JSONObject) Has(key string) bool { return o.index(key) >= 0 }

// Set stores value under key: in place of the existing member, keeping its
// position, or as a new last member. It copies the object first.
func (o *JSONObject) Set(key string, value any) {
	i := o.index(key)
	n := len(*o)
	if i < 0 {
		n++
	}
	out := make(JSONObject, len(*o), n)
	copy(out, *o)
	if i >= 0 {
		out[i].Value = value
	} else {
		out = append(out, Member{Key: key, Value: value})
	}
	*o = out
}

// Delete removes key, keeping the order of the other members. It copies
// the object first; deleting an absent key does nothing.
func (o *JSONObject) Delete(key string) {
	i := o.index(key)
	if i < 0 {
		return
	}
	out := make(JSONObject, 0, len(*o)-1)
	out = append(out, (*o)[:i]...)
	*o = append(out, (*o)[i+1:]...)
}

// Keys returns the keys in order.
func (o JSONObject) Keys() []string {
	keys := make([]string, len(o))
	for i := range o {
		keys[i] = o[i].Key
	}
	return keys
}

// All iterates the members in order: for key, value := range obj.All().
func (o JSONObject) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for _, m := range o {
			if !yield(m.Key, m.Value) {
				return
			}
		}
	}
}

// Clone returns a shallow copy (nested objects and lists are shared).
// It keeps nil distinct from empty.
func (o JSONObject) Clone() JSONObject { return slices.Clone(o) }

// ObjectFromMap converts a Go map into a JSONObject. A Go map has no key
// order, so the keys come out sorted, the order encoding/json writes a map
// in; nested maps are converted the same way. Build a JSONObject directly
// (or parse one with DecodeJSONObject) when the order matters, as it does
// for a JSON Schema's properties.
func ObjectFromMap(m map[string]any) JSONObject {
	if m == nil {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make(JSONObject, 0, len(keys))
	for _, k := range keys {
		out = append(out, Member{Key: k, Value: fromGoMaps(m[k])})
	}
	return out
}

// fromGoMaps converts nested map[string]any values (and []any lists of
// them) to JSONObjects, sorted; other values are returned unchanged.
func fromGoMaps(v any) any {
	switch x := v.(type) {
	case map[string]any:
		return ObjectFromMap(x)
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = fromGoMaps(item)
		}
		return out
	}
	return v
}

// asObject reads a JSON object from a value that may be a JSONObject (what
// lm15 decodes and builds) or a Go map a caller nested inside an opaque
// payload (read in sorted key order, the only order a map has).
func asObject(v any) (JSONObject, bool) {
	switch x := v.(type) {
	case JSONObject:
		return x, true
	case map[string]any:
		return ObjectFromMap(x), true
	case nil:
		return nil, false
	}
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Map || rv.Type().Key().Kind() != reflect.String {
		return nil, false
	}
	if rv.IsNil() {
		return nil, true
	}
	keys := make([]string, 0, rv.Len())
	for _, k := range rv.MapKeys() {
		keys = append(keys, k.String())
	}
	sort.Strings(keys)
	out := make(JSONObject, 0, len(keys))
	for _, k := range keys {
		out = append(out, Member{Key: k, Value: rv.MapIndex(reflect.ValueOf(k).Convert(rv.Type().Key())).Interface()})
	}
	return out, true
}

// String is the object as compact JSON, in member order, so fmt.Println
// and %v show what would be sent. An object that cannot be written (a
// repeated key, a value with no JSON form) shows its members instead.
func (o JSONObject) String() string {
	b, err := EncodeJSON(o)
	if err != nil {
		return fmt.Sprintf("%v", []Member(o))
	}
	return string(b)
}

// MarshalJSON writes the members in order, compactly, without HTML
// escaping. A repeated key is an error.
func (o JSONObject) MarshalJSON() ([]byte, error) {
	if o == nil {
		return []byte("null"), nil
	}
	return appendJSON(nil, o, 0)
}

// UnmarshalJSON reads a JSON object keeping its key order; numbers stay
// json.Number, nested objects become JSONObjects.
func (o *JSONObject) UnmarshalJSON(data []byte) error {
	v, err := DecodeJSON(data)
	if err != nil {
		return err
	}
	switch x := v.(type) {
	case JSONObject:
		*o = x
		return nil
	case nil:
		*o = nil
		return nil
	}
	return fmt.Errorf("expected a JSON object, got %s", jsonTypeName(v))
}

// ─── Order-preserving decoder ────────────────────────────────────────

// decodeValue reads one JSON value from dec: objects as JSONObject in wire
// order (a repeated key keeps its first position and its last value, as
// JavaScript and Python read it), lists as []any, numbers as json.Number.
//
// It walks encoding/json's token stream, the standard library's validated
// JSON reader, rather than a hand-written parser: provider replies are
// untrusted input. The cost is speed on object-heavy bodies (measured
// 2026-09-25: about 3x slower than decoding into maps for a 2 KB reply or
// a 200 KB model list; unchanged for a 5 MB base64 image), microseconds to
// milliseconds against network time. encoding/json/jsontext (experimental
// in Go 1.26) is the faster reader to move to once it is stable.
func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	switch delim {
	case '{':
		obj := JSONObject{}
		var seen map[string]int
		for dec.More() {
			kt, err := dec.Token()
			if err != nil {
				return nil, err
			}
			key, ok := kt.(string)
			if !ok {
				return nil, fmt.Errorf("invalid object key %v", kt)
			}
			val, err := decodeValue(dec)
			if err != nil {
				return nil, err
			}
			if seen == nil && len(obj) >= 8 {
				seen = make(map[string]int, 2*len(obj))
				for i, m := range obj {
					seen[m.Key] = i
				}
			}
			i := -1
			if seen != nil {
				if j, ok := seen[key]; ok {
					i = j
				}
			} else {
				i = obj.index(key)
			}
			if i >= 0 {
				obj[i].Value = val
				continue
			}
			if seen != nil {
				seen[key] = len(obj)
			}
			obj = append(obj, Member{Key: key, Value: val})
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return obj, nil
	case '[':
		list := []any{}
		for dec.More() {
			val, err := decodeValue(dec)
			if err != nil {
				return nil, err
			}
			list = append(list, val)
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return list, nil
	}
	return nil, fmt.Errorf("unexpected delimiter %v", delim)
}

// ─── Order-preserving encoder ────────────────────────────────────────

const maxEncodeDepth = 10000

// appendJSON writes v compactly. JSONObjects, lists, strings, booleans,
// integers and json.Numbers are written here; every other value (floats,
// typed values with their own MarshalJSON, Go maps and slices) goes
// through encoding/json with HTML escaping off, so a Go map inside a
// payload is written with sorted keys, as encoding/json writes it.
func appendJSON(buf []byte, v any, depth int) ([]byte, error) {
	if depth > maxEncodeDepth {
		return nil, fmt.Errorf("JSON value nests too deeply")
	}
	switch x := v.(type) {
	case nil:
		return append(buf, "null"...), nil
	case JSONObject:
		if x == nil {
			return append(buf, "null"...), nil
		}
		if k, dup := duplicateKey(x); dup {
			return nil, fmt.Errorf("JSON object has the key %q twice", k)
		}
		buf = append(buf, '{')
		for i, m := range x {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = appendString(buf, m.Key)
			buf = append(buf, ':')
			var err error
			if buf, err = appendJSON(buf, m.Value, depth+1); err != nil {
				return nil, err
			}
		}
		return append(buf, '}'), nil
	case []any:
		if x == nil {
			return append(buf, "null"...), nil
		}
		buf = append(buf, '[')
		for i, item := range x {
			if i > 0 {
				buf = append(buf, ',')
			}
			var err error
			if buf, err = appendJSON(buf, item, depth+1); err != nil {
				return nil, err
			}
		}
		return append(buf, ']'), nil
	case []JSONObject:
		if x == nil {
			return append(buf, "null"...), nil
		}
		buf = append(buf, '[')
		for i, item := range x {
			if i > 0 {
				buf = append(buf, ',')
			}
			var err error
			if buf, err = appendJSON(buf, item, depth+1); err != nil {
				return nil, err
			}
		}
		return append(buf, ']'), nil
	case string:
		return appendString(buf, x), nil
	case bool:
		return strconv.AppendBool(buf, x), nil
	case int:
		return strconv.AppendInt(buf, int64(x), 10), nil
	case int64:
		return strconv.AppendInt(buf, x, 10), nil
	case int32:
		return strconv.AppendInt(buf, int64(x), 10), nil
	}
	var enc bytes.Buffer
	e := json.NewEncoder(&enc)
	e.SetEscapeHTML(false)
	if err := e.Encode(v); err != nil {
		return nil, err
	}
	return append(buf, bytes.TrimRight(enc.Bytes(), "\n")...), nil
}

func duplicateKey(o JSONObject) (string, bool) {
	if len(o) <= 8 {
		for i := 1; i < len(o); i++ {
			for j := 0; j < i; j++ {
				if o[i].Key == o[j].Key {
					return o[i].Key, true
				}
			}
		}
		return "", false
	}
	seen := make(map[string]struct{}, len(o))
	for _, m := range o {
		if _, ok := seen[m.Key]; ok {
			return m.Key, true
		}
		seen[m.Key] = struct{}{}
	}
	return "", false
}

// appendString quotes s exactly as encoding/json does with HTML escaping
// off: plain printable ASCII is copied, anything else goes through
// encoding/json so the escapes (\n, \u001f, \u2028, ...) stay identical.
func appendString(buf []byte, s string) []byte {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < 0x20 || c == '"' || c == '\\' || c >= utf8.RuneSelf {
			var enc bytes.Buffer
			e := json.NewEncoder(&enc)
			e.SetEscapeHTML(false)
			_ = e.Encode(s)
			return append(buf, bytes.TrimRight(enc.Bytes(), "\n")...)
		}
	}
	buf = append(buf, '"')
	buf = append(buf, s...)
	return append(buf, '"')
}

// setIn stores value at a nested path, creating the objects on the way.
// JSONObjects are values, not references, so each level is written back to
// its parent; a level that holds something other than an object is replaced.
func setIn(o *JSONObject, value any, path ...string) {
	if len(path) == 1 {
		o.Set(path[0], value)
		return
	}
	child, _ := asObject(o.Get(path[0]))
	if child == nil {
		child = JSONObject{}
	}
	setIn(&child, value, path[1:]...)
	o.Set(path[0], child)
}

// sortedMapKeys lists a Go map's keys sorted: a map has no order of its own,
// and sorted is the order encoding/json writes one in.
func sortedMapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// jsonView is the value a type switch over JSON reads: a Go map a caller
// nested inside a payload is seen as the JSONObject it encodes as (sorted
// keys); every other value is returned unchanged.
func jsonView(v any) any {
	switch v.(type) {
	case nil, JSONObject, string, bool, []any:
		return v
	}
	if o, ok := asObject(v); ok {
		return o
	}
	return v
}
