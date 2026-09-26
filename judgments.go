package lm15

// Judgments: declared keys in, a distribution out (MAP-14).
//
// changes/2026-09-17-judgments.md. A judgment is a top-level property of a
// json_schema response_format that declares its answer set: a boolean, a
// string enum / anyOf-of-const, or an ordered integer enum /
// anyOf-of-const 0..n-1. This file reads that convention off a schema
// (§1), rewrites judgment properties for the two wires that need it (§2),
// folds a model's JSON text into a DataPart (§3), and offers the sugar that
// EMITS the convention (Choice, YesNo, Score, Judgments) the way tool
// helpers emit a tool schema. Nothing here touches the network.

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// JudgmentKind is one of "boolean", "choice", "ordered".
type JudgmentKind string

const (
	JudgmentBoolean JudgmentKind = "boolean"
	JudgmentChoice  JudgmentKind = "choice"
	JudgmentOrdered JudgmentKind = "ordered"
)

const (
	// MaxOrderedLevels is Jev's Score ceiling (docs.typesafe.ai/primitives/score).
	MaxOrderedLevels = 10
	// MaxChoiceKeys is Jev's Choice ceiling (docs.typesafe.ai/primitives/choice).
	MaxChoiceKeys = 255
)

// Judgment is one declared judgment read off a schema property.
type Judgment struct {
	Name         string
	Kind         JudgmentKind
	Keys         []string // declared answer keys, in order
	Instruction  string   // the property's description (the question); "" = none
	Descriptions map[string]string
	Titles       map[string]string
}

// Ordered reports whether the judgment is a level scale.
func (j Judgment) Ordered() bool { return j.Kind == JudgmentOrdered }

// ─── §1 reading the convention ────────────────────────────────────────

func constBranches(prop JSONObject) []JSONObject {
	raw, ok := prop.Get("anyOf").([]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	out := make([]JSONObject, 0, len(raw))
	for _, b := range raw {
		obj, ok := asObject(b)
		if !ok {
			return nil
		}
		if _, has := obj.Lookup("const"); !has {
			return nil
		}
		out = append(out, obj)
	}
	return out
}

func optDescription(prop JSONObject, key string) string {
	if s, ok := prop.Get(key).(string); ok {
		return s
	}
	return ""
}

// jsonInteger reads an integer literal (a json.Number without a fraction,
// or a Go int); bools and floats are not integers.
func jsonInteger(v any) (int, bool) {
	switch x := v.(type) {
	case json.Number:
		if strings.ContainsAny(x.String(), ".eE") {
			return 0, false
		}
		n, err := strconv.Atoi(x.String())
		return n, err == nil
	case int:
		return x, true
	case int64:
		return int(x), true
	case float64:
		if x == math.Trunc(x) && !math.IsInf(x, 0) {
			return int(x), true
		}
	}
	return 0, false
}

func judgmentOf(name string, raw any) (Judgment, bool) {
	prop, ok := asObject(raw)
	if !ok {
		return Judgment{}, false
	}
	instruction := optDescription(prop, "description")
	typ, _ := prop.Get("type").(string)
	if typ == "boolean" {
		return Judgment{Name: name, Kind: JudgmentBoolean, Keys: []string{"true", "false"}, Instruction: instruction,
			Descriptions: map[string]string{}, Titles: map[string]string{}}, true
	}
	enum, hasEnum := prop.Get("enum").([]any)
	branches := constBranches(prop)
	var values []any
	descs := map[string]string{}
	titles := map[string]string{}
	switch {
	case hasEnum && len(enum) > 0 && branches == nil:
		values = enum
	case branches != nil && prop.Get("enum") == nil:
		for _, b := range branches {
			values = append(values, b.Get("const"))
			key := wireStr(b.Get("const"))
			if d := optDescription(b, "description"); d != "" {
				descs[key] = d
			}
			if t := optDescription(b, "title"); t != "" {
				titles[key] = t
			}
		}
	default:
		return Judgment{}, false
	}
	allStrings := true
	for _, v := range values {
		if s, ok := v.(string); !ok || s == "" {
			allStrings = false
			break
		}
	}
	if allStrings {
		if typ != "" && typ != "string" {
			return Judgment{}, false
		}
		keys := make([]string, 0, len(values))
		seen := map[string]bool{}
		for _, v := range values {
			k := v.(string)
			if seen[k] {
				return Judgment{}, false
			}
			seen[k] = true
			keys = append(keys, k)
		}
		return Judgment{Name: name, Kind: JudgmentChoice, Keys: keys, Instruction: instruction, Descriptions: descs, Titles: titles}, true
	}
	ints := make([]int, 0, len(values))
	for _, v := range values {
		n, ok := jsonInteger(v)
		if !ok {
			return Judgment{}, false
		}
		ints = append(ints, n)
	}
	if typ != "" && typ != "integer" {
		return Judgment{}, false
	}
	if len(ints) < 2 {
		return Judgment{}, false
	}
	for i, n := range ints {
		if n != i {
			return Judgment{}, false
		}
	}
	keys := make([]string, 0, len(ints))
	for _, n := range ints {
		keys = append(keys, strconv.Itoa(n))
	}
	return Judgment{Name: name, Kind: JudgmentOrdered, Keys: keys, Instruction: instruction, Descriptions: descs, Titles: titles}, true
}

// schemaPropertyOrder is the order a schema lists its properties in: the
// order a model fills structured output in, and the order judgments are
// reported in (MAP-14 §1). A properties object given as a Go map has no
// order of its own and reads sorted.
func schemaPropertyOrder(schema JSONObject) []string {
	props, _ := asObject(schema.Get("properties"))
	return props.Keys()
}

// JudgmentsInSchema lists the judgments a json_schema declares, in
// property order (MAP-14 §1). Any property that is not one of the three
// shapes is ordinary structured output and is absent from the result.
func JudgmentsInSchema(schema any) []Judgment {
	obj, ok := asObject(schema)
	if !ok {
		return nil
	}
	if typ, has := obj.Lookup("type"); has && typ != "object" {
		return nil
	}
	props, ok := asObject(obj.Get("properties"))
	if !ok {
		return nil
	}
	var out []Judgment
	for _, name := range schemaPropertyOrder(obj) {
		if j, ok := judgmentOf(name, props.Get(name)); ok {
			out = append(out, j)
		}
	}
	return out
}

// RequestJudgments lists the judgments a request's response_format declares.
func RequestJudgments(req *Request) []Judgment {
	if req == nil {
		return nil
	}
	f := req.Config.ResponseFormat
	if f == nil || f.Get("type") != "json_schema" {
		return nil
	}
	return JudgmentsInSchema(f.Get("schema"))
}

// nonJudgmentProperties lists the schema's properties that are not judgments.
func nonJudgmentProperties(schema any, found []Judgment) []string {
	obj, ok := asObject(schema)
	if !ok {
		return nil
	}
	isJudgment := map[string]bool{}
	for _, j := range found {
		isJudgment[j.Name] = true
	}
	var out []string
	for _, name := range schemaPropertyOrder(obj) {
		if !isJudgment[name] {
			out = append(out, name)
		}
	}
	return out
}

// ─── §2 what a wire that measures nothing does with probabilities ─────

// noteUnmeasurableProbabilities is MAP-14 §3 on a wire with no
// distribution: if_available records dropped; required refuses before the
// wire (MAP-13 b).
func noteUnmeasurableProbabilities(scope *adaptScope, req *Request, provider string) error {
	policy := req.Config.Probabilities
	if policy == "" || policy == ProbabilitiesOff || len(RequestJudgments(req)) == 0 {
		return nil
	}
	if policy == ProbabilitiesRequired {
		return UnsupportedFeature(provider, "config.probabilities",
			"%s: config.probabilities='required' but this wire cannot measure a distribution over the declared keys (it returns a pick only); use 'if_available' or a provider that can (typesafe, or a vLLM/SGLang server that honours logprob_token_ids)", provider)
	}
	return scope.dropped("config.probabilities",
		"this wire cannot measure a distribution over the declared keys; the answer carries the pick only", policy)
}

func deepCopyJSON(v any) any {
	switch x := jsonView(v).(type) {
	case JSONObject:
		out := make(JSONObject, len(x))
		for i, m := range x {
			out[i] = Member{m.Key, deepCopyJSON(m.Value)}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = deepCopyJSON(val)
		}
		return out
	}
	return v
}

// anthropicSchema: a judgment property carrying both type and anyOf has
// its type moved into every branch: the Messages wire answers 400 "For
// 'anyOf', 'type' is not supported" otherwise (receipted 2026-09-17).
// Every other keyword stays verbatim (INV-050 exception).
func anthropicSchema(schema JSONObject, found []Judgment) JSONObject {
	if len(found) == 0 {
		return schema
	}
	out := deepCopyJSON(schema).(JSONObject)
	props, _ := asObject(out.Get("properties"))
	for _, j := range found {
		prop, ok := asObject(props.Get(j.Name))
		if !ok {
			continue
		}
		branches, isList := prop.Get("anyOf").([]any)
		kind, hasType := prop.Lookup("type")
		if hasType && isList {
			prop.Delete("type")
			for i, b := range branches {
				if obj, ok := asObject(b); ok && !obj.Has("type") {
					obj.Set("type", kind)
					branches[i] = obj // branches is the deep copy's own list
				}
			}
			props.Set(j.Name, prop)
		}
	}
	if props != nil {
		out.Set("properties", props)
	}
	return out
}

// geminiSchema: judgment properties go as enum with the per-key
// descriptions folded into the property description: responseJsonSchema
// ignores anyOf/const (it answered "Bordeaux-blend" for a declared set;
// receipted 2026-09-17) and honours enum.
func geminiSchema(schema JSONObject, found []Judgment) JSONObject {
	if len(found) == 0 {
		return schema
	}
	out := deepCopyJSON(schema).(JSONObject)
	props, _ := asObject(out.Get("properties"))
	for _, j := range found {
		prop, ok := asObject(props.Get(j.Name))
		if !ok {
			continue
		}
		if j.Kind == JudgmentBoolean {
			continue
		}
		if _, has := prop.Lookup("anyOf"); !has {
			continue
		}
		prop.Delete("anyOf")
		if j.Ordered() {
			prop.Set("type", "integer")
			enum := make([]any, 0, len(j.Keys))
			for _, k := range j.Keys {
				n, _ := strconv.Atoi(k)
				enum = append(enum, n)
			}
			prop.Set("enum", enum)
		} else {
			prop.Set("type", "string")
			prop.Set("enum", toAnyList(j.Keys, func(k string) any { return k }))
		}
		var lines []string
		any := false
		for _, k := range j.Keys {
			label, desc := j.Titles[k], j.Descriptions[k]
			if label == "" && desc == "" {
				lines = append(lines, k) // a bare key still tells the model it is an option
				continue
			}
			any = true
			switch {
			case label != "" && desc != "":
				lines = append(lines, k+" = "+label+": "+desc)
			case label != "":
				lines = append(lines, k+" = "+label)
			default:
				lines = append(lines, k+" = "+desc)
			}
		}
		if any {
			head := optDescription(prop, "description")
			word := "Options: "
			if j.Ordered() {
				word = "Levels: "
			}
			if head != "" {
				head += " "
			}
			prop.Set("description", strings.TrimSpace(head+word+strings.Join(lines, "; ")))
		}
		props.Set(j.Name, prop)
	}
	if props != nil {
		out.Set("properties", props)
	}
	return out
}

// ─── §3 the answer ────────────────────────────────────────────────────

// DataPartFromText is the model's JSON object as a DataPart (value only),
// or false when the text is not a JSON object (a truncated answer stays a
// TextPart).
func DataPartFromText(text string, found []Judgment) (DataPart, bool) {
	if len(found) == 0 {
		return DataPart{}, false
	}
	value, err := DecodeJSON([]byte(strings.TrimSpace(text)))
	if err != nil {
		return DataPart{}, false
	}
	obj, ok := asObject(value)
	if !ok {
		return DataPart{}, false
	}
	return DataPart{Value: obj}, true
}

// ReplaceTextWithData swaps the single text part of a judgment answer for
// its DataPart (MAP-14 §3). Any other shape is returned unchanged.
func ReplaceTextWithData(parts []Part, found []Judgment) []Part {
	if len(found) == 0 {
		return parts
	}
	textIndex, texts := -1, 0
	for i, p := range parts {
		if _, ok := p.(TextPart); ok {
			texts++
			textIndex = i
		}
	}
	if texts != 1 {
		return parts
	}
	text := parts[textIndex].(TextPart)
	part, ok := DataPartFromText(text.Text, found)
	if !ok {
		return parts
	}
	part.Continuation = text.Continuation
	out := append([]Part(nil), parts...)
	out[textIndex] = part
	return out
}

// DataPartText is a data part on a wire that takes only text: its value as
// compact canonical JSON, nothing added (changes/2026-09-19-jev-state.md
// D3; types.md §DataPart). An opaque payload: numbers as written.
func DataPartText(part DataPart) string {
	return string(mustJSON(deref(part.Value)))
}

// normalizeLogprobs is a softmax over log-scores: one normalisation over
// the key set.
func normalizeLogprobs(scores map[string]float64) map[string]float64 {
	top := math.Inf(-1)
	for _, v := range scores {
		if v > top {
			top = v
		}
	}
	weights := make(map[string]float64, len(scores))
	total := 0.0
	for k, v := range scores {
		w := math.Exp(v - top)
		weights[k] = w
		total += w
	}
	for k, w := range weights {
		weights[k] = w / total
	}
	return weights
}

// ExpectedLevel is Σ p·i over an ordered judgment's distribution.
func ExpectedLevel(distribution map[string]float64) (float64, error) {
	total := 0.0
	for k, p := range distribution {
		i, err := strconv.Atoi(k)
		if err != nil {
			return 0, valueErrorf("expected level: key %q is not a level index", k)
		}
		total += p * float64(i)
	}
	return total, nil
}

// ─── §4 sugar that emits the convention ───────────────────────────────

// ChoiceOption is one option of a Choice judgment.
type ChoiceOption struct {
	Key         string
	Description string
}

// Choice emits a choice judgment property: keys with optional descriptions.
func Choice(instruction string, options ...ChoiceOption) (JSONObject, error) {
	if len(options) == 0 {
		return nil, valueErrorf("choice needs at least one option")
	}
	seen := map[string]bool{}
	anyDesc := false
	for _, o := range options {
		if o.Key == "" {
			return nil, typeErrorf("choice option keys must be non-empty strings")
		}
		if seen[o.Key] {
			return nil, valueErrorf("choice option keys must be unique")
		}
		seen[o.Key] = true
		if o.Description != "" {
			anyDesc = true
		}
	}
	prop := JSONObject{{"type", "string"}, {"description", instruction}}
	if !anyDesc {
		prop.Set("enum", toAnyList(options, func(o ChoiceOption) any { return o.Key }))
	} else {
		prop.Set("anyOf", toAnyList(options, func(o ChoiceOption) any {
			b := JSONObject{{"const", o.Key}}
			if o.Description != "" {
				b.Set("description", o.Description)
			}
			return b
		}))
	}
	return prop, nil
}

// Options builds ChoiceOptions from bare keys.
func Options(keys ...string) []ChoiceOption {
	out := make([]ChoiceOption, 0, len(keys))
	for _, k := range keys {
		out = append(out, ChoiceOption{Key: k})
	}
	return out
}

// YesNo emits a boolean judgment property.
func YesNo(instruction string) JSONObject {
	return JSONObject{{"type", "boolean"}, {"description", instruction}}
}

// ScoreLevel is one level of a Score judgment, low → high.
type ScoreLevel struct {
	Name        string
	Description string
}

// Score emits an ordered judgment: levels low → high.
func Score(instruction string, levels ...ScoreLevel) (JSONObject, error) {
	if len(levels) < 2 {
		return nil, valueErrorf("score needs at least two levels")
	}
	if len(levels) > MaxOrderedLevels {
		return nil, valueErrorf("score takes at most %d levels", MaxOrderedLevels)
	}
	branches := make([]any, 0, len(levels))
	for i, l := range levels {
		b := JSONObject{{"const", i}}
		if l.Name != "" {
			b.Set("title", l.Name)
		}
		if l.Description != "" {
			b.Set("description", l.Description)
		}
		branches = append(branches, b)
	}
	return JSONObject{{"type", "integer"}, {"description", instruction}, {"anyOf", branches}}, nil
}

// JudgmentProperty is one named judgment for Judgments.
type JudgmentProperty struct {
	Name   string
	Schema JSONObject
}

// Judgments emits a response_format declaring the given judgment
// properties in the order given: the schema lists them, and required
// names them, in that order.
func Judgments(name string, strict bool, properties ...JudgmentProperty) (JSONObject, error) {
	if len(properties) == 0 {
		return nil, valueErrorf("judgments needs at least one property")
	}
	if name == "" {
		name = "judgments"
	}
	props := JSONObject{}
	required := make([]any, 0, len(properties))
	for _, p := range properties {
		props.Set(p.Name, p.Schema)
		required = append(required, p.Name)
	}
	schema := JSONObject{{"type", "object"}, {"properties", props}, {"required", required}, {"additionalProperties", false}}
	return JSONObject{{"type", "json_schema"}, {"name", name}, {"strict", strict}, {"schema", schema}}, nil
}
