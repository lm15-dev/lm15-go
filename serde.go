package lm15

import (
	"encoding/base64"
	"encoding/json"
)

// Canonical JSON serde: one ToDict / FromDict pair per type, obeying the
// omission rule (docs/serde-rules.md) and the from_dict leniencies
// (spec/invariants.md INV-040..048). FromDict delegates every check to
// Validate (INV-046). json.Marshal / json.Unmarshal on the canonical types
// go through the same functions.

// ─── generic plumbing ────────────────────────────────────────────────

func marshalVia(f func() JSONObject) ([]byte, error) { return EncodeJSON(f()) }

func unmarshalVia(data []byte, f func(JSONObject) error) error {
	obj, err := DecodeJSONObject(data)
	if err != nil {
		return err
	}
	return f(obj)
}

func continuationToJSON(states []ContinuationState) []any {
	if len(states) == 0 {
		return nil
	}
	return toAnyList(states, func(s ContinuationState) any { return ContinuationToDict(s) })
}

func continuationFromJSON(v any) ([]ContinuationState, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, typeErrorf("continuation must be a list")
	}
	out := make([]ContinuationState, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, typeErrorf("continuation entries must be objects")
		}
		s, err := ContinuationFromDict(obj)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// ─── ContinuationState ───────────────────────────────────────────────

// ContinuationToDict serializes a state (data is always emitted).
func ContinuationToDict(s ContinuationState) JSONObject {
	data := s.Data
	if data == nil {
		data = JSONObject{}
	}
	return JSONObject{"provider": s.Provider, "kind": s.Kind, "data": data}
}

// ContinuationFromDict reads a state.
func ContinuationFromDict(d JSONObject) (ContinuationState, error) {
	provider, err := reqString(d, "provider")
	if err != nil {
		return ContinuationState{}, err
	}
	kind, err := reqString(d, "kind")
	if err != nil {
		return ContinuationState{}, err
	}
	data, err := optObject(d, "data")
	if err != nil {
		return ContinuationState{}, err
	}
	if data == nil {
		data = JSONObject{}
	}
	s := ContinuationState{Provider: provider, Kind: kind, Data: data}
	return s, s.Validate()
}

func (s ContinuationState) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ContinuationToDict(s) })
}

func (s *ContinuationState) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := ContinuationFromDict(d)
		if err != nil {
			return err
		}
		*s = v
		return nil
	})
}

// ─── Parts ───────────────────────────────────────────────────────────

// PartToDict serializes a part.
func PartToDict(p Part) JSONObject {
	d := JSONObject{"type": p.Type()}
	switch x := p.(type) {
	case TextPart:
		d["text"] = x.Text
	case ThinkingPart:
		d["text"] = x.Text
	case RefusalPart:
		d["text"] = x.Text
	case CitationPart:
		if x.Text != "" {
			d["text"] = x.Text
		}
		if x.URL != "" {
			d["url"] = x.URL
		}
		if x.Title != "" {
			d["title"] = x.Title
		}
	case ImagePart:
		mediaToDict(d, x.Media)
		if x.Detail != "" {
			d["detail"] = x.Detail
		}
	case AudioPart:
		mediaToDict(d, x.Media)
	case VideoPart:
		mediaToDict(d, x.Media)
	case DocumentPart:
		mediaToDict(d, x.Media)
	case BinaryPart:
		mediaToDict(d, x.Media)
	case ToolCallPart:
		d["id"] = x.ID
		d["name"] = x.Name
		input := x.Input
		if input == nil {
			input = JSONObject{}
		}
		d["input"] = input
	case ToolResultPart:
		d["id"] = x.ID
		if x.Name != "" {
			d["name"] = x.Name
		}
		d["content"] = toAnyList(x.Content, func(c Part) any { return PartToDict(c) })
		if x.IsError {
			d["is_error"] = true
		}
	case DataPart:
		// value is always emitted, whatever it is (null is a value; the
		// cleaner never looks inside — serde-rules.md "Data parts").
		d["value"] = deref(x.Value)
		if x.Probabilities != nil {
			d["probabilities"] = probabilitiesToJSON(x.Probabilities)
		}
		if x.Method != "" {
			d["method"] = x.Method
		}
	}
	if c := continuationToJSON(p.ContinuationStates()); c != nil {
		d["continuation"] = c
	}
	return d
}

func mediaToDict(d JSONObject, m Media) {
	d["media_type"] = m.MediaType
	if m.Data != "" {
		d["data"] = m.Data
	}
	if m.URL != "" {
		d["url"] = m.URL
	}
	if m.FileID != "" {
		d["file_id"] = m.FileID
	}
	if m.Path != "" {
		d["path"] = m.Path
	}
}

func mediaFromDict(d JSONObject) (Media, error) {
	var m Media
	var err error
	if m.MediaType, err = optString(d, "media_type"); err != nil {
		return m, err
	}
	if m.Data, err = optString(d, "data"); err != nil {
		return m, err
	}
	if m.URL, err = optString(d, "url"); err != nil {
		return m, err
	}
	if m.FileID, err = optString(d, "file_id"); err != nil {
		return m, err
	}
	if m.Path, err = optString(d, "path"); err != nil {
		return m, err
	}
	if m.Continuation, err = continuationFromJSON(d["continuation"]); err != nil {
		return m, err
	}
	return m, nil
}

// PartFromDict reads a part (INV-040, INV-041, INV-044).
func PartFromDict(d JSONObject) (Part, error) {
	t, err := reqString(d, "type")
	if err != nil {
		return nil, err
	}
	continuation, err := continuationFromJSON(d["continuation"])
	if err != nil {
		return nil, err
	}
	var part Part
	switch t {
	case PartTypeText, PartTypeThinking, PartTypeRefusal:
		text, err := optString(d, "text")
		if err != nil {
			return nil, err
		}
		switch t {
		case PartTypeText:
			part = TextPart{Text: text, Continuation: continuation}
		case PartTypeThinking:
			part = ThinkingPart{Text: text, Continuation: continuation}
		default:
			part = RefusalPart{Text: text, Continuation: continuation}
		}
	case PartTypeCitation:
		text, err := optString(d, "text")
		if err != nil {
			return nil, err
		}
		url, err := optString(d, "url")
		if err != nil {
			return nil, err
		}
		title, err := optString(d, "title")
		if err != nil {
			return nil, err
		}
		part = CitationPart{Text: text, URL: url, Title: title, Continuation: continuation}
	case PartTypeImage, PartTypeAudio, PartTypeVideo, PartTypeDocument, PartTypeBinary:
		m, err := mediaFromDict(d)
		if err != nil {
			return nil, err
		}
		switch t {
		case PartTypeImage:
			detail, err := optString(d, "detail")
			if err != nil {
				return nil, err
			}
			part = ImagePart{Media: m, Detail: detail}
		case PartTypeAudio:
			part = AudioPart{Media: m}
		case PartTypeVideo:
			part = VideoPart{Media: m}
		case PartTypeDocument:
			part = DocumentPart{Media: m}
		default:
			part = BinaryPart{Media: m}
		}
	case PartTypeToolCall:
		id, err := reqString(d, "id")
		if err != nil {
			return nil, err
		}
		name, err := reqString(d, "name")
		if err != nil {
			return nil, err
		}
		input, err := optObject(d, "input")
		if err != nil {
			return nil, err
		}
		if input == nil {
			input = JSONObject{}
		}
		part = ToolCallPart{ID: id, Name: name, Input: input, Continuation: continuation}
	case PartTypeToolResult:
		id, err := reqString(d, "id")
		if err != nil {
			return nil, err
		}
		var content []Part
		switch raw := d["content"].(type) {
		case string:
			if raw != "" {
				content = []Part{TextPart{Text: raw}}
			}
		case []any:
			for _, c := range raw {
				if obj, ok := c.(map[string]any); ok {
					p, err := PartFromDict(obj)
					if err != nil {
						return nil, err
					}
					content = append(content, p)
				} else {
					content = append(content, TextPart{Text: wireStr(c)})
				}
			}
		}
		name, err := optString(d, "name")
		if err != nil {
			return nil, err
		}
		isError := false
		if v, ok := d["is_error"]; ok && v != nil {
			b, ok := v.(bool)
			if !ok {
				return nil, typeErrorf("ToolResultPart.is_error must be a bool")
			}
			isError = b
		}
		part = ToolResultPart{ID: id, Content: content, Name: name, IsError: isError, Continuation: continuation}
	case PartTypeData:
		value, present := d["value"]
		if !present {
			return nil, keyError("value")
		}
		probabilities, err := probabilitiesFromJSON(d["probabilities"])
		if err != nil {
			return nil, err
		}
		method, err := optString(d, "method")
		if err != nil {
			return nil, err
		}
		part = DataPart{Value: value, Probabilities: probabilities, Method: method, Continuation: continuation}
	default:
		return nil, valueErrorf("unsupported part type: %s", t)
	}
	return part, part.Validate()
}

// probabilitiesToJSON renders the nested maps with every probability as a
// JSON float (1.0, never 1).
func probabilitiesToJSON(p map[string]map[string]float64) JSONObject {
	out := JSONObject{}
	for name, dist := range p {
		inner := JSONObject{}
		for key, prob := range dist {
			inner[key] = jsonFloat(prob)
		}
		out[name] = inner
	}
	return out
}

func probabilitiesFromJSON(v any) (map[string]map[string]float64, error) {
	if v == nil {
		return nil, nil
	}
	outer, ok := v.(map[string]any)
	if !ok {
		return nil, typeErrorf("DataPart.probabilities must be a mapping of field -> {key: probability}")
	}
	out := make(map[string]map[string]float64, len(outer))
	for name, raw := range outer {
		inner, ok := raw.(map[string]any)
		if !ok {
			return nil, typeErrorf("DataPart.probabilities[%q] must be a mapping of key -> probability", name)
		}
		dist := make(map[string]float64, len(inner))
		for key, prob := range inner {
			if _, isBool := prob.(bool); isBool {
				return nil, typeErrorf("DataPart.probabilities[%q][%q] must be a number", name, key)
			}
			f, err := jsonFloat64(prob, "DataPart.probabilities")
			if err != nil {
				return nil, typeErrorf("DataPart.probabilities[%q][%q] must be a number", name, key)
			}
			dist[key] = f
		}
		out[name] = dist
	}
	return out, nil
}

// partsFromList reads a list of part dicts (INV-047: scalars become text).
func partsFromList(v any) ([]Part, error) {
	list, _ := v.([]any)
	out := make([]Part, 0, len(list))
	for _, item := range list {
		if obj, ok := item.(map[string]any); ok {
			p, err := PartFromDict(obj)
			if err != nil {
				return nil, err
			}
			out = append(out, p)
		} else {
			out = append(out, TextPart{Text: wireStr(item)})
		}
	}
	return out, nil
}

func partsToList(parts []Part) []any {
	return toAnyList(parts, func(p Part) any { return PartToDict(p) })
}

// MarshalJSON on every part variant.
func (p TextPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p ThinkingPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p RefusalPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p CitationPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p ImagePart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p AudioPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p VideoPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p DocumentPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p BinaryPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p ToolCallPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p DataPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}
func (p ToolResultPart) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return PartToDict(p) })
}

// ─── Messages ────────────────────────────────────────────────────────

// MessageToDict serializes a message.
func MessageToDict(m Message) JSONObject {
	d := JSONObject{"role": m.Role, "parts": partsToList(m.Parts)}
	if c := continuationToJSON(m.Continuation); c != nil {
		d["continuation"] = c
	}
	return d
}

// MessageFromDict reads a message (INV-047).
func MessageFromDict(d JSONObject) (Message, error) {
	role, err := reqString(d, "role")
	if err != nil {
		return Message{}, err
	}
	parts, err := partsFromList(d["parts"])
	if err != nil {
		return Message{}, err
	}
	if len(parts) == 0 {
		return Message{}, valueErrorf("message for role '%s' has no parts", role)
	}
	continuation, err := continuationFromJSON(d["continuation"])
	if err != nil {
		return Message{}, err
	}
	m := Message{Role: role, Parts: parts, Continuation: continuation}
	return m, m.Validate()
}

func (m Message) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return MessageToDict(m) })
}

func (m *Message) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := MessageFromDict(d)
		if err != nil {
			return err
		}
		*m = v
		return nil
	})
}

// ─── Tools ───────────────────────────────────────────────────────────

// ToolToDict serializes a tool (parameters always emitted, INV-033).
func ToolToDict(t Tool) JSONObject {
	switch x := t.(type) {
	case FunctionTool:
		d := dict{"type": "function", "name": x.Name}.omit("description", x.Description)
		d["parameters"] = x.EffectiveParameters()
		return JSONObject(d)
	case BuiltinTool:
		return JSONObject(dict{"type": "builtin", "name": x.Name}.omit("config", x.Config))
	}
	return nil
}

// ToolFromDict reads a tool (INV-034 dispatch).
func ToolFromDict(d JSONObject) (Tool, error) {
	name, err := reqString(d, "name")
	if err != nil {
		return nil, err
	}
	if t, _ := d["type"].(string); t == "builtin" {
		config, err := optObject(d, "config")
		if err != nil {
			return nil, err
		}
		tool := BuiltinTool{Name: name, Config: config}
		return tool, tool.Validate()
	}
	description, err := optString(d, "description")
	if err != nil {
		return nil, err
	}
	params, err := optObject(d, "parameters")
	if err != nil {
		return nil, err
	}
	if _, present := d["parameters"]; present && params == nil {
		if d["parameters"] != nil {
			return nil, typeErrorf("parameters must be a JSON object")
		}
	}
	if params == nil {
		params = DefaultParameters()
	}
	tool := FunctionTool{Name: name, Description: description, Parameters: params}
	return tool, tool.Validate()
}

func (t FunctionTool) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ToolToDict(t) })
}

func (t BuiltinTool) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ToolToDict(t) })
}

// ─── Config family ───────────────────────────────────────────────────

// ToolChoiceToDict serializes a tool choice.
func ToolChoiceToDict(tc ToolChoice) JSONObject {
	return JSONObject(dict{"mode": tc.EffectiveMode()}.omit("allowed", tc.Allowed).omit("parallel", tc.Parallel))
}

// ToolChoiceFromDict reads a tool choice (INV-045 mode → auto).
func ToolChoiceFromDict(d JSONObject) (ToolChoice, error) {
	mode, err := optString(d, "mode")
	if err != nil {
		return ToolChoice{}, err
	}
	if mode == "" {
		mode = "auto"
	}
	allowed, err := stringList(d["allowed"], "allowed")
	if err != nil {
		return ToolChoice{}, err
	}
	parallel, err := optBool(d, "parallel")
	if err != nil {
		return ToolChoice{}, typeErrorf("ToolChoice.parallel must be a bool")
	}
	tc := ToolChoice{Mode: mode, Allowed: allowed, Parallel: parallel}
	return tc, tc.Validate()
}

// ReasoningToDict serializes reasoning.
func ReasoningToDict(r Reasoning) JSONObject {
	return JSONObject(dict{"effort": r.Effort}.omit("thinking_budget", r.ThinkingBudget).omit("summary", r.Summary))
}

// ReasoningFromDict reads reasoning with the legacy leniencies (INV-043).
func ReasoningFromDict(d JSONObject) (Reasoning, error) {
	defaultEffort := "medium"
	if enabled, ok := d["enabled"].(bool); ok && !enabled {
		defaultEffort = "off"
	}
	effort := defaultEffort
	if v, ok := d["effort"]; ok && v != nil {
		s, ok := v.(string)
		if !ok {
			return Reasoning{}, valueErrorf("unsupported reasoning effort: %v", v)
		}
		effort = s
	}
	if effort == "adaptive" {
		effort = "medium"
	}
	if effort == "off" {
		r := Reasoning{Effort: "off"}
		return r, r.Validate()
	}
	budget, err := optInt(d, "thinking_budget")
	if err != nil {
		return Reasoning{}, err
	}
	if budget == nil {
		if budget, err = optInt(d, "budget"); err != nil {
			return Reasoning{}, err
		}
	}
	summary, err := optString(d, "summary")
	if err != nil {
		return Reasoning{}, err
	}
	r := Reasoning{Effort: effort, ThinkingBudget: budget, Summary: summary}
	return r, r.Validate()
}

// CacheConfigToDict serializes a cache config.
func CacheConfigToDict(c CacheConfig) JSONObject {
	return JSONObject(dict{"mode": c.EffectiveMode()}.
		omit("retention", c.Retention).omit("key", c.Key).omit("prefix_until_index", c.PrefixUntilIndex).
		omit("prefix", c.Prefix).omit("resource", c.Resource))
}

// CacheConfigFromDict reads a cache config.
func CacheConfigFromDict(d JSONObject) (CacheConfig, error) {
	mode, err := optString(d, "mode")
	if err != nil {
		return CacheConfig{}, err
	}
	if mode == "" {
		mode = "auto"
	}
	if !inVocab(mode, CacheModes) {
		return CacheConfig{}, valueErrorf("unsupported cache mode: %s", mode)
	}
	retention, err := optString(d, "retention")
	if err != nil {
		return CacheConfig{}, err
	}
	if retention != "" && !inVocab(retention, CacheRetentions) {
		return CacheConfig{}, valueErrorf("unsupported cache retention: %s", retention)
	}
	key, err := optString(d, "key")
	if err != nil {
		return CacheConfig{}, err
	}
	idx, err := optInt(d, "prefix_until_index")
	if err != nil {
		return CacheConfig{}, err
	}
	prefix, err := optString(d, "prefix")
	if err != nil {
		return CacheConfig{}, err
	}
	resource, err := optString(d, "resource")
	if err != nil {
		return CacheConfig{}, err
	}
	c := CacheConfig{Mode: mode, Retention: retention, Key: key, PrefixUntilIndex: idx, Prefix: prefix, Resource: resource}
	return c, c.Validate()
}

// ConfigToDict serializes a config (store=false and logprobs=0 are data).
func ConfigToDict(c Config) JSONObject {
	d := dict{}.
		omit("max_tokens", c.MaxTokens).omit("temperature", c.Temperature).omit("top_p", c.TopP).omit("top_k", c.TopK).
		omit("stop", c.Stop).omit("response_format", c.ResponseFormat)
	if c.ToolChoice != nil {
		d["tool_choice"] = ToolChoiceToDict(*c.ToolChoice)
	}
	if c.Reasoning != nil {
		d["reasoning"] = ReasoningToDict(*c.Reasoning)
	}
	if c.Cache != nil {
		d["cache"] = CacheConfigToDict(*c.Cache)
	}
	// seed 0 and a 0.0 penalty are data, emitted (spec/types.md § Config).
	if c.Seed != nil {
		d["seed"] = *c.Seed
	}
	if c.FrequencyPenalty != nil {
		d["frequency_penalty"] = jsonFloat(*c.FrequencyPenalty)
	}
	if c.PresencePenalty != nil {
		d["presence_penalty"] = jsonFloat(*c.PresencePenalty)
	}
	d.omit("service_tier", c.ServiceTier).omit("user_id", c.UserID).omit("store", c.Store).omit("logprobs", c.Logprobs).
		omit("probabilities", c.Probabilities).omit("extensions", c.Extensions)
	return JSONObject(d)
}

func configNest(d JSONObject, key string) (JSONObject, error) {
	v, ok := d[key]
	if !ok || v == nil {
		return nil, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, typeErrorf("config.%s must be a JSON object, got %s", key, jsonTypeName(v))
	}
	return m, nil
}

// ConfigFromDict reads a config (INV-042: malformed nests are errors).
func ConfigFromDict(d JSONObject) (Config, error) {
	var c Config
	var err error
	if c.MaxTokens, err = optInt(d, "max_tokens"); err != nil {
		return c, err
	}
	if c.Temperature, err = optFloat(d, "temperature"); err != nil {
		return c, err
	}
	if c.TopP, err = optFloat(d, "top_p"); err != nil {
		return c, err
	}
	if c.TopK, err = optInt(d, "top_k"); err != nil {
		return c, err
	}
	if c.Stop, err = stringList(d["stop"], "stop"); err != nil {
		return c, err
	}
	if rf, ok := d["response_format"]; ok && rf != nil {
		m, ok := rf.(map[string]any)
		if !ok {
			return c, typeErrorf("response_format must be a JSON object")
		}
		c.ResponseFormat = m
	}
	tc, err := configNest(d, "tool_choice")
	if err != nil {
		return c, err
	}
	if tc != nil {
		v, err := ToolChoiceFromDict(tc)
		if err != nil {
			return c, err
		}
		c.ToolChoice = &v
	}
	rs, err := configNest(d, "reasoning")
	if err != nil {
		return c, err
	}
	if rs != nil {
		v, err := ReasoningFromDict(rs)
		if err != nil {
			return c, err
		}
		c.Reasoning = &v
	}
	cc, err := configNest(d, "cache")
	if err != nil {
		return c, err
	}
	if cc != nil {
		v, err := CacheConfigFromDict(cc)
		if err != nil {
			return c, err
		}
		c.Cache = &v
	}
	if c.ServiceTier, err = optString(d, "service_tier"); err != nil {
		return c, err
	}
	if c.UserID, err = optString(d, "user_id"); err != nil {
		return c, err
	}
	if c.Store, err = optBool(d, "store"); err != nil {
		return c, typeErrorf("Config.store must be a bool or None")
	}
	if c.Logprobs, err = optInt(d, "logprobs"); err != nil {
		return c, err
	}
	if c.Seed, err = optInt(d, "seed"); err != nil {
		return c, err
	}
	if c.FrequencyPenalty, err = optFloat(d, "frequency_penalty"); err != nil {
		return c, err
	}
	if c.PresencePenalty, err = optFloat(d, "presence_penalty"); err != nil {
		return c, err
	}
	if c.Probabilities, err = optString(d, "probabilities"); err != nil {
		return c, err
	}
	if ext, ok := d["extensions"]; ok && ext != nil {
		m, ok := ext.(map[string]any)
		if !ok {
			return c, typeErrorf("extensions must be a JSON object")
		}
		c.Extensions = m
	}
	if c.Extensions, err = normalizeExtensions(c.Extensions); err != nil {
		return c, err
	}
	return c, c.Validate()
}

func (c Config) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ConfigToDict(c) })
}
func (t ToolChoice) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ToolChoiceToDict(t) })
}
func (r Reasoning) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ReasoningToDict(r) })
}
func (c CacheConfig) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return CacheConfigToDict(c) })
}

func (c *Config) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := ConfigFromDict(d)
		if err != nil {
			return err
		}
		*c = v
		return nil
	})
}

// ─── Logprobs ────────────────────────────────────────────────────────

func topLogprobToDict(t TopLogprob) JSONObject {
	d := JSONObject{"token": t.Token, "logprob": jsonFloat(t.Logprob)}
	if t.Bytes != nil {
		d["bytes"] = toAnyList(t.Bytes, func(b int) any { return b })
	}
	if t.TokenID != nil {
		d["token_id"] = *t.TokenID
	}
	return d
}

// TokenLogprobToDict serializes a token logprob.
func TokenLogprobToDict(t TokenLogprob) JSONObject {
	d := topLogprobToDict(TopLogprob{Token: t.Token, Logprob: t.Logprob, Bytes: t.Bytes, TokenID: t.TokenID})
	if len(t.Top) > 0 {
		d["top"] = toAnyList(t.Top, func(x TopLogprob) any { return topLogprobToDict(x) })
	}
	return d
}

func logprobBytes(v any) ([]int, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, typeErrorf("bytes must contain non-negative ints")
	}
	out := make([]int, 0, len(list))
	for _, item := range list {
		i, err := jsonInt(item, "bytes")
		if err != nil {
			return nil, typeErrorf("bytes must contain non-negative ints")
		}
		out = append(out, i)
	}
	return out, nil
}

func topLogprobFromDict(d JSONObject) (TopLogprob, error) {
	token, err := reqString(d, "token")
	if err != nil {
		return TopLogprob{}, err
	}
	v, ok := d["logprob"]
	if !ok {
		return TopLogprob{}, keyError("logprob")
	}
	logprob, err := jsonFloat64(v, "logprob")
	if err != nil {
		return TopLogprob{}, typeErrorf("logprob must be a float")
	}
	bytesList, err := logprobBytes(d["bytes"])
	if err != nil {
		return TopLogprob{}, err
	}
	tokenID, err := optInt(d, "token_id")
	if err != nil {
		return TopLogprob{}, err
	}
	t := TopLogprob{Token: token, Logprob: logprob, Bytes: bytesList, TokenID: tokenID}
	return t, t.Validate()
}

// TokenLogprobFromDict reads a token logprob.
func TokenLogprobFromDict(d JSONObject) (TokenLogprob, error) {
	base, err := topLogprobFromDict(d)
	if err != nil {
		return TokenLogprob{}, err
	}
	var top []TopLogprob
	if raw, ok := d["top"]; ok && raw != nil {
		list, ok := raw.([]any)
		if !ok {
			return TokenLogprob{}, typeErrorf("TokenLogprob.top must contain TopLogprob objects")
		}
		for _, item := range list {
			obj, ok := item.(map[string]any)
			if !ok {
				return TokenLogprob{}, typeErrorf("TokenLogprob.top must contain TopLogprob objects")
			}
			x, err := topLogprobFromDict(obj)
			if err != nil {
				return TokenLogprob{}, err
			}
			top = append(top, x)
		}
	}
	t := TokenLogprob{Token: base.Token, Logprob: base.Logprob, Bytes: base.Bytes, TokenID: base.TokenID, Top: top}
	return t, t.Validate()
}

func logprobsToJSON(lps []TokenLogprob) []any {
	if len(lps) == 0 {
		return nil
	}
	return toAnyList(lps, func(t TokenLogprob) any { return TokenLogprobToDict(t) })
}

func logprobsFromJSON(v any) ([]TokenLogprob, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, typeErrorf("logprobs must be a list")
	}
	out := make([]TokenLogprob, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, typeErrorf("logprobs must contain TokenLogprob objects")
		}
		t, err := TokenLogprobFromDict(obj)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

func (t TokenLogprob) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return TokenLogprobToDict(t) })
}

// ─── ErrorDetail ─────────────────────────────────────────────────────

// ErrorDetailToDict serializes an error detail.
func ErrorDetailToDict(e ErrorDetail) JSONObject {
	d := dict{"code": e.Code}.omit("message", e.Message).omit("provider_code", e.ProviderCode)
	if h := e.HTTPResponse.toDict(); h != nil {
		d["http_response"] = h
	}
	return JSONObject(d)
}

// ErrorDetailFromDict reads an error detail (message defaults to "").
func ErrorDetailFromDict(d JSONObject) (ErrorDetail, error) {
	code, err := reqString(d, "code")
	if err != nil {
		return ErrorDetail{}, err
	}
	message, err := optString(d, "message")
	if err != nil {
		return ErrorDetail{}, err
	}
	providerCode, err := optString(d, "provider_code")
	if err != nil {
		return ErrorDetail{}, err
	}
	http, err := httpResponseDetailFromJSON(d["http_response"])
	if err != nil {
		return ErrorDetail{}, err
	}
	e := ErrorDetail{Code: code, Message: message, ProviderCode: providerCode, HTTPResponse: http}
	return e, e.Validate()
}

func (e ErrorDetail) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ErrorDetailToDict(e) })
}

// ─── Deltas ──────────────────────────────────────────────────────────

// DeltaToDict serializes a delta: only nil fields are dropped; empty strings
// are emitted; part_index is always emitted (except a nil ContinuationDelta index).
func DeltaToDict(dl Delta) JSONObject {
	out := dict{"type": dl.Type()}
	switch x := dl.(type) {
	case TextDelta:
		out["part_index"] = x.PartIndex
		out["text"] = x.Text
		if len(x.Logprobs) > 0 {
			out["logprobs"] = logprobsToJSON(x.Logprobs)
		}
		if x.LogprobsIncomplete {
			out["logprobs_complete"] = false
		}
	case ThinkingDelta:
		out["part_index"] = x.PartIndex
		out["text"] = x.Text
	case AudioDelta:
		out["part_index"] = x.PartIndex
		out.omitNull("data", x.Data).omitNull("url", x.URL).omitNull("file_id", x.FileID)
		if x.MediaType != "" {
			out["media_type"] = x.MediaType
		}
	case ImageDelta:
		out["part_index"] = x.PartIndex
		out.omitNull("data", x.Data).omitNull("url", x.URL).omitNull("file_id", x.FileID)
		if x.MediaType != "" {
			out["media_type"] = x.MediaType
		}
	case ToolCallDelta:
		out["part_index"] = x.PartIndex
		out["input"] = x.Input
		if x.ID != "" {
			out["id"] = x.ID
		}
		if x.Name != "" {
			out["name"] = x.Name
		}
	case CitationDelta:
		out["part_index"] = x.PartIndex
		out.omitNull("text", x.Text).omitNull("url", x.URL).omitNull("title", x.Title)
	case ContinuationDelta:
		out["provider"] = x.Provider
		out["kind"] = x.Kind
		data := x.Data
		if data == nil {
			data = JSONObject{}
		}
		out["data"] = data
		if x.PartIndex != nil {
			out["part_index"] = *x.PartIndex
		}
	}
	return JSONObject(out)
}

// DeltaFromDict reads a delta (part_index defaults to 0).
func DeltaFromDict(d JSONObject) (Delta, error) {
	t, err := reqString(d, "type")
	if err != nil {
		return nil, err
	}
	partIndex := 0
	if t != DeltaTypeContinuation {
		if p, err := optInt(d, "part_index"); err != nil {
			return nil, err
		} else if p != nil {
			partIndex = *p
		}
	}
	var dl Delta
	switch t {
	case DeltaTypeText:
		text, err := optString(d, "text")
		if err != nil {
			return nil, err
		}
		lps, err := logprobsFromJSON(d["logprobs"])
		if err != nil {
			return nil, err
		}
		complete, err := optBool(d, "logprobs_complete")
		if err != nil {
			return nil, typeErrorf("TextDelta.logprobs_complete must be a bool")
		}
		dl = TextDelta{Text: text, PartIndex: partIndex, Logprobs: lps, LogprobsIncomplete: complete != nil && !*complete}
	case DeltaTypeThinking:
		text, err := optString(d, "text")
		if err != nil {
			return nil, err
		}
		dl = ThinkingDelta{Text: text, PartIndex: partIndex}
	case DeltaTypeAudio, DeltaTypeImage:
		data, err := optStringPtr(d, "data")
		if err != nil {
			return nil, err
		}
		url, err := optStringPtr(d, "url")
		if err != nil {
			return nil, err
		}
		fileID, err := optStringPtr(d, "file_id")
		if err != nil {
			return nil, err
		}
		mediaType, err := optString(d, "media_type")
		if err != nil {
			return nil, err
		}
		if t == DeltaTypeAudio {
			dl = AudioDelta{Data: data, URL: url, FileID: fileID, PartIndex: partIndex, MediaType: mediaType}
		} else {
			dl = ImageDelta{Data: data, URL: url, FileID: fileID, PartIndex: partIndex, MediaType: mediaType}
		}
	case DeltaTypeToolCall:
		input, err := optString(d, "input")
		if err != nil {
			return nil, err
		}
		id, err := optString(d, "id")
		if err != nil {
			return nil, err
		}
		name, err := optString(d, "name")
		if err != nil {
			return nil, err
		}
		dl = ToolCallDelta{Input: input, PartIndex: partIndex, ID: id, Name: name}
	case DeltaTypeCitation:
		text, err := optStringPtr(d, "text")
		if err != nil {
			return nil, err
		}
		url, err := optStringPtr(d, "url")
		if err != nil {
			return nil, err
		}
		title, err := optStringPtr(d, "title")
		if err != nil {
			return nil, err
		}
		dl = CitationDelta{Text: text, URL: url, Title: title, PartIndex: partIndex}
	case DeltaTypeContinuation:
		provider, err := reqString(d, "provider")
		if err != nil {
			return nil, err
		}
		kind, err := reqString(d, "kind")
		if err != nil {
			return nil, err
		}
		data, err := optObject(d, "data")
		if err != nil {
			return nil, err
		}
		if data == nil {
			data = JSONObject{}
		}
		idx, err := optInt(d, "part_index")
		if err != nil {
			return nil, err
		}
		dl = ContinuationDelta{Provider: provider, Kind: kind, Data: data, PartIndex: idx}
	default:
		return nil, valueErrorf("unsupported delta type: %s", t)
	}
	return dl, dl.Validate()
}

func (d TextDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}
func (d ThinkingDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}
func (d AudioDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}
func (d ImageDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}
func (d ToolCallDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}
func (d CitationDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}
func (d ContinuationDelta) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return DeltaToDict(d) })
}

// ─── Usage ───────────────────────────────────────────────────────────

// UsageToDict serializes usage (every counter omit-empty).
func UsageToDict(u Usage) JSONObject {
	return JSONObject(dict{}.
		omit("input_tokens", u.InputTokens).omit("output_tokens", u.OutputTokens).omit("total_tokens", u.TotalTokens).
		omit("cache_read_tokens", u.CacheReadTokens).omit("cache_write_tokens", u.CacheWriteTokens).
		omit("reasoning_tokens", u.ReasoningTokens).omit("input_audio_tokens", u.InputAudioTokens).
		omit("output_audio_tokens", u.OutputAudioTokens))
}

// UsageFromDict reads usage (total auto-computed per INV-029).
func UsageFromDict(d JSONObject) (Usage, error) {
	var u Usage
	var err error
	fields := []struct {
		key string
		dst **int
	}{
		{"input_tokens", &u.InputTokens}, {"output_tokens", &u.OutputTokens}, {"total_tokens", &u.TotalTokens},
		{"cache_read_tokens", &u.CacheReadTokens}, {"cache_write_tokens", &u.CacheWriteTokens},
		{"reasoning_tokens", &u.ReasoningTokens}, {"input_audio_tokens", &u.InputAudioTokens},
		{"output_audio_tokens", &u.OutputAudioTokens},
	}
	for _, f := range fields {
		if *f.dst, err = optInt(d, f.key); err != nil {
			return u, err
		}
	}
	u = u.Normalize()
	return u, u.Validate()
}

func (u Usage) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return UsageToDict(u) })
}

func (u *Usage) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := UsageFromDict(d)
		if err != nil {
			return err
		}
		*u = v
		return nil
	})
}

// ─── Stream events ───────────────────────────────────────────────────

// StreamEventToDict serializes a stream event.
func StreamEventToDict(e StreamEvent) JSONObject {
	switch x := e.(type) {
	case StreamStartEvent:
		d := dict{"type": "start"}.omit("id", x.ID).omit("model", x.Model)
		if a := adaptationsToJSON(x.Adaptations); a != nil {
			d["adaptations"] = a
		}
		return JSONObject(d)
	case StreamDeltaEvent:
		return JSONObject{"type": "delta", "delta": DeltaToDict(x.Delta)}
	case StreamEndEvent:
		d := dict{"type": "end"}.omit("finish_reason", x.FinishReason)
		if x.Usage != nil {
			d.omit("usage", UsageToDict(*x.Usage))
		}
		d.omit("provider_data", x.ProviderData)
		return JSONObject(d)
	case StreamErrorEvent:
		return JSONObject{"type": "error", "error": ErrorDetailToDict(x.Error)}
	}
	return nil
}

// StreamEventFromDict reads a stream event.
func StreamEventFromDict(d JSONObject) (StreamEvent, error) {
	t, err := reqString(d, "type")
	if err != nil {
		return nil, err
	}
	switch t {
	case "start":
		id, err := optString(d, "id")
		if err != nil {
			return nil, err
		}
		model, err := optString(d, "model")
		if err != nil {
			return nil, err
		}
		adaptations, err := adaptationsFromJSON(d["adaptations"])
		if err != nil {
			return nil, err
		}
		e := StreamStartEvent{ID: id, Model: model, Adaptations: adaptations}
		return e, e.Validate()
	case "delta":
		obj, ok := d["delta"].(map[string]any)
		if !ok {
			if _, present := d["delta"]; !present {
				return nil, keyError("delta")
			}
			return nil, typeErrorf("delta must be a JSON object")
		}
		dl, err := DeltaFromDict(obj)
		if err != nil {
			return nil, err
		}
		return StreamDeltaEvent{Delta: dl}, nil
	case "end":
		finish, err := optString(d, "finish_reason")
		if err != nil {
			return nil, err
		}
		var usage *Usage
		if obj, ok := d["usage"].(map[string]any); ok {
			u, err := UsageFromDict(obj)
			if err != nil {
				return nil, err
			}
			usage = &u
		}
		pd, err := optObject(d, "provider_data")
		if err != nil {
			return nil, err
		}
		e := StreamEndEvent{FinishReason: finish, Usage: usage, ProviderData: pd}
		return e, e.Validate()
	case "error":
		obj, ok := d["error"].(map[string]any)
		if !ok {
			return nil, keyError("error")
		}
		detail, err := ErrorDetailFromDict(obj)
		if err != nil {
			return nil, err
		}
		return StreamErrorEvent{Error: detail}, nil
	}
	return nil, valueErrorf("unsupported stream event type: %s", t)
}

func (e StreamStartEvent) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return StreamEventToDict(e) })
}
func (e StreamDeltaEvent) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return StreamEventToDict(e) })
}
func (e StreamEndEvent) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return StreamEventToDict(e) })
}
func (e StreamErrorEvent) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return StreamEventToDict(e) })
}

// ─── Request / Response ──────────────────────────────────────────────

func systemToJSON(s *SystemPrompt) any {
	if s == nil {
		return nil
	}
	if s.IsText() {
		return s.Text()
	}
	return partsToList(s.Parts())
}

func systemFromJSON(v any) (*SystemPrompt, error) {
	switch x := v.(type) {
	case nil:
		return nil, nil
	case string:
		return System(x), nil
	case []any:
		parts, err := partsFromList(x)
		if err != nil {
			return nil, err
		}
		return SystemParts(parts...), nil
	}
	return nil, typeErrorf("content must be a string, Part, or sequence of Parts")
}

func toolsFromJSON(v any) ([]Tool, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, typeErrorf("tools must be a list")
	}
	out := make([]Tool, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, typeErrorf("tools must contain objects")
		}
		t, err := ToolFromDict(obj)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

// RequestToDict serializes a request.
func RequestToDict(r *Request) JSONObject {
	d := dict{"model": r.Model, "messages": toAnyList(r.Messages, func(m Message) any { return MessageToDict(m) })}
	d.omit("system", systemToJSON(r.System))
	d.omit("tools", toAnyList(r.Tools, func(t Tool) any { return ToolToDict(t) }))
	d.omit("config", ConfigToDict(r.Config))
	return JSONObject(d)
}

// RequestFromDict reads a request.
func RequestFromDict(d JSONObject) (*Request, error) {
	model, err := reqString(d, "model")
	if err != nil {
		return nil, err
	}
	rawMessages, ok := d["messages"]
	if !ok {
		return nil, keyError("messages")
	}
	list, ok := rawMessages.([]any)
	if !ok {
		return nil, typeErrorf("messages must be a list")
	}
	messages := make([]Message, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, typeErrorf("Request.messages must contain Message objects — wrap plain text with Message.user(\"...\").")
		}
		m, err := MessageFromDict(obj)
		if err != nil {
			return nil, err
		}
		messages = append(messages, m)
	}
	system, err := systemFromJSON(d["system"])
	if err != nil {
		return nil, err
	}
	tools, err := toolsFromJSON(d["tools"])
	if err != nil {
		return nil, err
	}
	config := Config{}
	if raw, ok := d["config"]; ok && raw != nil {
		obj, ok := raw.(map[string]any)
		if !ok {
			return nil, typeErrorf("Request.config must be a Config")
		}
		if config, err = ConfigFromDict(obj); err != nil {
			return nil, err
		}
	}
	r := &Request{Model: model, Messages: messages, System: system, Tools: tools, Config: config}
	return r, r.Validate()
}

func (r Request) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return RequestToDict(&r) })
}

func (r *Request) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := RequestFromDict(d)
		if err != nil {
			return err
		}
		*r = *v
		return nil
	})
}

// ResponseToDict serializes a response; provider_data only when asked.
func ResponseToDict(r *Response, includeProviderData bool) JSONObject {
	d := dict{"model": r.Model, "message": MessageToDict(r.Message), "finish_reason": r.FinishReason}
	d.omit("id", r.ID)
	d.omit("usage", UsageToDict(r.Usage))
	d.omit("logprobs", logprobsToJSON(r.Logprobs))
	if r.LogprobsIncomplete {
		d["logprobs_complete"] = false
	}
	if includeProviderData && r.ProviderData != nil {
		d["provider_data"] = r.ProviderData
	}
	if a := adaptationsToJSON(r.Adaptations); a != nil {
		d["adaptations"] = a
	}
	return JSONObject(d)
}

// ResponseFromDict reads a response.
func ResponseFromDict(d JSONObject) (*Response, error) {
	id, err := optString(d, "id")
	if err != nil {
		return nil, err
	}
	model, err := reqString(d, "model")
	if err != nil {
		return nil, err
	}
	msgObj, ok := d["message"].(map[string]any)
	if !ok {
		if _, present := d["message"]; !present {
			return nil, keyError("message")
		}
		return nil, typeErrorf("Response.message must be a Message")
	}
	message, err := MessageFromDict(msgObj)
	if err != nil {
		return nil, err
	}
	finish, err := reqString(d, "finish_reason")
	if err != nil {
		return nil, err
	}
	usage := Usage{}
	if obj, ok := d["usage"].(map[string]any); ok {
		if usage, err = UsageFromDict(obj); err != nil {
			return nil, err
		}
	}
	lps, err := logprobsFromJSON(d["logprobs"])
	if err != nil {
		return nil, err
	}
	pd, err := optObject(d, "provider_data")
	if err != nil {
		return nil, err
	}
	adaptations, err := adaptationsFromJSON(d["adaptations"])
	if err != nil {
		return nil, err
	}
	complete, err := optBool(d, "logprobs_complete")
	if err != nil {
		return nil, typeErrorf("Response.logprobs_complete must be a bool")
	}
	r := &Response{ID: id, Model: model, Message: message, FinishReason: finish, Usage: usage, Logprobs: lps, ProviderData: pd,
		Adaptations: adaptations, LogprobsIncomplete: complete != nil && !*complete}
	return r, r.Validate()
}

func (r Response) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ResponseToDict(&r, false) })
}

func (r *Response) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := ResponseFromDict(d)
		if err != nil {
			return err
		}
		*r = *v
		return nil
	})
}

// ─── Batch ───────────────────────────────────────────────────────────

// BatchRequestToDict serializes a batch request.
func BatchRequestToDict(b BatchRequest) JSONObject {
	return JSONObject(dict{"requests": toAnyList(b.Requests, func(r *Request) any { return RequestToDict(r) })}.
		omit("model", b.EffectiveModel()).omit("label", b.Label).omit("extensions", b.Extensions))
}

// BatchRequestFromDict reads a batch request.
func BatchRequestFromDict(d JSONObject) (BatchRequest, error) {
	model, err := optString(d, "model")
	if err != nil {
		return BatchRequest{}, err
	}
	var requests []*Request
	list, _ := d["requests"].([]any)
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return BatchRequest{}, typeErrorf("BatchRequest.requests must contain Request objects")
		}
		r, err := RequestFromDict(obj)
		if err != nil {
			return BatchRequest{}, err
		}
		requests = append(requests, r)
	}
	label, err := optString(d, "label")
	if err != nil {
		return BatchRequest{}, err
	}
	ext, err := optObject(d, "extensions")
	if err != nil {
		return BatchRequest{}, err
	}
	if ext, err = normalizeExtensions(ext); err != nil {
		return BatchRequest{}, err
	}
	b := BatchRequest{Model: model, Requests: requests, Label: label, Extensions: ext}
	if err := b.Validate(); err != nil {
		return BatchRequest{}, err
	}
	b.Model = b.EffectiveModel()
	return b, nil
}

// BatchJobToDict serializes a batch job snapshot.
func BatchJobToDict(j BatchJobInfo) JSONObject {
	return JSONObject(dict{"id": j.ID, "status": j.Status}.omit("label", j.Label).omit("created_at", j.CreatedAt).omit("provider_data", j.ProviderData))
}

// BatchJobFromDict reads a batch job snapshot.
func BatchJobFromDict(d JSONObject) (BatchJobInfo, error) {
	id, err := reqString(d, "id")
	if err != nil {
		return BatchJobInfo{}, err
	}
	status, err := reqString(d, "status")
	if err != nil {
		return BatchJobInfo{}, err
	}
	label, err := optString(d, "label")
	if err != nil {
		return BatchJobInfo{}, err
	}
	created, err := optString(d, "created_at")
	if err != nil {
		return BatchJobInfo{}, err
	}
	pd, err := optObject(d, "provider_data")
	if err != nil {
		return BatchJobInfo{}, err
	}
	j := BatchJobInfo{ID: id, Status: status, Label: label, CreatedAt: created, ProviderData: pd}
	return j, j.Validate()
}

// BatchEntryToDict serializes a batch entry (its response WITH provider_data).
func BatchEntryToDict(e BatchEntry) JSONObject {
	d := dict{"index": e.Index, "outcome": e.Outcome}
	if e.Response != nil {
		d["response"] = ResponseToDict(e.Response, true)
	}
	if e.Error != nil {
		d["error"] = ErrorDetailToDict(*e.Error)
	}
	return JSONObject(d)
}

// BatchEntryFromDict reads a batch entry.
func BatchEntryFromDict(d JSONObject) (BatchEntry, error) {
	v, ok := d["index"]
	if !ok {
		return BatchEntry{}, keyError("index")
	}
	index, err := jsonInt(v, "index")
	if err != nil {
		return BatchEntry{}, valueErrorf("BatchEntry.index must be a non-negative int")
	}
	outcome, err := reqString(d, "outcome")
	if err != nil {
		return BatchEntry{}, err
	}
	e := BatchEntry{Index: index, Outcome: outcome}
	if obj, ok := d["response"].(map[string]any); ok {
		r, err := ResponseFromDict(obj)
		if err != nil {
			return BatchEntry{}, err
		}
		e.Response = r
	}
	if obj, ok := d["error"].(map[string]any); ok {
		detail, err := ErrorDetailFromDict(obj)
		if err != nil {
			return BatchEntry{}, err
		}
		e.Error = &detail
	}
	return e, e.Validate()
}

// ─── Files ───────────────────────────────────────────────────────────

// FileUploadRequestToDict serializes an upload request (bytes as base64).
func FileUploadRequestToDict(r FileUploadRequest) JSONObject {
	d := dict{"filename": r.Filename, "media_type": r.EffectiveMediaType()}
	if r.Bytes != nil {
		d["bytes_data"] = base64.StdEncoding.EncodeToString(r.Bytes)
	}
	d.omit("extensions", r.Extensions).omit("path", r.Path)
	return JSONObject(d)
}

// FileUploadRequestFromDict reads an upload request.
func FileUploadRequestFromDict(d JSONObject) (FileUploadRequest, error) {
	filename, err := reqString(d, "filename")
	if err != nil {
		return FileUploadRequest{}, err
	}
	var data []byte
	if raw, ok := d["bytes_data"].(string); ok {
		decoded, err := base64.StdEncoding.DecodeString(raw)
		if err != nil {
			return FileUploadRequest{}, valueErrorf("bytes_data must be base64")
		}
		data = decoded
		if data == nil {
			data = []byte{}
		}
	}
	mediaType, err := optString(d, "media_type")
	if err != nil {
		return FileUploadRequest{}, err
	}
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	ext, err := optObject(d, "extensions")
	if err != nil {
		return FileUploadRequest{}, err
	}
	if ext, err = normalizeExtensions(ext); err != nil {
		return FileUploadRequest{}, err
	}
	path, err := optString(d, "path")
	if err != nil {
		return FileUploadRequest{}, err
	}
	r := FileUploadRequest{Filename: filename, Bytes: data, MediaType: mediaType, Extensions: ext, Path: path}
	return r, r.Validate()
}

// FileInfoToDict serializes a file snapshot (readiness always; downloadable=false is data).
func FileInfoToDict(f FileInfo) JSONObject {
	return JSONObject(dict{"id": f.ID, "readiness": f.EffectiveReadiness()}.
		omit("filename", f.Filename).omit("media_type", f.MediaType).omit("size_bytes", f.SizeBytes).
		omit("created_at", f.CreatedAt).omit("expires_at", f.ExpiresAt).omit("downloadable", f.Downloadable).
		omit("provider_data", f.ProviderData))
}

// FileInfoFromDict reads a file snapshot.
func FileInfoFromDict(d JSONObject) (FileInfo, error) {
	var f FileInfo
	var err error
	if f.ID, err = reqString(d, "id"); err != nil {
		return f, err
	}
	if f.Filename, err = optString(d, "filename"); err != nil {
		return f, err
	}
	if f.MediaType, err = optString(d, "media_type"); err != nil {
		return f, err
	}
	if f.SizeBytes, err = optInt(d, "size_bytes"); err != nil {
		return f, err
	}
	if f.CreatedAt, err = optString(d, "created_at"); err != nil {
		return f, err
	}
	if f.ExpiresAt, err = optString(d, "expires_at"); err != nil {
		return f, err
	}
	if f.Readiness, err = optString(d, "readiness"); err != nil {
		return f, err
	}
	if f.Readiness == "" {
		f.Readiness = "ready"
	}
	if f.Downloadable, err = optBool(d, "downloadable"); err != nil {
		return f, typeErrorf("FileInfo.downloadable must be a bool or None")
	}
	if f.ProviderData, err = optObject(d, "provider_data"); err != nil {
		return f, err
	}
	return f, f.Validate()
}

// FilePageToDict serializes a page.
func FilePageToDict(p FilePage) JSONObject {
	return JSONObject(dict{}.omit("items", toAnyList(p.Items, func(f FileInfo) any { return FileInfoToDict(f) })).omit("next_cursor", p.NextCursor))
}

// FilePageFromDict reads a page.
func FilePageFromDict(d JSONObject) (FilePage, error) {
	var p FilePage
	list, _ := d["items"].([]any)
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return p, typeErrorf("FilePage.items must contain FileInfo objects")
		}
		f, err := FileInfoFromDict(obj)
		if err != nil {
			return p, err
		}
		p.Items = append(p.Items, f)
	}
	var err error
	if p.NextCursor, err = optString(d, "next_cursor"); err != nil {
		return p, err
	}
	return p, p.Validate()
}

// ─── Cache resources ─────────────────────────────────────────────────

// CacheInfoToDict serializes a cache snapshot.
func CacheInfoToDict(c CacheInfo) JSONObject {
	return JSONObject(dict{"id": c.ID, "model": c.Model}.omit("tokens", c.Tokens).omit("created_at", c.CreatedAt).
		omit("expires_at", c.ExpiresAt).omit("label", c.Label).omit("provider_data", c.ProviderData))
}

// CacheInfoFromDict reads a cache snapshot.
func CacheInfoFromDict(d JSONObject) (CacheInfo, error) {
	var c CacheInfo
	var err error
	if c.ID, err = reqString(d, "id"); err != nil {
		return c, err
	}
	if c.Model, err = reqString(d, "model"); err != nil {
		return c, err
	}
	if c.Tokens, err = optInt(d, "tokens"); err != nil {
		return c, err
	}
	if c.CreatedAt, err = optString(d, "created_at"); err != nil {
		return c, err
	}
	if c.ExpiresAt, err = optString(d, "expires_at"); err != nil {
		return c, err
	}
	if c.Label, err = optString(d, "label"); err != nil {
		return c, err
	}
	if c.ProviderData, err = optObject(d, "provider_data"); err != nil {
		return c, err
	}
	return c, c.Validate()
}

// CachePageToDict serializes a cache page.
func CachePageToDict(p CachePage) JSONObject {
	return JSONObject(dict{}.omit("items", toAnyList(p.Items, func(c CacheInfo) any { return CacheInfoToDict(c) })).omit("next_cursor", p.NextCursor))
}

// CachePageFromDict reads a cache page.
func CachePageFromDict(d JSONObject) (CachePage, error) {
	var p CachePage
	list, _ := d["items"].([]any)
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return p, typeErrorf("CachePage.items must contain CacheInfo objects")
		}
		c, err := CacheInfoFromDict(obj)
		if err != nil {
			return p, err
		}
		p.Items = append(p.Items, c)
	}
	var err error
	if p.NextCursor, err = optString(d, "next_cursor"); err != nil {
		return p, err
	}
	return p, p.Validate()
}

// CachedPrefixToDict serializes a cached prefix.
func CachedPrefixToDict(c CachedPrefix) JSONObject {
	d := dict{"prefix": RequestToDict(c.Prefix)}
	if c.Resource != nil {
		d["resource"] = CacheInfoToDict(*c.Resource)
	}
	return JSONObject(d)
}

// CachedPrefixFromDict reads a cached prefix.
func CachedPrefixFromDict(d JSONObject) (CachedPrefix, error) {
	obj, ok := d["prefix"].(map[string]any)
	if !ok {
		if _, present := d["prefix"]; !present {
			return CachedPrefix{}, keyError("prefix")
		}
		return CachedPrefix{}, typeErrorf("CachedPrefix.prefix must be a Request")
	}
	prefix, err := RequestFromDict(obj)
	if err != nil {
		return CachedPrefix{}, err
	}
	c := CachedPrefix{Prefix: prefix}
	if res, ok := d["resource"].(map[string]any); ok {
		info, err := CacheInfoFromDict(res)
		if err != nil {
			return CachedPrefix{}, err
		}
		c.Resource = &info
	}
	return c, c.Validate()
}

// ─── Generation ──────────────────────────────────────────────────────

func imagePartsToJSON(images []ImagePart) []any {
	if len(images) == 0 {
		return nil
	}
	return toAnyList(images, func(i ImagePart) any { return PartToDict(i) })
}

func imagePartsFromJSON(v any) ([]ImagePart, error) {
	list, _ := v.([]any)
	out := make([]ImagePart, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, typeErrorf("images must contain ImagePart objects")
		}
		p, err := PartFromDict(obj)
		if err != nil {
			return nil, err
		}
		img, ok := p.(ImagePart)
		if !ok {
			return nil, typeErrorf("images must contain ImagePart objects")
		}
		out = append(out, img)
	}
	return out, nil
}

// ImageGenerationRequestToDict serializes an image generation request.
func ImageGenerationRequestToDict(r ImageGenerationRequest) JSONObject {
	return JSONObject(dict{"model": r.Model, "prompt": r.Prompt}.omit("size", r.Size).omit("images", imagePartsToJSON(r.Images)).omit("extensions", r.Extensions))
}

// ImageGenerationRequestFromDict reads an image generation request.
func ImageGenerationRequestFromDict(d JSONObject) (ImageGenerationRequest, error) {
	var r ImageGenerationRequest
	var err error
	if r.Model, err = reqString(d, "model"); err != nil {
		return r, err
	}
	if r.Prompt, err = reqString(d, "prompt"); err != nil {
		return r, err
	}
	if r.Size, err = optString(d, "size"); err != nil {
		return r, err
	}
	if r.Images, err = imagePartsFromJSON(d["images"]); err != nil {
		return r, err
	}
	if r.Extensions, err = optObject(d, "extensions"); err != nil {
		return r, err
	}
	if r.Extensions, err = normalizeExtensions(r.Extensions); err != nil {
		return r, err
	}
	return r, r.Validate()
}

// ImageGenerationResponseToDict serializes an image generation response.
func ImageGenerationResponseToDict(r ImageGenerationResponse) JSONObject {
	return JSONObject(dict{"images": imagePartsToJSON(r.Images)}.omit("text", r.Text).omit("id", r.ID).omit("model", r.Model).
		omit("usage", UsageToDict(r.Usage)).omit("provider_data", r.ProviderData))
}

// ImageGenerationResponseFromDict reads an image generation response.
func ImageGenerationResponseFromDict(d JSONObject) (ImageGenerationResponse, error) {
	var r ImageGenerationResponse
	var err error
	if r.Images, err = imagePartsFromJSON(d["images"]); err != nil {
		return r, err
	}
	if r.Text, err = optString(d, "text"); err != nil {
		return r, err
	}
	if r.ID, err = optString(d, "id"); err != nil {
		return r, err
	}
	if r.Model, err = optString(d, "model"); err != nil {
		return r, err
	}
	if obj, ok := d["usage"].(map[string]any); ok {
		if r.Usage, err = UsageFromDict(obj); err != nil {
			return r, err
		}
	}
	if r.ProviderData, err = optObject(d, "provider_data"); err != nil {
		return r, err
	}
	return r, r.Validate()
}

// SpeechGenerationRequestToDict serializes a speech request.
func SpeechGenerationRequestToDict(r SpeechGenerationRequest) JSONObject {
	return JSONObject(dict{"model": r.Model, "prompt": r.Prompt}.omit("voice", r.Voice).omit("format", r.Format).omit("extensions", r.Extensions))
}

// SpeechGenerationRequestFromDict reads a speech request.
func SpeechGenerationRequestFromDict(d JSONObject) (SpeechGenerationRequest, error) {
	var r SpeechGenerationRequest
	var err error
	if r.Model, err = reqString(d, "model"); err != nil {
		return r, err
	}
	if r.Prompt, err = reqString(d, "prompt"); err != nil {
		return r, err
	}
	if r.Voice, err = optString(d, "voice"); err != nil {
		return r, err
	}
	if r.Format, err = optString(d, "format"); err != nil {
		return r, err
	}
	if r.Extensions, err = optObject(d, "extensions"); err != nil {
		return r, err
	}
	if r.Extensions, err = normalizeExtensions(r.Extensions); err != nil {
		return r, err
	}
	return r, r.Validate()
}

// SpeechGenerationResponseToDict serializes a speech response.
func SpeechGenerationResponseToDict(r SpeechGenerationResponse) JSONObject {
	return JSONObject(dict{"audio": PartToDict(r.Audio)}.omit("id", r.ID).omit("model", r.Model).
		omit("usage", UsageToDict(r.Usage)).omit("provider_data", r.ProviderData))
}

// SpeechGenerationResponseFromDict reads a speech response.
func SpeechGenerationResponseFromDict(d JSONObject) (SpeechGenerationResponse, error) {
	var r SpeechGenerationResponse
	obj, ok := d["audio"].(map[string]any)
	if !ok {
		return r, keyError("audio")
	}
	p, err := PartFromDict(obj)
	if err != nil {
		return r, err
	}
	audio, ok := p.(AudioPart)
	if !ok {
		return r, typeErrorf("audio must be an AudioPart")
	}
	r.Audio = audio
	if r.ID, err = optString(d, "id"); err != nil {
		return r, err
	}
	if r.Model, err = optString(d, "model"); err != nil {
		return r, err
	}
	if uobj, ok := d["usage"].(map[string]any); ok {
		if r.Usage, err = UsageFromDict(uobj); err != nil {
			return r, err
		}
	}
	if r.ProviderData, err = optObject(d, "provider_data"); err != nil {
		return r, err
	}
	return r, r.Validate()
}

// VideoGenerationRequestToDict serializes a video request.
func VideoGenerationRequestToDict(r VideoGenerationRequest) JSONObject {
	return JSONObject(dict{"model": r.Model, "prompt": r.Prompt}.omit("seconds", r.Seconds).omit("images", imagePartsToJSON(r.Images)).omit("extensions", r.Extensions))
}

// VideoGenerationRequestFromDict reads a video request.
func VideoGenerationRequestFromDict(d JSONObject) (VideoGenerationRequest, error) {
	var r VideoGenerationRequest
	var err error
	if r.Model, err = reqString(d, "model"); err != nil {
		return r, err
	}
	if r.Prompt, err = reqString(d, "prompt"); err != nil {
		return r, err
	}
	if v, ok := d["seconds"]; ok && v != nil {
		s, err := jsonInt(v, "seconds")
		if err != nil {
			return r, valueErrorf("VideoGenerationRequest.seconds must be a positive int")
		}
		r.Seconds = &s
	}
	if r.Images, err = imagePartsFromJSON(d["images"]); err != nil {
		return r, err
	}
	if r.Extensions, err = optObject(d, "extensions"); err != nil {
		return r, err
	}
	if r.Extensions, err = normalizeExtensions(r.Extensions); err != nil {
		return r, err
	}
	return r, r.Validate()
}

// VideoJobToDict serializes a video job snapshot.
func VideoJobToDict(j VideoJobInfo) JSONObject {
	return JSONObject(dict{"id": j.ID, "status": j.Status}.omit("progress", j.Progress).omit("created_at", j.CreatedAt).
		omit("model", j.Model).omit("provider_data", j.ProviderData))
}

// VideoJobFromDict reads a video job snapshot.
func VideoJobFromDict(d JSONObject) (VideoJobInfo, error) {
	var j VideoJobInfo
	var err error
	if j.ID, err = reqString(d, "id"); err != nil {
		return j, err
	}
	if j.Status, err = reqString(d, "status"); err != nil {
		return j, err
	}
	if v, ok := d["progress"]; ok && v != nil {
		p, err := jsonInt(v, "progress")
		if err != nil {
			return j, valueErrorf("VideoJobInfo.progress must be an int percentage 0-100")
		}
		j.Progress = &p
	}
	if j.CreatedAt, err = optString(d, "created_at"); err != nil {
		return j, err
	}
	if j.Model, err = optString(d, "model"); err != nil {
		return j, err
	}
	if j.ProviderData, err = optObject(d, "provider_data"); err != nil {
		return j, err
	}
	return j, j.Validate()
}

// ─── ModelInfo ───────────────────────────────────────────────────────

func pricingToDict(p InferencePricing) JSONObject {
	return JSONObject(dict{"currency": p.EffectiveCurrency()}.
		omit("input_per_million", p.InputPerMillion).omit("output_per_million", p.OutputPerMillion).
		omit("cache_read_per_million", p.CacheReadPerMillion).omit("cache_write_per_million", p.CacheWritePerMillion).
		omit("dimensions", p.Dimensions))
}

func pricingFromDict(d JSONObject) (InferencePricing, error) {
	var p InferencePricing
	var err error
	if p.InputPerMillion, err = optFloat(d, "input_per_million"); err != nil {
		return p, err
	}
	if p.OutputPerMillion, err = optFloat(d, "output_per_million"); err != nil {
		return p, err
	}
	if p.CacheReadPerMillion, err = optFloat(d, "cache_read_per_million"); err != nil {
		return p, err
	}
	if p.CacheWritePerMillion, err = optFloat(d, "cache_write_per_million"); err != nil {
		return p, err
	}
	if p.Currency, err = optString(d, "currency"); err != nil {
		return p, err
	}
	if p.Currency == "" {
		p.Currency = "USD"
	}
	if p.Dimensions, err = optObject(d, "dimensions"); err != nil {
		return p, err
	}
	return p, p.Validate()
}

func inferenceToDict(i InferenceModelInfo) JSONObject {
	d := dict{}.omit("input_modalities", i.effectiveInput()).omit("output_modalities", i.effectiveOutput()).
		omit("context_window", i.ContextWindow).omit("max_output_tokens", i.MaxOutputTokens)
	if i.SupportsReasoning {
		d["supports_reasoning"] = true
	}
	d.omit("reasoning_efforts", i.ReasoningEfforts)
	if i.Pricing != nil {
		d.omit("pricing", pricingToDict(*i.Pricing))
	}
	d.omit("extensions", i.Extensions)
	return JSONObject(d)
}

func (i InferenceModelInfo) effectiveInput() []string {
	if i.InputModalities == nil {
		return []string{"text"}
	}
	return i.InputModalities
}

func (i InferenceModelInfo) effectiveOutput() []string {
	if i.OutputModalities == nil {
		return []string{"text"}
	}
	return i.OutputModalities
}

func inferenceFromDict(d JSONObject) (InferenceModelInfo, error) {
	var i InferenceModelInfo
	var err error
	i.InputModalities = []string{"text"}
	i.OutputModalities = []string{"text"}
	if v, ok := d["input_modalities"]; ok && v != nil {
		if i.InputModalities, err = stringList(v, "input_modalities"); err != nil {
			return i, err
		}
	}
	if v, ok := d["output_modalities"]; ok && v != nil {
		if i.OutputModalities, err = stringList(v, "output_modalities"); err != nil {
			return i, err
		}
	}
	if i.ContextWindow, err = optInt(d, "context_window"); err != nil {
		return i, err
	}
	if i.MaxOutputTokens, err = optInt(d, "max_output_tokens"); err != nil {
		return i, err
	}
	if v, ok := d["supports_reasoning"].(bool); ok {
		i.SupportsReasoning = v
	}
	if i.ReasoningEfforts, err = stringList(d["reasoning_efforts"], "reasoning_efforts"); err != nil {
		return i, err
	}
	if obj, ok := d["pricing"].(map[string]any); ok {
		p, err := pricingFromDict(obj)
		if err != nil {
			return i, err
		}
		i.Pricing = &p
	}
	if i.Extensions, err = optObject(d, "extensions"); err != nil {
		return i, err
	}
	return i, i.Validate()
}

func originToDict(o ModelOrigin) JSONObject {
	return JSONObject(dict{"type": o.EffectiveType()}.omit("id", o.ID).omit("base_model", o.BaseModel).omit("provider_data", o.ProviderData))
}

func originFromDict(d JSONObject) (ModelOrigin, error) {
	var o ModelOrigin
	var err error
	if o.Type, err = optString(d, "type"); err != nil {
		return o, err
	}
	if o.Type == "" {
		o.Type = "provider"
	}
	if o.ID, err = optString(d, "id"); err != nil {
		return o, err
	}
	if o.BaseModel, err = optString(d, "base_model"); err != nil {
		return o, err
	}
	if o.ProviderData, err = optObject(d, "provider_data"); err != nil {
		return o, err
	}
	return o, nil
}

// ModelInfoToDict serializes model metadata (a default origin is dropped).
func ModelInfoToDict(m ModelInfo) JSONObject {
	origin := originToDict(m.Origin)
	if len(origin) == 1 && origin["type"] == "provider" {
		origin = JSONObject{}
	}
	d := dict{"id": m.ID, "provider": m.Provider, "api_family": m.APIFamily}.omit("aliases", m.Aliases).omit("origin", origin)
	if m.Inference != nil {
		d.omit("inference", inferenceToDict(*m.Inference))
	}
	d.omit("extensions", m.Extensions)
	return JSONObject(d)
}

// ModelInfoFromDict reads model metadata.
func ModelInfoFromDict(d JSONObject) (ModelInfo, error) {
	var m ModelInfo
	var err error
	if m.ID, err = reqString(d, "id"); err != nil {
		return m, err
	}
	if m.Provider, err = reqString(d, "provider"); err != nil {
		return m, err
	}
	if m.APIFamily, err = reqString(d, "api_family"); err != nil {
		return m, err
	}
	if m.Aliases, err = stringList(d["aliases"], "aliases"); err != nil {
		return m, err
	}
	m.Origin = ModelOrigin{Type: "provider"}
	if obj, ok := d["origin"].(map[string]any); ok {
		if m.Origin, err = originFromDict(obj); err != nil {
			return m, err
		}
	}
	if obj, ok := d["inference"].(map[string]any); ok {
		inf, err := inferenceFromDict(obj)
		if err != nil {
			return m, err
		}
		m.Inference = &inf
	}
	if m.Extensions, err = optObject(d, "extensions"); err != nil {
		return m, err
	}
	return m, m.Validate()
}

func (m ModelInfo) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return ModelInfoToDict(m) })
}

func (m *ModelInfo) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := ModelInfoFromDict(d)
		if err != nil {
			return err
		}
		*m = v
		return nil
	})
}

// ─── AudioFormat / Live ──────────────────────────────────────────────

// AudioFormatToDict serializes an audio format.
func AudioFormatToDict(a AudioFormat) JSONObject {
	return JSONObject{"encoding": a.Encoding, "sample_rate": a.SampleRate, "channels": a.EffectiveChannels()}
}

// AudioFormatFromDict reads an audio format (channels defaults to 1).
func AudioFormatFromDict(d JSONObject) (AudioFormat, error) {
	encoding, err := reqString(d, "encoding")
	if err != nil {
		return AudioFormat{}, err
	}
	v, ok := d["sample_rate"]
	if !ok {
		return AudioFormat{}, keyError("sample_rate")
	}
	rate, err := jsonInt(v, "sample_rate")
	if err != nil {
		return AudioFormat{}, err
	}
	channels := 1
	if c, err := optInt(d, "channels"); err != nil {
		return AudioFormat{}, err
	} else if c != nil {
		channels = *c
	}
	a := AudioFormat{Encoding: encoding, SampleRate: rate, Channels: channels}
	return a, a.Validate()
}

// LiveConfigToDict serializes a live config.
func LiveConfigToDict(c LiveConfig) JSONObject {
	d := dict{"model": c.Model}.omit("system", systemToJSON(c.System)).
		omit("tools", toAnyList(c.Tools, func(t Tool) any { return ToolToDict(t) })).omit("voice", c.Voice)
	if c.InputFormat != nil {
		d["input_format"] = AudioFormatToDict(*c.InputFormat)
	}
	if c.OutputFormat != nil {
		d["output_format"] = AudioFormatToDict(*c.OutputFormat)
	}
	d.omit("extensions", c.Extensions)
	return JSONObject(d)
}

// LiveConfigFromDict reads a live config.
func LiveConfigFromDict(d JSONObject) (LiveConfig, error) {
	var c LiveConfig
	var err error
	if c.Model, err = reqString(d, "model"); err != nil {
		return c, err
	}
	if c.System, err = systemFromJSON(d["system"]); err != nil {
		return c, err
	}
	if c.Tools, err = toolsFromJSON(d["tools"]); err != nil {
		return c, err
	}
	if c.Voice, err = optString(d, "voice"); err != nil {
		return c, err
	}
	if obj, ok := d["input_format"].(map[string]any); ok {
		f, err := AudioFormatFromDict(obj)
		if err != nil {
			return c, err
		}
		c.InputFormat = &f
	}
	if obj, ok := d["output_format"].(map[string]any); ok {
		f, err := AudioFormatFromDict(obj)
		if err != nil {
			return c, err
		}
		c.OutputFormat = &f
	}
	if c.Extensions, err = optObject(d, "extensions"); err != nil {
		return c, err
	}
	if c.Extensions, err = normalizeExtensions(c.Extensions); err != nil {
		return c, err
	}
	return c, c.Validate()
}

// LiveClientEventToDict serializes a client event (no cleaning: every field verbatim).
func LiveClientEventToDict(e LiveClientEvent) JSONObject {
	switch x := e.(type) {
	case LiveClientTurnEvent:
		return JSONObject{"type": "turn", "parts": partsToList(x.Parts), "turn_complete": x.TurnComplete}
	case LiveClientAudioEvent:
		return JSONObject{"type": "audio", "data": x.Data, "media_type": x.EffectiveMediaType()}
	case LiveClientImageEvent:
		return JSONObject{"type": "image", "data": x.Data, "media_type": x.EffectiveMediaType()}
	case LiveClientTextEvent:
		return JSONObject{"type": "text", "text": x.Text}
	case LiveClientToolResultEvent:
		return JSONObject{"type": "tool_result", "id": x.ID, "content": partsToList(x.Content)}
	case LiveClientInterruptEvent:
		return JSONObject{"type": "interrupt"}
	case LiveClientEndAudioEvent:
		return JSONObject{"type": "end_audio"}
	}
	return nil
}

// LiveClientEventFromDict reads a client event.
func LiveClientEventFromDict(d JSONObject) (LiveClientEvent, error) {
	t, err := reqString(d, "type")
	if err != nil {
		return nil, err
	}
	var e LiveClientEvent
	switch t {
	case "turn":
		parts, err := partsFromList(d["parts"])
		if err != nil {
			return nil, err
		}
		complete := true
		if v, ok := d["turn_complete"]; ok && v != nil {
			b, ok := v.(bool)
			if !ok {
				return nil, typeErrorf("LiveClientTurnEvent.turn_complete must be a bool")
			}
			complete = b
		}
		e = LiveClientTurnEvent{Parts: parts, TurnComplete: complete}
	case "audio", "image":
		data, err := reqString(d, "data")
		if err != nil {
			return nil, err
		}
		mediaType, err := optString(d, "media_type")
		if err != nil {
			return nil, err
		}
		if t == "audio" {
			e = LiveClientAudioEvent{Data: data, MediaType: mediaType}
		} else {
			e = LiveClientImageEvent{Data: data, MediaType: mediaType}
		}
	case "text":
		text, err := optString(d, "text")
		if err != nil {
			return nil, err
		}
		e = LiveClientTextEvent{Text: text}
	case "tool_result":
		id, err := reqString(d, "id")
		if err != nil {
			return nil, err
		}
		content, err := partsFromList(d["content"])
		if err != nil {
			return nil, err
		}
		e = LiveClientToolResultEvent{ID: id, Content: content}
	case "interrupt":
		e = LiveClientInterruptEvent{}
	case "end_audio":
		e = LiveClientEndAudioEvent{}
	default:
		return nil, valueErrorf("unsupported live client event type: %s", t)
	}
	return e, e.Validate()
}

// LiveServerEventToDict serializes a server event.
func LiveServerEventToDict(e LiveServerEvent) JSONObject {
	switch x := e.(type) {
	case LiveServerAudioEvent:
		return JSONObject(dict{"type": "audio", "data": x.Data}.omit("media_type", x.MediaType))
	case LiveServerTextEvent:
		return JSONObject{"type": "text", "text": x.Text}
	case LiveServerToolCallEvent:
		input := x.Input
		if input == nil {
			input = JSONObject{}
		}
		return JSONObject{"type": "tool_call", "id": x.ID, "name": x.Name, "input": input}
	case LiveServerToolCallDeltaEvent:
		return JSONObject(dict{"type": "tool_call_delta"}.omit("id", x.ID).omit("name", x.Name).omit("input_delta", x.InputDelta))
	case LiveServerInterruptedEvent:
		return JSONObject{"type": "interrupted"}
	case LiveServerTurnEndEvent:
		return JSONObject(dict{"type": "turn_end"}.omit("usage", UsageToDict(x.Usage)))
	case LiveServerUsageEvent:
		return JSONObject(dict{"type": "usage"}.omit("usage", UsageToDict(x.Usage)))
	case LiveServerErrorEvent:
		return JSONObject{"type": "error", "error": ErrorDetailToDict(x.Error)}
	}
	return nil
}

// LiveServerEventFromDict reads a server event.
func LiveServerEventFromDict(d JSONObject) (LiveServerEvent, error) {
	t, err := reqString(d, "type")
	if err != nil {
		return nil, err
	}
	var e LiveServerEvent
	switch t {
	case "audio":
		data, err := reqString(d, "data")
		if err != nil {
			return nil, err
		}
		mediaType, err := optString(d, "media_type")
		if err != nil {
			return nil, err
		}
		e = LiveServerAudioEvent{Data: data, MediaType: mediaType}
	case "text":
		text, err := optString(d, "text")
		if err != nil {
			return nil, err
		}
		e = LiveServerTextEvent{Text: text}
	case "tool_call":
		id, err := reqString(d, "id")
		if err != nil {
			return nil, err
		}
		name, err := reqString(d, "name")
		if err != nil {
			return nil, err
		}
		input, err := optObject(d, "input")
		if err != nil {
			return nil, err
		}
		if input == nil {
			input = JSONObject{}
		}
		e = LiveServerToolCallEvent{ID: id, Name: name, Input: input}
	case "tool_call_delta":
		delta, err := optString(d, "input_delta")
		if err != nil {
			return nil, err
		}
		id, err := optString(d, "id")
		if err != nil {
			return nil, err
		}
		name, err := optString(d, "name")
		if err != nil {
			return nil, err
		}
		e = LiveServerToolCallDeltaEvent{InputDelta: delta, ID: id, Name: name}
	case "interrupted":
		e = LiveServerInterruptedEvent{}
	case "turn_end", "usage":
		usage := Usage{}
		if obj, ok := d["usage"].(map[string]any); ok {
			if usage, err = UsageFromDict(obj); err != nil {
				return nil, err
			}
		}
		if t == "turn_end" {
			e = LiveServerTurnEndEvent{Usage: usage}
		} else {
			e = LiveServerUsageEvent{Usage: usage}
		}
	case "error":
		obj, ok := d["error"].(map[string]any)
		if !ok {
			return nil, keyError("error")
		}
		detail, err := ErrorDetailFromDict(obj)
		if err != nil {
			return nil, err
		}
		e = LiveServerErrorEvent{Error: detail}
	default:
		return nil, valueErrorf("unsupported live server event type: %s", t)
	}
	return e, e.Validate()
}

// jsonRaw returns the compact JSON of a dict (used by adapters for
// arguments strings).
func jsonRaw(v any) string {
	out, err := EncodeJSON(v)
	if err != nil {
		return "{}"
	}
	return string(out)
}

var _ = json.Marshal
