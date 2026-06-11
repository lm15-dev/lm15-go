package lm15

import "encoding/json"

// ─── Tools ───────────────────────────────────────────────────────────

// Tool is the tagged union of FunctionTool and BuiltinTool.
type Tool interface {
	ToolType() string
	ToolName() string
	toMap() jmap
	isTool()
}

// DefaultParameters returns the default FunctionTool JSON-Schema payload.
func DefaultParameters() jmap {
	return jmap{"type": "object", "properties": jmap{}}
}

type FunctionTool struct {
	Name        string  `json:"name"`
	Description *string `json:"description"`
	Parameters  jmap    `json:"parameters"`
}

func (t FunctionTool) ToolType() string { return "function" }
func (t FunctionTool) ToolName() string { return t.Name }
func (t FunctionTool) isTool()          {}
func (t FunctionTool) toMap() jmap {
	m := jmap{"type": "function", "name": t.Name}
	if t.Description != nil && *t.Description != "" {
		m["description"] = *t.Description
	}
	// parameters is required-with-shape (INV-033): always emitted, even {} —
	// an opaque payload, round-tripped verbatim.
	params := t.Parameters
	if params == nil {
		params = DefaultParameters()
	}
	m["parameters"] = params
	return m
}

type BuiltinTool struct {
	Name   string `json:"name"`
	Config jmap   `json:"config"`
}

func (t BuiltinTool) ToolType() string { return "builtin" }
func (t BuiltinTool) ToolName() string { return t.Name }
func (t BuiltinTool) isTool()          {}
func (t BuiltinTool) toMap() jmap {
	m := jmap{"type": "builtin", "name": t.Name}
	putOpt(m, "config", t.Config)
	return m
}

func (t FunctionTool) MarshalJSON() ([]byte, error) { return json.Marshal(t.toMap()) }
func (t BuiltinTool) MarshalJSON() ([]byte, error)  { return json.Marshal(t.toMap()) }

// ToolFromMap dispatches on "type": "builtin" → BuiltinTool; anything else
// (including absent) → FunctionTool (INV-034).
func ToolFromMap(m jmap) (Tool, error) {
	t, _ := optString(m, "type", "")
	name, err := reqString(m, "name")
	if err != nil {
		return nil, err
	}
	if name == "" {
		return nil, valueErr("tool name must be non-empty")
	}
	if t == "builtin" {
		config, err := optObj(m, "config")
		if err != nil {
			return nil, err
		}
		return BuiltinTool{Name: name, Config: config}, nil
	}
	desc, err := optStringPtr(m, "description")
	if err != nil {
		return nil, err
	}
	params, err := optObj(m, "parameters")
	if err != nil {
		return nil, err
	}
	if _, present := m["parameters"]; !present {
		params = DefaultParameters()
	} else if params == nil {
		params = DefaultParameters()
	}
	return FunctionTool{Name: name, Description: desc, Parameters: params}, nil
}

// ─── ToolChoice ──────────────────────────────────────────────────────

type ToolChoice struct {
	Mode     string   `json:"mode"`
	Allowed  []string `json:"allowed"`
	Parallel *bool    `json:"parallel"`
}

func (tc ToolChoice) toMap() jmap {
	m := jmap{"mode": tc.Mode}
	if len(tc.Allowed) > 0 {
		allowed := make([]any, len(tc.Allowed))
		for i, a := range tc.Allowed {
			allowed[i] = a
		}
		m["allowed"] = allowed
	}
	if tc.Parallel != nil {
		m["parallel"] = *tc.Parallel
	}
	return m
}

func toolChoiceFromMap(m jmap) (*ToolChoice, error) {
	mode, err := optString(m, "mode", "auto")
	if err != nil {
		return nil, err
	}
	if !inVocab(ToolChoiceModes, mode) {
		return nil, valueErr("unsupported tool choice mode: %s", mode)
	}
	allowed, err := stringList(m, "allowed")
	if err != nil {
		return nil, err
	}
	for _, a := range allowed {
		if a == "" {
			return nil, valueErr("tool choice allowed names must be non-empty")
		}
	}
	parallel, err := optBoolPtr(m, "parallel")
	if err != nil {
		return nil, err
	}
	if mode == "none" && (len(allowed) > 0 || parallel != nil) {
		return nil, valueErr("tool choice mode 'none' forbids allowed/parallel")
	}
	return &ToolChoice{Mode: mode, Allowed: allowed, Parallel: parallel}, nil
}

// ─── Reasoning ───────────────────────────────────────────────────────

type Reasoning struct {
	Effort         string  `json:"effort"`
	ThinkingBudget *int64  `json:"thinking_budget"`
	TotalBudget    *int64  `json:"total_budget"`
	Summary        *string `json:"summary"`
}

func (r Reasoning) IsOff() bool { return r.Effort == "off" }

func (r Reasoning) toMap() jmap {
	m := jmap{"effort": r.Effort}
	if r.ThinkingBudget != nil {
		m["thinking_budget"] = jsonInt(*r.ThinkingBudget)
	}
	if r.TotalBudget != nil {
		m["total_budget"] = jsonInt(*r.TotalBudget)
	}
	if r.Summary != nil && *r.Summary != "" {
		m["summary"] = *r.Summary
	}
	return m
}

func reasoningFromMap(m jmap) (*Reasoning, error) {
	defaultEffort := "medium"
	if enabled, ok := m["enabled"].(bool); ok && !enabled {
		defaultEffort = "off"
	}
	effort, err := optString(m, "effort", defaultEffort)
	if err != nil {
		return nil, err
	}
	if !inVocab(ReasoningEfforts, effort) {
		return nil, valueErr("unsupported reasoning effort: %s", effort)
	}
	if effort == "off" {
		// Legacy payloads could combine enabled=false with budgets; off
		// reasoning carries no budgets or summary (INV-026).
		return &Reasoning{Effort: "off"}, nil
	}
	thinking, err := optInt(m, "thinking_budget")
	if err != nil {
		return nil, err
	}
	if thinking == nil {
		if thinking, err = optInt(m, "budget"); err != nil {
			return nil, err
		}
	}
	total, err := optInt(m, "total_budget")
	if err != nil {
		return nil, err
	}
	if (thinking != nil && *thinking <= 0) || (total != nil && *total <= 0) {
		return nil, valueErr("reasoning budgets must be positive")
	}
	summary, err := optStringPtr(m, "summary")
	if err != nil {
		return nil, err
	}
	if summary != nil && !inVocab(ReasoningSummaries, *summary) {
		return nil, valueErr("unsupported reasoning summary: %s", *summary)
	}
	return &Reasoning{Effort: effort, ThinkingBudget: thinking, TotalBudget: total, Summary: summary}, nil
}

// ─── CacheConfig ─────────────────────────────────────────────────────

type CacheConfig struct {
	Mode             string  `json:"mode"`
	Retention        *string `json:"retention"`
	Key              *string `json:"key"`
	PrefixUntilIndex *int64  `json:"prefix_until_index"`
}

func (c CacheConfig) toMap() jmap {
	m := jmap{"mode": c.Mode}
	if c.Retention != nil && *c.Retention != "" {
		m["retention"] = *c.Retention
	}
	if c.Key != nil && *c.Key != "" {
		m["key"] = *c.Key
	}
	if c.PrefixUntilIndex != nil {
		m["prefix_until_index"] = jsonInt(*c.PrefixUntilIndex)
	}
	return m
}

func cacheConfigFromMap(m jmap) (*CacheConfig, error) {
	mode, err := optString(m, "mode", "auto")
	if err != nil {
		return nil, err
	}
	if !inVocab(CacheModes, mode) {
		return nil, valueErr("unsupported cache mode: %s", mode)
	}
	retention, err := optStringPtr(m, "retention")
	if err != nil {
		return nil, err
	}
	if retention != nil && !inVocab(CacheRetentions, *retention) {
		return nil, valueErr("unsupported cache retention: %s", *retention)
	}
	key, err := optStringPtr(m, "key")
	if err != nil {
		return nil, err
	}
	if mode == "off" && (retention != nil || key != nil) {
		return nil, valueErr("cache mode 'off' forbids retention/key")
	}
	prefix, err := optInt(m, "prefix_until_index")
	if err != nil {
		return nil, err
	}
	if prefix != nil && *prefix < 0 {
		return nil, valueErr("prefix_until_index must be >= 0")
	}
	return &CacheConfig{Mode: mode, Retention: retention, Key: key, PrefixUntilIndex: prefix}, nil
}

// ─── Config ──────────────────────────────────────────────────────────

type Config struct {
	MaxTokens      *int64       `json:"max_tokens"`
	Temperature    *float64     `json:"temperature"`
	TopP           *float64     `json:"top_p"`
	TopK           *int64       `json:"top_k"`
	Stop           []string     `json:"stop"`
	ResponseFormat jmap         `json:"response_format"`
	ToolChoice     *ToolChoice  `json:"tool_choice"`
	Reasoning      *Reasoning   `json:"reasoning"`
	Cache          *CacheConfig `json:"cache"`
	Extensions     jmap         `json:"extensions"`
}

func (c Config) toMap() jmap {
	m := jmap{}
	if c.MaxTokens != nil {
		m["max_tokens"] = jsonInt(*c.MaxTokens)
	}
	if c.Temperature != nil {
		m["temperature"] = jsonFloat(*c.Temperature)
	}
	if c.TopP != nil {
		m["top_p"] = jsonFloat(*c.TopP)
	}
	if c.TopK != nil {
		m["top_k"] = jsonInt(*c.TopK)
	}
	if len(c.Stop) > 0 {
		stop := make([]any, len(c.Stop))
		for i, s := range c.Stop {
			stop[i] = s
		}
		m["stop"] = stop
	}
	putOpt(m, "response_format", c.ResponseFormat)
	if c.ToolChoice != nil {
		putOpt(m, "tool_choice", c.ToolChoice.toMap())
	}
	if c.Reasoning != nil {
		putOpt(m, "reasoning", c.Reasoning.toMap())
	}
	if c.Cache != nil {
		putOpt(m, "cache", c.Cache.toMap())
	}
	putOpt(m, "extensions", c.Extensions)
	return m
}

// configNest returns config.<key> as an object; a present non-object is
// malformed canonical JSON and is rejected (INV-042).
func configNest(m jmap, key string) (jmap, error) {
	v, ok := m[key]
	if !ok || v == nil {
		return nil, nil
	}
	o, ok := v.(jmap)
	if !ok {
		return nil, typeErr("config.%s must be a JSON object", key)
	}
	return o, nil
}

func configFromMap(m jmap) (*Config, error) {
	maxTokens, err := optInt(m, "max_tokens")
	if err != nil {
		return nil, err
	}
	if maxTokens != nil && *maxTokens <= 0 {
		return nil, valueErr("max_tokens must be positive")
	}
	temperature, err := optFloat(m, "temperature")
	if err != nil {
		return nil, err
	}
	if temperature != nil && *temperature < 0 {
		return nil, valueErr("temperature must be >= 0")
	}
	topP, err := optFloat(m, "top_p")
	if err != nil {
		return nil, err
	}
	if topP != nil && (*topP < 0 || *topP > 1) {
		return nil, valueErr("top_p must be in [0, 1]")
	}
	topK, err := optInt(m, "top_k")
	if err != nil {
		return nil, err
	}
	if topK != nil && *topK <= 0 {
		return nil, valueErr("top_k must be positive")
	}
	stop, err := stringList(m, "stop")
	if err != nil {
		return nil, err
	}
	for _, s := range stop {
		if s == "" {
			return nil, valueErr("stop entries must be non-empty strings")
		}
	}
	responseFormat, err := optObj(m, "response_format")
	if err != nil {
		return nil, err
	}
	var toolChoice *ToolChoice
	if nest, err := configNest(m, "tool_choice"); err != nil {
		return nil, err
	} else if nest != nil {
		if toolChoice, err = toolChoiceFromMap(nest); err != nil {
			return nil, err
		}
	}
	var reasoning *Reasoning
	if nest, err := configNest(m, "reasoning"); err != nil {
		return nil, err
	} else if nest != nil {
		if reasoning, err = reasoningFromMap(nest); err != nil {
			return nil, err
		}
	}
	var cache *CacheConfig
	if nest, err := configNest(m, "cache"); err != nil {
		return nil, err
	} else if nest != nil {
		if cache, err = cacheConfigFromMap(nest); err != nil {
			return nil, err
		}
	}
	extensions, err := optObj(m, "extensions")
	if err != nil {
		return nil, err
	}
	if extensions != nil && len(extensions) == 0 {
		extensions = nil // {} normalized to null (INV-004)
	}
	return &Config{
		MaxTokens: maxTokens, Temperature: temperature, TopP: topP, TopK: topK,
		Stop: stop, ResponseFormat: responseFormat, ToolChoice: toolChoice,
		Reasoning: reasoning, Cache: cache, Extensions: extensions,
	}, nil
}

func (c Config) MarshalJSON() ([]byte, error) { return json.Marshal(c.toMap()) }

func (c *Config) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := configFromMap(m)
	if err != nil {
		return err
	}
	*c = *v
	return nil
}
