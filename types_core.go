package lm15

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// ─── Pointer helpers ─────────────────────────────────────────────────

// I returns a pointer to an int (Config{MaxTokens: lm15.I(100)}).
func I(v int) *int { return &v }

// F returns a pointer to a float (Config{Temperature: lm15.F(0.2)}).
func F(v float64) *float64 { return &v }

// B returns a pointer to a bool.
func B(v bool) *bool { return &v }

// S returns a pointer to a string.
func S(v string) *string { return &v }

// ─── Messages ────────────────────────────────────────────────────────

// Message is a contribution to a conversation, attributed to a speaker.
type Message struct {
	Role         string
	Parts        []Part
	Continuation []ContinuationState
}

// Validate checks role/part compatibility (INV-022..024).
func (m Message) Validate() error {
	if !inVocab(m.Role, Roles) {
		return valueErrorf("unsupported role: %s", m.Role)
	}
	if len(m.Parts) == 0 {
		return valueErrorf("Message requires at least one part")
	}
	for _, p := range m.Parts {
		if p == nil {
			return typeErrorf("Message.parts must contain Part objects")
		}
		if err := p.Validate(); err != nil {
			return err
		}
	}
	if err := validateContinuation(m.Continuation); err != nil {
		return err
	}
	return validateMessageParts(m.Role, m.Parts)
}

func validateMessageParts(role string, parts []Part) error {
	switch role {
	case RoleTool:
		for _, p := range parts {
			if _, ok := p.(ToolResultPart); !ok {
				return typeErrorf("tool messages may only contain ToolResultPart objects")
			}
		}
		return nil
	case RoleAssistant:
		for _, p := range parts {
			if _, ok := p.(ToolResultPart); ok {
				return typeErrorf("assistant messages cannot contain ToolResultPart objects")
			}
		}
		return nil
	}
	for _, p := range parts {
		if isPromptForbidden(p) {
			return typeErrorf("%s messages cannot contain model/tool protocol parts", role)
		}
	}
	return validateInputDataParts(role, parts)
}

// UserMessage creates a user message from text.
func UserMessage(text string) Message { return Message{Role: RoleUser, Parts: []Part{Text(text)}} }

// UserParts creates a user message from parts.
func UserParts(parts ...Part) Message { return Message{Role: RoleUser, Parts: parts} }

// DeveloperMessage creates a developer (high-authority instruction) message.
func DeveloperMessage(text string) Message {
	return Message{Role: RoleDeveloper, Parts: []Part{Text(text)}}
}

// DeveloperParts creates a developer message from parts.
func DeveloperParts(parts ...Part) Message { return Message{Role: RoleDeveloper, Parts: parts} }

// AssistantMessage creates an assistant message from parts.
func AssistantMessage(parts ...Part) Message { return Message{Role: RoleAssistant, Parts: parts} }

// AssistantText creates an assistant message from text.
func AssistantText(text string) Message {
	return Message{Role: RoleAssistant, Parts: []Part{Text(text)}}
}

// ToolMessage answers one tool call with text output.
func ToolMessage(callID, output string) Message {
	return Message{Role: RoleTool, Parts: []Part{ToolResult(callID, output)}}
}

// ToolMessageParts creates a tool message from explicit results.
func ToolMessageParts(results ...ToolResultPart) Message {
	parts := make([]Part, 0, len(results))
	for _, r := range results {
		parts = append(parts, r)
	}
	return Message{Role: RoleTool, Parts: parts}
}

// PartsOf returns the parts matching the predicate.
func (m Message) PartsOf(match func(Part) bool) []Part {
	var out []Part
	for _, p := range m.Parts {
		if match(p) {
			out = append(out, p)
		}
	}
	return out
}

// First returns the first part matching the predicate.
func (m Message) First(match func(Part) bool) (Part, bool) {
	for _, p := range m.Parts {
		if match(p) {
			return p, true
		}
	}
	return nil, false
}

// Text is the joined text when EVERY part is a TextPart (nil otherwise).
func (m Message) Text() *string {
	var texts []string
	for _, p := range m.Parts {
		t, ok := p.(TextPart)
		if !ok {
			return nil
		}
		texts = append(texts, t.Text)
	}
	s := strings.Join(texts, "\n")
	return &s
}

// ─── System prompt ───────────────────────────────────────────────────

// SystemPrompt is a system instruction: a string, or prompt parts.
type SystemPrompt struct {
	text  string
	parts []Part
}

// System creates a text system prompt.
func System(text string) *SystemPrompt { return &SystemPrompt{text: text} }

// SystemParts creates a system prompt from prompt parts.
func SystemParts(parts ...Part) *SystemPrompt { return &SystemPrompt{parts: parts} }

// IsText reports whether the prompt is the string form.
func (s *SystemPrompt) IsText() bool { return s != nil && s.parts == nil }

// Text returns the string form ("" for the parts form).
func (s *SystemPrompt) Text() string {
	if s == nil {
		return ""
	}
	return s.text
}

// Parts returns the parts form (nil for the string form).
func (s *SystemPrompt) Parts() []Part {
	if s == nil {
		return nil
	}
	return s.parts
}

// Validate checks INV-024 and the non-empty string rule.
func (s *SystemPrompt) Validate() error {
	if s == nil {
		return nil
	}
	if s.parts == nil {
		if s.text == "" {
			return valueErrorf("system cannot be empty")
		}
		return nil
	}
	if len(s.parts) == 0 {
		return valueErrorf("content sequence cannot be empty")
	}
	for _, p := range s.parts {
		if p == nil {
			return typeErrorf("content sequence must contain strings or Part objects")
		}
		if isPromptForbidden(p) {
			return typeErrorf("system parts cannot contain model/tool protocol parts")
		}
		if err := p.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ─── Tools ───────────────────────────────────────────────────────────

// Tool is FunctionTool or BuiltinTool.
type Tool interface {
	ToolName() string
	Validate() error
	sealedTool()
}

// FunctionTool is a function specification sent to the model.
// Parameters is an opaque JSON Schema; nil reads as the default
// {"type": "object", "properties": {}} (INV-033).
type FunctionTool struct {
	Name        string
	Description string
	Parameters  JSONObject
}

// BuiltinTool is a provider-native tool (web_search, code_execution, ...).
type BuiltinTool struct {
	Name   string
	Config JSONObject
}

func (t FunctionTool) ToolName() string { return t.Name }
func (t BuiltinTool) ToolName() string  { return t.Name }
func (FunctionTool) sealedTool()        {}
func (BuiltinTool) sealedTool()         {}

// DefaultParameters is the schema an omitted FunctionTool.Parameters means.
func DefaultParameters() JSONObject {
	return JSONObject{"type": "object", "properties": JSONObject{}}
}

// EffectiveParameters returns Parameters, or the default schema when nil.
func (t FunctionTool) EffectiveParameters() JSONObject {
	if t.Parameters == nil {
		return DefaultParameters()
	}
	return t.Parameters
}

// Validate implements Tool.
func (t FunctionTool) Validate() error {
	if t.Name == "" {
		return valueErrorf("FunctionTool.name cannot be empty")
	}
	if t.Parameters != nil {
		return checkJSONObject(t.Parameters, "parameters", true)
	}
	return nil
}

// Validate implements Tool.
func (t BuiltinTool) Validate() error {
	if t.Name == "" {
		return valueErrorf("BuiltinTool.name cannot be empty")
	}
	return checkJSONObject(t.Config, "config", false)
}

// ─── Configuration ───────────────────────────────────────────────────

// Reasoning is the one dial (MAP-7): Effort required; ThinkingBudget a cap
// on budget wires; Summary the visibility knob.
type Reasoning struct {
	Effort         string
	ThinkingBudget *int
	Summary        string
}

// IsOff reports effort == "off".
func (r Reasoning) IsOff() bool { return r.Effort == EffortOff }

// Validate checks INV-026.
func (r Reasoning) Validate() error {
	if !inVocab(r.Effort, ReasoningEfforts) {
		return valueErrorf("unsupported reasoning effort: %s", r.Effort)
	}
	if r.Summary != "" && !inVocab(r.Summary, ReasoningSummaries) {
		return valueErrorf("unsupported reasoning summary: %s", r.Summary)
	}
	if r.ThinkingBudget != nil && *r.ThinkingBudget <= 0 {
		return valueErrorf("thinking_budget must be > 0")
	}
	if r.IsOff() && (r.ThinkingBudget != nil || r.Summary != "") {
		return valueErrorf("Reasoning(effort='off') cannot specify thinking_budget or summary")
	}
	return nil
}

// CacheConfig names caching INTENTS (MAP-6). Mode "" reads as "auto".
type CacheConfig struct {
	Mode             string
	Retention        string
	Key              string
	PrefixUntilIndex *int
	Prefix           string
	Resource         string
}

// EffectiveMode returns Mode, or "auto" when unset.
func (c CacheConfig) EffectiveMode() string {
	if c.Mode == "" {
		return "auto"
	}
	return c.Mode
}

// Validate checks INV-027.
func (c CacheConfig) Validate() error {
	if !inVocab(c.EffectiveMode(), CacheModes) {
		return valueErrorf("unsupported cache mode: %s", c.Mode)
	}
	if c.Retention != "" && !inVocab(c.Retention, CacheRetentions) {
		return valueErrorf("unsupported cache retention: %s", c.Retention)
	}
	if c.Prefix != "" && !inVocab(c.Prefix, CachePrefixes) {
		return valueErrorf("unsupported cache prefix: %s", c.Prefix)
	}
	if c.EffectiveMode() == "off" && (c.Retention != "" || c.Key != "" || c.Prefix != "" || c.PrefixUntilIndex != nil || c.Resource != "") {
		return valueErrorf("CacheConfig(mode='off') cannot specify retention, key, prefix, prefix_until_index, or resource")
	}
	if c.Prefix != "" && c.PrefixUntilIndex != nil {
		return valueErrorf("CacheConfig cannot specify both prefix and prefix_until_index")
	}
	if c.PrefixUntilIndex != nil && *c.PrefixUntilIndex < 0 {
		return valueErrorf("prefix_until_index must be >= 0")
	}
	return nil
}

// ToolChoice says how the model should use tools. Mode "" reads as "auto".
type ToolChoice struct {
	Mode     string
	Allowed  []string
	Parallel *bool
}

// EffectiveMode returns Mode, or "auto" when unset.
func (t ToolChoice) EffectiveMode() string {
	if t.Mode == "" {
		return "auto"
	}
	return t.Mode
}

// Validate checks INV-028.
func (t ToolChoice) Validate() error {
	if !inVocab(t.EffectiveMode(), ToolChoiceModes) {
		return valueErrorf("unsupported tool choice mode: %s", t.Mode)
	}
	for _, name := range t.Allowed {
		if name == "" {
			return valueErrorf("ToolChoice.allowed must contain non-empty tool names")
		}
	}
	if t.EffectiveMode() == "none" && (len(t.Allowed) > 0 || t.Parallel != nil) {
		return valueErrorf("ToolChoice(mode='none') cannot specify allowed or parallel")
	}
	return nil
}

// ToolChoiceFrom builds a choice from tools or names.
func ToolChoiceFrom(mode string, tools ...Tool) ToolChoice {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.ToolName())
	}
	return ToolChoice{Mode: mode, Allowed: names}
}

// Config holds generation parameters. Universal fields are typed;
// provider-specific settings go in Extensions.
type Config struct {
	MaxTokens      *int
	Temperature    *float64
	TopP           *float64
	TopK           *int
	Stop           []string
	ResponseFormat JSONObject
	ToolChoice     *ToolChoice
	Reasoning      *Reasoning
	Cache          *CacheConfig
	ServiceTier    string
	UserID         string
	Store          *bool
	Logprobs       *int
	// Sampling knobs promoted from extensions 2026-09-14 (MAP-13 audit):
	// OpenAI (both dialects), Gemini and every OpenAI-compatible server
	// carry them; a wire without them (Anthropic) drops and records.
	Seed             *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	// Probabilities asks for a distribution over the keys the json_schema
	// declares (MAP-14). "" = off. "if_available": a wire that cannot
	// measure one records dropped; "required": it refuses before the wire.
	Probabilities string
	Extensions    JSONObject
}

// IsDefault reports whether every field is unset (serializes to {}).
func (c Config) IsDefault() bool {
	return c.MaxTokens == nil && c.Temperature == nil && c.TopP == nil && c.TopK == nil &&
		len(c.Stop) == 0 && len(c.ResponseFormat) == 0 && c.ToolChoice == nil && c.Reasoning == nil &&
		c.Cache == nil && c.ServiceTier == "" && c.UserID == "" && c.Store == nil && c.Logprobs == nil &&
		c.Seed == nil && c.FrequencyPenalty == nil && c.PresencePenalty == nil && c.Probabilities == "" &&
		len(c.Extensions) == 0
}

// Validate checks the field constraints and INV-050.
func (c Config) Validate() error {
	if c.MaxTokens != nil && *c.MaxTokens <= 0 {
		return valueErrorf("max_tokens must be > 0")
	}
	if c.TopK != nil && *c.TopK <= 0 {
		return valueErrorf("top_k must be > 0")
	}
	// The canonical range is 0–2 (OpenAI's and Gemini's); a wire whose
	// ceiling is 1 (Anthropic) clamps and records it (MAP-13), never rescales.
	if c.Temperature != nil && (*c.Temperature < 0 || *c.Temperature > 2) {
		return valueErrorf("temperature must be in [0, 2]")
	}
	if c.FrequencyPenalty != nil && (*c.FrequencyPenalty < -2 || *c.FrequencyPenalty > 2) {
		return valueErrorf("frequency_penalty must be in [-2, 2]")
	}
	if c.PresencePenalty != nil && (*c.PresencePenalty < -2 || *c.PresencePenalty > 2) {
		return valueErrorf("presence_penalty must be in [-2, 2]")
	}
	if c.Probabilities != "" && !inVocab(c.Probabilities, ProbabilityPolicies) {
		return valueErrorf("Config.probabilities must be one of %v, got %q", ProbabilityPolicies, c.Probabilities)
	}
	if c.TopP != nil && (*c.TopP < 0 || *c.TopP > 1) {
		return valueErrorf("top_p must be in [0, 1]")
	}
	for _, s := range c.Stop {
		if s == "" {
			return valueErrorf("stop must contain non-empty strings")
		}
	}
	if c.ToolChoice != nil {
		if err := c.ToolChoice.Validate(); err != nil {
			return err
		}
	}
	if c.Reasoning != nil {
		if err := c.Reasoning.Validate(); err != nil {
			return err
		}
	}
	if c.Cache != nil {
		if err := c.Cache.Validate(); err != nil {
			return err
		}
	}
	if c.Logprobs != nil && *c.Logprobs < 0 {
		return valueErrorf("logprobs must be >= 0")
	}
	if err := checkJSONObject(c.ResponseFormat, "response_format", false); err != nil {
		return err
	}
	if err := validateResponseFormat(c.ResponseFormat); err != nil {
		return err
	}
	if c.Extensions != nil {
		if err := checkJSONObject(c.Extensions, "extensions", false); err != nil {
			return err
		}
	}
	return nil
}

// validateResponseFormat is INV-050: exactly two shapes.
func validateResponseFormat(value JSONObject) error {
	if len(value) == 0 {
		return nil
	}
	fmtType, _ := value["type"].(string)
	if fmtType != "json_object" && fmtType != "json_schema" {
		keys := sortedKeys(value)
		return valueErrorf("response_format must be {'type': 'json_object'} or {'type': 'json_schema', 'schema': {...}, 'name'?: str, 'strict'?: bool}; provider-native shapes go in Config.extensions (got keys %v)", keys)
	}
	allowed := map[string]bool{"type": true}
	if fmtType == "json_schema" {
		allowed["schema"], allowed["name"], allowed["strict"] = true, true, true
	}
	var extra []string
	for k := range value {
		if !allowed[k] {
			extra = append(extra, k)
		}
	}
	if len(extra) > 0 {
		return valueErrorf("response_format %q does not take keys %v; provider-native shapes go in Config.extensions", fmtType, sortStrings(extra))
	}
	if fmtType == "json_schema" {
		if _, ok := value["schema"].(map[string]any); !ok {
			return valueErrorf("response_format json_schema requires a 'schema' object")
		}
		if name, ok := value["name"]; ok {
			if s, isStr := name.(string); !isStr || s == "" {
				return valueErrorf("response_format name must be a non-empty string")
			}
		}
		if strict, ok := value["strict"]; ok {
			if _, isBool := strict.(bool); !isBool {
				return typeErrorf("response_format strict must be a bool")
			}
		}
	}
	return nil
}

// ─── Request ─────────────────────────────────────────────────────────

// Request is a complete request to a foundation model.
type Request struct {
	Model    string
	Messages []Message
	System   *SystemPrompt
	Tools    []Tool
	Config   Config
}

// NewRequest builds and validates a request.
func NewRequest(model string, messages []Message, opts ...RequestOption) (*Request, error) {
	r := &Request{Model: model, Messages: messages}
	for _, o := range opts {
		o(r)
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return r, nil
}

// RequestOption configures NewRequest.
type RequestOption func(*Request)

// WithSystem sets the system prompt.
func WithSystem(text string) RequestOption { return func(r *Request) { r.System = System(text) } }

// WithTools sets the tools.
func WithTools(tools ...Tool) RequestOption { return func(r *Request) { r.Tools = tools } }

// WithConfig sets the config.
func WithConfig(c Config) RequestOption { return func(r *Request) { r.Config = c } }

// Validate checks INV-030 / INV-031 and every nested value.
func (r *Request) Validate() error {
	if r.Model == "" {
		return valueErrorf("model is required")
	}
	if len(r.Messages) == 0 {
		return valueErrorf("at least one message is required")
	}
	for _, m := range r.Messages {
		if err := m.Validate(); err != nil {
			return err
		}
	}
	if err := r.System.Validate(); err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, t := range r.Tools {
		if t == nil {
			return typeErrorf("Request.tools must contain Tool objects")
		}
		if err := t.Validate(); err != nil {
			return err
		}
		if seen[t.ToolName()] {
			return valueErrorf("Request.tools cannot contain duplicate tool names")
		}
		seen[t.ToolName()] = true
	}
	if err := r.Config.Validate(); err != nil {
		return err
	}
	if r.Config.ToolChoice != nil && len(r.Config.ToolChoice.Allowed) > 0 {
		var missing []string
		for _, name := range r.Config.ToolChoice.Allowed {
			if !seen[name] {
				missing = append(missing, name)
			}
		}
		if len(missing) > 0 {
			return valueErrorf("ToolChoice.allowed contains tools not present in Request.tools: %v", sortStrings(missing))
		}
	}
	return nil
}

// WithModel returns a copy with another model string.
func (r *Request) WithModel(model string) *Request {
	out := *r
	out.Model = model
	return &out
}

// ToolByName finds a declared tool.
func (r *Request) ToolByName(name string) Tool {
	for _, t := range r.Tools {
		if t.ToolName() == name {
			return t
		}
	}
	return nil
}

// ─── Usage / logprobs / Response ─────────────────────────────────────

// Usage carries provider-verbatim token counters. nil = not reported,
// distinct from a reported 0 (INV-029).
type Usage struct {
	InputTokens       *int
	OutputTokens      *int
	TotalTokens       *int
	CacheReadTokens   *int
	CacheWriteTokens  *int
	ReasoningTokens   *int
	InputAudioTokens  *int
	OutputAudioTokens *int
}

// Normalize returns the Usage with total_tokens auto-computed when both
// primaries are present and no total was reported (INV-029).
func (u Usage) Normalize() Usage {
	if u.TotalTokens == nil && u.InputTokens != nil && u.OutputTokens != nil {
		total := *u.InputTokens + *u.OutputTokens
		u.TotalTokens = &total
	}
	return u
}

// IsEmpty reports whether nothing was reported (serializes to {}).
func (u Usage) IsEmpty() bool {
	return u.InputTokens == nil && u.OutputTokens == nil && u.TotalTokens == nil && u.CacheReadTokens == nil &&
		u.CacheWriteTokens == nil && u.ReasoningTokens == nil && u.InputAudioTokens == nil && u.OutputAudioTokens == nil
}

// Validate checks every counter is >= 0.
func (u Usage) Validate() error {
	for name, v := range map[string]*int{
		"input_tokens": u.InputTokens, "output_tokens": u.OutputTokens, "total_tokens": u.TotalTokens,
		"cache_read_tokens": u.CacheReadTokens, "cache_write_tokens": u.CacheWriteTokens, "reasoning_tokens": u.ReasoningTokens,
		"input_audio_tokens": u.InputAudioTokens, "output_audio_tokens": u.OutputAudioTokens,
	} {
		if v != nil && *v < 0 {
			return valueErrorf("%s must be >= 0", name)
		}
	}
	return nil
}

// TopLogprob is one scored alternative token at a decoding step.
type TopLogprob struct {
	Token   string
	Logprob float64
	Bytes   []int
	TokenID *int
}

// TokenLogprob is the chosen token at one decoding step with alternatives.
type TokenLogprob struct {
	Token   string
	Logprob float64
	Bytes   []int
	TokenID *int
	Top     []TopLogprob
}

func validateLogprobBytes(b []int) error {
	for _, x := range b {
		if x < 0 {
			return typeErrorf("bytes must contain non-negative ints")
		}
	}
	return nil
}

// Validate implements the constraints.
func (t TopLogprob) Validate() error { return validateLogprobBytes(t.Bytes) }

// Validate implements the constraints.
func (t TokenLogprob) Validate() error {
	if err := validateLogprobBytes(t.Bytes); err != nil {
		return err
	}
	for _, top := range t.Top {
		if err := top.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// Response is what a foundation model returned.
//
// LogprobsIncomplete (the wire's logprobs_complete=false) means local text
// editing (a client-side stop inside a token) left retained text without
// its original scores; remaining scores describe whole original tokens
// only. False (the default, "complete") does not promise the provider
// supplied scores at all.
type Response struct {
	ID           string
	Model        string
	Message      Message
	FinishReason string
	Usage        Usage
	Logprobs     []TokenLogprob // nil = not reported
	ProviderData JSONObject
	// Adaptations is what the wire got that differs from what was asked
	// (MAP-13): a dropped hint, a clamped dial, a client-side stop. Empty
	// when the request went out exactly as written. Data, never printed.
	Adaptations []Adaptation
	// LogprobsIncomplete: see the type comment.
	LogprobsIncomplete bool
}

// LogprobsComplete is the wire's spelling of !LogprobsIncomplete.
func (r *Response) LogprobsComplete() bool { return !r.LogprobsIncomplete }

// Validate checks INV-036.
func (r *Response) Validate() error {
	if r.Model == "" {
		return valueErrorf("Response.model cannot be empty")
	}
	if err := r.Message.Validate(); err != nil {
		return err
	}
	if r.Message.Role != RoleAssistant {
		return valueErrorf("Response.message must have role 'assistant'")
	}
	if !inVocab(r.FinishReason, FinishReasons) {
		return valueErrorf("unsupported finish reason: %s", r.FinishReason)
	}
	if err := r.Usage.Validate(); err != nil {
		return err
	}
	for _, lp := range r.Logprobs {
		if err := lp.Validate(); err != nil {
			return err
		}
	}
	if err := validateAdaptations(r.Adaptations); err != nil {
		return err
	}
	return checkJSONObject(r.ProviderData, "provider_data", false)
}

// ─── Judgments (changes/2026-09-17-judgments.md D12) ────────────────

// DataPart returns the message's first data part, if any.
func (r *Response) DataPart() (DataPart, bool) {
	for _, p := range r.Message.Parts {
		if d, ok := p.(DataPart); ok {
			return d, true
		}
	}
	return DataPart{}, false
}

// Data is the answer of a judgment request: the DataPart's value. Falls
// back to the parsed JSON text of a plain structured-output response so
// Data reads the same on a wire that answered with text; nil when neither.
func (r *Response) Data() any {
	if d, ok := r.DataPart(); ok {
		return d.Value
	}
	return r.JSON()
}

// Probabilities are the per-judgment distributions over the declared keys,
// or nil when none was measured (never a fabricated one).
func (r *Response) Probabilities() map[string]map[string]float64 {
	if d, ok := r.DataPart(); ok {
		return d.Probabilities
	}
	return nil
}

// Method is how the distributions were measured (JudgmentMethod), or "".
func (r *Response) Method() string {
	if d, ok := r.DataPart(); ok {
		return d.Method
	}
	return ""
}

// Expected is Σ p·i over an ordered judgment's levels (Jev's score); ok is
// false when the field has no distribution or its keys are not level indexes.
func (r *Response) Expected(field string) (float64, bool) {
	dist := r.Probabilities()[field]
	if len(dist) == 0 {
		return 0, false
	}
	total := 0.0
	for key, p := range dist {
		i, err := strconv.Atoi(key)
		if err != nil {
			return 0, false
		}
		total += p * float64(i)
	}
	return total, true
}

// Text is the visible answer text: the joined TextParts when the message
// holds only text, citation and thinking parts; nil otherwise.
func (r *Response) Text() *string {
	if t := r.Message.Text(); t != nil {
		return t
	}
	var texts []string
	for _, p := range r.Message.Parts {
		switch x := p.(type) {
		case TextPart:
			texts = append(texts, x.Text)
		case CitationPart, ThinkingPart:
		default:
			return nil
		}
	}
	if len(texts) == 0 {
		return nil
	}
	s := strings.Join(texts, "\n")
	return &s
}

// TextOr returns Text() or the fallback.
func (r *Response) TextOr(fallback string) string {
	if t := r.Text(); t != nil {
		return *t
	}
	return fallback
}

// ToolCalls returns the tool call parts.
func (r *Response) ToolCalls() []ToolCallPart {
	var out []ToolCallPart
	for _, p := range r.Message.Parts {
		if tc, ok := p.(ToolCallPart); ok {
			out = append(out, tc)
		}
	}
	return out
}

// Citations returns the citation parts.
func (r *Response) Citations() []CitationPart {
	var out []CitationPart
	for _, p := range r.Message.Parts {
		if c, ok := p.(CitationPart); ok {
			out = append(out, c)
		}
	}
	return out
}

// ParseJSON parses the response text as exact JSON.
func (r *Response) ParseJSON(out any) error {
	t := r.Text()
	if t == nil {
		types := make([]string, 0, len(r.Message.Parts))
		for _, p := range r.Message.Parts {
			types = append(types, p.Type())
		}
		return valueErrorf("Cannot parse response as JSON: response is not pure text. Parts: %v", types)
	}
	stripped := strings.TrimSpace(*t)
	if err := json.Unmarshal([]byte(stripped), out); err != nil {
		preview := stripped
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		return valueErrorf("Cannot parse response as JSON: %v\nRaw text: %s", err, preview)
	}
	return nil
}

// JSON returns the parsed JSON text, or nil when parsing fails.
func (r *Response) JSON() any {
	var out any
	if err := r.ParseJSON(&out); err != nil {
		return nil
	}
	return out
}

// ToolCallInfo is the callback view of a tool call (no discriminator).
type ToolCallInfo struct {
	ID    string
	Name  string
	Input JSONObject
}

// ToPart converts back to a ToolCallPart.
func (t ToolCallInfo) ToPart() ToolCallPart {
	return ToolCallPart{ID: t.ID, Name: t.Name, Input: t.Input}
}

// ToolCallInfoFrom builds the info from a part.
func ToolCallInfoFrom(p ToolCallPart) ToolCallInfo {
	return ToolCallInfo{ID: p.ID, Name: p.Name, Input: p.Input}
}

func sortStrings(s []string) []string {
	out := append([]string(nil), s...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return sortStrings(keys)
}

func fmtList(items []string) string { return fmt.Sprintf("%v", items) }
