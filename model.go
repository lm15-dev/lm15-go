package lm15

import (
	"fmt"
	"mime"
	"os"
	"path/filepath"
)

// HistoryEntry records a request/response pair.
type HistoryEntry struct {
	Request  LMRequest
	Response LMResponse
}

// ModelOpts configures a Model.
type ModelOpts struct {
	LM            *UniversalLM
	Model         string
	System        string
	Tools         []Tool
	OnToolCall    func(ToolCallInfo) interface{}
	Provider      string
	Retries       int
	Cache         bool
	PromptCaching bool
	Temperature   *float64
	MaxTokens     *int
	MaxToolRounds int
}

// Model is a reusable, stateful model object with conversation memory.
type Model struct {
	opts             ModelOpts
	conversation     []Message
	history          []HistoryEntry
	pendingToolCalls []Part
	localCache       map[string]LMResponse // nil if caching disabled
}

// NewModel creates a new Model.
func NewModel(opts ModelOpts) *Model {
	if opts.MaxToolRounds == 0 {
		opts.MaxToolRounds = 8
	}
	var cache map[string]LMResponse
	if opts.Cache {
		cache = make(map[string]LMResponse)
	}
	return &Model{opts: opts, localCache: cache}
}

// History returns the model's history entries.
func (m *Model) History() []HistoryEntry { return m.history }

// ClearHistory resets conversation and history.
func (m *Model) ClearHistory() {
	m.history = nil
	m.conversation = nil
	m.pendingToolCalls = nil
}

// TotalCost returns the cumulative estimated cost across all history, or nil if cost tracking is not enabled.
func (m *Model) TotalCost() *CostBreakdown {
	if len(m.history) == 0 {
		zero := zeroCostBreakdown()
		return &zero
	}

	var costs []CostBreakdown
	for _, entry := range m.history {
		cost, ok := LookupCost(entry.Response.Model, entry.Response.Usage)
		if !ok {
			continue
		}
		costs = append(costs, *cost)
	}
	if len(costs) == 0 {
		return nil
	}
	total := SumCosts(costs)
	return &total
}

// Copy creates a copy of the model with optional overrides.
func (m *Model) Copy(overrides *ModelOpts) *Model {
	o := m.opts
	if overrides != nil {
		if overrides.Model != "" {
			o.Model = overrides.Model
		}
		if overrides.System != "" {
			o.System = overrides.System
		}
		if overrides.Tools != nil {
			o.Tools = overrides.Tools
		}
		if overrides.OnToolCall != nil {
			o.OnToolCall = overrides.OnToolCall
		}
		if overrides.Provider != "" {
			o.Provider = overrides.Provider
		}
		if overrides.Temperature != nil {
			o.Temperature = overrides.Temperature
		}
		if overrides.MaxTokens != nil {
			o.MaxTokens = overrides.MaxTokens
		}
		if overrides.MaxToolRounds != 0 {
			o.MaxToolRounds = overrides.MaxToolRounds
		}
		if overrides.Retries != 0 {
			o.Retries = overrides.Retries
		}
	}
	newModel := NewModel(o)
	// Copy conversation and history
	newModel.conversation = append([]Message{}, m.conversation...)
	newModel.history = append([]HistoryEntry{}, m.history...)
	newModel.pendingToolCalls = append([]Part{}, m.pendingToolCalls...)
	return newModel
}

// Prepare builds an LMRequest without sending it.
func (m *Model) Prepare(prompt string, opts *CallOpts) LMRequest {
	req, _ := m.buildRequest(prompt, opts)
	return req
}

// Call sends a prompt and returns a Result.
func (m *Model) Call(prompt string, opts *CallOpts) *Result {
	if opts == nil {
		opts = &CallOpts{}
	}
	request, callableRegistry := m.buildRequest(prompt, opts)
	resolvedProvider := opts.Provider
	if resolvedProvider == "" {
		resolvedProvider = m.opts.Provider
	}

	startStream := func(req LMRequest) (<-chan StreamEvent, error) {
		if m.localCache != nil {
			key := cacheKey(req, resolvedProvider)
			if cached, ok := m.localCache[key]; ok {
				return responseToEventChan(cached), nil
			}
		}
		return m.opts.LM.Stream(req, resolvedProvider)
	}

	onFinished := func(finalReq LMRequest, resp LMResponse) {
		if m.localCache != nil {
			m.localCache[cacheKey(finalReq, resolvedProvider)] = resp
		}
		m.history = append(m.history, HistoryEntry{Request: finalReq, Response: resp})
		m.pendingToolCalls = resp.ToolCalls()
		m.conversation = append(append([]Message{}, finalReq.Messages...), resp.Message)
	}

	onToolCall := opts.OnToolCall
	if onToolCall == nil {
		onToolCall = m.opts.OnToolCall
	}

	maxRounds := opts.MaxToolRounds
	if maxRounds == 0 {
		maxRounds = m.opts.MaxToolRounds
	}

	return NewResult(ResultOpts{
		Request:          request,
		StartStream:      startStream,
		OnFinished:       onFinished,
		CallableRegistry: callableRegistry,
		OnToolCall:       onToolCall,
		MaxToolRounds:    maxRounds,
		Retries:          m.opts.Retries,
	})
}

// Stream is an alias for Call — streaming is the default consumption mode.
func (m *Model) Stream(prompt string, opts *CallOpts) *Result {
	return m.Call(prompt, opts)
}

// SubmitTools submits tool results and continues the conversation.
func (m *Model) SubmitTools(results map[string]string, opts *CallOpts) *Result {
	if len(m.pendingToolCalls) == 0 {
		r := &Result{}
		r.failure = &ProviderError{ULMError{"no pending tool calls"}}
		r.done = true
		return r
	}

	var parts []Part
	for _, tc := range m.pendingToolCalls {
		if tc.ID == "" {
			continue
		}
		val, ok := results[tc.ID]
		if !ok {
			continue
		}
		parts = append(parts, ToolResultPart(tc.ID, []Part{TextPart(val)}, tc.Name))
	}

	toolMsg := Message{Role: RoleTool, Parts: parts}
	followMessages := append(append([]Message{}, m.conversation...), toolMsg)

	tools, callableRegistry := m.normalizeTools()
	config := m.baseConfig(nil)

	request := LMRequest{
		Model:    m.opts.Model,
		Messages: followMessages,
		System:   m.opts.System,
		Tools:    tools,
		Config:   config,
	}

	resolvedProvider := m.opts.Provider
	if opts != nil && opts.Provider != "" {
		resolvedProvider = opts.Provider
	}

	startStream := func(req LMRequest) (<-chan StreamEvent, error) {
		return m.opts.LM.Stream(req, resolvedProvider)
	}

	onFinished := func(finalReq LMRequest, resp LMResponse) {
		if m.localCache != nil {
			m.localCache[cacheKey(finalReq, resolvedProvider)] = resp
		}
		m.history = append(m.history, HistoryEntry{Request: finalReq, Response: resp})
		m.pendingToolCalls = resp.ToolCalls()
		m.conversation = append(append([]Message{}, finalReq.Messages...), resp.Message)
	}

	return NewResult(ResultOpts{
		Request:          request,
		StartStream:      startStream,
		OnFinished:       onFinished,
		CallableRegistry: callableRegistry,
		OnToolCall:       m.opts.OnToolCall,
		MaxToolRounds:    m.opts.MaxToolRounds,
		Retries:          m.opts.Retries,
	})
}

// Upload uploads a file via the provider's file API and returns a Part.
func (m *Model) Upload(filePath string) (Part, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return Part{}, err
	}
	filename := filepath.Base(filePath)
	ext := filepath.Ext(filePath)
	mediaType := mime.TypeByExtension(ext)
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}

	provider := m.opts.Provider
	if provider == "" {
		provider, _ = ResolveProvider(m.opts.Model)
	}

	req := FileUploadRequest{
		Model:     m.opts.Model,
		Filename:  filename,
		BytesData: data,
		MediaType: mediaType,
	}
	resp, err := m.opts.LM.FileUpload(req, provider)
	if err != nil {
		return Part{}, err
	}

	ds := DataSource{Type: "file", FileID: resp.ID, MediaType: mediaType}
	switch {
	case len(mediaType) > 6 && mediaType[:6] == "image/":
		return ImagePart(ds), nil
	case len(mediaType) > 6 && mediaType[:6] == "audio/":
		return AudioPart(ds), nil
	case len(mediaType) > 6 && mediaType[:6] == "video/":
		return VideoPart(ds), nil
	default:
		return DocumentPart(ds), nil
	}
}

// ── Private ────────────────────────────────────────────────────────

func (m *Model) buildRequest(prompt string, opts *CallOpts) (LMRequest, map[string]func(map[string]any) (any, error)) {
	if opts == nil {
		opts = &CallOpts{}
	}

	messages := append([]Message{}, m.conversation...)
	messages = append(messages, UserMessage(prompt))
	if opts.Prefill != "" {
		messages = append(messages, AssistantMessage(opts.Prefill))
	}

	tools, callableRegistry := m.normalizeTools()
	if opts.Tools != nil {
		tools = opts.Tools
		callableRegistry = make(map[string]func(map[string]any) (any, error))
		for _, t := range opts.Tools {
			if t.Type == "function" && t.Fn != nil {
				callableRegistry[t.Name] = t.Fn
			}
		}
	}

	config := m.baseConfig(opts)
	system := m.opts.System
	if opts.System != "" {
		system = opts.System
	}

	request := LMRequest{
		Model:    m.opts.Model,
		Messages: messages,
		System:   system,
		Tools:    tools,
		Config:   config,
	}

	return request, callableRegistry
}

func (m *Model) normalizeTools() ([]Tool, map[string]func(map[string]any) (any, error)) {
	registry := make(map[string]func(map[string]any) (any, error))
	for _, t := range m.opts.Tools {
		if t.Type == "function" && t.Fn != nil {
			registry[t.Name] = t.Fn
		}
	}
	return m.opts.Tools, registry
}

func (m *Model) baseConfig(opts *CallOpts) Config {
	cfg := Config{
		MaxTokens:   m.opts.MaxTokens,
		Temperature: m.opts.Temperature,
	}

	providerCfg := map[string]any{}
	if m.opts.PromptCaching {
		providerCfg["prompt_caching"] = true
	}

	if opts != nil {
		if opts.Temperature != nil {
			cfg.Temperature = opts.Temperature
		}
		if opts.MaxTokens != nil {
			cfg.MaxTokens = opts.MaxTokens
		}
		if opts.TopP != nil {
			cfg.TopP = opts.TopP
		}
		if opts.Stop != nil {
			cfg.Stop = opts.Stop
		}
		if opts.PromptCaching {
			providerCfg["prompt_caching"] = true
		}
		if opts.Output != "" {
			providerCfg["output"] = opts.Output
		}

		switch v := opts.Reasoning.(type) {
		case bool:
			if v {
				cfg.Reasoning = map[string]any{"enabled": true}
			}
		case map[string]any:
			r := map[string]any{"enabled": true}
			for k, val := range v {
				r[k] = val
			}
			cfg.Reasoning = r
		}
	}

	if len(providerCfg) > 0 {
		cfg.Provider = providerCfg
	}

	return cfg
}

func cacheKey(req LMRequest, provider string) string {
	return fmt.Sprintf("%s|%s|%v", provider, req.Model, req.Messages)
}

func responseToEventChan(resp LMResponse) <-chan StreamEvent {
	ch := make(chan StreamEvent, len(resp.Message.Parts)+2)
	go func() {
		defer close(ch)
		ch <- StreamEvent{Type: "start", ID: resp.ID, Model: resp.Model}
		for i, p := range resp.Message.Parts {
			idx := i
			switch p.Type {
			case PartText, PartRefusal:
				ch <- StreamEvent{Type: "delta", PartIndex: &idx, Delta: &PartDelta{Type: "text", Text: p.Text}}
			case PartThinking:
				ch <- StreamEvent{Type: "delta", PartIndex: &idx, Delta: &PartDelta{Type: "thinking", Text: p.Text}}
			case PartToolCall:
				ch <- StreamEvent{Type: "delta", PartIndex: &idx, DeltaRaw: map[string]any{
					"type": "tool_call", "id": p.ID, "name": p.Name, "input": "{}",
				}}
			}
		}
		ch <- StreamEvent{Type: "end", FinishReason: resp.FinishReason, Usage: &resp.Usage}
	}()
	return ch
}
