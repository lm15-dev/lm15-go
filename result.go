package lm15

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// StartStreamFn opens a streaming connection for a request.
type StartStreamFn func(req LMRequest) (<-chan StreamEvent, error)

// OnFinishedFn is called when a result is finalized.
type OnFinishedFn func(req LMRequest, resp LMResponse)

// ResultOpts configures a Result.
type ResultOpts struct {
	Request          LMRequest
	StartStream      StartStreamFn
	OnFinished       OnFinishedFn
	CallableRegistry map[string]func(map[string]any) (any, error)
	OnToolCall       func(ToolCallInfo) interface{}
	MaxToolRounds    int
	Retries          int
}

// Result is a lazy stream-backed response.
// Call Text() to block and get the text, or Stream() to iterate chunks.
type Result struct {
	opts     ResultOpts
	response *LMResponse
	failure  error
	done     bool
	consumed bool
	mu       sync.Mutex
}

// NewResult creates a Result from options.
func NewResult(opts ResultOpts) *Result {
	if opts.MaxToolRounds == 0 {
		opts.MaxToolRounds = 8
	}
	return &Result{opts: opts}
}

// Response blocks until the full response is available.
func (r *Result) Response() (LMResponse, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.done && r.failure != nil {
		return LMResponse{}, r.failure
	}
	if r.response != nil {
		return *r.response, nil
	}
	// Consume the stream
	for range r.streamLocked() {
	}
	if r.failure != nil {
		return LMResponse{}, r.failure
	}
	if r.response == nil {
		return LMResponse{}, &ProviderError{ULMError{"no response"}}
	}
	return *r.response, nil
}

// Text blocks and returns the response text.
func (r *Result) Text() string {
	resp, err := r.Response()
	if err != nil {
		return ""
	}
	return resp.Text()
}

// Thinking blocks and returns the thinking text.
func (r *Result) Thinking() string {
	resp, err := r.Response()
	if err != nil {
		return ""
	}
	return resp.Thinking()
}

// ToolCalls blocks and returns any tool call parts.
func (r *Result) ToolCalls() []Part {
	resp, err := r.Response()
	if err != nil {
		return nil
	}
	return resp.ToolCalls()
}

// Image blocks and returns the first image part, or nil.
func (r *Result) Image() *Part {
	resp, err := r.Response()
	if err != nil {
		return nil
	}
	return resp.Image()
}

// Audio blocks and returns the first audio part, or nil.
func (r *Result) Audio() *Part {
	resp, err := r.Response()
	if err != nil {
		return nil
	}
	return resp.Audio()
}

// Citations blocks and returns all citation parts.
func (r *Result) Citations() []Part {
	resp, err := r.Response()
	if err != nil {
		return nil
	}
	var citations []Part
	for _, p := range resp.Message.Parts {
		if p.Type == PartCitation {
			citations = append(citations, p)
		}
	}
	return citations
}

// FinishReason blocks and returns the finish reason.
func (r *Result) FinishReason() FinishReason {
	resp, err := r.Response()
	if err != nil {
		return FinishError
	}
	return resp.FinishReason
}

// Usage blocks and returns the usage.
func (r *Result) Usage() Usage {
	resp, err := r.Response()
	if err != nil {
		return Usage{}
	}
	return resp.Usage
}

// Err returns the error, if any. Must be called after Response/Text/Stream.
func (r *Result) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.failure
}

// Stream returns a channel of StreamChunks for incremental consumption.
func (r *Result) Stream() <-chan StreamChunk {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.streamLocked()
}

func (r *Result) streamLocked() <-chan StreamChunk {
	if r.consumed {
		ch := make(chan StreamChunk)
		close(ch)
		return ch
	}
	r.consumed = true

	ch := make(chan StreamChunk, 32)
	go r.runStream(ch)
	return ch
}

func (r *Result) runStream(ch chan<- StreamChunk) {
	defer func() {
		r.done = true
		close(ch)
	}()

	currentRequest := r.opts.Request
	rounds := 0

	for {
		state := newRoundState(currentRequest)
		events, err := r.opts.StartStream(currentRequest)
		if err != nil {
			r.failure = err
			return
		}

		for evt := range events {
			if evt.Type == "error" {
				if evt.Error != nil {
					r.failure = ErrorForCode(evt.Error.Code, evt.Error.Message)
				} else {
					r.failure = &ProviderError{ULMError{"stream error"}}
				}
				return
			}
			for _, chunk := range state.apply(evt) {
				ch <- chunk
			}
			if evt.Type == "end" {
				break
			}
		}

		resp := state.materialize()
		r.response = &resp

		// Yield tool_call chunks
		toolCalls := resp.ToolCalls()
		for _, tc := range toolCalls {
			ch <- StreamChunk{Type: "tool_call", Name: tc.Name, Input: tc.Input}
		}

		// Auto-execute tools
		if resp.FinishReason == FinishToolCall && len(toolCalls) > 0 && rounds < r.opts.MaxToolRounds {
			executed := r.executeTools(toolCalls)
			if len(executed) == len(toolCalls) {
				for _, outcome := range executed {
					ch <- StreamChunk{Type: "tool_result", Text: outcome.preview, Name: outcome.name}
				}
				toolMsg := Message{Role: RoleTool, Parts: make([]Part, len(executed))}
				for i, e := range executed {
					toolMsg.Parts[i] = e.part
				}
				currentRequest = LMRequest{
					Model:    currentRequest.Model,
					Messages: append(append(currentRequest.Messages, resp.Message), toolMsg),
					System:   currentRequest.System,
					Tools:    currentRequest.Tools,
					Config:   currentRequest.Config,
				}
				rounds++
				continue
			}
		}

		// Finalize
		if r.opts.OnFinished != nil {
			r.opts.OnFinished(currentRequest, resp)
		}
		ch <- StreamChunk{Type: "finished", Response: &resp}
		return
	}
}

type executedTool struct {
	name    string
	part    Part
	preview string
}

func (r *Result) executeTools(toolCalls []Part) []executedTool {
	var results []executedTool
	for _, tc := range toolCalls {
		info := ToolCallInfo{ID: tc.ID, Name: tc.Name, Input: tc.Input}

		// Check on_tool_call callback
		if r.opts.OnToolCall != nil {
			override := r.opts.OnToolCall(info)
			if override != nil {
				content := []Part{TextPart(fmt.Sprint(override))}
				results = append(results, executedTool{
					name:    info.Name,
					part:    ToolResultPart(info.ID, content, info.Name),
					preview: fmt.Sprint(override),
				})
				continue
			}
		}

		// Check callable registry
		fn := r.opts.CallableRegistry[info.Name]
		if fn == nil {
			return results // Can't execute, return partial
		}

		output, err := fn(info.Input)
		text := fmt.Sprint(output)
		if err != nil {
			text = fmt.Sprintf("error: %v", err)
		}
		content := []Part{TextPart(text)}
		results = append(results, executedTool{
			name:    info.Name,
			part:    ToolResultPart(info.ID, content, info.Name),
			preview: text,
		})
	}
	return results
}

// ── RoundState ─────────────────────────────────────────────────────

type roundState struct {
	request      LMRequest
	startedID    string
	startedModel string
	finishReason FinishReason
	usage        *Usage
	textParts    []string
	thinkingParts []string
	audioChunks  []string
	toolCallRaw  map[int]string
	toolCallMeta map[int]map[string]any
}

func newRoundState(request LMRequest) *roundState {
	return &roundState{
		request:      request,
		toolCallRaw:  make(map[int]string),
		toolCallMeta: make(map[int]map[string]any),
	}
}

func (s *roundState) apply(event StreamEvent) []StreamChunk {
	var chunks []StreamChunk

	switch event.Type {
	case "start":
		if event.ID != "" {
			s.startedID = event.ID
		}
		if event.Model != "" {
			s.startedModel = event.Model
		}
	case "end":
		if event.FinishReason != "" {
			s.finishReason = event.FinishReason
		}
		s.usage = event.Usage
	case "delta":
		if event.Delta != nil {
			switch event.Delta.Type {
			case "text":
				s.textParts = append(s.textParts, event.Delta.Text)
				chunks = append(chunks, StreamChunk{Type: "text", Text: event.Delta.Text})
			case "thinking":
				s.thinkingParts = append(s.thinkingParts, event.Delta.Text)
				chunks = append(chunks, StreamChunk{Type: "thinking", Text: event.Delta.Text})
			case "audio":
				s.audioChunks = append(s.audioChunks, event.Delta.Data)
				chunks = append(chunks, StreamChunk{Type: "audio", Text: event.Delta.Data})
			case "tool_call":
				idx := 0
				if event.PartIndex != nil {
					idx = *event.PartIndex
				}
				s.pushToolCall(idx, event.Delta.Input)
			}
		}
		// Handle raw delta (tool_call with structured fields)
		if event.DeltaRaw != nil {
			dt, _ := event.DeltaRaw["type"].(string)
			if dt == "tool_call" {
				idx := 0
				if event.PartIndex != nil {
					idx = *event.PartIndex
				}
				meta := s.toolCallMeta[idx]
				if meta == nil {
					meta = make(map[string]any)
					s.toolCallMeta[idx] = meta
				}
				if id, ok := event.DeltaRaw["id"]; ok && id != nil {
					meta["id"] = id
				}
				if name, ok := event.DeltaRaw["name"]; ok && name != nil {
					meta["name"] = name
				}
				if input, ok := event.DeltaRaw["input"]; ok {
					s.pushToolCall(idx, fmt.Sprint(input))
				}
			}
		}
	}

	return chunks
}

func (s *roundState) pushToolCall(idx int, rawInput string) {
	agg := s.toolCallRaw[idx] + rawInput
	s.toolCallRaw[idx] = agg

	meta := s.toolCallMeta[idx]
	if meta == nil {
		meta = make(map[string]any)
		s.toolCallMeta[idx] = meta
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(agg), &parsed); err == nil {
		meta["input"] = parsed
	}
}

func (s *roundState) materialize() LMResponse {
	var parts []Part

	if len(s.thinkingParts) > 0 {
		parts = append(parts, ThinkingPart(join(s.thinkingParts)))
	}
	if len(s.textParts) > 0 {
		parts = append(parts, TextPart(join(s.textParts)))
	}
	if len(s.audioChunks) > 0 {
		parts = append(parts, AudioBase64(join(s.audioChunks), "audio/wav"))
	}

	// Tool calls
	toolNames := make([]string, 0)
	for _, t := range s.request.Tools {
		if t.Type == "function" {
			toolNames = append(toolNames, t.Name)
		}
	}

	indices := sortedKeys(s.toolCallMeta)
	for pos, idx := range indices {
		meta := s.toolCallMeta[idx]
		payload, _ := meta["input"].(map[string]any)
		if payload == nil {
			var parsed map[string]any
			if err := json.Unmarshal([]byte(s.toolCallRaw[idx]), &parsed); err == nil {
				payload = parsed
			} else {
				payload = map[string]any{}
			}
		}
		name, _ := meta["name"].(string)
		if name == "" {
			if len(toolNames) == 1 {
				name = toolNames[0]
			} else if pos < len(toolNames) {
				name = toolNames[pos]
			} else {
				name = "tool"
			}
		}
		id, _ := meta["id"].(string)
		if id == "" {
			id = fmt.Sprintf("tool_call_%d", idx)
		}
		parts = append(parts, ToolCallPart(id, name, payload))
	}

	if len(parts) == 0 {
		parts = append(parts, TextPart(""))
	}

	finish := s.finishReason
	if finish == "" {
		finish = FinishStop
		for _, p := range parts {
			if p.Type == PartToolCall {
				finish = FinishToolCall
				break
			}
		}
	} else if finish == FinishStop {
		for _, p := range parts {
			if p.Type == PartToolCall {
				finish = FinishToolCall
				break
			}
		}
	}

	usage := Usage{}
	if s.usage != nil {
		usage = *s.usage
	}

	model := s.startedModel
	if model == "" {
		model = s.request.Model
	}

	return LMResponse{
		ID:           s.startedID,
		Model:        model,
		Message:      Message{Role: RoleAssistant, Parts: parts},
		FinishReason: finish,
		Usage:        usage,
	}
}

// ── Helpers ────────────────────────────────────────────────────────

func join(parts []string) string {
	result := ""
	for _, p := range parts {
		result += p
	}
	return result
}

func sortedKeys(m map[int]map[string]any) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Simple insertion sort (small N)
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

// sleep helper for retries
func sleep(d time.Duration) {
	time.Sleep(d)
}
