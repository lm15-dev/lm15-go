package lm15

// The client layer: provider LM types with Complete/Stream over net/http.
// This mirrors the reference implementation's public surface (OpenAILM,
// OpenAIChatLM, AnthropicLM, GeminiLM with complete()/stream()) adapted to
// Go idiom: context-first methods, functional options, iter.Seq2 streams.
// Stdlib only, per the zero-dependency budget.

import (
	"bufio"
	"context"
	"errors"
	"io"
	"iter"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ─── Shared HTTP transport ───────────────────────────────────────────

// defaultHTTPClient is shared across all LM instances; its Transport pools
// keep-alive connections natively. No overall Timeout — streams are
// long-lived — but dial/TLS/header phases are bounded, and callers bound
// requests with context.
var defaultHTTPClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConns:          32,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 120 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

// ─── Options ─────────────────────────────────────────────────────────

// Option configures an LM constructor.
type Option func(*providerClient) error

// WithBaseURL overrides the provider's default base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *providerClient) error {
		c.baseURL = strings.TrimRight(baseURL, "/")
		return nil
	}
}

// WithHTTPClient substitutes the shared http.Client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *providerClient) error {
		c.httpClient = client
		return nil
	}
}

// WithCompat selects an OpenAI Chat Completions compat preset by name
// (CompatOllama, CompatGroq, ...). The preset bundles the server's wire
// quirks and its default base URL; an explicit WithBaseURL still wins.
// Only meaningful for NewOpenAIChatLM.
func WithCompat(preset string) Option {
	return func(c *providerClient) error {
		if c.provider != "openai_chat" {
			return ConfigurationError{ErrorMeta{Message: "compat presets apply only to OpenAIChatLM"}}
		}
		compat, err := ChatCompatPreset(preset)
		if err != nil {
			return err
		}
		c.compat = compat
		c.compatName = preset
		return nil
	}
}

// Compat preset names for WithCompat (see ChatCompatPreset).
const (
	CompatOpenAI     = "openai"
	CompatOllama     = "ollama"
	CompatGroq       = "groq"
	CompatOpenRouter = "openrouter"
	CompatVLLM       = "vllm"
	CompatSGLang     = "sglang"
	CompatLMStudio   = "lmstudio"
)

// ─── Provider clients ────────────────────────────────────────────────

type providerClient struct {
	provider   string
	apiKey     string
	baseURL    string
	compat     ChatCompat
	compatName string
	httpClient *http.Client
}

// OpenAILM talks to the OpenAI Responses API.
type OpenAILM struct{ providerClient }

// OpenAIChatLM talks to the OpenAI Chat Completions dialect — OpenAI itself
// or any compatible server via WithCompat / WithBaseURL.
type OpenAIChatLM struct{ providerClient }

// AnthropicLM talks to the Anthropic Messages API.
type AnthropicLM struct{ providerClient }

// GeminiLM talks to the Google Gemini generateContent API.
type GeminiLM struct{ providerClient }

func newClient(provider, apiKey string, opts []Option) (providerClient, error) {
	c := providerClient{provider: provider, apiKey: apiKey, compat: defaultChatCompat(), httpClient: defaultHTTPClient}
	for _, opt := range opts {
		if err := opt(&c); err != nil {
			return providerClient{}, err
		}
	}
	if c.baseURL == "" && c.compatName != "" {
		if u, ok := ChatPresetBaseURLs[strings.ToLower(c.compatName)]; ok {
			c.baseURL = u
		}
	}
	return c, nil
}

// NewOpenAILM builds an OpenAI Responses API client.
func NewOpenAILM(apiKey string, opts ...Option) (*OpenAILM, error) {
	c, err := newClient("openai", apiKey, opts)
	return &OpenAILM{c}, err
}

// NewOpenAIChatLM builds a Chat Completions client (OpenAI-compatible servers).
func NewOpenAIChatLM(apiKey string, opts ...Option) (*OpenAIChatLM, error) {
	c, err := newClient("openai_chat", apiKey, opts)
	return &OpenAIChatLM{c}, err
}

// NewAnthropicLM builds an Anthropic Messages API client.
func NewAnthropicLM(apiKey string, opts ...Option) (*AnthropicLM, error) {
	c, err := newClient("anthropic", apiKey, opts)
	return &AnthropicLM{c}, err
}

// NewGeminiLM builds a Gemini client.
func NewGeminiLM(apiKey string, opts ...Option) (*GeminiLM, error) {
	c, err := newClient("gemini", apiKey, opts)
	return &GeminiLM{c}, err
}

// Provider reports the adapter name ("openai", "openai_chat", ...).
func (c *providerClient) Provider() string { return c.provider }

func (c *providerClient) buildWire(req Request, stream bool) (TransportRequest, error) {
	if c.provider == "openai_chat" {
		base := c.baseURL
		if base == "" {
			base = DefaultBaseURLs["openai_chat"]
		}
		return buildOpenAIChat(req, stream, c.apiKey, strings.TrimRight(base, "/"), c.compat)
	}
	return BuildRequest(c.provider, req, stream, c.apiKey, c.baseURL)
}

func transportErr(err error) LM15Error {
	return TransportError{ErrorMeta{Message: err.Error()}}
}

func (c *providerClient) do(ctx context.Context, tr TransportRequest) (*http.Response, error) {
	u := tr.URL
	if len(tr.Params) > 0 {
		q := url.Values{}
		for k, v := range tr.Params {
			q.Set(k, v)
		}
		u += "?" + q.Encode()
	}
	var body io.Reader
	if tr.Body != nil {
		body = strings.NewReader(compactJSON(tr.Body))
	}
	httpReq, err := http.NewRequestWithContext(ctx, tr.Method, u, body)
	if err != nil {
		return nil, transportErr(err)
	}
	for k, v := range tr.Headers {
		httpReq.Header.Set(k, v)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, TimeoutError{ProviderError{ErrorMeta{Message: err.Error(), Provider: c.provider}}}
		}
		return nil, transportErr(err)
	}
	return resp, nil
}

// Complete sends the canonical Request and returns the canonical Response.
func (c *providerClient) Complete(ctx context.Context, req Request) (Response, error) {
	tr, err := c.buildWire(req, false)
	if err != nil {
		return Response{}, err
	}
	httpResp, err := c.do(ctx, tr)
	if err != nil {
		return Response{}, err
	}
	defer httpResp.Body.Close()
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return Response{}, transportErr(err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return Response{}, c.normalizedHTTPError(httpResp.StatusCode, body)
	}
	resp, _, err := ParseResponse(c.provider, req, httpResp.StatusCode, body)
	return resp, err
}

func (c *providerClient) normalizedHTTPError(status int, body []byte) error {
	e, err := NormalizeError(c.provider, status, string(body))
	if err != nil {
		return MapHTTPError(status, string(body), c.provider, "", "")
	}
	return e
}

// Stream sends the Request in streaming mode and yields canonical stream
// events. MAP-3 holds: exactly one StreamEndEvent ends the stream, last,
// carrying merged finish_reason/usage. Iteration stops at the first
// non-nil error; abandoning the loop closes the connection.
func (c *providerClient) Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		tr, err := c.buildWire(req, true)
		if err != nil {
			yield(nil, err)
			return
		}
		httpResp, err := c.do(ctx, tr)
		if err != nil {
			yield(nil, err)
			return
		}
		defer httpResp.Body.Close()
		if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(httpResp.Body, 1<<20))
			yield(nil, c.normalizedHTTPError(httpResp.StatusCode, body))
			return
		}

		// Incremental SSE scan + MAP-3 coalescing: adapters may emit one
		// end event per provider terminal frame; absorb them all and emit
		// the single merged end event when the body finishes.
		var (
			eventName  *string
			dataLines  []string
			eventBytes int
			sawEnd     bool
			endEvent   StreamEndEvent
			stopped    bool
		)
		emit := func(ev StreamEvent) bool {
			if end, ok := ev.(StreamEndEvent); ok {
				sawEnd = true
				if end.FinishReason != nil {
					endEvent.FinishReason = end.FinishReason
				}
				if end.Usage != nil {
					endEvent.Usage = end.Usage
				}
				if end.ProviderData != nil {
					endEvent.ProviderData = end.ProviderData
				}
				return true
			}
			return yield(ev, nil)
		}
		flush := func() bool {
			if len(dataLines) == 0 {
				eventName = nil
				eventBytes = 0
				return true
			}
			raw := SSEEvent{Event: eventName, Data: strings.Join(dataLines, "\n")}
			eventName = nil
			dataLines = nil
			eventBytes = 0
			events, mapErr := ParseStreamEvents(c.provider, req, raw)
			if mapErr != nil {
				yield(nil, mapErr)
				return false
			}
			for _, ev := range events {
				if !emit(ev) {
					return false
				}
			}
			return true
		}

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), sseMaxLineBytes)
		for scanner.Scan() {
			line := strings.TrimRight(scanner.Text(), "\r")
			eventBytes += len(line) + 1
			if eventBytes > sseMaxEventBytes {
				yield(nil, TransportError{ErrorMeta{Message: "SSE event exceeds limit"}})
				return
			}
			switch {
			case line == "":
				if !flush() {
					stopped = true
				}
			case strings.HasPrefix(line, ":"):
			case strings.HasPrefix(line, "event:"):
				name := strings.TrimSpace(line[len("event:"):])
				eventName = &name
			case strings.HasPrefix(line, "data:"):
				dataLines = append(dataLines, strings.TrimLeft(line[len("data:"):], " \t\n\r\v\f"))
			}
			if stopped {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			if errors.Is(err, bufio.ErrTooLong) {
				yield(nil, TransportError{ErrorMeta{Message: "SSE line exceeds limit"}})
				return
			}
			yield(nil, transportErr(err))
			return
		}
		if !flush() {
			return
		}
		if sawEnd {
			yield(endEvent, nil)
		}
	}
}

// CollectResponse runs Stream and materializes the events into a Response —
// the streaming twin of Complete (same shape: message, finish_reason, usage).
func (c *providerClient) CollectResponse(ctx context.Context, req Request) (Response, error) {
	var events []StreamEvent
	for ev, err := range c.Stream(ctx, req) {
		if err != nil {
			return Response{}, err
		}
		events = append(events, ev)
	}
	return MaterializeResponse(events, req), nil
}

// ─── Message constructors (mirror of Message.user / .assistant / .tool) ──

// Text creates a text part.
func Text(content string) TextPart { return TextPart{Text: content} }

// UserMessage creates a user message from text.
func UserMessage(text string) Message {
	return Message{Role: "user", Parts: []Part{TextPart{Text: text}}}
}

// UserMessageParts creates a user message from explicit parts.
func UserMessageParts(parts ...Part) Message {
	return Message{Role: "user", Parts: parts}
}

// AssistantMessage creates an assistant message from text.
func AssistantMessage(text string) Message {
	return Message{Role: "assistant", Parts: []Part{TextPart{Text: text}}}
}

// AssistantMessageParts creates an assistant message from explicit parts
// (e.g. replaying the model's ToolCallParts in a tools round-trip).
func AssistantMessageParts(parts ...Part) Message {
	return Message{Role: "assistant", Parts: parts}
}

// ToolResults creates a tool message carrying one or more tool results.
func ToolResults(results ...ToolResultPart) Message {
	parts := make([]Part, len(results))
	for i, r := range results {
		parts[i] = r
	}
	return Message{Role: "tool", Parts: parts}
}

// ToolResult builds a text tool result answering the tool call with this ID.
func ToolResult(callID, content string) ToolResultPart {
	return ToolResultPart{ID: callID, Content: []Part{TextPart{Text: content}}}
}

// ─── Response accessors (mirror of Response.text / .tool_calls) ─────

// Text returns the concatenated assistant text, or "" when the response
// carries no text. Citation and thinking parts are metadata around the
// visible answer and don't block extraction; any other part type does.
func (r Response) Text() string {
	var texts []string
	for _, p := range r.Message.Parts {
		switch x := p.(type) {
		case TextPart:
			texts = append(texts, x.Text)
		case CitationPart, ThinkingPart:
		default:
			return ""
		}
	}
	return strings.Join(texts, "\n")
}

// ToolCalls returns the response's tool-call parts, in order.
func (r Response) ToolCalls() []ToolCallPart {
	var calls []ToolCallPart
	for _, p := range r.Message.Parts {
		if tc, ok := p.(ToolCallPart); ok {
			calls = append(calls, tc)
		}
	}
	return calls
}
