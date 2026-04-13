package lm15

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
)

// HTTPRequestDict is a JSON-serializable representation of an HTTP request
// for cross-SDK comparison.
type HTTPRequestDict struct {
	Method  string            `json:"method"`
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Params  map[string]string `json:"params,omitempty"`
	Body    any               `json:"body"`
}

var authHeaders = map[string]bool{
	"authorization":  true,
	"x-api-key":      true,
	"x-goog-api-key": true,
}

// HTTPRequestToDict converts an HTTPRequest to a JSON-serializable dict.
// Auth headers are redacted for safe sharing.
func HTTPRequestToDict(req HTTPRequest) HTTPRequestDict {
	var body any
	if req.Body != nil {
		var parsed any
		if err := json.Unmarshal(req.Body, &parsed); err == nil {
			body = parsed
		} else {
			body = "<binary>"
		}
	}

	headers := make(map[string]string)
	for k, v := range req.Headers {
		if authHeaders[strings.ToLower(k)] {
			headers[k] = "REDACTED"
		} else {
			headers[k] = v
		}
	}

	var params map[string]string
	if len(req.Params) > 0 {
		params = req.Params
	}

	return HTTPRequestDict{
		Method:  req.Method,
		URL:     req.URL,
		Headers: headers,
		Params:  params,
		Body:    body,
	}
}

// HTTPRequestToCurl converts an HTTPRequest to a curl command string.
func HTTPRequestToCurl(req HTTPRequest, redactAuth bool) string {
	var parts []string
	parts = append(parts, "curl")

	if req.Method != "GET" {
		parts = append(parts, fmt.Sprintf("-X %s", req.Method))
	}

	reqURL := req.URL
	if len(req.Params) > 0 {
		v := url.Values{}
		for k, val := range req.Params {
			v.Set(k, val)
		}
		reqURL = reqURL + "?" + v.Encode()
	}
	parts = append(parts, shellQuote(reqURL))

	for k, v := range req.Headers {
		val := v
		if redactAuth && authHeaders[strings.ToLower(k)] {
			val = "REDACTED"
		}
		parts = append(parts, fmt.Sprintf("-H %s", shellQuote(k+": "+val)))
	}

	if req.Body != nil {
		var parsed any
		if err := json.Unmarshal(req.Body, &parsed); err == nil {
			pretty, _ := json.MarshalIndent(parsed, "", "  ")
			parts = append(parts, fmt.Sprintf("-d %s", shellQuote(string(pretty))))
		} else {
			parts = append(parts, "--data-binary @-")
		}
	}

	return strings.Join(parts, " \\\n  ")
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\\''") + "'"
}

// BuildHTTPRequest builds the provider-level HTTPRequest without sending it.
func BuildHTTPRequest(
	model string,
	prompt string,
	opts *CurlOpts,
) (HTTPRequest, error) {
	if opts == nil {
		opts = &CurlOpts{}
	}

	provider := opts.Provider
	if provider == "" {
		var err error
		provider, err = ResolveProvider(model)
		if err != nil {
			return HTTPRequest{}, err
		}
	}

	lm := getClient(opts.APIKey, provider, opts.Env)

	// Build LMRequest
	messages := []Message{UserMessage(prompt)}
	if opts.Prefill != "" {
		messages = append(messages, AssistantMessage(opts.Prefill))
	}

	config := Config{
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}
	providerCfg := map[string]any{}
	if opts.PromptCaching {
		providerCfg["prompt_caching"] = true
	}
	if opts.Output != "" {
		providerCfg["output"] = opts.Output
	}
	if len(providerCfg) > 0 {
		config.Provider = providerCfg
	}

	var reasoning map[string]any
	switch v := opts.Reasoning.(type) {
	case bool:
		if v {
			reasoning = map[string]any{"enabled": true}
		}
	case map[string]any:
		reasoning = map[string]any{"enabled": true}
		for k, val := range v {
			reasoning[k] = val
		}
	}
	config.Reasoning = reasoning

	lmReq := LMRequest{
		Model:    model,
		Messages: messages,
		System:   opts.System,
		Tools:    opts.Tools,
		Config:   config,
	}

	// Resolve adapter and build request
	adapter, err := lm.resolveAdapter(model, provider)
	if err != nil {
		return HTTPRequest{}, err
	}

	return adapter.BuildRequest(lmReq, opts.Stream), nil
}

// CurlOpts configures BuildHTTPRequest.
type CurlOpts struct {
	Stream        bool
	Provider      string
	APIKey        interface{}
	Env           string
	System        string
	Tools         []Tool
	Reasoning     interface{}
	Prefill       string
	Output        string
	PromptCaching bool
	Temperature   *float64
	MaxTokens     *int
}

// DumpCurl builds a curl command for the given call parameters.
func DumpCurl(model, prompt string, opts *CurlOpts) (string, error) {
	req, err := BuildHTTPRequest(model, prompt, opts)
	if err != nil {
		return "", err
	}
	redact := true
	return HTTPRequestToCurl(req, redact), nil
}

// DumpHTTP builds the HTTP request dict for cross-SDK comparison.
func DumpHTTP(model, prompt string, opts *CurlOpts) (HTTPRequestDict, error) {
	req, err := BuildHTTPRequest(model, prompt, opts)
	if err != nil {
		return HTTPRequestDict{}, err
	}
	return HTTPRequestToDict(req), nil
}
