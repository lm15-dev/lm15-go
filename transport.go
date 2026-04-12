package lm15

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// TransportPolicy configures HTTP behavior.
type TransportPolicy struct {
	Timeout        time.Duration
	ConnectTimeout time.Duration
}

// DefaultPolicy returns sensible transport defaults.
func DefaultPolicy() TransportPolicy {
	return TransportPolicy{
		Timeout:        30 * time.Second,
		ConnectTimeout: 10 * time.Second,
	}
}

// HTTPRequest is a normalized HTTP request.
type HTTPRequest struct {
	Method  string
	URL     string
	Headers map[string]string
	Params  map[string]string
	Body    []byte
	Timeout time.Duration
}

// HTTPResponse is a normalized HTTP response.
type HTTPResponse struct {
	Status  int
	Headers map[string]string
	Body    []byte
}

// Text returns the response body as a string.
func (r HTTPResponse) Text() string { return string(r.Body) }

// JSON decodes the response body into the target.
func (r HTTPResponse) JSON(target any) error { return json.Unmarshal(r.Body, target) }

// Transport executes HTTP requests.
type Transport interface {
	Request(req HTTPRequest) (HTTPResponse, error)
	Stream(req HTTPRequest) (io.ReadCloser, error)
}

// StdTransport uses net/http (zero dependencies).
type StdTransport struct {
	Policy TransportPolicy
	Client *http.Client
}

// NewStdTransport creates a transport using the standard library.
func NewStdTransport(policy TransportPolicy) *StdTransport {
	return &StdTransport{
		Policy: policy,
		Client: &http.Client{Timeout: policy.Timeout},
	}
}

func (t *StdTransport) buildURL(req HTTPRequest) string {
	if len(req.Params) == 0 {
		return req.URL
	}
	v := url.Values{}
	for k, val := range req.Params {
		v.Set(k, val)
	}
	return req.URL + "?" + v.Encode()
}

func (t *StdTransport) buildHTTPRequest(req HTTPRequest) (*http.Request, error) {
	fullURL := t.buildURL(req)
	var body io.Reader
	if req.Body != nil {
		body = bytes.NewReader(req.Body)
	}
	httpReq, err := http.NewRequest(req.Method, fullURL, body)
	if err != nil {
		return nil, err
	}
	for k, v := range req.Headers {
		httpReq.Header.Set(k, v)
	}
	return httpReq, nil
}

// Request executes a synchronous HTTP request.
func (t *StdTransport) Request(req HTTPRequest) (HTTPResponse, error) {
	httpReq, err := t.buildHTTPRequest(req)
	if err != nil {
		return HTTPResponse{}, &TransportError{ULMError{err.Error()}}
	}

	client := t.Client
	if req.Timeout > 0 {
		client = &http.Client{Timeout: req.Timeout}
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return HTTPResponse{}, &TransportError{ULMError{err.Error()}}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return HTTPResponse{}, &TransportError{ULMError{err.Error()}}
	}

	headers := make(map[string]string)
	for k := range resp.Header {
		headers[http.CanonicalHeaderKey(k)] = resp.Header.Get(k)
	}

	return HTTPResponse{Status: resp.StatusCode, Headers: headers, Body: body}, nil
}

// Stream opens a streaming HTTP connection and returns a line reader.
func (t *StdTransport) Stream(req HTTPRequest) (io.ReadCloser, error) {
	httpReq, err := t.buildHTTPRequest(req)
	if err != nil {
		return nil, &TransportError{ULMError{err.Error()}}
	}

	client := t.Client
	if req.Timeout > 0 {
		client = &http.Client{Timeout: req.Timeout}
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, &TransportError{ULMError{err.Error()}}
	}

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, &TransportError{ULMError{fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body))}}
	}

	return resp.Body, nil
}

// SSEEvent is a parsed Server-Sent Event.
type SSEEvent struct {
	Event string
	Data  string
}

// ParseSSE reads SSE events from a reader.
func ParseSSE(r io.Reader) <-chan SSEEvent {
	ch := make(chan SSEEvent, 16)
	go func() {
		defer close(ch)
		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 64*1024), 1024*1024)

		var eventName string
		var dataLines []string

		for scanner.Scan() {
			line := scanner.Text()

			if line == "" {
				if len(dataLines) > 0 {
					data := dataLines[0]
					for _, d := range dataLines[1:] {
						data += "\n" + d
					}
					ch <- SSEEvent{Event: eventName, Data: data}
				}
				eventName = ""
				dataLines = nil
				continue
			}

			if line[0] == ':' {
				continue // comment
			}

			if len(line) > 6 && line[:6] == "event:" {
				eventName = trimLeft(line[6:])
			} else if len(line) > 5 && line[:5] == "data:" {
				dataLines = append(dataLines, trimLeft(line[5:]))
			}
		}

		// Flush remaining
		if len(dataLines) > 0 {
			data := dataLines[0]
			for _, d := range dataLines[1:] {
				data += "\n" + d
			}
			ch <- SSEEvent{Event: eventName, Data: data}
		}
	}()
	return ch
}

func trimLeft(s string) string {
	if len(s) > 0 && s[0] == ' ' {
		return s[1:]
	}
	return s
}
