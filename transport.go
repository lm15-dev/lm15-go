package lm15

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// TransportRequest is a finished wire request: method, URL (query included),
// ordered headers, body. Dialects produce it; a Transport sends it.
type TransportRequest struct {
	Method         string
	URL            string
	Headers        [][2]string
	Body           []byte
	ConnectTimeout time.Duration
	ReadTimeout    time.Duration
}

// Header returns the first header value with this name (case-insensitive).
func (r *TransportRequest) Header(name string) string {
	for _, h := range r.Headers {
		if strings.EqualFold(h[0], name) {
			return h[1]
		}
	}
	return ""
}

// SetHeader replaces or appends a header (case-insensitive).
func (r *TransportRequest) SetHeader(name, value string) {
	for i, h := range r.Headers {
		if strings.EqualFold(h[0], name) {
			r.Headers[i][1] = value
			return
		}
	}
	r.Headers = append(r.Headers, [2]string{name, value})
}

// TransportResponse is a streaming HTTP response.
type TransportResponse struct {
	Status  int
	Reason  string
	Headers [][2]string
	Body    io.ReadCloser
}

// Header returns the first header value with this name.
func (r *TransportResponse) Header(name string) string {
	for _, h := range r.Headers {
		if strings.EqualFold(h[0], name) {
			return h[1]
		}
	}
	return ""
}

// Transport sends wire requests. The default is net/http; on js/wasm Go's
// net/http rides the browser's fetch, so the same transport works there.
type Transport interface {
	Do(ctx context.Context, req *TransportRequest) (*TransportResponse, error)
}

// HTTPTransport is the net/http transport.
type HTTPTransport struct {
	Client *http.Client
}

// NewHTTPTransport creates the default transport.
func NewHTTPTransport() *HTTPTransport {
	return &HTTPTransport{Client: &http.Client{}}
}

// Do implements Transport.
func (t *HTTPTransport) Do(ctx context.Context, req *TransportRequest) (*TransportResponse, error) {
	client := t.Client
	if client == nil {
		client = http.DefaultClient
	}
	if req.ReadTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.ReadTimeout)
		// The body outlives Do; tie the cancel to the body's Close.
		resp, err := t.do(ctx, client, req)
		if err != nil {
			cancel()
			return nil, err
		}
		resp.Body = &cancelOnClose{ReadCloser: resp.Body, cancel: cancel}
		return resp, nil
	}
	return t.do(ctx, client, req)
}

func (t *HTTPTransport) do(ctx context.Context, client *http.Client, req *TransportRequest) (*TransportResponse, error) {
	var body io.Reader
	if len(req.Body) > 0 {
		body = bytes.NewReader(req.Body)
	}
	httpReq, err := http.NewRequestWithContext(ctx, req.Method, req.URL, body)
	if err != nil {
		return nil, newError(KindTransport, err.Error()).WithCause(err)
	}
	for _, h := range req.Headers {
		httpReq.Header.Add(h[0], h[1])
	}
	if len(req.Body) > 0 {
		httpReq.ContentLength = int64(len(req.Body))
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, newError(KindTransport, err.Error()).WithCause(err)
	}
	headers := make([][2]string, 0, len(resp.Header))
	for name, values := range resp.Header {
		for _, v := range values {
			headers = append(headers, [2]string{strings.ToLower(name), v})
		}
	}
	return &TransportResponse{Status: resp.StatusCode, Reason: resp.Status, Headers: headers, Body: resp.Body}, nil
}

type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (c *cancelOnClose) Close() error {
	err := c.ReadCloser.Close()
	c.cancel()
	return err
}

// HTTPResponse is a buffered provider-level response.
type HTTPResponse struct {
	Status  int
	Reason  string
	Headers [][2]string
	Body    []byte
}

// Header returns the first header value with this name.
func (r *HTTPResponse) Header(name string) string {
	for _, h := range r.Headers {
		if strings.EqualFold(h[0], name) {
			return h[1]
		}
	}
	return ""
}

// Text returns the body as UTF-8 text.
func (r *HTTPResponse) Text() string { return string(r.Body) }

// JSON decodes the body as a JSON object.
func (r *HTTPResponse) JSON() (JSONObject, error) { return DecodeJSONObject(r.Body) }

// JSONResponse wraps a JSON body as an HTTPResponse (batch entries, tests).
func JSONResponse(status int, body JSONObject) *HTTPResponse {
	reason := "OK"
	if status >= 400 {
		reason = "Error"
	}
	return &HTTPResponse{Status: status, Reason: reason, Headers: [][2]string{{"content-type", "application/json"}}, Body: mustJSON(body)}
}

// buildURL appends query parameters (nil values dropped).
func buildURL(base string, params map[string]string) string {
	if len(params) == 0 {
		return base
	}
	values := url.Values{}
	for k, v := range params {
		values.Set(k, v)
	}
	sep := "?"
	if strings.Contains(base, "?") {
		sep = "&"
	}
	return base + sep + values.Encode()
}

// makeJSONRequest serializes a payload (or a raw body) into a TransportRequest.
func makeJSONRequest(method, rawURL string, headers [][2]string, params map[string]string, payload any, body []byte, readTimeout time.Duration) (*TransportRequest, error) {
	hdrs := append([][2]string(nil), headers...)
	if payload != nil {
		encoded, err := EncodeJSON(payload)
		if err != nil {
			return nil, err
		}
		body = encoded
		hasCT := false
		for _, h := range hdrs {
			if strings.EqualFold(h[0], "content-type") {
				hasCT = true
			}
		}
		if !hasCT {
			hdrs = append(hdrs, [2]string{"Content-Type", "application/json"})
		}
	}
	return &TransportRequest{Method: method, URL: buildURL(rawURL, params), Headers: hdrs, Body: body, ReadTimeout: readTimeout}, nil
}

// errNotSupportedTransport is the wasm placeholder error.
var errNotSupportedTransport = errors.New("transport not supported on this platform")
