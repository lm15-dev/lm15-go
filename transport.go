package lm15

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"compress/zlib"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// TransportRequest is a finished wire request: method, URL (query included),
// ordered headers, body. Dialects produce it; a Transport sends it.
type TransportRequest struct {
	Method         string
	URL            string
	Headers        [][2]string
	Body           []byte
	ConnectTimeout time.Duration // 0 = the transport's
	ReadTimeout    time.Duration // 0 = the transport's
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

// TransportResponse is a streaming HTTP response. Body is the DECODED body:
// a Content-Encoding the transport can inflate has been inflated (INV-053).
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

// ─── Connection budget (spec/vocabularies.md § Connection budget) ────

// Timeouts is how long to wait at each step of a request. Defaults follow
// the provider SDKs, not general-purpose HTTP clients: a model that thinks
// for minutes before its first byte is ordinary, and a client that gives up
// at 60 s turns that into a "network failure" and, under a retry loop,
// restarts the generation each time. Every timeout is per operation, not
// per request: Read bounds the wait for the NEXT byte, so a slow stream
// that keeps trickling never times out.
//
//   - Connect: TCP + TLS to the host.
//   - Read: the next byte of the reply — headers first, then each chunk.
//   - Write: sending the request bytes.
//   - Pool: a free connection when MaxConnections are all busy (0 waits).
//
// A zero field takes the default; negative is refused.
type Timeouts struct {
	Connect time.Duration
	Read    time.Duration
	Write   time.Duration
	Pool    time.Duration
}

// The shared defaults (ratified 2026-09-15, A1).
const (
	DefaultConnectTimeout = 10 * time.Second
	DefaultReadTimeout    = 600 * time.Second
	DefaultWriteTimeout   = 600 * time.Second
	DefaultPoolTimeout    = 600 * time.Second
	DefaultMaxConnections = 100
)

// DefaultTimeouts are the provider SDKs' defaults.
func DefaultTimeouts() Timeouts {
	return Timeouts{Connect: DefaultConnectTimeout, Read: DefaultReadTimeout, Write: DefaultWriteTimeout, Pool: DefaultPoolTimeout}
}

// Validate refuses negative values.
func (t Timeouts) Validate() error {
	for _, v := range []struct {
		name string
		d    time.Duration
	}{{"connect", t.Connect}, {"read", t.Read}, {"write", t.Write}, {"pool", t.Pool}} {
		if v.d < 0 {
			return valueErrorf("Timeouts.%s must not be negative", v.name)
		}
	}
	return nil
}

func (t Timeouts) withDefaults() Timeouts {
	d := DefaultTimeouts()
	if t.Connect == 0 {
		t.Connect = d.Connect
	}
	if t.Read == 0 {
		t.Read = d.Read
	}
	if t.Write == 0 {
		t.Write = d.Write
	}
	if t.Pool == 0 {
		t.Pool = d.Pool
	}
	return t
}

// checkMaxConnections refuses a non-positive cap (0 = the default).
func checkMaxConnections(n int) error {
	if n < 0 {
		return valueErrorf("max_connections must be positive, got %d", n)
	}
	return nil
}

// ─── The net/http transport ──────────────────────────────────────────

// HTTPTransport is the net/http transport with the connection budget:
// one pool shared by every LM of a router, MaxConnections wide, each
// operation bounded by Timeouts. Requests advertise Accept-Encoding:
// identity (a compressed SSE body buffers in proxies); a reply that
// arrives gzip/x-gzip/deflate anyway is inflated incrementally, and br,
// zstd or any other coding is a transport error naming the coding —
// encoded bytes never reach a parser (INV-053).
type HTTPTransport struct {
	Client         *http.Client
	Timeouts       Timeouts
	MaxConnections int

	once sync.Once
	sem  chan struct{}
}

// NewHTTPTransport creates the default transport with the shared defaults.
func NewHTTPTransport() *HTTPTransport {
	return NewHTTPTransportWith(DefaultTimeouts(), DefaultMaxConnections)
}

// NewHTTPTransportWith creates a transport with an explicit budget (zero
// fields take the defaults).
func NewHTTPTransportWith(timeouts Timeouts, maxConnections int) *HTTPTransport {
	timeouts = timeouts.withDefaults()
	if maxConnections <= 0 {
		maxConnections = DefaultMaxConnections
	}
	dialer := &net.Dialer{Timeout: timeouts.Connect}
	inner := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		DialContext:         dialer.DialContext,
		TLSHandshakeTimeout: timeouts.Connect,
		MaxConnsPerHost:     maxConnections,
		MaxIdleConns:        maxConnections,
		MaxIdleConnsPerHost: maxConnections,
		IdleConnTimeout:     90 * time.Second,
		// The request sets Accept-Encoding itself; net/http must not add
		// (and silently decode) its own.
		DisableCompression: true,
	}
	return &HTTPTransport{Client: &http.Client{Transport: inner}, Timeouts: timeouts, MaxConnections: maxConnections}
}

func (t *HTTPTransport) budget() Timeouts { return t.Timeouts.withDefaults() }

// acquire takes a connection slot, waiting up to Pool.
func (t *HTTPTransport) acquire(ctx context.Context) (func(), error) {
	t.once.Do(func() {
		n := t.MaxConnections
		if n <= 0 {
			n = DefaultMaxConnections
		}
		t.sem = make(chan struct{}, n)
	})
	pool := t.budget().Pool
	timer := time.NewTimer(pool)
	defer timer.Stop()
	select {
	case t.sem <- struct{}{}:
		return func() { <-t.sem }, nil
	case <-ctx.Done():
		return nil, newError(KindTransport, ctx.Err().Error()).WithCause(ctx.Err())
	case <-timer.C:
		return nil, newError(KindTimeout, fmt.Sprintf("lm15 transport: no free connection within the pool timeout (%s; %d connections in use) — this is the client's limit, raise Timeouts.Pool or MaxConnections", pool, cap(t.sem)))
	}
}

// Do implements Transport.
func (t *HTTPTransport) Do(ctx context.Context, req *TransportRequest) (*TransportResponse, error) {
	client := t.Client
	if client == nil {
		client = http.DefaultClient
	}
	release, err := t.acquire(ctx)
	if err != nil {
		return nil, err
	}
	read := req.ReadTimeout
	if read <= 0 {
		read = t.budget().Read
	}
	ctx, cancel := context.WithCancel(ctx)
	watch := newIdleWatch(read, cancel)
	resp, err := t.do(ctx, client, req)
	if err != nil {
		watch.stop()
		cancel()
		release()
		if watch.fired() {
			return nil, readTimeoutError(read)
		}
		return nil, err
	}
	resp.Body = &decodedBody{ReadCloser: resp.Body, watch: watch, cancel: cancel, release: release, read: read}
	if err := (resp.Body.(*decodedBody)).decode(resp); err != nil {
		resp.Body.Close()
		return nil, err
	}
	return resp, nil
}

func readTimeoutError(read time.Duration) *Error {
	return newError(KindTimeout, fmt.Sprintf("lm15 transport: no reply bytes within the read timeout (%s) — this is the client's limit, not a dead server; raise Timeouts.Read for slow models", read))
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
	hasAccept := false
	for _, h := range req.Headers {
		httpReq.Header.Add(h[0], h[1])
		if strings.EqualFold(h[0], "accept-encoding") {
			hasAccept = true
		}
	}
	if !hasAccept {
		httpReq.Header.Set("Accept-Encoding", "identity")
	}
	if len(req.Body) > 0 {
		httpReq.ContentLength = int64(len(req.Body))
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return nil, newError(KindTimeout, "lm15 transport: "+err.Error()+" — this is the client's limit (Timeouts.Connect / Timeouts.Read)").WithCause(err)
		}
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

// idleWatch cancels the request when no read completes within the read
// timeout: a per-operation limit, reset on every byte.
type idleWatch struct {
	mu      sync.Mutex
	timer   *time.Timer
	d       time.Duration
	fire    bool
	stopped bool
}

func newIdleWatch(d time.Duration, cancel context.CancelFunc) *idleWatch {
	w := &idleWatch{d: d}
	w.timer = time.AfterFunc(d, func() {
		w.mu.Lock()
		w.fire = true
		w.mu.Unlock()
		cancel()
	})
	return w
}

func (w *idleWatch) reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.stopped {
		w.timer.Reset(w.d)
	}
}

func (w *idleWatch) stop() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.stopped = true
	w.timer.Stop()
}

func (w *idleWatch) fired() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.fire
}

// decodedBody is the response body: content-decoded (INV-053), read
// timeout per operation, and the connection slot released on Close.
type decodedBody struct {
	io.ReadCloser
	reader  io.Reader
	watch   *idleWatch
	cancel  context.CancelFunc
	release func()
	read    time.Duration
	once    sync.Once
}

// decode wraps the raw body according to Content-Encoding (in reverse
// order of application); an unknown coding is refused before any byte
// reaches a parser.
func (b *decodedBody) decode(resp *TransportResponse) error {
	var reader io.Reader = b.ReadCloser
	encoding := strings.TrimSpace(resp.Header("content-encoding"))
	if encoding == "" {
		b.reader = reader
		return nil
	}
	codings := strings.Split(encoding, ",")
	for i := len(codings) - 1; i >= 0; i-- {
		coding := strings.ToLower(strings.TrimSpace(codings[i]))
		switch coding {
		case "", "identity":
		case "gzip", "x-gzip":
			reader = &gzipMembers{src: reader}
		case "deflate":
			reader = &deflateReader{src: reader}
		default:
			return newError(KindTransport, fmt.Sprintf("lm15 transport: the reply is Content-Encoding %q, which this client cannot decode (gzip, x-gzip and deflate are); encoded bytes are never handed to a parser (INV-053)", coding))
		}
	}
	b.reader = reader
	return nil
}

func (b *decodedBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	if n > 0 {
		b.watch.reset()
	}
	if err != nil && err != io.EOF && b.watch.fired() {
		return n, readTimeoutError(b.read)
	}
	return n, err
}

func (b *decodedBody) Close() error {
	var err error
	b.once.Do(func() {
		b.watch.stop()
		err = b.ReadCloser.Close()
		b.cancel()
		b.release()
	})
	return err
}

// gzipMembers reads a gzip body as the sequence of members RFC 1952 §2.2
// defines; every member has its own trailer, and bytes after a member feed
// the next one. Zero padding between members is accepted; other trailing
// bytes must form a valid member.
type gzipMembers struct {
	src    io.Reader
	buf    *bufferedReader
	member *gzip.Reader
	done   bool
}

func (g *gzipMembers) Read(p []byte) (int, error) {
	if g.buf == nil {
		g.buf = &bufferedReader{r: g.src}
	}
	for {
		if g.done {
			return 0, io.EOF
		}
		if g.member == nil {
			// Skip zero padding; EOF here ends the body.
			for {
				b, err := g.buf.peekByte()
				if err == io.EOF {
					g.done = true
					return 0, io.EOF
				}
				if err != nil {
					return 0, newError(KindTransport, "lm15 transport: gzip body: "+err.Error()).WithCause(err)
				}
				if b != 0 {
					break
				}
				g.buf.discard(1)
			}
			member, err := gzip.NewReader(g.buf)
			if err != nil {
				return 0, newError(KindTransport, "lm15 transport: gzip body: "+err.Error()+" (bytes after a member must form a valid member)").WithCause(err)
			}
			member.Multistream(false)
			g.member = member
		}
		n, err := g.member.Read(p)
		if err == io.EOF {
			g.member = nil
			if n > 0 {
				return n, nil
			}
			continue
		}
		if err != nil {
			return n, newError(KindTransport, "lm15 transport: gzip body: "+err.Error()).WithCause(err)
		}
		return n, nil
	}
}

// deflateReader reads an HTTP deflate body: RFC 1950 zlib-wrapped when the
// two-byte header is valid, else the legacy raw-deflate convention. A
// valid wrapper header wins in ambiguous cases; the choice is made once
// from two bytes, never re-tried after output.
type deflateReader struct {
	src   io.Reader
	inner io.Reader
}

func (d *deflateReader) Read(p []byte) (int, error) {
	if d.inner == nil {
		head := make([]byte, 2)
		n, err := io.ReadFull(d.src, head)
		if err != nil && err != io.ErrUnexpectedEOF {
			if err == io.EOF {
				return 0, io.EOF
			}
			return 0, newError(KindTransport, "lm15 transport: deflate body: "+err.Error()).WithCause(err)
		}
		prefix := io.MultiReader(bytes.NewReader(head[:n]), d.src)
		if n == 2 && head[0]&0x0f == 8 && (uint16(head[0])<<8|uint16(head[1]))%31 == 0 {
			z, zerr := zlib.NewReader(prefix)
			if zerr != nil {
				return 0, newError(KindTransport, "lm15 transport: deflate body: "+zerr.Error()).WithCause(zerr)
			}
			d.inner = z
		} else {
			d.inner = flate.NewReader(prefix)
		}
	}
	n, err := d.inner.Read(p)
	if err != nil && err != io.EOF {
		return n, newError(KindTransport, "lm15 transport: deflate body: "+err.Error()).WithCause(err)
	}
	return n, err
}

// bufferedReader is a one-byte look-ahead over a reader.
type bufferedReader struct {
	r    io.Reader
	head []byte
}

func (b *bufferedReader) peekByte() (byte, error) {
	if len(b.head) == 0 {
		one := make([]byte, 1)
		n, err := b.r.Read(one)
		if n == 0 {
			if err == nil {
				err = io.ErrNoProgress
			}
			return 0, err
		}
		b.head = one[:n]
	}
	return b.head[0], nil
}

func (b *bufferedReader) discard(n int) { b.head = b.head[n:] }

func (b *bufferedReader) Read(p []byte) (int, error) {
	if len(b.head) > 0 {
		n := copy(p, b.head)
		b.head = b.head[n:]
		return n, nil
	}
	return b.r.Read(p)
}

// ─── Buffered responses ──────────────────────────────────────────────

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
