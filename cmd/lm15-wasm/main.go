//go:build js && wasm

// lm15-wasm exposes the actual Go SDK to a browser worker through wasm_exec.js.
package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"syscall/js"

	lm15 "github.com/lm15-dev/lm15-go"
	"github.com/lm15-dev/lm15-go/internal/browserbridge"
)

// Fetch decodes every HTTP content coding before exposing body bytes, including
// deflate/br/zstd, but may retain Content-Encoding. Don't inflate a second time.
// A Transport with a DialContext would bypass Fetch on js/wasm, so use a fresh
// one with ALL dial hooks nil, not DefaultTransport or NewHTTPTransport's pool.
type fetchRoundTripper struct{ inner *http.Transport }

func (t fetchRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.inner.RoundTrip(req)
	if resp != nil {
		resp.Header.Del("Content-Encoding")
		resp.Header.Del("Content-Length")
		resp.ContentLength = -1
		// Go's Fetch RoundTrip observes cancellation only until headers arrive.
		// Its streamReader.Read does not select on context, so explicitly close
		// the ReadableStream on cancellation (including the SDK idle timeout).
		body := &onceBody{ReadCloser: resp.Body}
		stop := context.AfterFunc(req.Context(), func() { _ = body.Close() })
		resp.Body = &cancelBody{onceBody: body, stop: stop}
	}
	return resp, err
}

type onceBody struct {
	io.ReadCloser
	once sync.Once
	err  error
}

func (b *onceBody) Close() error {
	b.once.Do(func() { b.err = b.ReadCloser.Close() })
	return b.err
}

type cancelBody struct {
	*onceBody
	stop func() bool
}

func (b *cancelBody) Close() error {
	b.stop()
	return b.onceBody.Close()
}

type browserTransport struct{ inner *lm15.HTTPTransport }

func (t browserTransport) Do(ctx context.Context, req *lm15.TransportRequest) (*lm15.TransportResponse, error) {
	// Inject only on Anthropic-dialect requests, after the SDK built its wire.
	copy := *req
	copy.Headers = append([][2]string(nil), req.Headers...)
	if copy.Header("anthropic-version") != "" {
		copy.SetHeader("anthropic-dangerous-direct-browser-access", "true")
	}
	return t.inner.Do(ctx, &copy)
}

func newTransport() lm15.Transport {
	return browserTransport{&lm15.HTTPTransport{
		Client: &http.Client{Transport: fetchRoundTripper{&http.Transport{}}},
	}}
}

func invalid(message string) string {
	return browserbridge.JSON(browserbridge.Failure(lm15.Errorf(lm15.KindInvalidRequest, "%s", message)))
}

func run(args []js.Value, resolve js.Value) {
	reply := invalid("invalid input or JavaScript callback failure")
	// This guard also catches host getter/listener exceptions. Never unwind into
	// the parked main goroutine or leave a Promise pending on invalid input.
	defer func() { _ = recover(); resolve.Invoke(reply) }()
	if len(args) < 2 || args[0].Type() != js.TypeString || args[1].Type() != js.TypeString {
		reply = invalid("call requires an operation and JSON input string")
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if len(args) > 2 && !args[2].IsNull() && !args[2].IsUndefined() {
		signal := args[2]
		if signal.Type() != js.TypeObject || signal.Get("addEventListener").Type() != js.TypeFunction || signal.Get("removeEventListener").Type() != js.TypeFunction || signal.Get("aborted").Type() != js.TypeBoolean {
			reply = invalid("signal must be an AbortSignal")
			return
		}
		listener := js.FuncOf(func(js.Value, []js.Value) any { cancel(); return nil })
		defer listener.Release()
		// Register the cleanup before calling into user-supplied JavaScript.
		defer signal.Call("removeEventListener", "abort", listener)
		signal.Call("addEventListener", "abort", listener)
		if signal.Get("aborted").Bool() {
			cancel()
		}
	}
	var emit func(string) error
	if len(args) > 3 && !args[3].IsUndefined() && !args[3].IsNull() {
		callback := args[3]
		if callback.Type() != js.TypeFunction {
			reply = invalid("onEvent must be a function")
			return
		}
		// callback belongs to JS, not js.FuncOf: retaining it only for this call
		// needs no Release. Exceptions end the iterator, closing the HTTP body.
		emit = func(event string) (err error) {
			defer func() {
				if recover() != nil {
					err = errors.New("onEvent callback failed")
				}
			}()
			callback.Invoke(event)
			return nil
		}
	}
	reply = browserbridge.Call(ctx, args[0].String(), args[1].String(), newTransport(), emit)
}

func main() {
	call := js.FuncOf(func(_ js.Value, args []js.Value) any {
		// Promise executors run synchronously, but I/O must run on a new goroutine
		// so Fetch and JS event callbacks can resume the Go scheduler.
		executor := js.FuncOf(func(_ js.Value, callbacks []js.Value) any {
			go run(args, callbacks[0])
			return nil
		})
		defer executor.Release()
		return js.Global().Get("Promise").New(executor)
	})
	js.Global().Set("lm15Go", map[string]any{"call": call})
	select {} // call lives for the worker's lifetime
}
