//go:build js

package ws

import (
	"context"
	"errors"
	"sync"
	"syscall/js"
)

// browserConn wraps the browser's WebSocket. Custom headers cannot be set
// on a browser WebSocket (the platform forbids it); providers that need a
// header-carried credential in the browser must use a query-key door or a
// proxy — stated in the README.
type browserConn struct {
	ws       js.Value
	messages chan []byte
	errs     chan error
	closed   chan struct{}
	once     sync.Once
	funcs    []js.Func
}

// Dial opens a browser WebSocket. headers are ignored (unsupported by browsers).
func Dial(ctx context.Context, rawURL string, headers [][2]string) (Conn, error) {
	ctor := js.Global().Get("WebSocket")
	if ctor.IsUndefined() {
		return nil, errors.New("websocket: no WebSocket in this runtime")
	}
	ws := ctor.New(rawURL)
	ws.Set("binaryType", "arraybuffer")
	c := &browserConn{ws: ws, messages: make(chan []byte, 256), errs: make(chan error, 1), closed: make(chan struct{})}
	opened := make(chan struct{})
	onOpen := js.FuncOf(func(this js.Value, args []js.Value) any { close(opened); return nil })
	onMessage := js.FuncOf(func(this js.Value, args []js.Value) any {
		data := args[0].Get("data")
		var payload []byte
		if data.Type() == js.TypeString {
			payload = []byte(data.String())
		} else {
			buf := js.Global().Get("Uint8Array").New(data)
			payload = make([]byte, buf.Get("length").Int())
			js.CopyBytesToGo(payload, buf)
		}
		select {
		case c.messages <- payload:
		case <-c.closed:
		}
		return nil
	})
	onError := js.FuncOf(func(this js.Value, args []js.Value) any {
		select {
		case c.errs <- errors.New("websocket: error event"):
		default:
		}
		return nil
	})
	onClose := js.FuncOf(func(this js.Value, args []js.Value) any { c.markClosed(); return nil })
	c.funcs = []js.Func{onOpen, onMessage, onError, onClose}
	ws.Set("onopen", onOpen)
	ws.Set("onmessage", onMessage)
	ws.Set("onerror", onError)
	ws.Set("onclose", onClose)
	select {
	case <-opened:
		return c, nil
	case err := <-c.errs:
		c.Close()
		return nil, err
	case <-c.closed:
		return nil, ErrClosed
	case <-ctx.Done():
		c.Close()
		return nil, ctx.Err()
	}
}

func (c *browserConn) markClosed() {
	c.once.Do(func() { close(c.closed) })
}

func (c *browserConn) Send(ctx context.Context, data []byte) error {
	select {
	case <-c.closed:
		return ErrClosed
	default:
	}
	c.ws.Call("send", string(data))
	return nil
}

func (c *browserConn) Recv(ctx context.Context) ([]byte, error) {
	select {
	case m := <-c.messages:
		return m, nil
	case err := <-c.errs:
		return nil, err
	case <-c.closed:
		select {
		case m := <-c.messages:
			return m, nil
		default:
		}
		return nil, ErrClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *browserConn) Close() error {
	c.markClosed()
	c.ws.Call("close")
	for _, f := range c.funcs {
		f.Release()
	}
	return nil
}
