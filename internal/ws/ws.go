// Package ws is a minimal WebSocket client (RFC 6455) with no dependencies:
// a client handshake, masked text/binary frames, fragmentation, ping/pong,
// and the close handshake. On js/wasm the browser's WebSocket is used.
package ws

import (
	"context"
	"errors"
)

// Conn is one WebSocket connection.
type Conn interface {
	// Send writes one text message.
	Send(ctx context.Context, data []byte) error
	// Recv reads the next text or binary message.
	Recv(ctx context.Context) ([]byte, error)
	Close() error
}

// ErrClosed is returned after the connection is closed.
var ErrClosed = errors.New("websocket: connection closed")
