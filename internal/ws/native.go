//go:build !js

package ws

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"crypto/tls"
	"encoding/base64"
	"encoding/binary"
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

const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

type nativeConn struct {
	conn     net.Conn
	reader   *bufio.Reader
	writeMu  sync.Mutex
	readMu   sync.Mutex
	closed   bool
	closeMu  sync.Mutex
	fragment []byte
	fragOp   byte
}

// Dial opens a WebSocket connection with the given headers.
func Dial(ctx context.Context, rawURL string, headers [][2]string) (Conn, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}
	host := u.Host
	tlsOn := false
	switch u.Scheme {
	case "wss", "https":
		tlsOn = true
		if u.Port() == "" {
			host += ":443"
		}
	case "ws", "http":
		if u.Port() == "" {
			host += ":80"
		}
	default:
		return nil, fmt.Errorf("websocket: unsupported scheme %q", u.Scheme)
	}
	dialer := &net.Dialer{Timeout: 30 * time.Second}
	var conn net.Conn
	if tlsOn {
		conn, err = (&tls.Dialer{NetDialer: dialer, Config: &tls.Config{ServerName: u.Hostname()}}).DialContext(ctx, "tcp", host)
	} else {
		conn, err = dialer.DialContext(ctx, "tcp", host)
	}
	if err != nil {
		return nil, err
	}
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	} else {
		_ = conn.SetDeadline(time.Now().Add(30 * time.Second))
	}
	keyBytes := make([]byte, 16)
	if _, err := rand.Read(keyBytes); err != nil {
		conn.Close()
		return nil, err
	}
	key := base64.StdEncoding.EncodeToString(keyBytes)
	path := u.EscapedPath()
	if path == "" {
		path = "/"
	}
	if u.RawQuery != "" {
		path += "?" + u.RawQuery
	}
	var req strings.Builder
	fmt.Fprintf(&req, "GET %s HTTP/1.1\r\n", path)
	fmt.Fprintf(&req, "Host: %s\r\n", u.Host)
	req.WriteString("Upgrade: websocket\r\nConnection: Upgrade\r\n")
	fmt.Fprintf(&req, "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n", key)
	for _, h := range headers {
		lk := strings.ToLower(h[0])
		if lk == "host" || lk == "upgrade" || lk == "connection" || strings.HasPrefix(lk, "sec-websocket-") {
			continue
		}
		fmt.Fprintf(&req, "%s: %s\r\n", h[0], h[1])
	}
	req.WriteString("\r\n")
	if _, err := io.WriteString(conn, req.String()); err != nil {
		conn.Close()
		return nil, err
	}
	reader := bufio.NewReaderSize(conn, 64*1024)
	resp, err := http.ReadResponse(reader, &http.Request{Method: "GET"})
	if err != nil {
		conn.Close()
		return nil, err
	}
	if resp.StatusCode != http.StatusSwitchingProtocols {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		conn.Close()
		return nil, fmt.Errorf("websocket: handshake failed: %s %s", resp.Status, strings.TrimSpace(string(body)))
	}
	sum := sha1.Sum([]byte(key + wsGUID))
	if resp.Header.Get("Sec-WebSocket-Accept") != base64.StdEncoding.EncodeToString(sum[:]) {
		conn.Close()
		return nil, errors.New("websocket: bad Sec-WebSocket-Accept")
	}
	_ = conn.SetDeadline(time.Time{})
	return &nativeConn{conn: conn, reader: reader}, nil
}

func (c *nativeConn) writeFrame(opcode byte, payload []byte) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	header := []byte{0x80 | opcode}
	n := len(payload)
	switch {
	case n < 126:
		header = append(header, 0x80|byte(n))
	case n <= 0xFFFF:
		header = append(header, 0x80|126)
		header = binary.BigEndian.AppendUint16(header, uint16(n))
	default:
		header = append(header, 0x80|127)
		header = binary.BigEndian.AppendUint64(header, uint64(n))
	}
	var mask [4]byte
	if _, err := rand.Read(mask[:]); err != nil {
		return err
	}
	header = append(header, mask[:]...)
	masked := make([]byte, n)
	for i, b := range payload {
		masked[i] = b ^ mask[i%4]
	}
	if _, err := c.conn.Write(append(header, masked...)); err != nil {
		return err
	}
	return nil
}

func (c *nativeConn) Send(ctx context.Context, data []byte) error {
	if c.isClosed() {
		return ErrClosed
	}
	if deadline, ok := ctx.Deadline(); ok {
		_ = c.conn.SetWriteDeadline(deadline)
	} else {
		_ = c.conn.SetWriteDeadline(time.Time{})
	}
	return c.writeFrame(0x1, data)
}

func (c *nativeConn) readFrame() (opcode byte, fin bool, payload []byte, err error) {
	var head [2]byte
	if _, err = io.ReadFull(c.reader, head[:]); err != nil {
		return 0, false, nil, err
	}
	fin = head[0]&0x80 != 0
	opcode = head[0] & 0x0F
	masked := head[1]&0x80 != 0
	length := uint64(head[1] & 0x7F)
	switch length {
	case 126:
		var ext [2]byte
		if _, err = io.ReadFull(c.reader, ext[:]); err != nil {
			return 0, false, nil, err
		}
		length = uint64(binary.BigEndian.Uint16(ext[:]))
	case 127:
		var ext [8]byte
		if _, err = io.ReadFull(c.reader, ext[:]); err != nil {
			return 0, false, nil, err
		}
		length = binary.BigEndian.Uint64(ext[:])
	}
	if length > 64<<20 {
		return 0, false, nil, errors.New("websocket: frame too large")
	}
	var mask [4]byte
	if masked {
		if _, err = io.ReadFull(c.reader, mask[:]); err != nil {
			return 0, false, nil, err
		}
	}
	payload = make([]byte, length)
	if _, err = io.ReadFull(c.reader, payload); err != nil {
		return 0, false, nil, err
	}
	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return opcode, fin, payload, nil
}

func (c *nativeConn) Recv(ctx context.Context) ([]byte, error) {
	c.readMu.Lock()
	defer c.readMu.Unlock()
	if c.isClosed() {
		return nil, ErrClosed
	}
	if deadline, ok := ctx.Deadline(); ok {
		_ = c.conn.SetReadDeadline(deadline)
	} else {
		_ = c.conn.SetReadDeadline(time.Time{})
	}
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			_ = c.conn.SetReadDeadline(time.Now())
		case <-stop:
		}
	}()
	for {
		opcode, fin, payload, err := c.readFrame()
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, err
		}
		switch opcode {
		case 0x0: // continuation
			c.fragment = append(c.fragment, payload...)
			if fin {
				out := c.fragment
				c.fragment = nil
				return out, nil
			}
		case 0x1, 0x2:
			if !fin {
				c.fragment = append([]byte(nil), payload...)
				c.fragOp = opcode
				continue
			}
			return payload, nil
		case 0x8:
			c.markClosed()
			_ = c.writeFrame(0x8, payload)
			c.conn.Close()
			return nil, ErrClosed
		case 0x9:
			if err := c.writeFrame(0xA, payload); err != nil {
				return nil, err
			}
		case 0xA:
		default:
			return nil, fmt.Errorf("websocket: unknown opcode %d", opcode)
		}
	}
}

func (c *nativeConn) isClosed() bool {
	c.closeMu.Lock()
	defer c.closeMu.Unlock()
	return c.closed
}

func (c *nativeConn) markClosed() {
	c.closeMu.Lock()
	c.closed = true
	c.closeMu.Unlock()
}

func (c *nativeConn) Close() error {
	if c.isClosed() {
		return nil
	}
	c.markClosed()
	_ = c.conn.SetWriteDeadline(time.Now().Add(2 * time.Second))
	_ = c.writeFrame(0x8, binary.BigEndian.AppendUint16(nil, 1000))
	return c.conn.Close()
}
