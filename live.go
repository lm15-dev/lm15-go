package lm15

import (
	"context"
	"encoding/base64"
	"sync"

	"github.com/lm15-dev/lm15-go/internal/ws"
)

// wsConn is the socket the live sessions speak over.
type wsConn = ws.Conn

func dialWebSocket(ctx context.Context, url string, headers [][2]string) (wsConn, error) {
	conn, err := ws.Dial(ctx, url, headers)
	if err != nil {
		return nil, newError(KindTransport, err.Error()).WithCause(err)
	}
	return conn, nil
}

// LiveSession is a realtime session: send typed client events, receive
// typed server events. The codec is the contract; the session mechanics
// are this port's idiom.
type LiveSession interface {
	Send(ctx context.Context, event LiveClientEvent) error
	SendTurn(ctx context.Context, parts ...Part) error
	SendAudio(ctx context.Context, data []byte, mediaType string) error
	SendImage(ctx context.Context, data []byte, mediaType string) error
	SendText(ctx context.Context, text string) error
	SendToolResult(ctx context.Context, callID string, output string) error
	Interrupt(ctx context.Context) error
	EndAudio(ctx context.Context) error
	Recv(ctx context.Context) (LiveServerEvent, error)
	// Turn reads until the turn's boundary (turn_end / interrupted / error)
	// or a tool_call the caller must answer (LIVE-1).
	Turn(ctx context.Context) (*Turn, error)
	Close() error
}

type webSocketLiveSession struct {
	conn    wsConn
	encode  func(LiveClientEvent) ([]JSONObject, error)
	decode  func([]byte) ([]LiveServerEvent, error)
	pending []LiveServerEvent
	sendMu  sync.Mutex
	recvMu  sync.Mutex
	closed  bool
}

func newWebSocketLiveSession(conn wsConn, encode func(LiveClientEvent) ([]JSONObject, error), decode func([]byte) ([]LiveServerEvent, error)) *webSocketLiveSession {
	return &webSocketLiveSession{conn: conn, encode: encode, decode: decode}
}

func (s *webSocketLiveSession) Send(ctx context.Context, event LiveClientEvent) error {
	if err := event.Validate(); err != nil {
		return err
	}
	frames, err := s.encode(event)
	if err != nil {
		return err
	}
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	if s.closed {
		return newError(KindTransport, "live session is closed")
	}
	for _, f := range frames {
		if err := s.conn.Send(ctx, mustJSON(f)); err != nil {
			return newError(KindTransport, err.Error()).WithCause(err)
		}
	}
	return nil
}

func (s *webSocketLiveSession) SendTurn(ctx context.Context, parts ...Part) error {
	return s.Send(ctx, NewLiveClientTurnEvent(parts...))
}

func (s *webSocketLiveSession) SendAudio(ctx context.Context, data []byte, mediaType string) error {
	return s.Send(ctx, LiveClientAudioEvent{Data: base64.StdEncoding.EncodeToString(data), MediaType: mediaType})
}

func (s *webSocketLiveSession) SendImage(ctx context.Context, data []byte, mediaType string) error {
	return s.Send(ctx, LiveClientImageEvent{Data: base64.StdEncoding.EncodeToString(data), MediaType: mediaType})
}

func (s *webSocketLiveSession) SendText(ctx context.Context, text string) error {
	return s.Send(ctx, LiveClientTextEvent{Text: text})
}

func (s *webSocketLiveSession) SendToolResult(ctx context.Context, callID, output string) error {
	return s.Send(ctx, LiveClientToolResultEvent{ID: callID, Content: []Part{Text(output)}})
}

func (s *webSocketLiveSession) Interrupt(ctx context.Context) error {
	return s.Send(ctx, LiveClientInterruptEvent{})
}

func (s *webSocketLiveSession) EndAudio(ctx context.Context) error {
	return s.Send(ctx, LiveClientEndAudioEvent{})
}

func (s *webSocketLiveSession) Recv(ctx context.Context) (LiveServerEvent, error) {
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	for {
		if len(s.pending) > 0 {
			ev := s.pending[0]
			s.pending = s.pending[1:]
			return ev, nil
		}
		if s.closed {
			return nil, newError(KindTransport, "live session is closed")
		}
		raw, err := s.conn.Recv(ctx)
		if err != nil {
			return nil, newError(KindTransport, err.Error()).WithCause(err)
		}
		events, err := s.decode(raw)
		if err != nil {
			return nil, err
		}
		s.pending = append(s.pending, events...)
	}
}

func (s *webSocketLiveSession) Turn(ctx context.Context) (*Turn, error) {
	var events []LiveServerEvent
	for {
		ev, err := s.Recv(ctx)
		if err != nil {
			return nil, err
		}
		events = append(events, ev)
		switch ev.Type() {
		case "turn_end", "interrupted", "error", "tool_call":
			return MaterializeTurn(events), nil
		}
	}
}

func (s *webSocketLiveSession) Close() error {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.conn.Close()
}

// Turn is one materialized live turn (LIVE-1, LIVE-2).
type Turn struct {
	EndedBy        string // turn_end | interrupted | error | tool_call | incomplete
	Text           string
	Audio          []byte
	AudioMediaType string
	ToolCalls      []ToolCallInfo
	Usage          *Usage
	Error          *ErrorDetail
	Events         []LiveServerEvent
}

// OK reports whether the turn ended normally.
func (t *Turn) OK() bool { return t.EndedBy == "turn_end" }

// SumUsage is the field-wise sum; absent on either side stays absent (INV-029).
func SumUsage(acc *Usage, more Usage) Usage {
	if acc == nil {
		return more
	}
	add := func(a, b *int) *int {
		if a == nil || b == nil {
			return nil
		}
		s := *a + *b
		return &s
	}
	return Usage{
		InputTokens: add(acc.InputTokens, more.InputTokens), OutputTokens: add(acc.OutputTokens, more.OutputTokens),
		TotalTokens: add(acc.TotalTokens, more.TotalTokens), CacheReadTokens: add(acc.CacheReadTokens, more.CacheReadTokens),
		CacheWriteTokens: add(acc.CacheWriteTokens, more.CacheWriteTokens), ReasoningTokens: add(acc.ReasoningTokens, more.ReasoningTokens),
		InputAudioTokens: add(acc.InputAudioTokens, more.InputAudioTokens), OutputAudioTokens: add(acc.OutputAudioTokens, more.OutputAudioTokens),
	}
}

// MaterializeTurn collects events into a Turn.
func MaterializeTurn(events []LiveServerEvent) *Turn {
	t := &Turn{Events: events, EndedBy: "incomplete"}
	for _, ev := range events {
		switch e := ev.(type) {
		case LiveServerTextEvent:
			t.Text += e.Text
		case LiveServerAudioEvent:
			if decoded, err := base64.StdEncoding.DecodeString(e.Data); err == nil {
				t.Audio = append(t.Audio, decoded...)
			}
			if t.AudioMediaType == "" && e.MediaType != "" {
				t.AudioMediaType = e.MediaType
			}
		case LiveServerToolCallEvent:
			t.ToolCalls = append(t.ToolCalls, ToolCallInfo{ID: e.ID, Name: e.Name, Input: e.Input})
		case LiveServerTurnEndEvent:
			u := SumUsage(t.Usage, e.Usage)
			t.Usage = &u
		case LiveServerUsageEvent:
			u := SumUsage(t.Usage, e.Usage)
			t.Usage = &u
		case LiveServerErrorEvent:
			err := e.Error
			t.Error = &err
		}
	}
	if len(events) > 0 {
		switch last := events[len(events)-1].Type(); last {
		case "turn_end", "interrupted", "error", "tool_call":
			t.EndedBy = last
		}
	}
	return t
}
