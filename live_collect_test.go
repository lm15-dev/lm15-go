package lm15

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

type scriptedSource struct {
	events []LiveServerEvent
	reads  int
}

func (s *scriptedSource) Recv(context.Context) (LiveServerEvent, error) {
	if s.reads >= len(s.events) {
		return nil, newError(KindTransport, "no more events")
	}
	ev := s.events[s.reads]
	s.reads++
	return ev, nil
}

// The shared consumer vectors (lm15-contract/consumer/live-collection-limits.json):
// exact byte and event boundaries, terminal admission, count-before-read,
// empty events, Unicode, DEL, audio, tool input.
func TestLiveCollectionLimitVectors(t *testing.T) {
	raw, err := os.ReadFile("../lm15-contract/consumer/live-collection-limits.json")
	if err != nil {
		t.Skip("contract corpus not present:", err)
	}
	var doc struct {
		Cases []struct {
			ID     string `json:"id"`
			Limits struct {
				MaxBytes  int `json:"max_bytes"`
				MaxEvents int `json:"max_events"`
			} `json:"limits"`
			Events []JSONObject `json:"events"`
			Expect struct {
				Accepted      int     `json:"accepted"`
				Reads         int     `json:"reads"`
				RetainedBytes int     `json:"retained_bytes"`
				Limit         *string `json:"limit"`
				RejectedIndex *int    `json:"rejected_index"`
			} `json:"expect"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	for _, c := range doc.Cases {
		var events []LiveServerEvent
		for _, e := range c.Events {
			// Re-decode through the canonical reader (numbers as json.Number).
			b, _ := json.Marshal(e)
			obj, _ := DecodeJSONObject(b)
			ev, err := LiveServerEventFromDict(obj)
			if err != nil {
				t.Fatalf("%s: %v", c.ID, err)
			}
			events = append(events, ev)
		}
		src := &scriptedSource{events: events}
		view, err := NewTurnView(src, TurnLimits{MaxBytes: c.Limits.MaxBytes, MaxEvents: c.Limits.MaxEvents})
		if err != nil {
			t.Fatal(err)
		}
		var failure *Error
		for {
			_, ok, err := view.Next(context.Background())
			if err != nil {
				failure = AsError(err)
				break
			}
			if !ok {
				break
			}
		}
		if len(view.Events()) != c.Expect.Accepted {
			t.Errorf("%s: accepted %d want %d", c.ID, len(view.Events()), c.Expect.Accepted)
		}
		if src.reads != c.Expect.Reads {
			t.Errorf("%s: reads %d want %d", c.ID, src.reads, c.Expect.Reads)
		}
		if view.RetainedBytes() != c.Expect.RetainedBytes {
			t.Errorf("%s: retained_bytes %d want %d", c.ID, view.RetainedBytes(), c.Expect.RetainedBytes)
		}
		switch {
		case c.Expect.Limit == nil && failure != nil:
			t.Errorf("%s: unexpected failure %v", c.ID, failure)
		case c.Expect.Limit != nil && (failure == nil || failure.Limit != *c.Expect.Limit || failure.Code != CodeCollectionLimit):
			t.Errorf("%s: failure %v want limit %s", c.ID, failure, *c.Expect.Limit)
		}
		if c.Expect.RejectedIndex != nil {
			if failure == nil || failure.RejectedEvent == nil || !jsonEqual(LiveServerEventToDict(failure.RejectedEvent), LiveServerEventToDict(events[*c.Expect.RejectedIndex])) {
				t.Errorf("%s: rejected event not carried", c.ID)
			}
		} else if failure != nil && failure.RejectedEvent != nil {
			t.Errorf("%s: a count overflow carries no rejected event", c.ID)
		}
		if failure != nil {
			// Sealed: the same failure again, no further read.
			before := src.reads
			if _, _, err := view.Next(context.Background()); AsError(err) != failure || src.reads != before {
				t.Errorf("%s: the view is not sealed", c.ID)
			}
		}
	}
}
