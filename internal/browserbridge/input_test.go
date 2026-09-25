package browserbridge

import (
	"context"
	"strings"
	"testing"

	lm15 "github.com/lm15-dev/lm15-go"
)

func TestBuildRejectsLocalFilesAndFixtureCredentials(t *testing.T) {
	msg := input("openai-chat")
	canonical := msg.Get("canonical_request").(lm15.JSONObject)
	messages, err := lm15.DecodeJSON([]byte(`[{"role": "user", "parts": [{"type": "image", "path": "/not-a-browser-input.png", "media_type": "image/png"}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	canonical.Set("messages", messages)
	msg.Set("canonical_request", canonical)
	reply := Call(context.Background(), "build_request", JSON(msg), nil, nil)
	if !strings.Contains(reply, "cannot read local media paths") {
		t.Fatal(reply)
	}
	// A non-JWT Codex key cannot acquire the vet shim's test-account identity.
	reply = Call(context.Background(), "build_request", JSON(input("openai-codex")), nil, nil)
	if !strings.Contains(reply, `"error"`) || strings.Contains(reply, "test-account") {
		t.Fatal(reply)
	}
}
