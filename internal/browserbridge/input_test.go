package browserbridge

import (
	"context"
	"strings"
	"testing"

	lm15 "github.com/lm15-dev/lm15-go"
)

func TestBuildRejectsLocalFilesAndFixtureCredentials(t *testing.T) {
	msg := input("openai-chat")
	msg["canonical_request"].(lm15.JSONObject)["messages"] = []any{lm15.JSONObject{"role": "user", "parts": []any{lm15.JSONObject{"type": "image", "path": "/not-a-browser-input.png", "media_type": "image/png"}}}}
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
