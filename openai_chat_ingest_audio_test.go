package lm15

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// MAP-12 rule 4 (amended 2026-09-29): each input_audio format reads as its
// true media type; an unknown format is malformed; a builder with no audio
// slot refuses at send (MAP-10).

func audioBody(format string) JSONObject {
	body, err := DecodeJSONObject([]byte(`{"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": [
		{"type": "text", "text": "Transcribe."},
		{"type": "input_audio", "input_audio": {"data": "T2dnUw==", "format": "` + format + `"}}]}]}`))
	if err != nil {
		panic(err)
	}
	return body
}

func audioOf(t *testing.T, req *Request) AudioPart {
	t.Helper()
	switch p := req.Messages[0].Parts[1].(type) {
	case AudioPart:
		return p
	case *AudioPart:
		return *p
	default:
		t.Fatalf("second part is %T, not audio", p)
		return AudioPart{}
	}
}

func TestInputAudioReadsItsTrueMediaType(t *testing.T) {
	for format, mediaType := range map[string]string{
		"wav": "audio/wav", "mp3": "audio/mpeg", "mpeg": "audio/mpeg", "ogg": "audio/ogg", "opus": "audio/opus",
		"flac": "audio/flac", "aac": "audio/aac", "aiff": "audio/aiff", "webm": "audio/webm",
	} {
		req, err := RequestFromOpenAIChat(audioBody(format), "")
		if err != nil {
			t.Fatalf("%s: %v", format, err)
		}
		if got := audioOf(t, req); got.MediaType != mediaType || got.Data != "T2dnUw==" {
			t.Errorf("%s: got %q %q, want %q", format, got.MediaType, got.Data, mediaType)
		}
	}
}

func TestInputAudioUnknownFormatIsMalformed(t *testing.T) {
	_, err := RequestFromOpenAIChat(audioBody("midi"), "")
	if err == nil || !strings.Contains(err.Error(), "input_audio.format must be one of") {
		t.Fatalf("midi: want a malformed-format error, got %v", err)
	}
}

func TestOggAudioReachesGeminiInlineAndTheChatWireRefusesIt(t *testing.T) {
	req, err := RequestFromOpenAIChat(audioBody("ogg"), "")
	if err != nil {
		t.Fatal(err)
	}
	gemini, err := NewGeminiLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	wire, err := gemini.BuildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	sent, err := DecodeJSONObject(wire.Body)
	if err != nil {
		t.Fatal(err)
	}
	part := sent.Get("contents").([]any)[0].(JSONObject).Get("parts").([]any)[1]
	want, _ := DecodeJSONObject([]byte(`{"inlineData": {"mimeType": "audio/ogg", "data": "T2dnUw=="}}`))
	if !reflect.DeepEqual(part, any(want)) {
		t.Fatalf("gemini part: got %v, want %v", part, want)
	}
	chat, err := NewOpenAIChatLM(WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = chat.BuildRequest(req, false)
	var e *Error
	if !errors.As(err, &e) || e.Kind != KindUnsupportedFeature {
		t.Fatalf("chat wire: want UnsupportedFeatureError, got %v", err)
	}
}
