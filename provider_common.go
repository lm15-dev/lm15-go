package lm15

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// Shared adapter helpers (reference lm15/providers/common.py).

// EffortThinkingBudgets is the one effort→budget grading table (MAP-7 rule 3).
var EffortThinkingBudgets = map[string]int{
	"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 32768,
}

var openaiFileReadiness = map[string]string{
	"uploaded": "pending", "pending": "pending", "error": "failed", "failed": "failed", "processed": "ready",
}

// openaiFileReadinessOf folds an OpenAI-shaped file status (D6).
func openaiFileReadinessOf(status any) string {
	s, ok := status.(string)
	if !ok {
		return "ready"
	}
	if r, ok := openaiFileReadiness[s]; ok {
		return r
	}
	return "ready"
}

// partsToText renders text-bearing parts for a text-only wire field. A media
// part RAISES before any wire (MAP-10 rule 2).
func partsToText(parts []Part, provider, where string) (string, error) {
	if where == "" {
		where = "a text-only wire field"
	}
	var out []string
	for _, p := range parts {
		if IsMediaPart(p) {
			head := ""
			if provider != "" {
				head = provider + ": "
			}
			return "", UnsupportedFeature(provider, "messages[*].parts["+p.Type()+"]", "%sa %s part cannot reach %s, which takes text only; no text rendering of a media part is made (MAP-10)", head, p.Type(), where)
		}
		switch x := p.(type) {
		case TextPart:
			out = append(out, x.Text)
		case DataPart:
			out = append(out, DataPartText(x))
		case ThinkingPart:
			if x.Text != "" {
				out = append(out, x.Text)
			}
		case CitationPart:
			var bits []string
			for _, b := range []string{x.Title, x.URL, x.Text} {
				if b != "" {
					bits = append(bits, b)
				}
			}
			if len(bits) > 0 {
				out = append(out, strings.Join(bits, " — "))
			}
		}
	}
	return strings.Join(out, "\n"), nil
}

// systemText renders a system prompt to text.
func systemText(s *SystemPrompt, provider string) (string, error) {
	if s == nil {
		return "", nil
	}
	if s.IsText() {
		return s.Text(), nil
	}
	return partsToText(s.Parts(), provider, "")
}

func mediaDataURI(m Media) (string, error) {
	b64, err := m.Base64()
	if err != nil {
		return "", err
	}
	return "data:" + m.MediaType + ";base64," + b64, nil
}

// ─── MAP-10: tool-result media policy ────────────────────────────────

var toolResultMediaAdmits = map[string]map[string]bool{
	"native": {"image": true, "document": true},
	"images": {"image": true},
	"reject": {},
}

var mediaDoors = map[string]string{
	"image":    "the OpenAI Responses, Anthropic Messages and Gemini dialects (and the xai/moonshotai/zai chat presets)",
	"document": "the OpenAI Responses, Anthropic Messages and Gemini dialects",
}

// noMessageSlot is MAP-10 for message parts: the cells where a dialect has no
// slot at all. Found 2026-09-24 (lm15-contract
// changes/2026-09-24-message-media.md): the builders turned such a part into
// an empty text block or dropped it, silently. lm15-rs refused; this is the
// same preflight.
var noMessageSlot = map[string]func(role, kind string) bool{
	// The Messages API has image and document blocks only, in either role.
	"anthropic": func(_, kind string) bool { return kind == "audio" || kind == "video" || kind == "binary" },
	// Assistant content is output text (and refusals) on both OpenAI wires.
	"openai":      func(role, _ string) bool { return role == "assistant" },
	"openai_chat": func(role, _ string) bool { return role == "assistant" },
}

// checkMessageMedia raises before any wire when a message holds a media part
// the dialect has no content slot for in that role (MAP-10).
func checkMessageMedia(messages []Message, dialect, provider string) error {
	gap, ok := noMessageSlot[dialect]
	if !ok {
		return nil
	}
	for i, m := range messages {
		for j, p := range m.Parts {
			if IsMediaPart(p) && gap(m.Role, p.Type()) {
				return UnsupportedFeature(provider, fmt.Sprintf("messages[%d].parts[%d]", i, j),
					"%s: messages[%d].parts[%d]: the program depends on this %s %s part; no native %s content slot carries it (MAP-10)",
					provider, i, j, m.Role, p.Type(), dialect)
			}
		}
	}
	return nil
}

func checkToolResultMedia(provider string, part ToolResultPart, policy, wire string) error {
	admits := toolResultMediaAdmits[policy]
	for _, p := range part.Content {
		if IsMediaPart(p) && !admits[p.Type()] {
			why := fmt.Sprintf("this server carries images but not %s parts in a tool result", p.Type())
			if policy == "reject" {
				why = "this server takes text-only tool results"
			}
			door, ok := mediaDoors[p.Type()]
			if !ok {
				door = "no lm15 door yet"
			}
			return UnsupportedFeature(provider, "messages[*].tool_result["+part.ID+"].content["+p.Type()+"]",
				"%s: a %s part in tool_result %q cannot reach %s — %s (compat tool_result_media=%q, measured: lm15-contract/research/tool-result-content/). Carried natively by %s; or render the part to text yourself before building the tool result (MAP-10)",
				provider, p.Type(), part.ID, wire, why, policy, door)
		}
	}
	return nil
}

func toolResultErrorText(part ToolResultPart, text string) string {
	if part.IsError {
		return "[error] " + text
	}
	return text
}

func hasMediaParts(parts []Part) bool {
	for _, p := range parts {
		if IsMediaPart(p) {
			return true
		}
	}
	return false
}

// pathID percent-encodes a provider id for a URL path (MAP-11).
func pathID(value string, resourceName bool) string {
	var b strings.Builder
	for i := 0; i < len(value); i++ {
		c := value[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '-', c == '.', c == '_', c == '~':
			b.WriteByte(c)
		case c == '/' && resourceName:
			b.WriteByte(c)
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

// quoteSafe percent-encodes like urllib.parse.quote(value, safe=safe).
func quoteSafe(value, safe string) string {
	var b strings.Builder
	for i := 0; i < len(value); i++ {
		c := value[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '-', c == '.', c == '_', c == '~':
			b.WriteByte(c)
		case strings.IndexByte(safe, c) >= 0:
			b.WriteByte(c)
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

var fractionRe = regexp.MustCompile(`\.(\d{6})\d+`)

// isoUTC normalizes an epoch or ISO-8601 timestamp to YYYY-MM-DDTHH:MM:SSZ ("" when unparseable).
func isoUTC(value any) string {
	switch x := value.(type) {
	case nil, bool:
		return ""
	case string:
		text := strings.TrimSpace(x)
		if text == "" {
			return ""
		}
		text = fractionRe.ReplaceAllString(text, ".$1")
		t, err := ParseRFC3339(text)
		if err != nil {
			return ""
		}
		return FormatRFC3339(t)
	default:
		f, err := jsonFloat64(value, "")
		if err != nil {
			return ""
		}
		sec, frac := math.Modf(f)
		t := time.Unix(int64(sec), int64(frac*1e9)).UTC()
		return FormatRFC3339(t)
	}
}

func randomHex(n int) string {
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return strings.Repeat("0", n*2)
	}
	return hex.EncodeToString(buf)
}

// multipartFile is one file part of a multipart/form-data body.
type multipartFile struct {
	Field       string
	Filename    string
	ContentType string
	Data        []byte
}

// multipartFormBody builds a multipart/form-data body; returns (content type, body).
func multipartFormBody(fields [][2]string, files []multipartFile) (string, []byte) {
	boundary := "lm15-" + randomHex(16)
	var b strings.Builder
	for _, f := range fields {
		b.WriteString("--" + boundary + "\r\n")
		b.WriteString("Content-Disposition: form-data; name=\"" + f[0] + "\"\r\n\r\n")
		b.WriteString(f[1] + "\r\n")
	}
	body := []byte(b.String())
	for _, f := range files {
		safe := strings.ReplaceAll(f.Filename, "\"", "%22")
		body = append(body, []byte("--"+boundary+"\r\n")...)
		body = append(body, []byte("Content-Disposition: form-data; name=\""+f.Field+"\"; filename=\""+safe+"\"\r\n")...)
		body = append(body, []byte("Content-Type: "+f.ContentType+"\r\n\r\n")...)
		body = append(body, f.Data...)
		body = append(body, []byte("\r\n")...)
	}
	body = append(body, []byte("--"+boundary+"--\r\n")...)
	return "multipart/form-data; boundary=" + boundary, body
}

// multipartRelatedBody builds a multipart/related body (Gemini upload).
func multipartRelatedBody(metadata JSONObject, mediaType string, data []byte) (string, []byte) {
	boundary := "lm15-" + randomHex(16)
	var body []byte
	body = append(body, []byte("--"+boundary+"\r\n")...)
	body = append(body, []byte("Content-Type: application/json; charset=UTF-8\r\n\r\n")...)
	body = append(body, mustJSON(metadata)...)
	body = append(body, []byte("\r\n")...)
	body = append(body, []byte("--"+boundary+"\r\n")...)
	body = append(body, []byte("Content-Type: "+mediaType+"\r\n\r\n")...)
	body = append(body, data...)
	body = append(body, []byte("\r\n")...)
	body = append(body, []byte("--"+boundary+"--\r\n")...)
	return "multipart/related; boundary=" + boundary, body
}

// modelInfosFromEntries maps list-models entries to ModelInfo (verbatim entry in origin.provider_data).
func modelInfosFromEntries(entries any, provider, apiFamily string, idOf func(map[string]any) string) []ModelInfo {
	list, ok := entries.([]any)
	if !ok {
		return nil
	}
	var out []ModelInfo
	for _, e := range list {
		entry, ok := e.(map[string]any)
		if !ok {
			continue
		}
		id := idOf(entry)
		if id == "" {
			continue
		}
		out = append(out, ModelInfo{ID: id, Provider: provider, APIFamily: apiFamily, Origin: ModelOrigin{Type: "provider", ProviderData: entry}})
	}
	return out
}

// partToOpenAIInput maps a prompt part to a Responses input block (MAP-10).
func partToOpenAIInput(p Part, provider string) (JSONObject, error) {
	switch x := p.(type) {
	case TextPart:
		return JSONObject{"type": "input_text", "text": x.Text}, nil
	case ImagePart:
		if x.FileID != "" {
			return JSONObject{"type": "input_image", "file_id": x.FileID}, nil
		}
		src := x.URL
		if src == "" {
			uri, err := mediaDataURI(x.Media)
			if err != nil {
				return nil, err
			}
			src = uri
		}
		out := JSONObject{"type": "input_image", "image_url": src}
		if x.Detail != "" {
			out["detail"] = x.Detail
		}
		return out, nil
	case AudioPart:
		if x.URL != "" {
			return JSONObject{"type": "input_audio", "audio_url": x.URL}, nil
		}
		if x.FileID != "" {
			return JSONObject{"type": "input_audio", "file_id": x.FileID}, nil
		}
		media := x.MediaType
		if media == "" {
			media = "audio/wav"
		}
		if i := strings.Index(media, "/"); i >= 0 {
			media = media[i+1:]
		}
		if media == "mpeg" || media == "mp3" {
			media = "mp3"
		}
		b64, err := x.Base64()
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "input_audio", "audio": b64, "format": media}, nil
	case DocumentPart, BinaryPart:
		m, _ := MediaOf(p)
		if m.URL != "" {
			return JSONObject{"type": "input_file", "file_url": m.URL}, nil
		}
		if m.FileID != "" {
			return JSONObject{"type": "input_file", "file_id": m.FileID}, nil
		}
		mt := m.MediaType
		if mt == "" {
			mt = "application/octet-stream"
		}
		ext := mt
		if i := strings.Index(ext, "/"); i >= 0 {
			ext = ext[i+1:]
		}
		ext, _, _ = strings.Cut(ext, "+")
		if ext == "" {
			ext = "bin"
		}
		uri, err := mediaDataURI(m)
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "input_file", "filename": "file." + ext, "file_data": uri}, nil
	case VideoPart:
		if x.URL != "" {
			return JSONObject{"type": "input_video", "video_url": x.URL}, nil
		}
		if x.FileID != "" {
			return JSONObject{"type": "input_video", "file_id": x.FileID}, nil
		}
		uri, err := mediaDataURI(x.Media)
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "input_video", "video_data": uri}, nil
	case DataPart:
		// 2026-09-19 D3: a data part on a text wire is its compact JSON.
		return JSONObject{"type": "input_text", "text": DataPartText(x)}, nil
	case CitationPart, ThinkingPart:
		text, err := partsToText([]Part{p}, provider, "")
		if err != nil {
			return nil, err
		}
		return JSONObject{"type": "input_text", "text": text}, nil
	}
	head := ""
	if provider != "" {
		head = provider + ": "
	}
	return nil, UnsupportedFeature(provider, "messages[*].parts["+p.Type()+"]", "%sa %s part has no input block on the Responses wire (MAP-10)", head, p.Type())
}

// toolResultOutputOpenAI renders function_call_output.output (MAP-10).
func toolResultOutputOpenAI(provider string, part ToolResultPart, policy string) (any, error) {
	if err := checkToolResultMedia(provider, part, policy, "function_call_output"); err != nil {
		return nil, err
	}
	if !hasMediaParts(part.Content) {
		text, err := partsToText(part.Content, provider, "function_call_output")
		if err != nil {
			return nil, err
		}
		return toolResultErrorText(part, text), nil
	}
	var blocks []JSONObject
	for _, p := range part.Content {
		block, err := partToOpenAIInput(p, provider)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	if part.IsError {
		found := false
		for _, b := range blocks {
			if b["type"] == "input_text" {
				b["text"] = "[error] " + wireStr(b["text"])
				found = true
				break
			}
		}
		if !found {
			blocks = append([]JSONObject{{"type": "input_text", "text": "[error]"}}, blocks...)
		}
	}
	return toAnyList(blocks, func(b JSONObject) any { return b }), nil
}

// anthropicSource maps a media part to an Anthropic source block.
func anthropicSource(m Media) (JSONObject, error) {
	if m.URL != "" {
		return JSONObject{"type": "url", "url": m.URL}, nil
	}
	if m.FileID != "" {
		return JSONObject{"type": "file", "file_id": m.FileID}, nil
	}
	b64, err := m.Base64()
	if err != nil {
		return nil, valueErrorf("media part has no usable source")
	}
	return JSONObject{"type": "base64", "media_type": m.MediaType, "data": b64}, nil
}

// openaiTokenLogprobs maps OpenAI-style logprob entries to TokenLogprobs.
func openaiTokenLogprobs(entries any) []TokenLogprob {
	list, ok := entries.([]any)
	if !ok {
		return nil
	}
	var out []TokenLogprob
	for _, e := range list {
		entry, ok := e.(map[string]any)
		if !ok {
			continue
		}
		if _, has := entry["token"]; !has {
			continue
		}
		if _, has := entry["logprob"]; !has {
			continue
		}
		var top []TopLogprob
		for _, a := range wireList(entry["top_logprobs"]) {
			alt, ok := a.(map[string]any)
			if !ok {
				continue
			}
			if _, has := alt["token"]; !has {
				continue
			}
			if _, has := alt["logprob"]; !has {
				continue
			}
			top = append(top, TopLogprob{Token: wireStr(alt["token"]), Logprob: wireFloat(alt["logprob"], 0), Bytes: intList(alt["bytes"])})
		}
		out = append(out, TokenLogprob{Token: wireStr(entry["token"]), Logprob: wireFloat(entry["logprob"], 0), Bytes: intList(entry["bytes"]), Top: top})
	}
	return out
}

func intList(v any) []int {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int, 0, len(list))
	for _, item := range list {
		out = append(out, wireInt(item, 0))
	}
	return out
}

// unnamedToolCallError is MAP-9 on the complete path.
func unnamedToolCallError(provider, path string) *Error {
	e := newError(KindProvider, fmt.Sprintf("%s: %s is a tool call with no name; lm15 does not guess which tool the model meant (MAP-9)", provider, path))
	e.Provider = provider
	return e
}

// parseJSONObject parses provider-emitted arguments leniently.
func parseJSONObject(v any) JSONObject {
	switch x := v.(type) {
	case map[string]any:
		return x
	case string:
		if x == "" {
			return JSONObject{}
		}
		parsed, err := DecodeJSON([]byte(x))
		if err != nil {
			return JSONObject{"partial_json": x}
		}
		if obj, ok := parsed.(map[string]any); ok {
			return obj
		}
		return JSONObject{"value": parsed}
	}
	return JSONObject{}
}

// parseJSONBestEffort is the stream assembler's lenient argument parse.
func parseJSONBestEffort(raw string) JSONObject {
	if raw == "" {
		return JSONObject{}
	}
	parsed, err := DecodeJSON([]byte(raw))
	if err != nil {
		return JSONObject{"partial_json": raw}
	}
	if obj, ok := parsed.(map[string]any); ok {
		return obj
	}
	return JSONObject{"value": parsed}
}

// nonJSONReplyError is INV-054: a 2xx whose body is not JSON is a
// ProviderError (code provider) carrying the status, the content type, the
// first 200 bytes and the request id — never ServerError (bound to 5xx),
// never retried by lm15 (the request may have been served and billed).
func nonJSONReplyError(provider string, resp *HTTPResponse, cause error) *Error {
	contentType := strings.TrimSpace(strings.SplitN(resp.Header("content-type"), ";", 2)[0])
	if contentType == "" {
		contentType = "unknown"
	}
	preview := resp.Body
	if len(preview) > 200 {
		preview = preview[:200]
	}
	e := providerErrorf(KindProvider, provider, nil, fmt.Sprintf("%s: the reply (HTTP %d, %s) is not JSON: %q", provider, resp.Status, contentType, string(preview)))
	e.Code = CodeProvider
	e.Status = resp.Status
	e.RequestID = resp.Header("x-request-id")
	return e.WithCause(cause)
}

// jsonBody decodes a 2xx reply as a JSON object, refusing a non-JSON body
// as the provider fault it is (INV-054).
func (c *lmCore) jsonBody(resp *HTTPResponse) (JSONObject, error) {
	data, err := resp.JSON()
	if err != nil {
		return nil, c.replyError(resp, err)
	}
	return data, nil
}

// splitURL separates a URL into its query-less form and decoded params
// (the vet protocol's build_request shape).
func splitURL(raw string) (string, map[string]string) {
	u, err := url.Parse(raw)
	if err != nil {
		return raw, map[string]string{}
	}
	params := map[string]string{}
	for _, pair := range strings.Split(u.RawQuery, "&") {
		if pair == "" {
			continue
		}
		k, v, _ := strings.Cut(pair, "=")
		k, _ = url.QueryUnescape(k)
		v, _ = url.QueryUnescape(v)
		params[k] = v
	}
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), params
}

func contentTypeOf(headers [][2]string) string {
	for _, h := range headers {
		if strings.EqualFold(h[0], "content-type") {
			ct, _, _ := strings.Cut(h[1], ";")
			return strings.TrimSpace(ct)
		}
	}
	return ""
}

var _ = json.Marshal
