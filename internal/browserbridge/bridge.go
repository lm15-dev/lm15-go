// Package browserbridge is the JSON boundary for the browser's Go runtime.
// All wire building, provider I/O, event parsing and assembly stay in the SDK.
package browserbridge

import (
	"context"
	"encoding/base64"
	"fmt"
	"iter"

	lm15 "github.com/lm15-dev/lm15-go"
)

func invalid(message string) error { return lm15.Errorf(lm15.KindInvalidRequest, "%s", message) }

// Failure preserves the SDK's formatted diagnostics and bounded header evidence.
func Failure(err error) lm15.JSONObject {
	d := lm15.JSONObject{lm15.KV("name", "Error"), lm15.KV("code", "invalid_request"), lm15.KV("message", err.Error())}
	if e := lm15.AsError(err); e != nil {
		d.Set("name", e.ClassName())
		d.Set("code", e.Code)
		if e.Status != 0 {
			d.Set("status", e.Status)
		}
		if e.ProviderCode != "" {
			d.Set("provider_code", e.ProviderCode)
		}
		if e.Feature != "" {
			d.Set("feature", e.Feature)
		}
		h := lm15.JSONObject{}
		if e.RequestID != "" {
			h.Set("request_id", e.RequestID)
		}
		if e.RetryAfter != nil {
			h.Set("retry_after", *e.RetryAfter)
		}
		if len(e.RateLimitHeaders) > 0 {
			h.Set("rate_limit_headers", e.RateLimitHeaders.Clone())
		}
		if len(h) > 0 {
			d.Set("http_response", h)
		}
	} else if name := lm15.NativeErrorKind(err); name != "" {
		d.Set("name", name)
	}
	return lm15.JSONObject{lm15.KV("error", d)}
}

// JSON returns a reply string, including when serialization itself fails.
func JSON(value any) string {
	b, err := lm15.EncodeJSON(value)
	if err != nil {
		b, _ = lm15.EncodeJSON(Failure(invalid("cannot serialize bridge reply")))
	}
	return string(b)
}

// Call never panics across the host boundary. emit is synchronous, allowing the
// host to display an event before the next provider read; it may abort the call.
func Call(ctx context.Context, op, input string, transport lm15.Transport, emit func(string) error) (reply string) {
	defer func() {
		if recover() != nil {
			reply = JSON(Failure(invalid("invalid input or callback failure in Go bridge")))
		}
	}()
	out, err := call(ctx, op, input, transport, emit)
	if ctx.Err() != nil {
		return JSON(lm15.JSONObject{lm15.KV("error", lm15.JSONObject{lm15.KV("name", "AbortError"), lm15.KV("code", "transport"), lm15.KV("message", "request aborted")})})
	}
	if err != nil {
		return JSON(Failure(err))
	}
	return JSON(out)
}

func call(ctx context.Context, op, input string, transport lm15.Transport, emit func(string) error) (lm15.JSONObject, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if op == "version" {
		return lm15.JSONObject{lm15.KV("version", lm15.Version), lm15.KV("language", "go")}, nil
	}
	if op != "build_request" && op != "complete" && op != "stream" {
		return nil, invalid("unknown operation: " + op)
	}
	msg, err := lm15.DecodeJSONObject([]byte(input))
	if err != nil {
		return nil, err
	}
	provider, ok := msg.Get("provider").(string)
	if !ok || provider == "" {
		return nil, invalid("provider must be a non-empty string")
	}
	key, ok := msg.Get("api_key").(string)
	if !ok || key == "" {
		return nil, lm15.NotConfiguredErrorf(provider, nil, "", "pass an explicit non-empty api_key (a placeholder for keyless endpoints)")
	}
	baseURL := ""
	if v := msg.Get("base_url"); v != nil {
		if baseURL, ok = v.(string); !ok {
			return nil, invalid("base_url must be a string")
		}
	}
	settings := map[string]string{}
	if v := msg.Get("settings"); v != nil {
		obj, ok := v.(lm15.JSONObject)
		if !ok {
			return nil, invalid("settings must be an object of strings")
		}
		for k, v := range obj.All() {
			s, ok := v.(string)
			if !ok {
				return nil, invalid(fmt.Sprintf("settings.%s must be a string", k))
			}
			settings[k] = s
		}
	}
	stream := false
	if v := msg.Get("stream"); v != nil {
		if stream, ok = v.(bool); !ok {
			return nil, invalid("stream must be a boolean")
		}
	}
	canonical, ok := msg.Get("canonical_request").(lm15.JSONObject)
	if !ok {
		return nil, invalid("canonical_request must be an object")
	}
	req, err := lm15.RequestFromDict(canonical)
	if err != nil {
		return nil, err
	}
	// Filesystem paths are not browser inputs. Reject rather than attempting I/O
	// in an otherwise pure build; data, URLs and provider file IDs remain valid.
	if hasLocalPath(req) {
		return nil, invalid("browser requests cannot read local media paths; use data, url or file_id")
	}
	lm, err := lm15.AdapterForProvider(provider, key, baseURL, settings, nil, lm15.WithTransport(transport))
	if err != nil {
		return nil, err
	}
	defer lm.Close()
	if op == "build_request" {
		wire, adaptations, err := lm.Build(req, stream)
		if err != nil {
			return nil, err
		}
		out := lm15.NormalizeTransportRequest(wire)
		out.Set("body_b64", base64.StdEncoding.EncodeToString(wire.Body))
		if len(adaptations) > 0 {
			records := make([]any, 0, len(adaptations))
			for _, a := range adaptations {
				d := lm15.AdaptationToDict(a)
				d.Delete("reason") // same shape as the vet build_request protocol
				records = append(records, d)
			}
			out.Set("adaptations", records)
		}
		return out, nil
	}
	var resp *lm15.Response
	if op == "complete" {
		resp, err = lm.Complete(ctx, req)
	} else {
		events := lm.Stream(ctx, req)
		tee := iter.Seq2[lm15.StreamEvent, error](func(yield func(lm15.StreamEvent, error) bool) {
			for event, err := range events {
				if err == nil && emit != nil {
					if err = emit(JSON(lm15.StreamEventToDict(event))); err != nil {
						yield(nil, err)
						return
					}
				}
				if !yield(event, err) {
					return
				}
			}
		})
		resp, err = lm15.MaterializeResponse(tee, req)
	}
	if err != nil {
		return nil, err
	}
	return lm15.JSONObject{lm15.KV("canonical_response", lm15.ResponseToDict(resp, false))}, nil
}

func hasLocalPath(req *lm15.Request) bool {
	var check func([]lm15.Part) bool
	check = func(parts []lm15.Part) bool {
		for _, part := range parts {
			if media, ok := lm15.MediaOf(part); ok && media.Path != "" {
				return true
			}
			if tool, ok := part.(lm15.ToolResultPart); ok && check(tool.Content) {
				return true
			}
		}
		return false
	}
	if req.System != nil && check(req.System.Parts()) {
		return true
	}
	for _, msg := range req.Messages {
		if check(msg.Parts) {
			return true
		}
	}
	return false
}
