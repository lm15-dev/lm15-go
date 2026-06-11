// lm15-vet — the lm15-go vet shim.
//
// Speaks the newline-delimited JSON protocol of lm15-contract/harness/
// PROTOCOL.md on stdin/stdout: one JSON request per line, one reply per
// line, same order. Never touches the network.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	lm15 "github.com/lm15-dev/lm15-go"
)

type jmap = map[string]any

func errReply(id any, errType, message string) jmap {
	return jmap{"id": id, "ok": false, "error": jmap{"type": errType, "message": message}}
}

func handle(msg jmap) jmap {
	id := msg["id"]
	op, _ := msg["op"].(string)
	switch op {
	case "capabilities":
		return jmap{"id": id, "ok": true, "result": jmap{
			"language":     "go",
			"ops":          []string{"capabilities", "serde_roundtrip", "validate", "surface_dump", "normalize_error", "build_request", "parse_response", "replay_stream"},
			"impl_version": lm15.ImplVersion,
		}}
	case "serde_roundtrip", "validate":
		kind, _ := msg["kind"].(string)
		value, ok := msg["value"].(jmap)
		if !ok {
			return errReply(id, "TypeError", "value must be a JSON object")
		}
		out, err := lm15.SerdeRoundtrip(kind, value)
		if err != nil {
			return errReply(id, lm15.ErrTypeName(err), err.Error())
		}
		if op == "validate" {
			return jmap{"id": id, "ok": true, "result": jmap{"ok": true, "normalized": out}}
		}
		return jmap{"id": id, "ok": true, "result": jmap{"value": out}}
	case "surface_dump":
		return jmap{"id": id, "ok": true, "result": lm15.SurfaceDump()}
	case "normalize_error":
		provider, _ := msg["provider"].(string)
		bodyText, _ := msg["body_text"].(string)
		status := 0
		if n, ok := msg["status"].(json.Number); ok {
			if i, err := n.Int64(); err == nil {
				status = int(i)
			}
		}
		lmErr, err := lm15.NormalizeError(provider, status, bodyText)
		if err != nil {
			return errReply(id, lm15.ErrTypeName(err), err.Error())
		}
		return jmap{"id": id, "ok": true, "result": lm15.NormalizedErrorMap(lmErr)}
	case "build_request":
		provider, _ := msg["provider"].(string)
		canonical, ok := msg["canonical_request"].(jmap)
		if !ok {
			return errReply(id, "TypeError", "canonical_request must be a JSON object")
		}
		stream, _ := msg["stream"].(bool)
		apiKey, _ := msg["api_key"].(string)
		baseURL, _ := msg["base_url"].(string)
		out, err := lm15.BuildRequestMap(provider, canonical, stream, apiKey, baseURL)
		if err != nil {
			return errReply(id, lm15.ErrTypeName(err), err.Error())
		}
		return jmap{"id": id, "ok": true, "result": out}
	case "parse_response":
		provider, _ := msg["provider"].(string)
		canonical, ok := msg["canonical_request"].(jmap)
		if !ok {
			return errReply(id, "TypeError", "canonical_request must be a JSON object")
		}
		bodyB64, _ := msg["body_b64"].(string)
		status := 0
		if n, ok := msg["status"].(json.Number); ok {
			if i, err := n.Int64(); err == nil {
				status = int(i)
			}
		}
		out, err := lm15.ParseResponseMap(provider, canonical, status, bodyB64)
		if err != nil {
			return errReply(id, lm15.ErrTypeName(err), err.Error())
		}
		return jmap{"id": id, "ok": true, "result": out}
	case "replay_stream":
		provider, _ := msg["provider"].(string)
		canonical, ok := msg["canonical_request"].(jmap)
		if !ok {
			return errReply(id, "TypeError", "canonical_request must be a JSON object")
		}
		bodyB64, _ := msg["body_b64"].(string)
		out, err := lm15.ReplayStreamMap(provider, canonical, bodyB64)
		if err != nil {
			return errReply(id, lm15.ErrTypeName(err), err.Error())
		}
		return jmap{"id": id, "ok": true, "result": out}
	}
	return errReply(id, "ValueError", fmt.Sprintf("unknown op: %s", op))
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1<<20), 64<<20)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	enc := json.NewEncoder(out)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var msg any
		dec := json.NewDecoder(strings.NewReader(line))
		dec.UseNumber()
		var reply jmap
		if err := dec.Decode(&msg); err != nil {
			reply = errReply(nil, "JSONDecodeError", err.Error())
		} else if m, ok := msg.(jmap); !ok {
			reply = errReply(nil, "ValueError", "request must be a JSON object")
		} else {
			reply = handle(m)
		}
		if err := enc.Encode(reply); err != nil {
			fmt.Fprintln(os.Stderr, "encode error:", err)
			os.Exit(1)
		}
		out.Flush()
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "stdin error:", err)
		os.Exit(1)
	}
}
