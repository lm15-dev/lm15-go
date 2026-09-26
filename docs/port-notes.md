# Port notes

Details for contributors and for readers who compare lm15 languages: where
each part of the contract lives in this module, and where Go states a
difference from the Python reference. The [README](../README.md) is the
user-facing overview.

## Where things are

| Module | Where |
|---|---|
| canonical types, invariants, serde | `types_*.go`, `serde.go`, `vocab.go`, `json.go` |
| ordered JSON objects (`JSONObject`, `KV`), the decoder and encoder | `object.go`, `json.go` |
| adaptations (MAP-13), `Plan` | `adaptation.go`, `provider_base.go` |
| judgments (MAP-14), `DataPart`, the schema sugar | `judgments.go`, `types_parts.go` |
| client-side stop, score preservation | `stop.go` |
| errors, rate-limit diagnostics | `errors.go` (`*lm15.Error`, `ErrorKind`, `errors.As`), `rate_limits.go` |
| credentials, access policies, presets, registry | `credentials.go`, `features.go`, `access.go`, `compat.go`, `registry.go` |
| auth stores, refresh under lock, PKCE, device code, `Login` | `auth_store.go`, `internal/fslock` |
| managed sign-in (`Auth`, `Connect`, `BoundClient`, the flows, the loopback listener) | `login_*.go` |
| doctor (`ExplainAuth`) | `doctor.go` |
| cloud chains, named credentials, hosts, endpoints, SigV4, RS256 | `cloud_chains.go`, `cloud_hosts.go`, `internal/sigv4`, `internal/rs256` |
| dialects | `provider_openai*.go`, `provider_openai_chat*.go`, `provider_anthropic.go`, `provider_gemini*.go`, `provider_typesafe.go` |
| token-trie judgments (vLLM) | `provider_openai_chat_trie.go` |
| shared drivers (complete, stream, files, batch, cache, video, generation) | `provider_base.go`, `provider_common.go` |
| stream assembly (MAP-3/4/9), `ResponseStream` | `result.go` |
| transport, connection budget, content decoding | `transport.go` |
| router | `router.go` |
| live sessions, bounded turn collection | `live.go`, `live_collect.go`, `internal/ws` |
| vet shim | `vet.go`, `cmd/lm15-vet` |

## The browser runtime (wasm)

The website bridge runs the actual Go SDK, not a JavaScript translation:

```bash
GOOS=js GOARCH=wasm go build -trimpath -ldflags='-s -w' \
  -o /tmp/lm15-go.wasm ./cmd/lm15-wasm
# Use wasm_exec.js from the SAME Go toolchain as the build:
node cmd/lm15-wasm/smoke.mjs "$(go env GOROOT)/lib/wasm/wasm_exec.js" /tmp/lm15-go.wasm
```

Start `go.run(instance)` without awaiting its lifetime promise; wait for
`globalThis.lm15Go`. Its `call(op, inputJSON, signal?, onEvent?)` returns a
Promise of a JSON string. Operations: `version`, `build_request`, `complete`,
`stream`. Input is `{provider, api_key, base_url?, settings?, canonical_request,
stream?}`; pass the key explicitly, or a nonempty placeholder for keyless
endpoints. Custom endpoints use `openai-chat` with `base_url`. Local media
paths are refused; use inline data, URLs or provider file IDs.

`build_request` returns the vet wire shape plus **always** `body_b64`: the
actual SDK bytes (keys in the order the SDK wrote them), not reserialized
JavaScript.
`complete`/`stream` return `{canonical_response}`. Streaming delivers canonical
JSON events immediately through `onEvent` and uses the SDK's stop handling,
adaptations and final assembler. Failures resolve `{error:{name,code,message,
status?,provider_code?,http_response?}}`; messages include SDK diagnostics.
Cancellation resolves `AbortError` and closes the HTTP stream. Host callbacks
and abort listeners are released/detached when the call ends.

The browser-only transport uses Go's Fetch path (nil dial hooks), lets Fetch
handle compression, and injects Anthropic's direct-browser opt-in **after**
SDK request building. Native presets are unchanged. CORS still applies;
servers must expose diagnostic headers for browsers to see them. The smoke
test uses only loopback HTTP, no provider credentials or paid calls. Build and
smoke verified with Go 1.26.7.

## Stated deviations

Each row names the spec line and the reason (port.md rule 8).

- **Optional strings are `""`-as-absent.** Invariants of the form "non-empty
  when present" (INV-016's neighbours on optional fields) are unrepresentable:
  a wire `""` for an optional string reads as absent. Exceptions where an
  empty string is data: `AudioDelta`/`ImageDelta`/`CitationDelta` address
  fields are `*string`, and every required text field is a plain `string`
  whose `""` is emitted. Optional numbers and booleans are pointers
  (`lm15.I`, `lm15.F`, `lm15.B`): `0` and `false` are data.
- **Go maps have no key order.** A Go map placed inside a payload (a
  nested `map[string]any`, `ObjectFromMap`) is accepted and written with
  sorted keys, the way `encoding/json` writes it; build a `JSONObject` when
  the order matters. Two typed fields are Go maps for lookup and are
  written sorted in canonical JSON where the reference keeps insertion
  order: `DataPart.Probabilities` (and `Response.Probabilities()`) and
  `RateLimitHeaders` (net/http hands response headers over as a map, so
  their arrival order is gone before lm15 sees them). Both are typed data,
  compared order-free by the contract (serde-rules.md), not opaque payloads.
- **Numbers:** every wire document is decoded with `UseNumber`, so opaque
  payloads round-trip `1` vs `1.0` verbatim; typed floats always emit a
  fraction (Number rule). User-built Go payloads with `float64(1)` emit `1`
  — Go has no distinct "1.0" value.
- **`Message.tool({call_id: output})`** (INV-025, dict order) has no Go
  equivalent: Go maps are unordered. Use `lm15.ToolMessageParts(...)`.
- **`LiveClientTurnEvent.TurnComplete`** defaults to `true` on the wire and
  in `NewLiveClientTurnEvent`; a struct literal must set it.
- **Live transport in the browser:** browser WebSockets cannot carry custom
  headers, so the OpenAI Realtime door needs a proxy there; Gemini Live
  (query-key) works as-is. Native builds use an in-tree RFC 6455 client
  (`internal/ws`) rather than a dependency — a stated choice: zero
  dependencies over a battle-tested library.
- **File lock on wasm** is an in-process mutex: there is no shared
  filesystem to lock.
- **Deprecated Python surfaces are not ported**: `ProviderProfile`,
  `EndpointProfile`, `from_profile`, and the base-URL compat guess. The
  same facts are `WithCompatPreset` + `WithBaseURL`.
- **`surface_dump`** field names come from struct reflection with a
  snake-case rendering; the type list itself is a table.
- **RS256 JWT claims** are encoded UTF-8 (the reference escapes non-ASCII);
  the pinned vectors are ASCII.
- **Async** is Go's own: `context.Context` everywhere, `iter.Seq2` streams.
- **Adaptations reach builders explicitly.** The reference collects MAP-13
  records through a context variable; here every builder takes the build's
  `*adaptScope` and `adapt` returns the refusal under `"refuse"`.
  `Plan` runs the build with a scope that skips credentials and signing.
- **Timeouts** are Go durations on `lm15.Timeouts` (connect 10 s, read/
  write/pool 600 s, 100 connections); the read timeout is an idle limit
  reset on every byte, enforced by cancelling the request's context.
- **Content decoding** (INV-053) inflates gzip/x-gzip (every member) and
  deflate (zlib-wrapped by its header, else raw); `br`/`zstd` are a
  `TransportError` naming the coding. Browser builds see fetch-decoded
  bodies and cannot check the coding.
- **A lone surrogate** (INV-055) is invalid UTF-8 in a Go string; the
  canonical encoder refuses it (and any other invalid UTF-8) with a
  `ValueError` naming the code point, instead of Go's silent U+FFFD.
- **Bounded live collection** is `TurnView` (`session.TurnView(TurnLimits{...})`);
  `Turn` uses the defaults. The byte charge is computed by a dedicated
  ASCII-JSON sizer (`asciiJSONSize`) because Go's encoder does not use the
  short `\b`/`\f` escapes or escape DEL.
- **`Response.LogprobsIncomplete`** is the field; the wire's
  `logprobs_complete` is its negation, so the zero value is the default.

## What has been run against real servers

`examples/live_smoke` (2026-09-26, `receipts/`): chat, streaming, a
structured-output schema, a tool loop and model listing on every API-key
provider family, the Codex CLI login, and five sign-ins saved in the lm15
store, which Go renewed and used. Not yet measured against a server: the
token-trie judgment driver (`provider_openai_chat_trie.go`), the
connection budget's idle read timeout under a stalled server, and
`TurnView` over a real socket.
