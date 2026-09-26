# lm15-go

The Go port of lm15: one canonical request/response model over every
provider the [lm15-contract](https://github.com/lm15-dev/lm15-contract)
names. Zero dependencies (standard library only). Builds for Linux, macOS,
Windows and `GOOS=js GOARCH=wasm` (the browser).

The contract commit this port is written against is in `CONTRACT_PIN`
(2026-09-25). At that commit the port passes all 1,492 of the contract's
cases.

## Versions

There is no released version yet. `v1.0.0` was tagged by mistake on
2026-06-11, from an early prototype, and is **retracted**: `go get` skips it
and warns anyone who pinned it. Until the first release candidate
(`v1.1.0-rc.1`), `go get github.com/lm15-dev/lm15-go` installs the current
`main` as an untagged pseudo-version. See [RELEASING.md](RELEASING.md).

## Browser runtime

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

## Status

**Implemented; the shared harness passes every direction.** Every module of `playbooks/port.md` is
written (1 types+serde, 2 errors, 3a core auth, 3b cloud chains, 4 the
four chat dialects plus `typesafe`, 4b ingest, 5 response/stream
assembly, 5c router, 6 models, 7 files/batch/cache, 8 generation/video,
9 live), and the 2026-09-14 → 2026-09-20 amendments are in: MAP-13
adaptations and `Plan`, MAP-14 judgments and `DataPart`, the `typesafe`
provider and the token-trie driver, client-side stop with score
preservation (`logprobs_complete`), bounded live-turn collection
(`collection_limit`), rate-limit diagnostics, the connection budget,
reply faults (INV-053..055), named cloud credentials with provenance,
endpoint overrides, and a JWT string as bearer.

Harness, from `lm15-contract` (pin: `CONTRACT_PIN`):

```bash
go build -o bin/lm15-vet ./cmd/lm15-vet     # harness/shims.json runs ./bin/lm15-vet
cd ../lm15-contract && python3 harness/check.py --shim go --direction all
```

Last run (2026-09-25): request 398/398, response 308/308, stream 40/40,
error 90/90, serde 129/129, auth 43/43, token 43/43, models 36/36, live
24/24, files 48/48, batch 41/41, generation 20/20, video 27/27, cache
11/11, router 22/22, ingest 169/169, managed (sign-in) 43/43. The request
and serde directions include the opaque key-order check (INV-002).
Live traffic: `go run ./examples/live_smoke` (keys from the environment;
`-managed` adds the saved sign-ins, `-control` the sorted-schema control)
sends real requests and writes receipts; the last run is
[receipts/2026-09-26-live-smoke](receipts/2026-09-26-live-smoke/SUMMARY.md).

The shared consumer vectors `consumer/live-collection-limits.json`,
`errors/diagnostic-headers.json` and `auth/named-credentials.json` pass
natively (`go test ./...` for the first two; the third through the vet
shim's `explain_auth`).

Sign in once, use everywhere: `Auth`, `Connect`, `BoundClient` and
`RouterConfig.Auth` are the managed authentication lm15-python, lm15-ts and
lm15-rs have — same rules, same store file, graded by the same contract runs
and by mixed-language runs on one store (`login_*.go`,
[managed login](docs/managed-login.md)).

| Module | Where |
|---|---|
| canonical types, invariants, serde | `types_*.go`, `serde.go`, `vocab.go`, `json.go` |
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

## Quick start

```go
router := lm15.NewRouter() // keys from the environment (AUTH-1)
req := &lm15.Request{
    Model:    "groq:openai/gpt-oss-20b", // or "claude-haiku-4-5", "gpt-4.1-mini"
    Messages: []lm15.Message{lm15.UserMessage("hi")},
    Config:   lm15.Config{MaxTokens: lm15.I(100)},
}
resp, err := router.Complete(ctx, req)
fmt.Println(resp.TextOr(""))

// Streamed: text as it arrives, then the same Response Complete returns.
rs := lm15.NewResponseStream(router.Stream(ctx, req), req)
for text, err := range rs.Text() { /* ... */ }
resp, err = rs.Response()

// Direct provider, explicit key.
lm, err := lm15.NewAnthropicLM(lm15.WithAPIKey(os.Getenv("MY_KEY")))
```

Tool loop: run the function, answer with `lm15.ToolMessage(call.ID, result)`,
call `Complete` again.

JSON objects keep their order. Tool parameters, `ResponseFormat`,
`Extensions`, tool-call input and `ProviderData` are `lm15.JSONObject`, a
list of key/value members: a model fills structured output in the order
the schema lists its properties, so write them in the order you want them
answered.

```go
schema := lm15.JSONObject{
    lm15.KV("type", "object"),
    lm15.KV("properties", lm15.JSONObject{
        lm15.KV("reasoning", lm15.JSONObject{lm15.KV("type", "string")}), // written first
        lm15.KV("answer", lm15.JSONObject{lm15.KV("type", "string")}),
    }),
    lm15.KV("required", []any{"reasoning", "answer"}),
}
// Or keep the schema as JSON text; the order written is the order sent:
//   schema, err := lm15.DecodeJSONObject([]byte(`{"type": "object", ...}`))

req.Config.ResponseFormat = lm15.JSONObject{
    lm15.KV("type", "json_schema"), lm15.KV("name", "verdict"), lm15.KV("schema", schema)}

v, ok := resp.ProviderData.Lookup("id")        // read: Get, Lookup, Has, Keys
for key, value := range schema.All() { fmt.Println(key, value) } // in order
schema.Set("title", "Verdict")                 // write: Set (keeps a key's place), Delete
```

`Set` and `Delete` copy before they write, so an object you passed to lm15,
or got back from it, never changes through another copy; to change a nested
object, change it and `Set` it back. `lm15.ObjectFromMap(m)` converts a Go
map (sorted keys: a map has no order). `lm15.KV` keeps `go vet` quiet;
`{Key: "type", Value: "object"}` works too.

Adaptations (MAP-13): change the model string and the program keeps
working; what the wire got that differs from what was asked is on the
response, never printed.

```go
resp, _ := router.Complete(ctx, req)
for _, a := range resp.Adaptations { fmt.Println(a.Field, a.Action, a.Reason) }
plan, err := router.Plan(req)            // the same record, no network, no key
lm, _ := lm15.NewAnthropicLM(lm15.WithAdaptations("refuse")) // the old strictness
```

Judgments (MAP-14): declared keys in, a distribution out.

```go
quality, _ := lm15.Score("How good is this wine?", lm15.ScoreLevel{Name: "poor"}, lm15.ScoreLevel{Name: "great"})
style, _ := lm15.Choice("Dominant style?", lm15.Options("fruit", "oak", "mineral")...)
format, _ := lm15.Judgments("judgments", true,
    lm15.JudgmentProperty{Name: "quality", Schema: quality},
    lm15.JudgmentProperty{Name: "style", Schema: style},
    lm15.JudgmentProperty{Name: "ageing", Schema: lm15.YesNo("Will it improve with age?")})
req := &lm15.Request{Model: "jev-latest", Messages: []lm15.Message{lm15.UserMessage(note)},
    Config: lm15.Config{ResponseFormat: format, Probabilities: "required"}}
resp, _ := router.Complete(ctx, req)   // typesafe, or a vLLM server that scores tokens
fmt.Println(resp.Data(), resp.Probabilities()["style"])
expected, _ := resp.Expected("quality")
```

Cloud identity (AUTH-1): name one identity, or read who was picked.

```go
router, _ := lm15.NewRouterWithConfig(lm15.RouterConfig{
    Credentials: map[string]string{"azure": "platform"},        // that rung only, never the chain
    BaseURLs:    map[string]string{"azure": "https://acct.services.ai.azure.com"}, // the door appends /openai/v1
    Timeouts:    &lm15.Timeouts{Read: 30 * time.Minute},        // one pool, shared by every LM
})
```

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

## Not exercised, stated

Nothing has been run against a live server. The harness (offline, above)
and the three native vector files are the evidence. The token-trie driver
(`provider_openai_chat_trie.go`), the connection budget's idle read
timeout, the content decoders and `TurnView` over a real socket are
written to the contract and the reference, not measured against a server;
the trie driver's harness direction is deferred contract-side as well.
