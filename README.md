# lm15-go

The Go port of lm15: one canonical request/response model over every
provider the [lm15-contract](https://github.com/lm15-dev/lm15-contract)
names. Zero dependencies (standard library only). Builds for Linux, macOS,
Windows and `GOOS=js GOARCH=wasm` (the browser).

The contract commit this port is written against is in `CONTRACT_PIN`.

## Status

**Implemented, not yet graded.** Every module of `playbooks/port.md` is
written (1 types+serde, 2 errors, 3a core auth, 3b cloud chains, 4 the four
dialects, 4b ingest, 5 response/stream assembly, 5c router, 6 models, 7
files/batch/cache, 8 generation/video, 9 live). The vet shim answers every
op of `harness/PROTOCOL.md`. `go build ./...` and `go vet ./...` pass on all
four targets. **No harness direction has been run yet**; the numbers below
are the surface, not a grade.

```bash
go build -o bin/lm15-vet ./cmd/lm15-vet     # harness/shims.json runs ./bin/lm15-vet
cd ../lm15-contract && python3 harness/check.py --shim go --direction all
```

| Module | Where |
|---|---|
| canonical types, invariants, serde | `types_*.go`, `serde.go`, `vocab.go`, `json.go` |
| errors | `errors.go` (`*lm15.Error`, `ErrorKind`, `errors.As`) |
| credentials, access policies, presets, registry | `credentials.go`, `features.go`, `access.go`, `compat.go`, `registry.go` |
| auth stores, refresh under lock, PKCE, device code, `Login` | `auth_store.go`, `internal/fslock` |
| doctor (`ExplainAuth`) | `doctor.go` |
| cloud chains, hosts, SigV4, RS256 | `cloud_chains.go`, `cloud_hosts.go`, `internal/sigv4`, `internal/rs256` |
| dialects | `provider_openai*.go`, `provider_openai_chat*.go`, `provider_anthropic.go`, `provider_gemini*.go` |
| shared drivers (complete, stream, files, batch, cache, video, generation) | `provider_base.go`, `provider_common.go` |
| stream assembly (MAP-3/4/9), `ResponseStream` | `result.go` |
| router | `router.go` |
| live sessions | `live.go`, `internal/ws` |
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

## Stated deviations

Each row names the spec line and the reason (port.md rule 8).

- **Optional strings are `""`-as-absent.** Invariants of the form "non-empty
  when present" (INV-016's neighbours on optional fields) are unrepresentable:
  a wire `""` for an optional string reads as absent. Exceptions where an
  empty string is data: `AudioDelta`/`ImageDelta`/`CitationDelta` address
  fields are `*string`, and every required text field is a plain `string`
  whose `""` is emitted. Optional numbers and booleans are pointers
  (`lm15.I`, `lm15.F`, `lm15.B`): `0` and `false` are data.
- **No key-order preservation inside opaque payloads.** `JSONObject` is
  `map[string]any`; canonical JSON is compared parsed, so nothing pinned
  changes, but where an opaque object is serialized *into a wire string*
  (tool-call `arguments`) keys come out sorted and UTF-8 where the reference
  emits insertion order and ASCII escapes. No contract case carries such an
  input today; the harness compares those bodies parsed.
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

## Not exercised, stated

Nothing has been run against a live server or the harness in this phase.
The token-refresh wire, the xAI device-code login and the cloud chains are
copied as data from the reference and compile, and that is all that is
claimed.
