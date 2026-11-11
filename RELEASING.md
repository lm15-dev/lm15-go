# Releasing lm15-go

## Version numbers

Go's module proxy keeps every tagged version forever, so a tag cannot be
taken back, only retracted (a `retract` directive in `go.mod`, published in a
later version).

- **`v1.0.0` (2026-06-11) is retracted.** It was tagged from an early
  prototype of the client layer, before the shared lm15 contract existed. It
  is not the lm15 1.0 API.
- **`v1.0.1` (2026-09-24) is retracted too.** It exists only to carry that
  retraction: Go reads retractions from the newest version, and a retraction
  must be published by a version above the one it retracts.
- **The v1 releases were candidates only,** `v1.1.0-rc.1` to `-rc.10`.
  A version number can never be reused, so `v1.0.x` is closed.
- **Major version 2 (2026-11-01):** every lm15 language shares its major
  version, the contract's generation (lm15-contract `playbooks/api-family.md`
  rule 8). Generation 2 is MAP-19. The module path is
  `github.com/lm15-dev/lm15-go/v2`; `v2.0.0-rc.1`, then `v2.0.0`, the first
  stable release (2026-11-01).
  The v1 retractions stay with the v1 module (in `v1.0.1`); the v2 `go.mod`
  carries none, since Go refuses a v1 version in a v2 module's `retract`.

Checked on a private copy before tagging (2026-09-24): with both versions
retracted, `go get` installs `main` as a pseudo-version; a module pinned to
`v1.0.0` sees "(retracted)" and the reason; once `v1.1.0-rc.1` exists,
`go get ...@latest` selects it.

## Releases

| Version | Date | Contract pin | Notes |
|---|---|---|---|
| `v1.1.0-rc.1` | 2026-09-26 | `3763eec` | The first release. 1,583 of 1,583 contract checks. |
| `v1.1.0-rc.2` | 2026-09-26 | `fe5cdf9` | DeepInfra, Together AI, Fireworks AI, Parasail; the Google Cloud pass. 1,788 of 1,788 contract checks. |
| `v1.1.0-rc.3` | 2026-09-30 | `57e33d1` | The Claude Code release as a setting (`client_version`, `LM15_CLAUDE_CODE_VERSION`), its refusal guided; Claude's output ceiling as the default `max_tokens`; `input_audio` formats in ingest. 1,838 of 1,838 contract checks. |
| `v1.1.0-rc.4` | 2026-10-06 | `0f3ea82` | Long streamed replies: no default SSE line or event limit (INV-056); a tool with no description is left off the wire (MAP-17). 1,901 of 1,901 contract checks. |
| `v1.1.0-rc.5` | 2026-10-10 | `880f72c` | A refused key is `KindAuth` on Gemini and xAI, which answer it with HTTP 400 (MAP-18); a key passed as a named credential is never repeated in the refusal (AUTH-1, AUTH-5). 1,904 of 1,904 contract checks. |
| `v2.3.1` | 2026-11-11 | `81b519f` | A conversation that changes provider keeps working (MAP-24, lm15-contract issue #3): a conversation answered by an OpenAI Responses model and continued on Claude failed on every request with `messages: text content blocks must be non-empty`, because OpenAI's hidden reasoning (a `ThinkingPart` with empty text and OpenAI's state) was written to the Messages wire as an empty text block. A part with nothing for the wire it is sent to now writes nothing there, and a message left with nothing is not sent: hidden thinking carrying another dialect's state, an empty `TextPart`, a citation that quotes nothing, an empty system prompt given as parts, on the Messages, generateContent, Responses and Chat Completions wires. Claude's own thinking with display omitted still goes back as a thinking block; a tool result is always sent (an empty output is the empty string). Nothing is recorded in `Response.Adaptations`; the caller's messages are not changed. Not changed, still refused by the provider: thinking signed by another server of the same API family, a tool loop moved to Gemini 3 or DeepSeek between a call and its result, a message whose only text is whitespace on Anthropic. An assistant `DataPart` (a structured answer kept in the history) is sent as its compact JSON on the Responses and Chat Completions wires too; it was dropped there. No API change. 4,269 of 4,269 contract checks. |
| `v2.3.0` | 2026-11-10 | `18b3788` | Tinker (Thinking Machines): providers `tinker` (Chat Completions, alias `thinkingmachines`) and `tinker-anthropic` (Messages), one key; a checkpoint by its `tinker://` path (a URI is one model name); reasoning switched per base family, a checkpoint following its base through `RouterConfig.BaseModels` or `WithBaseModels`; FastAPI `detail` error bodies keep their message (MAP-21); a stop sequence the server returns is cut (MAP-22); a catalog entry naming its base model is a fine-tune (MAP-23). 2,266 of 2,266 contract checks. |
| `v2.2.0` | 2026-11-07 | `9d8e1ce` | Claude shows its thinking when asked (MAP-7 rule 7 amended): a reasoning summary sends `thinking.display: "summarized"`; Opus 4.7, 4.8 and every Claude 5 return empty thinking without it. No API change. 2,140 of 2,140 contract checks. |
| `v2.1.0` | 2026-11-05 | `77f22f4` | A stream's final copy completes its fragments (MAP-20): parallel tool calls on the Codex backend keep their inputs (their arguments arrive only in each call's final copy); a streamed reasoning summary is joined with newlines as on the complete path; `response.incomplete` ends a Responses stream with `length`; Gemini calls sent one per chunk stay separate calls; the realtime completion path counts a call's arguments once and never names a call `tool`. No API change. 2,132 of 2,132 contract checks. |
| `v2.0.0` | 2026-11-01 | `cfc3edc` | The first stable release: v2.0.0-rc.1 marked stable, the same code (the vet shim reports `2.0.0`). The chat core keeps working as documented through v2.x. Released without the fresh sign-in per account flow listed under `v1.1.0-rc.1`; sign-in is provisional, outside that promise. 1,968 of 1,968 contract checks. |
| `v2.0.0-rc.1` | 2026-11-01 | `cfc3edc` | A usage counter means one thing on every provider (MAP-19): `InputTokens` includes cached tokens, `OutputTokens` includes reasoning, parts never exceed their wholes; `InferencePricing.Estimate(usage)` returns `(cost, ok)` and is exact or unknown, with `PricingTier` for long prompts; `ReasoningUsage` compat (xAI counts reasoning beside the completion). Breaking for code that read Anthropic input or Gemini/xAI output; major version 2 like every lm15 language, so the import path gains `/v2`. 1,968 of 1,968 contract checks. |
| `v1.1.0-rc.10` | 2026-10-15 | `6dc1c5c` | Bounding boxes: `BoundingBox(...)` asks where something is in a picture, on every provider (Jev 1.14's Box question on TypeSafe). The scoring driver answers every property it does not measure with one generated call instead of dropping it. 1,921 of 1,921 contract checks. |
| `v1.1.0-rc.9` | 2026-10-15 | `1cbdc12` | A saved cloud login's settings (project, location, region) build the managed route; an explicit settings entry changes only the settings it names. 1,912 of 1,912 contract checks. |
| `v1.1.0-rc.8` | 2026-10-14 | `10645e5` | Jev reads pictures: a PNG or JPEG image part is a reference in the state and its bytes an asset, a message holding an image beside other parts is the ordered list of its parts, and the request carries `TypeSafe-Danger-Zone-Images: true`; a URL, file id, other format or other media is refused before the wire. 1,911 of 1,911 contract checks. |
| `v1.1.0-rc.7` | 2026-10-12 | `88bb560` | The Meta login reaches the Muse Code subscription: requests go to the address the key mint names (an https `meta.ai` host, anything else refused before the key is sent), and the connection's account label names the plan. 1,905 of 1,905 contract checks. |
| `v1.1.0-rc.6` | 2026-10-10 | `8eedbae` | A structured answer reads through `Response.Text` (a `DataPart` answer as its JSON); `ReasoningBudget` and a `Reasoning` with only `ThinkingBudget` fill `Effort` from the MAP-7 table; messages that name the fix. 1,904 of 1,904 contract checks. |

### `v1.1.0-rc.4`: what was checked

- `go vet ./...`, `go test ./...`, `gofmt`, the wasm build (CI on Linux,
  macOS and Windows); first tests for `internal/sse`.
- The contract at the pin: 1,901 of 1,901 checks, including the live case
  `openai.streaming_long_line` (three 1.1 MB lines).
- Over real HTTP from a local server, in 16 KiB chunks: the pinned body, and a
  Gemini-shaped stream with one 30,000,000-character image line (image intact
  by SHA-256, 0.29 s).
- Live, 2026-10-06: `openai:gpt-4.1-mini` streamed through the router with a
  74 KB system prompt (its echo lines are 75 KB): "OK", 18,244 input tokens.
  `v1.1.0-rc.3` refuses the same stream.

### `v1.1.0-rc.3`: what was checked

- `go vet ./...`, `go test ./...`, `gofmt`, the wasm build (CI on Linux,
  macOS and Windows).
- The contract at the pin: 1,838 of 1,838 checks, including the
  claude-code cases that now compare `user-agent`.
- Live, 2026-09-30: `claude-code:claude-opus-5-5` through the router with
  no settings (the default release, 2.1.285, and the default `max_tokens`,
  128000): HTTP 200, `stop`, the `defaulted` record carried.

### `v1.1.0-rc.2`: what was checked

- `go vet ./...`, `go test ./...`, `gofmt`, the wasm build (CI on Linux,
  macOS and Windows).
- The contract at the pin: 1,788 of 1,788 checks.
- Live: the four open-model hosts, 16 of 16 checks
  (`receipts/2026-09-26-inference-hosts-live-smoke`); Google Cloud, the 13
  identity setups of lm15-contract `changes/2026-09-26-vertex-live.md`, all
  200 through Go's own chain.

### `v1.1.0-rc.1`: what was checked

- `go vet ./...`, `go test ./...` on Linux, macOS and Windows (CI), the
  wasm build and its smoke test, `gofmt`.
- The contract at the pin: 1,583 of 1,583 checks, including the opaque
  key-order check (2026-09-25) and the Gemini schema-field vectors
  (MAP-16, 2026-09-26). Mixed-language sign-in runs with Python: pass.
- Live traffic (`receipts/2026-09-26-live-smoke`, and
  `receipts/2026-09-26-live-smoke-map16`): every API-key provider family,
  the Codex CLI's login file and five saved sign-ins, 77 checks, no Go
  defect; Go renewed three saved connections and lm15-python used them
  afterwards.

Released at the maintainer's request without one step this file listed:
a fresh sign-in through Go for each account flow (xAI and Copilot device
code, Claude and ChatGPT browser, OpenRouter loopback, Kimi Code, Meta).
Each needs a person to approve it. Go's flows are graded by the shared
managed runs, and Go renewed and used real connections that lm15-python
had created; a fresh Go sign-in per flow remains to do (v2.0.0 was released without it,
at the maintainer's request: sign-in is provisional).

## Cutting a release

1. `go vet ./... && go test ./...`, and the contract harness above: all green.
2. Tag an annotated tag on `main` (`git tag -a v2.0.1 -m "..."`) and push it.
3. Ask the proxy to fetch it, so the first user does not wait:
   `GOPROXY=https://proxy.golang.org go list -m github.com/lm15-dev/lm15-go/v2@v2.0.1`.
4. Check from a scratch module: `go get github.com/lm15-dev/lm15-go/v2@latest`.
