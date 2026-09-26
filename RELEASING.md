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
- **The first real release is `v1.1.0`,** preceded by release candidates
  `v1.1.0-rc.1`, `-rc.2`, and so on. A version number can never be reused, so
  `v1.0.x` is closed.

Checked on a private copy before tagging (2026-09-24): with both versions
retracted, `go get` installs `main` as a pseudo-version; a module pinned to
`v1.0.0` sees "(retracted)" and the reason; once `v1.1.0-rc.1` exists,
`go get ...@latest` selects it.

## Releases

| Version | Date | Contract pin | Notes |
|---|---|---|---|
| `v1.1.0-rc.1` | 2026-09-26 | `3763eec` | The first release. 1,583 of 1,583 contract checks. |
| `v1.1.0-rc.2` | 2026-09-26 | `fe5cdf9` | DeepInfra, Together AI, Fireworks AI, Parasail; the Google Cloud pass. 1,788 of 1,788 contract checks. |

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
had created; a fresh Go sign-in per flow remains to do before v1.1.0.

## Cutting a release

1. `go vet ./... && go test ./...`, and the contract harness above: all green.
2. Tag an annotated tag on `main` (`git tag -a v1.1.0-rc.1 -m "..."`) and push it.
3. Ask the proxy to fetch it, so the first user does not wait:
   `GOPROXY=https://proxy.golang.org go list -m github.com/lm15-dev/lm15-go@v1.1.0-rc.1`.
4. Check from a scratch module: `go get github.com/lm15-dev/lm15-go@latest`.
