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

## Before `v1.1.0-rc.1`

The port must pass the contract at its `CONTRACT_PIN`
(`python3 harness/check.py --shim go --direction all` in lm15-contract). At
`b0ff3c0` it fails 24 cases:

- 23 where Go writes JSON object keys in sorted order and the recorded bytes
  keep insertion order: 20 Bedrock Chat / Mantle requests (the SigV4 signature
  covers the body bytes), 2 batch JSONL uploads, 1 image-edit multipart field
  order. Providers accept either order; a signature or an uploaded file pins
  the bytes.
- 1: `cached_prefix.routed`, the `CachedPrefix.provider` field (contract
  2026-09-20).

## Cutting a release

1. `go vet ./... && go test ./...`, and the contract harness above: all green.
2. Tag an annotated tag on `main` (`git tag -a v1.1.0-rc.1 -m "..."`) and push it.
3. Ask the proxy to fetch it, so the first user does not wait:
   `GOPROXY=https://proxy.golang.org go list -m github.com/lm15-dev/lm15-go@v1.1.0-rc.1`.
4. Check from a scratch module: `go get github.com/lm15-dev/lm15-go@latest`.
