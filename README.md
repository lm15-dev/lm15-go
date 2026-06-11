# lm15-go

Go implementation of the lm15 canonical model, ported from the
[lm15 contract](../lm15-contract) (spec/types.md, spec/vocabularies.md,
spec/invariants.md, docs/serde-rules.md, docs/mapping-rules.md).

## Status

This port implements the frozen **chat core** per `spec/SCOPE.md`:
canonical types and serde, error normalization, request building, response
parsing, and stream replay (SSE parsing, per-provider event mapping, MAP-3
coalescing) for the four adapters (`openai`, `openai_chat`, `anthropic`,
`gemini`).

Against the contract harness (all five directions) it passes
**304 checks, 0 failures** (request 110, response 102, stream 8, error 16,
serde 68; 4 skips are cases not applicable to the shim protocol).

Not yet implemented: non-chat endpoints (embeddings, files, batch,
image/audio generation), live sessions, and any HTTP transport — this
library is the pure transformation core plus the vet shim.

## Build

```bash
go mod download
go build -o bin/lm15-vet ./cmd/lm15-vet
go test ./...
go vet ./...
```

## Conformance

```bash
cd ../lm15-contract
../lm15-python2/.venv/bin/python harness/check.py --shim go --direction all
```

The harness drives `bin/lm15-vet` (see `harness/shims.json`) inside a
no-network sandbox; build the binary first.
