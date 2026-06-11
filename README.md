# lm15-go

Go implementation of the lm15 canonical model, ported from the
[lm15 contract](../lm15-contract) (spec/types.md, spec/vocabularies.md,
spec/invariants.md, docs/serde-rules.md).

Stage A: canonical types, canonical serde (one omission rule, Number rule,
opaque payloads verbatim), and the vet shim (`cmd/lm15-vet`) speaking the
harness JSONL protocol with `capabilities`, `serde_roundtrip`, `validate`,
and `surface_dump`.

## Build

```bash
go build -o bin/lm15-vet ./cmd/lm15-vet
go test ./...
```

## Conformance

```bash
cd ../lm15-contract
../lm15-python2/.venv/bin/python harness/check.py --shim go --direction serde
```
