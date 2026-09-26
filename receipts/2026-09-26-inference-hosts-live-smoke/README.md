# Live smoke, open-model inference hosts, 2026-09-26

`go run ./examples/live_smoke -out receipts/2026-09-26-inference-hosts-live-smoke -only deepinfra,together,fireworks,parasail`
against lm15-go with the four providers added (lm15-contract `fe5cdf9`,
changes/2026-09-26-inference-hosts-live.md): 16 checks, 0 failed — hello
(complete equals stream), order (json_schema key order kept), tools (a call
and its result round trip), models (DeepInfra 187, Together 272, Fireworks 27,
Parasail 91).

The per-check `ms` field was removed from these files before commit: Parasail's
terms (§ 2.2(h)) forbid publishing performance information, and the others
forbid benchmarking. The receipts record wire shapes only.
