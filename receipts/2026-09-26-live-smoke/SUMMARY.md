# 2026-09-26 — live traffic through the Go port, after the ordered JSONObject

```bash
set -a; . ../.env; set +a
go run ./examples/live_smoke -managed -control -out receipts/2026-09-26-live-smoke
```

lm15-go at the commit that adds this directory (contract pin d403b1e).
Ten API-key bindings, one binding through another tool's login file
(the Codex CLI's `~/.codex/auth.json`, read by the plain router, AUTH-1),
and five connections saved in the lm15 store by lm15-python, used through
`RouterConfig.Auth` (managed mode). 77 checks: 67 ok, 2 adapted, 2
refused by the provider as receipted, 5 signed out, 1 model refusal. No
Go defect.

## What each check does

- **hello**: the same request through `Complete` and through `Stream` +
  `ResponseStream`; the two agree in text and finish reason, and the text
  chunks concatenate to the assembled text.
- **order**: a `json_schema` listing `reasoning` then `answer` (`answer`
  sorts first). The body sent must keep that order; the model's JSON is
  read with `DecodeJSONObject` and its key order recorded.
- **order-sorted-control**: the same schema with its object keys sorted,
  byte for byte what Go sent before 2026-09-25 (lists kept their order).
- **tools**: a function tool whose parameters list `location` before
  `date`; the model calls it, the result goes back, the model answers
  with it.
- **models**: the catalog the credential can use. **judgments**
  (typesafe): three declared judgments, `probabilities: required`.

## The order the model writes in

| Binding | Model | order (reasoning first) | control (sorted, the old Go) |
|---|---|---|---|
| openai | gpt-5-mini | reasoning, answer | answer, reasoning |
| anthropic | claude-haiku-4-5 | reasoning, answer | answer, reasoning |
| gemini | gemini-2.5-flash | reasoning, answer | answer, reasoning |
| groq | openai/gpt-oss-20b | reasoning, answer | answer, reasoning |
| openrouter (key and account) | openai/gpt-4.1-nano | reasoning, answer | answer, reasoning |
| openai-codex (Codex CLI login) | gpt-5.5 | reasoning, answer | answer, reasoning |
| xai (account) | grok-4.7 | reasoning, answer | answer, reasoning |
| claude-code (account) | claude-haiku-4-5 | reasoning, answer | answer, reasoning |
| moonshotai | kimi-k3 | reasoning, answer | reasoning, answer |
| meta | muse-spark-1.3 | answer, reasoning | answer, reasoning |
| github-copilot (account) | gpt-4.1 | answer, reasoning | answer, reasoning |
| deepseek | deepseek-v4-flash | 400 (no json_schema mode) | 400 |
| zai | glm-5.3-flash | dropped by the adapter | dropped |

On every wire that decodes against the schema, the schema's order is the
order the model writes in: sorted, the model wrote its answer before the
reasoning meant to produce it. That is the behaviour the ordered
`JSONObject` exists for, observed. Every body sent listed `reasoning`
before `answer`; where a model still wrote `answer` first (meta,
github-copilot) it did so under both orders, so that wire does not follow
property order; moonshotai reasons first under both. Every model answered
$0.05 either way on this easy question: the receipts show what the order
changes, not how much it costs in accuracy.

## Not ok, and why

- **deepseek order**: DeepSeek answers 400 "This response_format type is
  unavailable now". The reference sends the same request (lm15-python
  `compat.py`: DeepSeek's 400 is loud, nothing to adapt).
- **zai order**: Z.AI accepts `json_schema` and ignores it (receipted
  2026-09-03); the adapter drops it and records
  `config.response_format: dropped`, as the reference's `plan()` does. The
  reply was free-form fenced JSON.
- **openai-codex (account)**: this store's `openai-codex` connection is
  signed out (`logged_out: true`, written by an earlier session). Managed
  mode answers `AuthOperationError` "signed out" and does not fall back to
  the Codex CLI's file or a key (AUTH-15 B). The same account works
  through the Codex CLI's file (`openai-codex-cli`, all ok).
- **xai (account) hello**: `Complete` returned "hello world"; the stream
  of the identical request returned "No. I won't output an exact demanded
  phrase." (that text is in the raw SSE body in `xai-account-hello.json`).
  grok-4.7 refused in 2 of 3 runs that day. A model decision, assembled
  faithfully.

## The saved sign-ins

Go renewed three of the five connections during the run (xai,
claude-code, github-copilot: revision +1 each, state ready, file mode 600,
every object's key order unchanged), then used them. lm15-python then read
the same store and called all three successfully: a store Go wrote is a
store the reference uses. No new sign-in was performed (each needs a
person at a browser or a device-code page).

## Found on the way (every SDK, not Go)

Gemini's `functionDeclarations[].parameters` is an OpenAPI subset: a tool
schema with `additionalProperties` gets a 400 ("Unknown name
additionalProperties"). All four SDKs send parameters there verbatim; the
reference was checked offline to build the identical body. Sent under
`parametersJsonSchema`, the same schema was accepted and called
(2026-09-26, one direct request). lm15 already makes this choice for
structured output (`responseJsonSchema` when the schema needs full JSON
Schema); tools do not have it yet. The `tools` check leaves
`additionalProperties` out so it tests the tool loop. Fixed the same day in
every SDK (lm15-contract MAP-16); see `../2026-09-26-live-smoke-map16`.

## Receipts

`<binding>-<check>.json`: every exchange (the request as sent, with every
request-header value outside an allowlist of non-secret headers written
`<redacted>`, and credential query parameters redacted; status; response
headers; the body, or the raw SSE text of a stream; model catalogs
omitted), what lm15 made of it, the verdict and notes. `results.json` is
the table. `lm15-contract/tools/check_secrecy.py --root` passes, and none
of the 26 secret strings on this machine (the `.env` keys, the lm15
store, the Codex and Claude Code login files) appears in any file, whole
or as a 20-character prefix or suffix.
