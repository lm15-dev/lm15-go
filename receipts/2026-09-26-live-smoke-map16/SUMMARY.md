# 2026-09-26 — one tool schema, three providers, after MAP-16

```bash
go run ./examples/live_smoke -only gemini,openai,anthropic -out receipts/2026-09-26-live-smoke-map16
```

The `tools` check's schema now carries `"additionalProperties": false`, as
OpenAI strict mode wants. Earlier the same day (`../2026-09-26-live-smoke`)
Gemini answered that schema with a 400 in every SDK, and the check had to
leave the keyword out. With MAP-16 (lm15-contract
`changes/2026-09-26-gemini-schema-fields.md`) Go sends it to Gemini as
`functionDeclarations[].parametersJsonSchema`: 12 checks, 0 failed; the
model called `get_forecast` and used its result on all three providers.
`gemini-tools.json` shows the declaration as sent.
