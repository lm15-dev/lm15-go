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

On top of the core sits the client layer: `OpenAILM`, `OpenAIChatLM`,
`AnthropicLM`, `GeminiLM` with context-first `Complete`/`Stream` methods
over a shared `net/http` client (stdlib only). Streams are normalized per
MAP-3: exactly one `StreamEndEvent` ends the stream, carrying merged
`finish_reason` and `usage`.

Not implemented: non-chat endpoints (embeddings, files, batch, image/audio
generation) and live sessions (PROVISIONAL in `spec/SCOPE.md`).

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"os"

	lm15 "github.com/lm15-dev/lm15-go"
)

func main() {
	lm, _ := lm15.NewOpenAILM(os.Getenv("OPENAI_API_KEY"))
	maxTokens := int64(50)
	resp, err := lm.Complete(context.Background(), lm15.Request{
		Model:    "gpt-4.1-mini",
		System:   "You are terse.",
		Messages: []lm15.Message{lm15.UserMessage("Say hello in three words.")},
		Config:   &lm15.Config{MaxTokens: &maxTokens},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())       // Hello there!
	fmt.Println(resp.FinishReason) // stop
	fmt.Println(*resp.Usage.TotalTokens)
}
```

The mental model is one straight line, same as the reference
implementation: parts → `Message` → `Request` → LM → `Response`, with
`Stream` as the event-wise twin.

Any OpenAI-compatible server is one compat preset away (the preset bundles
the server's wire quirks and its default base URL):

```go
lm, _ := lm15.NewOpenAIChatLM("ollama", lm15.WithCompat(lm15.CompatOllama))
// base URL -> http://localhost:11434/v1; swap CompatGroq, CompatOpenRouter, ...
resp, _ := lm.Complete(ctx, lm15.Request{
	Model:    "qwen3.5:0.8b",
	Messages: []lm15.Message{lm15.UserMessage("Say hello in five words or fewer.")},
	Config:   &lm15.Config{MaxTokens: &maxTokens, Extensions: map[string]any{"reasoning_effort": "none"}},
})
```

### Streaming

`Stream` returns an `iter.Seq2[StreamEvent, error]`. Text arrives as
`StreamDeltaEvent{Delta: TextDelta{...}}`; exactly one `StreamEndEvent`
ends the stream with `finish_reason` and `usage` (MAP-3):

```go
for ev, err := range lm.Stream(ctx, req) {
	if err != nil {
		panic(err)
	}
	if d, ok := ev.(lm15.StreamDeltaEvent); ok {
		if td, ok := d.Delta.(lm15.TextDelta); ok {
			fmt.Print(td.Text)
		}
	}
}
```

To consume a stream into a full `Response` (identical in shape to one from
`Complete`): `resp, err := lm.CollectResponse(ctx, req)`.

### Tools: the full round-trip

```go
desc := "Get the current weather for a city."
weather := lm15.FunctionTool{
	Name:        "get_weather",
	Description: &desc,
	Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []any{"city"},
	},
}
req := lm15.Request{
	Model:    "gpt-4.1-mini",
	Messages: []lm15.Message{lm15.UserMessage("What is the weather in Montreal? Use the tool.")},
	Tools:    []lm15.Tool{weather},
}
first, _ := lm.Complete(ctx, req)
call := first.ToolCalls()[0] // typed ToolCallPart: ID, Name, Input
req.Messages = append(req.Messages,
	first.Message, // replay the assistant turn
	lm15.ToolResults(lm15.ToolResult(call.ID, "Sunny, 22C")),
)
final, _ := lm.Complete(ctx, req)
fmt.Println(final.Text()) // "The weather in Montreal is currently sunny ... 22°C."
```

All of the above ran live (June 2026): the quickstart and tools round-trip
against OpenAI (`gpt-4.1-mini`), the compat example against local Ollama
(`qwen3.5:0.8b`) and Groq (`llama-3.1-8b-instant`), and streaming against
all three — see `client_live_test.go` for the exact assertions.

## Live smoke tests

`go test` runs the live smokes when targets are available and skips them
otherwise (CI-safe): `-short` skips all, each provider test skips without
its key (`OPENAI_API_KEY`, `GROQ_API_KEY`), and the Ollama tests skip
unless localhost:11434 answers.

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

## Dependency budget

Zero third-party dependencies — standard library only, by policy.
