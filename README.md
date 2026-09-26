# lm15 for Go

One request and response model for every major AI model provider, in Go.
Write a `Request` once and send it to OpenAI, Anthropic, Gemini, xAI, Groq,
DeepSeek, OpenRouter, Z.AI, Moonshot, Meta, a cloud (Azure, Bedrock,
Vertex) or a model on your own machine: change the model string, keep the
program.

```go
router := lm15.NewRouter()
response, err := router.Complete(ctx, &lm15.Request{
    Model:    "anthropic:claude-haiku-4-5",
    Messages: []lm15.Message{lm15.UserMessage("What eats acorns at night?")},
})
fmt.Println(response.TextOr(""))
```

- **Standard library only.** No dependencies; builds for Linux, macOS,
  Windows and the browser (`GOOS=js GOARCH=wasm`).
- **Low-level on purpose.** Typed requests, responses, stream events, tools,
  media, errors and exact JSON serialization. No hidden tool loop, no
  retries you did not ask for, no prompt templates: the library you build
  on top decides those.
- **The same behavior in every language.** lm15 exists for Python,
  TypeScript, Rust and Go, graded by one shared
  [contract](https://github.com/lm15-dev/lm15-contract): the same request
  produces the same wire bytes and the same response in all four.

Documentation: **[lm15.dev](https://lm15.dev/docs/)** (every guide has Go
examples) · API reference:
[pkg.go.dev](https://pkg.go.dev/github.com/lm15-dev/lm15-go).

## Install

Requires Go 1.26.2 or newer.

```bash
go get github.com/lm15-dev/lm15-go@v1.1.0-rc.1
```

```go
import lm15 "github.com/lm15-dev/lm15-go"
```

**v1.1.0-rc.1 is a release candidate**: the first release of lm15 for Go,
open for trying before v1.1.0. Python's lm15 1.0 is stable; TypeScript and
Rust are release candidates too. Why Go starts at 1.1: see
[Versions](#versions).

Set the key of the provider you call (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...). The router reads it from the
environment; nothing else is configured.

## Guide

### Ask, stream, continue

```go
ctx := context.Background()
router := lm15.NewRouter()
request := &lm15.Request{
    Model:    "gpt-4.1-mini",
    System:   lm15.System("Answer in one sentence."),
    Messages: []lm15.Message{lm15.UserMessage("What eats acorns at night?")},
    Config:   lm15.Config{MaxTokens: lm15.I(200)},
}

response, err := router.Complete(ctx, request)
if err != nil {
    return err
}
fmt.Println(response.TextOr(""), response.FinishReason, *response.Usage.OutputTokens)

// Streaming: text as it arrives, then the same Response Complete returns.
stream := lm15.NewResponseStream(router.Stream(ctx, request), request)
for text, err := range stream.Text() {
    if err != nil {
        return err
    }
    fmt.Print(text)
}
final, err := stream.Response()

// A conversation is the messages so far, plus the reply.
request.Messages = append(request.Messages, final.Message, lm15.UserMessage("And by day?"))
```

A model string is either `provider:model` (`"gemini:gemini-2.5-flash"`,
`"groq:openai/gpt-oss-20b"`, `"ollama:qwen3.5:0.8b"`) or a bare name the
router recognizes (`"gpt-4.1-mini"`, `"claude-haiku-4-5"`).

### Tools

lm15 returns the model's tool calls; your program runs them and answers.

```go
weather := lm15.FunctionTool{
    Name:        "get_weather",
    Description: "Current weather for a city.",
    Parameters: lm15.JSONObject{
        lm15.KV("type", "object"),
        lm15.KV("properties", lm15.JSONObject{
            lm15.KV("city", lm15.JSONObject{lm15.KV("type", "string")}),
        }),
        lm15.KV("required", []any{"city"}),
    },
}
request.Tools = []lm15.Tool{weather}
response, err := router.Complete(ctx, request)
var results []lm15.ToolResultPart
for _, call := range response.ToolCalls() {
    city, _ := call.Input.Get("city").(string)
    results = append(results, lm15.ToolResult(call.ID, lookUpWeather(city)))
}
request.Messages = append(request.Messages, response.Message, lm15.ToolMessageParts(results...))
response, err = router.Complete(ctx, request) // the model answers with the results
```

### JSON objects keep their order

Tool parameters, response formats, tool-call input, extensions and provider
data are `lm15.JSONObject`: a list of key/value members, in order. Order is
data. A model fills a structured answer in the order its schema lists the
fields, so a schema that lists `reasoning` before `answer` gets reasoning
first.

```go
schema := lm15.JSONObject{
    lm15.KV("type", "object"),
    lm15.KV("properties", lm15.JSONObject{
        lm15.KV("reasoning", lm15.JSONObject{lm15.KV("type", "string")}),
        lm15.KV("answer", lm15.JSONObject{lm15.KV("type", "string")}),
    }),
    lm15.KV("required", []any{"reasoning", "answer"}),
    lm15.KV("additionalProperties", false),
}
request.Config.ResponseFormat = lm15.JSONObject{
    lm15.KV("type", "json_schema"), lm15.KV("name", "worked_answer"), lm15.KV("schema", schema),
}
```

Read with `Get`, `Lookup`, `Has`, `Keys` and `All` (in order); write with
`Set` (a key keeps its place) and `Delete`. Both copy before writing, so an
object you passed to lm15 never changes behind your back. JSON text keeps
its order too: `lm15.DecodeJSONObject(data)`. `fmt.Println(obj)` prints the
JSON lm15 would send. A Go `map` has no order: `lm15.ObjectFromMap(m)`
converts one with its keys sorted.

### Images, documents, audio

```go
photo := lm15.Image(lm15.WithPath("camera-trap.jpg")) // or WithURL, WithData, WithFileID
request.Messages = []lm15.Message{lm15.UserParts(lm15.Text("What animal is this?"), photo)}
```

### Errors

Every failure is an `*lm15.Error` with a `Kind` you can branch on
(`KindRateLimit`, `KindAuth`, `KindContextLength`, `KindUnsupportedFeature`,
...), the provider's own code and message, and rate-limit evidence when the
provider sent it.

```go
var e *lm15.Error
if errors.As(err, &e) && e.Kind.IsA(lm15.KindRateLimit) {
    time.Sleep(time.Duration(*e.RetryAfter * float64(time.Second)))
}
```

### Sign in once, use everywhere

Besides API keys, lm15 can use an account you sign in to (a ChatGPT, Claude,
xAI, GitHub Copilot or OpenRouter login), saved in one file that every lm15
language shares. Sign-in is **provisional**: whether each provider permits
it, and how it is billed, is the provider's call.

```go
client, err := lm15.Connect(ctx, lm15.ConnectOptions{}) // pick a provider and model in the terminal
response, err := client.Ask(ctx, "Explain drought stress in oaks.")
```

See [docs/managed-login.md](docs/managed-login.md).

### More

Judgments with probabilities (`lm15.Judgments`, `Response.Probabilities`),
reasoning controls, prompt caching, built-in provider tools (web search,
code execution), files and batches, image and speech generation, video,
realtime sessions, the model catalog, and reading an OpenAI Chat
Completions request into a `Request` are all in the
[guides](https://lm15.dev/docs/) and on
[pkg.go.dev](https://pkg.go.dev/github.com/lm15-dev/lm15-go).

## Stability

The chat core is stable within v1.x once v1.1.0 is released: requests,
responses, streaming, tools, structured output, errors, credentials and
model listing. These ship as **provisional** and may still change in 1.x
with a notice in the contract: files, batches, media generation, stored
caches, realtime sessions, Chat Completions ingest and sign-in.

## Conformance

This module is graded by [lm15-contract](https://github.com/lm15-dev/lm15-contract)
at the commit in `CONTRACT_PIN`: **1,583 of 1,583 checks pass**, the same
count as Python, TypeScript and Rust. The checks compare the exact requests
lm15 builds and the responses it reads against recorded provider traffic.

```bash
go build -o bin/lm15-vet ./cmd/lm15-vet
cd ../lm15-contract && python3 harness/check.py --shim go --direction all
```

Real calls: `go run ./examples/live_smoke` sends a small set of checks
(chat, streaming, structured output, a tool loop, model listing) to every
provider you have a key for, and writes receipts. The latest run is in
[receipts/](receipts/).

Go-specific differences from the Python reference, and where each part of
the contract lives in this module: [docs/port-notes.md](docs/port-notes.md).

## Versions

`v1.0.0` was tagged by mistake on 2026-06-11 from an early prototype, before
the shared contract existed. Go's module proxy keeps every tag forever, so
it is retracted (with `v1.0.1`, which only carries the retraction) and the
first real release is v1.1.0. See [RELEASING.md](RELEASING.md).

## Development

```bash
go vet ./... && go test ./...
GOOS=js GOARCH=wasm go build ./cmd/lm15-wasm   # the browser runtime
```

## License

MIT. See [LICENSE](LICENSE).
