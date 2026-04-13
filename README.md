# lm15

[![Go Reference](https://pkg.go.dev/badge/github.com/lm15-dev/lm15-go.svg)](https://pkg.go.dev/github.com/lm15-dev/lm15-go)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One interface for OpenAI, Anthropic, and Gemini. Zero dependencies.

Go implementation — conforms to the [lm15 spec](https://github.com/lm15-dev/spec).

```go
package main

import (
    "fmt"
    lm15 "github.com/lm15-dev/lm15-go"
)

func main() {
    result := lm15.Call("gpt-4.1-mini", "Hello.", nil)
    fmt.Println(result.Text())
}
```

Switch models by changing the string. Same types, same streaming, same tool calling.

## Install

```bash
go get github.com/lm15-dev/lm15-go
```

Set at least one provider key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

## Usage

### Blocking

```go
result := lm15.Call("gpt-4.1-mini", "Hello.", nil)
fmt.Println(result.Text())

resp, err := result.Response()
fmt.Println(resp.Usage.TotalTokens)
```

### Streaming

```go
for chunk := range lm15.Call("gpt-4.1-mini", "Write a haiku.", nil).Stream() {
    switch chunk.Type {
    case "text":
        fmt.Print(chunk.Text)
    case "thinking":
        fmt.Printf("💭 %s", chunk.Text)
    case "tool_call":
        fmt.Printf("🔧 %s(%v)\n", chunk.Name, chunk.Input)
    case "finished":
        fmt.Printf("\n📊 %+v\n", chunk.Response.Usage)
    }
}
```

### Tools (auto-execute)

```go
result := lm15.Call("gpt-4.1-mini", "Weather in Montreal?", &lm15.CallOpts{
    Tools: []lm15.Tool{
        {
            Type: "function", Name: "get_weather",
            Description: "Get weather by city",
            Parameters: map[string]any{
                "type": "object",
                "properties": map[string]any{"city": map[string]any{"type": "string"}},
                "required": []string{"city"},
            },
            Fn: func(args map[string]any) (any, error) {
                return fmt.Sprintf("22°C in %s", args["city"]), nil
            },
        },
    },
})
fmt.Println(result.Text()) // "It's 22°C in Montreal."
```

### Multimodal

```go
result := lm15.Call("gemini-2.5-flash", "Describe this image.", &lm15.CallOpts{
    // Attach parts via messages
})

// Or build the request manually:
req := lm15.LMRequest{
    Model: "gemini-2.5-flash",
    Messages: []lm15.Message{{
        Role: lm15.RoleUser,
        Parts: []lm15.Part{
            lm15.TextPart("Describe this image."),
            lm15.ImageURL("https://example.com/cat.jpg"),
        },
    }},
}
```

### Reasoning

```go
result := lm15.Call("claude-sonnet-4-5", "Prove √2 is irrational.", &lm15.CallOpts{
    Reasoning: true,
})
fmt.Println(result.Thinking())
fmt.Println(result.Text())
```

### Structured output (JSON)

```go
result := lm15.Call("gpt-4.1-mini", "Extract: 'Alice is 30.'.", &lm15.CallOpts{
    System: "Return JSON: {name, age}",
    Prefill: "{",
})
var data struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}
resp, _ := result.Response()
resp.JSON(&data)
fmt.Printf("%s is %d\n", data.Name, data.Age)
```

### Cost tracking

```go
err := lm15.ConfigureWithOptions(lm15.ConfigureOpts{TrackCosts: true})
if err != nil {
    panic(err)
}

result := lm15.Call("gpt-4.1-mini", "Explain TCP.", nil)
fmt.Printf("cost: %+v\n", result.Cost())

m := lm15.ModelObj("claude-sonnet-4", nil)
m.Call("What is TCP?", nil).Text()
m.Call("What is UDP?", nil).Text()
fmt.Printf("total cost: %+v\n", m.TotalCost())
```

You can also estimate manually with `EstimateCost()`, install pricing with
`EnableCostTracking()`, or inject your own catalog with `SetCostIndex()`.

## Architecture

```
Call()                           ← high-level surface
  │
  ▼
Result (lazy, channel-based)
  │
  ▼
LMRequest → UniversalLM → Adapter → Transport (net/http)
                             │
                    provider/{openai,anthropic,gemini}.go
```

## Implementation status

| Layer | Status |
|---|---|
| Types (`Part`, `Message`, `LMRequest`, etc.) | ✅ |
| Error hierarchy | ✅ |
| SSE parser | ✅ |
| Transport (`net/http`, zero deps) | ✅ |
| Provider adapters (OpenAI, Anthropic, Gemini) | ✅ |
| UniversalLM client | ✅ |
| Result (lazy stream, auto tool execution) | ✅ |
| Conversation helper | ✅ |
| Factory (`BuildDefault`, env file parsing) | ✅ |
| Capability resolver | ✅ |
| High-level API (`Call`, `Configure`, `Providers`) | ✅ |

## Related

- [lm15 spec](https://github.com/lm15-dev/spec) — canonical type definitions and test fixtures
- [lm15 Python](https://github.com/lm15-dev/lm15-python) — reference implementation
- [lm15 TypeScript](https://github.com/lm15-dev/lm15-ts) — TypeScript implementation

## License

MIT
