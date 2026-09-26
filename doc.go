// Package lm15 is one request and response model for every major AI model
// provider: write a [Request] once and send it to OpenAI, Anthropic, Gemini,
// xAI, Groq, DeepSeek, OpenRouter, Z.AI, Moonshot, Meta, a cloud (Azure,
// Bedrock, Vertex) or a local server, by changing the model string.
//
//	router := lm15.NewRouter() // keys from the environment
//	response, err := router.Complete(ctx, &lm15.Request{
//		Model:    "anthropic:claude-haiku-4-5",
//		Messages: []lm15.Message{lm15.UserMessage("What eats acorns at night?")},
//	})
//	if err != nil {
//		return err
//	}
//	fmt.Println(response.TextOr(""))
//
// The package is low-level on purpose: typed requests ([Request], [Message],
// the [Part] kinds), responses ([Response]), stream events ([StreamEvent],
// [ResponseStream]), tools ([FunctionTool], [BuiltinTool]), errors
// ([Error], [ErrorKind]) and exact JSON serialization. It runs no tool loop
// and retries nothing on its own; programs and libraries built on it decide
// those.
//
// # Routing
//
// [LMRouter] resolves a model string to a provider: "provider:model"
// ("gemini:gemini-2.5-flash", "ollama:qwen3.5:0.8b") or a bare name it
// recognizes ("gpt-4.1-mini"). [NewRouterWithConfig] sets keys, endpoints,
// cloud identities, timeouts and saved sign-ins ([Auth]) explicitly. Each
// provider also has its own constructor (NewOpenAILM, NewAnthropicLM,
// NewGeminiLM, ...) returning an [LM].
//
// # JSON objects
//
// Tool parameters, response formats, tool-call input, extensions and
// provider data are [JSONObject]: key/value [Member]s in order. The order is
// sent as written and read as received, because it is data: a model fills a
// structured answer in the order its schema lists the fields. Build one with
// [KV] or [DecodeJSONObject]; read it with Get, Lookup and All.
//
// # Contract
//
// lm15 exists for Python, TypeScript, Rust and Go, graded by one shared
// contract (github.com/lm15-dev/lm15-contract): the same request produces
// the same wire request and the same response in every language. Guides
// with Go examples: https://lm15.dev/docs/.
package lm15
