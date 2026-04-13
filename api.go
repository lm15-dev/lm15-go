package lm15

import (
	"encoding/json"
	"fmt"
	"sync"
)

// ── Module-level defaults ──────────────────────────────────────────

var (
	defaultsMu    sync.RWMutex
	defaults      = make(map[string]interface{})
	clientCacheMu sync.RWMutex
	clientCache   = make(map[string]*UniversalLM)
	factories     []AdapterFactory // set by RegisterCoreAdapters
)

// Configure sets module-level defaults so you don't repeat them on every call.
func Configure(env string, apiKey interface{}) {
	defaultsMu.Lock()
	defer defaultsMu.Unlock()
	defaults = make(map[string]interface{})
	clientCacheMu.Lock()
	clientCache = make(map[string]*UniversalLM)
	clientCacheMu.Unlock()
	if env != "" {
		defaults["env"] = env
	}
	if apiKey != nil {
		defaults["apiKey"] = apiKey
	}
}

// RegisterCoreAdapters sets the adapter factories used by Call/Model/etc.
// Must be called once at init time (typically from the provider package).
func RegisterCoreAdapters(f []AdapterFactory) {
	factories = f
}

func resolveDefault(key string, explicit interface{}) interface{} {
	if explicit != nil {
		return explicit
	}
	defaultsMu.RLock()
	defer defaultsMu.RUnlock()
	return defaults[key]
}

func getClient(apiKey interface{}, providerHint, env string) *UniversalLM {
	resolvedKey := resolveDefault("apiKey", apiKey)
	resolvedEnv, _ := resolveDefault("env", nilIfEmpty(env)).(string)

	keyJSON, _ := json.Marshal(resolvedKey)
	cacheKey := fmt.Sprintf("%s|%s|%s", string(keyJSON), providerHint, resolvedEnv)

	clientCacheMu.RLock()
	client, ok := clientCache[cacheKey]
	clientCacheMu.RUnlock()
	if ok {
		return client
	}

	var ak interface{}
	if resolvedKey != nil {
		ak = resolvedKey
	}

	client = BuildDefaultWithFactories(factories, &BuildOpts{
		APIKey:       ak,
		ProviderHint: providerHint,
		Env:          resolvedEnv,
	})

	clientCacheMu.Lock()
	clientCache[cacheKey] = client
	clientCacheMu.Unlock()

	return client
}

func nilIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

// ── CallOpts ───────────────────────────────────────────────────────

// CallOpts configures a Call.
type CallOpts struct {
	System        string
	Tools         []Tool
	OnToolCall    func(ToolCallInfo) interface{}
	Reasoning     interface{} // bool or map[string]any
	Prefill       string
	Output        string // "image", "audio"
	PromptCaching bool
	Temperature   *float64
	MaxTokens     *int
	TopP          *float64
	Stop          []string
	MaxToolRounds int
	Retries       int
	Provider      string
	APIKey        interface{} // string or map[string]string
	Env           string
}

// Call makes a single request to any model. Returns a Result.
//
//	result := lm15.Call("gpt-4.1-mini", "Hello.")
//	fmt.Println(result.Text())
//
//	for chunk := range lm15.Call("gpt-4.1-mini", "Write a haiku.").Stream() {
//	    fmt.Print(chunk.Text)
//	}
func Call(model string, prompt string, opts *CallOpts) *Result {
	if opts == nil {
		opts = &CallOpts{}
	}

	lm := getClient(opts.APIKey, opts.Provider, opts.Env)
	provider := opts.Provider
	if provider == "" {
		p, _ := ResolveProvider(model)
		provider = p
	}

	// Build the request
	messages := []Message{UserMessage(prompt)}
	if opts.Prefill != "" {
		messages = append(messages, AssistantMessage(opts.Prefill))
	}

	providerCfg := map[string]any{}
	if opts.PromptCaching {
		providerCfg["prompt_caching"] = true
	}
	if opts.Output != "" {
		providerCfg["output"] = opts.Output
	}

	var reasoning map[string]any
	switch v := opts.Reasoning.(type) {
	case bool:
		if v {
			reasoning = map[string]any{"enabled": true}
		}
	case map[string]any:
		reasoning = map[string]any{"enabled": true}
		for k, val := range v {
			reasoning[k] = val
		}
	}

	config := Config{
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
		TopP:        opts.TopP,
		Stop:        opts.Stop,
		Reasoning:   reasoning,
	}
	if len(providerCfg) > 0 {
		config.Provider = providerCfg
	}

	request := LMRequest{
		Model:    model,
		Messages: messages,
		System:   opts.System,
		Tools:    opts.Tools,
		Config:   config,
	}

	// Build callable registry
	callableRegistry := make(map[string]func(map[string]any) (any, error))
	for _, t := range opts.Tools {
		if t.Type == "function" && t.Fn != nil {
			callableRegistry[t.Name] = t.Fn
		}
	}

	maxRounds := opts.MaxToolRounds
	if maxRounds == 0 {
		maxRounds = 8
	}

	return NewResult(ResultOpts{
		Request: request,
		StartStream: func(req LMRequest) (<-chan StreamEvent, error) {
			return lm.Stream(req, provider)
		},
		CallableRegistry: callableRegistry,
		OnToolCall:       opts.OnToolCall,
		MaxToolRounds:    maxRounds,
		Retries:          opts.Retries,
	})
}

// ModelObj creates a reusable, stateful model object.
//
//	gpt := lm15.ModelObj("gpt-4.1-mini", &lm15.ModelObjOpts{System: "You are terse."})
//	resp := gpt.Call("Hello.", nil)
//	fmt.Println(resp.Text())
func ModelObj(modelName string, opts *ModelObjOpts) *Model {
	if opts == nil {
		opts = &ModelObjOpts{}
	}
	lm := getClient(opts.APIKey, opts.Provider, opts.Env)

	return NewModel(ModelOpts{
		LM:            lm,
		Model:         modelName,
		System:        opts.System,
		Tools:         opts.Tools,
		OnToolCall:    opts.OnToolCall,
		Provider:      opts.Provider,
		Retries:       opts.Retries,
		Cache:         opts.Cache,
		PromptCaching: opts.PromptCaching,
		Temperature:   opts.Temperature,
		MaxTokens:     opts.MaxTokens,
		MaxToolRounds: opts.MaxToolRounds,
	})
}

// ModelObjOpts configures ModelObj.
type ModelObjOpts struct {
	System        string
	Tools         []Tool
	OnToolCall    func(ToolCallInfo) interface{}
	Provider      string
	Retries       int
	Cache         bool
	PromptCaching bool
	Temperature   *float64
	MaxTokens     *int
	MaxToolRounds int
	APIKey        interface{}
	Env           string
}

// Prepare builds an LMRequest without sending it.
func Prepare(model, prompt string, opts *CallOpts) LMRequest {
	if opts == nil {
		opts = &CallOpts{}
	}
	m := ModelObj(model, &ModelObjOpts{
		Provider:      opts.Provider,
		PromptCaching: opts.PromptCaching,
		System:        opts.System,
		APIKey:        opts.APIKey,
		Env:           opts.Env,
	})
	return m.Prepare(prompt, opts)
}

// Send sends a pre-built LMRequest. Returns a Result.
func Send(request LMRequest, opts *SendOpts) *Result {
	if opts == nil {
		opts = &SendOpts{}
	}
	provider := opts.Provider
	if provider == "" {
		provider, _ = ResolveProvider(request.Model)
	}
	lm := getClient(opts.APIKey, provider, opts.Env)

	callableRegistry := make(map[string]func(map[string]any) (any, error))
	for _, t := range request.Tools {
		if t.Type == "function" && t.Fn != nil {
			callableRegistry[t.Name] = t.Fn
		}
	}

	return NewResult(ResultOpts{
		Request: request,
		StartStream: func(req LMRequest) (<-chan StreamEvent, error) {
			return lm.Stream(req, provider)
		},
		CallableRegistry: callableRegistry,
	})
}

// SendOpts configures Send.
type SendOpts struct {
	Provider string
	APIKey   interface{}
	Env      string
}

// Upload uploads a file and returns a Part referencing it.
func Upload(modelName, filePath string, opts *UploadOpts) (Part, error) {
	if opts == nil {
		opts = &UploadOpts{}
	}
	provider := opts.Provider
	if provider == "" {
		provider, _ = ResolveProvider(modelName)
	}
	m := ModelObj(modelName, &ModelObjOpts{
		Provider: provider,
		APIKey:   opts.APIKey,
		Env:      opts.Env,
	})
	return m.Upload(filePath)
}

// UploadOpts configures Upload.
type UploadOpts struct {
	Provider string
	APIKey   interface{}
	Env      string
}

// Providers returns {provider: [env_var_names]} for all core adapters.
func Providers() map[string][]string {
	return ProviderEnvKeys()
}
