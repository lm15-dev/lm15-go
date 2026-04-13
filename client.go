package lm15

import (
	"fmt"
	"strings"
)

// UniversalLM routes requests to the correct provider adapter.
type UniversalLM struct {
	adapters map[string]Adapter
}

// NewUniversalLM creates a new client.
func NewUniversalLM() *UniversalLM {
	return &UniversalLM{adapters: make(map[string]Adapter)}
}

// Register adds a provider adapter.
func (lm *UniversalLM) Register(adapter Adapter) {
	lm.adapters[adapter.ProviderName()] = adapter
}

func (lm *UniversalLM) resolveAdapter(model, provider string) (Adapter, error) {
	p := provider
	if p == "" {
		var err error
		p, err = ResolveProvider(model)
		if err != nil {
			return nil, err
		}
	}
	a, ok := lm.adapters[p]
	if !ok {
		registered := make([]string, 0, len(lm.adapters))
		for k := range lm.adapters {
			registered = append(registered, k)
		}
		if len(registered) == 0 {
			registered = []string{"(none)"}
		}
		return nil, &ProviderError{ULMError{fmt.Sprintf(
			"no adapter registered for provider '%s'\n\n"+
				"  Registered providers: %s\n"+
				"\n"+
				"  To fix, set the API key: export %s_API_KEY=...\n",
			p, strings.Join(registered, ", "), strings.ToUpper(p),
		)}}
	}
	return a, nil
}

// Complete executes a non-streaming request via the adapter's BaseAdapter.
func (lm *UniversalLM) Complete(request LMRequest, provider string) (LMResponse, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return LMResponse{}, err
	}
	req := a.BuildRequest(request, false)
	transport := adapterTransport(a)
	resp, err := transport.Request(req)
	if err != nil {
		return LMResponse{}, err
	}
	if resp.Status >= 400 {
		return LMResponse{}, a.NormalizeError(resp.Status, resp.Text())
	}
	return a.ParseResponse(request, resp)
}

// Stream opens a streaming request and returns a channel of events.
func (lm *UniversalLM) Stream(request LMRequest, provider string) (<-chan StreamEvent, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return nil, err
	}
	req := a.BuildRequest(request, true)
	transport := adapterTransport(a)
	body, err := transport.Stream(req)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEvent, 32)
	go func() {
		defer close(ch)
		defer body.Close()
		for raw := range ParseSSE(body) {
			evt, err := a.ParseStreamEvent(request, raw)
			if err != nil {
				ch <- StreamEvent{Type: "error", Error: &ErrorInfo{Code: "provider", Message: err.Error()}}
				return
			}
			if evt != nil {
				ch <- *evt
			}
		}
	}()
	return ch, nil
}

// Embeddings runs an embedding request.
func (lm *UniversalLM) Embeddings(request EmbeddingRequest, provider string) (EmbeddingResponse, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return EmbeddingResponse{}, err
	}
	return a.Embeddings(request)
}

// FileUpload uploads a file.
func (lm *UniversalLM) FileUpload(request FileUploadRequest, provider string) (FileUploadResponse, error) {
	p := provider
	if p == "" && request.Model != "" {
		p2, _ := ResolveProvider(request.Model)
		p = p2
	}
	a, err := lm.resolveAdapter("", p)
	if err != nil {
		return FileUploadResponse{}, err
	}
	return a.FileUpload(request)
}

// ImageGenerate generates images.
func (lm *UniversalLM) ImageGenerate(request ImageGenerationRequest, provider string) (ImageGenerationResponse, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return ImageGenerationResponse{}, err
	}
	return a.ImageGenerate(request)
}

// AudioGenerate generates audio.
func (lm *UniversalLM) AudioGenerate(request AudioGenerationRequest, provider string) (AudioGenerationResponse, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return AudioGenerationResponse{}, err
	}
	return a.AudioGenerate(request)
}

// adapterTransport extracts the Transport from an adapter.
func adapterTransport(a Adapter) Transport {
	type hasTransport interface {
		GetTransport() Transport
	}
	if ht, ok := a.(hasTransport); ok {
		return ht.GetTransport()
	}
	// All our adapters embed BaseAdapter which has Tport
	// Use reflection-free approach: check for the Tport field via BaseAdapter
	if ba, ok := a.(interface{ Base() *BaseAdapter }); ok {
		return ba.Base().Tport
	}
	return NewStdTransport(DefaultPolicy())
}
