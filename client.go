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

// Complete executes a non-streaming request.
func (lm *UniversalLM) Complete(request LMRequest, provider string) (LMResponse, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return LMResponse{}, err
	}
	// All adapters embed BaseAdapter which has Complete
	type completer interface {
		Complete(Adapter, LMRequest) (LMResponse, error)
	}
	if c, ok := a.(completer); ok {
		return c.Complete(a, request)
	}
	// Fallback: use the adapter interface directly
	req := a.BuildRequest(request, false)
	_ = req
	return LMResponse{}, &UnsupportedFeatureError{ProviderError{ULMError{a.ProviderName() + ": complete not supported via this path"}}}
}

// Stream opens a streaming request and returns a channel of events.
func (lm *UniversalLM) Stream(request LMRequest, provider string) (<-chan StreamEvent, error) {
	a, err := lm.resolveAdapter(request.Model, provider)
	if err != nil {
		return nil, err
	}
	type streamer interface {
		StreamEvents(Adapter, LMRequest) (<-chan StreamEvent, error)
	}
	if s, ok := a.(streamer); ok {
		return s.StreamEvents(a, request)
	}
	return nil, &UnsupportedFeatureError{ProviderError{ULMError{a.ProviderName() + ": stream not supported"}}}
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
	a, err := lm.resolveAdapter("", provider)
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
