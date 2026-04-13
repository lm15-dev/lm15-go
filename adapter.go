package lm15



// EndpointSupport declares which operations an adapter supports.
type EndpointSupport struct {
	Complete   bool
	Stream     bool
	Live       bool
	Embeddings bool
	Files      bool
	Batches    bool
	Images     bool
	Audio      bool
}

// ProviderManifest describes a provider adapter's metadata.
type ProviderManifest struct {
	Provider string
	Supports EndpointSupport
	EnvKeys  []string
}

// Adapter is the interface every provider must implement.
type Adapter interface {
	ProviderName() string
	Manifest() ProviderManifest
	BuildRequest(request LMRequest, stream bool) HTTPRequest
	ParseResponse(request LMRequest, response HTTPResponse) (LMResponse, error)
	ParseStreamEvent(request LMRequest, raw SSEEvent) (*StreamEvent, error)
	NormalizeError(status int, body string) error

	// Optional methods — check Manifest().Supports before calling.
	Embeddings(request EmbeddingRequest) (EmbeddingResponse, error)
	FileUpload(request FileUploadRequest) (FileUploadResponse, error)
	ImageGenerate(request ImageGenerationRequest) (ImageGenerationResponse, error)
	AudioGenerate(request AudioGenerationRequest) (AudioGenerationResponse, error)
}

// BaseAdapter provides default implementations for optional methods.
type BaseAdapter struct {
	Provider  string
	Tport     Transport
}

func (a *BaseAdapter) ProviderName() string { return a.Provider }

// GetTransport returns the underlying transport.
func (a *BaseAdapter) GetTransport() Transport { return a.Tport }

// Base returns the BaseAdapter (for client.go).
func (a *BaseAdapter) Base() *BaseAdapter { return a }

// Complete executes a non-streaming request.
func (a *BaseAdapter) Complete(adapter Adapter, request LMRequest) (LMResponse, error) {
	req := adapter.BuildRequest(request, false)
	resp, err := a.Tport.Request(req)
	if err != nil {
		return LMResponse{}, err
	}
	if resp.Status >= 400 {
		return LMResponse{}, adapter.NormalizeError(resp.Status, resp.Text())
	}
	return adapter.ParseResponse(request, resp)
}

// StreamEvents opens a streaming request and yields parsed events.
func (a *BaseAdapter) StreamEvents(adapter Adapter, request LMRequest) (<-chan StreamEvent, error) {
	req := adapter.BuildRequest(request, true)
	body, err := a.Tport.Stream(req)
	if err != nil {
		return nil, err
	}

	ch := make(chan StreamEvent, 32)
	go func() {
		defer close(ch)
		defer body.Close()
		for raw := range ParseSSE(body) {
			evt, err := adapter.ParseStreamEvent(request, raw)
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

// Default unsupported implementations
func (a *BaseAdapter) Embeddings(request EmbeddingRequest) (EmbeddingResponse, error) {
	return EmbeddingResponse{}, &UnsupportedFeatureError{ProviderError{ULMError{a.Provider + ": embeddings not supported"}}}
}
func (a *BaseAdapter) FileUpload(request FileUploadRequest) (FileUploadResponse, error) {
	return FileUploadResponse{}, &UnsupportedFeatureError{ProviderError{ULMError{a.Provider + ": files not supported"}}}
}
func (a *BaseAdapter) ImageGenerate(request ImageGenerationRequest) (ImageGenerationResponse, error) {
	return ImageGenerationResponse{}, &UnsupportedFeatureError{ProviderError{ULMError{a.Provider + ": images not supported"}}}
}
func (a *BaseAdapter) AudioGenerate(request AudioGenerationRequest) (AudioGenerationResponse, error) {
	return AudioGenerationResponse{}, &UnsupportedFeatureError{ProviderError{ULMError{a.Provider + ": audio not supported"}}}
}
