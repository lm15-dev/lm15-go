package lm15

// Non-chat endpoint types (PROVISIONAL per spec/SCOPE.md — stubs for Stage A;
// canonical shapes from spec/types.md "Other endpoints").

type EmbeddingRequest struct {
	Model      string   `json:"model"`
	Inputs     []string `json:"inputs"`
	Extensions jmap     `json:"extensions"`
}

type EmbeddingResponse struct {
	Model        string      `json:"model"`
	Vectors      [][]float64 `json:"vectors"`
	Usage        Usage       `json:"usage"`
	ProviderData jmap        `json:"provider_data"`
}

// FileUploadRequest is in-memory only (no canonical JSON serializer).
type FileUploadRequest struct {
	Filename   string `json:"filename"`
	BytesData  []byte `json:"bytes_data"`
	MediaType  string `json:"media_type"`
	Model      string `json:"model"`
	Extensions jmap   `json:"extensions"`
	Path       string `json:"path"`
}

type FileUploadResponse struct {
	ID           string `json:"id"`
	ProviderData jmap   `json:"provider_data"`
}

type BatchRequest struct {
	Model      string    `json:"model"`
	Requests   []Request `json:"requests"`
	Extensions jmap      `json:"extensions"`
}

type BatchResponse struct {
	ID           string `json:"id"`
	Status       string `json:"status"`
	ProviderData jmap   `json:"provider_data"`
}

type ImageGenerationRequest struct {
	Model      string  `json:"model"`
	Prompt     string  `json:"prompt"`
	Size       *string `json:"size"`
	Extensions jmap    `json:"extensions"`
}

type ImageGenerationResponse struct {
	Images       []ImagePart `json:"images"`
	ID           *string     `json:"id"`
	Model        *string     `json:"model"`
	Usage        Usage       `json:"usage"`
	ProviderData jmap        `json:"provider_data"`
}

type AudioGenerationRequest struct {
	Model      string  `json:"model"`
	Prompt     string  `json:"prompt"`
	Voice      *string `json:"voice"`
	Format     *string `json:"format"`
	Extensions jmap    `json:"extensions"`
}

type AudioGenerationResponse struct {
	Audio        AudioPart `json:"audio"`
	ID           *string   `json:"id"`
	Model        *string   `json:"model"`
	Usage        Usage     `json:"usage"`
	ProviderData jmap      `json:"provider_data"`
}

// ToolCallInfo is the in-memory callback view of a ToolCallPart.
type ToolCallInfo struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input jmap   `json:"input"`
}

// FromPart builds a ToolCallInfo from a ToolCallPart.
func ToolCallInfoFromPart(p ToolCallPart) ToolCallInfo {
	return ToolCallInfo{ID: p.ID, Name: p.Name, Input: p.Input}
}

// ToPart converts back to a ToolCallPart.
func (i ToolCallInfo) ToPart() ToolCallPart {
	return ToolCallPart{ID: i.ID, Name: i.Name, Input: i.Input}
}
