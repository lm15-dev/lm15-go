package lm15

import (
	"encoding/json"
	"io"
	"net/http"
	"time"
)

// ModelSpec describes a model from models.dev or live provider APIs.
type ModelSpec struct {
	ID               string
	Provider         string
	ContextWindow    *int
	MaxOutput        *int
	InputModalities  []string
	OutputModalities []string
	ToolCall         bool
	StructuredOutput bool
	Reasoning        bool
	Raw              map[string]any
}

// FetchModelsDev fetches the model catalog from models.dev.
func FetchModelsDev(timeout time.Duration) ([]ModelSpec, error) {
	client := &http.Client{Timeout: timeout}
	req, _ := http.NewRequest("GET", "https://models.dev/api.json", nil)
	req.Header.Set("User-Agent", "lm15")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	var out []ModelSpec
	providers, ok := data["providers"].(map[string]any)
	if !ok {
		providers = data
	}

	for providerID, providerPayload := range providers {
		pp, ok := providerPayload.(map[string]any)
		if !ok {
			continue
		}
		models, ok := pp["models"].(map[string]any)
		if !ok {
			continue
		}
		for modelID, modelPayload := range models {
			m, ok := modelPayload.(map[string]any)
			if !ok {
				continue
			}
			limit, _ := m["limit"].(map[string]any)
			modalities, _ := m["modalities"].(map[string]any)

			spec := ModelSpec{
				ID:       modelID,
				Provider: providerID,
				Raw:      m,
			}
			if limit != nil {
				if v, ok := limit["context"].(float64); ok {
					i := int(v)
					spec.ContextWindow = &i
				}
				if v, ok := limit["output"].(float64); ok {
					i := int(v)
					spec.MaxOutput = &i
				}
			}
			if modalities != nil {
				spec.InputModalities = toStringSlice(modalities["input"])
				spec.OutputModalities = toStringSlice(modalities["output"])
			}
			if v, ok := m["tool_call"].(bool); ok {
				spec.ToolCall = v
			}
			if v, ok := m["structured_output"].(bool); ok {
				spec.StructuredOutput = v
			}
			if v, ok := m["reasoning"].(bool); ok {
				spec.Reasoning = v
			}
			out = append(out, spec)
		}
	}
	return out, nil
}

func toStringSlice(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, x := range arr {
		if s, ok := x.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
