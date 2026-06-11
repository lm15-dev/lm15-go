package lm15

import "encoding/json"

// Model metadata (the "model_info" serde kind). Spec gap: spec/types.md does
// not table these types; field shapes mirror the Python reference
// (lm15-python2/lm15/models.py + serde.py) per the task's disambiguation rule.

type InferencePricing struct {
	InputPerMillion      *float64 `json:"input_per_million"`
	OutputPerMillion     *float64 `json:"output_per_million"`
	CacheReadPerMillion  *float64 `json:"cache_read_per_million"`
	CacheWritePerMillion *float64 `json:"cache_write_per_million"`
	Currency             string   `json:"currency"`
	Dimensions           jmap     `json:"dimensions"`
}

func (p InferencePricing) toMap() jmap {
	m := jmap{}
	put := func(k string, v *float64) {
		if v != nil {
			m[k] = jsonFloat(*v)
		}
	}
	put("input_per_million", p.InputPerMillion)
	put("output_per_million", p.OutputPerMillion)
	put("cache_read_per_million", p.CacheReadPerMillion)
	put("cache_write_per_million", p.CacheWritePerMillion)
	putOpt(m, "currency", p.Currency)
	putOpt(m, "dimensions", p.Dimensions)
	return m
}

func inferencePricingFromMap(m jmap) (*InferencePricing, error) {
	var p InferencePricing
	for _, f := range []struct {
		key string
		dst **float64
	}{
		{"input_per_million", &p.InputPerMillion},
		{"output_per_million", &p.OutputPerMillion},
		{"cache_read_per_million", &p.CacheReadPerMillion},
		{"cache_write_per_million", &p.CacheWritePerMillion},
	} {
		v, err := optFloat(m, f.key)
		if err != nil {
			return nil, err
		}
		if v != nil && *v < 0 {
			return nil, valueErr("%s must be >= 0", f.key)
		}
		*f.dst = v
	}
	currency, err := optString(m, "currency", "USD")
	if err != nil {
		return nil, err
	}
	if currency == "" {
		return nil, valueErr("currency must be a non-empty string")
	}
	p.Currency = currency
	if p.Dimensions, err = optObj(m, "dimensions"); err != nil {
		return nil, err
	}
	return &p, nil
}

type TrainingPricing struct {
	TrainingTokensPerMillion *float64 `json:"training_tokens_per_million"`
	GPUSecond                *float64 `json:"gpu_second"`
	Currency                 string   `json:"currency"`
	Dimensions               jmap     `json:"dimensions"`
}

func (p TrainingPricing) toMap() jmap {
	m := jmap{}
	if p.TrainingTokensPerMillion != nil {
		m["training_tokens_per_million"] = jsonFloat(*p.TrainingTokensPerMillion)
	}
	if p.GPUSecond != nil {
		m["gpu_second"] = jsonFloat(*p.GPUSecond)
	}
	putOpt(m, "currency", p.Currency)
	putOpt(m, "dimensions", p.Dimensions)
	return m
}

func trainingPricingFromMap(m jmap) (*TrainingPricing, error) {
	var p TrainingPricing
	var err error
	if p.TrainingTokensPerMillion, err = optFloat(m, "training_tokens_per_million"); err != nil {
		return nil, err
	}
	if p.GPUSecond, err = optFloat(m, "gpu_second"); err != nil {
		return nil, err
	}
	if p.Currency, err = optString(m, "currency", "USD"); err != nil {
		return nil, err
	}
	if p.Dimensions, err = optObj(m, "dimensions"); err != nil {
		return nil, err
	}
	return &p, nil
}

type InferenceModelInfo struct {
	InputModalities   []string          `json:"input_modalities"`
	OutputModalities  []string          `json:"output_modalities"`
	ContextWindow     *int64            `json:"context_window"`
	MaxOutputTokens   *int64            `json:"max_output_tokens"`
	SupportsReasoning bool              `json:"supports_reasoning"`
	ReasoningEfforts  []string          `json:"reasoning_efforts"`
	Pricing           *InferencePricing `json:"pricing"`
	Extensions        jmap              `json:"extensions"`
}

func strsToAny(ss []string) []any {
	out := make([]any, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}

func (i InferenceModelInfo) toMap() jmap {
	m := jmap{}
	putOpt(m, "input_modalities", strsToAny(i.InputModalities))
	putOpt(m, "output_modalities", strsToAny(i.OutputModalities))
	if i.ContextWindow != nil {
		m["context_window"] = jsonInt(*i.ContextWindow)
	}
	if i.MaxOutputTokens != nil {
		m["max_output_tokens"] = jsonInt(*i.MaxOutputTokens)
	}
	if i.SupportsReasoning {
		m["supports_reasoning"] = true
	}
	putOpt(m, "reasoning_efforts", strsToAny(i.ReasoningEfforts))
	if i.Pricing != nil {
		putOpt(m, "pricing", i.Pricing.toMap())
	}
	putOpt(m, "extensions", i.Extensions)
	return m
}

func inferenceModelInfoFromMap(m jmap) (*InferenceModelInfo, error) {
	var i InferenceModelInfo
	var err error
	if i.InputModalities, err = stringList(m, "input_modalities"); err != nil {
		return nil, err
	}
	if i.InputModalities == nil {
		i.InputModalities = []string{"text"}
	}
	if i.OutputModalities, err = stringList(m, "output_modalities"); err != nil {
		return nil, err
	}
	if i.OutputModalities == nil {
		i.OutputModalities = []string{"text"}
	}
	if i.ContextWindow, err = optInt(m, "context_window"); err != nil {
		return nil, err
	}
	if i.MaxOutputTokens, err = optInt(m, "max_output_tokens"); err != nil {
		return nil, err
	}
	if (i.ContextWindow != nil && *i.ContextWindow <= 0) || (i.MaxOutputTokens != nil && *i.MaxOutputTokens <= 0) {
		return nil, valueErr("context_window/max_output_tokens must be positive")
	}
	if i.SupportsReasoning, err = optBool(m, "supports_reasoning", false); err != nil {
		return nil, err
	}
	if i.ReasoningEfforts, err = stringList(m, "reasoning_efforts"); err != nil {
		return nil, err
	}
	if obj, err := optObj(m, "pricing"); err != nil {
		return nil, err
	} else if obj != nil {
		if i.Pricing, err = inferencePricingFromMap(obj); err != nil {
			return nil, err
		}
	}
	if i.Extensions, err = optObj(m, "extensions"); err != nil {
		return nil, err
	}
	return &i, nil
}

type TrainingModelInfo struct {
	SupportsLora         bool             `json:"supports_lora"`
	SupportsFullFinetune bool             `json:"supports_full_finetune"`
	TrainableModalities  []string         `json:"trainable_modalities"`
	Pricing              *TrainingPricing `json:"pricing"`
	Extensions           jmap             `json:"extensions"`
}

func (t TrainingModelInfo) toMap() jmap {
	m := jmap{}
	if t.SupportsLora {
		m["supports_lora"] = true
	}
	if t.SupportsFullFinetune {
		m["supports_full_finetune"] = true
	}
	putOpt(m, "trainable_modalities", strsToAny(t.TrainableModalities))
	if t.Pricing != nil {
		putOpt(m, "pricing", t.Pricing.toMap())
	}
	putOpt(m, "extensions", t.Extensions)
	return m
}

func trainingModelInfoFromMap(m jmap) (*TrainingModelInfo, error) {
	var t TrainingModelInfo
	var err error
	if t.SupportsLora, err = optBool(m, "supports_lora", false); err != nil {
		return nil, err
	}
	if t.SupportsFullFinetune, err = optBool(m, "supports_full_finetune", false); err != nil {
		return nil, err
	}
	if t.TrainableModalities, err = stringList(m, "trainable_modalities"); err != nil {
		return nil, err
	}
	if obj, err := optObj(m, "pricing"); err != nil {
		return nil, err
	} else if obj != nil {
		if t.Pricing, err = trainingPricingFromMap(obj); err != nil {
			return nil, err
		}
	}
	if t.Extensions, err = optObj(m, "extensions"); err != nil {
		return nil, err
	}
	return &t, nil
}

type ModelOrigin struct {
	Type         string  `json:"type"`
	ID           *string `json:"id"`
	BaseModel    *string `json:"base_model"`
	ProviderData jmap    `json:"provider_data"`
}

func (o ModelOrigin) toMap() jmap {
	m := jmap{}
	putOpt(m, "type", o.Type)
	if o.ID != nil && *o.ID != "" {
		m["id"] = *o.ID
	}
	if o.BaseModel != nil && *o.BaseModel != "" {
		m["base_model"] = *o.BaseModel
	}
	putOpt(m, "provider_data", o.ProviderData)
	return m
}

func modelOriginFromMap(m jmap) (ModelOrigin, error) {
	var o ModelOrigin
	var err error
	if o.Type, err = optString(m, "type", "provider"); err != nil {
		return o, err
	}
	if o.Type == "" {
		return o, valueErr("ModelOrigin.type must be non-empty")
	}
	if o.ID, err = optStringPtr(m, "id"); err != nil {
		return o, err
	}
	if o.BaseModel, err = optStringPtr(m, "base_model"); err != nil {
		return o, err
	}
	if o.ProviderData, err = optObj(m, "provider_data"); err != nil {
		return o, err
	}
	return o, nil
}

type ModelInfo struct {
	ID         string              `json:"id"`
	Provider   string              `json:"provider"`
	APIFamily  string              `json:"api_family"`
	Aliases    []string            `json:"aliases"`
	Origin     ModelOrigin         `json:"origin"`
	Inference  *InferenceModelInfo `json:"inference"`
	Training   *TrainingModelInfo  `json:"training"`
	Extensions jmap                `json:"extensions"`
}

func (mi ModelInfo) toMap() jmap {
	origin := mi.Origin.toMap()
	if len(origin) == 1 && origin["type"] == "provider" {
		origin = jmap{} // the default origin carries no information
	}
	m := jmap{"id": mi.ID, "provider": mi.Provider, "api_family": mi.APIFamily}
	putOpt(m, "aliases", strsToAny(mi.Aliases))
	putOpt(m, "origin", origin)
	if mi.Inference != nil {
		putOpt(m, "inference", mi.Inference.toMap())
	}
	if mi.Training != nil {
		putOpt(m, "training", mi.Training.toMap())
	}
	putOpt(m, "extensions", mi.Extensions)
	return m
}

func modelInfoFromMap(m jmap) (ModelInfo, error) {
	var mi ModelInfo
	var err error
	if mi.ID, err = reqString(m, "id"); err != nil {
		return mi, err
	}
	if mi.Provider, err = reqString(m, "provider"); err != nil {
		return mi, err
	}
	if mi.APIFamily, err = reqString(m, "api_family"); err != nil {
		return mi, err
	}
	if mi.ID == "" || mi.Provider == "" || mi.APIFamily == "" {
		return mi, valueErr("ModelInfo id/provider/api_family must be non-empty")
	}
	if mi.Aliases, err = stringList(m, "aliases"); err != nil {
		return mi, err
	}
	if obj, err := optObj(m, "origin"); err != nil {
		return mi, err
	} else if obj != nil {
		if mi.Origin, err = modelOriginFromMap(obj); err != nil {
			return mi, err
		}
	} else {
		mi.Origin = ModelOrigin{Type: "provider"}
	}
	if obj, err := optObj(m, "inference"); err != nil {
		return mi, err
	} else if obj != nil {
		if mi.Inference, err = inferenceModelInfoFromMap(obj); err != nil {
			return mi, err
		}
	}
	if obj, err := optObj(m, "training"); err != nil {
		return mi, err
	} else if obj != nil {
		if mi.Training, err = trainingModelInfoFromMap(obj); err != nil {
			return mi, err
		}
	}
	if mi.Extensions, err = optObj(m, "extensions"); err != nil {
		return mi, err
	}
	return mi, nil
}

func (mi ModelInfo) MarshalJSON() ([]byte, error) { return json.Marshal(mi.toMap()) }

func (mi *ModelInfo) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := modelInfoFromMap(m)
	if err != nil {
		return err
	}
	*mi = v
	return nil
}
