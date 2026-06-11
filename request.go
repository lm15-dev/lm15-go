package lm15

import "encoding/json"

// systemToJSON renders Request/LiveConfig system: nil, a string, or []Part.
func systemToJSON(system any) (any, error) {
	switch s := system.(type) {
	case nil:
		return nil, nil
	case string:
		return s, nil
	case []Part:
		out := make([]any, len(s))
		for i, p := range s {
			out[i] = p.toMap()
		}
		return out, nil
	}
	return nil, typeErr("system must be a string or a list of parts")
}

func systemFromJSON(m jmap) (any, error) {
	v, ok := m["system"]
	if !ok || v == nil {
		return nil, nil
	}
	switch s := v.(type) {
	case string:
		if s == "" {
			return nil, nil
		}
		return s, nil
	case []any:
		parts := make([]Part, 0, len(s))
		for _, item := range s {
			obj, ok := item.(jmap)
			if !ok {
				return nil, typeErr("system parts must be objects")
			}
			p, err := PartFromMap(obj)
			if err != nil {
				return nil, err
			}
			parts = append(parts, p)
		}
		return parts, nil
	}
	return nil, typeErr("system must be a string or a list of parts")
}

func toolsToJSON(tools []Tool) []any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]any, len(tools))
	for i, t := range tools {
		out[i] = t.toMap()
	}
	return out
}

func toolsFromJSON(m jmap) ([]Tool, error) {
	raw, err := optList(m, "tools")
	if err != nil {
		return nil, err
	}
	var tools []Tool
	seen := map[string]bool{}
	for _, item := range raw {
		obj, ok := item.(jmap)
		if !ok {
			return nil, typeErr("tools entries must be objects")
		}
		t, err := ToolFromMap(obj)
		if err != nil {
			return nil, err
		}
		if seen[t.ToolName()] {
			return nil, valueErr("duplicate tool name: %s", t.ToolName())
		}
		seen[t.ToolName()] = true
		tools = append(tools, t)
	}
	return tools, nil
}

// ─── Request ─────────────────────────────────────────────────────────

type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	System   any       `json:"system"` // string or []Part
	Tools    []Tool    `json:"tools"`
	Config   *Config   `json:"config"`
}

func (r Request) toMap() (jmap, error) {
	messages := make([]any, len(r.Messages))
	for i, msg := range r.Messages {
		messages[i] = msg.toMap()
	}
	m := jmap{"model": r.Model, "messages": messages}
	system, err := systemToJSON(r.System)
	if err != nil {
		return nil, err
	}
	putOpt(m, "system", system)
	putOpt(m, "tools", toolsToJSON(r.Tools))
	if r.Config != nil {
		putOpt(m, "config", r.Config.toMap())
	}
	return m, nil
}

func requestFromMap(m jmap) (Request, error) {
	model, err := reqString(m, "model")
	if err != nil {
		return Request{}, err
	}
	if model == "" {
		return Request{}, valueErr("request model must be non-empty")
	}
	rawMessages, err := optList(m, "messages")
	if err != nil {
		return Request{}, err
	}
	var messages []Message
	for _, item := range rawMessages {
		obj, ok := item.(jmap)
		if !ok {
			return Request{}, typeErr("messages entries must be objects")
		}
		msg, err := messageFromMap(obj)
		if err != nil {
			return Request{}, err
		}
		messages = append(messages, msg)
	}
	if len(messages) == 0 {
		return Request{}, valueErr("request messages must be non-empty")
	}
	system, err := systemFromJSON(m)
	if err != nil {
		return Request{}, err
	}
	tools, err := toolsFromJSON(m)
	if err != nil {
		return Request{}, err
	}
	var config *Config
	if obj, err := optObj(m, "config"); err != nil {
		return Request{}, err
	} else if obj != nil {
		if config, err = configFromMap(obj); err != nil {
			return Request{}, err
		}
	} else {
		config = &Config{}
	}
	return Request{Model: model, Messages: messages, System: system, Tools: tools, Config: config}, nil
}

func (r Request) MarshalJSON() ([]byte, error) {
	m, err := r.toMap()
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

func (r *Request) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := requestFromMap(m)
	if err != nil {
		return err
	}
	*r = v
	return nil
}

// ─── Response ────────────────────────────────────────────────────────

type Response struct {
	ID           *string `json:"id"`
	Model        string  `json:"model"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
	Usage        Usage   `json:"usage"`
	ProviderData jmap    `json:"provider_data"`
}

// toMap serializes WITHOUT provider_data (the vet protocol's default form).
func (r Response) toMap() jmap {
	m := jmap{
		"model":         r.Model,
		"message":       r.Message.toMap(),
		"finish_reason": r.FinishReason,
	}
	if r.ID != nil && *r.ID != "" {
		m["id"] = *r.ID
	}
	putOpt(m, "usage", r.Usage.toMap())
	return m
}

func responseFromMap(m jmap) (Response, error) {
	id, err := optStringPtr(m, "id")
	if err != nil {
		return Response{}, err
	}
	model, err := reqString(m, "model")
	if err != nil {
		return Response{}, err
	}
	msgObj, err := optObj(m, "message")
	if err != nil {
		return Response{}, err
	}
	if msgObj == nil {
		return Response{}, keyErr("message")
	}
	msg, err := messageFromMap(msgObj)
	if err != nil {
		return Response{}, err
	}
	if msg.Role != "assistant" {
		return Response{}, valueErr("response message role must be assistant")
	}
	finish, err := reqString(m, "finish_reason")
	if err != nil {
		return Response{}, err
	}
	if !inVocab(FinishReasons, finish) {
		return Response{}, valueErr("unsupported finish reason: %s", finish)
	}
	var usage Usage
	if obj, err := optObj(m, "usage"); err != nil {
		return Response{}, err
	} else if obj != nil {
		if usage, err = usageFromMap(obj); err != nil {
			return Response{}, err
		}
	}
	providerData, err := optObj(m, "provider_data")
	if err != nil {
		return Response{}, err
	}
	return Response{ID: id, Model: model, Message: msg, FinishReason: finish, Usage: usage, ProviderData: providerData}, nil
}

func (r Response) MarshalJSON() ([]byte, error) { return json.Marshal(r.toMap()) }

func (r *Response) UnmarshalJSON(data []byte) error {
	m, err := decodeMap(data)
	if err != nil {
		return err
	}
	v, err := responseFromMap(m)
	if err != nil {
		return err
	}
	*r = v
	return nil
}
