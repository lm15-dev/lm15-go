package lm15

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// ToolFromFunc creates a Tool from a Go function.
//
// The function must take a single struct argument (or pointer to struct) and
// return (any, error) or just a single value. The struct's fields are reflected
// to build the JSON Schema for the tool's parameters.
//
// Field names come from `json` tags. Field descriptions come from `description`
// tags. Required fields are those without `omitempty` in their json tag.
//
// Example:
//
//	type WeatherInput struct {
//	    City string `json:"city" description:"The city to get weather for"`
//	    Unit string `json:"unit,omitempty" description:"celsius or fahrenheit"`
//	}
//
//	func GetWeather(input WeatherInput) (string, error) {
//	    return fmt.Sprintf("22°C in %s", input.City), nil
//	}
//
//	tool := lm15.ToolFromFunc("get_weather", "Get weather by city", GetWeather)
func ToolFromFunc(name, description string, fn any) Tool {
	v := reflect.ValueOf(fn)
	t := v.Type()

	if t.Kind() != reflect.Func {
		panic(fmt.Sprintf("ToolFromFunc: expected a function, got %s", t.Kind()))
	}

	// Extract input struct type
	var inputType reflect.Type
	if t.NumIn() == 1 {
		inputType = t.In(0)
		if inputType.Kind() == reflect.Ptr {
			inputType = inputType.Elem()
		}
		if inputType.Kind() != reflect.Struct {
			panic(fmt.Sprintf("ToolFromFunc: function argument must be a struct, got %s", inputType.Kind()))
		}
	} else if t.NumIn() == 0 {
		// No-arg function — no parameters
		inputType = nil
	} else {
		panic(fmt.Sprintf("ToolFromFunc: function must take 0 or 1 arguments, got %d", t.NumIn()))
	}

	// Build JSON Schema from struct
	parameters := structToJSONSchema(inputType)

	// Build the wrapper Fn
	wrapperFn := buildWrapper(v, t, inputType)

	return Tool{
		Type:        "function",
		Name:        name,
		Description: description,
		Parameters:  parameters,
		Fn:          wrapperFn,
	}
}

// structToJSONSchema reflects on a struct type to produce a JSON Schema.
func structToJSONSchema(t reflect.Type) map[string]any {
	if t == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}

	properties := map[string]any{}
	var required []string

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		name, opts := parseJSONTag(jsonTag)
		if name == "" {
			name = field.Name
		}

		prop := goTypeToJSONSchema(field.Type)

		if desc := field.Tag.Get("description"); desc != "" {
			prop["description"] = desc
		}

		// Enum support via tag
		if enum := field.Tag.Get("enum"); enum != "" {
			prop["enum"] = strings.Split(enum, ",")
		}

		properties[name] = prop

		if !opts.omitempty {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// goTypeToJSONSchema maps a Go type to a JSON Schema type.
func goTypeToJSONSchema(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Slice, reflect.Array:
		items := goTypeToJSONSchema(t.Elem())
		return map[string]any{"type": "array", "items": items}
	case reflect.Map:
		return map[string]any{"type": "object"}
	case reflect.Ptr:
		return goTypeToJSONSchema(t.Elem())
	default:
		return map[string]any{"type": "string"}
	}
}

type jsonTagOpts struct {
	omitempty bool
}

func parseJSONTag(tag string) (string, jsonTagOpts) {
	if tag == "" {
		return "", jsonTagOpts{}
	}
	parts := strings.Split(tag, ",")
	name := parts[0]
	opts := jsonTagOpts{}
	for _, p := range parts[1:] {
		if p == "omitempty" {
			opts.omitempty = true
		}
	}
	return name, opts
}

// buildWrapper creates a func(map[string]any) (any, error) that unmarshals
// the map into the struct and calls the original function.
func buildWrapper(v reflect.Value, t reflect.Type, inputType reflect.Type) func(map[string]any) (any, error) {
	return func(args map[string]any) (any, error) {
		var results []reflect.Value

		if inputType == nil {
			// No-arg function
			results = v.Call(nil)
		} else {
			// Marshal map → JSON → unmarshal into struct
			jsonBytes, err := json.Marshal(args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal tool args: %w", err)
			}

			inputPtr := reflect.New(inputType)
			if err := json.Unmarshal(jsonBytes, inputPtr.Interface()); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool args into %s: %w", inputType.Name(), err)
			}

			input := inputPtr.Elem()

			// Handle pointer vs value receiver
			if t.In(0).Kind() == reflect.Ptr {
				results = v.Call([]reflect.Value{inputPtr})
			} else {
				results = v.Call([]reflect.Value{input})
			}
		}

		// Extract return values
		switch len(results) {
		case 0:
			return nil, nil
		case 1:
			// Could be (value) or (error)
			r := results[0]
			if r.Type().Implements(reflect.TypeOf((*error)(nil)).Elem()) {
				if r.IsNil() {
					return nil, nil
				}
				return nil, r.Interface().(error)
			}
			return r.Interface(), nil
		case 2:
			// (value, error)
			val := results[0].Interface()
			if results[1].IsNil() {
				return val, nil
			}
			return val, results[1].Interface().(error)
		default:
			return nil, fmt.Errorf("tool function returned %d values, expected 0-2", len(results))
		}
	}
}
