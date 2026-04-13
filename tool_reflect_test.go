package lm15

import (
	"fmt"
	"testing"
)

type WeatherInput struct {
	City string `json:"city" description:"The city to get weather for"`
	Unit string `json:"unit,omitempty" description:"celsius or fahrenheit" enum:"celsius,fahrenheit"`
}

func getWeather(input WeatherInput) (string, error) {
	return fmt.Sprintf("22°C in %s", input.City), nil
}

func TestToolFromFunc(t *testing.T) {
	tool := ToolFromFunc("get_weather", "Get weather by city", getWeather)

	if tool.Type != "function" {
		t.Errorf("expected function, got %s", tool.Type)
	}
	if tool.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", tool.Name)
	}
	if tool.Description != "Get weather by city" {
		t.Errorf("expected description, got %s", tool.Description)
	}

	// Check schema
	props, ok := tool.Parameters["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}
	cityProp, ok := props["city"].(map[string]any)
	if !ok {
		t.Fatal("expected city property")
	}
	if cityProp["type"] != "string" {
		t.Errorf("expected string type for city, got %v", cityProp["type"])
	}
	if cityProp["description"] != "The city to get weather for" {
		t.Errorf("expected description for city, got %v", cityProp["description"])
	}

	// Check unit has enum
	unitProp, ok := props["unit"].(map[string]any)
	if !ok {
		t.Fatal("expected unit property")
	}
	enumVals, ok := unitProp["enum"].([]string)
	if !ok || len(enumVals) != 2 {
		t.Errorf("expected enum with 2 values, got %v", unitProp["enum"])
	}

	// Check required
	required, ok := tool.Parameters["required"].([]string)
	if !ok {
		t.Fatal("expected required array")
	}
	if len(required) != 1 || required[0] != "city" {
		t.Errorf("expected [city] required, got %v", required)
	}
}

func TestToolFromFuncExecution(t *testing.T) {
	tool := ToolFromFunc("get_weather", "Get weather", getWeather)

	result, err := tool.Fn(map[string]any{"city": "Montreal"})
	if err != nil {
		t.Fatal(err)
	}
	if result != "22°C in Montreal" {
		t.Errorf("expected '22°C in Montreal', got %v", result)
	}
}

type MathInput struct {
	A int `json:"a" description:"First number"`
	B int `json:"b" description:"Second number"`
}

func add(input MathInput) int {
	return input.A + input.B
}

func TestToolFromFuncSingleReturn(t *testing.T) {
	tool := ToolFromFunc("add", "Add two numbers", add)

	result, err := tool.Fn(map[string]any{"a": float64(3), "b": float64(4)})
	if err != nil {
		t.Fatal(err)
	}
	if result != 7 {
		t.Errorf("expected 7, got %v", result)
	}

	// Check schema
	props := tool.Parameters["properties"].(map[string]any)
	aProp := props["a"].(map[string]any)
	if aProp["type"] != "integer" {
		t.Errorf("expected integer type for a, got %v", aProp["type"])
	}
}

func TestToolFromFuncNoArgs(t *testing.T) {
	fn := func() string { return "hello" }
	tool := ToolFromFunc("greet", "Say hello", fn)

	result, err := tool.Fn(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if result != "hello" {
		t.Errorf("expected hello, got %v", result)
	}
}

type ArrayInput struct {
	Tags []string `json:"tags" description:"List of tags"`
}

func TestToolFromFuncArrayField(t *testing.T) {
	fn := func(input ArrayInput) string { return fmt.Sprintf("%d tags", len(input.Tags)) }
	tool := ToolFromFunc("tag", "Tag stuff", fn)

	props := tool.Parameters["properties"].(map[string]any)
	tagsProp := props["tags"].(map[string]any)
	if tagsProp["type"] != "array" {
		t.Errorf("expected array type, got %v", tagsProp["type"])
	}
	items := tagsProp["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("expected string items, got %v", items["type"])
	}

	result, _ := tool.Fn(map[string]any{"tags": []any{"a", "b", "c"}})
	if result != "3 tags" {
		t.Errorf("expected '3 tags', got %v", result)
	}
}

type BoolInput struct {
	Verbose bool    `json:"verbose" description:"Enable verbose output"`
	Factor  float64 `json:"factor" description:"Scale factor"`
}

func TestToolFromFuncBoolFloat(t *testing.T) {
	fn := func(input BoolInput) string { return fmt.Sprintf("v=%v f=%.1f", input.Verbose, input.Factor) }
	tool := ToolFromFunc("test", "Test", fn)

	props := tool.Parameters["properties"].(map[string]any)
	if props["verbose"].(map[string]any)["type"] != "boolean" {
		t.Error("expected boolean")
	}
	if props["factor"].(map[string]any)["type"] != "number" {
		t.Error("expected number")
	}

	result, _ := tool.Fn(map[string]any{"verbose": true, "factor": 2.5})
	if result != "v=true f=2.5" {
		t.Errorf("unexpected: %v", result)
	}
}
