package provider

import (
	"strings"
	"time"

	lm15 "github.com/lm15-dev/lm15-go"
)

func partsToText(parts []lm15.Part) string {
	var texts []string
	for _, p := range parts {
		if p.Type == lm15.PartText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func toString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	default:
		return 0
	}
}

func toFloat64Slice(v any) []float64 {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]float64, len(arr))
	for i, x := range arr {
		if f, ok := x.(float64); ok {
			out[i] = f
		}
	}
	return out
}

func toSlice(v any) []any {
	if v == nil {
		return nil
	}
	if s, ok := v.([]any); ok {
		return s
	}
	return nil
}

func orDefault(s, fallback string) string {
	if s != "" {
		return s
	}
	return fallback
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func intPtr(i int) *int { return &i }

func durationMs(ms int) time.Duration {
	return time.Duration(ms) * time.Millisecond
}

func boolPtr(b bool) *bool { return &b }
