package lm15

import (
	"fmt"
	"time"
)

// CompleteFn is a function that performs a complete request.
type CompleteFn func(req LMRequest) (LMResponse, error)

// CompleteMiddleware wraps a CompleteFn.
type CompleteMiddleware func(req LMRequest, next CompleteFn) (LMResponse, error)

// MiddlewarePipeline chains middleware for complete operations.
type MiddlewarePipeline struct {
	completeMw []CompleteMiddleware
}

// Add appends a middleware to the pipeline.
func (p *MiddlewarePipeline) Add(mw CompleteMiddleware) {
	p.completeMw = append(p.completeMw, mw)
}

// WrapComplete wraps a CompleteFn with all middleware.
func (p *MiddlewarePipeline) WrapComplete(fn CompleteFn) CompleteFn {
	wrapped := fn
	for i := len(p.completeMw) - 1; i >= 0; i-- {
		mw := p.completeMw[i]
		prev := wrapped
		wrapped = func(req LMRequest) (LMResponse, error) {
			return mw(req, prev)
		}
	}
	return wrapped
}

// WithRetries retries on transient errors.
func WithRetries(maxRetries int, sleepBase time.Duration) CompleteMiddleware {
	return func(req LMRequest, next CompleteFn) (LMResponse, error) {
		var lastErr error
		for i := 0; i <= maxRetries; i++ {
			resp, err := next(req)
			if err == nil {
				return resp, nil
			}
			lastErr = err
			if i == maxRetries || !IsTransient(err) {
				return LMResponse{}, err
			}
			time.Sleep(sleepBase * time.Duration(1<<uint(i)))
		}
		return LMResponse{}, lastErr
	}
}

// WithCache caches responses by request key.
func WithCache(cache map[string]LMResponse) CompleteMiddleware {
	return func(req LMRequest, next CompleteFn) (LMResponse, error) {
		key := fmt.Sprintf("%v", req)
		if cached, ok := cache[key]; ok {
			return cached, nil
		}
		resp, err := next(req)
		if err != nil {
			return resp, err
		}
		cache[key] = resp
		return resp, nil
	}
}

// MiddlewareHistoryEntry is a log entry from WithHistory.
type MiddlewareHistoryEntry struct {
	Timestamp    time.Time
	Model        string
	Messages     int
	FinishReason FinishReason
	Usage        Usage
}

// WithHistory logs each request/response.
func WithHistory(history *[]MiddlewareHistoryEntry) CompleteMiddleware {
	return func(req LMRequest, next CompleteFn) (LMResponse, error) {
		started := time.Now()
		resp, err := next(req)
		if err != nil {
			return resp, err
		}
		*history = append(*history, MiddlewareHistoryEntry{
			Timestamp:    started,
			Model:        req.Model,
			Messages:     len(req.Messages),
			FinishReason: resp.FinishReason,
			Usage:        resp.Usage,
		})
		return resp, nil
	}
}
