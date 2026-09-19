//go:build js

package fslock

import "sync"

// The browser has no shared filesystem: the lock serializes goroutines of
// this one process only (stated in the README).
var (
	mu   sync.Mutex
	held = map[string]bool{}
)

func tryLock(lockPath string) (func(), bool, error) {
	mu.Lock()
	defer mu.Unlock()
	if held[lockPath] {
		return nil, false, nil
	}
	held[lockPath] = true
	return func() {
		mu.Lock()
		delete(held, lockPath)
		mu.Unlock()
	}, true, nil
}
