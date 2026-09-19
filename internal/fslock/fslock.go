// Package fslock is an advisory, cross-process, exclusive file lock with
// atomic private JSON writes (spec/auth.md AUTH-4). Unix uses flock,
// Windows LockFileEx, js/wasm an in-process mutex (no filesystem there).
package fslock

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"time"
)

// ErrTimeout is returned when the lock cannot be taken before the deadline.
var ErrTimeout = errors.New("fslock: timeout")

// LockDir is the lm15-owned lock directory ($LM15_LOCK_DIR, else
// $XDG_CACHE_HOME/lm15/locks, else ~/.cache/lm15/locks).
func LockDir(env func(string) string, home string) string {
	if v := env("LM15_LOCK_DIR"); v != "" {
		return expand(v, home)
	}
	if v := env("XDG_CACHE_HOME"); v != "" {
		return filepath.Join(expand(v, home), "lm15", "locks")
	}
	return filepath.Join(home, ".cache", "lm15", "locks")
}

func expand(p, home string) string {
	if len(p) > 0 && p[0] == '~' {
		return filepath.Join(home, p[1:])
	}
	return p
}

// LockPathFor is the deterministic lock-file path for a guarded file.
func LockPathFor(path string, env func(string) string, home string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		abs = resolved
	}
	sum := sha256.Sum256([]byte(abs))
	return filepath.Join(LockDir(env, home), hex.EncodeToString(sum[:])[:32]+".lock")
}

// Lock is a held lock.
type Lock struct {
	release func()
}

// Release frees the lock.
func (l *Lock) Release() {
	if l != nil && l.release != nil {
		l.release()
		l.release = nil
	}
}

// Acquire takes the exclusive lock at lockPath, polling until the context
// ends or the timeout elapses.
func Acquire(ctx context.Context, lockPath string, timeout time.Duration) (*Lock, error) {
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o700); err != nil {
		return nil, err
	}
	deadline := time.Now().Add(timeout)
	for {
		release, ok, err := tryLock(lockPath)
		if err != nil {
			return nil, err
		}
		if ok {
			return &Lock{release: release}, nil
		}
		if time.Now().After(deadline) {
			return nil, ErrTimeout
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}

// WritePrivateAtomic writes data to path via a private temp file, fsync
// and rename, so a reader sees the old or the new file, never a partial.
func WritePrivateAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	cleanup := func() { tmp.Close(); os.Remove(name) }
	if err := tmp.Chmod(0o600); err != nil && !errors.Is(err, errors.ErrUnsupported) {
		_ = err // best effort on platforms without modes
	}
	if _, err := tmp.Write(data); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return err
	}
	_ = os.Chmod(path, 0o600)
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		d.Close()
	}
	return nil
}
