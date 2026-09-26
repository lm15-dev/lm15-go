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
	"runtime"
	"strings"
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

// LockPathFor is the deterministic lock-file path for a guarded file: the
// AUTH-4 lock identity, including missing leaves and dangling symlinks.
// Ordinary realpath hashes are unchanged; a path whose leaf does not exist
// yet resolves every existing component in filesystem order (so the
// process that creates the file and the one that later reads it agree),
// and a resolution error falls back to the absolute path rather than
// guessing. On Windows the key is lowercased with verbatim prefixes
// removed, deliberately over-locking case-sensitive directories.
func LockPathFor(path string, env func(string) string, home string) string {
	key, err := RealPathAllowMissing(path)
	if err != nil {
		if abs, aerr := filepath.Abs(path); aerr == nil {
			key = abs
		} else {
			key = path
		}
	}
	if runtime.GOOS == "windows" {
		key = strings.ToLower(stripWindowsVerbatim(key))
	}
	sum := sha256.Sum256([]byte(key))
	return filepath.Join(LockDir(env, home), hex.EncodeToString(sum[:])[:32]+".lock")
}

func stripWindowsVerbatim(path string) string {
	path = strings.ReplaceAll(path, "/", "\\")
	if len(path) >= 8 && strings.EqualFold(path[:8], "\\\\?\\unc\\") {
		return "\\\\" + path[8:]
	}
	if strings.HasPrefix(path, "\\\\?\\") {
		return path[4:]
	}
	return path
}

// RealPathAllowMissing resolves symlinks component by component in
// filesystem order; only a missing component is recoverable (the walk
// continues, so a later ".." can return to an existing ancestor). Forty
// link expansions bound loops, like every other SDK.
func RealPathAllowMissing(target string) (string, error) {
	if strings.ContainsRune(target, 0) {
		return "", errors.New("fslock: NUL in credential path")
	}
	if target == "~" || strings.HasPrefix(target, "~/") || strings.HasPrefix(target, "~\\") {
		return "", errors.New("fslock: no home directory for credential lock path")
	}
	if strings.HasPrefix(target, "~") {
		return "", errors.New("fslock: named-user home expansion is unsupported for credential locks")
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if cwd, err = filepath.EvalSymlinks(cwd); err != nil {
		return "", err
	}
	resolved, pending := splitPath(target, cwd)
	links := 0
	for len(pending) > 0 {
		name := pending[0]
		pending = pending[1:]
		switch name {
		case ".":
			continue
		case "..":
			resolved = filepath.Dir(resolved)
			continue
		}
		candidate := filepath.Join(resolved, name)
		info, err := os.Lstat(candidate)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				// Windows compares names with the filesystem's upcase
				// table, not Unicode case folding: with no on-disk spelling
				// to canonicalize, a non-ASCII name could hash apart from
				// the same file named by another SDK. Refuse, as they do.
				if runtime.GOOS == "windows" && !isASCII(name) {
					return "", errors.New("fslock: a missing non-ASCII Windows credential path component is unsupported")
				}
				resolved = candidate
				continue
			}
			return "", err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			links++
			if links > 40 {
				return "", errors.New("fslock: too many credential path symlinks")
			}
			linked, err := os.Readlink(candidate)
			if err != nil {
				return "", err
			}
			base, rest := splitPath(linked, resolved)
			resolved = base
			pending = append(rest, pending...)
			continue
		}
		resolved = candidate
		if runtime.GOOS == "windows" {
			// One file, one identity: an existing component is spelled as
			// the filesystem stores it (long name for an 8.3 short name
			// such as RUNNER~1, stored case), as every other SDK's realpath
			// spells it. EvalSymlinks normalizes on Windows; the walk has
			// already expanded every link, so it changes the spelling only.
			long, err := filepath.EvalSymlinks(candidate)
			if err != nil {
				return "", err
			}
			resolved = stripWindowsVerbatim(long)
		}
	}
	return resolved, nil
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

// splitPath separates a path into the base it starts from (the root for an
// absolute path, base otherwise) and its remaining components.
func splitPath(value, base string) (string, []string) {
	if filepath.IsAbs(value) {
		vol := filepath.VolumeName(value)
		base = vol + string(filepath.Separator)
		value = value[len(vol):]
	}
	var names []string
	for _, part := range strings.FieldsFunc(value, func(r rune) bool { return r == '/' || r == filepath.Separator }) {
		names = append(names, part)
	}
	return base, names
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
