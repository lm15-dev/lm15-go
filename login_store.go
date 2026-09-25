package lm15

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

// The managed credential store (spec/auth-managed.md AUTH-25): port of
// lm15-python lm15/login/store.py, in the layout of lm15-contract
// auth/managed/store-layout.md — a file one SDK writes is another's. One JSON
// object per scope: provider entries (secret) plus one non-secret "_lm15"
// block. An unreadable or unrecognised document is a typed AuthOperationError
// (storage_unavailable / unsupported_store_version), never an empty store and
// never overwritten.

const (
	storeMetaKey = "_lm15"
	storeVersion = 1
)

func storageError(message, reason, stage string) *Error {
	if reason == "" {
		reason = "storage_unavailable"
	}
	if stage == "" {
		stage = "persistence"
	}
	e := authOperation(message, reason, stage, "not_committed", "repair_storage")
	e.Operation = "store"
	return e
}

// validateDocument rejects anything that is not the document shape.
func validateDocument(data any, where string) (JSONObject, error) {
	doc, ok := data.(map[string]any)
	if !ok {
		return nil, storageError(fmt.Sprintf("Credential store at %s is not a JSON object; not touching it.", where), "", "")
	}
	if raw, present := doc[storeMetaKey]; present {
		meta, ok := raw.(map[string]any)
		if !ok {
			return nil, storageError(fmt.Sprintf("Credential store at %s has a malformed %q block; not touching it.", where, storeMetaKey), "", "")
		}
		version, isNumber := meta["version"].(json.Number)
		if !isNumber {
			if f, ok := meta["version"].(int); ok {
				version, isNumber = json.Number(strconv.Itoa(f)), true
			}
		}
		if !isNumber || string(version) != strconv.Itoa(storeVersion) {
			shown := "None"
			if meta["version"] != nil {
				shown = fmt.Sprint(meta["version"])
			}
			return nil, storageError(fmt.Sprintf("Credential store at %s is managed-store version %s; this lm15 reads version %d. Upgrade lm15 or point LM15_CREDENTIALS_PATH at another file.", where, shown, storeVersion), "unsupported_store_version", "")
		}
		if slots, present := meta["slots"]; present {
			m, ok := slots.(map[string]any)
			if !ok {
				return nil, storageError(fmt.Sprintf("Credential store at %s has malformed slot metadata; not touching it.", where), "", "")
			}
			for _, v := range m {
				if _, ok := v.(map[string]any); !ok {
					return nil, storageError(fmt.Sprintf("Credential store at %s has malformed slot metadata; not touching it.", where), "", "")
				}
			}
		}
	}
	for key, value := range doc {
		if key == storeMetaKey {
			continue
		}
		if _, ok := value.(map[string]any); !ok {
			return nil, storageError(fmt.Sprintf("Credential store at %s: entry %q is not an object; not touching it.", where, key), "", "")
		}
	}
	return doc, nil
}

func copyDocument(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, item := range x {
			out[k] = copyDocument(item)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = copyDocument(item)
		}
		return out
	case []string:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = item
		}
		return out
	case map[string]string:
		out := make(map[string]any, len(x))
		for k, item := range x {
			out[k] = item
		}
		return out
	}
	return v
}

// StoreGuard is the store, locked: Read is the document now; Write replaces
// it durably (a guard may write more than once, AUTH-20.4). Unlock releases.
type StoreGuard interface {
	Read() (JSONObject, error)
	Write(document JSONObject) error
	Unlock()
}

// Store is the document contract every store implements (file, memory, an application's own).
type Store interface {
	// Description is where it lives, for people ("memory", a path). Never contents.
	Description() string
	// Read is a private copy of the whole document, unlocked: for status and selection.
	Read() (JSONObject, error)
	// Lock takes exclusive access (cross-process where the backend can).
	Lock(ctx context.Context) (StoreGuard, error)
	// Reserve proves the store can be written before any external authorization starts (AUTH-17).
	Reserve(ctx context.Context) error
}

// mutateStore is a serialized read-modify-write: update gets a private copy and
// returns the new document, or nil to leave the store untouched.
func mutateStore(ctx context.Context, s Store, update func(JSONObject) (JSONObject, error)) (JSONObject, error) {
	guard, err := s.Lock(ctx)
	if err != nil {
		return nil, err
	}
	defer guard.Unlock()
	current, err := guard.Read()
	if err != nil {
		return nil, err
	}
	next, err := update(copyDocument(current).(map[string]any))
	if err != nil {
		return nil, err
	}
	if next == nil {
		return current, nil
	}
	if err := guard.Write(next); err != nil {
		return nil, err
	}
	return next, nil
}

// MemoryStore is a process-lifetime document: Auth.Memory, tests, short-lived tools.
type MemoryStore struct {
	mu   sync.Mutex
	gate chan struct{}
	data JSONObject
}

// NewMemoryStore is an empty memory store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{gate: make(chan struct{}, 1), data: JSONObject{}}
}

func (m *MemoryStore) Description() string { return "memory" }

func (m *MemoryStore) Read() (JSONObject, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return copyDocument(m.data).(map[string]any), nil
}

type memoryGuard struct{ store *MemoryStore }

func (g memoryGuard) Read() (JSONObject, error) { return g.store.Read() }
func (g memoryGuard) Write(document JSONObject) error {
	checked, err := validateDocument(normalizeDocument(document), "memory")
	if err != nil {
		return err
	}
	g.store.mu.Lock()
	g.store.data = checked
	g.store.mu.Unlock()
	return nil
}
func (g memoryGuard) Unlock() { <-g.store.gate }

func (m *MemoryStore) Lock(ctx context.Context) (StoreGuard, error) {
	select {
	case m.gate <- struct{}{}:
		return memoryGuard{m}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m *MemoryStore) Reserve(context.Context) error { return nil }

// normalizeDocument round-trips a document through JSON so what a store
// holds is exactly what a file would (numbers as json.Number).
func normalizeDocument(document JSONObject) any {
	data, err := marshalDocument(document)
	if err != nil {
		return document
	}
	v, err := DecodeJSON(data)
	if err != nil {
		return document
	}
	return v
}

func marshalDocument(document JSONObject) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(document); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// FileStore is the AUTH-8 private file with the AUTH-4 lock every lm15 SDK
// takes on this path and atomic writes; AUTH-25 strictness.
type FileStore struct {
	path string
}

// NewFileStore anchors path (or the AUTH-8 default) now; nothing is read or created.
func NewFileStore(path string) (*FileStore, error) {
	chosen := path
	if chosen == "" {
		chosen = DefaultCredentialsPath()
	} else {
		chosen = expandHome(chosen)
	}
	abs, err := filepath.Abs(chosen)
	if err != nil {
		return nil, storageError("could not resolve the credential store path", "", "")
	}
	return &FileStore{path: abs}, nil
}

// Path is the file.
func (f *FileStore) Path() string { return f.path }

func (f *FileStore) Description() string { return f.path }

func (f *FileStore) load() (JSONObject, error) {
	data, err := os.ReadFile(f.path)
	if errors.Is(err, os.ErrNotExist) {
		return JSONObject{}, nil
	}
	if err != nil {
		return nil, storageError(fmt.Sprintf("Could not read credential store at %s", f.path), "", "")
	}
	parsed, err := decodeStrictJSON(data)
	if err != nil {
		return nil, storageError(fmt.Sprintf("Credential store at %s is not valid JSON; not touching it.", f.path), "", "")
	}
	return validateDocument(parsed, f.path)
}

func (f *FileStore) Read() (JSONObject, error) { return f.load() }

type fileGuard struct {
	store   *FileStore
	release func()
}

func (g fileGuard) Read() (JSONObject, error) { return g.store.load() }
func (g fileGuard) Write(document JSONObject) error {
	if _, err := validateDocument(normalizeDocument(document), g.store.path); err != nil {
		return err
	}
	if err := writePrivateJSON(g.store.path, document); err != nil {
		return storageError(fmt.Sprintf("Could not write credential store at %s", g.store.path), "", "")
	}
	return nil
}
func (g fileGuard) Unlock() { g.release() }

func (f *FileStore) Lock(ctx context.Context) (StoreGuard, error) {
	lock, err := holdFileLock(ctx, f.path)
	if err != nil {
		return nil, err
	}
	return fileGuard{store: f, release: lock.Release}, nil
}

func (f *FileStore) Reserve(ctx context.Context) error {
	if err := os.MkdirAll(filepath.Dir(f.path), 0o700); err != nil {
		return storageError(fmt.Sprintf("Cannot create %s for the credential store", filepath.Dir(f.path)), "", "reservation")
	}
	if _, err := os.Stat(f.path); err == nil {
		file, err := os.OpenFile(f.path, os.O_WRONLY|os.O_APPEND, 0)
		if err != nil {
			return storageError(fmt.Sprintf("Credential store at %s is not writable.", f.path), "", "reservation")
		}
		file.Close()
	}
	guard, err := f.Lock(ctx)
	if err != nil {
		return err
	}
	defer guard.Unlock()
	_, err = f.load()
	return err
}

// ─── Ordered JSON bodies (the reference's key order on the wire) ─────

type orderedBody [][2]any

func (o orderedBody) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, kv := range o {
		if i > 0 {
			buf.WriteByte(',')
		}
		key, _ := json.Marshal(kv[0])
		buf.Write(key)
		buf.WriteByte(':')
		var value bytes.Buffer
		enc := json.NewEncoder(&value)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(kv[1]); err != nil {
			return nil, err
		}
		buf.Write(bytes.TrimRight(value.Bytes(), "\n"))
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

func jsonNumberOf(s string) json.Number { return json.Number(s) }

func formatFloat(f float64) string { return strconv.FormatFloat(f, 'f', -1, 64) }
