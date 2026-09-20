package fslock

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRealPathAllowMissing(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "real")
	os.Mkdir(real, 0o700)
	link := filepath.Join(dir, "link")
	os.Symlink(real, link)
	missing := filepath.Join(link, "sub", "credentials.json")
	got, err := RealPathAllowMissing(missing)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := filepath.EvalSymlinks(real)
	if got != filepath.Join(want, "sub", "credentials.json") {
		t.Fatalf("got %s", got)
	}
	// Existing paths agree with EvalSymlinks.
	os.MkdirAll(filepath.Join(real, "sub"), 0o700)
	os.WriteFile(filepath.Join(real, "sub", "credentials.json"), nil, 0o600)
	again, _ := RealPathAllowMissing(missing)
	if again != got {
		t.Fatalf("identity changed once the file exists: %s vs %s", again, got)
	}
	// A loop is bounded.
	loop := filepath.Join(dir, "loop")
	os.Symlink(loop, loop)
	if _, err := RealPathAllowMissing(filepath.Join(loop, "x")); err == nil {
		t.Fatal("a symlink loop must fail")
	}
}
