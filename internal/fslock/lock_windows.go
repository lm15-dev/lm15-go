//go:build windows

package fslock

import (
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

const (
	lockfileExclusiveLock   = 0x2
	lockfileFailImmediately = 0x1
	errorLockViolation      = 33
)

func tryLock(lockPath string) (func(), bool, error) {
	f, err := os.OpenFile(lockPath, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, false, err
	}
	var overlapped syscall.Overlapped
	r, _, e := procLockFileEx.Call(f.Fd(), uintptr(lockfileExclusiveLock|lockfileFailImmediately), 0, 1, 0, uintptr(unsafe.Pointer(&overlapped)))
	if r == 0 {
		f.Close()
		if errno, ok := e.(syscall.Errno); ok && errno == errorLockViolation {
			return nil, false, nil
		}
		return nil, false, e
	}
	return func() {
		var ov syscall.Overlapped
		procUnlockFileEx.Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(&ov)))
		f.Close()
	}, true, nil
}
