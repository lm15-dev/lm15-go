package lm15

import (
	"context"
	"fmt"
	"html"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// callbackListener is a one-shot loopback listener for an authorization-code
// return (AUTH-18): port of lm15-python CallbackListener. Loopback bind only
// (127.0.0.1 or ::1), never a wildcard or LAN address; the registered
// redirect URI (which may say localhost) is a separate value, never
// rewritten. Only the exact path; the state checked on success AND error
// returns; a wrong state, both a code and an error, neither, or a repeated
// parameter gets a generic rejection and the wait goes on. Bounded request
// target (8 KiB) and headers (32 KiB); no access log; pages are no-store and
// no-referrer. A busy registered port is method_unavailable.
type callbackListener struct {
	redirectURI string
	results     chan listenerResult
	server      *http.Server
	mu          sync.Mutex
	done        bool
}

type listenerResult struct {
	value callbackReturn
	err   error
}

func openCallbackListener(path string, expectedState string, checkState bool, port int, bindHost, redirectHost string) (*callbackListener, error) {
	if bindHost == "" {
		bindHost = "127.0.0.1"
	}
	if bindHost != "127.0.0.1" && bindHost != "::1" {
		return nil, authOperation(fmt.Sprintf("callback listener may bind loopback only, not %q", bindHost), "method_unavailable", "reservation", "not_committed", "operator_action")
	}
	if !strings.HasPrefix(path, "/") {
		return nil, authOperation("callback path must start with '/'", "method_unavailable", "reservation", "not_committed", "operator_action")
	}
	socket, err := net.Listen("tcp", net.JoinHostPort(bindHost, strconv.Itoa(port)))
	if err != nil {
		where := "ephemeral"
		if port != 0 {
			where = strconv.Itoa(port)
		}
		return nil, authOperation(fmt.Sprintf("could not listen on %s:%s for the sign-in return; another program may be using the port", bindHost, where), "method_unavailable", "reservation", "not_committed", "choose_method")
	}
	host := redirectHost
	if host == "" {
		host = bindHost
	}
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") {
		host = "[" + host + "]"
	}
	l := &callbackListener{
		redirectURI: fmt.Sprintf("http://%s:%d%s", host, socket.Addr().(*net.TCPAddr).Port, path),
		results:     make(chan listenerResult, 1),
	}
	page := func(w http.ResponseWriter, status int, title, message string) {
		body := "<!doctype html><meta charset='utf-8'><meta name='referrer' content='no-referrer'><title>" + html.EscapeString(title) + "</title><p>" + html.EscapeString(message) + "</p>"
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}
	var settled bool
	var lock sync.Mutex
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || len(r.RequestURI) > callbackTargetLimit {
			page(w, 414, "Rejected", "Request too large.")
			return
		}
		if r.URL.Path != path {
			page(w, 404, "Not found", "Callback route not found.")
			return
		}
		lock.Lock()
		defer lock.Unlock()
		if settled {
			page(w, 409, "Already used", "This sign-in return was already handled.")
			return
		}
		params := parseQueryPairs(r.URL.RawQuery)
		seen := map[string]bool{}
		get := map[string]string{}
		for _, p := range params {
			if seen[p[0]] {
				page(w, 400, "Rejected", "Sign-in return was not accepted.")
				return
			}
			seen[p[0]] = true
			get[p[0]] = p[1]
		}
		if checkState && (!seen["state"] || get["state"] != expectedState) {
			page(w, 400, "Rejected", "Sign-in return was not accepted.")
			return
		}
		hasCode := get["code"] != ""
		if hasCode == seen["error"] {
			page(w, 400, "Rejected", "Sign-in return was not accepted.")
			return
		}
		settled = true
		if seen["error"] {
			page(w, 400, "Not completed", "Sign-in was not completed.")
			l.results <- listenerResult{err: denied("the provider returned an error to the sign-in callback")}
		} else {
			page(w, 200, "Signed in", "Sign-in completed. You can close this window.")
			l.results <- listenerResult{value: callbackReturn{code: get["code"], state: get["state"], has: seen["state"]}}
		}
		go l.stop()
	})
	l.server = &http.Server{Handler: handler, MaxHeaderBytes: 32 * 1024, ReadHeaderTimeout: 10 * time.Second, ErrorLog: nil}
	go func() { _ = l.server.Serve(socket) }()
	return l, nil
}

func (l *callbackListener) isDone() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.done
}

func (l *callbackListener) markDone() {
	l.mu.Lock()
	l.done = true
	l.mu.Unlock()
}

func (l *callbackListener) stop() {
	l.markDone()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = l.server.Shutdown(ctx)
}
