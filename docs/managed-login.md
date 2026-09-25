# Sign in once, use everywhere (managed login)

`Auth`, `Connect` and `BoundClient` implement lm15-contract's managed
authentication (`spec/auth-managed.md`, AUTH-12–26): the same component as
lm15-python's `lm15.login`, lm15-ts's `Auth` and lm15-rs's `lm15::login`,
on the same store file, graded by the same contract runs and by
mixed-language runs on one store (a login saved from Python is renewed from Go
and signed out from Rust; two processes in any two languages renew one token
exactly once).

```go
lm, err := lm15.Connect(ctx, lm15.ConnectOptions{})           // pickers in the terminal
resp, err := lm.Ask(ctx, "Explain drought stress.")

auth, _ := lm15.LocalAuth("")                                   // reads nothing yet
auth.Login(ctx, "xai", lm15.LoginOptions{Method: "device", UI: lm15.InteractiveTerminalUI(false)})
auth.SetAPIKey(ctx, "groq", "gsk-…", "")
auth.Configure(ctx, "gemini", "env", map[string]string{"name": "GEMINI_API_KEY"}, nil, "")
auth.Status("xai")                                              // no secrets
auth.Logout(ctx, "xai")                                         // remembered across restarts

router, _ := lm15.NewRouterWithConfig(lm15.RouterConfig{Auth: auth})
```

With `RouterConfig.Auth`: an explicit `APIKeys` entry, then an explicit named
cloud identity, then the saved connection (renewed if due), then — keyless
local servers only — the placeholder key. Never an environment variable,
another tool's login file or the machine's cloud identity; a missing, expired,
rejected or signed-out connection is an `AuthOperationError` (`err.Reason`).
A managed router also routes `kimi-code` and `github-copilot`.

A cancelled login returns an error that wraps `context.Canceled`
(`ErrLoginCancelled`): cancellation is Go's own, never an lm15 error. Renewal,
sign-in returns (a loopback listener raced against a paste; a wrong paste is
rejected and asked for again) and per-method evidence are as in lm15-python's
`docs/managed-login.md`. In a browser (`GOOS=js`) there is no loopback
listener: the loopback-only OpenRouter method is unavailable there.
