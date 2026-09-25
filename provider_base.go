package lm15

import (
	"context"
	"fmt"
	"io"
	"iter"
	"strings"
	"time"

	"github.com/lm15-dev/lm15-go/internal/sse"
)

// LM is a provider adapter: a dialect bound to an access policy. Every
// concrete adapter (OpenAILM, OpenAIChatLM, AnthropicLM, GeminiLM) and the
// router's returned values satisfy it.
type LM interface {
	Provider() string
	Access() AccessPolicy
	Supports() EndpointSupport
	BaseURL() string

	Complete(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error]
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// Plan is what a call WOULD adapt (MAP-13), with no network and no
	// credential invoked; it returns the refusal the call would return.
	Plan(req *Request) ([]Adaptation, error)
	// Adaptations is the MAP-13 policy this adapter runs under.
	Adaptations() string

	// Pure hooks (what the contract pins).
	BuildRequest(req *Request, stream bool) (*TransportRequest, error)
	// Build is BuildRequest with the MAP-13 record of the build.
	Build(req *Request, stream bool) (*TransportRequest, []Adaptation, error)
	ParseResponse(req *Request, resp *HTTPResponse) (*Response, error)
	ParseStreamEvents(req *Request, event sse.Event) ([]StreamEvent, error)
	NormalizeError(status int, body string) *Error

	// Files.
	FileUpload(ctx context.Context, req *FileUploadRequest) (FileInfo, error)
	FileGet(ctx context.Context, fileID string) (FileInfo, error)
	FileList(ctx context.Context, limit int, cursor string) (FilePage, error)
	FileDelete(ctx context.Context, fileID string) error
	FileDownload(ctx context.Context, fileID string) ([]byte, error)
	FileWaitReady(ctx context.Context, fileID string, pollEvery time.Duration) (FileInfo, error)

	// Batch.
	BatchSubmit(ctx context.Context, req *BatchRequest) (BatchJobInfo, error)
	BatchStatus(ctx context.Context, batchID string) (BatchJobInfo, error)
	BatchResults(ctx context.Context, batchID string) ([]BatchEntry, error)
	BatchCancel(ctx context.Context, batchID string) (BatchJobInfo, error)
	BatchList(ctx context.Context, limit int) ([]BatchJobInfo, error)
	Batch(ctx context.Context, req *BatchRequest) (*BatchJob, error)
	BatchJob(ctx context.Context, batchID string) (*BatchJob, error)
	Batches(ctx context.Context, limit int) ([]*BatchJob, error)

	// Cache resources.
	CacheCreate(ctx context.Context, prefix *Request, ttlSeconds *int, label string) (CacheInfo, error)
	CacheGet(ctx context.Context, cacheID string) (CacheInfo, error)
	CacheList(ctx context.Context, limit int, cursor string) (CachePage, error)
	CacheDelete(ctx context.Context, cacheID string) error
	CacheUpdate(ctx context.Context, cacheID string, ttlSeconds int) (CacheInfo, error)
	Cache(ctx context.Context, prefix *Request, ttlSeconds *int, label string) (CachedPrefix, error)

	// Generation.
	ImageGenerate(ctx context.Context, req *ImageGenerationRequest) (ImageGenerationResponse, error)
	SpeechGenerate(ctx context.Context, req *SpeechGenerationRequest) (SpeechGenerationResponse, error)

	// Video.
	VideoSubmit(ctx context.Context, req *VideoGenerationRequest) (VideoJobInfo, error)
	VideoStatus(ctx context.Context, videoID string) (VideoJobInfo, error)
	VideoResult(ctx context.Context, videoID string) (VideoPart, error)
	VideoList(ctx context.Context, limit int, model string) ([]VideoJobInfo, error)
	VideoGenerate(ctx context.Context, req *VideoGenerationRequest) (*VideoJob, error)
	VideoJob(ctx context.Context, videoID string) (*VideoJob, error)
	VideoJobs(ctx context.Context, limit int, model string) ([]*VideoJob, error)

	// Live.
	Live(ctx context.Context, config *LiveConfig) (LiveSession, error)

	// Ingest (MAP-12); non-chat dialects refuse.
	RequestFromOpenAIChat(body JSONObject) (*Request, error)
	// ResponseFromOpenAIChat reads a Chat Completions body; responseFormat
	// (optional) is the request's, so a judgment answer folds into a DataPart.
	ResponseFromOpenAIChat(body JSONObject, model string, choice *int, responseFormat JSONObject) (*Response, error)

	Close() error
}

// dialect is the hook set a concrete adapter implements; lmCore supplies
// refusing defaults and drives every surface through self.
type dialect interface {
	buildRequest(req *Request, stream bool, scope *adaptScope) (*TransportRequest, error)
	parseResponse(req *Request, resp *HTTPResponse) (*Response, error)
	parseStreamEvents(req *Request, event sse.Event) ([]StreamEvent, error)
	normalizeError(status int, body string) *Error
	completeOverride(ctx context.Context, req *Request) (*Response, bool, error)
	streamOverride(ctx context.Context, req *Request) (iter.Seq2[StreamEvent, error], bool)

	modelsRequest() (*TransportRequest, error)
	modelsFromBody(body string) ([]ModelInfo, error)

	fileUploadRequest(req *FileUploadRequest) (*TransportRequest, error)
	fileInfoFromBody(body string) (FileInfo, error)
	fileGetRequest(fileID string) (*TransportRequest, error)
	fileListRequest(limit int, cursor string) (*TransportRequest, error)
	filePageFromListBody(body string) (FilePage, error)
	fileDeleteRequest(fileID string) (*TransportRequest, error)
	fileDownloadRequest(fileID string) (*TransportRequest, error)

	batchUploadRequest(req *BatchRequest, scope *adaptScope) (*TransportRequest, error)
	batchSubmitRequest(req *BatchRequest, uploadBody JSONObject, scope *adaptScope) (*TransportRequest, error)
	batchJobFromBody(body string) (BatchJobInfo, error)
	batchStatusRequest(batchID string) (*TransportRequest, error)
	batchCancelRequest(batchID string) (*TransportRequest, error)
	batchResultFetches(statusBody JSONObject) ([]*TransportRequest, error)
	batchEntries(statusBody JSONObject, fetched []string) ([]BatchEntry, error)
	batchListRequest(limit int) (*TransportRequest, error)
	batchJobsFromListBody(body string) ([]BatchJobInfo, error)

	cacheCreateRequest(prefix *Request, ttlSeconds *int, label string) (*TransportRequest, error)
	cacheInfoFromBody(body string) (CacheInfo, error)
	cacheGetRequest(cacheID string) (*TransportRequest, error)
	cacheListRequest(limit int, cursor string) (*TransportRequest, error)
	cachePageFromListBody(body string) (CachePage, error)
	cacheDeleteRequest(cacheID string) (*TransportRequest, error)
	cacheUpdateRequest(cacheID string, ttlSeconds int) (*TransportRequest, error)

	videoSubmitRequest(req *VideoGenerationRequest) (*TransportRequest, error)
	videoJobFromBody(body string, videoID string) (VideoJobInfo, error)
	videoStatusRequest(videoID string) (*TransportRequest, error)
	videoResultFetch(statusBody JSONObject) (*TransportRequest, error)
	videoPart(statusBody JSONObject, fetched *HTTPResponse) (VideoPart, error)
	videoListRequest(limit int, model string) (*TransportRequest, error)
	videoJobsFromListBody(body string) ([]VideoJobInfo, error)

	imageGenerateRequest(req *ImageGenerationRequest) (*TransportRequest, error)
	imageGenerationFromResponse(req *ImageGenerationRequest, resp *HTTPResponse) (ImageGenerationResponse, error)
	speechGenerateRequest(req *SpeechGenerationRequest) (*TransportRequest, error)
	speechGenerationFromResponse(req *SpeechGenerationRequest, resp *HTTPResponse) (SpeechGenerationResponse, error)

	liveSetupFrames(config *LiveConfig) ([]JSONObject, error)
	liveEncoder(config *LiveConfig) func(LiveClientEvent) ([]JSONObject, error)
	liveDecode(raw []byte) ([]LiveServerEvent, error)
	live(ctx context.Context, config *LiveConfig) (LiveSession, error)

	requestFromOpenAIChat(body JSONObject) (*Request, error)
	responseFromOpenAIChat(body JSONObject, model string, choice *int, responseFormat JSONObject) (*Response, error)
}

// Option configures an adapter constructor.
type Option func(*lmOptions)

type lmOptions struct {
	credential         CredentialProvider
	credentialErr      error
	baseURL            string
	transport          Transport
	access             *AccessPolicy
	compatPreset       string
	responsesCompat    *OpenAIResponsesCompat
	chatCompat         *OpenAIChatCompat
	anthropicCompat    *AnthropicCompat
	credentialsPath    string
	settings           map[string]string
	clock              func() time.Time
	accountID          string
	apiVersion         string
	uploadBaseURL      string
	claudeCodeVersion  string
	codexOriginator    string
	codexClientVersion string
	adaptations        string
	namedCredential    string
}

// WithAdaptations sets the MAP-13 policy: "note" (default: adapt and
// record), "silent" (adapt, record nothing), "refuse" (every deviation is
// an UnsupportedFeatureError before the wire).
func WithAdaptations(policy string) Option { return func(o *lmOptions) { o.adaptations = policy } }

// WithNamedCredential names one identity on a cloud door instead of its
// chain (AUTH-1, 2026-09-19): "platform", "workload", "environment", "cli".
func WithNamedCredential(name string) Option { return func(o *lmOptions) { o.namedCredential = name } }

// WithAPIKey sets the credential: a string (APIKey shorthand), a Credential
// value, a CredentialProvider, or a func returning one.
func WithAPIKey(credential CredentialLike) Option {
	return func(o *lmOptions) {
		p, err := coerceCredentialLike(credential)
		o.credential, o.credentialErr = p, err
	}
}

// WithCredential sets a credential value.
func WithCredential(c Credential) Option { return WithAPIKey(c) }

// WithCredentialProvider sets a credential provider.
func WithCredentialProvider(p CredentialProvider) Option { return WithAPIKey(p) }

// WithBaseURL overrides the base URL.
func WithBaseURL(url string) Option { return func(o *lmOptions) { o.baseURL = url } }

// WithTransport sets the transport.
func WithTransport(t Transport) Option { return func(o *lmOptions) { o.transport = t } }

// WithAccess binds an access policy (AUTH-10).
func WithAccess(p AccessPolicy) Option { return func(o *lmOptions) { o.access = &p } }

// WithCompatPreset names the server dialect preset (also supplies its address).
func WithCompatPreset(name string) Option { return func(o *lmOptions) { o.compatPreset = name } }

// WithOpenAIResponsesCompat sets a Responses compat value.
func WithOpenAIResponsesCompat(c OpenAIResponsesCompat) Option {
	return func(o *lmOptions) { o.responsesCompat = &c }
}

// WithOpenAIChatCompat sets a Chat Completions compat value.
func WithOpenAIChatCompat(c OpenAIChatCompat) Option { return func(o *lmOptions) { o.chatCompat = &c } }

// WithAnthropicCompat sets an Anthropic compat value.
func WithAnthropicCompat(c AnthropicCompat) Option {
	return func(o *lmOptions) { o.anthropicCompat = &c }
}

// WithCredentialsPath overrides the stored-login file path.
func WithCredentialsPath(path string) Option { return func(o *lmOptions) { o.credentialsPath = path } }

// WithSettings sets host settings (region, project, resource, ...).
func WithSettings(settings map[string]string) Option {
	return func(o *lmOptions) { o.settings = settings }
}

// WithClock fixes the clock every time-dependent byte reads.
func WithClock(clock func() time.Time) Option { return func(o *lmOptions) { o.clock = clock } }

// WithAccountID sets the ChatGPT account id (Codex backend).
func WithAccountID(id string) Option { return func(o *lmOptions) { o.accountID = id } }

// WithAPIVersion sets the anthropic-version header.
func WithAPIVersion(v string) Option { return func(o *lmOptions) { o.apiVersion = v } }

// WithUploadBaseURL sets Gemini's upload root.
func WithUploadBaseURL(url string) Option { return func(o *lmOptions) { o.uploadBaseURL = url } }

// WithClaudeCodeVersion sets the claude-cli user-agent version.
func WithClaudeCodeVersion(v string) Option { return func(o *lmOptions) { o.claudeCodeVersion = v } }

// WithCodexOriginator sets the Codex originator header.
func WithCodexOriginator(v string) Option { return func(o *lmOptions) { o.codexOriginator = v } }

// WithCodexClientVersion sets the Codex client_version option.
func WithCodexClientVersion(v string) Option { return func(o *lmOptions) { o.codexClientVersion = v } }

func applyOptions(opts []Option) (*lmOptions, error) {
	o := &lmOptions{}
	for _, opt := range opts {
		opt(o)
	}
	if o.credentialErr != nil {
		return nil, o.credentialErr
	}
	if o.adaptations == "" {
		o.adaptations = AdaptationsNote
	}
	if err := checkAdaptationPolicy(o.adaptations); err != nil {
		return nil, err
	}
	return o, nil
}

// lmCore is the shared adapter state and the shared surface drivers.
type lmCore struct {
	self             dialect
	provider         string
	access           AccessPolicy
	manifest         AccessPolicy
	credential       CredentialProvider
	credentialSource string
	credentialOrigin string
	namedCredential  string
	adaptations      string
	accountID        string
	baseURL          string
	defaultBaseURL   string
	endpoint         string
	hostSettings     map[string]string
	clock            func() time.Time
	transport        Transport
	apiKeyHeader     string
}

// Provider returns the canonical provider string.
func (c *lmCore) Provider() string { return c.provider }

// wireRequest strips exactly this binding's own "provider:" prefix, once, at
// the codec boundary (a CachedPrefix's qualified request sent straight to
// the LM a router built); any other colon-bearing model id is opaque.
func (c *lmCore) wireRequest(req *Request) *Request {
	head, model, ok := strings.Cut(req.Model, ":")
	if ok && model != "" && CanonicalProvider(head) == CanonicalProvider(c.provider) {
		return req.WithModel(model)
	}
	return req
}

// Access returns the bound access policy.
func (c *lmCore) Access() AccessPolicy { return c.access }

// Supports returns the surfaces this access path carries.
func (c *lmCore) Supports() EndpointSupport { return c.access.Supports }

// BaseURL returns the base URL.
func (c *lmCore) BaseURL() string { return c.baseURL }

// AccountID returns the account id the credential carries (Codex).
func (c *lmCore) AccountID() string { return c.accountID }

// Adaptations returns the MAP-13 policy ("note", "silent", "refuse").
func (c *lmCore) Adaptations() string { return c.adaptations }

// CredentialOrigin says where this adapter's credential comes from (AUTH-1
// provenance): a label, never the value. For a cloud chain provider this
// is the rung that last won, or what will be walked when no request has
// been sent yet.
func (c *lmCore) CredentialOrigin() string {
	if cp, ok := c.credential.(*cachingProvider); ok {
		if src, ok := cp.Source(); ok {
			return src.Describe(c.now())
		}
		if cp.named != "" {
			return fmt.Sprintf("named credential %q (%s; not yet resolved)", cp.named, NamedMeaningFor(c.access, cp.named))
		}
		return "the " + c.access.EffectiveCredentialPolicy() + " (not yet resolved)"
	}
	if c.credentialOrigin != "" {
		return c.credentialOrigin
	}
	return originLabel(c.credential, c.credentialSource)
}

// SetCredentialOrigin names the credential's source when the constructor
// could not know it (the router: an env variable, a placeholder key).
func (c *lmCore) SetCredentialOrigin(label string) { c.credentialOrigin = label }

// HostSettings returns the resolved host settings.
func (c *lmCore) HostSettings() map[string]string { return c.hostSettings }

// Close releases the transport.
func (c *lmCore) Close() error {
	if closer, ok := c.transport.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

func (c *lmCore) now() time.Time {
	if c.clock != nil {
		return c.clock()
	}
	return time.Now().UTC()
}

// bindAccess binds the policy and resolves the credential it calls for.
func (c *lmCore) bindAccess(self dialect, manifest AccessPolicy, o *lmOptions, defaultBaseURL string) error {
	c.self = self
	c.manifest = manifest
	policy := manifest
	if o.access != nil {
		policy = *o.access
	}
	if err := policy.Validate(); err != nil {
		return err
	}
	c.access = policy
	c.provider = policy.Provider
	c.defaultBaseURL = defaultBaseURL
	c.baseURL = defaultBaseURL
	if o.baseURL != "" {
		c.baseURL = o.baseURL
	}
	c.transport = o.transport
	if c.transport == nil {
		c.transport = NewHTTPTransport()
	}
	c.clock = o.clock
	c.accountID = o.accountID
	c.adaptations = o.adaptations
	if c.adaptations == "" {
		c.adaptations = AdaptationsNote
	}
	if c.apiKeyHeader == "" {
		c.apiKeyHeader = "x-api-key"
	}
	c.namedCredential = o.namedCredential
	loaded, err := LoadCredentialNamed(policy, o.credential, o.credentialsPath, o.namedCredential)
	if err != nil {
		return err
	}
	c.credential = loaded.Provider
	c.credentialSource = loaded.Source
	c.credentialOrigin = loaded.Origin
	if loaded.AccountID != "" && c.accountID == "" {
		c.accountID = loaded.AccountID
	}
	// A static credential of the wrong kind for this door fails now.
	if static, ok := loaded.Provider.(StaticCredential); ok {
		if _, err := SelectScheme(policy, static.Value); err != nil {
			return err
		}
	}
	// An explicit base URL on a cloud door is the endpoint root; the door's
	// path is appended unless already present (AUTH-10, amended 2026-09-19).
	endpoint := ""
	if policy.Host != nil && o.baseURL != "" && o.baseURL != defaultBaseURL {
		endpoint = o.baseURL
	}
	settings, err := resolveSettingsWithEndpoint(policy.Host, o.settings, nil, policy.Provider, nil, endpoint)
	if err != nil {
		return err
	}
	c.hostSettings = settings
	c.endpoint = endpoint
	if policy.Host != nil {
		rendered, err := renderBaseURLAt(*policy.Host, settings, endpoint, policy.Provider)
		if err != nil {
			return err
		}
		c.baseURL = rendered
	} else if policy.BaseURL != "" && c.baseURL == defaultBaseURL {
		c.baseURL = policy.BaseURL
	}
	return nil
}

// Endpoint is the endpoint root a cloud door was given ("" = the template).
func (c *lmCore) Endpoint() string { return c.endpoint }

// registryCompat is the preset the bound provider names in the registry.
func (c *lmCore) registryCompat() string {
	if c.access.Provider == c.manifest.Provider {
		return ""
	}
	if d, ok := LookupProvider(c.access.Provider); ok {
		return d.Compat
	}
	return ""
}

func (c *lmCore) isCodex() bool { return c.access.EffectiveBackend() == CodexBackend }

// resolveCredential invokes the provider once (AUTH-2).
func (c *lmCore) resolveCredential(ctx context.Context) (Credential, error) {
	if c.credential == nil {
		return nil, nil
	}
	cred, err := c.credential.Credential(ctx)
	if err != nil {
		return nil, err
	}
	if cred == nil {
		return nil, nil
	}
	return cred, validateCredential(cred)
}

// credentialString is the bearer/key string of the credential (never AWS).
func (c *lmCore) credentialString(ctx context.Context) (string, error) {
	cred, err := c.resolveCredential(ctx)
	if err != nil {
		return "", err
	}
	switch x := cred.(type) {
	case APIKey:
		return x.Value, nil
	case BearerToken:
		return x.Value, nil
	case nil:
		return "", nil
	}
	return "", NotConfiguredErrorf(c.provider, nil, "", "AWS credentials cannot be sent as a header value; the door must accept sigv4")
}

// emitSpec is what a dialect hands to emit.
type emitSpec struct {
	method      string
	url         string
	headers     [][2]string
	params      map[string]string
	payload     any
	body        []byte
	endpoint    string
	stream      bool
	model       string
	readTimeout time.Duration
	// scope is the build's MAP-13 record; under Plan the bytes are
	// discarded, so no credential is invoked and nothing is signed.
	scope *adaptScope
}

// emit finishes a dialect-built request through the bound host (AUTH-10)
// and signs it. Pure apart from the credential provider call.
func (c *lmCore) emit(spec emitSpec) (*TransportRequest, error) {
	var cred Credential
	if !spec.scope.isPlanning() {
		var err error
		if cred, err = c.resolveCredential(context.Background()); err != nil {
			return nil, err
		}
	}
	headers := append([][2]string(nil), spec.headers...)
	if cred != nil {
		if _, isAws := cred.(AwsCredentials); !isAws {
			name, value, ok, err := AuthHeaderFor(c.access, cred, c.apiKeyHeader)
			if err != nil {
				return nil, err
			}
			if ok && !hasHeader(headers, name) {
				headers = append(headers, [2]string{name, value})
			}
		}
	}
	finished, err := finishRequest(c.access, c.hostSettings, c.baseURL, spec, headers, cred)
	if err != nil {
		return nil, err
	}
	req, err := makeJSONRequest(spec.method, finished.url, finished.headers, finished.params, finished.payload, spec.body, spec.readTimeout)
	if err != nil {
		return nil, err
	}
	if aws, ok := cred.(AwsCredentials); ok {
		signed, err := signRequest(c.access, c.hostSettings, req, aws, c.now())
		if err != nil {
			return nil, err
		}
		req.Headers = signed
	}
	return req, nil
}

func hasHeader(headers [][2]string, name string) bool {
	for _, h := range headers {
		if strings.EqualFold(h[0], name) {
			return true
		}
	}
	return false
}

var surfaceWord = map[string]string{
	"files": "files", "batches": "batch", "images": "image generation", "speech": "speech generation",
	"video": "video generation", "live": "live", "caches": "caches", "models": "model listing",
}

// require refuses unless the bound access path carries the surface.
func (c *lmCore) require(surface string) error {
	if !c.access.Supports.SupportsEndpoint(surface) {
		word := surfaceWord[surface]
		if word == "" {
			word = surface
		}
		return UnsupportedFeatureErrorf(c.provider, "%s: %s not supported", c.provider, word)
	}
	return nil
}

// ─── HTTP plumbing ───────────────────────────────────────────────────

func (c *lmCore) send(ctx context.Context, req *TransportRequest) (*HTTPResponse, error) {
	resp, err := c.transport.Do(ctx, req)
	if err != nil {
		if e := AsError(err); e != nil {
			return nil, e
		}
		return nil, newError(KindTransport, err.Error()).WithCause(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, newError(KindTransport, err.Error()).WithCause(err)
	}
	return &HTTPResponse{Status: resp.Status, Reason: resp.Reason, Headers: resp.Headers, Body: body}, nil
}

func (c *lmCore) httpError(resp *HTTPResponse) *Error {
	e := c.self.normalizeError(resp.Status, resp.Text())
	attachErrorMetadata(e, resp.Headers)
	return c.withOrigin(e)
}

// CredentialSource is the provenance of a cloud chain credential after its
// first resolution (nil for a key, a stored login, or before any request).
func (c *lmCore) CredentialSource() *CredentialSource { return ChainCredentialSource(c.credential) }

// withOrigin names the credential's source on an AuthError (AUTH-1
// provenance): the rung, the variable, "an explicit api_key", the stored
// login, or the caller's callable. Every other error passes through.
func (c *lmCore) withOrigin(e *Error) *Error {
	if e == nil || !e.Kind.IsA(KindAuth) {
		return e
	}
	return WithCredentialOrigin(e, c.CredentialOrigin())
}

// replyError is INV-054: a 2xx whose body is not JSON is a ProviderError
// (code provider) carrying the status, the content type, the first 200
// bytes and the request id — never a ServerError, never retried.
func (c *lmCore) replyError(resp *HTTPResponse, cause error) *Error {
	e := nonJSONReplyError(c.provider, resp, cause)
	attachErrorMetadata(e, resp.Headers)
	return e
}

// sendOK sends and refuses non-2xx.
func (c *lmCore) sendOK(ctx context.Context, req *TransportRequest, err error) (*HTTPResponse, error) {
	if err != nil {
		return nil, err
	}
	resp, err := c.send(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.Status >= 400 {
		return nil, c.httpError(resp)
	}
	return resp, nil
}

// NormalizeError maps an HTTP failure body to a typed error (dialects override).
func (c *lmCore) NormalizeError(status int, body string) *Error {
	return c.self.normalizeError(status, body)
}

func (c *lmCore) normalizeError(status int, body string) *Error {
	msg := strings.TrimSpace(body)
	if len(msg) > 500 {
		msg = msg[:500]
	}
	if msg == "" {
		msg = "HTTP " + itoa(status)
	}
	return c.withLoginHint(MapHTTPError(status, msg, c.provider, c.access.EnvKeys, "", "", nil))
}

// providerError builds a typed provider error with this adapter's metadata.
func (c *lmCore) providerError(kind ErrorKind, message string, status int, providerCode, requestID string) *Error {
	e := providerErrorf(kind, c.provider, c.access.EnvKeys, message)
	e.Status = status
	e.ProviderCode = providerCode
	e.RequestID = requestID
	if kind.IsA(KindAuth) {
		return c.withLoginHint(e)
	}
	return e
}

// withLoginHint rewrites AuthError guidance for stored logins.
func (c *lmCore) withLoginHint(e *Error) *Error {
	hint := c.access.LoginHint
	if hint != "" && (c.access.EffectiveCredentialPolicy() == "oauth" || c.credentialSource == "stored") {
		return WithCredentialHint(e, hint)
	}
	return e
}

// ─── Public surface ──────────────────────────────────────────────────

// BuildRequest builds the wire request (pure apart from the credential call).
func (c *lmCore) BuildRequest(req *Request, stream bool) (*TransportRequest, error) {
	wire, _, err := c.Build(req, stream)
	return wire, err
}

// Build is BuildRequest with the MAP-13 record of what the build adapted.
func (c *lmCore) Build(req *Request, stream bool) (*TransportRequest, []Adaptation, error) {
	if err := req.Validate(); err != nil {
		return nil, nil, err
	}
	return c.build(req, stream, false)
}

// build runs one request build inside an adaptation scope: the wire
// request and the record of what differs from what was asked. The one
// place a scope is opened.
func (c *lmCore) build(req *Request, stream bool, planning bool) (*TransportRequest, []Adaptation, error) {
	req = c.wireRequest(req)
	scope := newAdaptScope(c.adaptations, c.provider, planning)
	wire, err := c.self.buildRequest(req, stream, scope)
	if err != nil {
		return nil, nil, err
	}
	return wire, scope.records, nil
}

// Plan is what a call with this request WOULD adapt, with no network and
// no credential invoked (offline, like Resolve). It returns the refusal
// the call would return (under any policy, or every deviation under
// "refuse") and the FULL record under every policy, "silent" included: a
// preview that hid what it saw would be no preview.
func (c *lmCore) Plan(req *Request) ([]Adaptation, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	_, records, err := c.build(req, false, true)
	return records, err
}

// visible is what the response carries: everything under "note" (and
// "refuse", which only ever holds satisfied/defaulted), nothing under
// "silent". Behaviour is decided from the full record, never from this.
func (c *lmCore) visible(records []Adaptation) []Adaptation {
	if c.adaptations == AdaptationsSilent {
		return nil
	}
	return records
}

// finishResponse stamps the visible record on the response and applies the
// client-side steps the record asks for.
func (c *lmCore) finishResponse(req *Request, resp *Response, records []Adaptation) *Response {
	if clientSideStop(records) {
		resp = ApplyClientSideStop(resp, req.Config.Stop)
	}
	if visible := c.visible(records); len(visible) > 0 && len(resp.Adaptations) == 0 {
		resp.Adaptations = visible
	}
	return resp
}

// ParseResponse parses a complete body.
func (c *lmCore) ParseResponse(req *Request, resp *HTTPResponse) (*Response, error) {
	return c.self.parseResponse(req, resp)
}

// ParseStreamEvents parses one SSE event into canonical events.
func (c *lmCore) ParseStreamEvents(req *Request, event sse.Event) ([]StreamEvent, error) {
	return c.self.parseStreamEvents(req, event)
}

// Complete performs one call.
func (c *lmCore) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if resp, handled, err := c.self.completeOverride(ctx, req); handled {
		return resp, err
	}
	wire, records, err := c.build(req, false, false)
	if err != nil {
		return nil, err
	}
	if clientSideStop(records) {
		// MAP-13 (decision 2026-09-14): a stop sequence the wire cannot take
		// is honoured by streaming under the hood and closing the
		// connection at the cut. Whether the provider then stops
		// generating (and billing) on a closed connection is its own
		// behaviour, not a promise made here. The price is the usage
		// report, which only the final frame carries: it is "not
		// reported", never estimated. A stream that never hits the
		// sequence completes normally, usage included.
		return MaterializeResponse(c.Stream(ctx, req), req)
	}
	resp, err := c.send(ctx, wire)
	if err != nil {
		return nil, err
	}
	if resp.Status >= 400 {
		return nil, c.httpError(resp)
	}
	parsed, err := c.self.parseResponse(req, resp)
	if err != nil {
		if e := AsError(err); e != nil && e.Kind.IsA(KindProvider) {
			attachErrorMetadata(e, resp.Headers)
		}
		return nil, err
	}
	return c.finishResponse(req, parsed, records), nil
}

// Stream yields canonical events: exactly one start, deltas, exactly one
// final end (MAP-3 / MAP-4 via the coalescer).
func (c *lmCore) Stream(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error] {
	if err := req.Validate(); err != nil {
		return errSeq(err)
	}
	if it, ok := c.self.streamOverride(ctx, req); ok {
		return it
	}
	wire, records, err := c.build(req, true, false)
	if err != nil {
		return errSeq(err)
	}
	events := CoalesceStreamWith(c.streamRaw(ctx, req, wire), req.Model, c.visible(records))
	if clientSideStop(records) {
		events = TruncateStreamAtStop(events, req.Config.Stop)
	}
	return events
}

func errSeq(err error) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) { yield(nil, err) }
}

func (c *lmCore) streamRaw(ctx context.Context, req *Request, wire *TransportRequest) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		if wire == nil {
			var err error
			if wire, err = c.self.buildRequest(req, true, nil); err != nil {
				yield(nil, err)
				return
			}
		}
		resp, err := c.transport.Do(ctx, wire)
		if err != nil {
			if e := AsError(err); e != nil {
				yield(nil, e)
			} else {
				yield(nil, newError(KindTransport, err.Error()).WithCause(err))
			}
			return
		}
		defer resp.Body.Close()
		if resp.Status >= 400 {
			body, _ := io.ReadAll(resp.Body)
			e := c.self.normalizeError(resp.Status, string(body))
			attachErrorMetadata(e, resp.Headers)
			yield(nil, c.withOrigin(e))
			return
		}
		// The handshake's diagnostics ride every in-stream error event
		// (docs/error-diagnostics.md): evidence from the HTTP reply, not
		// proof of the quota at the later moment the error occurred.
		handshake := httpResponseDetailOf(resp.Headers)
		reader := sse.NewReader(resp.Body, sse.Limits{})
		for {
			ev, err := reader.Next()
			if err == io.EOF {
				return
			}
			if err != nil {
				if e := AsError(err); e != nil {
					yield(nil, e)
				} else {
					yield(nil, newError(KindTransport, err.Error()).WithCause(err))
				}
				return
			}
			events, err := c.self.parseStreamEvents(req, ev)
			if err != nil {
				if e := AsError(err); e != nil && e.Kind.IsA(KindProvider) {
					attachErrorMetadata(e, resp.Headers)
				}
				yield(nil, err)
				return
			}
			for _, e := range events {
				if e == nil {
					continue
				}
				if se, ok := e.(StreamErrorEvent); ok && se.Error.HTTPResponse.IsEmpty() {
					se.Error.HTTPResponse = handshake
					e = se
				}
				if !yield(e, nil) {
					return
				}
			}
		}
	}
}

// ListModels fetches the models this credential can use.
func (c *lmCore) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := c.require("models"); err != nil {
		return nil, err
	}
	req, err := c.self.modelsRequest()
	resp, err := c.sendOK(ctx, req, err)
	if err != nil {
		return nil, err
	}
	return c.self.modelsFromBody(resp.Text())
}

// ─── Files ───────────────────────────────────────────────────────────

func (c *lmCore) FileUpload(ctx context.Context, req *FileUploadRequest) (FileInfo, error) {
	if err := c.require("files"); err != nil {
		return FileInfo{}, err
	}
	if err := req.Validate(); err != nil {
		return FileInfo{}, err
	}
	wire, err := c.self.fileUploadRequest(req)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return FileInfo{}, err
	}
	return c.self.fileInfoFromBody(resp.Text())
}

func (c *lmCore) FileGet(ctx context.Context, fileID string) (FileInfo, error) {
	if err := c.require("files"); err != nil {
		return FileInfo{}, err
	}
	wire, err := c.self.fileGetRequest(fileID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return FileInfo{}, err
	}
	return c.self.fileInfoFromBody(resp.Text())
}

func (c *lmCore) FileList(ctx context.Context, limit int, cursor string) (FilePage, error) {
	if err := c.require("files"); err != nil {
		return FilePage{}, err
	}
	if limit <= 0 {
		limit = 20
	}
	wire, err := c.self.fileListRequest(limit, cursor)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return FilePage{}, err
	}
	return c.self.filePageFromListBody(resp.Text())
}

func (c *lmCore) FileDelete(ctx context.Context, fileID string) error {
	if err := c.require("files"); err != nil {
		return err
	}
	wire, err := c.self.fileDeleteRequest(fileID)
	_, err = c.sendOK(ctx, wire, err)
	return err
}

func (c *lmCore) FileDownload(ctx context.Context, fileID string) ([]byte, error) {
	if err := c.require("files"); err != nil {
		return nil, err
	}
	wire, err := c.self.fileDownloadRequest(fileID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}

// FileWaitReady polls until the file leaves pending (ctx bounds the wait).
func (c *lmCore) FileWaitReady(ctx context.Context, fileID string, pollEvery time.Duration) (FileInfo, error) {
	if pollEvery <= 0 {
		pollEvery = 2 * time.Second
	}
	info, err := c.FileGet(ctx, fileID)
	if err != nil {
		return FileInfo{}, err
	}
	for info.EffectiveReadiness() == "pending" {
		select {
		case <-ctx.Done():
			return info, ctx.Err()
		case <-time.After(pollEvery):
		}
		if info, err = c.FileGet(ctx, fileID); err != nil {
			return FileInfo{}, err
		}
	}
	return info, nil
}

// ─── Batch ───────────────────────────────────────────────────────────

func (c *lmCore) BatchSubmit(ctx context.Context, req *BatchRequest) (BatchJobInfo, error) {
	if err := c.require("batches"); err != nil {
		return BatchJobInfo{}, err
	}
	if err := req.Validate(); err != nil {
		return BatchJobInfo{}, err
	}
	var uploadBody JSONObject
	// MAP-13: the batch builders run under the adapter's policy so "refuse"
	// refuses here too; a batch ticket has no adaptations field
	// (provisional surface), so under "note" the record is not kept.
	scope := newAdaptScope(c.adaptations, c.provider, false)
	upload, err := c.self.batchUploadRequest(req, scope)
	if err != nil {
		return BatchJobInfo{}, err
	}
	if upload != nil {
		resp, err := c.sendOK(ctx, upload, nil)
		if err != nil {
			return BatchJobInfo{}, err
		}
		if uploadBody, err = resp.JSON(); err != nil {
			return BatchJobInfo{}, c.replyError(resp, err)
		}
	}
	wire, err := c.self.batchSubmitRequest(req, uploadBody, scope)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return BatchJobInfo{}, err
	}
	return c.self.batchJobFromBody(resp.Text())
}

func (c *lmCore) BatchStatus(ctx context.Context, batchID string) (BatchJobInfo, error) {
	if err := c.require("batches"); err != nil {
		return BatchJobInfo{}, err
	}
	wire, err := c.self.batchStatusRequest(batchID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return BatchJobInfo{}, err
	}
	return c.self.batchJobFromBody(resp.Text())
}

func (c *lmCore) BatchResults(ctx context.Context, batchID string) ([]BatchEntry, error) {
	if err := c.require("batches"); err != nil {
		return nil, err
	}
	wire, err := c.self.batchStatusRequest(batchID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return nil, err
	}
	job, err := c.self.batchJobFromBody(resp.Text())
	if err != nil {
		return nil, err
	}
	if !job.Done() {
		return nil, valueErrorf("batch %s is not finished (status=%q); wait() or poll batch_status() until done", batchID, job.Status)
	}
	statusBody, err := resp.JSON()
	if err != nil {
		return nil, c.replyError(resp, err)
	}
	fetches, err := c.self.batchResultFetches(statusBody)
	if err != nil {
		return nil, err
	}
	var texts []string
	for _, f := range fetches {
		fetched, err := c.sendOK(ctx, f, nil)
		if err != nil {
			return nil, err
		}
		texts = append(texts, fetched.Text())
	}
	return c.self.batchEntries(statusBody, texts)
}

func (c *lmCore) BatchCancel(ctx context.Context, batchID string) (BatchJobInfo, error) {
	if err := c.require("batches"); err != nil {
		return BatchJobInfo{}, err
	}
	wire, err := c.self.batchCancelRequest(batchID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return BatchJobInfo{}, err
	}
	return c.self.batchJobFromBody(resp.Text())
}

func (c *lmCore) BatchList(ctx context.Context, limit int) ([]BatchJobInfo, error) {
	if err := c.require("batches"); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 20
	}
	wire, err := c.self.batchListRequest(limit)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return nil, err
	}
	return c.self.batchJobsFromListBody(resp.Text())
}

// Batch submits and wraps the ticket in a handle.
func (c *lmCore) Batch(ctx context.Context, req *BatchRequest) (*BatchJob, error) {
	info, err := c.BatchSubmit(ctx, req)
	if err != nil {
		return nil, err
	}
	return &BatchJob{lm: c.self.(LM), info: info}, nil
}

// BatchJob re-attaches to a job by id.
func (c *lmCore) BatchJob(ctx context.Context, batchID string) (*BatchJob, error) {
	info, err := c.BatchStatus(ctx, batchID)
	if err != nil {
		return nil, err
	}
	return &BatchJob{lm: c.self.(LM), info: info}, nil
}

// Batches lists this credential's jobs as handles.
func (c *lmCore) Batches(ctx context.Context, limit int) ([]*BatchJob, error) {
	infos, err := c.BatchList(ctx, limit)
	if err != nil {
		return nil, err
	}
	out := make([]*BatchJob, 0, len(infos))
	for _, info := range infos {
		out = append(out, &BatchJob{lm: c.self.(LM), info: info})
	}
	return out, nil
}

// ─── Cache resources ─────────────────────────────────────────────────

func checkCachePrefix(prefix *Request, ttlSeconds *int) error {
	if err := prefix.Validate(); err != nil {
		return err
	}
	if !prefix.Config.IsDefault() {
		return valueErrorf("cache_create: the prefix Request must carry a default Config (a stored cache has no generation settings)")
	}
	if ttlSeconds != nil && *ttlSeconds <= 0 {
		return valueErrorf("ttl_seconds must be a positive int")
	}
	return nil
}

func (c *lmCore) CacheCreate(ctx context.Context, prefix *Request, ttlSeconds *int, label string) (CacheInfo, error) {
	if err := c.require("caches"); err != nil {
		return CacheInfo{}, err
	}
	if err := checkCachePrefix(prefix, ttlSeconds); err != nil {
		return CacheInfo{}, err
	}
	wire, err := c.self.cacheCreateRequest(prefix, ttlSeconds, label)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return CacheInfo{}, err
	}
	return c.self.cacheInfoFromBody(resp.Text())
}

func (c *lmCore) CacheGet(ctx context.Context, cacheID string) (CacheInfo, error) {
	if err := c.require("caches"); err != nil {
		return CacheInfo{}, err
	}
	wire, err := c.self.cacheGetRequest(cacheID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return CacheInfo{}, err
	}
	return c.self.cacheInfoFromBody(resp.Text())
}

func (c *lmCore) CacheList(ctx context.Context, limit int, cursor string) (CachePage, error) {
	if err := c.require("caches"); err != nil {
		return CachePage{}, err
	}
	if limit <= 0 {
		limit = 20
	}
	wire, err := c.self.cacheListRequest(limit, cursor)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return CachePage{}, err
	}
	return c.self.cachePageFromListBody(resp.Text())
}

func (c *lmCore) CacheDelete(ctx context.Context, cacheID string) error {
	if err := c.require("caches"); err != nil {
		return err
	}
	wire, err := c.self.cacheDeleteRequest(cacheID)
	_, err = c.sendOK(ctx, wire, err)
	return err
}

func (c *lmCore) CacheUpdate(ctx context.Context, cacheID string, ttlSeconds int) (CacheInfo, error) {
	if err := c.require("caches"); err != nil {
		return CacheInfo{}, err
	}
	if ttlSeconds <= 0 {
		return CacheInfo{}, valueErrorf("ttl_seconds must be a positive int")
	}
	wire, err := c.self.cacheUpdateRequest(cacheID, ttlSeconds)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return CacheInfo{}, err
	}
	return c.self.cacheInfoFromBody(resp.Text())
}

// Cache makes a prompt beginning reusable with the best tier the provider has.
func (c *lmCore) Cache(ctx context.Context, prefix *Request, ttlSeconds *int, label string) (CachedPrefix, error) {
	if c.access.Supports.Caches {
		info, err := c.CacheCreate(ctx, prefix, ttlSeconds, label)
		if err != nil {
			return CachedPrefix{}, err
		}
		return CachedPrefix{Prefix: prefix, Resource: &info}, nil
	}
	if err := checkCachePrefix(prefix, ttlSeconds); err != nil {
		return CachedPrefix{}, err
	}
	return CachedPrefix{Prefix: prefix}, nil
}

// ─── Generation ──────────────────────────────────────────────────────

func (c *lmCore) ImageGenerate(ctx context.Context, req *ImageGenerationRequest) (ImageGenerationResponse, error) {
	if err := c.require("images"); err != nil {
		return ImageGenerationResponse{}, err
	}
	if err := req.Validate(); err != nil {
		return ImageGenerationResponse{}, err
	}
	wire, err := c.self.imageGenerateRequest(req)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return ImageGenerationResponse{}, err
	}
	return c.self.imageGenerationFromResponse(req, resp)
}

func (c *lmCore) SpeechGenerate(ctx context.Context, req *SpeechGenerationRequest) (SpeechGenerationResponse, error) {
	if err := c.require("speech"); err != nil {
		return SpeechGenerationResponse{}, err
	}
	if err := req.Validate(); err != nil {
		return SpeechGenerationResponse{}, err
	}
	wire, err := c.self.speechGenerateRequest(req)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return SpeechGenerationResponse{}, err
	}
	return c.self.speechGenerationFromResponse(req, resp)
}

// ─── Video ───────────────────────────────────────────────────────────

func (c *lmCore) VideoSubmit(ctx context.Context, req *VideoGenerationRequest) (VideoJobInfo, error) {
	if err := c.require("video"); err != nil {
		return VideoJobInfo{}, err
	}
	if err := req.Validate(); err != nil {
		return VideoJobInfo{}, err
	}
	wire, err := c.self.videoSubmitRequest(req)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return VideoJobInfo{}, err
	}
	return c.self.videoJobFromBody(resp.Text(), "")
}

func (c *lmCore) VideoStatus(ctx context.Context, videoID string) (VideoJobInfo, error) {
	if err := c.require("video"); err != nil {
		return VideoJobInfo{}, err
	}
	wire, err := c.self.videoStatusRequest(videoID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return VideoJobInfo{}, err
	}
	return c.self.videoJobFromBody(resp.Text(), videoID)
}

func (c *lmCore) VideoResult(ctx context.Context, videoID string) (VideoPart, error) {
	if err := c.require("video"); err != nil {
		return VideoPart{}, err
	}
	wire, err := c.self.videoStatusRequest(videoID)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return VideoPart{}, err
	}
	job, err := c.self.videoJobFromBody(resp.Text(), videoID)
	if err != nil {
		return VideoPart{}, err
	}
	if !job.Done() {
		return VideoPart{}, valueErrorf("video %s is not finished (status=%q); wait() or poll video_status() until done", videoID, job.Status)
	}
	statusBody, err := resp.JSON()
	if err != nil {
		return VideoPart{}, err
	}
	fetch, err := c.self.videoResultFetch(statusBody)
	if err != nil {
		return VideoPart{}, err
	}
	var fetched *HTTPResponse
	if fetch != nil {
		if fetched, err = c.sendOK(ctx, fetch, nil); err != nil {
			return VideoPart{}, err
		}
	}
	return c.self.videoPart(statusBody, fetched)
}

func (c *lmCore) VideoList(ctx context.Context, limit int, model string) ([]VideoJobInfo, error) {
	if err := c.require("video"); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 20
	}
	wire, err := c.self.videoListRequest(limit, model)
	resp, err := c.sendOK(ctx, wire, err)
	if err != nil {
		return nil, err
	}
	return c.self.videoJobsFromListBody(resp.Text())
}

func (c *lmCore) VideoGenerate(ctx context.Context, req *VideoGenerationRequest) (*VideoJob, error) {
	info, err := c.VideoSubmit(ctx, req)
	if err != nil {
		return nil, err
	}
	return &VideoJob{lm: c.self.(LM), info: info}, nil
}

func (c *lmCore) VideoJob(ctx context.Context, videoID string) (*VideoJob, error) {
	info, err := c.VideoStatus(ctx, videoID)
	if err != nil {
		return nil, err
	}
	return &VideoJob{lm: c.self.(LM), info: info}, nil
}

func (c *lmCore) VideoJobs(ctx context.Context, limit int, model string) ([]*VideoJob, error) {
	infos, err := c.VideoList(ctx, limit, model)
	if err != nil {
		return nil, err
	}
	out := make([]*VideoJob, 0, len(infos))
	for _, info := range infos {
		out = append(out, &VideoJob{lm: c.self.(LM), info: info})
	}
	return out, nil
}

// Live opens a live session (dialects with the surface override live()).
func (c *lmCore) Live(ctx context.Context, config *LiveConfig) (LiveSession, error) {
	if err := c.require("live"); err != nil {
		return nil, err
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return c.self.live(ctx, config)
}

// RequestFromOpenAIChat is MAP-12 (chat dialect only).
func (c *lmCore) RequestFromOpenAIChat(body JSONObject) (*Request, error) {
	return c.self.requestFromOpenAIChat(body)
}

// ResponseFromOpenAIChat is MAP-12 rule 9 (chat dialect only).
func (c *lmCore) ResponseFromOpenAIChat(body JSONObject, model string, choice *int, responseFormat JSONObject) (*Response, error) {
	return c.self.responseFromOpenAIChat(body, model, choice, responseFormat)
}

// LiveSetupFrames / LiveEncoder / LiveDecode expose the pure live codec.
func (c *lmCore) LiveSetupFrames(config *LiveConfig) ([]JSONObject, error) {
	return c.self.liveSetupFrames(config)
}
func (c *lmCore) LiveEncoder(config *LiveConfig) func(LiveClientEvent) ([]JSONObject, error) {
	return c.self.liveEncoder(config)
}
func (c *lmCore) LiveDecode(raw []byte) ([]LiveServerEvent, error) { return c.self.liveDecode(raw) }

// ─── Default (refusing) hooks ────────────────────────────────────────

func (c *lmCore) unsupported(what string) *Error {
	return UnsupportedFeatureErrorf(c.provider, "%s: %s not supported", c.provider, what)
}

func (c *lmCore) completeOverride(context.Context, *Request) (*Response, bool, error) {
	return nil, false, nil
}
func (c *lmCore) streamOverride(context.Context, *Request) (iter.Seq2[StreamEvent, error], bool) {
	return nil, false
}
func (c *lmCore) modelsRequest() (*TransportRequest, error) {
	return nil, c.unsupported("model listing")
}
func (c *lmCore) modelsFromBody(string) ([]ModelInfo, error) {
	return nil, c.unsupported("model listing")
}
func (c *lmCore) fileUploadRequest(*FileUploadRequest) (*TransportRequest, error) {
	return nil, c.unsupported("files")
}
func (c *lmCore) fileInfoFromBody(string) (FileInfo, error) {
	return FileInfo{}, c.unsupported("files")
}
func (c *lmCore) fileGetRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("files")
}
func (c *lmCore) fileListRequest(int, string) (*TransportRequest, error) {
	return nil, c.unsupported("files")
}
func (c *lmCore) filePageFromListBody(string) (FilePage, error) {
	return FilePage{}, c.unsupported("files")
}
func (c *lmCore) fileDeleteRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("files")
}
func (c *lmCore) fileDownloadRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("files")
}
func (c *lmCore) batchUploadRequest(*BatchRequest, *adaptScope) (*TransportRequest, error) {
	return nil, nil
}
func (c *lmCore) batchSubmitRequest(*BatchRequest, JSONObject, *adaptScope) (*TransportRequest, error) {
	return nil, c.unsupported("batch")
}
func (c *lmCore) batchJobFromBody(string) (BatchJobInfo, error) {
	return BatchJobInfo{}, c.unsupported("batch")
}
func (c *lmCore) batchStatusRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("batch")
}
func (c *lmCore) batchCancelRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("batch")
}
func (c *lmCore) batchResultFetches(JSONObject) ([]*TransportRequest, error) {
	return nil, c.unsupported("batch")
}
func (c *lmCore) batchEntries(JSONObject, []string) ([]BatchEntry, error) {
	return nil, c.unsupported("batch")
}
func (c *lmCore) batchListRequest(int) (*TransportRequest, error) { return nil, c.unsupported("batch") }
func (c *lmCore) batchJobsFromListBody(string) ([]BatchJobInfo, error) {
	return nil, c.unsupported("batch")
}
func (c *lmCore) cacheCreateRequest(*Request, *int, string) (*TransportRequest, error) {
	return nil, c.unsupported("stored caches")
}
func (c *lmCore) cacheInfoFromBody(string) (CacheInfo, error) {
	return CacheInfo{}, c.unsupported("stored caches")
}
func (c *lmCore) cacheGetRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("stored caches")
}
func (c *lmCore) cacheListRequest(int, string) (*TransportRequest, error) {
	return nil, c.unsupported("stored caches")
}
func (c *lmCore) cachePageFromListBody(string) (CachePage, error) {
	return CachePage{}, c.unsupported("stored caches")
}
func (c *lmCore) cacheDeleteRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("stored caches")
}
func (c *lmCore) cacheUpdateRequest(string, int) (*TransportRequest, error) {
	return nil, c.unsupported("stored caches")
}
func (c *lmCore) videoSubmitRequest(*VideoGenerationRequest) (*TransportRequest, error) {
	return nil, c.unsupported("video generation")
}
func (c *lmCore) videoJobFromBody(string, string) (VideoJobInfo, error) {
	return VideoJobInfo{}, c.unsupported("video generation")
}
func (c *lmCore) videoStatusRequest(string) (*TransportRequest, error) {
	return nil, c.unsupported("video generation")
}
func (c *lmCore) videoResultFetch(JSONObject) (*TransportRequest, error) {
	return nil, c.unsupported("video generation")
}
func (c *lmCore) videoPart(JSONObject, *HTTPResponse) (VideoPart, error) {
	return VideoPart{}, c.unsupported("video generation")
}
func (c *lmCore) videoListRequest(int, string) (*TransportRequest, error) {
	return nil, c.unsupported("video generation")
}
func (c *lmCore) videoJobsFromListBody(string) ([]VideoJobInfo, error) {
	return nil, c.unsupported("video generation")
}
func (c *lmCore) imageGenerateRequest(*ImageGenerationRequest) (*TransportRequest, error) {
	return nil, c.unsupported("image generation")
}
func (c *lmCore) imageGenerationFromResponse(*ImageGenerationRequest, *HTTPResponse) (ImageGenerationResponse, error) {
	return ImageGenerationResponse{}, c.unsupported("image generation")
}
func (c *lmCore) speechGenerateRequest(*SpeechGenerationRequest) (*TransportRequest, error) {
	return nil, c.unsupported("speech generation")
}
func (c *lmCore) speechGenerationFromResponse(*SpeechGenerationRequest, *HTTPResponse) (SpeechGenerationResponse, error) {
	return SpeechGenerationResponse{}, c.unsupported("speech generation")
}
func (c *lmCore) liveSetupFrames(*LiveConfig) ([]JSONObject, error) {
	return nil, c.unsupported("live")
}
func (c *lmCore) liveEncoder(*LiveConfig) func(LiveClientEvent) ([]JSONObject, error) {
	return func(LiveClientEvent) ([]JSONObject, error) { return nil, c.unsupported("live") }
}
func (c *lmCore) liveDecode([]byte) ([]LiveServerEvent, error) { return nil, c.unsupported("live") }
func (c *lmCore) live(context.Context, *LiveConfig) (LiveSession, error) {
	return nil, c.unsupported("live")
}
func (c *lmCore) requestFromOpenAIChat(JSONObject) (*Request, error) {
	return nil, valueErrorf("provider %q does not speak the Chat Completions wire; nothing to ingest", c.provider)
}
func (c *lmCore) responseFromOpenAIChat(JSONObject, string, *int, JSONObject) (*Response, error) {
	return nil, valueErrorf("provider %q does not speak the Chat Completions wire; nothing to read", c.provider)
}

// batchEntryRequest is the synthetic Request for parsing a batch entry body.
func batchEntryRequest(model string) *Request {
	if model == "" {
		model = "batch"
	}
	return &Request{Model: model, Messages: []Message{UserMessage("-")}}
}

func itoa(i int) string { return strings.TrimSpace(strings.Repeat(" ", 0) + intToStr(i)) }

func intToStr(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
