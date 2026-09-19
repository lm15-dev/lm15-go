// Package sigv4 implements AWS Signature Version 4 (standard library only).
//
// Rules from the AWS reference, pinned by the AWS test suite in
// lm15-contract/auth/sigv4-vectors.json: canonical request, string to sign,
// HMAC signing-key chain, trimmed header values, no x-amz-content-sha256.
package sigv4

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"sort"
	"strings"
	"time"
)

const algorithm = "AWS4-HMAC-SHA256"

// Credentials are the signing keys.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

// Signature is the signing output: every header to send, lowercase names.
type Signature struct {
	CanonicalRequest string
	StringToSign     string
	Authorization    string
	Headers          map[string]string
}

func encode(text string) string {
	var b strings.Builder
	for i := 0; i < len(text); i++ {
		c := text[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		default:
			b.WriteString("%" + strings.ToUpper(hex.EncodeToString([]byte{c})))
		}
	}
	return b.String()
}

// removeDotSegments is RFC 3986 §5.2.4 as the AWS SDKs apply it to non-S3 paths.
func removeDotSegments(path string) string {
	var kept []string
	for _, seg := range strings.Split(path, "/") {
		switch {
		case seg == "..":
			if len(kept) > 0 {
				kept = kept[:len(kept)-1]
			}
		case seg != "" && seg != ".":
			kept = append(kept, seg)
		}
	}
	first := ""
	if strings.HasPrefix(path, "/") {
		first = "/"
	}
	last := ""
	if strings.HasSuffix(path, "/") && len(kept) > 0 {
		last = "/"
	}
	return first + strings.Join(kept, "/") + last
}

func canonicalPath(path string) string {
	if path == "" {
		return "/"
	}
	normalized := removeDotSegments(path)
	if normalized == "" {
		normalized = "/"
	}
	segs := strings.Split(normalized, "/")
	for i, s := range segs {
		unescaped, err := url.PathUnescape(s)
		if err != nil {
			unescaped = s
		}
		segs[i] = encode(unescaped)
	}
	return strings.Join(segs, "/")
}

func canonicalQuery(query string) string {
	if query == "" {
		return ""
	}
	type pair struct{ k, v string }
	var pairs []pair
	for _, item := range strings.Split(query, "&") {
		if item == "" {
			continue
		}
		k, v, _ := strings.Cut(item, "=")
		k = queryUnescape(k)
		v = queryUnescape(v)
		pairs = append(pairs, pair{encode(k), encode(v)})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].k != pairs[j].k {
			return pairs[i].k < pairs[j].k
		}
		return pairs[i].v < pairs[j].v
	})
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		parts = append(parts, p.k+"="+p.v)
	}
	return strings.Join(parts, "&")
}

func queryUnescape(s string) string {
	out, err := url.QueryUnescape(s)
	if err != nil {
		return s
	}
	return out
}

func trim(value string) string { return strings.Join(strings.Fields(value), " ") }

// Canonicalize returns (canonical request, signed headers) for complete headers.
func Canonicalize(method, rawURL string, headers map[string]string, payload []byte) (string, string) {
	u, _ := url.Parse(rawURL)
	path, query := "", ""
	if u != nil {
		path = u.EscapedPath()
		if path == "" && u.Path != "" {
			path = u.Path
		}
		query = u.RawQuery
	}
	lowered := map[string]string{}
	for k, v := range headers {
		lowered[strings.ToLower(k)] = trim(v)
	}
	names := make([]string, 0, len(lowered))
	for k := range lowered {
		names = append(names, k)
	}
	sort.Strings(names)
	var ch strings.Builder
	for _, k := range names {
		ch.WriteString(k + ":" + lowered[k] + "\n")
	}
	signed := strings.Join(names, ";")
	sum := sha256.Sum256(payload)
	canonical := strings.Join([]string{
		strings.ToUpper(method), canonicalPath(path), canonicalQuery(query), ch.String(), signed, hex.EncodeToString(sum[:]),
	}, "\n")
	return canonical, signed
}

func hmacSHA256(key, data []byte) []byte {
	m := hmac.New(sha256.New, key)
	m.Write(data)
	return m.Sum(nil)
}

// Sign signs a request. headers are the caller's (any case); the result
// carries every header to send, lowercase, with host, x-amz-date,
// x-amz-security-token (when a session token exists) and authorization.
func Sign(method, rawURL string, headers map[string]string, payload []byte, creds Credentials, region, service string, now time.Time) Signature {
	now = now.UTC()
	amzDate := now.Format("20060102T150405Z")
	date := now.Format("20060102")
	u, _ := url.Parse(rawURL)
	host := ""
	if u != nil {
		host = u.Host
	}
	toSign := map[string]string{}
	for k, v := range headers {
		lk := strings.ToLower(k)
		if lk == "authorization" {
			continue
		}
		toSign[lk] = v
	}
	toSign["host"] = host
	toSign["x-amz-date"] = amzDate
	delete(toSign, "x-amz-security-token")
	if creds.SessionToken != "" {
		toSign["x-amz-security-token"] = creds.SessionToken
	}
	canonical, signed := Canonicalize(method, rawURL, toSign, payload)
	scope := date + "/" + region + "/" + service + "/aws4_request"
	canonicalSum := sha256.Sum256([]byte(canonical))
	stringToSign := strings.Join([]string{algorithm, amzDate, scope, hex.EncodeToString(canonicalSum[:])}, "\n")
	key := []byte("AWS4" + creds.SecretAccessKey)
	for _, piece := range []string{date, region, service, "aws4_request"} {
		key = hmacSHA256(key, []byte(piece))
	}
	signature := hex.EncodeToString(hmacSHA256(key, []byte(stringToSign)))
	authorization := algorithm + " Credential=" + creds.AccessKeyID + "/" + scope + ", SignedHeaders=" + signed + ", Signature=" + signature
	out := map[string]string{}
	for k, v := range toSign {
		out[k] = trim(v)
	}
	out["authorization"] = authorization
	return Signature{CanonicalRequest: canonical, StringToSign: stringToSign, Authorization: authorization, Headers: out}
}
