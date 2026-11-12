package lm15

// Google external_account with an AWS source (lm15-contract spec/auth.md AUTH-1
// gcp-chain, amended 2026-11-12; changes/2026-11-12-gcp-aws-source.md). The subject
// token is pinned byte for byte by auth/token-vectors.json (computed by google-auth);
// these walk the acquisition: where the AWS identity comes from, then STS, then
// impersonation.

import (
	"context"
	"encoding/json"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"
)

type awsWifCall struct {
	method, url string
	headers     map[string]string
	body        []byte
}

var awsWifSource = map[string]any{
	"environment_id":                 "aws1",
	"region_url":                     "http://169.254.169.254/latest/meta-data/placement/availability-zone",
	"url":                            "http://169.254.169.254/latest/meta-data/iam/security-credentials",
	"regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
}

const awsWifAudience = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/aws"
const awsWifImpersonate = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken"

var awsWifGoogle = [][2]any{
	{"https://sts.googleapis.com/", `{"access_token":"federated","expires_in":3600,"token_type":"Bearer"}`},
	{"https://iamcredentials.googleapis.com/", `{"accessToken":"sa-token","expireTime":"2026-09-20T01:00:00Z"}`},
}

func awsWifFile(t *testing.T, source map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"type": "external_account", "audience": awsWifAudience, "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
		"token_url": "https://sts.googleapis.com/v1/token", "credential_source": source, "service_account_impersonation_url": awsWifImpersonate,
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func awsWif(t *testing.T, env map[string]string, answers [][2]any, source map[string]any) (*[]awsWifCall, CredentialProvider) {
	t.Helper()
	calls := &[]awsWifCall{}
	full := map[string]string{"GOOGLE_APPLICATION_CREDENTIALS": "/h/wif.json"}
	for k, v := range env {
		full[k] = v
	}
	ctx := &ChainContext{
		Env: full, Home: "/h", Files: map[string]string{"/h/wif.json": awsWifFile(t, source)},
		Now: func() time.Time { return time.Date(2026, 9, 20, 0, 0, 0, 0, time.UTC) },
		HTTP: func(method, u string, headers map[string]string, body []byte, _ time.Duration) (int, map[string]string, []byte, error) {
			copied := map[string]string{}
			for k, v := range headers {
				copied[k] = v
			}
			*calls = append(*calls, awsWifCall{method, u, copied, body})
			for _, a := range answers {
				if strings.HasPrefix(u, a[0].(string)) {
					return 200, map[string]string{}, []byte(a[1].(string)), nil
				}
			}
			t.Fatalf("unexpected request %s %s", method, u)
			return 0, nil, nil, nil
		},
	}
	p, err := NamedCredentialProviderFor(Vertex, ctx, "workload")
	if err != nil {
		t.Fatal(err)
	}
	return calls, p
}

func awsWifSigned(t *testing.T, calls []awsWifCall) (string, map[string]string) {
	t.Helper()
	for _, c := range calls {
		if !strings.HasPrefix(c.url, "https://sts.googleapis.com/") {
			continue
		}
		var body map[string]string
		if err := json.Unmarshal(c.body, &body); err != nil {
			t.Fatal(err)
		}
		if body["subjectTokenType"] != "urn:ietf:params:aws:token-type:aws4_request" {
			t.Fatal(body["subjectTokenType"])
		}
		decoded, err := url.QueryUnescape(body["subjectToken"])
		if err != nil {
			t.Fatal(err)
		}
		var signed struct {
			URL     string `json:"url"`
			Headers []struct{ Key, Value string }
		}
		if err := json.Unmarshal([]byte(decoded), &signed); err != nil {
			t.Fatal(err)
		}
		headers := map[string]string{}
		for _, h := range signed.Headers {
			headers[h.Key] = h.Value
		}
		return signed.URL, headers
	}
	t.Fatal("no STS exchange")
	return "", nil
}

func TestGCPAwsSourceOnECS(t *testing.T) {
	calls, p := awsWif(t, map[string]string{"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/abc", "AWS_REGION": "us-east-2"}, append([][2]any{
		{"http://169.254.170.2/v2/credentials/abc", `{"AccessKeyId":"ASIATASK","SecretAccessKey":"s","Token":"task-session","Expiration":"2026-09-20T06:00:00Z"}`},
	}, awsWifGoogle...), awsWifSource)
	cred, err := p.Credential(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok, ok := cred.(BearerToken); !ok || tok.Value != "sa-token" {
		t.Fatalf("%#v", cred)
	}
	var urls []string
	for _, c := range *calls {
		urls = append(urls, c.url)
	}
	if strings.Join(urls, " ") != "http://169.254.170.2/v2/credentials/abc https://sts.googleapis.com/v1/token "+awsWifImpersonate {
		t.Fatal(urls)
	}
	target, h := awsWifSigned(t, *calls)
	if target != "https://sts.us-east-2.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15" {
		t.Fatal(target)
	}
	if !regexp.MustCompile(`^AWS4-HMAC-SHA256 Credential=ASIATASK/20260920/us-east-2/sts/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, `).MatchString(h["Authorization"]) {
		t.Fatal(h["Authorization"])
	}
	if h["x-amz-security-token"] != "task-session" || h["x-goog-cloud-target-resource"] != awsWifAudience {
		t.Fatal(h)
	}
	if (*calls)[2].headers["authorization"] != "Bearer federated" {
		t.Fatal((*calls)[2].headers)
	}
}

func TestGCPAwsSourceOnEC2(t *testing.T) {
	source := map[string]any{}
	for k, v := range awsWifSource {
		source[k] = v
	}
	source["imdsv2_session_token_url"] = "http://169.254.169.254/latest/api/token"
	calls, p := awsWif(t, map[string]string{}, append([][2]any{
		{"http://169.254.169.254/latest/api/token", "imds-session"},
		{"http://169.254.169.254/latest/meta-data/placement/availability-zone", "eu-west-1b"},
		{"http://169.254.169.254/latest/meta-data/iam/security-credentials/web", `{"Code":"Success","AccessKeyId":"ASIAEC2","SecretAccessKey":"s","Token":"t"}`},
		{"http://169.254.169.254/latest/meta-data/iam/security-credentials", "web\n"},
	}, awsWifGoogle...), source)
	if _, err := p.Credential(context.Background()); err != nil {
		t.Fatal(err)
	}
	puts := 0
	for _, c := range *calls {
		if c.method == "PUT" {
			puts++
		}
	}
	if puts != 1 || (*calls)[0].headers["X-aws-ec2-metadata-token-ttl-seconds"] != "300" {
		t.Fatal(*calls)
	}
	for _, c := range (*calls)[1:4] {
		if c.headers["X-aws-ec2-metadata-token"] != "imds-session" {
			t.Fatal(c)
		}
	}
	if target, _ := awsWifSigned(t, *calls); !strings.HasPrefix(target, "https://sts.eu-west-1.amazonaws.com?") {
		t.Fatal(target)
	}
}

func TestGCPAwsSourceEnvironmentKeysWin(t *testing.T) {
	calls, p := awsWif(t, map[string]string{"AWS_ACCESS_KEY_ID": "AKIDENV", "AWS_SECRET_ACCESS_KEY": "s", "AWS_DEFAULT_REGION": "ca-central-1",
		"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/abc"}, awsWifGoogle, awsWifSource)
	if _, err := p.Credential(context.Background()); err != nil {
		t.Fatal(err)
	}
	_, h := awsWifSigned(t, *calls)
	if _, has := h["x-amz-security-token"]; has || !strings.Contains(h["Authorization"], "AKIDENV/20260920/ca-central-1/sts") {
		t.Fatal(h)
	}
}

func TestGCPAwsSourceRefusals(t *testing.T) {
	for _, c := range []struct {
		source map[string]any
		why    string
	}{
		{map[string]any{"environment_id": "aws2", "regional_cred_verification_url": "https://sts.{region}.amazonaws.com"}, "only aws1"},
		{map[string]any{"environment_id": "aws1"}, "regional_cred_verification_url"},
		{map[string]any{"environment_id": "azure1", "regional_cred_verification_url": "https://x"}, "environment_id"},
	} {
		_, p := awsWif(t, map[string]string{"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"}, nil, c.source)
		if _, err := p.Credential(context.Background()); err == nil || !strings.Contains(err.Error(), c.why) {
			t.Fatalf("%v: %v", c.source, err)
		}
	}
	noURL := map[string]any{}
	for k, v := range awsWifSource {
		if k != "url" {
			noURL[k] = v
		}
	}
	_, p := awsWif(t, map[string]string{"AWS_REGION": "us-east-1"}, nil, noURL)
	if _, err := p.Credential(context.Background()); err == nil || !strings.Contains(err.Error(), "AWS_ACCESS_KEY_ID nor a container endpoint") {
		t.Fatal(err)
	}
}
