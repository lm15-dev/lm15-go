package lm15

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// MAP-18 (lm15-contract spec/auth-failed.json, 2026-10-10): the pinned
// "this key is not valid" forms are the contract's, verbatim.
func TestAuthFailedFormsAreTheContracts(t *testing.T) {
	raw, err := os.ReadFile("../lm15-contract/spec/auth-failed.json")
	if err != nil {
		t.Skip("contract spec not present:", err)
	}
	var spec struct {
		Forms []map[string]any `json:"forms"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		t.Fatal(err)
	}
	if len(spec.Forms) != len(AuthFailedForms) {
		t.Fatalf("%d forms in the contract, %d here", len(spec.Forms), len(AuthFailedForms))
	}
	for i, f := range spec.Forms {
		field := func(k string) string { s, _ := f[k].(string); return s }
		want := AuthFailedForm{Code: field("code"), Reason: field("reason"), Prefix: field("prefix"), Contains: field("contains"), Suffix: field("suffix")}
		if AuthFailedForms[i] != want {
			t.Fatalf("form %d: %+v, contract %+v", i, AuthFailedForms[i], want)
		}
	}
}

func TestGoogleReasonOnlyFromErrorInfo(t *testing.T) {
	help, err := DecodeJSONObject([]byte(`{"details": [{"@type": "type.googleapis.com/google.rpc.Help", "reason": "API_KEY_INVALID"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if r := googleErrorReasons(help); len(r) != 0 || IsPinnedAuthFailure("INVALID_ARGUMENT", "API key not valid.", r) {
		t.Fatalf("a Help detail counted as a reason: %v", r)
	}
}

// AUTH-1/AUTH-5 (amended 2026-10-10): a key given where a name goes is
// refused without being repeated, on the doctor and on an adapter.
func TestKeyGivenAsNamedCredentialIsNeverShown(t *testing.T) {
	const secret = "SECRET-SENTINEL-DO-NOT-PRINT"
	checks := map[string]error{}
	_, checks["doctor, key door"] = ExplainAuth("anthropic", ExplainOptions{Env: map[string]string{}, Credential: secret})
	_, checks["doctor, cloud door"] = ExplainAuth("vertex", ExplainOptions{Env: map[string]string{"GOOGLE_CLOUD_PROJECT": "p"}, Credential: secret})
	_, checks["adapter"] = NewAnthropicLM(WithNamedCredential(secret))
	for name, err := range checks {
		if err == nil {
			t.Fatalf("%s: no refusal", name)
		}
		if strings.Contains(err.Error(), secret) {
			t.Fatalf("%s: the value leaked: %v", name, err)
		}
		if !strings.Contains(err.Error(), "may be a key") {
			t.Fatalf("%s: %v", name, err)
		}
	}
}
