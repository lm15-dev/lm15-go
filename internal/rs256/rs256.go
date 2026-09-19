// Package rs256 signs JWTs with RSASSA-PKCS1-v1_5 / SHA-256 using the
// platform RSA (crypto/rsa), and pins lm15's compact JWS serialization:
// compact JSON with keys in the caller's order, base64url without padding.
package rs256

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"strings"
)

// ErrEncrypted is returned for encrypted PEM keys.
var ErrEncrypted = errors.New("encrypted private keys are not supported; decrypt it first: openssl pkey -in key.pem -out key-plain.pem")

// ErrNoKey is returned when no PEM private key is found.
var ErrNoKey = errors.New("no PEM private key found; PKCS#12 (.pfx/.p12) is not parsed — convert with: openssl pkcs12 -in cert.pfx -nodes -out cert.pem")

// ErrEC is returned for EC keys.
var ErrEC = errors.New("EC private keys are not supported (RS256 needs an RSA key)")

// B64URL is unpadded base64url.
func B64URL(data []byte) string { return base64.RawURLEncoding.EncodeToString(data) }

func pemBlock(text, label string) ([]byte, error) {
	rest := []byte(text)
	for {
		block, remaining := pem.Decode(rest)
		if block == nil {
			return nil, nil
		}
		if block.Type == label {
			return block.Bytes, nil
		}
		rest = remaining
	}
}

// LoadPrivateKey parses an unencrypted RSA private key (PKCS#8 or PKCS#1).
func LoadPrivateKey(pemText string) (*rsa.PrivateKey, error) {
	if strings.Contains(pemText, "ENCRYPTED PRIVATE KEY") || strings.Contains(pemText, "Proc-Type: 4,ENCRYPTED") {
		return nil, ErrEncrypted
	}
	if der, _ := pemBlock(pemText, "PRIVATE KEY"); der != nil {
		key, err := x509.ParsePKCS8PrivateKey(der)
		if err != nil {
			return nil, err
		}
		rsaKey, ok := key.(*rsa.PrivateKey)
		if !ok {
			return nil, errors.New("PKCS#8: not an RSA key (only rsaEncryption is supported)")
		}
		return rsaKey, nil
	}
	if der, _ := pemBlock(pemText, "RSA PRIVATE KEY"); der != nil {
		return x509.ParsePKCS1PrivateKey(der)
	}
	if strings.Contains(pemText, "BEGIN EC PRIVATE KEY") {
		return nil, ErrEC
	}
	return nil, ErrNoKey
}

// CertificateDER returns the DER bytes of the first CERTIFICATE block.
func CertificateDER(pemText string) ([]byte, error) {
	der, err := pemBlock(pemText, "CERTIFICATE")
	if err != nil {
		return nil, err
	}
	if der == nil {
		return nil, errors.New("no PEM CERTIFICATE block found")
	}
	return der, nil
}

// SignPKCS1v15SHA256 signs a message.
func SignPKCS1v15SHA256(key *rsa.PrivateKey, message []byte) ([]byte, error) {
	sum := sha256.Sum256(message)
	return rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, sum[:])
}

// OrderedObject is a JSON object with a fixed key order.
type OrderedObject struct {
	Keys   []string
	Values map[string]any
}

// Set appends or replaces a key.
func (o *OrderedObject) Set(key string, value any) {
	if o.Values == nil {
		o.Values = map[string]any{}
	}
	if _, exists := o.Values[key]; !exists {
		o.Keys = append(o.Keys, key)
	}
	o.Values[key] = value
}

// MarshalJSON writes the keys in order, compactly.
func (o OrderedObject) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, k := range o.Keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		kb, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		buf.Write(kb)
		buf.WriteByte(':')
		var vb bytes.Buffer
		enc := json.NewEncoder(&vb)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(o.Values[k]); err != nil {
			return nil, err
		}
		buf.Write(bytes.TrimRight(vb.Bytes(), "\n"))
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// JWTEncode builds the compact JWS.
func JWTEncode(header, payload OrderedObject, key *rsa.PrivateKey) (string, error) {
	head, err := header.MarshalJSON()
	if err != nil {
		return "", err
	}
	body, err := payload.MarshalJSON()
	if err != nil {
		return "", err
	}
	signingInput := B64URL(head) + "." + B64URL(body)
	sig, err := SignPKCS1v15SHA256(key, []byte(signingInput))
	if err != nil {
		return "", err
	}
	return signingInput + "." + B64URL(sig), nil
}
