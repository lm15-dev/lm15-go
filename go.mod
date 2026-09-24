module github.com/lm15-dev/lm15-go

go 1.26.2

// v1.0.0 was tagged on 2026-06-11 from an early prototype of the client
// layer, before the shared lm15 contract existed; it is not lm15-go 1.0.
// v1.0.1 exists only to carry this retraction. See RELEASING.md.
retract (
	v1.0.1 // Carries this retraction only.
	v1.0.0 // Early prototype tagged by mistake; not the lm15 1.0 API.
)
