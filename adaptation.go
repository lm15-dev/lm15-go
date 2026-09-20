package lm15

// MAP-13 — adapt freely, never invisibly; refuse only when a guess could hurt.
//
// lm15 makes two promises, ranked: first, change the model or provider
// string and the program keeps working; second, never change what the
// caller asked for. The second is kept by VISIBILITY, not refusal. When a
// wire cannot take a setting as asked, the adapter does the obvious thing
// and records it here; the record rides Response.Adaptations and
// StreamStartEvent.Adaptations, and lm.Plan(request) previews it with no
// network.
//
// Translations — the adapter's ordinary job (stop → stop_sequences, effort
// → budget by the MAP-7 table) — are never recorded. A record exists only
// where the wire got something other than what was asked.
//
// The policy (AdaptationPolicy) is set on the LM (WithAdaptations) and on
// RouterConfig.Adaptations:
//
//   - "note" (default): adapt and record.
//   - "silent": adapt exactly as "note" does; the response carries no
//     record. The policy never changes what goes to the wire.
//   - "refuse": every DEVIATION — dropped, clamped, substituted,
//     client_side — is an UnsupportedFeatureError before the wire,
//     carrying Feature = the config path. satisfied and defaulted change
//     nothing the caller asked for and are recorded, not refused.
//
// Nothing here prints. The record is data on the response.
//
// How the record reaches the response: every request build runs inside one
// *adaptScope that the builder threads to the places that adapt; the
// scope collects records under every policy (the adapter's own behaviour —
// a client-side stop, a narrowed tool list — is read from them), and the
// response shows them or not. A nil scope adapts under "note" and keeps
// nothing (a pure hook called outside a build).

import (
	"fmt"
	"strings"
)

// Adaptation is one thing the wire got that differs from what was asked.
//
//   - Field: the config path ("config.seed", "config.temperature",
//     "config.reasoning.summary", "config.tool_choice.allowed").
//   - Action: dropped (a hint with no home), clamped (a dial to its
//     nearest level), substituted (the closest spelling), client_side
//     (lm15 does it after the wire), satisfied (the provider's default
//     already is what was asked), defaulted (the wire requires a value the
//     caller did not set).
//   - Asked / Applied: what the caller set / what went to the wire (JSON
//     values; nil = absent). Asked is absent for defaulted; Applied is
//     absent for dropped and satisfied.
//   - Reason: one sentence naming the provider fact; the adapter's own
//     wording, never pinned by a case.
type Adaptation struct {
	Field   string
	Action  string
	Reason  string
	Asked   any
	Applied any
}

// Validate checks the field constraints.
func (a Adaptation) Validate() error {
	if a.Field == "" {
		return valueErrorf("Adaptation.field must be a non-empty string")
	}
	if !inVocab(a.Action, AdaptationActions) {
		return valueErrorf("unsupported adaptation action: %s", a.Action)
	}
	if a.Reason == "" {
		return valueErrorf("Adaptation.reason must be a non-empty string")
	}
	if a.Asked != nil {
		if err := ValidateJSONValue(a.Asked); err != nil {
			return typeErrorf("Adaptation.asked: %v", err)
		}
	}
	if a.Applied != nil {
		if err := ValidateJSONValue(a.Applied); err != nil {
			return typeErrorf("Adaptation.applied: %v", err)
		}
	}
	return nil
}

func validateAdaptations(list []Adaptation) error {
	for _, a := range list {
		if err := a.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// AdaptationToDict serializes one record (asked/applied omitted when nil).
func AdaptationToDict(a Adaptation) JSONObject {
	d := JSONObject{"field": a.Field, "action": a.Action, "reason": a.Reason}
	if a.Asked != nil {
		d["asked"] = deref(a.Asked)
	}
	if a.Applied != nil {
		d["applied"] = deref(a.Applied)
	}
	return d
}

// AdaptationFromDict reads one record.
func AdaptationFromDict(d JSONObject) (Adaptation, error) {
	field, err := reqString(d, "field")
	if err != nil {
		return Adaptation{}, err
	}
	action, err := reqString(d, "action")
	if err != nil {
		return Adaptation{}, err
	}
	reason, err := reqString(d, "reason")
	if err != nil {
		return Adaptation{}, err
	}
	a := Adaptation{Field: field, Action: action, Reason: reason, Asked: d["asked"], Applied: d["applied"]}
	return a, a.Validate()
}

func adaptationsToJSON(list []Adaptation) []any {
	if len(list) == 0 {
		return nil
	}
	return toAnyList(list, func(a Adaptation) any { return AdaptationToDict(a) })
}

func adaptationsFromJSON(v any) ([]Adaptation, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, typeErrorf("adaptations must be a list of Adaptation")
	}
	out := make([]Adaptation, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, typeErrorf("adaptations must be a list of Adaptation")
		}
		a, err := AdaptationFromDict(obj)
		if err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, nil
}

// MarshalJSON implements json.Marshaler.
func (a Adaptation) MarshalJSON() ([]byte, error) {
	return marshalVia(func() JSONObject { return AdaptationToDict(a) })
}

// UnmarshalJSON implements json.Unmarshaler.
func (a *Adaptation) UnmarshalJSON(b []byte) error {
	return unmarshalVia(b, func(d JSONObject) error {
		v, err := AdaptationFromDict(d)
		if err != nil {
			return err
		}
		*a = v
		return nil
	})
}

// IsDeviation reports whether the action changes what the caller asked for
// (the actions "refuse" refuses).
func (a Adaptation) IsDeviation() bool {
	switch a.Action {
	case AdaptDropped, AdaptClamped, AdaptSubstituted, AdaptClientSide:
		return true
	}
	return false
}

// checkAdaptationPolicy validates a policy word.
func checkAdaptationPolicy(value string) error {
	if !inVocab(value, AdaptationPolicies) {
		return valueErrorf("adaptations must be one of %s, got %q", strings.Join(AdaptationPolicies, ", "), value)
	}
	return nil
}

// adaptScope is one request build's record. Builders receive it and call
// adapt at the point where they would have refused before MAP-13, then do
// the adapted thing. Nil is a valid scope: "note" policy, nothing kept.
type adaptScope struct {
	policy   string
	provider string
	records  []Adaptation
	// planning: the bytes are discarded, so signing and credential
	// providers are skipped (Plan invokes no credential and needs no key).
	planning bool
}

func newAdaptScope(policy, provider string, planning bool) *adaptScope {
	if policy == "" {
		policy = AdaptationsNote
	}
	return &adaptScope{policy: policy, provider: provider, planning: planning}
}

var adaptationVerb = map[string]string{
	AdaptDropped:     "would be dropped",
	AdaptClamped:     "would be clamped",
	AdaptSubstituted: "would be substituted",
	AdaptClientSide:  "would be applied client-side",
	AdaptSatisfied:   "is already satisfied here",
	AdaptDefaulted:   "would be defaulted",
}

// adapt records one adaptation, or returns the refusal under "refuse".
// The message under "refuse" is the sentence the note carries, so the two
// policies never say different things about the same fact.
func (s *adaptScope) adapt(field, action, reason string, asked, applied any) error {
	policy, who := AdaptationsNote, ""
	if s != nil {
		policy, who = s.policy, s.provider
	}
	a := Adaptation{Field: field, Action: action, Reason: reason, Asked: asked, Applied: applied}
	if policy == AdaptationsRefuse && a.IsDeviation() {
		head := ""
		if who != "" {
			head = who + ": "
		}
		return UnsupportedFeature(who, field, "%s%s %s: %s (adaptations='refuse')", head, field, adaptationVerb[action], reason)
	}
	if s == nil {
		return nil
	}
	// Every policy records into the scope: behaviour (a client-side stop,
	// a narrowed tool list) is read from these records, so "silent" must
	// not empty them — it hides them on the response instead.
	s.records = append(s.records, a)
	return nil
}

// dropped / clamped / substituted / clientSide / satisfied / defaulted are
// the six spellings of adapt, one per action.
func (s *adaptScope) dropped(field, reason string, asked any) error {
	return s.adapt(field, AdaptDropped, reason, asked, nil)
}
func (s *adaptScope) clamped(field, reason string, asked, applied any) error {
	return s.adapt(field, AdaptClamped, reason, asked, applied)
}
func (s *adaptScope) substituted(field, reason string, asked, applied any) error {
	return s.adapt(field, AdaptSubstituted, reason, asked, applied)
}
func (s *adaptScope) clientSide(field, reason string, asked, applied any) error {
	return s.adapt(field, AdaptClientSide, reason, asked, applied)
}
func (s *adaptScope) satisfied(field, reason string, asked any) error {
	return s.adapt(field, AdaptSatisfied, reason, asked, nil)
}
func (s *adaptScope) defaulted(field, reason string, applied any) error {
	return s.adapt(field, AdaptDefaulted, reason, nil, applied)
}

// isPlanning reports whether the build's bytes are discarded.
func (s *adaptScope) isPlanning() bool { return s != nil && s.planning }

// clientSideStop reports whether the record asks lm15 to cut at a stop
// sequence after the wire (MAP-13 client_side on config.stop).
func clientSideStop(list []Adaptation) bool {
	for _, a := range list {
		if a.Field == "config.stop" && a.Action == AdaptClientSide {
			return true
		}
	}
	return false
}

// ─── Shared clamps ───────────────────────────────────────────────────

// effortLadder is the ordinal effort dial (off excluded).
var effortLadder = []string{EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}

func effortIndex(level string) int {
	for i, l := range effortLadder {
		if l == level {
			return i
		}
	}
	return -1
}

// nearestEffort is the closest level to asked among available on the
// ordinal ladder. A tie goes to the lower level: the cheaper guess is the
// one a caller who set a dial would rather see recorded.
func nearestEffort(asked string, available []string) (string, error) {
	var levels []string
	for _, l := range available {
		if effortIndex(l) >= 0 {
			levels = append(levels, l)
		}
	}
	if len(levels) == 0 {
		return "", valueErrorf("no comparable effort levels in %v", available)
	}
	for _, l := range levels {
		if l == asked {
			return asked, nil
		}
	}
	want := effortIndex(asked)
	if want < 0 {
		want = 0
	}
	best, bestKey := "", [2]int{1 << 30, 1 << 30}
	for _, l := range levels {
		i := effortIndex(l)
		dist := i - want
		if dist < 0 {
			dist = -dist
		}
		key := [2]int{dist, i}
		if key[0] < bestKey[0] || (key[0] == bestKey[0] && key[1] < bestKey[1]) {
			best, bestKey = l, key
		}
	}
	return best, nil
}

// adaptationsSummary renders the record compactly for Response.String.
func adaptationsSummary(list []Adaptation) string {
	parts := make([]string, 0, len(list))
	for _, a := range list {
		parts = append(parts, fmt.Sprintf("%s:%s", a.Field, a.Action))
	}
	return strings.Join(parts, ", ")
}
