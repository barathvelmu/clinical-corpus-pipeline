"""Quality-validation gates.

This is the heart of the artifact. A gate takes one example plus the case facts it was
built from and returns a verdict. The gates are identical regardless of how the example
was generated (templated or LLM), which is the whole design: we do not trust the
generator, we verify its output against the source record.

What each gate defends against:
  - format:             a structurally invalid example.
  - link_confidence:    a fused example built on a cross-source link we do not trust.
  - identity:           a token collision fusing two different patients into one example.
  - phi:                an identifier resurfacing in generated text (a privacy incident).
  - grounding:          a clinical claim that does not trace back to the record (a
                        hallucination), cited or uncited. The anti-poisoning gate.
  - answerability:      a claim that is not supported by the context actually presented.
  - clinical_plausibility: EHR/claims disagreement, routed to mandatory human review
                        (HOLD) rather than adjudicated automatically.
  - completeness:       a flagged-abnormal finding omitted from the summary (HOLD).
  - dedup:              near-duplicate examples inflating the set without adding signal.

Honesty about scope: `clinical_plausibility` is a *plausibility and consistency* check,
not a correctness oracle. Real clinical validity needs a clinician or an LLM-judge; this
gate enforces the cheap, automatable invariants and routes anything genuinely judgemental
to `needs_clinician_review`. We are not claiming to have solved medicine. Blocking gates
gate the data; the review flag gates the human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline import deid
from pipeline.casebuild import CaseFacts, _icd_category, _phrase_match
from pipeline.linkage import DEFAULT_CONFIDENCE_THRESHOLD
from pipeline.posttrain.schema import TrainingExample, validate_format
from pipeline.quality.lexicon import clinical_terms_in, term_kind

_ICD_RE = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b")

# Negation cues that precede the entity ("no X", "ruled out X", "denies X") and ones that
# follow it ("X was ruled out", "X negative"). A recognised entity in a negation context is
# a differential being excluded, not a fact being asserted, so grounding does not require it
# to resolve. We keep the following-window cues specific (not a bare "no") so "semaglutide,
# no issues" does not get mistaken for a ruled-out mention. We do not judge whether a
# negated *plan* is clinically safe; that is the clinician/judge seam.
_PRE_NEGATION = (
    "no ", "not ", "without ", "ruled out", "rule out", "denies", "denied",
    "negative for", "never ", "declined", "free of", "absence of", "no evidence of",
)
_POST_NEGATION = ("ruled out", "ruled-out", "negative", "excluded", "not present", "not seen")


def _all_occurrences_negated(term: str, text: str) -> bool:
    """True only if every occurrence of `term` sits in a negation context. If the entity is
    asserted even once, it must be grounded."""
    low = text.lower()
    positions = [m.start() for m in re.finditer(r"\b" + re.escape(term) + r"\b", low)]
    if not positions:
        return False
    for pos in positions:
        before = low[max(0, pos - 30):pos]
        after = low[pos + len(term):pos + len(term) + 16]
        pre = any(cue in before for cue in _PRE_NEGATION)
        post = any(cue in after for cue in _POST_NEGATION)
        if not (pre or post):
            return False
    return True


@dataclass
class GateResult:
    name: str
    passed: bool
    blocking: bool          # a failed blocking gate rejects the example
    message: str
    needs_review: bool = False


def link_confidence_gate(ex: TrainingExample, facts: CaseFacts,
                         threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> GateResult:
    conf = facts.link_confidence
    if conf is None:
        return GateResult("link_confidence", False, True,
                          "no parseable link confidence; treating as untrusted")
    if conf < threshold:
        return GateResult("link_confidence", False, True,
                          f"link confidence {conf:.2f} < {threshold:.2f}; refusing to ship "
                          f"an example built on an untrusted cross-source link")
    return GateResult("link_confidence", True, True, f"link confidence {conf:.2f}")


def grounding_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    """Three checks, because a citation-only check is gameable by simply not citing a
    hallucination. (1) Every cited fact must resolve into the source record. (2) Every
    ICD-shaped code in the response must belong to a diagnosis category in the record.
    (3) Every recognised clinical entity (medication / lab) in the response must resolve,
    whether or not it was cited; this catches an uncited free-text fabrication like
    "started on insulin glargine"."""
    for cf in ex.grounding.cited_facts:
        if not facts.contains(cf.field, cf.value):
            return GateResult("grounding", False, True,
                              f"unsupported claim: cited {cf.field}={cf.value!r} does not "
                              f"resolve to the source record (hallucination)")

    allowed_categories = {_icd_category(c) for c in facts.all_dx_codes()}
    for m in _ICD_RE.finditer(ex.response):
        cat = _icd_category(m.group(0))
        if cat and cat not in allowed_categories:
            return GateResult("grounding", False, True,
                              f"unsupported claim: code {m.group(0)} in the response is not "
                              f"present in the record")

    for term in sorted(clinical_terms_in(ex.response)):
        if facts.contains(term_kind(term), term):
            continue
        if _all_occurrences_negated(term, ex.response):
            continue   # a ruled-out / negated entity need not resolve
        return GateResult("grounding", False, True,
                          f"unsupported claim: {term_kind(term)} {term!r} stated in the "
                          f"response does not resolve to the source record "
                          f"(uncited hallucination)")
    return GateResult("grounding", True, True,
                      f"all cited facts resolve; no uncited clinical entities ungrounded")


def identity_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    """Reject when the linked timeline shows a token-collision footprint (two distinct
    patient keys from one source under one token, or contradictory sex). The link
    confidence can read a perfect 1.0 and still fuse two people if the token table assigned
    one token to two patients; confidence describes the match the table claims, not whether
    the table is right. A wrong-patient fusion is a privacy incident, so we drop it."""
    if facts.identity_conflict:
        return GateResult("identity", False, True,
                          "token-collision footprint: records from one source under this "
                          "token map to different patients; refusing to ship a possibly "
                          "wrong-patient fusion")
    return GateResult("identity", True, True, "no identity conflict")


def completeness_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    """A grounded summary can still be dangerously incomplete: every stated fact is true,
    but a life-threatening abnormal value is silently omitted, leaving a falsely reassuring
    picture. Every gate before this is a positive-assertion check (is what is said true?);
    this is the only recall check (is what matters present?). When a flagged-abnormal
    observation is not addressed, we do not auto-reject (a summary need not list every
    value) but we route to clinician review rather than auto-ship."""
    abnormal = [o for o in facts.observations if o.get("flag") in ("H", "L", "A")]
    missing = [o["name"] for o in abnormal
               if o.get("name") and not _phrase_match(o["name"], ex.response)]
    if missing:
        return GateResult("completeness", True, False,
                          f"flagged-abnormal finding(s) not addressed in the summary "
                          f"({', '.join(missing)}); routed to clinician review",
                          needs_review=True)
    return GateResult("completeness", True, False, "abnormal findings addressed")


def phi_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    """Re-scan the generated text. The user turn is redacted at generation, but we check
    both turns: a leak anywhere in the example is a leak."""
    for role, text in (("assistant", ex.response), ("user", ex.user_prompt)):
        findings = deid.find_phi(text)
        if findings:
            labels = ", ".join(sorted({f.label for f in findings}))
            return GateResult("phi", False, True,
                              f"PHI detected in {role} turn ({labels}); generated output "
                              f"must not contain identifiers")
    return GateResult("phi", True, True, "no PHI detected")


def answerability_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    """Every asserted fact must be present in the context the example actually shows the
    model (the user turn). Catches a response that reaches beyond what it was given, even
    if that fact happens to exist elsewhere in the record."""
    context = ex.user_prompt.lower()
    for cf in ex.grounding.cited_facts:
        if cf.value.strip().lower() not in context:
            return GateResult("answerability", False, True,
                              f"cited {cf.field}={cf.value!r} is not present in the case "
                              f"context shown to the model")
    return GateResult("answerability", True, True, "all cited facts appear in context")


def _response_asserts_claim_only_code(response: str, facts: CaseFacts) -> bool:
    claim_only = facts.claim_only_dx_categories()
    return any(_icd_category(m.group(0)) in claim_only for m in _ICD_RE.finditer(response))


def clinical_plausibility_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    """Plausibility + EHR/claims consistency. Deliberately NOT a correctness oracle, and
    deliberately NOT an automatic adjudicator of clinical disagreement.

    When the chart and the claim disagree, we do not try to decide (with a brittle phrasing
    heuristic) whether the response handled it 'honestly enough'. A chart/claims
    disagreement is an inherently clinical judgement, so we route the example to mandatory
    human review (HOLD) rather than auto-accepting or auto-rejecting it. This both closes
    the failure mode where a misleading-but-fluent example slips through, and reflects the
    right product decision: we detect and surface disagreement, we do not pretend to
    resolve it automatically.

    The same applies when the chart is non-ICD (SNOMED / free text) so we cannot do a
    code-level comparison but the response leans on a claim diagnosis: that is exactly the
    case where the consistency gate would otherwise be silently off, so we route to review
    instead."""
    if facts.has_ehr_claims_disagreement():
        return GateResult("clinical_plausibility", True, False,
                          "EHR/claims diagnosis disagreement detected; routed to mandatory "
                          "clinician review rather than adjudicated automatically",
                          needs_review=True)

    chart_non_icd = bool(facts.diagnoses) and not facts.ehr_dx_categories()
    if (chart_non_icd and facts.claim_dx_categories()
            and _response_asserts_claim_only_code(ex.response, facts)):
        return GateResult("clinical_plausibility", True, False,
                          "chart coded in a non-ICD system; claim diagnosis asserted and "
                          "cannot be auto-verified, routed to clinician review",
                          needs_review=True)

    return GateResult("clinical_plausibility", True, False, "no consistency issues detected")


class Deduplicator:
    """Near-duplicate detection across a batch. Exact match on a normalised response, plus
    a token-Jaccard check so trivially-reworded duplicates are still caught."""

    def __init__(self, jaccard_threshold: float = 0.9):
        self._seen_shingles: list[set[str]] = []
        self._threshold = jaccard_threshold

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def gate(self, ex: TrainingExample, facts: CaseFacts) -> GateResult:
        toks = self._tokens(ex.response)
        for prev in self._seen_shingles:
            union = toks | prev
            if union and len(toks & prev) / len(union) >= self._threshold:
                return GateResult("dedup", False, True,
                                  "near-duplicate of an example already in the set")
        self._seen_shingles.append(toks)
        return GateResult("dedup", True, True, "unique")


def format_gate(ex: TrainingExample, facts: CaseFacts) -> GateResult:
    problems = validate_format(ex)
    if problems:
        return GateResult("format", False, True, "; ".join(problems))
    return GateResult("format", True, True, "well-formed")


def run_gates(ex: TrainingExample, facts: CaseFacts,
              deduper: Deduplicator | None = None) -> list[GateResult]:
    """Run every gate and return all verdicts. Order is cheap-to-expensive and
    safety-first, but all gates run so the report shows the full picture."""
    deduper = deduper or Deduplicator()
    return [
        format_gate(ex, facts),
        link_confidence_gate(ex, facts),
        identity_gate(ex, facts),
        phi_gate(ex, facts),
        grounding_gate(ex, facts),
        answerability_gate(ex, facts),
        clinical_plausibility_gate(ex, facts),
        completeness_gate(ex, facts),
        deduper.gate(ex, facts),
    ]
