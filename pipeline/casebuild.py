"""Assemble a clinical "case" from a linked patient timeline, and extract the
verifiable case-facts that anchor grounding.

A case is one index encounter plus the claims that line up with it, drawn from a single
patient timeline. The important output is `CaseFacts`: a structured, de-duplicated set of
the atomic, checkable facts in the record (diagnoses, medications, labs, demographics,
and the separately-kept billing codes from claims). CaseFacts is the single source of
truth for the grounding gate; a generated example may assert a clinical fact only if that
fact resolves here. Keeping clinical facts (from the EHR) and billing facts (from claims)
in separate buckets is deliberate: claims are billing-driven and are not clinical ground
truth, and the disagreement check below depends on that separation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from pipeline.canonical import Claim, Encounter
from pipeline.linkage import PatientTimeline


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


# Generic diagnosis modifiers carry no clinical identity on their own. A cited diagnosis
# made only of these ("uncomplicated", "without complications") must not resolve as if it
# were the diagnosis itself.
_DX_MODIFIERS = {
    "without", "with", "complication", "complications", "uncomplicated", "unspecified",
    "chronic", "acute", "mild", "moderate", "severe", "type", "stage", "and", "the", "of",
    "due", "to", "other", "not", "elsewhere", "classified", "history",
}


def _phrase_match(a: str, b: str) -> bool:
    """True when the shorter phrase appears as a contiguous run of whole words in the
    longer one. Word-aware on purpose: the old bidirectional substring test matched "e"
    against "metformin" and "female" against "male". We also require the shorter phrase to
    carry at least one token of length >= 3, so single stray letters never resolve."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    short, long = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    if not any(len(t) >= 3 for t in short):
        return False
    n = len(short)
    return any(long[i:i + n] == short for i in range(len(long) - n + 1))


def _icd_category(code: str | None) -> str | None:
    """The 3-character ICD-10 category (e.g. 'J45.909' -> 'J45'). Used to judge whether
    two diagnoses are in the same clinical family rather than demanding exact equality."""
    if not code:
        return None
    base = code.strip().split(".")[0]
    return base[:3].upper() if base else None


def _is_icd(system: str | None) -> bool:
    return bool(system) and "icd" in system.strip().lower()


@dataclass
class CaseFacts:
    patient_token: str
    link_confidence: float | None
    trusted: bool
    identity_conflict: bool
    source_encounter_ids: list[str]
    source_providers: list[str]

    age: int | None
    sex: str | None
    specialty: str | None
    encounter_type: str | None
    encounter_date: date | None

    # Clinical facts, from the EHR. These are the groundable clinical truth.
    diagnoses: list[dict] = field(default_factory=list)     # {code, system, description}
    medications: list[dict] = field(default_factory=list)   # {drug, dose, route, frequency}
    observations: list[dict] = field(default_factory=list)  # {name, value, unit, flag, kind}

    # Billing facts, from claims. Kept apart on purpose; never promoted to clinical truth.
    claim_procedures: list[str] = field(default_factory=list)
    claim_diagnoses: list[str] = field(default_factory=list)

    # Raw note text. Carries PHI; used by the de-id utility, never copied into output.
    notes: list[str] = field(default_factory=list)

    def ehr_dx_categories(self) -> set[str]:
        """ICD categories of the *chart* diagnoses. Only ICD-coded diagnoses count: a
        SNOMED or free-text problem cannot be compared to a billing ICD code without a
        crosswalk, so we do not pretend to."""
        return {c for c in (_icd_category(d["code"]) for d in self.diagnoses
                            if _is_icd(d.get("system"))) if c}

    def claim_dx_categories(self) -> set[str]:
        return {c for c in (_icd_category(c2) for c2 in self.claim_diagnoses) if c}

    def has_ehr_claims_disagreement(self) -> bool:
        """True when the claim diagnoses share no ICD category with the chart. That is the
        signature of a billing code that contradicts the clinical record (e.g. COPD billed
        on an asthma visit). Extra claim codes that include a chart category are not a
        disagreement, just claims being broader.

        Honest limitation: this only fires when the chart is ICD-coded. When the chart
        uses SNOMED or free text (Provider B), we cannot do a code-level comparison
        without a SNOMED->ICD crosswalk, so we report no disagreement rather than a false
        one. The crosswalk is named as future work in DESIGN.md."""
        ehr = self.ehr_dx_categories()
        claim = self.claim_dx_categories()
        if not ehr or not claim:
            return False
        return ehr.isdisjoint(claim)

    def claim_only_dx_categories(self) -> set[str]:
        return self.claim_dx_categories() - self.ehr_dx_categories()

    def contains(self, field_kind: str, value: str) -> bool:
        """Does `value` resolve to a real fact of kind `field_kind` in this case?
        This is what the grounding gate calls to verify a cited fact. Matching is
        case-insensitive and word-aware (see `_phrase_match`): a faithful summary may say
        "metformin" while the record says "Metformin 1000mg", but a stray letter or the
        wrong sex must not resolve."""
        v = (value or "").strip().lower()
        if not v:
            return False

        if field_kind == "diagnosis":
            # A cited diagnosis must share a content word with the chart description, not
            # just a generic modifier; "uncomplicated" alone never resolves as a diagnosis.
            if all(t in _DX_MODIFIERS for t in _words(v)):
                return False
            for d in self.diagnoses:
                if d.get("code") and v == d["code"].strip().lower():
                    return True
                if d.get("description") and _phrase_match(v, d["description"]):
                    return True
            return False

        if field_kind == "medication":
            return any(m.get("drug") and _phrase_match(v, m["drug"])
                       for m in self.medications)

        if field_kind == "lab":
            # Match by lab name only. We deliberately do not resolve on a bare value, which
            # let "the dose was 187 mg" ground itself against a glucose of 187.
            return any(o.get("name") and _phrase_match(v, o["name"])
                       for o in self.observations)

        if field_kind == "demographic":
            tokens = _words(v)
            if self.age is not None and str(self.age) in tokens:
                return True
            if self.sex and self.sex.lower() in tokens:   # exact token: "male" != "female"
                return True
            return False

        if field_kind == "claim_procedure":
            return any(p.strip().lower() == v for p in self.claim_procedures)

        if field_kind == "claim_diagnosis":
            return any(c.strip().lower() == v for c in self.claim_diagnoses)

        return False

    def all_dx_codes(self) -> set[str]:
        """Every ICD code present anywhere (chart or claims), for the uncited-code scan in
        the grounding gate."""
        codes = {d["code"].strip().upper() for d in self.diagnoses if d.get("code")}
        codes |= {c.strip().upper() for c in self.claim_diagnoses if c}
        return codes


@dataclass
class ClinicalCase:
    """An index encounter plus its aligned claims, ready to summarise."""

    timeline: PatientTimeline
    index_encounter: Encounter
    aligned_claims: list[Claim]
    facts: CaseFacts


def _abnormal_first(observations: list[dict]) -> list[dict]:
    """Surface flagged-abnormal observations first; they carry the clinical signal."""
    return sorted(observations, key=lambda o: o.get("flag") not in ("H", "L", "A"))


# How close a claim's service date must be to the encounter to count as the same episode.
# A window, not exact equality, because claim service dates and chart dates routinely
# differ by a few days (and the billed/adjudicated date differs again).
_CLAIM_ALIGN_WINDOW_DAYS = 7


def build_case(timeline: PatientTimeline) -> ClinicalCase | None:
    """Build a case from a timeline's most recent encounter and the claims that fall near
    its service date. Returns None if the timeline has no usable encounter."""
    if not timeline.encounters:
        return None
    index = timeline.encounters[-1]   # timelines are date-sorted; take the latest

    # Align claims within a date window of the index encounter. If the encounter has no
    # date we do NOT fall back to fusing every claim the patient ever had (that would drag
    # unrelated billing into the case); we simply align nothing.
    if index.start_date is None:
        aligned = []
    else:
        aligned = [c for c in timeline.claims if c.service_date
                   and abs((c.service_date - index.start_date).days) <= _CLAIM_ALIGN_WINDOW_DAYS]

    diagnoses = [{"code": d.concept.code, "system": d.concept.system,
                  "description": d.concept.description} for d in index.diagnoses]
    medications = [{"drug": m.drug_name, "dose": m.dose, "route": m.route,
                    "frequency": m.frequency} for m in index.medications]
    observations = _abnormal_first([
        {"name": o.name, "value": o.value, "unit": o.unit,
         "flag": o.abnormal_flag, "kind": o.obs_kind} for o in index.observations])

    claim_procs, claim_dx = [], []
    for c in aligned:
        if c.procedure and c.procedure.code:
            claim_procs.append(c.procedure.code)
        claim_dx.extend(d.code for d in c.diagnoses if d.code)

    facts = CaseFacts(
        patient_token=timeline.token,
        link_confidence=timeline.min_link_confidence,
        trusted=timeline.is_trusted(),
        identity_conflict=timeline.has_identity_conflict(),
        source_encounter_ids=[index.source_encounter_id],
        source_providers=sorted({index.source_id} | {c.source_id for c in aligned}),
        age=index.patient_age,
        sex=index.patient_sex,
        specialty=index.specialty,
        encounter_type=index.encounter_type,
        encounter_date=index.start_date,
        diagnoses=diagnoses,
        medications=medications,
        observations=observations,
        claim_procedures=claim_procs,
        claim_diagnoses=claim_dx,
        notes=[n.text for n in index.notes],
    )
    return ClinicalCase(timeline=timeline, index_encounter=index,
                        aligned_claims=aligned, facts=facts)
