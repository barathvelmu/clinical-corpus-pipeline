"""Canonical clinical event model.

Every EHR/claims provider has a slightly different schema. Rather than teach the rest
of the pipeline 30 dialects, each provider gets a small adapter that maps its raw rows
into the shapes below. Downstream code (linkage, case assembly, example generation,
quality gates) only ever sees canonical objects, so "support a new provider" means
"write and test one adapter," not "touch the pipeline."

The model is intentionally small and clinical-event shaped: a patient has encounters,
and each encounter carries notes, observations (labs and vitals unified), medications,
and diagnoses. Claims live alongside as billing events, deliberately kept separate from
the clinical record because they are billing-driven and do not always agree with it.

Identity note: `patient_token` is the cross-source identity (the same person yields the
same token everywhere). Adapters populate `source_patient_key` (the provider's own id);
linkage fills `patient_token` from the token table. We never invent a token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True)
class CodedConcept:
    """A clinical code plus the system it came from and its human description.

    `normalized_code` / `normalized_system` are filled when we map the raw code to a
    standard vocabulary (ICD-10, SNOMED, LOINC, RxNorm). Left None when we cannot map
    it confidently; we never guess a standard code, because a wrong code is a clinical
    error, not a formatting one.
    """

    system: Optional[str]            # raw coding system as the provider reported it
    code: Optional[str]              # raw code; may be None when only free text exists
    description: Optional[str]       # free-text label
    normalized_system: Optional[str] = None
    normalized_code: Optional[str] = None

    @property
    def is_coded(self) -> bool:
        return bool(self.code)


@dataclass
class Observation:
    """A lab or a vital. Provider A keeps labs in their own table; Provider B merges
    labs and vitals into one observations table. Both collapse to here."""

    obs_kind: str                    # "lab" or "vital"
    name: str                        # raw test/observation name; not standardized
    value: Optional[str]             # often numeric, sometimes text; kept as-is
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    abnormal_flag: Optional[str] = None   # H / L / A; frequently absent
    observed_at: Optional[datetime] = None
    code: Optional[CodedConcept] = None   # LOINC where we can map it

    @property
    def numeric_value(self) -> Optional[float]:
        """Best-effort numeric parse; None when the value is genuinely non-numeric.
        Rejects nan/inf, which float() would otherwise accept and let leak into output."""
        if self.value is None:
            return None
        try:
            v = float(str(self.value).strip())
        except ValueError:
            return None
        return v if math.isfinite(v) else None


@dataclass
class Medication:
    drug_name: str                   # free text; brand/generic mix
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None  # frequently null
    code: Optional[CodedConcept] = None   # RxNorm where we can map it


@dataclass
class Diagnosis:
    concept: CodedConcept            # ICD-10 (Provider A) or SNOMED / free text (B)
    diagnosed_at: Optional[date] = None


@dataclass
class ClinicalNote:
    """Free-text note. Carries PHI in the raw record; this is the highest-risk field
    in the whole pipeline and the one the PHI gate scrutinises hardest."""

    doc_type: Optional[str]          # progress note, discharge summary, inline, etc.
    text: str
    authored_at: Optional[datetime] = None


@dataclass
class Encounter:
    """One clinical encounter / visit, normalized across providers.

    Provider A has a single encounter_date; Provider B splits admit/discharge. We keep
    both `start_date` and `end_date` (end_date None for single-date or outpatient) so we
    never lose information in normalization.
    """

    source_id: str                   # which provider this came from
    source_encounter_id: str         # provider's own encounter/visit id
    source_patient_key: str          # provider's own patient/member id
    encounter_type: Optional[str] = None      # office visit / inpatient / ER, normalized
    specialty: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None

    notes: list[ClinicalNote] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    diagnoses: list[Diagnosis] = field(default_factory=list)

    # Filled by linkage; the cross-source patient identity.
    patient_token: Optional[str] = None
    link_confidence: Optional[float] = None


@dataclass
class Claim:
    """A claims line. Billing-driven: codes reflect what was billed, which does not
    always match the clinical record. Kept separate from Encounter on purpose."""

    source_id: str
    claim_id: str
    source_patient_key: str
    service_date: Optional[date] = None
    claim_type: Optional[str] = None          # professional / institutional / pharmacy
    place_of_service: Optional[str] = None
    procedure: Optional[CodedConcept] = None  # CPT / HCPCS
    diagnoses: list[CodedConcept] = field(default_factory=list)  # the delimited field, split
    billed_amount: Optional[float] = None
    paid_amount: Optional[float] = None       # null until adjudicated; lags months
    rendering_npi: Optional[str] = None

    patient_token: Optional[str] = None
    link_confidence: Optional[float] = None


# Normalization helpers shared by adapters. Small, boring, and tested: the point of a
# canonical model is undermined if each adapter normalizes "ER" differently.

_ENCOUNTER_TYPE_MAP = {
    "office visit": "ambulatory", "office": "ambulatory", "op": "ambulatory",
    "outpatient": "ambulatory", "ambulatory": "ambulatory",
    "inpatient": "inpatient", "ip": "inpatient",
    "er": "emergency", "emer": "emergency", "emergency": "emergency",
}

_SEX_MAP = {
    "m": "male", "male": "male",
    "f": "female", "female": "female",
}


def normalize_encounter_type(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()   # str(): some providers code these as integers
    return _ENCOUNTER_TYPE_MAP.get(s, s)


def normalize_sex(raw) -> Optional[str]:
    if raw is None:
        return None
    return _SEX_MAP.get(str(raw).strip().lower())


def parse_date(raw) -> Optional[date]:
    """Lenient date parse. Accepts date objects, ISO strings, and a couple of common
    formats. Returns None rather than raising, because partial/garbage dates are a
    routine reality in this data and should not crash a 100M-row pipeline."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_datetime(raw) -> Optional[datetime]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, date):
        return datetime(raw.year, raw.month, raw.day)
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
