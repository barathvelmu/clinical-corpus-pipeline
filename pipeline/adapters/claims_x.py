"""Adapter: Claims Provider X -> canonical.

Claims are billing-driven. The one structural wrinkle worth handling carefully: a single
claim line packs MULTIPLE ICD codes into one delimited `diagnosis_codes` field. We split
them into separate CodedConcepts so downstream logic can reason over each. We do NOT
treat any of them as clinical ground truth; that judgement lives in the quality gates.
"""

from __future__ import annotations

import re

from pipeline.canonical import Claim, CodedConcept, parse_date

SOURCE_ID = "claims_provider_x"

# Providers delimit the multi-code field differently, and a single field can even mix
# delimiters. Split on any of them in one pass so "E11.9|E78.5, I10" yields three codes,
# not a mangled one. str() guards a code field that arrives as a number.
_DX_SPLIT_RE = re.compile(r"[|,;^]+")


def _split_dx(raw) -> list[CodedConcept]:
    if raw is None or raw == "":
        return []
    parts = _DX_SPLIT_RE.split(str(raw))
    return [
        CodedConcept(system="ICD-10", code=p.strip(), description=None)
        for p in parts if p.strip()
    ]


def build_claims(tables: dict[str, list[dict]]) -> list[Claim]:
    claims = []
    for row in tables.get(f"{SOURCE_ID}.claim_lines", []):
        if not row.get("claim_id") or not row.get("member_id"):
            continue   # drop a row missing its identity keys rather than crash the batch
        proc_code = row.get("procedure_code")
        procedure = None
        if proc_code:
            procedure = CodedConcept(
                system=row.get("procedure_system"), code=proc_code, description=None)
        claims.append(Claim(
            source_id=SOURCE_ID,
            claim_id=row["claim_id"],
            source_patient_key=row["member_id"],
            service_date=parse_date(row.get("service_date")),
            claim_type=row.get("claim_type"),
            place_of_service=row.get("place_of_service"),
            procedure=procedure,
            diagnoses=_split_dx(row.get("diagnosis_codes")),
            billed_amount=row.get("billed_amount"),
            paid_amount=row.get("paid_amount"),
            rendering_npi=row.get("rendering_npi"),
        ))
    return claims
