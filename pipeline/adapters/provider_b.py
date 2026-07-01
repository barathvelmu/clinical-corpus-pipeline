"""Adapter: EHR Provider B -> canonical.

Provider B is visit-centric with different names and structure than A:
  - "visit" not "encounter", "member_id" not "patient_id".
  - split admit_date / discharge_date instead of a single date.
  - notes live in a separate `documents` table, not inline.
  - labs AND vitals are merged into one `observations` table (split by obs_type).
  - problems coded in SNOMED, sometimes only free text (concept_code null).
  - no medications table at all.

The whole point of the canonical model: after this adapter, downstream code cannot tell
a Provider B record from a Provider A record. Same shape, same field names, same units of
meaning. Adding a 31st provider is one more file like this one, plus its tests.
"""

from __future__ import annotations

from collections import defaultdict

from pipeline.canonical import (
    ClinicalNote, CodedConcept, Diagnosis, Encounter, Observation,
    normalize_encounter_type, normalize_sex, parse_date, parse_datetime,
)

SOURCE_ID = "ehr_provider_b"


def build_encounters(tables: dict[str, list[dict]]) -> list[Encounter]:
    # Skip child rows missing their parent key rather than crash the batch (see provider_a).
    docs_by_visit = defaultdict(list)
    for row in tables.get(f"{SOURCE_ID}.documents", []):
        if row.get("visit_key"):
            docs_by_visit[row["visit_key"]].append(row)

    obs_by_visit = defaultdict(list)
    for row in tables.get(f"{SOURCE_ID}.observations", []):
        if row.get("visit_key"):
            obs_by_visit[row["visit_key"]].append(row)

    problems_by_visit = defaultdict(list)
    for row in tables.get(f"{SOURCE_ID}.problems", []):
        if row.get("visit_key"):
            problems_by_visit[row["visit_key"]].append(row)

    encounters = []
    for row in tables.get(f"{SOURCE_ID}.visits", []):
        visit_key = row.get("visit_key")
        if not visit_key or not row.get("member_id"):
            continue   # drop a row missing its identity keys rather than crash
        enc = Encounter(
            source_id=SOURCE_ID,
            source_encounter_id=visit_key,
            source_patient_key=row["member_id"],
            encounter_type=normalize_encounter_type(row.get("visit_class")),
            specialty=row.get("department"),       # rough proxy for specialty, not standardized
            start_date=parse_date(row.get("admit_date")),
            end_date=parse_date(row.get("discharge_date")),
            patient_age=row.get("age_years"),
            patient_sex=normalize_sex(row.get("gender")),
        )

        for doc in docs_by_visit.get(visit_key, []):
            text = doc.get("doc_text")
            if text:
                enc.notes.append(ClinicalNote(
                    doc_type=doc.get("doc_type"),
                    text=text,
                    authored_at=parse_datetime(doc.get("authored_at")),
                ))

        for obs in obs_by_visit.get(visit_key, []):
            kind = (obs.get("obs_type") or "").strip().lower()
            enc.observations.append(Observation(
                obs_kind=kind if kind in ("lab", "vital") else "lab",
                name=obs.get("obs_name"),
                value=obs.get("obs_value"),
                unit=obs.get("obs_unit"),
                observed_at=parse_datetime(obs.get("obs_datetime")),
            ))

        for prob in problems_by_visit.get(visit_key, []):
            # SNOMED when coded; free text only when concept_code is null. We keep the raw
            # system and never coerce a SNOMED code into an ICD slot.
            enc.diagnoses.append(Diagnosis(
                concept=CodedConcept(
                    system=prob.get("coding_system") or None,
                    code=prob.get("concept_code") or None,
                    description=prob.get("problem_text"),
                ),
                diagnosed_at=parse_date(prob.get("noted_date")),
            ))

        encounters.append(enc)
    return encounters
