"""Adapter: EHR Provider A -> canonical.

Provider A is encounter-centric. The clinical note is inline on the encounter row.
Labs, medications, and diagnoses live in their own tables keyed by encounter_id.
Diagnoses are ICD-10.
"""

from __future__ import annotations

from collections import defaultdict

from pipeline.canonical import (
    ClinicalNote, CodedConcept, Diagnosis, Encounter, Medication, Observation,
    normalize_encounter_type, normalize_sex, parse_date,
)

SOURCE_ID = "ehr_provider_a"


def build_encounters(tables: dict[str, list[dict]]) -> list[Encounter]:
    # Group child rows by their parent key, skipping any row missing it. A malformed row
    # is dropped, not allowed to crash the batch: at 10M-1B encounters per provider we
    # cannot let one bad row take down the run.
    labs_by_enc = defaultdict(list)
    for row in tables.get(f"{SOURCE_ID}.labs", []):
        if row.get("encounter_id"):
            labs_by_enc[row["encounter_id"]].append(row)

    meds_by_enc = defaultdict(list)
    for row in tables.get(f"{SOURCE_ID}.medications", []):
        if row.get("encounter_id"):
            meds_by_enc[row["encounter_id"]].append(row)

    dx_by_enc = defaultdict(list)
    for row in tables.get(f"{SOURCE_ID}.diagnoses", []):
        if row.get("encounter_id"):
            dx_by_enc[row["encounter_id"]].append(row)

    encounters = []
    for row in tables.get(f"{SOURCE_ID}.encounters", []):
        enc_id = row.get("encounter_id")
        if not enc_id or not row.get("patient_id"):
            continue   # drop a row missing its identity keys rather than crash
        enc = Encounter(
            source_id=SOURCE_ID,
            source_encounter_id=enc_id,
            source_patient_key=row["patient_id"],
            encounter_type=normalize_encounter_type(row.get("encounter_type")),
            specialty=row.get("specialty"),
            start_date=parse_date(row.get("encounter_date")),
            end_date=None,
            patient_age=row.get("patient_age"),
            patient_sex=normalize_sex(row.get("patient_sex")),
        )
        note = row.get("clinical_note")
        if note:
            enc.notes.append(ClinicalNote(doc_type="inline", text=note))

        for lab in labs_by_enc.get(enc_id, []):
            enc.observations.append(Observation(
                obs_kind="lab",
                name=lab.get("test_name"),
                value=lab.get("value"),
                unit=lab.get("unit"),
                reference_range=lab.get("reference_range"),
                abnormal_flag=lab.get("abnormal_flag"),
                observed_at=None,
            ))

        for med in meds_by_enc.get(enc_id, []):
            enc.medications.append(Medication(
                drug_name=med.get("drug_name"),
                dose=med.get("dose"),
                route=med.get("route"),
                frequency=med.get("frequency"),
                start_date=parse_date(med.get("start_date")),
                end_date=parse_date(med.get("end_date")),
            ))

        for dx in dx_by_enc.get(enc_id, []):
            enc.diagnoses.append(Diagnosis(
                concept=CodedConcept(
                    system=dx.get("code_system"),
                    code=dx.get("code"),
                    description=dx.get("description"),
                ),
                diagnosed_at=parse_date(dx.get("dx_date")),
            ))

        encounters.append(enc)
    return encounters
