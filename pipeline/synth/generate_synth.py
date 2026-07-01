"""Synthetic source data, generated to match the source provider DDLs.

Why synthetic, and why hand-built rather than random: the deep piece is an
evidence-grounded clinical reasoning summary, so the cases have to be clinically
*coherent* (a diabetic with a high A1c on metformin, not random noise) or the summaries
are meaningless and the grounding gate has nothing real to check. We therefore author a
small set of coherent patient cases, then deliberately plant the four failure modes the
quality gates exist to catch:

    1. low linkage confidence   (token confidence below threshold)
    2. PHI leak                 (a raw note carrying a fake name / DOB / MRN)
    3. unsupported claim         (material for a hallucinated-fact candidate)
    4. EHR / claims disagreement (claim dx that contradicts the clinical record)

Every identifier, name, and date here is invented. There is zero real PHI in this file.

Output shape mirrors a warehouse extract: a dict of "source_id.table" -> list of row
dicts, with keys exactly matching the DDL columns. Adapters consume these rows.
"""

from __future__ import annotations

EHR_A = "ehr_provider_a"
EHR_B = "ehr_provider_b"
CLAIMS_X = "claims_provider_x"


def _empty_tables() -> dict[str, list[dict]]:
    return {
        f"{EHR_A}.encounters": [], f"{EHR_A}.labs": [],
        f"{EHR_A}.medications": [], f"{EHR_A}.diagnoses": [],
        f"{EHR_B}.visits": [], f"{EHR_B}.documents": [],
        f"{EHR_B}.observations": [], f"{EHR_B}.problems": [],
        f"{CLAIMS_X}.claim_lines": [],
        "linkage.patient_tokens": [],
    }


def _token_row(source_id, patient_key, token_value, confidence):
    return {
        "source_id": source_id, "patient_key": patient_key,
        "token_value": token_value, "confidence": confidence,
        "generated_at": "2026-06-01 00:00:00",
    }


def generate() -> dict[str, list[dict]]:
    """Build the full synthetic dataset. Deterministic: same output every run, so the
    tests and RESULTS numbers are reproducible."""
    t = _empty_tables()

    # ---- TOKEN_001: type 2 diabetes, Provider A, clean and coherent. The "happy path"
    # case that should produce an accepted example, and the source for the planted
    # unsupported-claim candidate later.
    t[f"{EHR_A}.encounters"].append({
        "encounter_id": "A-ENC-001", "patient_id": "A-PT-1001",
        "encounter_date": "2026-03-10", "encounter_type": "office visit",
        "specialty": "Endocrinology", "patient_age": 58, "patient_sex": "F",
        "clinical_note": ("58F with type 2 diabetes here for routine follow-up. Reports "
                          "polyuria and fatigue over the past month. Adherent to "
                          "metformin. No chest pain, no vision changes."),
        "ingested_at": "2026-03-10 14:00:00",
    })
    t[f"{EHR_A}.labs"].extend([
        {"lab_id": "A-LAB-001", "encounter_id": "A-ENC-001", "patient_id": "A-PT-1001",
         "test_name": "Hemoglobin A1c", "value": "9.1", "unit": "%",
         "reference_range": "4.0-5.6", "abnormal_flag": "H", "result_date": "2026-03-10"},
        {"lab_id": "A-LAB-002", "encounter_id": "A-ENC-001", "patient_id": "A-PT-1001",
         "test_name": "Glucose, fasting", "value": "187", "unit": "mg/dL",
         "reference_range": "70-99", "abnormal_flag": "H", "result_date": "2026-03-10"},
    ])
    t[f"{EHR_A}.medications"].append(
        {"med_id": "A-MED-001", "encounter_id": "A-ENC-001", "patient_id": "A-PT-1001",
         "drug_name": "Metformin", "dose": "1000mg", "route": "oral",
         "frequency": "twice daily", "start_date": "2025-01-15", "end_date": None})
    t[f"{EHR_A}.diagnoses"].append(
        {"dx_id": "A-DX-001", "encounter_id": "A-ENC-001", "patient_id": "A-PT-1001",
         "code_system": "ICD-10", "code": "E11.9",
         "description": "Type 2 diabetes mellitus without complications", "dx_date": "2026-03-10"})
    t[f"{CLAIMS_X}.claim_lines"].append(
        {"claim_id": "X-CLM-001", "member_id": "X-MBR-9001", "service_date": "2026-03-10",
         "claim_type": "professional", "place_of_service": "11",
         "procedure_code": "99214", "procedure_system": "CPT",
         "diagnosis_codes": "E11.9|E78.5", "billed_amount": 240.0, "paid_amount": 156.0,
         "rendering_npi": "1598877766"})
    t["linkage.patient_tokens"].extend([
        _token_row(EHR_A, "A-PT-1001", "TOKEN_001", "0.99"),
        _token_row(CLAIMS_X, "X-MBR-9001", "TOKEN_001", "0.99"),
    ])

    # ---- TOKEN_002: CHF exacerbation, Provider B (visit-centric, notes in documents,
    # labs+vitals merged, SNOMED problems, admit/discharge dates). Tests the harder
    # adapter and produces a second clean accepted example.
    t[f"{EHR_B}.visits"].append({
        "visit_key": "B-VIS-002", "member_id": "B-MEM-2002",
        "admit_date": "2026-02-02", "discharge_date": "2026-02-06", "visit_class": "IP",
        "department": "Cardiology", "age_years": 71, "gender": "M",
        "loaded_at": "2026-02-06 09:00:00"})
    t[f"{EHR_B}.documents"].append({
        "doc_id": "B-DOC-002", "visit_key": "B-VIS-002", "member_id": "B-MEM-2002",
        "doc_type": "discharge summary",
        "doc_text": ("71M admitted with progressive dyspnea and bilateral leg swelling. "
                     "Exam notable for elevated JVP and bibasilar crackles. Treated with "
                     "IV diuresis with good response. Discharged on oral furosemide."),
        "authored_at": "2026-02-06 08:30:00"})
    t[f"{EHR_B}.observations"].extend([
        {"obs_id": "B-OBS-002a", "visit_key": "B-VIS-002", "member_id": "B-MEM-2002",
         "obs_type": "lab", "obs_name": "BNP", "obs_value": "1240", "obs_unit": "pg/mL",
         "obs_datetime": "2026-02-02 12:00:00"},
        {"obs_id": "B-OBS-002b", "visit_key": "B-VIS-002", "member_id": "B-MEM-2002",
         "obs_type": "vital", "obs_name": "Blood Pressure", "obs_value": "148/92",
         "obs_unit": "mmHg", "obs_datetime": "2026-02-02 12:05:00"},
    ])
    t[f"{EHR_B}.problems"].append({
        "problem_id": "B-PRB-002", "visit_key": "B-VIS-002", "member_id": "B-MEM-2002",
        "coding_system": "SNOMED", "concept_code": "42343007",
        "problem_text": "Congestive heart failure", "noted_date": "2026-02-02"})
    t[f"{CLAIMS_X}.claim_lines"].append(
        {"claim_id": "X-CLM-002", "member_id": "X-MBR-9002", "service_date": "2026-02-02",
         "claim_type": "institutional", "place_of_service": "21",
         "procedure_code": "99223", "procedure_system": "CPT",
         "diagnosis_codes": "I50.9", "billed_amount": 8200.0, "paid_amount": None,
         "rendering_npi": "1455667788"})
    t["linkage.patient_tokens"].extend([
        _token_row(EHR_B, "B-MEM-2002", "TOKEN_002", "0.98"),
        _token_row(CLAIMS_X, "X-MBR-9002", "TOKEN_002", "0.98"),
    ])

    # ---- TOKEN_003: PHI LEAK. The raw note deliberately carries identifiers (fake name,
    # DOB, MRN, phone). The structured facts are clean, so a grounded summary is fine; but
    # an LLM that copies note prose verbatim would leak PHI. We use this note to build a
    # planted-bad candidate that the PHI gate must catch.
    t[f"{EHR_A}.encounters"].append({
        "encounter_id": "A-ENC-003", "patient_id": "A-PT-1003",
        "encounter_date": "2026-04-01", "encounter_type": "office visit",
        "specialty": "Primary Care", "patient_age": 64, "patient_sex": "M",
        "clinical_note": ("Patient Robert Langdon (DOB 04/12/1961, MRN 5567281, phone "
                          "617-555-0148) seen for hypertension follow-up. Home BP log "
                          "shows readings around 150/95. Tolerating lisinopril well."),
        "ingested_at": "2026-04-01 10:00:00"})
    t[f"{EHR_A}.labs"].append(
        {"lab_id": "A-LAB-003", "encounter_id": "A-ENC-003", "patient_id": "A-PT-1003",
         "test_name": "Potassium", "value": "4.4", "unit": "mmol/L",
         "reference_range": "3.5-5.1", "abnormal_flag": None, "result_date": "2026-04-01"})
    t[f"{EHR_A}.medications"].append(
        {"med_id": "A-MED-003", "encounter_id": "A-ENC-003", "patient_id": "A-PT-1003",
         "drug_name": "Lisinopril", "dose": "20mg", "route": "oral", "frequency": "daily",
         "start_date": "2025-06-01", "end_date": None})
    t[f"{EHR_A}.diagnoses"].append(
        {"dx_id": "A-DX-003", "encounter_id": "A-ENC-003", "patient_id": "A-PT-1003",
         "code_system": "ICD-10", "code": "I10", "description": "Essential hypertension",
         "dx_date": "2026-04-01"})
    t[f"{CLAIMS_X}.claim_lines"].append(
        {"claim_id": "X-CLM-003", "member_id": "X-MBR-9003", "service_date": "2026-04-01",
         "claim_type": "professional", "place_of_service": "11",
         "procedure_code": "99213", "procedure_system": "CPT", "diagnosis_codes": "I10",
         "billed_amount": 180.0, "paid_amount": 110.0, "rendering_npi": "1598877766"})
    t["linkage.patient_tokens"].extend([
        _token_row(EHR_A, "A-PT-1003", "TOKEN_003", "0.97"),
        _token_row(CLAIMS_X, "X-MBR-9003", "TOKEN_003", "0.97"),
    ])

    # ---- TOKEN_004: LOW LINKAGE CONFIDENCE. The EHR<->claims token match is only 0.85,
    # below our 0.95 threshold. We must not fuse claims into a clinical example for this
    # patient on a link we do not trust; linkage flags it and the example is rejected.
    t[f"{EHR_A}.encounters"].append({
        "encounter_id": "A-ENC-004", "patient_id": "A-PT-1004",
        "encounter_date": "2026-05-12", "encounter_type": "ER",
        "specialty": "Emergency", "patient_age": 33, "patient_sex": "F",
        "clinical_note": ("33F presenting with acute right lower quadrant abdominal pain "
                          "and nausea. Tender to palpation at McBurney point."),
        "ingested_at": "2026-05-12 22:00:00"})
    t[f"{EHR_A}.diagnoses"].append(
        {"dx_id": "A-DX-004", "encounter_id": "A-ENC-004", "patient_id": "A-PT-1004",
         "code_system": "ICD-10", "code": "K35.80", "description": "Acute appendicitis",
         "dx_date": "2026-05-12"})
    t[f"{CLAIMS_X}.claim_lines"].append(
        {"claim_id": "X-CLM-004", "member_id": "X-MBR-9004", "service_date": "2026-05-12",
         "claim_type": "institutional", "place_of_service": "23",
         "procedure_code": "44970", "procedure_system": "CPT", "diagnosis_codes": "K35.80",
         "billed_amount": 15400.0, "paid_amount": None, "rendering_npi": "1455667788"})
    t["linkage.patient_tokens"].extend([
        _token_row(EHR_A, "A-PT-1004", "TOKEN_004", "0.99"),
        _token_row(CLAIMS_X, "X-MBR-9004", "TOKEN_004", "0.85"),   # the shaky link
    ])

    # ---- TOKEN_005: EHR / CLAIMS DISAGREEMENT. The clinical record says asthma; the
    # claim for the same date bills an unrelated, higher-reimbursing respiratory code.
    # Claims are billing-driven and are not clinical ground truth, so a summary that
    # silently asserts the claim dx as the diagnosis must be flagged.
    t[f"{EHR_A}.encounters"].append({
        "encounter_id": "A-ENC-005", "patient_id": "A-PT-1005",
        "encounter_date": "2026-01-20", "encounter_type": "office visit",
        "specialty": "Pulmonology", "patient_age": 27, "patient_sex": "M",
        "clinical_note": ("27M with intermittent wheeze and cough, worse with exercise. "
                          "Lungs with scattered expiratory wheeze. Started on albuterol."),
        "ingested_at": "2026-01-20 11:00:00"})
    t[f"{EHR_A}.medications"].append(
        {"med_id": "A-MED-005", "encounter_id": "A-ENC-005", "patient_id": "A-PT-1005",
         "drug_name": "Albuterol HFA", "dose": "90mcg", "route": "inhaled",
         "frequency": "as needed", "start_date": "2026-01-20", "end_date": None})
    t[f"{EHR_A}.diagnoses"].append(
        {"dx_id": "A-DX-005", "encounter_id": "A-ENC-005", "patient_id": "A-PT-1005",
         "code_system": "ICD-10", "code": "J45.909",
         "description": "Unspecified asthma, uncomplicated", "dx_date": "2026-01-20"})
    t[f"{CLAIMS_X}.claim_lines"].append(
        {"claim_id": "X-CLM-005", "member_id": "X-MBR-9005", "service_date": "2026-01-20",
         "claim_type": "professional", "place_of_service": "11",
         "procedure_code": "94060", "procedure_system": "CPT",
         "diagnosis_codes": "J44.9",   # COPD on the claim vs asthma in the chart
         "billed_amount": 320.0, "paid_amount": 210.0, "rendering_npi": "1598877766"})
    t["linkage.patient_tokens"].extend([
        _token_row(EHR_A, "A-PT-1005", "TOKEN_005", "0.96"),
        _token_row(CLAIMS_X, "X-MBR-9005", "TOKEN_005", "0.96"),
    ])

    return t


# Stable lists of which tokens carry which planted defect, so tests and the report can
# assert against ground truth rather than rediscovering it.
PLANTED = {
    "clean": ["TOKEN_001", "TOKEN_002"],
    "phi_leak_source": "TOKEN_003",
    "low_confidence": "TOKEN_004",
    "ehr_claims_disagreement": "TOKEN_005",
    "unsupported_claim_source": "TOKEN_001",
}


# ---------------------------------------------------------------------------------------
# Scaled corpus generator
#
# `generate()` above is the hand-built, fully-controlled set used by the unit tests and the
# defect demo. `generate_corpus()` below is for volume: it instantiates clinically coherent
# archetypes across both provider shapes, with seeded randomness, so we can run the whole
# pipeline over hundreds of patients and report aggregate metrics (see scripts/eval_corpus).
# It deliberately mixes in the same failure modes at controlled rates (low-confidence links,
# EHR/claims disagreements) so the metrics reflect realistic yield, not a clean best case.
# Still zero real PHI: every value is drawn from these templates.
# ---------------------------------------------------------------------------------------

import random

# Each archetype is clinically coherent: the dx, labs, and meds belong together. `claim_dx`
# is the ICD normally billed; `conflict_dx` is an unrelated code we sometimes bill instead
# to simulate an EHR/claims disagreement.
_ARCHETYPES = [
    {"specialty": "Endocrinology", "dx": ("E11.9", "Type 2 diabetes mellitus without complications"),
     "labs": [("Hemoglobin A1c", "9.1", "%", "H"), ("Glucose, fasting", "187", "mg/dL", "H")],
     "meds": [("Metformin", "1000mg", "twice daily")], "proc": "99214",
     "claim_dx": "E11.9", "conflict_dx": "M54.5",
     "note": "here for diabetes follow-up with polyuria and fatigue; adherent to therapy."},
    {"specialty": "Cardiology", "dx": ("I50.9", "Heart failure, unspecified"),
     "labs": [("BNP", "1240", "pg/mL", "H")], "meds": [("Furosemide", "40mg", "daily")],
     "proc": "99223", "claim_dx": "I50.9", "conflict_dx": "J44.9",
     "note": "admitted with dyspnea and bilateral leg swelling; elevated JVP."},
    {"specialty": "Pulmonology", "dx": ("J45.909", "Unspecified asthma, uncomplicated"),
     "labs": [], "meds": [("Albuterol HFA", "90mcg", "as needed")], "proc": "94060",
     "claim_dx": "J45.909", "conflict_dx": "J44.9",
     "note": "intermittent wheeze and cough worse with exercise; scattered wheeze on exam."},
    {"specialty": "Primary Care", "dx": ("I10", "Essential hypertension"),
     "labs": [("Potassium", "4.4", "mmol/L", None)], "meds": [("Lisinopril", "20mg", "daily")],
     "proc": "99213", "claim_dx": "I10", "conflict_dx": "E78.5",
     "note": "hypertension follow-up; home readings around 150/95; tolerating therapy."},
    {"specialty": "Nephrology", "dx": ("N18.3", "Chronic kidney disease, stage 3"),
     "labs": [("Creatinine", "2.1", "mg/dL", "H")], "meds": [("Losartan", "50mg", "daily")],
     "proc": "99214", "claim_dx": "N18.3", "conflict_dx": "I10",
     "note": "CKD follow-up; stable creatinine; no edema."},
    {"specialty": "Gastroenterology", "dx": ("K21.9", "Gastro-esophageal reflux disease"),
     "labs": [], "meds": [("Omeprazole", "20mg", "daily")], "proc": "99213",
     "claim_dx": "K21.9", "conflict_dx": "K35.80",
     "note": "reflux symptoms after meals; improved on acid suppression."},
    {"specialty": "Hematology", "dx": ("D64.9", "Anemia, unspecified"),
     "labs": [("Hemoglobin", "9.4", "g/dL", "L"), ("Ferritin", "12", "ng/mL", "L")],
     "meds": [], "proc": "85025", "claim_dx": "D64.9", "conflict_dx": "C81.9",
     "note": "fatigue with microcytic indices; iron studies consistent with deficiency."},
    {"specialty": "Endocrinology", "dx": ("E03.9", "Hypothyroidism, unspecified"),
     "labs": [("TSH", "11.2", "mIU/L", "H")], "meds": [("Levothyroxine", "75mcg", "daily")],
     "proc": "99214", "claim_dx": "E03.9", "conflict_dx": "F41.9",
     "note": "fatigue and cold intolerance; TSH elevated; dose titration ongoing."},
]

# SNOMED equivalents so Provider-B archetypes carry SNOMED codes (exercises the
# cross-system, can't-auto-compare path), keyed by the ICD dx.
_SNOMED = {
    "E11.9": "44054006", "I50.9": "42343007", "J45.909": "195967001", "I10": "38341003",
    "N18.3": "709044004", "K21.9": "235595009", "D64.9": "271737000", "E03.9": "40930008",
}


def generate_corpus(n_patients: int = 200, seed: int = 7,
                    low_conf_rate: float = 0.08,
                    disagreement_rate: float = 0.12) -> dict[str, list[dict]]:
    """Build a larger synthetic dataset: n_patients across both provider shapes, with a
    controlled fraction of low-confidence links and EHR/claims disagreements."""
    rng = random.Random(seed)
    t = _empty_tables()

    for i in range(n_patients):
        arch = rng.choice(_ARCHETYPES)
        token = f"TKN_{i:05d}"
        age = rng.randint(24, 88)
        sex = rng.choice(["M", "F"])
        disagree = rng.random() < disagreement_rate
        claim_dx = arch["conflict_dx"] if disagree else arch["claim_dx"]
        ehr_conf = "0.99"
        claim_conf = "0.85" if rng.random() < low_conf_rate else f"0.9{rng.randint(6, 9)}"

        if rng.random() < 0.5:
            _emit_provider_a(t, i, token, age, sex, arch, ehr_conf)
        else:
            _emit_provider_b(t, i, token, age, sex, arch, ehr_conf)
        _emit_claim(t, i, token, arch, claim_dx, claim_conf)

    return t


def _emit_provider_a(t, i, token, age, sex, arch, conf):
    pid = f"A-{i:05d}"
    enc = f"AE-{i:05d}"
    t[f"{EHR_A}.encounters"].append({
        "encounter_id": enc, "patient_id": pid, "encounter_date": "2026-03-10",
        "encounter_type": "office visit", "specialty": arch["specialty"],
        "patient_age": age, "patient_sex": sex,
        "clinical_note": f"{age}{sex} {arch['note']}", "ingested_at": "2026-03-10 12:00:00"})
    for j, (name, val, unit, flag) in enumerate(arch["labs"]):
        t[f"{EHR_A}.labs"].append({
            "lab_id": f"AL-{i:05d}-{j}", "encounter_id": enc, "patient_id": pid,
            "test_name": name, "value": val, "unit": unit, "reference_range": None,
            "abnormal_flag": flag, "result_date": "2026-03-10"})
    for j, (drug, dose, freq) in enumerate(arch["meds"]):
        t[f"{EHR_A}.medications"].append({
            "med_id": f"AM-{i:05d}-{j}", "encounter_id": enc, "patient_id": pid,
            "drug_name": drug, "dose": dose, "route": "oral", "frequency": freq,
            "start_date": "2025-06-01", "end_date": None})
    code, desc = arch["dx"]
    t[f"{EHR_A}.diagnoses"].append({
        "dx_id": f"AD-{i:05d}", "encounter_id": enc, "patient_id": pid,
        "code_system": "ICD-10", "code": code, "description": desc, "dx_date": "2026-03-10"})
    t["linkage.patient_tokens"].append(_token_row(EHR_A, pid, token, conf))


def _emit_provider_b(t, i, token, age, sex, arch, conf):
    mid = f"B-{i:05d}"
    vis = f"BV-{i:05d}"
    t[f"{EHR_B}.visits"].append({
        "visit_key": vis, "member_id": mid, "admit_date": "2026-02-02",
        "discharge_date": "2026-02-05" if arch["specialty"] == "Cardiology" else None,
        "visit_class": "IP" if arch["specialty"] == "Cardiology" else "OP",
        "department": arch["specialty"], "age_years": age, "gender": sex,
        "loaded_at": "2026-02-05 09:00:00"})
    t[f"{EHR_B}.documents"].append({
        "doc_id": f"BD-{i:05d}", "visit_key": vis, "member_id": mid,
        "doc_type": "progress note", "doc_text": f"{age}{sex} {arch['note']}",
        "authored_at": "2026-02-02 10:00:00"})
    for j, (name, val, unit, flag) in enumerate(arch["labs"]):
        t[f"{EHR_B}.observations"].append({
            "obs_id": f"BO-{i:05d}-{j}", "visit_key": vis, "member_id": mid,
            "obs_type": "lab", "obs_name": name, "obs_value": val, "obs_unit": unit,
            "obs_datetime": "2026-02-02 11:00:00"})
    code, desc = arch["dx"]
    t[f"{EHR_B}.problems"].append({
        "problem_id": f"BP-{i:05d}", "visit_key": vis, "member_id": mid,
        "coding_system": "SNOMED", "concept_code": _SNOMED.get(code), "problem_text": desc,
        "noted_date": "2026-02-02"})
    t["linkage.patient_tokens"].append(_token_row(EHR_B, mid, token, conf))


def _emit_claim(t, i, token, arch, claim_dx, conf):
    cid = f"X-{i:05d}"
    mbr = f"XM-{i:05d}"
    t[f"{CLAIMS_X}.claim_lines"].append({
        "claim_id": cid, "member_id": mbr, "service_date": "2026-03-10",
        "claim_type": "professional", "place_of_service": "11",
        "procedure_code": arch["proc"], "procedure_system": "CPT",
        "diagnosis_codes": claim_dx, "billed_amount": 240.0, "paid_amount": 150.0,
        "rendering_npi": "1598877766"})
    t["linkage.patient_tokens"].append(_token_row(CLAIMS_X, mbr, token, conf))


if __name__ == "__main__":
    import json
    tables = generate()
    counts = {name: len(rows) for name, rows in tables.items()}
    print(json.dumps(counts, indent=2))
