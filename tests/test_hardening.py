"""Regression tests for bypasses found during adversarial review.

Each test pins a hole that an earlier version of the gates let through. They are the proof
that the hardening is real: remove a fix and the matching test goes red. Keeping the
attacks as tests is the difference between "we think it is robust" and "we showed it."
"""

import unittest

from pipeline import deid
from pipeline.adapters.registry import build_all_claims, build_all_encounters
from pipeline.canonical import Claim, Encounter
from pipeline.casebuild import CaseFacts, build_case
from pipeline.linkage import PatientTimeline, build_timelines
from pipeline.posttrain import generate as gen
from pipeline.quality.gates import (
    Deduplicator, clinical_plausibility_gate, grounding_gate, link_confidence_gate,
    run_gates,
)
from pipeline.quality.score import summarize
from pipeline.synth.generate_synth import generate


def _cases():
    tables = generate()
    encs = build_all_encounters(tables)
    clms = build_all_claims(tables)
    tls = build_timelines(encs, clms, tables["linkage.patient_tokens"])
    return {tok: build_case(tl) for tok, tl in tls.items()}


class TestGroundingHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()

    def test_uncited_hallucinated_meds_in_prose_rejected(self):
        # The core bypass: state fabricated drugs in prose without citing them.
        case = self.cases["TOKEN_001"]
        ex = gen.build_summary_example(case)
        ex.messages[2]["content"] += (" Also started on insulin glargine 20 units and "
                                      "empagliflozin 10mg daily.")
        r = grounding_gate(ex, case.facts)
        self.assertFalse(r.passed)
        self.assertIn("uncited hallucination", r.message)

    def test_uncited_fabricated_lab_in_prose_rejected(self):
        case = self.cases["TOKEN_001"]
        ex = gen.build_summary_example(case)
        ex.messages[2]["content"] += " Troponin was elevated at 2.1 indicating infarction."
        r = grounding_gate(ex, case.facts)
        self.assertFalse(r.passed)

    def test_loose_substring_no_longer_resolves(self):
        f = self.cases["TOKEN_001"].facts
        self.assertFalse(f.contains("medication", "e"))       # used to match "metformin"
        self.assertFalse(f.contains("medication", "form"))    # used to match "metformin"
        self.assertTrue(f.contains("medication", "metformin"))

    def test_wrong_sex_not_grounded(self):
        f = self.cases["TOKEN_001"].facts   # patient is female
        self.assertFalse(f.contains("demographic", "male"))   # "male" must not match "female"
        self.assertTrue(f.contains("demographic", "female"))


class TestDisagreementHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()

    def test_adopting_billing_dx_is_never_auto_accepted(self):
        # The old hole: a fluent example that adopts the billing dx while name-dropping a
        # marker word got auto-accepted. Now any disagreement is HELD, so no phrasing can
        # auto-ship it.
        case = self.cases["TOKEN_005"]
        ex = gen.make_disagreement_asserted_candidate(case)
        ex.messages[2]["content"] = ("This is a 27-year-old male with COPD per the billing "
                                     "code J44.9; management should target COPD.")
        report = summarize(ex, run_gates(ex, case.facts, Deduplicator()))
        self.assertEqual(report.disposition, "hold")
        self.assertFalse(report.accepted)

    def test_morphology_catches_undisclosed_drug_classes(self):
        # The lexicon bypass: a hallucinated drug we never enumerated, caught by stem.
        case = self.cases["TOKEN_001"]
        for drug in ("semaglutide", "canagliflozin", "dabigatran"):
            ex = gen.build_summary_example(case)
            ex.messages[2]["content"] += f" Also started on {drug}."
            report = summarize(ex, run_gates(ex, case.facts, Deduplicator()))
            self.assertFalse(report.accepted, msg=f"{drug} should not have grounded")

    def test_diagnosis_modifier_fragment_does_not_resolve(self):
        f = self.cases["TOKEN_001"].facts   # dx "Type 2 diabetes mellitus without complications"
        self.assertFalse(f.contains("diagnosis", "uncomplicated"))
        self.assertFalse(f.contains("diagnosis", "without complications"))
        self.assertTrue(f.contains("diagnosis", "diabetes"))

    def test_phi_false_positives_reduced(self):
        # Smart over-flagging: these legitimate contents must NOT be flagged.
        self.assertFalse(deid.contains_phi("documented under SNOMED 442311008"))
        self.assertFalse(deid.contains_phi("seen in April 2024 for follow-up"))
        self.assertFalse(deid.contains_phi("reviewed by the Patient Care Team"))
        # ...while real identifiers still are.
        self.assertTrue(deid.contains_phi("seen on April 12, 2024"))
        self.assertTrue(deid.contains_phi("Patient Robert Langdon"))

    def test_third_pass_regressions(self):
        # Bypasses/regressions found in the THIRD adversarial pass, now closed.
        # 1. -tide/-tidine stems no longer flag common biochemistry words.
        self.assertFalse(deid.contains_phi("brain natriuretic peptide elevated"))
        # 2. The code-context filter is word-bounded: "Decode" must not suppress a real id.
        self.assertTrue(deid.contains_phi("Decode 987654321 quickly"))
        # 3. Ordinal-day dates are caught.
        self.assertTrue(deid.contains_phi("born 12th April 1961"))
        # 4. redact() does not corrupt on overlapping MRN + digit-run spans.
        self.assertEqual(deid.redact("MRN: 12345678901 stable"), "[MRN] stable")
        # 5. A real name survives a lowercase relationship cue (not dropped as a stopword).
        self.assertTrue(deid.contains_phi("Her husband John Carter drove her home"))


class TestNegationAndCompleteness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()

    def test_ruled_out_drug_is_not_a_hallucination(self):
        case = self.cases["TOKEN_001"]
        ex = gen.build_summary_example(case)
        ex.messages[2]["content"] += (" He was NOT started on empagliflozin and "
                                      "atorvastatin was ruled out.")
        report = summarize(ex, run_gates(ex, case.facts, Deduplicator()))
        self.assertTrue(report.accepted)

    def test_disguised_hallucination_still_caught(self):
        # A hallucination cannot hide behind an unrelated trailing "no issues".
        case = self.cases["TOKEN_001"]
        ex = gen.build_summary_example(case)
        ex.messages[2]["content"] += " Started semaglutide, no issues noted."
        report = summarize(ex, run_gates(ex, case.facts, Deduplicator()))
        self.assertFalse(report.accepted)

    def test_omitted_abnormal_lab_is_held(self):
        case = self.cases["TOKEN_001"]   # has abnormal A1c + glucose
        ex = gen.build_summary_example(case)
        ex.messages[2]["content"] = ("This is a 58-year-old female with type 2 diabetes; "
                                     "the visit was routine and the patient is stable.")
        report = summarize(ex, run_gates(ex, case.facts, Deduplicator()))
        self.assertEqual(report.disposition, "hold")


class TestRobustness(unittest.TestCase):
    def test_malformed_rows_do_not_crash_adapters(self):
        from pipeline.adapters import provider_a, claims_x
        tables = {
            "ehr_provider_a.encounters": [{"encounter_id": None},          # missing keys
                                          {"encounter_id": "E", "patient_id": "P",
                                           "patient_sex": 1, "encounter_type": 5}],
            "ehr_provider_a.labs": [{"value": "inf"}],                     # missing key + nan/inf
            "claims_provider_x.claim_lines": [{"diagnosis_codes": 12345,   # numeric, mixed
                                               "claim_id": "C", "member_id": "M"}],
        }
        encs = provider_a.build_encounters(tables)
        self.assertEqual(len(encs), 1)                 # the one valid row survived
        claims = claims_x.build_claims(tables)
        self.assertEqual([d.code for d in claims[0].diagnoses], ["12345"])

    def test_mixed_delimiter_claim_codes_split(self):
        from pipeline.adapters import claims_x
        claims = claims_x.build_claims({"claims_provider_x.claim_lines": [
            {"claim_id": "C", "member_id": "M", "diagnosis_codes": "E11.9|E78.5, I10"}]})
        self.assertEqual([d.code for d in claims[0].diagnoses], ["E11.9", "E78.5", "I10"])

    def test_token_collision_is_rejected(self):
        # Two different patients (different patient keys, same source) fused under one token
        # at perfect confidence must be caught as an identity conflict, not shipped.
        from pipeline.canonical import Encounter
        from pipeline.linkage import PatientTimeline
        e1 = Encounter(source_id="ehr_provider_a", source_encounter_id="E1",
                       source_patient_key="P1", patient_sex="male")
        e2 = Encounter(source_id="ehr_provider_a", source_encounter_id="E2",
                       source_patient_key="P2", patient_sex="female")
        e1.link_confidence = e2.link_confidence = 0.99
        tl = PatientTimeline(token="T", encounters=[e1, e2])
        self.assertTrue(tl.has_identity_conflict())


class TestPHIHardening(unittest.TestCase):
    def test_age_90_plus_hyphenated_caught(self):
        # The generator itself writes "92-year-old"; this must be flagged.
        self.assertTrue(deid.contains_phi("This is a 92-year-old female"))
        self.assertTrue(deid.contains_phi("105-year-old man"))

    def test_cued_name_caught_prose_not_flagged(self):
        self.assertTrue(deid.contains_phi("Patient Robert Langdon was seen"))
        self.assertTrue(deid.contains_phi("seen with her husband Robert Langdon"))
        # Must not flag ordinary clinical prose as a name.
        self.assertFalse(deid.contains_phi("patient reports fatigue and denies chest pain"))

    def test_record_number_cue_and_long_numeric_id(self):
        self.assertTrue(deid.contains_phi("Medical record number 5567281"))
        self.assertTrue(deid.contains_phi("id 12345678901"))   # 11-digit bare run


class TestLinkConfidenceHardening(unittest.TestCase):
    def test_none_confidence_forces_untrusted(self):
        # A populated 0.99 link must not mask a record whose confidence is unparseable.
        enc = Encounter(source_id="a", source_encounter_id="E", source_patient_key="p")
        enc.link_confidence = 0.99
        clm = Claim(source_id="b", claim_id="C", source_patient_key="p")
        clm.link_confidence = None
        tl = PatientTimeline(token="T", encounters=[enc], claims=[clm])
        self.assertIsNone(tl.min_link_confidence)
        self.assertFalse(tl.is_trusted())


if __name__ == "__main__":
    unittest.main()
