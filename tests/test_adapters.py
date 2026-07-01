"""Adapters are load-bearing: every downstream guarantee rests on raw rows mapping to the
canonical model correctly. These tests pin the tricky parts of each provider's dialect."""

import unittest

from pipeline.adapters import claims_x, provider_a, provider_b
from pipeline.synth.generate_synth import generate


class TestProviderA(unittest.TestCase):
    def setUp(self):
        self.encs = provider_a.build_encounters(generate())

    def test_inline_note_becomes_clinical_note(self):
        enc = self._by_id("A-ENC-001")
        self.assertEqual(len(enc.notes), 1)
        self.assertEqual(enc.notes[0].doc_type, "inline")
        self.assertIn("type 2 diabetes", enc.notes[0].text.lower())

    def test_labs_meds_dx_attached_by_encounter(self):
        enc = self._by_id("A-ENC-001")
        self.assertEqual(len(enc.observations), 2)
        self.assertEqual(len(enc.medications), 1)
        self.assertEqual(len(enc.diagnoses), 1)
        self.assertEqual(enc.diagnoses[0].concept.code, "E11.9")

    def test_encounter_type_and_sex_normalized(self):
        enc = self._by_id("A-ENC-001")
        self.assertEqual(enc.encounter_type, "ambulatory")  # "office visit" -> ambulatory
        self.assertEqual(enc.patient_sex, "female")          # "F" -> female

    def test_numeric_lab_parse(self):
        enc = self._by_id("A-ENC-001")
        a1c = next(o for o in enc.observations if "a1c" in o.name.lower())
        self.assertEqual(a1c.numeric_value, 9.1)

    def _by_id(self, enc_id):
        return next(e for e in self.encs if e.source_encounter_id == enc_id)


class TestProviderB(unittest.TestCase):
    def setUp(self):
        self.encs = provider_b.build_encounters(generate())

    def test_admit_discharge_map_to_start_end(self):
        enc = self.encs[0]
        self.assertIsNotNone(enc.start_date)       # admit_date
        self.assertIsNotNone(enc.end_date)         # discharge_date
        self.assertLess(enc.start_date, enc.end_date)

    def test_observations_split_lab_and_vital(self):
        enc = self.encs[0]
        kinds = {o.obs_kind for o in enc.observations}
        self.assertEqual(kinds, {"lab", "vital"})

    def test_documents_become_notes(self):
        enc = self.encs[0]
        self.assertEqual(len(enc.notes), 1)
        self.assertEqual(enc.notes[0].doc_type, "discharge summary")

    def test_snomed_problem_keeps_its_system(self):
        enc = self.encs[0]
        dx = enc.diagnoses[0]
        self.assertEqual(dx.concept.system, "SNOMED")
        self.assertEqual(dx.concept.code, "42343007")

    def test_visit_class_normalized(self):
        enc = self.encs[0]
        self.assertEqual(enc.encounter_type, "inpatient")  # "IP" -> inpatient


class TestClaimsX(unittest.TestCase):
    def setUp(self):
        self.claims = claims_x.build_claims(generate())

    def test_delimited_diagnoses_are_split(self):
        clm = next(c for c in self.claims if c.claim_id == "X-CLM-001")
        codes = [d.code for d in clm.diagnoses]
        self.assertEqual(codes, ["E11.9", "E78.5"])   # "E11.9|E78.5" split

    def test_procedure_mapped(self):
        clm = next(c for c in self.claims if c.claim_id == "X-CLM-001")
        self.assertEqual(clm.procedure.code, "99214")

    def test_null_paid_amount_preserved(self):
        clm = next(c for c in self.claims if c.claim_id == "X-CLM-002")
        self.assertIsNone(clm.paid_amount)   # claims lag; not yet adjudicated


if __name__ == "__main__":
    unittest.main()
