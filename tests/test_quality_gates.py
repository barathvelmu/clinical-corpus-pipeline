"""Each gate must catch the specific defect it exists for, and a clean example must pass
every gate. These tests are the proof that the quality layer actually works, not just that
it runs."""

import unittest

from pipeline.adapters.registry import build_all_claims, build_all_encounters
from pipeline.casebuild import build_case
from pipeline.linkage import build_timelines
from pipeline.posttrain import generate as gen
from pipeline.quality.gates import (
    Deduplicator, answerability_gate, clinical_plausibility_gate, grounding_gate,
    link_confidence_gate, phi_gate, run_gates,
)
from pipeline.quality.score import summarize
from pipeline.synth.generate_synth import generate


def _cases():
    tables = generate()
    encs = build_all_encounters(tables)
    clms = build_all_claims(tables)
    tls = build_timelines(encs, clms, tables["linkage.patient_tokens"])
    return {tok: build_case(tl) for tok, tl in tls.items()}


class TestGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()

    def _facts(self, token):
        return self.cases[token].facts

    def test_clean_example_passes_all_blocking_gates(self):
        case = self.cases["TOKEN_001"]
        ex = gen.build_summary_example(case)
        results = run_gates(ex, case.facts)
        report = summarize(ex, results)
        self.assertTrue(report.accepted, msg=report.notes)
        self.assertEqual(report.score, 1.0)

    def test_grounding_gate_catches_hallucinated_medication(self):
        case = self.cases["TOKEN_001"]
        ex = gen.make_unsupported_claim_candidate(case)
        r = grounding_gate(ex, case.facts)
        self.assertFalse(r.passed)
        self.assertIn("insulin glargine", r.message)

    def test_phi_gate_catches_leaked_identifiers(self):
        case = self.cases["TOKEN_003"]
        ex = gen.make_phi_leak_candidate(case)
        r = phi_gate(ex, case.facts)
        self.assertFalse(r.passed)

    def test_link_confidence_gate_rejects_untrusted_link(self):
        case = self.cases["TOKEN_004"]
        ex = gen.build_summary_example(case)
        r = link_confidence_gate(ex, case.facts)
        self.assertFalse(r.passed)
        self.assertIn("0.85", r.message)

    def test_disagreement_routes_to_review_not_auto_decision(self):
        # Any chart/claims disagreement is held for a clinician, never auto-adjudicated,
        # whether the example surfaced it well or adopted the billing code.
        case = self.cases["TOKEN_005"]
        for ex in (gen.build_summary_example(case),
                   gen.make_disagreement_asserted_candidate(case)):
            r = clinical_plausibility_gate(ex, case.facts)
            self.assertTrue(r.needs_review)
            report = summarize(ex, run_gates(ex, case.facts, Deduplicator()))
            self.assertEqual(report.disposition, "hold")
            self.assertFalse(report.accepted)   # held, so never auto-shipped

    def test_no_false_disagreement_on_snomed_vs_icd(self):
        # TOKEN_002: chart SNOMED CHF vs claim ICD I50.9 are the same condition. The
        # disagreement check must NOT fire across coding systems.
        facts = self._facts("TOKEN_002")
        self.assertFalse(facts.has_ehr_claims_disagreement())

    def test_answerability_gate_flags_fact_absent_from_context(self):
        case = self.cases["TOKEN_001"]
        ex = gen.make_unsupported_claim_candidate(case)
        r = answerability_gate(ex, case.facts)
        self.assertFalse(r.passed)

    def test_dedup_gate_catches_identical_second_example(self):
        case = self.cases["TOKEN_001"]
        ex = gen.build_summary_example(case)
        dd = Deduplicator()
        self.assertTrue(dd.gate(ex, case.facts).passed)        # first time: unique
        self.assertFalse(dd.gate(ex, case.facts).passed)       # second time: duplicate


if __name__ == "__main__":
    unittest.main()
