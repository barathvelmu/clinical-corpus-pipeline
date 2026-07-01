"""The whole pipeline, as one assertion: from synthetic rows to accepted/rejected
examples, every planted defect class is caught."""

import unittest

from pipeline.adapters.registry import build_all_claims, build_all_encounters
from pipeline.casebuild import build_case
from pipeline.linkage import build_timelines
from pipeline.posttrain import generate as gen
from pipeline.quality.gates import Deduplicator, run_gates
from pipeline.quality.score import summarize
from pipeline.synth.generate_synth import generate
from scripts.run_pipeline import build_candidates


class _StubLLM:
    """Deterministic stand-in for a frontier model: returns whatever text we hand it,
    through the same `.complete(system, user)` interface a real client would expose."""

    name = "stub-model"

    def __init__(self, text):
        self._text = text

    def complete(self, system, user):
        return self._text


class TestLLMSeam(unittest.TestCase):
    """The gates are path-agnostic: the same gates judge LLM output as templated output."""

    @classmethod
    def setUpClass(cls):
        tables = generate()
        encs, clms = build_all_encounters(tables), build_all_claims(tables)
        tls = build_timelines(encs, clms, tables["linkage.patient_tokens"])
        cls.case = build_case(tls["TOKEN_001"])

    def test_grounded_llm_output_is_accepted(self):
        grounded = gen.build_summary_example(self.case).response   # faithful text
        ex = gen.llm_generate(self.case, _StubLLM(grounded))
        self.assertEqual(ex.provenance.generation_method, "llm")
        report = summarize(ex, run_gates(ex, self.case.facts, Deduplicator()))
        self.assertTrue(report.accepted, msg=report.notes)

    def test_hallucinated_llm_output_is_rejected(self):
        bad = ("This is a 58-year-old female with type 2 diabetes, started on semaglutide "
               "and tirzepatide with good response.")
        ex = gen.llm_generate(self.case, _StubLLM(bad))
        report = summarize(ex, run_gates(ex, self.case.facts, Deduplicator()))
        self.assertFalse(report.accepted)   # the same grounding gate catches the drift


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates, cases = build_candidates()
        deduper = Deduplicator()
        cls.results = []
        for label, ex in candidates:
            facts = cases[ex.grounding.patient_token].facts
            report = summarize(ex, run_gates(ex, facts, deduper))
            cls.results.append((label, ex, report))

    def test_disposition_split(self):
        # 3 clean -> accept; 2 disagreement-related -> hold; 3 defects -> reject.
        by = {"accept": 0, "hold": 0, "reject": 0}
        for _, _, rep in self.results:
            by[rep.disposition] += 1
        self.assertEqual(by, {"accept": 3, "hold": 2, "reject": 3})

    def test_every_defect_class_is_caught(self):
        rejected_notes = " | ".join(
            note for _, _, rep in self.results if rep.disposition == "reject"
            for note in rep.notes)
        held_notes = " | ".join(
            note for _, _, rep in self.results if rep.disposition == "hold"
            for note in rep.notes)
        self.assertIn("link confidence", rejected_notes)        # low confidence -> reject
        self.assertIn("PHI detected", rejected_notes)            # PHI leak -> reject
        self.assertIn("hallucination", rejected_notes)           # unsupported claim -> reject
        self.assertIn("disagreement", held_notes)                # disagreement -> held for review

    def test_accepted_examples_are_fully_grounded(self):
        for label, ex, rep in self.results:
            if rep.accepted:
                self.assertTrue(ex.grounding.cited_facts)
                self.assertTrue(ex.grounding.source_encounter_ids)
                self.assertIsNotNone(ex.grounding.link_confidence)

    def test_accepted_examples_have_no_phi(self):
        from pipeline import deid
        for label, ex, rep in self.results:
            if rep.accepted:
                self.assertFalse(deid.contains_phi(ex.response), msg=ex.example_id)


if __name__ == "__main__":
    unittest.main()
