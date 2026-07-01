"""Linkage is the privacy-critical join. These tests pin confidence parsing (it arrives as
free-text VARCHAR) and the trust threshold that downstream gates depend on."""

import unittest

from pipeline.adapters.registry import build_all_claims, build_all_encounters
from pipeline.linkage import (
    DEFAULT_CONFIDENCE_THRESHOLD, build_timelines, parse_confidence,
)
from pipeline.synth.generate_synth import generate


class TestConfidenceParsing(unittest.TestCase):
    def test_decimal(self):
        self.assertEqual(parse_confidence("0.99"), 0.99)

    def test_percent(self):
        self.assertEqual(parse_confidence("98%"), 0.98)

    def test_gt_prefix(self):
        self.assertEqual(parse_confidence(">95%"), 0.95)

    def test_bare_integer_percent(self):
        self.assertEqual(parse_confidence("85"), 0.85)

    def test_unparseable_is_none_not_one(self):
        # Critical: garbage must be untrusted, never silently treated as a perfect match.
        self.assertIsNone(parse_confidence("high"))
        self.assertIsNone(parse_confidence(None))


class TestTimelines(unittest.TestCase):
    def setUp(self):
        tables = generate()
        encs = build_all_encounters(tables)
        clms = build_all_claims(tables)
        self.tls = build_timelines(encs, clms, tables["linkage.patient_tokens"])

    def test_records_grouped_by_token(self):
        self.assertEqual(len(self.tls), 5)
        for tl in self.tls.values():
            self.assertGreaterEqual(len(tl.encounters), 1)

    def test_same_patient_fused_across_sources(self):
        tl = self.tls["TOKEN_001"]
        self.assertEqual(len(tl.encounters), 1)
        self.assertEqual(len(tl.claims), 1)

    def test_low_confidence_timeline_is_untrusted(self):
        tl = self.tls["TOKEN_004"]
        self.assertEqual(tl.min_link_confidence, 0.85)
        self.assertFalse(tl.is_trusted(DEFAULT_CONFIDENCE_THRESHOLD))

    def test_high_confidence_timeline_is_trusted(self):
        self.assertTrue(self.tls["TOKEN_001"].is_trusted())


if __name__ == "__main__":
    unittest.main()
