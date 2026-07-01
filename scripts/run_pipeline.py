"""End-to-end pipeline run, no API key required.

    synthetic rows
      -> per-provider adapters -> canonical clinical event model
      -> token linkage -> per-patient timelines (with link confidence)
      -> case assembly -> grounded case facts
      -> example generation (templated, grounded by construction)
      -> quality gates -> accepted / rejected, with reasons
      -> report + JSONL artifacts

The run deliberately includes three adversarial candidates alongside the good examples so
the report shows each gate catching its failure mode. The four planted defect classes
(low link confidence, PHI leak, unsupported claim, EHR/claims disagreement) each surface
as a rejection with a reason.

Run:  python -m scripts.run_pipeline
"""

from __future__ import annotations

import json
import os

from pipeline.adapters.registry import build_all_claims, build_all_encounters
from pipeline.casebuild import build_case
from pipeline.linkage import build_timelines
from pipeline.posttrain import generate as gen
from pipeline.quality.gates import Deduplicator, run_gates
from pipeline.quality.score import summarize
from pipeline.synth.generate_synth import generate

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build_candidates():
    """Return (candidates, cases_by_token). candidates is a list of (label, example)."""
    tables = generate()
    encounters = build_all_encounters(tables)
    claims = build_all_claims(tables)
    timelines = build_timelines(encounters, claims, tables["linkage.patient_tokens"])

    cases = {}
    for token, tl in timelines.items():
        case = build_case(tl)
        if case is not None:
            cases[token] = case

    candidates = []
    # The good, grounded-by-construction example for every patient.
    for token in sorted(cases):
        candidates.append(("good", gen.build_summary_example(cases[token])))

    # Adversarial candidates that reproduce the LLM path's real failure modes.
    if "TOKEN_003" in cases:
        candidates.append(("adversarial:phi_leak",
                           gen.make_phi_leak_candidate(cases["TOKEN_003"])))
    if "TOKEN_001" in cases:
        candidates.append(("adversarial:unsupported_claim",
                           gen.make_unsupported_claim_candidate(cases["TOKEN_001"])))
    if "TOKEN_005" in cases:
        candidates.append(("adversarial:disagreement_asserted",
                           gen.make_disagreement_asserted_candidate(cases["TOKEN_005"])))

    return candidates, cases


def run():
    candidates, cases = build_candidates()
    deduper = Deduplicator()

    buckets = {"accept": [], "hold": [], "reject": []}
    print("=" * 78)
    print("CLINICAL REASONING EXAMPLE PIPELINE  -  per-candidate verdicts")
    print("=" * 78)

    for label, ex in candidates:
        facts = cases[ex.grounding.patient_token].facts
        results = run_gates(ex, facts, deduper)
        report = summarize(ex, results)

        print(f"\n[{report.disposition.upper()}] {ex.example_id}")
        print(f"    kind={label}  method={ex.provenance.generation_method}  "
              f"score={report.score}  link_conf={ex.grounding.link_confidence}")
        for r in results:
            mark = "ok " if r.passed else "XX "
            if not r.passed or r.needs_review:
                print(f"      {mark}{r.name}: {r.message}")
        buckets[report.disposition].append((label, ex))

    a, h, x = len(buckets["accept"]), len(buckets["hold"]), len(buckets["reject"])
    print("\n" + "=" * 78)
    print(f"SUMMARY: {a} accepted, {h} held for review, {x} rejected "
          f"out of {len(candidates)} candidates")
    print("\nHeld for clinician review:")
    for label, ex in buckets["hold"]:
        print(f"    - {label:32s} {next(iter(ex.quality.notes), '')}")
    print("Rejected, by reason:")
    for label, ex in buckets["reject"]:
        print(f"    - {label:32s} {next(iter(ex.quality.notes), '')}")
    print("=" * 78)

    _write_artifacts(buckets)
    return buckets


_FILENAME = {"accept": "accepted.jsonl", "hold": "held.jsonl", "reject": "rejected.jsonl"}


def _write_artifacts(buckets):
    os.makedirs(OUT_DIR, exist_ok=True)
    for disposition, rows in buckets.items():
        with open(os.path.join(OUT_DIR, _FILENAME[disposition]), "w") as f:
            for label, ex in rows:
                row = ex.to_dict()
                row["_candidate_kind"] = label
                f.write(json.dumps(row, default=str) + "\n")
    print(f"Wrote accepted.jsonl / held.jsonl / rejected.jsonl to "
          f"{os.path.normpath(OUT_DIR)}/")


if __name__ == "__main__":
    run()
