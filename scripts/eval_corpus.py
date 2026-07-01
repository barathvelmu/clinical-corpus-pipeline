"""Scaled end-to-end evaluation: run the whole pipeline over a large synthetic corpus and
report aggregate intrinsic metrics.

This is the intrinsic quality metrics made real. The 8-candidate demo in
run_pipeline.py shows the gates catching each planted defect; this shows how the pipeline
behaves at volume: the disposition mix (accept / hold / reject), which gate is responsible
for each non-accept, and the two numbers that matter most for a training-data vendor:

  - grounding faithfulness:  of accepted examples, the fraction whose every cited fact
                             resolves to the source record (target: 100%).
  - PHI-leak rate:           of accepted examples, the fraction with any detected PHI in
                             the output (target: 0%).

Run:  python -m scripts.eval_corpus [n_patients]
"""

from __future__ import annotations

import sys
from collections import Counter

from pipeline import deid
from pipeline.adapters.registry import build_all_claims, build_all_encounters
from pipeline.casebuild import build_case
from pipeline.linkage import build_timelines
from pipeline.posttrain import generate as gen
from pipeline.quality.gates import Deduplicator, run_gates
from pipeline.quality.score import summarize
from pipeline.synth.generate_synth import generate_corpus


def evaluate(n_patients: int = 300, seed: int = 7) -> dict:
    tables = generate_corpus(n_patients, seed=seed)
    encounters = build_all_encounters(tables)
    claims = build_all_claims(tables)
    timelines = build_timelines(encounters, claims, tables["linkage.patient_tokens"])

    deduper = Deduplicator()
    dispositions = Counter()
    gate_fail = Counter()        # blocking failures, by gate
    gate_review = Counter()      # review routes (holds), by gate
    accepted, accepted_phi_leaks, accepted_ungrounded = 0, 0, 0
    n_cases = 0

    for tl in timelines.values():
        case = build_case(tl)
        if case is None:
            continue
        n_cases += 1
        ex = gen.build_summary_example(case)
        results = run_gates(ex, case.facts, deduper)
        report = summarize(ex, results)
        dispositions[report.disposition] += 1
        for r in results:
            if not r.passed and r.blocking:
                gate_fail[r.name] += 1
            if r.needs_review:
                gate_review[r.name] += 1
        if report.accepted:
            accepted += 1
            if deid.contains_phi(ex.response):
                accepted_phi_leaks += 1
            if any(not case.facts.contains(cf.field, cf.value)
                   for cf in ex.grounding.cited_facts):
                accepted_ungrounded += 1

    return {
        "n_patients": n_patients, "n_cases": n_cases,
        "dispositions": dict(dispositions),
        "gate_fail": dict(gate_fail), "gate_review": dict(gate_review),
        "accepted": accepted,
        "grounding_faithfulness": 1.0 if accepted == 0 else
            round((accepted - accepted_ungrounded) / accepted, 4),
        "phi_leak_rate_accepted": 0.0 if accepted == 0 else
            round(accepted_phi_leaks / accepted, 4),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    m = evaluate(n)
    total = m["n_cases"]
    print("=" * 70)
    print(f"CORPUS EVALUATION  -  {m['n_patients']} synthetic patients, {total} cases")
    print("=" * 70)
    print("\nDisposition mix:")
    for d in ("accept", "hold", "reject"):
        c = m["dispositions"].get(d, 0)
        print(f"    {d:8s} {c:5d}  ({0 if not total else round(100*c/total,1)}%)")
    print("\nBlocking rejections by gate:")
    for g, c in sorted(m["gate_fail"].items(), key=lambda kv: -kv[1]):
        print(f"    {g:20s} {c}")
    print("\nRouted to clinician review by gate:")
    for g, c in sorted(m["gate_review"].items(), key=lambda kv: -kv[1]):
        print(f"    {g:20s} {c}")
    print("\nQuality of the ACCEPTED (auto-shippable) set:")
    print(f"    grounding faithfulness:  {m['grounding_faithfulness']*100:.1f}%  (target 100%)")
    print(f"    PHI-leak rate:           {m['phi_leak_rate_accepted']*100:.1f}%  (target 0%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
