"""Aggregate gate verdicts into the example's QualityReport.

Three dispositions, conservative on purpose:
  - REJECT  if any blocking gate failed.
  - HOLD    if no blocking failure but a gate routed the example to mandatory human review
            (a chart/claims disagreement, or anything we will not adjudicate automatically).
            Held examples are not auto-shipped; they wait for a clinician.
  - ACCEPT  otherwise (auto-shippable).

Separating HOLD from ACCEPT matters: "passed every automated gate" and "safe to ship into
training data without a human looking" are different questions, and conflating them is how
a clinically ambiguous example slips into the corpus. Score stays as a soft signal.
"""

from __future__ import annotations

from pipeline.posttrain.schema import QualityReport, TrainingExample
from pipeline.quality.gates import GateResult


def summarize(ex: TrainingExample, results: list[GateResult]) -> QualityReport:
    gates = {r.name: r.passed for r in results}
    blocking_failures = [r for r in results if r.blocking and not r.passed]
    needs_review = any(r.needs_review for r in results)
    score = sum(1 for r in results if r.passed) / len(results) if results else 0.0

    if blocking_failures:
        disposition = "reject"
    elif needs_review:
        disposition = "hold"
    else:
        disposition = "accept"

    notes = [f"{r.name}: {r.message}" for r in results if not r.passed or r.needs_review]

    report = QualityReport(
        gates=gates,
        score=round(score, 3),
        disposition=disposition,
        accepted=(disposition == "accept"),
        needs_clinician_review=needs_review,
        notes=notes,
    )
    ex.quality = report
    return report
