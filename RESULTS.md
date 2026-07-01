# Results (measured, not asserted)

Every number here is reproducible from a fresh clone with no install and no API key:

```bash
python -m unittest discover -s tests    # the test suite (54 tests)
python -m scripts.run_pipeline          # the 8-candidate demo + accept/hold/reject report
python -m scripts.eval_corpus 300       # the scaled corpus evaluation
```

## Test suite

**54 tests pass in well under a second**, standard library `unittest`. Coverage by area:

| area | tests | what is pinned |
|---|---|---|
| adapters | 12 | Provider A inline-note/labs/meds/dx attachment, type+sex normalization, numeric lab parse; Provider B admit/discharge → start/end, lab/vital split, SNOMED preserved; Claims X delimited-dx split, procedure mapping, null paid-amount |
| linkage | 9 | confidence parsing (`0.99` / `98%` / `>95%` / `85` / garbage→None); token grouping; cross-source fusion; trusted-vs-untrusted threshold |
| quality gates | 8 | a clean example passes all gates at 1.0; grounding catches a hallucinated med; PHI catches leaked identifiers; link-confidence rejects an untrusted link; a disagreement routes to HOLD (not auto-decided); no false disagreement on SNOMED-vs-ICD; answerability; dedup |
| hardening / regression | 19 | every bypass from three adversarial rounds, now closed (see below) |
| end to end | 6 | the 3/2/3 disposition split; every defect class caught; accepted examples fully grounded + PHI-free; the LLM path (via a stub) accepted when grounded and rejected when hallucinated by the *same* gates |

## End-to-end demo run

Eight candidates: five good, three adversarial. Result: **3 accepted, 2 held for review, 3
rejected.**

| candidate | kind | disposition | gate that decided it |
|---|---|---|---|
| TOKEN_001 (diabetes, Provider A) | good | ACCEPT (1.0) | all pass |
| TOKEN_002 (CHF, Provider B, SNOMED) | good | ACCEPT (1.0) | all pass; no false disagreement |
| TOKEN_003 (hypertension, Provider A) | good | ACCEPT (1.0) | all pass |
| TOKEN_005 (asthma, Provider A) | good | **HOLD** | EHR/claims disagreement → clinician review |
| TOKEN_004 (appendicitis, link 0.85) | good | REJECT | link_confidence: 0.85 < 0.95 |
| TOKEN_003 phi_leak | adversarial | REJECT | phi: DATE, MRN, NAME, PHONE in output |
| TOKEN_001 unsupported_claim | adversarial | REJECT | grounding: cited `insulin glargine` unresolved |
| TOKEN_005 disagreement_asserted | adversarial | **HOLD** | EHR/claims disagreement → clinician review |

Four defect classes, three dispositions. The disagreement candidates are **held**, not
auto-judged: a chart/claims conflict is an inherently clinical decision, so the pipeline
detects and routes it to a human rather than pretending to adjudicate it (the bad one is
then rejected by a reviewer; the automated pipeline never ships either). Accepted examples
land in `data/accepted.jsonl`, held in `data/held.jsonl`, rejected (with reasons) in
`data/rejected.jsonl`.

## Scaled corpus evaluation

`scripts/eval_corpus.py` runs the whole pipeline over a larger synthetic corpus (300
patients across both provider shapes, with planted low-confidence links and EHR/claims
disagreements). Headline numbers on the accepted (auto-shippable) set:

- **Grounding faithfulness: 100%**; every accepted example's every cited fact resolves to
  the source record.
- **PHI-leak rate: 0%**; no accepted example carries a detected identifier.
- The planted **low-confidence-link rate (~8%)** is recovered by the link-confidence gate
  (~7% of cases rejected for it).

Most rejections are **dedup**, and honestly so: the generator reuses a small set of
clinical archetypes, so many templated summaries are near-identical and the dedup gate
correctly collapses them. On real data with free-text note variety this rate drops sharply;
the number to read here is not the accept rate but the two quality guarantees above, which
hold at volume.

## What three adversarial rounds found, and what we did about it

After each green build we ran structured adversarial passes (the third with six concurrent
attackers: code-correctness, gate-bypass, privacy/re-identification, clinical-correctness,
and data-robustness). Every finding was verified by running code, then either fixed and
pinned by a regression test, or (when it is fundamentally beyond an automated offline gate)
mapped explicitly in [DESIGN.md](DESIGN.md#threat-model-and-guarantees). Highlights of what became fixes:

- **Uncited free-text hallucination → drug-stem morphology.** A citation-only check missed
  drugs stated in prose; a word-list missed drugs we did not enumerate. The grounding gate
  now recognises drugs by USAN stem (`-glutide`, `-gliflozin`, `-mab`, ...), catching
  `semaglutide` / `canagliflozin` / `dabigatran` without ever listing them.
- **Omitted critical lab → completeness gate (HOLD).** A grounded summary that silently
  drops a flagged-abnormal value is now held for review.
- **Negation false-reject → negation-aware grounding.** "atorvastatin was ruled out" no
  longer reads as a hallucination, while a disguised "started semaglutide, no issues" still
  does.
- **Token collision → identity gate.** Two patients fused under one token at perfect
  confidence is rejected by detecting the collision footprint.
- **Batch resilience.** One malformed row (missing key, non-string field, mixed-delimiter
  code) no longer crashes a whole-provider batch; the bad row is dropped.
- **PHI fixes.** Age-90+ (`92-year-old`, our own phrasing), ordinal dates (`12th April`),
  cued names with a lowercase relationship cue are now caught; SNOMED concept codes and
  bare month+year are no longer false-flagged; `redact()` no longer corrupts overlapping
  spans.

What gives me confidence here is not that the first version looked right, but that it was
attacked repeatedly and those attacks are now part of the test suite.

### A real false positive, found and fixed

The disagreement check originally fired on TOKEN_002, comparing a SNOMED chart code
(42343007, CHF) against an ICD claim code (I50.9, heart failure) for the same condition, by
naively parsing the SNOMED code as ICD. It now compares categories only when both sides are
ICD-coded; cross-system comparison waits for a crosswalk rather than producing a wrong
answer. `tests/test_quality_gates.py::test_no_false_disagreement_on_snomed_vs_icd` pins it.

## Honest limitations of these numbers

- The corpus is synthetic and built from a small archetype set, so the *accept rate* is
  distorted by dedup; the meaningful, robust numbers are the 100% grounding faithfulness and
  0% PHI-leak on the accepted set. Throughput/scale is argued structurally (per-record pure
  logic) in [DESIGN.md](DESIGN.md), not benchmarked.
- Clinical validity is checked as plausibility and consistency, not correctness; real
  validity is the clinician/LLM-judge seam (DESIGN.md), named not benchmarked.
- The PHI detector is a shape-based tripwire that over-flags by design; full Safe Harbor
  coverage (addresses, facility names, foreign dates, etc.) is the contracted de-id
  service's job, and the gaps are enumerated in DESIGN.md rather than hidden.
