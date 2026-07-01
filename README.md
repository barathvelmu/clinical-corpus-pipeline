# clinical-corpus-pipeline

**Turning messy multi-provider clinical + claims data into trustworthy training examples for an LLM, where every clinical claim in the output traces back to a specific field in a specific source record.**

The hard part of building training data from healthcare records is not moving bytes; it is trust. Data pulled from dozens of providers, each with its own schema, and joined to billing claims can teach a model two kinds of damage: it can leak a patient's identity (a privacy incident), or it can teach the model something clinically false (a poisoned model). This pipeline is built around a single invariant that attacks both at once: **provenance**. Every generated example carries the exact source field behind every clinical assertion, and a set of quality gates refuses to ship anything that cannot prove it.

```
Python 3.10+   ·   standard library only   ·   no install, no API key, no data download   ·   54 tests, sub-second
```

## Quickstart

No dependencies. If you have Python 3.10+, you can run the whole thing in under a minute.

```bash
git clone https://github.com/barathvelmu/clinical-corpus-pipeline.git
cd clinical-corpus-pipeline

python -m unittest discover -s tests     # full suite: 54 tests, runs in well under a second
python -m scripts.run_pipeline           # 8-example demo -> 3 accept / 2 hold / 3 reject, each with its reason
python -m scripts.eval_corpus 300        # scaled run over 300 synthetic patients -> aggregate quality scores
```

What you should see:

- the test suite reports `Ran 54 tests ... OK`;
- the demo prints one example per outcome: a clean accept, a chart/claims disagreement held for a clinician, a hallucination rejected, and a low-confidence link rejected;
- the scaled run reports **100% grounding faithfulness** and a **0% PHI-leak rate** on the accepted set.

On the scaled run, most rejections are `dedup`: the synthetic generator reuses a small set of clinical archetypes, so many templated summaries are near-identical and the dedup gate correctly collapses them. The number to read is not the accept rate but the two quality guarantees above, which hold at volume. Everything is synthetic, authored to match real provider schemas, with zero real PHI.

## The problem

Take encounters, labs, notes, and claims from many providers that each store things differently, and turn them into clinical-reasoning examples that are *good enough to train on*. "Good enough" carries two unforgiving constraints:

1. **Privacy.** The data cannot leak a patient's identity. Unlinked, de-identified data is one thing; the moment you *link* a patient's clinical record to their claims across a timeline, the combination of quasi-identifiers can re-identify them even when no single field is an identifier.
2. **Correctness.** The data cannot teach the model something clinically wrong. A language model writing a clinical summary will, left alone, produce fluent and occasionally invented facts. "Started on insulin glargine" reads perfectly even when the chart says metformin.

Get the first wrong and it is a HIPAA incident. Get the second wrong and you have actively made the trained model worse at the one thing the data was supposed to improve.

## The one idea: grounding

The central design decision is that **a generated example may only assert a clinical fact if that fact traces back to a specific field in the source record.** Everything else is machinery in service of that.

So the pipeline does not trust the generator. Every example carries an explicit list of cited facts, and a gate verifies each one resolves against the record before the example ships. Grounding is enforced two ways:

- **By construction:** the deterministic generator only writes what it can cite.
- **By verification:** the gate re-checks every declared citation against the record, scans for any diagnosis code that snuck in uncited, and scans for any recognized clinical entity (a medication or lab) asserted in prose that does not resolve. That last layer is what catches a free-text "started on insulin glargine" with no citation attached.

Verification is the load-bearing half, because in production the generator is a frontier model, not a template, and the gate does not care which one produced the text. The same idea generalizes: quality (grounding), privacy (every example is auditable back to its source and link confidence), and anti-memorization (deduplication removes the repetition models memorize) are all *provenance* enforced at different levels. That thesis is developed in [DESIGN.md](DESIGN.md).

## How it works, end to end

A straight line. Every stage is boring on purpose except the last two.

```
synthetic source rows      (match real provider EHR + claims schemas + the token table)
  -> per-provider adapters   normalize each provider's dialect into one canonical model
  -> token linkage           fuse records for the same patient across sources, with confidence
  -> case assembly           build a case + extract the verifiable "case facts"
  -> example generation      encounter + claims -> a grounded clinical-reasoning summary
  -> quality gates           format, link-confidence, identity, PHI, grounding,
                             answerability, plausibility, completeness, dedup
  -> accept / hold / reject  every example scored, provenanced, and auditable
```

- **Adapters and the canonical model** are the answer to "N providers, N schemas." Rather than teach the pipeline every dialect, each provider gets one small adapter that maps its raw rows into a single canonical clinical event model. Provider A is encounter-centric with inline notes and ICD-10 codes; Provider B is visit-centric with notes in a separate table, SNOMED codes, and split admit/discharge dates. After their adapters run, downstream code cannot tell them apart. Onboarding the next provider is one new file plus its tests, not a rewrite.

- **Linkage** is the privacy-critical join, not a routine one. The same patient yields the same token in every source. Two things make it dangerous: linking de-identified datasets can re-create identifiability, and the token match is only ~95% confident, not certain. A wrong link is a wrong patient, which is at once a quality bug and a privacy incident. So the pipeline parses the confidence (it arrives as free text like `">95%"` or `"0.85"`), carries it onto every record, and refuses to build a fused example on a link it does not trust.

- **Case assembly** produces the `CaseFacts` object: the de-duplicated set of atomic, checkable facts. Clinical facts (from the EHR) and billing facts (from claims) are kept in separate buckets on purpose, because **claims are billing-driven and are not clinical ground truth.** That separation is what makes the disagreement check possible.

- **Generation and the gates** are the deep piece described above.

## See it run

```bash
python -m scripts.run_pipeline
```

The run generates eight candidate examples (five good, three deliberately adversarial) and puts each through the gates, producing three dispositions: ACCEPT (auto-shippable), HOLD (routed to mandatory clinician review), REJECT (a blocking gate failed). The result is 3 accepted, 2 held, 3 rejected, showing each failure mode that actually happens with this data:

```
[REJECT] TOKEN_004 ... link_confidence: link confidence 0.85 < 0.95; refusing to ship
                       an example built on an untrusted cross-source link
[REJECT] TOKEN_003 ... phi: PHI detected in assistant turn (DATE, MRN, NAME, PHONE)
[REJECT] TOKEN_001 ... grounding: unsupported claim: cited medication='insulin glargine'
                       does not resolve to the source record (hallucination)
[HOLD]   TOKEN_005 ... clinical_plausibility: EHR/claims disagreement detected; routed to
                       mandatory clinician review rather than adjudicated automatically
```

That last one is the point of the three-state design: a chart/claims disagreement is an inherently clinical judgement, so the pipeline detects and routes it rather than auto-deciding. Here is a trimmed accepted example, so the shape of the product is concrete:

```jsonc
{
  "task_type": "clinical_reasoning_summary",
  "messages": [
    {"role": "system",    "content": "You are a clinical reasoning assistant ..."},
    {"role": "user",      "content": "Patient: 58 female. Setting: ambulatory visit, Endocrinology.\nObservations: Hemoglobin A1c 9.1 % (H); Glucose, fasting 187 mg/dL (H)\nMedications: Metformin 1000mg twice daily\n..."},
    {"role": "assistant", "content": "This is a 58-year-old female evaluated in Endocrinology with a documented diagnosis of Type 2 diabetes mellitus without complications. Key objective findings include Hemoglobin A1c 9.1 % (H), Glucose, fasting 187 mg/dL (H). Current management includes Metformin 1000mg twice daily. The associated claim (procedure 99214) is consistent with the documented encounter."}
  ],
  "grounding": {
    "patient_token": "TOKEN_001",
    "source_encounter_ids": ["A-ENC-001"],
    "source_providers": ["claims_provider_x", "ehr_provider_a"],
    "link_confidence": 0.99,
    "cited_facts": [
      {"field": "diagnosis",  "value": "E11.9"},
      {"field": "lab",        "value": "Hemoglobin A1c"},
      {"field": "medication", "value": "Metformin"},
      {"field": "claim_procedure", "value": "99214"}
    ]
  },
  "quality": {"gates": {"grounding": true, "phi": true, "...": true}, "score": 1.0,
              "disposition": "accept", "accepted": true, "needs_clinician_review": false},
  "provenance": {"generation_method": "templated", "model": null}
}
```

The example is fully auditable on its own: which encounter it came from, how confident the link was, and the exact field behind every clinical claim. That traceability is the product, as much as the text is.

## Does it actually work? (measured, not asserted)

Full evidence, with reproduce commands, is in [RESULTS.md](RESULTS.md). The short version:

- **54 unit tests pass**, covering the adapters, linkage and confidence parsing, every individual gate against its defect, the whole pipeline end to end (including the LLM path through a stub), and regression tests for the bypasses found in adversarial review.
- **Measured at volume.** A 300-patient synthetic corpus runs end to end with 100% grounding faithfulness and a 0% PHI-leak rate on the accepted set; the planted low-confidence-link rate (~8%) is recovered by the gates.
- **Every planted defect class is caught:** low link confidence (reject), PHI leak (reject), hallucinated fact (reject), and EHR/claims disagreement (held for clinician review).
- **The gates were stress-tested.** Several rounds of adversarial review aimed to slip a bad example past the gates; the holes that surfaced became the fixes and the 19 regression tests in `tests/test_hardening.py`. Examples: uncited hallucinations are caught by a drug-stem morphology heuristic (so `semaglutide` is flagged without being on any list); omitted critical labs are held by a completeness gate; ruled-out drugs are not false-rejected (negation-aware); a token collision is rejected by an identity gate.

## The honesty seam (what is real, what is staged)

This is a prototype, and the line is drawn on purpose rather than blurred.

- **Generation.** The runnable, tested path is deterministic and templated, so the whole pipeline works offline. The production path is a frontier model behind a clean interface (`pipeline/posttrain/generate.py:llm_generate`) with the constraining prompt shown. The gates are identical for both paths, which is the entire point: verify the output, do not trust the generator.
- **Clinical validity.** The plausibility gate enforces the cheap, automatable invariants (consistency, surfaced disagreement, sane values) and routes genuine clinical judgement to a `needs_clinician_review` flag. It is not a correctness oracle; real validity needs a clinician or an LLM-judge in the loop, and the gate builds the thing that *routes* to them.
- **The uncited-claim scan is lexicon-bounded.** It recognizes common medications and labs (`pipeline/quality/lexicon.py`), which covers the failure modes that dominate this data. In production this scan is a clinical NER / NLI model behind the same gate; the invariant ("a recognized clinical entity must be grounded") is the durable part.
- **De-identification.** The PHI detector catches common HIPAA Safe Harbor identifier shapes and deliberately over-flags (a false positive costs a regeneration; a false negative costs a breach). Heavy de-id belongs to a contracted privacy service; this module is the in-pipeline tripwire on our own generated output.

The exact boundary of what the gates guarantee, what is routed to a human, and what is left to the contracted privacy service is mapped in [DESIGN.md](DESIGN.md#threat-model-and-guarantees).

## Project layout

```
pipeline/
  canonical.py          the canonical clinical event model + normalization helpers
  adapters/
    provider_a.py       EHR Provider A (encounter-centric, inline notes, ICD-10) -> canonical
    provider_b.py       EHR Provider B (visit-centric, separate notes, SNOMED)   -> canonical
    claims_x.py         Claims Provider X (delimited multi-dx) -> canonical
    registry.py         source_id -> adapter; the onboarding seam
  linkage.py            token-based cross-source patient assembly + confidence
  casebuild.py          case assembly + the grounded CaseFacts object
  deid.py               PHI detection / redaction (the PHI gate's engine)
  posttrain/
    schema.py           the training-example schema + structural validation
    generate.py         encounter+claims -> grounded example (templated path + LLM seam)
  quality/
    gates.py            the nine quality gates (incl. identity + completeness)
    lexicon.py          clinical lexicon + drug-stem morphology for the uncited-claim scan
    score.py            aggregate verdicts -> accept / hold / reject
  synth/
    generate_synth.py   synthetic rows matching the schemas + planted defects, and a
                        scaled corpus generator for volume evaluation
scripts/
  run_pipeline.py       end-to-end run + accept/hold/reject report
  eval_corpus.py        scaled evaluation: aggregate metrics over a synthetic corpus
tests/                  54 unit tests, standard library only
DESIGN.md               architecture, the provenance thesis, key tradeoffs, threat model
RESULTS.md              measured evidence, with the exact commands to reproduce it
```

## Further reading

- **[DESIGN.md](DESIGN.md):** the architecture and the reasoning; the provenance thesis, why training data changes the privacy problem, the key design decisions and what was deliberately left out, and the full threat model (what the gates guarantee vs. route to a human vs. leave to a contracted service).
- **[RESULTS.md](RESULTS.md):** every measured number, with the commands to reproduce it from a clean clone.
