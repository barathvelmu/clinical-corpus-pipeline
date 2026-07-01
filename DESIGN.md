# Design

The architecture and the reasoning behind it: the one idea that organizes everything, the decisions made on purpose (including the ones made *not* to build), the threat model, and what productionizing this looks like. Paired with [README.md](README.md) (what it does) and [RESULTS.md](RESULTS.md) (the proof).

## The thesis: provenance ties quality and privacy together

One idea organizes the whole design. In a pipeline that turns linked clinical and claims data into training examples for a model, **data quality and patient privacy are tightly linked, and one property carries much of the weight for both: provenance. Every token shipped should trace back to a specific field in a specific source record.**

Provenance does not solve privacy on its own (re-identification risk and memorization still need dedicated controls), but it is the common backbone, and it pulls in three directions at once:

- **Quality.** If every clinical claim in an example traces to the record, the example is grounded; it cannot teach the model a fact that was not there. Grounding is provenance enforced at the level of each clinical assertion.
- **Privacy.** If every example carries where it came from (which encounter, which patient token, which link confidence), it is auditable: for any single shipped row, "why is this safe, and where is it from" is answerable. De-identification and re-identification control are provenance enforced at the level of identity.
- **Anti-memorization.** This one matters specifically because the deliverable is training data, not a static table.

### Why training data changes the privacy problem: it is weights, not a table

HIPAA's Expert Determination certifies re-identification risk on a *released dataset*, historically a table. But training data does not sit in a warehouse; it becomes **model weights**, and a large model can **memorize and regurgitate** a training example close to verbatim. That is a re-identification channel classical statistical disclosure control does not model at all.

Once you see that, the defenses line up under the same invariant:

- **Grounding** strips spurious, ungrounded specifics that a model would otherwise have to memorize rather than learn.
- **De-duplication** (corpus-wide and patient-level via the token) removes the repetition that drives memorization, because a unique string repeated across examples is exactly what a model memorizes.
- **Quasi-identifier generalization** (date shifting, rare-value suppression) reduces the uniqueness that makes a memorized example re-identifying.

So the same machinery that makes the data *correct* also does much of the work for keeping it *private* and *memorization-safe*. Provenance is the thread through all of it. That is why the pipeline is built around grounding-by-verification, why every example carries a full provenance and quality record, and why de-identifying a *linked* corpus is an Expert-Determination problem that must explicitly account for memorization, not a second pass of an identifier scrubber.

## Key decisions, and why

- **A canonical clinical event model with per-provider adapters.** The defining difficulty is many providers with many schemas. Mapping each pair of providers to each other is quadratic and hopeless; mapping each provider once into a shared canonical model is linear. After an adapter runs, the rest of the pipeline is provider-agnostic, so onboarding the next provider is one file plus its tests. This is the single most important structural choice, and it is why two very different providers (A encounter-centric/ICD, B visit-centric/SNOMED) collapse into identical downstream code.

- **Grounding as the core quality property, enforced twice.** Every clinical assertion must resolve to a real field in the source record. Enforced by construction (the generator only writes what it can cite) and by verification (the gate re-checks every citation and scans for uncited codes and entities). Verification is the load-bearing half, because the production generator is a frontier model whose fluent output cannot be trusted on its face.

- **Clinical facts and billing facts kept in separate buckets.** Claims are billing-driven and do not always match the chart. Folding them together would erase exactly the signal we need. Keeping them apart is what makes the EHR/claims disagreement check possible at all, and it encodes the domain truth that a billed code is not a diagnosis.

- **Link confidence parsed, carried, and thresholded.** The token match is ~95%, not certain, and arrives as free text (`">95%"`, `"0.85"`). A wrong link is a wrong patient, which is both a quality and a privacy failure, so confidence is parsed defensively (unparseable becomes untrusted, never a silent 1.0) and fused examples below a threshold are refused. The threshold is an explicit, configurable knob.

- **The gates are identical for the templated and the LLM path.** The generator is not trusted, whichever it is. Making the gates path-agnostic is what lets the offline templated path be a faithful stand-in for the production LLM path: the thing being validated (output against record) is the same.

- **Provenance travels with every example.** Each example carries its source encounters, token, link confidence, per-fact citations, generation method, and gate verdicts. In a HIPAA context, "show me why this example is safe and correct" must be answerable for any single row.

- **The PHI detector over-flags on purpose.** A false positive costs a regeneration; a false negative costs a breach. The asymmetry is enormous, so the detector errs toward catching too much.

- **Standard library only, synthetic data, deterministic core.** The whole pipeline runs on a fresh clone with no install and no API key, and produces the same output every run, so the tests and the numbers are reproducible and anyone can actually run it.

## Hardened after adversarial review

After the first green build, several structured adversarial passes were run against the gates (the last with six concurrent attackers: code-correctness, gate-bypass, privacy/re-identification, clinical-correctness, data-robustness). Each finding was verified by running code, then fixed and pinned with a regression test (`tests/test_hardening.py`), or (when fundamentally beyond an automated offline gate) mapped explicitly in the threat model below. What came out of it:

- **Grounding cannot rely on citations alone.** A citation-only check is defeated by simply not citing a hallucination. The gate now also scans the response for recognized clinical entities and requires each to resolve to the record, cited or not.
- **Recognize drugs by morphology, not a list.** A blocklist of drug names is defeated by naming a drug it omits. The scan recognizes drugs by USAN stem (`-glutide`, `-gliflozin`, `-mab`, ...), so `semaglutide` / `canagliflozin` / `dabigatran` are caught without ever being listed. Bounded by a small exclusion set to stop "peptide" / "histidine" false positives; the production form is clinical NER.
- **Grounding is negation-aware.** A ruled-out entity ("X was ruled out") is a differential being excluded, not a claim, so it need not resolve; a disguised hallucination still must.
- **Match on words, not substrings.** Bidirectional substring matching resolved a single letter against a drug name and the wrong sex against a patient. Matching is now word-aware.
- **A disagreement is never auto-adjudicated.** An earlier version tried to judge, with a phrasing heuristic, whether a disagreement was "surfaced honestly"; that was gameable by vocabulary. The honest design is a three-state disposition: any chart/claims disagreement is **HELD** for mandatory clinician review.
- **A completeness gate**, because every other gate checks positive assertions: a grounded summary that omits a flagged-abnormal value is held for review.
- **An identity gate for token collisions.** Confidence describes the match the table claims, not whether the table is right; two patients fused under one token are rejected.
- **Adapters drop bad rows, they do not crash.** One malformed row must not take down a large batch.

### A real false positive, found and fixed

The disagreement check originally fired on a case comparing a SNOMED chart code (42343007, CHF) against an ICD claim code (I50.9, heart failure) for the *same condition*, by naively parsing the SNOMED code as ICD. It now compares categories only when both sides are genuinely ICD-coded; cross-system comparison waits for a crosswalk rather than producing a wrong answer. `tests/test_quality_gates.py::test_no_false_disagreement_on_snomed_vs_icd` pins it. This is exactly the class of silent, domain-specific error that matters most here, and finding it is the difference between a check that looks right and one that is right.

## Threat model and guarantees

What the pipeline guarantees automatically, what it routes to a human, and what it does not attempt. This map is deliberate: a training-data pipeline that knows exactly where its automated guarantees end is more trustworthy than one that implies it catches everything. The boundary below was drawn the hard way, by running repeated adversarial passes and recording what the gates do and do not catch.

Three dispositions: **ACCEPT** (auto-shippable), **HOLD** (sound but routed to mandatory clinician / LLM-judge review before shipping), **REJECT** (a blocking gate failed).

### What the automated gates guarantee (ACCEPT-gating)

| Risk | Gate | Guarantee |
|---|---|---|
| Hallucinated clinical fact (cited) | grounding | every cited fact must resolve to the record |
| Hallucinated fact stated uncited in prose | grounding (lexicon + drug-stem morphology) | recognized medications/labs in the response must resolve, cited or not; morphology catches unseen drugs by USAN stem |
| Hallucinated ICD code | grounding | any ICD-shaped code in the response must be in a record diagnosis category |
| Claim asserted beyond the provided context | answerability | every cited fact must appear in the case context shown to the model |
| Untrusted cross-source link | link_confidence | refuse to fuse below the 0.95 threshold; an unparseable confidence is treated as untrusted |
| Wrong-patient token collision | identity | reject when one source's records under a token map to >1 patient, or sex is contradictory |
| Identifier shapes in output (cued names, dates, SSN, MRN, phone, email, age 90+) | phi | reject on detected Safe Harbor identifier shapes; over-flags by design |
| Near-duplicate examples | dedup | reject lexical near-duplicates (corpus-wide MinHash/LSH in production) |
| Malformed structure | format | reject; an example with no cited facts is not acceptable |

Measured on a 300-patient synthetic corpus: of the accepted set, **100% grounding faithfulness** and **0% PHI-leak rate** (see [RESULTS.md](RESULTS.md)).

### What is routed to HOLD (human review, never auto-shipped)

| Situation | Why a human, not a gate |
|---|---|
| EHR/claims diagnosis disagreement | claims are billing-driven; which is "right" is a clinical judgement, so we detect and route, we do not adjudicate |
| Chart coded in SNOMED/free text while the claim is ICD | no code-level comparison without a crosswalk; route rather than silently pass |
| A flagged-abnormal lab not addressed in the summary | omission is a recall question; a falsely-reassuring-by-omission summary is held |

### What is deliberately NOT attempted automatically (the human / contracted seam)

These are real, and pretending a regex catches them would be dishonest. Each is named, with where it actually belongs.

| Risk | Why not automated here | Where it belongs |
|---|---|---|
| **Memorization / verbatim regurgitation** of a unique note by the trained model | not a property of a single example | corpus-wide dedup + an extraction red-team on the trained model |
| **Quasi-identifier re-identification** (age + rare dx + dates + geography) | needs statistical disclosure-risk analysis over the whole linked set | a contracted privacy service's Expert Determination; date shifting and rare-value suppression |
| Full Safe Harbor identifier coverage (addresses, ZIP, facility/employer names, foreign or abbreviated dates, single surnames, vehicle/device/account ids, URLs/handles) | the detector is a shape-based tripwire on our own output, not a clinical NER de-id model | a contracted de-id service's NLP; `deid.py` is defense-in-depth, not the system of record |
| **Clinical safety of the plan** (a grounded summary recommending a harmful action) | grounding checks entities, not the safety of recommendations | LLM-judge + clinician review with a safety screen |
| **Wrong drug-to-indication / value-to-test relationships** (each entity grounds, the relationship is fabricated) | grounding is entity-level, not relation-level | a relation/NLI judge or clinician review |
| **Token-to-patient correctness upstream** | we catch the collision *footprint*, but cannot verify the token table is right | upstream linkage QA + the identity gate as a backstop |
| **Within-ICD-category clinical contradiction** ("without complications" vs "with complications") | same category, so the disagreement check does not fire | clinician / LLM-judge review |

## Two products, two definitions of quality

A pre-training corpus and a post-training example set are different products with different quality bars, and conflating them is an easy mistake.

- **Pre-training corpus: quality is "clean, safe, representative, at scale."** The model learns the distribution, so the questions are about the whole corpus, not each document: fully de-identified, de-duplicated, format-consistent, broadly representative (specialty/demographic/payer mix not skewed by which providers arrived first), and free of obvious garbage. Verification is mostly automated and statistical, plus sampling. Cheap per document.
- **Post-training examples: quality is "each example is correct and grounded."** The model learns to imitate, so a single wrong example teaches a wrong behavior. The unit of quality is the individual example, and it is expensive: every clinical claim grounded, no PHI, answerable, clinically plausible and consistent (surfacing disagreement rather than resolving it), not a near-duplicate, well-formed. The gates in this repo encode exactly this. The automated gates do the deterministic, high-volume work and route the genuinely judgemental part to a human, so the expensive clinician/LLM-judge review is spent on pre-filtered, already-grounded examples rather than raw model output.

The single most important property across both is **traceability**: every shipped item carries where it came from and why it passed, so quality is auditable rather than asserted.

## Notes for productionizing

The parts that are argued rather than built, and where each belongs.

- **De-duplication at scale.** Clinical notes are full of copy-forward text, and the same patient appears across providers and in claims. Per-batch string matching is not enough. The production design is corpus-wide near-duplicate detection (MinHash / LSH with a persisted signature store across rolling deliveries) plus **patient-level deduplication via the token**, since the same token across providers is the same person and the join is already computed. The `Deduplicator` here is the lexical, in-run version of this idea.
- **Evals.** Two layers: *intrinsic* metrics on the data (grounding/faithfulness rate, dedup rate, PHI-leak rate, clinician agreement on a stratified sample; the first three are computed here at volume) and *extrinsic* metrics on the model (fine-tune on the examples vs. a matched control arm and measure movement on a held-out clinical-reasoning benchmark, with multiple seeds to bound variance and a stated minimum detectable effect). **Contamination control matters most:** the eval set and the training corpus must be disjoint at the patient (token) and document level, which is the same token-keyed join as patient-level dedup.
- **Claims are not clinical ground truth.** Beyond "billing-driven": a claim can be denied, reversed, or adjusted; submitted/adjudicated/paid amounts differ and lag; service date is not the billing date (so claim-to-encounter alignment is a service-date window, not exact equality); and rule-out / upcoding means a diagnosis may be billed to justify a test for a condition the patient does not have. Claims stay in a separate bucket precisely so none of this contaminates the clinical ground truth.
- **Sensitive categories.** Substance-use-disorder records (42 CFR Part 2), behavioral/mental-health, HIV, genetic, and reproductive-health data carry protections beyond standard HIPAA. The workable posture at scale is detect-and-exclude, with the strongest signal being program/facility provenance (a source-level flag), backed by structured code ranges and a free-text classifier as a backstop.

## What was deliberately NOT built

Real production concerns. Half-building them would be worse than scoping them out and saying so.

- **The de-identification engine itself** is a contracted external service. `deid.py` is the in-pipeline tripwire on our own generated output and a testable stand-in, not a replacement for the system of record.
- **The LLM generation path wired to a key** is the production path, shown behind a clean seam with its constraining prompt; the runnable, tested core is deterministic so the whole thing works offline. The gates, which make either path safe, are fully built and tested.
- **Real terminology normalization** (LOINC / RxNorm / ICD↔SNOMED crosswalks). The canonical model has the slots for normalized codes; populating them is a reuse-not-build job against standard vocabularies. A SNOMED↔ICD crosswalk specifically would upgrade the disagreement check from "ICD-only" to cross-system.
- **A second and third example task type.** The schema supports more; one (evidence-grounded clinical reasoning summary) is built end to end and well, rather than three shallowly. Extension reuses the grounding and gate machinery.
- **Scale machinery** (Spark/Snowflake execution, partitioning, throughput). The logic is pure-Python and per-record, so it ports to a distributed engine cleanly; the prototype proves correctness, not throughput.
- **The clinician-review and LLM-judge tooling.** The plausibility gate routes to a `needs_clinician_review` flag; the human-in-the-loop UI and the held-out eval harness are named as reuse, not built here.

## The honesty seam in one line

The runnable, tested core is deterministic and offline; the LLM generator, the clinician and LLM-judge review, and a contracted de-identification service are real production steps behind clean interfaces. This repo builds the gates that gate the humans and the machines, and says plainly where each begins.
