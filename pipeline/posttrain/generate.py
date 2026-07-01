"""Turn a clinical case into a post-training example: an evidence-grounded clinical
reasoning summary built from the encounter and its claims.

There are two generation paths, and being honest about the split matters:

  - Templated (this module's tested core). Deterministic, grounded by construction: the
    response is assembled only from facts that exist in CaseFacts, and every clinical
    assertion is emitted with a citation. It runs with no API key, which is why the whole
    pipeline is testable end to end. It is rigid prose, but it is correct prose.

  - LLM (the production path, behind a seam). A frontier model writes richer reasoning
    from the same de-identified case, constrained to the case facts, and a citation step
    extracts what it asserted. We do NOT fake this: `llm_generate` shows the prompt and
    the seam, and the quality gates downstream are exactly what would catch an LLM that
    drifts off the record. The gates are identical for both paths, which is the point.

The de-identified case presentation (the user turn) redacts the raw note before it ever
enters an example. The structured facts the response is built from never contained PHI.

Also here: the planted-bad candidate builders. They construct the realistic failure modes
of the LLM path (a PHI leak, a hallucinated medication, a billing code asserted as the
diagnosis) so we can show the gates catching each. They are clearly labelled as adversarial.
"""

from __future__ import annotations

from typing import Protocol

from pipeline import deid
from pipeline.casebuild import ClinicalCase
from pipeline.posttrain.schema import (
    CitedFact, Grounding, Provenance, TrainingExample,
)
from pipeline.quality.lexicon import clinical_terms_in, term_kind

TASK_TYPE = "clinical_reasoning_summary"

SYSTEM_PROMPT = (
    "You are a clinical reasoning assistant. Given a de-identified encounter and its "
    "associated claims, write a concise, evidence-grounded summary of the case: the "
    "presentation, the key objective findings, the working assessment, and the current "
    "management. Use only information present in the record. When the billing claim "
    "diverges from the clinical documentation, say so explicitly rather than resolving it."
)


# --------------------------------------------------------------------------- helpers

def _fmt_obs(o: dict) -> str:
    parts = [o["name"]]
    if o.get("value"):
        parts.append(str(o["value"]))
    if o.get("unit"):
        parts[-1] = f"{parts[-1]} {o['unit']}"
    s = " ".join(parts)
    if o.get("flag") in ("H", "L", "A"):
        s += f" ({o['flag']})"
    return s


def _fmt_med(m: dict) -> str:
    bits = [m["drug"]]
    if m.get("dose"):
        bits.append(m["dose"])
    if m.get("frequency"):
        bits.append(m["frequency"])
    return " ".join(bits)


def _present_case(case: ClinicalCase) -> str:
    """The de-identified case the model reasons over. The note is redacted here."""
    f = case.facts
    lines = []
    demo = f"{f.age or 'unknown-age'} {f.sex or 'unknown-sex'}"
    lines.append(f"Patient: {demo}. Setting: {f.encounter_type or 'unspecified'} visit, "
                 f"{f.specialty or 'unspecified specialty'}.")
    if f.notes:
        lines.append("Note (de-identified): " + deid.redact(f.notes[0]))
    if f.observations:
        lines.append("Observations: " + "; ".join(_fmt_obs(o) for o in f.observations))
    if f.medications:
        lines.append("Medications: " + "; ".join(_fmt_med(m) for m in f.medications))
    if f.diagnoses:
        # Render the coding system with the code ("(SNOMED 235595009)"), both so it reads
        # as a clinical code and so the PHI tripwire does not mistake a 9-digit SNOMED id
        # for a patient identifier.
        def _dx(d):
            if not d.get("code"):
                return d["description"]
            sys = (d.get("system") or "").strip()
            return f"{d['description']} ({sys + ' ' if sys else ''}{d['code']})"
        lines.append("Documented diagnoses: " + "; ".join(_dx(d) for d in f.diagnoses))
    if f.claim_procedures or f.claim_diagnoses:
        lines.append("Associated claim: procedures " + (", ".join(f.claim_procedures) or "none")
                     + "; billed diagnoses " + (", ".join(f.claim_diagnoses) or "none"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- templated path

def build_summary_example(case: ClinicalCase) -> TrainingExample:
    """The grounded-by-construction example. Every sentence in the response is backed by a
    CitedFact that resolves into CaseFacts."""
    f = case.facts
    cited: list[CitedFact] = []
    sentences: list[str] = []

    # Presentation line, grounded in demographics + primary diagnosis.
    primary = f.diagnoses[0] if f.diagnoses else None
    demo = f"{f.age}-year-old {f.sex}" if (f.age and f.sex) else "patient"
    if primary:
        dx_label = primary.get("description") or primary.get("code")
        sentences.append(
            f"This is a {demo} evaluated in {f.specialty or 'clinic'} with a documented "
            f"diagnosis of {dx_label}.")
        cited.append(CitedFact("diagnosis", primary.get("code") or dx_label))
    else:
        sentences.append(f"This is a {demo} evaluated in {f.specialty or 'clinic'}.")
    if f.age:
        cited.append(CitedFact("demographic", str(f.age)))
    if f.sex:
        cited.append(CitedFact("demographic", f.sex))

    # Objective findings, abnormal first.
    abnormal = [o for o in f.observations if o.get("flag") in ("H", "L", "A")]
    if abnormal:
        sentences.append("Key objective findings include "
                         + ", ".join(_fmt_obs(o) for o in abnormal) + ".")
        for o in abnormal:
            cited.append(CitedFact("lab", o["name"]))

    # Current management.
    if f.medications:
        sentences.append("Current management includes "
                         + ", ".join(_fmt_med(m) for m in f.medications) + ".")
        for m in f.medications:
            cited.append(CitedFact("medication", m["drug"]))

    # Billing alignment: surface disagreement explicitly, never resolve it silently.
    if f.has_ehr_claims_disagreement():
        claim_dx = ", ".join(sorted(f.claim_only_dx_categories())) or ", ".join(f.claim_diagnoses)
        sentences.append(
            f"Note a discrepancy: the associated claim is billed under {claim_dx}, which "
            f"does not match the documented clinical diagnosis; this is flagged for "
            f"clinician review rather than reconciled here.")
        for c in f.claim_diagnoses:
            cited.append(CitedFact("claim_diagnosis", c))
    elif f.claim_procedures:
        sentences.append(
            f"The associated claim (procedure {', '.join(f.claim_procedures)}) is "
            f"consistent with the documented encounter.")
        for p in f.claim_procedures:
            cited.append(CitedFact("claim_procedure", p))

    response = " ".join(sentences)
    return _assemble(case, response, cited, Provenance(
        generation_method="templated", model=None, created_from="grounded_summary"))


def _assemble(case, response, cited, provenance) -> TrainingExample:
    f = case.facts
    ex_id = f"{f.patient_token}:{f.source_encounter_ids[0]}:{provenance.created_from}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _present_case(case)},
        {"role": "assistant", "content": response},
    ]
    grounding = Grounding(
        patient_token=f.patient_token,
        source_encounter_ids=list(f.source_encounter_ids),
        source_providers=list(f.source_providers),
        link_confidence=f.link_confidence,
        cited_facts=cited,
    )
    return TrainingExample(ex_id, TASK_TYPE, messages, grounding, provenance)


# --------------------------------------------------------------------------- LLM seam

class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def _extract_citations(facts, response: str) -> list[CitedFact]:
    """Citation-extraction over a free-text response: pull the clinical claims it makes and
    tie each to the source record. This is the step that turns prose into auditable,
    grounded structure. It is intentionally simple (recognised entities + demographics +
    diagnoses that resolve); in production it is a clinical NER/NLI model, but the contract
    is identical: only facts that resolve to the record become citations, and anything the
    model asserted that does NOT resolve is left for the grounding gate to catch."""
    cited: list[CitedFact] = []
    for term in sorted(clinical_terms_in(response)):
        kind = term_kind(term)
        if facts.contains(kind, term):
            cited.append(CitedFact(kind, term))
    if facts.age and str(facts.age) in response:
        cited.append(CitedFact("demographic", str(facts.age)))
    if facts.sex and facts.sex.lower() in response.lower():
        cited.append(CitedFact("demographic", facts.sex))
    for d in facts.diagnoses:
        if d.get("code") and d["code"].lower() in response.lower():
            cited.append(CitedFact("diagnosis", d["code"]))
        elif d.get("description") and d["description"].lower() in response.lower():
            cited.append(CitedFact("diagnosis", d["description"]))
    return cited


def llm_generate(case: ClinicalCase, client: LLMClient) -> TrainingExample:
    """Production path. Same de-identified case in, richer reasoning out from a frontier
    model, then the SAME gates as the templated path. The model sees only the de-identified
    case and is instructed to ground every claim; we then extract citations from its output
    and let the gates verify. If it drifts off the record, the grounding gate rejects it,
    which is the whole point: the gates do not care which path wrote the text.

    `client` is any object with `.complete(system, user) -> str`. The offline test suite
    passes a deterministic stub; in production it is a real model client behind the same
    interface, so no pipeline code changes between the two."""
    response = client.complete(SYSTEM_PROMPT, _present_case(case))
    cited = _extract_citations(case.facts, response)
    model = getattr(client, "name", "llm-client")
    return _assemble(case, response, cited,
                     Provenance(generation_method="llm", model=model, created_from="llm_generate"))


# --------------------------------------------------------------------------- adversarial candidates
# These deliberately reproduce the LLM path's realistic failure modes so the gates have
# something real to catch. Each is tagged in provenance as adversarial.

def make_phi_leak_candidate(case: ClinicalCase) -> TrainingExample:
    """An example whose response copies the raw note verbatim, dragging the name/MRN/DOB
    straight into the output. This is the single most dangerous LLM failure mode."""
    f = case.facts
    dx = f.diagnoses[0] if f.diagnoses else {"description": "the documented condition", "code": None}
    raw_note = f.notes[0] if f.notes else ""
    response = (f"This is a {f.age}-year-old {f.sex} with {dx['description']}. "
                f"From the chart: {raw_note}")
    cited = [CitedFact("diagnosis", dx["code"])] if dx.get("code") else []
    return _assemble(case, response, cited, Provenance(
        generation_method="llm", model="adversarial-stub", created_from="phi_leak"))


def make_unsupported_claim_candidate(case: ClinicalCase) -> TrainingExample:
    """An example that asserts a medication not in the record (a hallucination), with a
    citation that cannot be resolved against CaseFacts."""
    f = case.facts
    dx_desc = f.diagnoses[0]["description"] if f.diagnoses else "the documented condition"
    cited = [CitedFact("medication", "insulin glargine")]   # not in the record
    if f.diagnoses and f.diagnoses[0].get("code"):
        cited.insert(0, CitedFact("diagnosis", f.diagnoses[0]["code"]))
    response = (f"This is a {f.age}-year-old {f.sex} with {dx_desc}, "
                f"started on insulin glargine 20 units nightly.")
    return _assemble(case, response, cited, Provenance(
        generation_method="llm", model="adversarial-stub", created_from="unsupported_claim"))


def make_disagreement_asserted_candidate(case: ClinicalCase) -> TrainingExample:
    """An example that adopts the billing diagnosis as the clinical truth, silently
    overriding the chart. The disagreement is never surfaced."""
    f = case.facts
    claim_dx = f.claim_diagnoses[0] if f.claim_diagnoses else "the billed code"
    response = (f"This is a {f.age}-year-old {f.sex}. The diagnosis is {claim_dx} based on "
                f"the claim, and management should target that condition.")
    cited = [CitedFact("claim_diagnosis", claim_dx)]
    return _assemble(case, response, cited, Provenance(
        generation_method="llm", model="adversarial-stub", created_from="disagreement_asserted"))
