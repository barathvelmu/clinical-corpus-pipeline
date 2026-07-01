"""The post-training example schema.

There is no standard schema for post-training examples; designing one is part of the work.
The design goal is that every shipped example is auditable: you can take
any single example and trace every clinical claim in it back to a specific source field in
a specific encounter, see how confident the cross-source link was, see how it was
generated, and see which quality gates it passed. That traceability is what lets a
downstream consumer trust the data, and what lets us defend it under HIPAA.

So an example is not just messages. It is messages plus three sidecars:
  - grounding:   what real record this came from, and the explicit fact citations.
  - quality:     which gates ran, their verdicts, an aggregate score, review flag.
  - provenance:  how it was made (templated vs LLM), which model, source case.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CitedFact:
    """One clinical assertion in the response, tied to where it came from. `field` is the
    kind of fact (diagnosis, medication, lab, demographic, claim_procedure,
    claim_diagnosis); `value` is the asserted value the grounding gate must resolve back
    to the source record."""

    field: str
    value: str


@dataclass
class Grounding:
    patient_token: str
    source_encounter_ids: list[str]
    source_providers: list[str]
    link_confidence: float | None
    cited_facts: list[CitedFact] = field(default_factory=list)


@dataclass
class Provenance:
    generation_method: str          # "templated" | "llm"
    model: str | None = None        # None for templated; model id for the LLM path
    created_from: str | None = None  # short tag for the source case / candidate kind


@dataclass
class QualityReport:
    """Populated by the gates. Kept on the example so quality travels with the data.

    `disposition` is the real output: an example is ACCEPTed (auto-shippable), HELD (sound
    but routed to mandatory clinician review before it can ship, e.g. a chart/claims
    disagreement we will not adjudicate automatically), or REJECTed (a blocking gate
    failed). `accepted` is kept as the boolean "auto-shippable" shorthand."""

    gates: dict[str, bool] = field(default_factory=dict)
    score: float = 0.0
    disposition: str = "reject"          # "accept" | "hold" | "reject"
    accepted: bool = False               # disposition == "accept"
    needs_clinician_review: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class TrainingExample:
    example_id: str
    task_type: str
    messages: list[dict]            # [{role, content}, ...]
    grounding: Grounding
    provenance: Provenance
    quality: QualityReport = field(default_factory=QualityReport)

    @property
    def system_prompt(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "system"), "")

    @property
    def user_prompt(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "user"), "")

    @property
    def response(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "assistant"), "")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


VALID_ROLES = {"system", "user", "assistant"}
VALID_TASK_TYPES = {"clinical_reasoning_summary"}   # one task type for now; extend later


def validate_format(ex: TrainingExample) -> list[str]:
    """Structural validation only (not quality). Returns a list of problems; empty means
    well-formed. The format gate wraps this."""
    problems: list[str] = []

    if ex.task_type not in VALID_TASK_TYPES:
        problems.append(f"unknown task_type: {ex.task_type!r}")

    roles = [m.get("role") for m in ex.messages]
    if "user" not in roles or "assistant" not in roles:
        problems.append("messages must include a user turn and an assistant turn")
    for m in ex.messages:
        if m.get("role") not in VALID_ROLES:
            problems.append(f"invalid role: {m.get('role')!r}")
        if not (m.get("content") or "").strip():
            problems.append(f"empty content for role {m.get('role')!r}")

    if not ex.grounding.patient_token:
        problems.append("missing patient_token in grounding")
    if not ex.grounding.source_encounter_ids:
        problems.append("missing source_encounter_ids in grounding")
    if not ex.grounding.cited_facts:
        problems.append("no cited_facts: an ungrounded example is not acceptable")

    return problems
