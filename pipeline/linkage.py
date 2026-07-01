"""Token-based linkage: assemble one patient's records from across providers.

The same person yields the same `token_value` in every source, so the token table is how
we recognise that an EHR record and a claims record belong to one patient. Two things
make this the privacy-critical step, not a routine join:

  1. Linking de-identified datasets can re-create identifiability. Everything assembled
     here is therefore treated as living inside the PHI-restricted workspace until it has
     been re-de-identified downstream.
  2. The token match has a confidence (~95% typical, not 100%). A wrong link is a wrong
     patient, which is simultaneously a data-quality bug and a privacy incident. So we
     parse the confidence, carry it onto every record, and let downstream gates refuse to
     build a fused example on a link we do not trust.

We do not invent or infer tokens. A record whose (source, patient_key) is absent from the
token table stays unlinked rather than being guessed into a patient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.canonical import Claim, Encounter

# Below this, we treat the cross-source link as untrusted and will not fuse claims into a
# clinical example. Upstream linkage reports confidence "generally >95%"; 0.95 is the
# natural line and is a configurable threshold.
DEFAULT_CONFIDENCE_THRESHOLD = 0.95


def parse_confidence(raw) -> float | None:
    """Confidence arrives as a VARCHAR in many shapes: "0.99", "98%", ">95%", "0.85".
    Return a float in [0, 1], or None when it is genuinely unparseable (which we treat as
    untrusted, never as 1.0)."""
    if raw is None:
        return None
    s = str(raw).strip().lstrip(">").lstrip("<").strip()
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if pct:
        v /= 100.0
    if v > 1.0:          # bare "98" meaning 98%
        v /= 100.0
    if not 0.0 <= v <= 1.0:
        return None
    return v


@dataclass
class TokenMap:
    """(source_id, patient_key) -> (token_value, confidence)."""

    _by_key: dict[tuple[str, str], tuple[str, float | None]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "TokenMap":
        m = cls()
        for r in rows:
            if not (r.get("source_id") and r.get("patient_key") and r.get("token_value")):
                continue   # drop a malformed token row rather than crash linkage
            m._by_key[(r["source_id"], r["patient_key"])] = (
                r["token_value"], parse_confidence(r.get("confidence")))
        return m

    def lookup(self, source_id: str, patient_key: str) -> tuple[str | None, float | None]:
        return self._by_key.get((source_id, patient_key), (None, None))


@dataclass
class PatientTimeline:
    """All records for one patient (one token), across every source."""

    token: str
    encounters: list[Encounter] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    @property
    def _all_confidences(self) -> list:
        return ([e.link_confidence for e in self.encounters]
                + [c.link_confidence for c in self.claims])

    @property
    def min_link_confidence(self) -> float | None:
        """The weakest link. If any linked record has an unparseable confidence (None),
        the timeline's effective confidence is None: we never let a populated value mask a
        missing one, because that is exactly how an untrusted link would slip through."""
        confs = self._all_confidences
        if not confs:
            return None
        if any(c is None for c in confs):
            return None
        return min(confs)

    def is_trusted(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
        """A timeline is trusted for fusion only if every linked record meets the
        threshold. One shaky or unparseable link contaminates the fused view, so we are
        strict: a single None forces untrusted."""
        mc = self.min_link_confidence
        return mc is not None and mc >= threshold

    def has_identity_conflict(self) -> bool:
        """A signature of a wrong/colliding token: the link confidence can be a perfect
        1.0 and still fuse two different people if the token table itself assigned one
        token to two patients. We cannot verify token correctness (that is upstream), but
        we can catch its footprint: two distinct patient keys from the *same* source under
        one token, or contradictory sex across the fused records. The confidence gate does
        not catch this, because the problem is a correct-looking link to the wrong person."""
        from collections import defaultdict
        keys_per_source = defaultdict(set)
        for rec in self.encounters + self.claims:
            keys_per_source[rec.source_id].add(rec.source_patient_key)
        if any(len(keys) > 1 for keys in keys_per_source.values()):
            return True
        sexes = {e.patient_sex for e in self.encounters if e.patient_sex}
        return len(sexes) > 1


def attach_tokens(records, token_map: TokenMap):
    """Stamp token_value + link_confidence onto each Encounter/Claim in place."""
    for rec in records:
        token, conf = token_map.lookup(rec.source_id, rec.source_patient_key)
        rec.patient_token = token
        rec.link_confidence = conf
    return records


def build_timelines(
    encounters: list[Encounter],
    claims: list[Claim],
    token_rows: list[dict],
) -> dict[str, PatientTimeline]:
    """End-to-end linkage: stamp tokens, then group by token into per-patient timelines.
    Records with no token (unlinkable) are dropped from the timelines and counted by the
    caller; we never fold an unlinked record into a patient on a guess."""
    token_map = TokenMap.from_rows(token_rows)
    attach_tokens(encounters, token_map)
    attach_tokens(claims, token_map)

    timelines: dict[str, PatientTimeline] = {}
    for enc in encounters:
        if enc.patient_token is None:
            continue
        timelines.setdefault(enc.patient_token, PatientTimeline(enc.patient_token)).encounters.append(enc)
    for clm in claims:
        if clm.patient_token is None:
            continue
        timelines.setdefault(clm.patient_token, PatientTimeline(clm.patient_token)).claims.append(clm)

    for tl in timelines.values():
        tl.encounters.sort(key=lambda e: (e.start_date is None, e.start_date))
        tl.claims.sort(key=lambda c: (c.service_date is None, c.service_date))
    return timelines
