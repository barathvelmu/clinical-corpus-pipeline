"""PHI detection and redaction for free text.

Two jobs. As a utility, it redacts identifiers out of note text. As the engine behind the
PHI gate, it answers one question about a piece of generated output: does this contain
anything that looks like PHI? A generated example that resurfaces a name or an MRN from a
note is a privacy incident even when the source dataset was "de-identified," because
generation can copy identifiers straight through.

Honest about what this is: pattern/shape-based detection of the common HIPAA Safe Harbor
identifier shapes (names with explicit cues, dates finer than a year, MRNs, SSNs, phones,
emails, ages over 89). It catches the obvious shapes that dominate real leaks; it is not
a clinical NLP de-identification model. In production this stage is delegated to the
contracted privacy service (with its own SLA); this module is the in-pipeline
tripwire that runs on our own generated output before anything ships, plus a stand-in we
can test against. The detector deliberately errs toward over-flagging: a false positive
costs a regeneration, a false negative costs a breach.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern is (label, compiled regex). Order matters only for readability.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\d{3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # MRN / record number: an explicit cue followed by a digit run, so we do not flag
    # lab values. Several cue phrasings, case-insensitive.
    ("MRN", re.compile(
        r"\b(?:MRN|medical\s+record(?:\s+(?:number|no\.?|#))?|record\s+(?:number|no\.?|#))"
        r"[:#\s]*\d{3,}\b", re.IGNORECASE)),
    # Dates finer than a year: 04/12/1961, 04.12.1961, 2026-04-01, 1961/04/12,
    # "April 12, 1961", "12 April 1961". A bare month+year ("April 2024") is intentionally
    # NOT flagged here: it is routine prose and the day is the bigger identifier, so we
    # require a day to avoid killing legitimate examples.
    ("DATE", re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")),
    ("DATE", re.compile(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b")),
    ("DATE", re.compile(
        r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)|(?:January|February|March|April|May|"
        r"June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?),?"
        r"\s+\d{4}\b", re.IGNORECASE)),
    # Age >= 90 is a Safe Harbor identifier. Allow hyphenated and spaced forms, because
    # our own generator writes "92-year-old".
    ("AGE_90_PLUS", re.compile(
        r"\b(?:9\d|1\d\d)[\s-]*(?:years?[\s-]*old|y/?o|yo)\b", re.IGNORECASE)),
    # Long bare digit runs (9-11 digits): unseparated SSNs, phones, NPIs, account ids.
    # Over-flag on purpose; a false positive costs a regeneration.
    ("NUMERIC_ID", re.compile(r"\b\d{9,11}\b")),
    # Names introduced by a cue, including clinician and relationship cues. The cue is
    # case-insensitive (scoped `(?i:...)`, so sentence-initial "Patient Robert Langdon" is
    # caught) but the NAME itself stays case-sensitive, so "patient reports fatigue" is not
    # mistaken for a name. We do not NER every capitalised word (that flags drug brands and
    # cities); an un-cued bare name is left to the contracted de-id service's NLP, named as
    # a gap in the README.
    ("NAME", re.compile(
        r"\b(?i:patient|name|mr|mrs|ms|dr|doctor|spouse|husband|wife|son|daughter|"
        r"mother|father|brother|sister)\.?\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")),
]


@dataclass
class PHIFinding:
    label: str
    text: str
    start: int
    end: int


# Title-case words that are not personal names, so "Patient Care Team" is not flagged as a
# name. Smart over-flagging: we still err toward catching too much, just not obvious noise.
_NAME_STOPWORDS = {
    "care", "team", "self", "management", "syndrome", "center", "centre", "clinic",
    "department", "unit", "history", "exam", "education", "services", "service", "group",
    "plan", "program", "report", "summary", "note", "visit", "review", "control",
}

# Coding systems whose identifiers are genuinely long digit runs (so a number right after
# the cue is a code, not a patient id). Matched as whole words, not substrings, so "Decode"
# does not suppress. Deliberately excludes generic "code"/"icd" (an ICD code is not 9-11
# digits, and "code 123456789" is more likely a misplaced SSN we would rather over-flag).
_CODE_CONTEXT = ("snomed", "loinc", "rxnorm", "ndc")
_CODE_CONTEXT_RE = re.compile(r"\b(?:" + "|".join(_CODE_CONTEXT) + r")\b\s*$", re.IGNORECASE)


def _is_false_positive(label: str, match: re.Match, full: str) -> bool:
    if label == "NUMERIC_ID":
        prefix = full[max(0, match.start() - 16):match.start()]
        return bool(_CODE_CONTEXT_RE.search(prefix))
    if label == "NAME":
        # Use the captured name group (not a slice of the whole match, which mis-handles a
        # lowercase cue): if every name word is a stopword it is a phrase like "Patient Care
        # Team", not a person.
        name = match.group(1) if match.groups() else match.group(0)
        name_words = re.findall(r"[A-Z][a-z]+", name)
        return bool(name_words) and all(w.lower() in _NAME_STOPWORDS for w in name_words)
    return False


def _overlaps(span: tuple[int, int], accepted: list[tuple[int, int]]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in accepted)


def find_phi(text: str | None) -> list[PHIFinding]:
    if not text:
        return []
    findings: list[PHIFinding] = []
    spans: list[tuple[int, int]] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            # Skip overlapping spans (e.g. MRN-cue+digits vs the bare-digit NUMERIC_ID over
            # the same number), so redaction does not splice two overlapping replacements.
            if _overlaps(span, spans):
                continue
            if _is_false_positive(label, m, text):
                continue
            spans.append(span)
            findings.append(PHIFinding(label=label, text=m.group(0),
                                       start=m.start(), end=m.end()))
    return sorted(findings, key=lambda f: f.start)


def contains_phi(text: str | None) -> bool:
    return len(find_phi(text)) > 0


def redact(text: str | None) -> str:
    """Replace each finding with a typed placeholder, e.g. [NAME], [DATE]. Right-to-left
    so earlier offsets stay valid as we splice."""
    if not text:
        return text or ""
    out = text
    for f in sorted(find_phi(text), key=lambda f: f.start, reverse=True):
        out = out[:f.start] + f"[{f.label}]" + out[f.end:]
    return out
