"""Recognising clinical entities in generated prose, to catch uncited claims.

The grounding gate's first job is verifying declared citations. But a drifting language
model can also state a clinical fact in prose and simply not cite it ("started on insulin
glargine"), which a citation-only check never sees. Any recognised clinical entity in the
response must therefore resolve to the source record, cited or not.

Recognition works two ways, and the second is the important one:

  1. A lexicon of common medications and labs (below). Useful but bounded: enumerate drugs
     and an adversary just names one you forgot (we learned this the hard way; the first
     version listed `insulin` and the next attack used `semaglutide`).

  2. **Drug-stem morphology.** Generic drug names are not arbitrary: the WHO/USAN naming
     system assigns systematic stems by class (-glutide for GLP-1 agonists, -gliflozin for
     SGLT2 inhibitors, -mab for monoclonal antibodies, -pril for ACE inhibitors, -sartan
     for ARBs, -statin, -xaban, -gatran, ...). Recognising a drug by its *stem* catches
     semaglutide, tirzepatide, canagliflozin, and dabigatran without ever having listed
     them. This is the difference between a blocklist and a principle: morphology
     generalises to drugs we have never seen.

Honest about scope: this is morphology + a lexicon, not a clinical NER/NLI model. It will
miss an entity that neither matches a stem nor is listed, and it does not understand
negation or hypotheticals (a ruled-out drug named by stem would be flagged). In production
this scan is a clinical NER / NLI step, and it drops in behind the same gate. The durable
invariant is "a recognised clinical entity must be grounded"; morphology just makes
recognition far harder to evade offline.
"""

from __future__ import annotations

import re

# Common medication tokens (generic and brand). Lowercased, matched on word boundaries.
DRUG_TERMS = {
    "metformin", "insulin", "glargine", "lispro", "empagliflozin", "dapagliflozin",
    "lisinopril", "enalapril", "losartan", "amlodipine", "hydrochlorothiazide",
    "atorvastatin", "simvastatin", "rosuvastatin", "albuterol", "fluticasone",
    "furosemide", "spironolactone", "metoprolol", "carvedilol", "warfarin",
    "apixaban", "rivaroxaban", "aspirin", "clopidogrel", "omeprazole", "prednisone",
    "gabapentin", "levothyroxine", "amoxicillin", "azithromycin", "ceftriaxone",
    "morphine", "oxycodone", "heparin", "digoxin", "sertraline",
}

# Common laboratory / measurable observation tokens.
LAB_TERMS = {
    "a1c", "hemoglobin", "glucose", "creatinine", "potassium", "sodium", "troponin",
    "bnp", "ldl", "hdl", "cholesterol", "triglycerides", "tsh", "wbc", "platelet",
    "bilirubin", "albumin", "lactate", "inr", "ferritin",
}

CLINICAL_TERMS = DRUG_TERMS | LAB_TERMS

# USAN / INN generic drug stems (suffixes). A token ending in one of these, of reasonable
# length, is almost certainly a medication regardless of whether we have seen it before.
# Chosen to be specific enough that everyday English words do not end in them; the length
# guard below handles the few that could collide (e.g. "April" vs the -pril stem).
DRUG_STEMS = (
    "glutide", "tide", "gliflozin", "gliptin", "glitazone", "sartan", "pril", "olol",
    "dipine", "statin", "cillin", "cycline", "floxacin", "prazole", "conazole", "azole",
    "parin", "xaban", "gatran", "afil", "triptan", "tinib", "ciclib", "mab", "nib", "vir",
    "mycin", "tidine", "semide", "thiazide", "fenac", "profen", "caine", "barbital",
    "oxetine", "opram", "azepam", "zolam", "dronate", "lukast", "setron", "sone", "olone",
)

# Common clinical/biochemistry words that happen to end in a drug stem but are not drugs.
# Without this, "-tide"/"-tidine" flag "peptide", "nucleotide", "histidine", and a faithful
# summary mentioning "natriuretic peptide" gets wrongly rejected.
_NON_DRUG = {
    "peptide", "polypeptide", "dipeptide", "tripeptide", "oligopeptide", "phosphatide",
    "nucleotide", "oligonucleotide", "histidine", "cytidine", "uridine", "thymidine",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def is_drug_like(token: str) -> bool:
    """True when a token's morphology marks it as a generic drug name. The length guard
    (>= 7) keeps short English words that happen to end in a stem (e.g. 'April' -> -pril,
    'nib') from being mistaken for drugs; the `_NON_DRUG` set excludes the handful of
    common biochemistry words that share a stem ('peptide', 'histidine'). Real generic
    names clear both. This is still a bounded recogniser, not NER (the honest limit)."""
    if not token:
        return False
    t = token.lower()
    if len(t) < 7 or t in _NON_DRUG:
        return False
    return any(t.endswith(stem) for stem in DRUG_STEMS)


def clinical_terms_in(text: str) -> set[str]:
    """Recognised clinical entity tokens present in `text`: lexicon hits plus any token
    whose morphology marks it as a drug."""
    tokens = set(_TOKEN_RE.findall((text or "").lower()))
    hits = tokens & CLINICAL_TERMS
    hits |= {t for t in tokens if t not in hits and is_drug_like(t)}
    return hits


def term_kind(term: str) -> str:
    return "lab" if term in LAB_TERMS else "medication"
