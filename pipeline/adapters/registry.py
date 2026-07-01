"""Adapter registry: source_id -> how to turn its raw rows into canonical objects.

This indirection is the scaling story. A production deployment spans dozens of EHR
providers and several claims providers; onboarding one is "register an adapter," and the
pipeline below never changes.
The registry also records which kind of source each adapter handles (clinical encounters
vs claims) so the runner knows what to build.
"""

from __future__ import annotations

from pipeline.adapters import claims_x, provider_a, provider_b
from pipeline.canonical import Claim, Encounter

EHR_ADAPTERS = {
    provider_a.SOURCE_ID: provider_a.build_encounters,
    provider_b.SOURCE_ID: provider_b.build_encounters,
}

CLAIMS_ADAPTERS = {
    claims_x.SOURCE_ID: claims_x.build_claims,
}


def build_all_encounters(tables: dict[str, list[dict]]) -> list[Encounter]:
    encounters: list[Encounter] = []
    for build in EHR_ADAPTERS.values():
        encounters.extend(build(tables))
    return encounters


def build_all_claims(tables: dict[str, list[dict]]) -> list[Claim]:
    claims: list[Claim] = []
    for build in CLAIMS_ADAPTERS.values():
        claims.extend(build(tables))
    return claims
