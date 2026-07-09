#!/usr/bin/env python3
"""
test_parse.py
Tests _parse_structured_explanation without the GPU or any network call.
Covers the bug where clean_response_text ran on the raw blob before the
label split, destroying the RISK_FACTOR/SCENARIO/MITIGATION structure.
Run from the repo root:
    python3 scripts/test_parse.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_agent"))
from agent import _parse_structured_explanation

failures = []


def check(condition, message):
    if not condition:
        failures.append(f"FAIL: {message}")


def parse_failed(fields):
    return fields is None or not all(fields.values())


# ── Fixture 1: well-formed, uppercase labels, with markdown noise ─────────
well_formed = (
    "RISK_FACTOR: **Geographic concentration** poses a significant risk due to potential regional disruptions.\n"
    "SCENARIO: A single earthquake in the region could halt production for months.\n"
    "MITIGATION: Diversify sourcing to include a secondary supplier in a different region.\n"
)
fields = _parse_structured_explanation(well_formed)
print("=== Fixture 1: well-formed ===")
print(fields)
check(fields is not None, "well-formed fixture failed to parse")
if fields is not None:
    check(
        fields["risk_factor"] == "Geographic concentration poses a significant risk due to potential regional disruptions.",
        f"well-formed risk_factor mismatch: {fields['risk_factor']!r}",
    )
    check("**" not in fields["risk_factor"], "markdown bold survived per-field cleaning")
    check(
        fields["scenario"] == "A single earthquake in the region could halt production for months.",
        f"well-formed scenario mismatch: {fields['scenario']!r}",
    )
    check(
        fields["mitigation"] == "Diversify sourcing to include a secondary supplier in a different region.",
        f"well-formed mitigation mismatch: {fields['mitigation']!r}",
    )
check(parse_failed(fields) is False, "well-formed fixture incorrectly flagged parse_failed")
print()

# ── Fixture 2: lowercase labels ────────────────────────────────────────────
lowercase = (
    "risk_factor: Single-source dependency creates a critical bottleneck.\n"
    "scenario: If the sole supplier faces a labor strike, deliveries stop entirely.\n"
    "mitigation: Qualify a second supplier within twelve months.\n"
)
fields = _parse_structured_explanation(lowercase)
print("=== Fixture 2: lowercase labels ===")
print(fields)
check(fields is not None, "lowercase-label fixture failed to parse")
if fields is not None:
    check(
        fields["risk_factor"] == "Single-source dependency creates a critical bottleneck.",
        f"lowercase risk_factor mismatch: {fields['risk_factor']!r}",
    )
    check(
        fields["scenario"] == "If the sole supplier faces a labor strike, deliveries stop entirely.",
        f"lowercase scenario mismatch: {fields['scenario']!r}",
    )
    check(
        fields["mitigation"] == "Qualify a second supplier within twelve months.",
        f"lowercase mitigation mismatch: {fields['mitigation']!r}",
    )
check(parse_failed(fields) is False, "lowercase-label fixture incorrectly flagged parse_failed")
print()

# ── Fixture 3: missing MITIGATION — the exact bug repro from the report ──
missing_mitigation = (
    "RISK_FACTOR: Geographic concentration poses a significant risk due to "
    "potential regional disruptions. SCENARIO: "
)
fields = _parse_structured_explanation(missing_mitigation)
print("=== Fixture 3: missing MITIGATION ===")
print(fields)
check(fields is None, f"missing-MITIGATION fixture should fail to match, got {fields!r}")
check(parse_failed(fields) is True, "missing-MITIGATION fixture should set parse_failed")
print()

if failures:
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print("ALL ASSERTIONS PASSED")
