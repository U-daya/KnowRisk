#!/usr/bin/env python3
"""
test_clean.py
Tests clean_response_text without the GPU or any network call.
Run from the repo root:
    python3 scripts/test_clean.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_agent"))
from agent import clean_response_text

FIXTURE = (
    "## Why Is This Risky?\n"
    "**Bold claim** and __underline__ are stripped.\n"
    "6. Numbered list item that should lose its marker.\n"
    "Single Source: Being sourced from a single supplier increases vulnerability.\n"
    "Advanced Technology: Advanced logic dies require cutting-edge processes.\n"
    "Export Controls: The component faces additional regulatory hurdles.\n"
    "However the ratio is 3:1 in favor of risk, and http://example.com is fine.\n"
)

result = clean_response_text(FIXTURE)
print("=== OUTPUT ===")
print(result)
print()

failures = []

# 1. No markdown headers
if "##" in result:
    failures.append("FAIL: '##' header survived")

# 2. No bold/underline markup
if "**" in result or "__" in result:
    failures.append("FAIL: bold/underline markup survived")

# 3. No numbered list marker "6. "
if "6. " in result:
    failures.append("FAIL: numbered list marker '6.' survived")

# 4. Label-colon patterns removed at sentence boundaries
for label in ("Single Source: ", "Advanced Technology: ", "Export Controls: "):
    if label in result:
        failures.append(f"FAIL: label-colon '{label}' survived")

# 5. Mid-sentence colon "3:1" must survive
if "3:1" not in result:
    failures.append("FAIL: mid-sentence '3:1' was incorrectly stripped")

# 6. The substantive words after labels must still be present
for word in ("vulnerability", "processes", "hurdles"):
    if word not in result:
        failures.append(f"FAIL: content word '{word}' was removed along with its label")

if failures:
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print("ALL ASSERTIONS PASSED")
