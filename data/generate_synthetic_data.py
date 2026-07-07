#!/usr/bin/env python3
"""
generate_synthetic_data.py
Generates a synthetic multi-tier semiconductor supplier graph.
Outputs: data/suppliers.json
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

# ── Country risk profiles ──────────────────────────────────────────────────
COUNTRY_RISK = {
    "Taiwan":        {"geo_risk": 0.75, "export_ctrl_prob": 0.10},
    "South Korea":   {"geo_risk": 0.40, "export_ctrl_prob": 0.05},
    "Japan":         {"geo_risk": 0.25, "export_ctrl_prob": 0.05},
    "Netherlands":   {"geo_risk": 0.15, "export_ctrl_prob": 0.30},
    "USA":           {"geo_risk": 0.10, "export_ctrl_prob": 0.15},
    "China":         {"geo_risk": 0.65, "export_ctrl_prob": 0.60},
    "Germany":       {"geo_risk": 0.10, "export_ctrl_prob": 0.08},
    "Malaysia":      {"geo_risk": 0.30, "export_ctrl_prob": 0.05},
    "Vietnam":       {"geo_risk": 0.35, "export_ctrl_prob": 0.05},
    "Israel":        {"geo_risk": 0.45, "export_ctrl_prob": 0.20},
}

# ── Component catalog ──────────────────────────────────────────────────────
COMPONENT_CATALOG = [
    # (name, category, tier, preferred_country, lead_time_base, single_source_prob)
    ("Advanced Logic Die",       "Semiconductor",  1, "Taiwan",      180, 0.9),
    ("HBM3 Memory Stack",        "Memory",         1, "South Korea", 120, 0.7),
    ("EUV Photomask",            "Photonics",      1, "Netherlands", 200, 0.95),
    ("Advanced Packaging Substrate","Substrate",   1, "Taiwan",      150, 0.8),
    ("Wafer Fab Equipment",      "Equipment",      1, "Netherlands", 365, 0.85),
    ("GDDR7 Memory",             "Memory",         1, "South Korea", 90,  0.5),
    ("SiC Power Module",         "Power",          1, "USA",         60,  0.4),
    ("RF Front-End Module",      "RF",             2, "Taiwan",      75,  0.6),
    ("PCIe 5.0 Switch IC",       "Connectivity",   2, "USA",         45,  0.3),
    ("High-K Dielectric Film",   "Materials",      1, "Germany",     90,  0.7),
    ("Copper Interconnect Wire", "Materials",      2, "Japan",       30,  0.2),
    ("Silicon Wafer (300mm)",    "Materials",      1, "Japan",       60,  0.5),
    ("Photoresist (EUV-grade)",  "Chemicals",      1, "Japan",       120, 0.8),
    ("Process Control System",   "Equipment",      2, "USA",         180, 0.4),
    ("OSAT Assembly Service",    "Assembly",       2, "Malaysia",    45,  0.3),
    ("Advanced Thermal Interface","Thermal",       2, "Germany",     30,  0.2),
    ("CoWoS Interposer",         "Packaging",      1, "Taiwan",      120, 0.85),
    ("Sapphire Substrate",       "Materials",      1, "Japan",       90,  0.6),
    ("Rare Earth Magnets",       "Materials",      2, "China",       45,  0.7),
    ("Cobalt Slurry (CMP)",      "Chemicals",      2, "USA",         30,  0.3),
    ("Lithography System",       "Equipment",      1, "Netherlands", 400, 0.95),
    ("Etch System",              "Equipment",      2, "USA",         180, 0.4),
    ("CVD System",               "Equipment",      2, "Japan",       150, 0.5),
    ("Inspection System (e-beam)","Equipment",     2, "Israel",      200, 0.6),
    ("PCB (Multi-layer)",        "PCB",            3, "China",       21,  0.2),
    ("Passive Components (MLCC)","Passives",       3, "Japan",       14,  0.2),
    ("Voltage Regulator Module", "Power",          3, "USA",         30,  0.25),
    ("Ethernet PHY IC",          "Connectivity",   3, "USA",         28,  0.2),
    ("Clock Generator IC",       "Timing",         3, "USA",         28,  0.2),
    ("DDR5 DRAM Module",         "Memory",         2, "South Korea", 45,  0.3),
    ("NVMe SSD Controller",      "Storage",        3, "Taiwan",      35,  0.3),
    ("Board Management Controller","MCU",          3, "USA",         30,  0.2),
    ("Fiber Optic Transceiver",  "Optical",        3, "China",       45,  0.4),
    ("High-Voltage Capacitor",   "Passives",       3, "Japan",       21,  0.2),
    ("Custom ASIC (AI Engine)",  "Semiconductor",  1, "Taiwan",      210, 0.95),
    ("3D NAND Flash",            "Memory",         2, "South Korea", 60,  0.4),
    ("Tantalum Capacitor",       "Passives",       3, "China",       28,  0.5),
    ("Power Management IC",      "Power",          3, "USA",         30,  0.25),
    ("Optical Sensor Array",     "Optical",        2, "Japan",       60,  0.5),
    ("Thermal Paste (high-perf)","Thermal",        3, "Germany",     14,  0.2),
    ("Indium Phosphide Wafer",   "Materials",      1, "USA",         90,  0.7),
    ("Gallium Nitride Die",      "Semiconductor",  2, "USA",         75,  0.6),
    ("Electroplating Chemistry", "Chemicals",      3, "Germany",     21,  0.2),
    ("Specialty Gas (WF6)",      "Chemicals",      2, "USA",         45,  0.6),
    ("Argon/Nitrogen Supply",    "Chemicals",      3, "USA",         7,   0.1),
    ("Back-End Test Equipment",  "Equipment",      3, "Japan",       90,  0.35),
    ("Carrier Tape/Reel",        "Packaging",      3, "China",       14,  0.15),
    ("Shipping Container Slot",  "Logistics",      3, "Malaysia",    7,   0.1),
    ("Export License (ECCN 3E)", "Regulatory",     1, "USA",         90,  0.9),
    ("IP Core License (EDA)",    "Software",       1, "USA",         30,  0.5),
]

def compute_risk_score(component: dict) -> float:
    """Heuristic risk score [0, 1]."""
    country = component["country"]
    cr = COUNTRY_RISK.get(country, {"geo_risk": 0.3, "export_ctrl_prob": 0.1})

    score = 0.0
    score += cr["geo_risk"] * 0.30
    score += (1.0 if component["single_source"] else 0.0) * 0.30
    score += (1.0 if component["export_controlled"] else 0.0) * 0.20
    score += min(component["lead_time_days"] / 400.0, 1.0) * 0.20

    noise = random.gauss(0, 0.03)
    return round(max(0.0, min(1.0, score + noise)), 4)


def build_dependency_graph(components: list[dict]) -> dict[str, list[str]]:
    """Build tier-based dependency edges (tier N depends on tier N+1)."""
    by_tier: dict[int, list[str]] = {}
    for c in components:
        tier = c["tier"]
        by_tier.setdefault(tier, []).append(c["id"])

    deps: dict[str, list[str]] = {c["id"]: [] for c in components}

    # Tier 1 depends on tier 2; tier 2 depends on tier 3
    for tier in [1, 2]:
        next_tier = by_tier.get(tier + 1, [])
        if not next_tier:
            continue
        for cid in by_tier.get(tier, []):
            n_deps = random.randint(1, min(4, len(next_tier)))
            deps[cid] = random.sample(next_tier, n_deps)

    return deps


def main():
    out_dir = Path(__file__).parent
    suppliers_path = out_dir / "suppliers.json"

    components = []
    for i, (name, category, tier, country, lead_base, ss_prob) in enumerate(COMPONENT_CATALOG):
        cr = COUNTRY_RISK.get(country, {"geo_risk": 0.3, "export_ctrl_prob": 0.1})
        single_source = random.random() < ss_prob
        export_controlled = random.random() < cr["export_ctrl_prob"]
        lead_time = lead_base + random.randint(-10, 20)

        comp = {
            "id": f"COMP-{i+1:03d}",
            "name": name,
            "category": category,
            "tier": tier,
            "country": country,
            "single_source": single_source,
            "export_controlled": export_controlled,
            "lead_time_days": lead_time,
            "risk_score": 0.0,  # filled below
            "dependencies": [],
        }
        components.append(comp)

    # Build dependency graph
    dep_map = build_dependency_graph(components)
    for comp in components:
        comp["dependencies"] = dep_map[comp["id"]]
        comp["risk_score"] = compute_risk_score(comp)

    # Compute aggregate risk (propagate from dependencies)
    id_to_comp = {c["id"]: c for c in components}
    for comp in components:
        if comp["dependencies"]:
            dep_avg = sum(id_to_comp[d]["risk_score"] for d in comp["dependencies"]) / len(comp["dependencies"])
            comp["risk_score"] = round(0.7 * comp["risk_score"] + 0.3 * dep_avg, 4)

    payload = {
        "schema_version": "1.0",
        "generated_at": "2026-07-07T00:00:00Z",
        "component_count": len(components),
        "components": components,
    }

    with open(suppliers_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Generated {len(components)} components → {suppliers_path}")
    high_risk = [c for c in components if c["risk_score"] >= 0.6]
    print(f"   High-risk (≥0.6): {len(high_risk)}")
    single_src = [c for c in components if c["single_source"]]
    print(f"   Single-source:    {len(single_src)}")
    export_ctrl = [c for c in components if c["export_controlled"]]
    print(f"   Export-controlled:{len(export_ctrl)}")


if __name__ == "__main__":
    main()
