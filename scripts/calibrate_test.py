#!/usr/bin/env python3
import json
import numpy as np

def main():
    with open("data/suppliers.json") as f:
        data = json.load(f)
    components = data["components"]
    scores = sorted([c["risk_score"] for c in components])
    n = len(scores)
    
    # Quantiles:
    # CRITICAL: top 10% (5 components)
    # HIGH: next 20% (10 components)
    # MEDIUM: next 30% (15 components)
    # LOW: bottom 40% (20 components)
    idx_crit = max(0, n - 5)
    idx_high = max(0, n - 15)
    idx_med = max(0, n - 30)
    
    t_crit = scores[idx_crit]
    t_high = scores[idx_high]
    t_med = scores[idx_med]
    
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s in scores:
        if s >= t_crit:
            counts["CRITICAL"] += 1
        elif s >= t_high:
            counts["HIGH"] += 1
        elif s >= t_med:
            counts["MEDIUM"] += 1
        else:
            counts["LOW"] += 1
            
    print("=============================================================")
    print("📊 Calculated Risk Thresholds Calibration Results")
    print("=============================================================")
    print(f"Total components: {n}")
    print(f"Thresholds:")
    print(f"  - CRITICAL: >= {t_crit:.4f}")
    print(f"  - HIGH:     >= {t_high:.4f}")
    print(f"  - MEDIUM:   >= {t_med:.4f}")
    print(f"  - LOW:      <  {t_med:.4f}")
    print("\nBand distribution counts:")
    for band, count in counts.items():
        print(f"  {band:<10}: {count}")

if __name__ == "__main__":
    main()
