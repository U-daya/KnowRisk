#!/usr/bin/env python3
"""
agent.py — KnowRisk LLM Agent
Combines the classifier's quantitative risk score with an LLM-generated
plain-language explanation.

Reads from environment:
  LLM_BASE_URL  — e.g. http://localhost:8001/v1
  LLM_MODEL     — e.g. Qwen/Qwen2.5-70B-Instruct
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
SUPPLIERS    = REPO_ROOT / "data" / "suppliers.json"
MODEL_PATH   = REPO_ROOT / "classifier" / "risk_model.joblib"
FEATURES_PATH= REPO_ROOT / "classifier" / "feature_columns.json"

# ── Lazy-loaded globals ────────────────────────────────────────────────────
_model        = None
_features     = None
_suppliers    = None
_tokenizer    = None
_llm_model    = None

def _load_classifier():
    global _model, _features
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Classifier model not found: {MODEL_PATH}. Run classifier/train_risk_model.py first.")
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _features = json.load(f)
    return _model, _features

def _load_suppliers() -> dict:
    global _suppliers
    if _suppliers is None:
        if not SUPPLIERS.exists():
            raise FileNotFoundError(f"Supplier graph not found: {SUPPLIERS}. Run data/generate_synthetic_data.py first.")
        with open(SUPPLIERS) as f:
            data = json.load(f)
        _suppliers = {c["id"]: c for c in data["components"]}
    return _suppliers

def _print_rocm_smi(stage_name: str):
    """Print ROCm GPU status and utilization statistics."""
    print(f"\n⚡ [AMD Instinct MI300X] ROCm-SMI GPU Status at: {stage_name} ⚡")
    try:
        res = subprocess.run(["rocm-smi"], capture_output=True, text=True)
        print(res.stdout.strip())
    except Exception as e:
        print(f"rocm-smi execution failed: {e}")
    print("=" * 60)

def _get_llm():
    global _tokenizer, _llm_model
    if _llm_model is None:
        model_id = get_llm_model()
        print("=" * 60)
        print("🚀 [ROCm LLM Startup] Initializing local Hugging Face model...")
        print(f"   PyTorch ROCm/CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   AMD GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️ WARNING: No GPU detected by PyTorch, falling back to CPU!")
        print(f"   Loading {model_id} onto GPU...")
        print("=" * 60)
        
        t0 = time.time()
        _print_rocm_smi("Pre-Model Load")
        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ Local model loaded successfully in {time.time() - t0:.2f} seconds.")
        _print_rocm_smi("Post-Model Load")
    return _tokenizer, _llm_model

def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


# ── Feature extraction for a supplier component ───────────────────────────

COUNTRY_GEO_RISK = {
    "Taiwan": 0.75, "South Korea": 0.40, "Japan": 0.25,
    "Netherlands": 0.15, "USA": 0.10, "China": 0.65,
    "Germany": 0.10, "Malaysia": 0.30, "Vietnam": 0.35, "Israel": 0.45,
}

def _build_feature_row(component: dict, features: list[str]) -> pd.DataFrame:
    """Map a supplier component dict to model feature vector.

    Feature names must match classifier/feature_columns.json exactly.
    The classifier was trained on DataCo features known at order time:
    scheduled_ship_days, discount_rate, high_discount_flag, order_value,
    benefit_per_order, shipping_mode_enc, market_geo_risk, category_risk,
    product_price_log, order_quantity.
    """
    geo_risk = COUNTRY_GEO_RISK.get(component.get("country", ""), 0.35)
    lead_days = component.get("lead_time_days", 30)
    tier = component.get("tier", 3)
    single_src = component.get("single_source", False)
    export_ctrl = component.get("export_controlled", False)

    # Map supplier attributes to DataCo-analogue features
    # scheduled_ship_days: longer lead time = riskier (mapped to 1-6 day scale)
    sched_days = min(6, max(1, round(lead_days / 45)))
    # shipping_mode_enc: 3=Standard (high-risk), 0=Same Day (low-risk)
    ship_enc = 3 if lead_days > 90 else (2 if lead_days > 45 else 1)
    # discount_rate: export-controlled or single-source = cost premium (anomaly signal)
    discount = 0.35 if (single_src or export_ctrl) else 0.10
    # benefit_per_order: single-source suppliers have pricing power -> lower margins
    benefit = -100.0 if single_src else (50.0 if tier == 3 else 20.0)

    raw = {
        "scheduled_ship_days": float(sched_days),
        "discount_rate":       discount,
        "high_discount_flag":  1 if discount > 0.3 else 0,
        "order_value":         min(lead_days * 50, 10000.0),
        "benefit_per_order":   benefit,
        "shipping_mode_enc":   float(ship_enc),
        "market_geo_risk":     geo_risk,
        "category_risk":       0.5 if tier == 1 else (0.3 if tier == 2 else 0.2),
        "product_price_log":   np.log1p(lead_days * 20),
        "order_quantity":      float(max(1, 5 - tier)),
    }

    row = {f: raw.get(f, 0.0) for f in features}
    return pd.DataFrame([row], columns=features)


# ── Risk label helper ─────────────────────────────────────────────────────

def _risk_label(score: float) -> str:
    if score >= 0.70:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


# ── LLM explanation ───────────────────────────────────────────────────────

def _generate_explanation(component: dict, risk_score: float, risk_label: str,
                           feature_row: dict) -> str:
    """Generate a plain-language risk explanation directly using the local ROCm LLM."""
    try:
        tokenizer, model = _get_llm()
    except Exception as e:
        return f"[LLM initialization failed: {e}] Risk score {risk_score:.2f} ({risk_label}). " \
               f"Component sourced from {component['country']} with " \
               f"{'single-source' if component['single_source'] else 'multi-source'} supply."

    prompt = f"""You are a semiconductor supply-chain risk analyst. Analyze the following component and provide a concise, actionable risk assessment in 3-4 sentences.

Component: {component['name']}
Category: {component['category']}
Tier: {component['tier']} (1=most critical)
Country of origin: {component['country']}
Single-source supplier: {component['single_source']}
Export controlled: {component['export_controlled']}
Lead time: {component['lead_time_days']} days
Dependencies: {len(component.get('dependencies', []))} upstream components

ML Risk Score: {risk_score:.2f} / 1.00  ({risk_label})

Key risk drivers:
- Geographic risk (country): {feature_row.get('market_geo_risk', 0):.2f}
- Single-source vulnerability: {'Yes' if component['single_source'] else 'No'}
- Export control exposure: {'Yes' if component['export_controlled'] else 'No'}
- Supply lead time pressure: {component['lead_time_days']} days

Provide: (1) the primary risk factor, (2) potential supply disruption scenario, (3) a concrete mitigation recommendation. Be specific and concise."""

    try:
        _print_rocm_smi("Start of Explanation Generation")
        messages = [
            {"role": "system", "content": "You are an expert semiconductor supply chain risk analyst working for an AMD hardware team. Provide precise, actionable analysis."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        _print_rocm_smi("End of Explanation Generation")
        return response.strip()
    except Exception as e:
        return f"[LLM local inference failed: {e}] Risk score {risk_score:.2f} ({risk_label}). " \
               f"Component sourced from {component['country']} with " \
               f"{'single-source' if component['single_source'] else 'multi-source'} supply."


# ── Public API ────────────────────────────────────────────────────────────

def analyze_component(component_id: str) -> dict:
    """
    Full risk analysis for a single component.
    Returns structured dict with classifier score + LLM explanation.
    """
    suppliers   = _load_suppliers()
    model, features = _load_classifier()

    if component_id not in suppliers:
        raise ValueError(f"Unknown component: {component_id}. "
                         f"Valid IDs: {list(suppliers.keys())[:5]}...")

    component = suppliers[component_id]

    # ── Classifier inference ──────────────────────────────────────────────
    X = _build_feature_row(component, features)
    prob = float(model.predict_proba(X)[0][1])

    # Blend with heuristic risk_score from the graph for richer signal
    graph_score  = component.get("risk_score", prob)
    final_score  = round(0.6 * prob + 0.4 * graph_score, 4)
    risk_label   = _risk_label(final_score)

    feature_vals = X.iloc[0].to_dict()

    # ── LLM explanation ───────────────────────────────────────────────────
    explanation = _generate_explanation(component, final_score, risk_label, feature_vals)

    # ── Dependency risk summary ───────────────────────────────────────────
    dep_risks = []
    for dep_id in component.get("dependencies", []):
        if dep_id in suppliers:
            dep = suppliers[dep_id]
            dep_risks.append({
                "id":          dep_id,
                "name":        dep["name"],
                "country":     dep["country"],
                "risk_score":  dep.get("risk_score", 0.0),
                "single_source": dep.get("single_source", False),
            })

    return {
        "component_id":    component_id,
        "component_name":  component["name"],
        "category":        component["category"],
        "tier":            component["tier"],
        "country":         component["country"],
        "single_source":   component["single_source"],
        "export_controlled": component["export_controlled"],
        "lead_time_days":  component["lead_time_days"],
        "risk_score":      final_score,
        "classifier_prob": round(prob, 4),
        "graph_risk_score": round(graph_score, 4),
        "risk_label":      risk_label,
        "llm_explanation": explanation,
        "feature_values":  {k: round(v, 4) for k, v in feature_vals.items()},
        "dependency_risks": dep_risks,
    }


def answer_query(query: str, context_component_id: str | None = None) -> str:
    """
    Free-text Q&A about supply-chain risk.
    Optionally anchored to a specific component.
    """
    try:
        tokenizer, model = _get_llm()
    except Exception as e:
        return f"[LLM local initialization failed: {e}]"

    system_msg = (
        "You are a semiconductor supply-chain risk analyst with deep expertise in "
        "geopolitical risk, export controls, and semiconductor manufacturing. "
        "Answer questions concisely and with specific, actionable insights. "
        "Reference real supply chain dynamics when relevant."
    )

    context = ""
    if context_component_id:
        try:
            suppliers = _load_suppliers()
            if context_component_id in suppliers:
                comp = suppliers[context_component_id]
                context = (
                    f"\nContext — Currently viewing component: {comp['name']} "
                    f"(Tier {comp['tier']}, {comp['country']}, "
                    f"risk_score={comp.get('risk_score',0):.2f})\n"
                )
        except Exception:
            pass

    try:
        _print_rocm_smi("Start of Q&A Generation")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": context + query},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=400,
                temperature=0.4,
                do_sample=True
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        _print_rocm_smi("End of Q&A Generation")
        return response.strip()
    except Exception as e:
        return f"[LLM query failed: {e}]"


def list_components() -> list[dict]:
    """Return all components as a list."""
    suppliers = _load_suppliers()
    return list(suppliers.values())


# ── CLI test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    component_id = sys.argv[1] if len(sys.argv) > 1 else "COMP-001"
    print(f"Analyzing {component_id}...")
    result = analyze_component(component_id)
    print(json.dumps(result, indent=2))
