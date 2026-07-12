#!/usr/bin/env python3
"""
serve.py — KnowRisk public/edge server.

Runs on a small always-on host (Render/Railway/Fly/any container).
Serves the built frontend AND the API from one origin.

Generates risk briefs + Q&A by proxying to an OpenAI-compatible LLM endpoint
(your AMD MI300X droplet's vLLM server, or any compatible endpoint) when it is
reachable. When the droplet is down or unconfigured, falls back to a clearly-
labelled synthetic response — the UI shows "SYNTHETIC — LLM OFFLINE" and the
footer shows the mode.

Env vars (all optional — with none set, it runs pure synthetic):
  LLM_BASE_URL     OpenAI-compatible base URL, e.g. http://<droplet-ip>:8001/v1
  LLM_MODEL        model id served there, e.g. Qwen/Qwen2.5-7B-Instruct
  LLM_API_KEY      key if the endpoint needs one (vLLM usually doesn't)
  LLM_GPU_LABEL    footer label when live (default: AMD Instinct MI300X · ROCm)
  PORT             port to bind (default 8000)
"""

import json
import os
import re
import random
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
FRONTEND    = ROOT / "frontend" / "index.html"
SUPPLIERS_PATH = ROOT / "data" / "suppliers.json"

# ── Load supplier graph ────────────────────────────────────────────────────
with open(SUPPLIERS_PATH) as f:
    SUP = json.load(f)
COMPS: dict = {c["id"]: c for c in SUP["components"]}

# ── Geographic risk table ──────────────────────────────────────────────────
GEO = {
    "Taiwan": 0.75, "South Korea": 0.40, "Japan": 0.25,
    "Netherlands": 0.15, "USA": 0.10, "China": 0.65,
    "Germany": 0.10, "Malaysia": 0.30, "Vietnam": 0.35, "Israel": 0.45,
}

# ── LLM config ────────────────────────────────────────────────────────────
LLM_BASE_URL  = os.environ.get("LLM_BASE_URL", "").strip()
LLM_MODEL     = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_API_KEY   = os.environ.get("LLM_API_KEY", "not-needed")
LLM_GPU_LABEL = os.environ.get("LLM_GPU_LABEL", "AMD Instinct MI300X · ROCm")

# ── Lazy OpenAI client ────────────────────────────────────────────────────
_oai_client = None

def _llm_client():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        _oai_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=25.0,
        )
    return _oai_client


def _llm_chat(system: str, user: str, max_tokens: int = 320, temperature: float = 0.3) -> tuple[str, int]:
    """Call the configured LLM. Returns (text, latency_ms). Raises on failure."""
    t0 = time.time()
    resp = _llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()
    return text, int((time.time() - t0) * 1000)


def _llm_reachable(timeout: float = 3.0) -> bool:
    """Fast reachability probe — used by /api/health."""
    if not LLM_BASE_URL:
        return False
    try:
        r = httpx.get(
            LLM_BASE_URL.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=timeout,
        )
        return r.status_code < 500
    except Exception:
        return False


# ── Risk helpers ──────────────────────────────────────────────────────────
def _risk_label(score: float) -> str:
    if score >= 0.70:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    return "LOW"


# ── Synthetic brief (no LLM needed) ──────────────────────────────────────
def _synth_brief(c: dict) -> str:
    geo = GEO.get(c["country"], 0.35)
    parts = [
        f"[SYNTHETIC — LLM OFFLINE] {c['name']} is a Tier-{c['tier']} component sourced from "
        f"{c['country']} (geographic risk index: {geo:.2f}).",
    ]
    if c["single_source"]:
        parts.append("This is a single-source component, creating critical vulnerability to any supplier disruption.")
    if c["export_controlled"]:
        parts.append("Export-control regulations add regulatory and licensing risk.")
    parts.append(
        f"With a {c['lead_time_days']}-day lead time, recovery from any disruption would be slow. "
        "Recommended mitigations: qualify a second source, hold 6–12 months of safety stock, "
        "and design in alternate components where feasible."
    )
    return " ".join(parts)


# ── Live LLM brief ────────────────────────────────────────────────────────
def _live_brief(c: dict) -> str:
    """Call the AMD droplet for a structured risk brief, fall back to synthetic on any error."""
    geo = GEO.get(c["country"], 0.35)
    system = (
        "You are a semiconductor supply-chain risk analyst. "
        "Respond ONLY with a JSON object with exactly three string keys: "
        "risk_factor, scenario, mitigation. Each value must be 1–2 concise sentences. "
        "No markdown, no extra keys, no preamble."
    )
    user = (
        f"Component: {c['name']}\n"
        f"Category: {c['category']}\n"
        f"Tier: {c['tier']} (1 = most critical)\n"
        f"Country of origin: {c['country']} (geo risk {geo:.2f})\n"
        f"Single-source supplier: {c['single_source']}\n"
        f"Export-controlled: {c['export_controlled']}\n"
        f"Lead time: {c['lead_time_days']} days\n"
        f"Risk score: {c['risk_score']:.2f}/1.00 ({_risk_label(c['risk_score'])})\n\n"
        "Provide: (1) the primary risk_factor, (2) a concrete disruption scenario, "
        "(3) an actionable mitigation recommendation."
    )
    try:
        text, ms = _llm_chat(system, user, max_tokens=320, temperature=0.3)
        m = re.search(r"\{.*?\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
        rf = (data.get("risk_factor") or "").strip()
        sc = (data.get("scenario")    or "").strip()
        mt = (data.get("mitigation")  or "").strip()
        if rf and sc and mt:
            return (
                f"[AMD MI300X · LIVE] {rf}\n\n"
                f"Disruption Scenario: {sc}\n\n"
                f"Mitigation: {mt}\n\n"
                f"(Inference: {ms}ms · {LLM_MODEL})"
            )
    except Exception as e:
        print(f"[serve] LLM brief failed → synthetic: {e}")
    return _synth_brief(c)


# ── Feature values for UI bars (derived from suppliers.json fields) ────────
def _feature_values(c: dict) -> dict:
    geo = GEO.get(c["country"], 0.35)
    lead = c["lead_time_days"]
    single = c["single_source"]
    export = c["export_controlled"]
    tier   = c["tier"]
    return {
        "market_geo_risk":    geo,
        "late_delivery_risk": 0.9 if single else 0.2,
        "high_discount_flag": 0.85 if export else 0.1,
        "order_status_risk":  c["risk_score"] * 0.7,
        "shipping_mode_risk": min(lead / 400, 1.0),
        "product_price_log":  0.5 if tier == 1 else (0.3 if tier == 2 else 0.15),
        "category_risk":      0.5 if tier == 1 else (0.3 if tier == 2 else 0.2),
    }


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="KnowRisk API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────
class QueryReq(BaseModel):
    query: str
    component_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND), media_type="text/html")
    return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)


@app.get("/api/health")
def health():
    components = SUP["components"]
    live = _llm_reachable()
    if live:
        gpu_label = f"{LLM_GPU_LABEL} · LIVE"
    elif LLM_BASE_URL:
        gpu_label = "AMD droplet unreachable — SYNTHETIC FALLBACK"
    else:
        gpu_label = "SYNTHETIC (no AMD droplet configured)"

    return {
        "status":       "ok",
        "llm_model":    LLM_MODEL if live else "synthetic",
        "gpu":          gpu_label,
        "gpu_available": live,
        "llm_loaded":   live,
        "data_summary": {
            "components_count":       len(components),
            "single_source_count":    sum(1 for c in components if c["single_source"]),
            "export_controlled_count": sum(1 for c in components if c["export_controlled"]),
            "median_lead_time":       sorted(c["lead_time_days"] for c in components)[len(components) // 2],
        },
    }


@app.get("/api/components")
def list_components():
    return [
        {
            "id":               c["id"],
            "name":             c["name"],
            "category":         c["category"],
            "tier":             c["tier"],
            "country":          c["country"],
            "single_source":    c["single_source"],
            "export_controlled": c["export_controlled"],
            "lead_time_days":   c["lead_time_days"],
            "risk_score":       c["risk_score"],
            "risk_label":       _risk_label(c["risk_score"]),
            "n_dependencies":   len(c.get("dependencies", [])),
        }
        for c in SUP["components"]
    ]


@app.get("/api/risk/{cid}")
def get_risk(cid: str):
    if cid not in COMPS:
        raise HTTPException(status_code=404, detail=f"Unknown component: {cid}")
    c = COMPS[cid]

    deps = [
        {
            "id":           d,
            "name":         COMPS[d]["name"],
            "country":      COMPS[d]["country"],
            "risk_score":   COMPS[d]["risk_score"],
            "single_source": COMPS[d]["single_source"],
        }
        for d in c.get("dependencies", [])
        if d in COMPS
    ]

    explanation = _live_brief(c) if LLM_BASE_URL else _synth_brief(c)
    live = LLM_BASE_URL and "[AMD MI300X" in explanation

    return {
        "component_id":     cid,
        "component_name":   c["name"],
        "category":         c["category"],
        "tier":             c["tier"],
        "country":          c["country"],
        "single_source":    c["single_source"],
        "export_controlled": c["export_controlled"],
        "lead_time_days":   c["lead_time_days"],
        "risk_score":       c["risk_score"],
        "risk_label":       _risk_label(c["risk_score"]),
        "llm_explanation":  explanation,
        "model_id":         LLM_MODEL if live else "Synthetic fallback",
        "feature_values":   _feature_values(c),
        "dependency_risks": deps,
    }


@app.post("/api/query")
def query(r: QueryReq):
    if not r.query or len(r.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")

    ctx = COMPS.get(r.component_id or "")

    if LLM_BASE_URL:
        system = (
            "You are a semiconductor supply-chain risk analyst with deep expertise in "
            "geopolitical risk, export controls, and chip manufacturing. "
            "Answer concisely with specific, actionable insight."
        )
        ctx_prefix = (
            f"Context: currently viewing {ctx['name']} "
            f"(Tier {ctx['tier']}, {ctx['country']}, risk score {ctx['risk_score']:.2f}).\n\n"
            if ctx else ""
        )
        try:
            text, ms = _llm_chat(system, ctx_prefix + r.query, max_tokens=400, temperature=0.4)
            return {
                "query":  r.query,
                "answer": text,
                "model":  LLM_MODEL,
                "source": "mi300x",
                "latency_ms": ms,
            }
        except Exception as e:
            print(f"[serve] LLM query failed → synthetic: {e}")

    # Synthetic fallback
    ctx_note = f" Focused on {ctx['name']} ({ctx['country']})," if ctx else ""
    answer = (
        f"[SYNTHETIC — LLM OFFLINE]{ctx_note} risk in semiconductor supply chains is driven by "
        "geographic concentration (especially Taiwan/South Korea), single-source dependencies, "
        "export-control regulations, and extended lead times. "
        "Recommended actions: dual-source critical components, maintain 6–12 months safety stock, "
        "and implement continuous geopolitical monitoring. "
        "Connect the AMD MI300X droplet to enable live AI-powered analysis."
    )
    return {
        "query":  r.query,
        "answer": answer,
        "model":  "synthetic",
        "source": "synthetic",
        "latency_ms": random.randint(50, 120),
    }


@app.get("/api/model-metrics")
def model_metrics():
    for p in [
        ROOT / "classifier" / "metrics.json",
        ROOT / "data" / "classifier" / "metrics.json",
    ]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail="metrics.json not found")


@app.get("/api/graph")
def get_graph():
    return SUP


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
