#!/usr/bin/env python3
"""
app.py — KnowRisk FastAPI Backend
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_agent import agent as llm_agent

METRICS_PATH  = REPO_ROOT / "classifier" / "metrics.json"
SUPPLIERS_PATH = REPO_ROOT / "data" / "suppliers.json"
DIST = REPO_ROOT / "frontend-new" / "dist"

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="KnowRisk API",
    description="Semiconductor supply-chain risk analyzer powered by AMD MI300X + ROCm LLM",
    version="1.0.0",
)

# CORS configuration
env_mode = os.environ.get("KNOWRISK_ENV", "production")
if env_mode == "development":
    allow_origins = ["http://localhost:5173"]
else:
    allow_origins = []

print(f"CORS mode: {env_mode}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

@app.on_event("startup")
def startup_event():
    print("🚀 Starting KnowRisk application...")
    # Eagerly warm up and load local models
    llm_agent.init_models()
    print("✅ KnowRisk application started successfully.")

# ── Request/Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    component_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    stats = llm_agent.get_health_stats()
    gpu_available, gpu_name = llm_agent.get_gpu_info()
    data_stats = llm_agent.get_data_summary_stats()
    return {
        "status": "ok",
        "llm_model":    llm_agent.get_llm_model(),
        "gpu":          gpu_name,
        "gpu_available": gpu_available,
        "cache_hit_rate": stats["cache_hit_rate"],
        "p50_latency_ms": stats["p50_latency_ms"],
        "news_search_failures": stats["news_search_failures"],
        "news_empty_results": stats.get("news_empty_results", 0),
        "llm_loaded": stats["llm_loaded"],
        # Real supplier component counts
        "data_summary": data_stats,
    }


@app.get("/api/components")
async def list_components():
    """Return all supplier components from the graph."""
    try:
        components = llm_agent.list_components()
        # Summarize — don't include full dependency lists in list view
        return [
            {
                "id":               c["id"],
                "name":             c["name"],
                "category":         c["category"],
                "tier":             c["tier"],
                "country":          c["country"],
                "single_source":    c["single_source"],
                "export_controlled":c["export_controlled"],
                "lead_time_days":   c["lead_time_days"],
                "risk_score":       c["risk_score"],
                "risk_label":       llm_agent._risk_label(c["risk_score"]),
                "n_dependencies":   len(c.get("dependencies", [])),
            }
            for c in components
        ]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/risk/{component_id}")
async def get_risk(component_id: str):
    """
    Full risk analysis for a component:
    - ML classifier probability
    - LLM plain-language explanation
    - Dependency risk summary
    """
    try:
        return llm_agent.analyze_component(component_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/api/query")
async def query(request: QueryRequest):
    """Free-text Q&A via LLM about supply chain risk."""
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    try:
        return llm_agent.answer_query(request.query, request.component_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/api/graph")
async def get_graph():
    """Return full supplier graph for visualization."""
    if not SUPPLIERS_PATH.exists():
        raise HTTPException(status_code=503, detail="suppliers.json not found")
    with open(SUPPLIERS_PATH) as f:
        return json.load(f)


@app.get("/{full_path:path}")
async def spa(full_path: str):
    """Serve the built SPA for any non-API route (client-side routing)."""
    if full_path.startswith("api"):
        raise HTTPException(404, detail="Not found")
    return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
