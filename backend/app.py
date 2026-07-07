#!/usr/bin/env python3
"""
app.py — KnowRisk FastAPI Backend

Environment variables required:
  LLM_BASE_URL   e.g. http://localhost:8001/v1
  LLM_MODEL      e.g. Qwen/Qwen2.5-70B-Instruct

Run:
  LLM_BASE_URL=http://localhost:8001/v1 LLM_MODEL=Qwen/Qwen2.5-70B-Instruct \
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_agent import agent as llm_agent

FRONTEND_HTML = REPO_ROOT / "frontend" / "index.html"
METRICS_PATH  = REPO_ROOT / "classifier" / "metrics.json"
SUPPLIERS_PATH = REPO_ROOT / "data" / "suppliers.json"

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="KnowRisk API",
    description="Semiconductor supply-chain risk analyzer powered by AMD MI300X + ROCm LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    component_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the static frontend dashboard."""
    if FRONTEND_HTML.exists():
        return FileResponse(str(FRONTEND_HTML), media_type="text/html")
    return JSONResponse(
        {"error": "frontend/index.html not found"},
        status_code=404,
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "llm_base_url": os.environ.get("LLM_BASE_URL", "NOT SET"),
        "llm_model":    os.environ.get("LLM_MODEL",    "NOT SET"),
        "gpu":          "AMD MI300X (ROCm)",
    }


@app.get("/api/model-metrics")
async def model_metrics():
    """Return classifier training metrics."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found — run classifier/train_risk_model.py first",
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


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
        t0 = time.time()
        result = llm_agent.analyze_component(component_id)
        result["analysis_time_seconds"] = round(time.time() - t0, 2)
        return result
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
        answer = llm_agent.answer_query(request.query, request.component_id)
        return {
            "query":        request.query,
            "answer":       answer,
            "model":        llm_agent.get_llm_model(),
            "component_id": request.component_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/api/graph")
async def get_graph():
    """Return full supplier graph for visualization."""
    if not SUPPLIERS_PATH.exists():
        raise HTTPException(status_code=503, detail="suppliers.json not found")
    with open(SUPPLIERS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
