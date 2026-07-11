#!/usr/bin/env python3
"""
agent.py — KnowRisk LLM Agent
Combines the classifier's quantitative risk score with an LLM-generated
plain-language explanation.
"""

import json
import os
import sys
import time
import subprocess
import zipfile
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set HF_HOME inside python process only
os.environ["HF_HOME"] = str(Path("/root/.cache/huggingface"))

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
SUPPLIERS     = REPO_ROOT / "data" / "suppliers.json"
MODEL_PATH    = REPO_ROOT / "classifier" / "risk_model.joblib"
FEATURES_PATH = REPO_ROOT / "classifier" / "feature_columns.json"

# Resolve cache path
cache_path_env = os.environ.get("KNOWRISK_CACHE_PATH")
if cache_path_env:
    CACHE_PATH = Path(cache_path_env)
else:
    CACHE_PATH = Path(__file__).resolve().parent / "cache.npz"

# ── Globals ────────────────────────────────────────────────────────────────
_model                = None
_features             = None
_suppliers            = None
_tokenizer            = None
_llm_model            = None
_sentence_transformer = None
_cache                = None

# Server stats
news_search_failures = 0
news_empty_results = 0
latency_samples = []
cache_hits = 0
total_queries = 0

def _update_latency_stat(latency_ms: float):
    global latency_samples
    latency_samples.append(latency_ms)
    if len(latency_samples) > 1000:
        latency_samples = latency_samples[-1000:]

# ── Semantic Cache ─────────────────────────────────────────────────────────
class SemanticCache:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.namespaces = []
        self.prompts = []
        self.embeddings = []
        self.responses = []
        self.timestamps = []
        self.news_grounded = []
        self.load()

    def load(self):
        if not self.filepath.exists():
            return
        try:
            with np.load(self.filepath, allow_pickle=True) as data:
                self.namespaces = list(data.get("namespaces", []))
                self.prompts = list(data["prompts"])
                self.embeddings = list(data["embeddings"])
                self.responses = list(data["responses"])
                self.timestamps = list(data["timestamps"])
                # Handle boolean/None/str conversion safely
                self.news_grounded = list(data["news_grounded"])
        except (FileNotFoundError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as e:
            print(f"⚠️ Cache load failed ({type(e).__name__}): {e}. Falling back to empty cache.")
            self.namespaces = []
            self.prompts = []
            self.embeddings = []
            self.responses = []
            self.timestamps = []
            self.news_grounded = []

    def save(self):
        tmp_filepath = self.filepath.with_name(self.filepath.name + ".tmp.npz")
        try:
            np.savez_compressed(
                tmp_filepath,
                namespaces=np.array(self.namespaces, dtype=object),
                prompts=np.array(self.prompts, dtype=object),
                embeddings=np.array(self.embeddings, dtype=float),
                responses=np.array(self.responses, dtype=object),
                timestamps=np.array(self.timestamps, dtype=float),
                news_grounded=np.array(self.news_grounded, dtype=object) # Use object dtype to support None values
            )
            os.replace(tmp_filepath, self.filepath)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
            if tmp_filepath.exists():
                try:
                    tmp_filepath.unlink()
                except Exception:
                    pass

    def get(self, namespace: str, prompt_text: str, q_emb: np.ndarray, ttl_hours: float) -> tuple[str | None, object]:
        if len(self.embeddings) == 0:
            return None, None
        
        # Filter indices by matching namespace exactly
        matching_indices = [i for i, ns in enumerate(self.namespaces) if ns == namespace]
        if not matching_indices:
            return None, None

        norm_q = np.linalg.norm(q_emb)
        if norm_q > 0:
            q_emb = q_emb / norm_q
            
        # Extract matching embeddings
        sub_embeddings = np.array([self.embeddings[i] for i in matching_indices], dtype=float)
        norms = np.linalg.norm(sub_embeddings, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_embeddings = sub_embeddings / norms[:, np.newaxis]
        
        similarities = np.dot(normalized_embeddings, q_emb)
        best_sub_idx = int(np.argmax(similarities))
        best_sim = similarities[best_sub_idx]
        best_idx = matching_indices[best_sub_idx]
        
        if best_sim >= 0.95:
            entry_time = self.timestamps[best_idx]
            age_hours = (time.time() - entry_time) / 3600.0
            if age_hours <= ttl_hours:
                return self.responses[best_idx], self.news_grounded[best_idx]
            else:
                print(f"Evicting stale cache entry for prompt similarity {best_sim:.4f} (age: {age_hours:.2f} hours)")
                self.namespaces.pop(best_idx)
                self.prompts.pop(best_idx)
                self.embeddings.pop(best_idx)
                self.responses.pop(best_idx)
                self.timestamps.pop(best_idx)
                self.news_grounded.pop(best_idx)
                self.save()
        return None, None

    def add(self, namespace: str, prompt_text: str, q_emb: np.ndarray, response_text: str, news_grounded: object):
        norm_q = np.linalg.norm(q_emb)
        if norm_q > 0:
            q_emb = q_emb / norm_q
        self.namespaces.append(namespace)
        self.prompts.append(prompt_text)
        self.embeddings.append(q_emb)
        self.responses.append(response_text)
        self.timestamps.append(time.time())
        self.news_grounded.append(news_grounded)
        self.save()

# ── Startup & Helpers ──────────────────────────────────────────────────────

def get_gpu_info() -> tuple[bool, str]:
    available = False
    name = "N/A"
    try:
        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return available, name

def init_models():
    """Eagerly load models at startup. Fail startup loudly if either fails."""
    global _tokenizer, _llm_model, _sentence_transformer, _cache
    
    # 1. Load SentenceTransformer (MiniLM)
    print("🚀 [Startup] Initializing MiniLM for semantic caching...")
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"✅ MiniLM loaded in {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"❌ Failed to load MiniLM: {e}")
        raise RuntimeError(f"Failed to load MiniLM: {e}") from e
        
    # 2. Load Local LLM (Qwen)
    model_id = get_llm_model()
    print(f"🚀 [Startup] Initializing local Hugging Face model: {model_id}...")
    t0 = time.time()
    try:
        _print_rocm_smi("Pre-Model Load")
        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ Local model loaded in {time.time() - t0:.2f} seconds.")
        _print_rocm_smi("Post-Model Load")
    except Exception as e:
        print(f"❌ Failed to load LLM {model_id}: {e}")
        raise RuntimeError(f"Failed to load LLM {model_id}: {e}") from e

    # 3. Warm up GPU kernels
    print("🚀 [Startup] Warming up GPU kernels...")
    t_warm = time.time()
    try:
        warmup_inputs = _tokenizer("warmup", return_tensors="pt").to(_llm_model.device)
        with torch.no_grad():
            _ = _llm_model.generate(**warmup_inputs, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        print(f"✅ GPU kernels warmed up in {time.time() - t_warm:.2f} seconds.")
    except Exception as e:
        print(f"⚠️ GPU kernel warmup failed: {e}")

    # Initialize cache
    _cache = SemanticCache(CACHE_PATH)

    # Calibrate risk thresholds based on actual data
    calibrate_risk_thresholds()

def get_health_stats() -> dict:
    global news_search_failures, news_empty_results, latency_samples, cache_hits, total_queries, _llm_model
    p50_latency = 0.0
    if latency_samples:
        p50_latency = float(np.median(latency_samples))
    cache_hit_rate = 0.0
    if total_queries > 0:
        cache_hit_rate = float(cache_hits) / total_queries
    return {
        "news_search_failures": news_search_failures,
        "news_empty_results": news_empty_results,
        "p50_latency_ms": round(p50_latency, 2),
        "cache_hit_rate": round(cache_hit_rate, 4),
        "llm_loaded": _llm_model is not None
    }

def clean_response_text(text: str) -> str:
    if not text:
        return ""
    # Remove bold/italic markup
    text = re.sub(r"\*\*|__", "", text)
    # Remove headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Split into lines and strip list/numbered markers
    lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        line_cleaned = re.sub(r"^[\-\*\+]\s+", "", line_stripped)
        line_cleaned = re.sub(r"^\d+\.\s+", "", line_cleaned)
        if line_cleaned:
            lines.append(line_cleaned)
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined).strip()
    # Strip "Word:" or "Two Words:" label-colon patterns that survive bullet removal.
    # These are artifacts of the model producing structured lists as prose.
    # Use a capturing group for the boundary (sentence end or start of string)
    # because Python re requires fixed-width lookbehind.
    joined = re.sub(r"([.!?]\s+|^)[A-Z][a-z]+(?: [A-Z][a-z]+)?:\s+", r"\1", joined)
    return joined

def truncate_to_last_sentence(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    matches = list(re.finditer(r'[.!?](?=\s|$)', text))
    if not matches:
        return text
    return text[:matches[-1].end()].strip()

def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

def get_cache_ttl_hours() -> float:
    try:
        return float(os.environ.get("KNOWRISK_CACHE_TTL_HOURS", "6.0"))
    except ValueError:
        return 6.0

def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(CACHE_PATH)
    return _cache

def _get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer

def _get_llm():
    global _tokenizer, _llm_model
    if _llm_model is None or _tokenizer is None:
        raise RuntimeError("LLM not initialized! Call init_models() first.")
    return _tokenizer, _llm_model

def get_data_summary_stats() -> dict:
    try:
        suppliers = _load_suppliers()
        components = list(suppliers.values())
        count = len(components)
        single_source = sum(1 for c in components if c.get("single_source", False))
        export_ctrl = sum(1 for c in components if c.get("export_controlled", False))
        lead_times = [c.get("lead_time_days", 0) for c in components]
        median_lead = 0
        if lead_times:
            median_lead = int(np.median(lead_times))
        return {
            "components_count": count,
            "single_source_count": single_source,
            "export_controlled_count": export_ctrl,
            "median_lead_time": median_lead
        }
    except Exception as e:
        print(f"⚠️ Error computing data stats: {e}")
        return {
            "components_count": 0,
            "single_source_count": 0,
            "export_controlled_count": 0,
            "median_lead_time": 0
        }

def _load_suppliers() -> dict:
    global _suppliers
    if _suppliers is None:
        if not SUPPLIERS.exists():
            raise FileNotFoundError(f"Supplier graph not found: {SUPPLIERS}. Run data/generate_synthetic_data.py first.")
        with open(SUPPLIERS) as f:
            data = json.load(f)
        _suppliers = {c["id"]: c for c in data["components"]}
    return _suppliers

_dependents_map: dict[str, list[str]] | None = None

def _load_dependents_map() -> dict[str, list[str]]:
    """
    Reverse of each component's own `dependencies` list: for a given
    component ID, which OTHER components list it as a dependency (i.e. who
    consumes it downstream). Built once from the full supplier graph and
    cached, since it requires scanning every component's dependency list —
    not something we want to redo on every /api/risk/{id} request.
    """
    global _dependents_map
    if _dependents_map is None:
        suppliers = _load_suppliers()
        reverse: dict[str, list[str]] = {cid: [] for cid in suppliers}
        for cid, comp in suppliers.items():
            for dep_id in comp.get("dependencies", []):
                if dep_id in reverse:
                    reverse[dep_id].append(cid)
        _dependents_map = reverse
    return _dependents_map

def _print_rocm_smi(stage_name: str):
    """Print ROCm GPU status and utilization statistics."""
    print(f"\n⚡ [AMD Instinct MI300X] ROCm-SMI GPU Status at: {stage_name} ⚡")
    try:
        res = subprocess.run(["rocm-smi"], capture_output=True, text=True)
        print(res.stdout.strip())
    except Exception as e:
        print(f"rocm-smi execution failed: {e}")
    print("=" * 60)

# ── Feature extraction for a supplier component ───────────────────────────
COUNTRY_GEO_RISK = {
    "Taiwan": 0.75, "South Korea": 0.40, "Japan": 0.25,
    "Netherlands": 0.15, "USA": 0.10, "China": 0.65,
    "Germany": 0.10, "Malaysia": 0.30, "Vietnam": 0.35, "Israel": 0.45,
}

_risk_threshold_critical = 0.5298
_risk_threshold_high     = 0.3959
_risk_threshold_medium   = 0.1457

def calibrate_risk_thresholds():
    """
    Calibrate risk thresholds using quantiles over suppliers data.
    These are rank-based (top 10% critical, next 20% high, next 30% medium, bottom 40% low)
    rather than absolute severity values.
    """
    global _risk_threshold_critical, _risk_threshold_high, _risk_threshold_medium
    try:
        suppliers = _load_suppliers()
        scores = sorted([c.get("risk_score", 0.0) for c in suppliers.values()])
        if not scores:
            return
        n = len(scores)
        idx_crit = max(0, n - 5)
        idx_high = max(0, n - 15)
        idx_med = max(0, n - 30)
        
        _risk_threshold_critical = scores[idx_crit]
        _risk_threshold_high = scores[idx_high]
        _risk_threshold_medium = scores[idx_med]
        
        print(f"📊 [Calibration] Recalibrated risk thresholds based on {n} components:")
        print(f"   - CRITICAL (top 10%) >= {_risk_threshold_critical:.4f}")
        print(f"   - HIGH (next 20%)    >= {_risk_threshold_high:.4f}")
        print(f"   - MEDIUM (next 30%)  >= {_risk_threshold_medium:.4f}")
        print(f"   - LOW (bottom 40%)   <  {_risk_threshold_medium:.4f}")
    except Exception as e:
        print(f"⚠️ Risk threshold calibration failed: {e}")

def _risk_label(score: float) -> str:
    global _risk_threshold_critical, _risk_threshold_high, _risk_threshold_medium
    if score >= _risk_threshold_critical:
        return "CRITICAL"
    elif score >= _risk_threshold_high:
        return "HIGH"
    elif score >= _risk_threshold_medium:
        return "MEDIUM"
    else:
        return "LOW"

# ── Shared news search helper ──────────────────────────────────────────────

def _ddg_search(search_query: str, timeout: float = 1.5) -> list[str]:
    """
    Run a DuckDuckGo search in a daemon thread and return filtered headlines.
    Returns [] immediately if the deadline passes — the worker thread is not
    joined; it runs to completion in the background and is discarded.

    The socket-level timeout on the DDGS client ensures the underlying HTTP
    request dies at the source (connect + read) rather than merely being
    abandoned at the future level while the TCP teardown blocks.

    Increments news_search_failures on exception, news_empty_results when
    DDG returns results that all fail filters. Caller reads the return value
    to determine news_grounded.
    """
    global news_search_failures, news_empty_results
    from ddgs import DDGS

    def _fetch():
        # timeout= sets httpx connect+read timeout so the socket dies at source
        with DDGS(timeout=timeout) as ddgs:
            return list(ddgs.text(search_query, max_results=3))

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch)
    # Shut down the executor immediately without waiting for the thread.
    executor.shutdown(wait=False)
    try:
        results = future.result(timeout=timeout)
    except Exception as e:
        news_search_failures += 1
        print(f"⚠️ News search failed/timed out: {e}")
        return []

    headlines = []
    for r in results:
        title = r.get("title", "").strip()
        href = r.get("href", "").strip().lower()
        if href.endswith(".pdf"):
            continue
        if len(title) < 20:
            continue
        headlines.append(title)
    headlines = headlines[:3]
    if not headlines and results:
        # DDG returned rows but all were filtered out
        news_empty_results += 1
    elif not results:
        news_empty_results += 1
    return headlines

_EXPLANATION_LABEL_RE = re.compile(
    r"RISK_FACTOR:\s*(.*?)\s*SCENARIO:\s*(.*?)\s*MITIGATION:\s*(.*)",
    re.S | re.I,
)

def _parse_structured_explanation(raw: str) -> dict[str, str] | None:
    """
    Parse RAW model output (before clean_response_text) for three labeled
    fields via a single regex anchored on all three labels. Must run on the
    raw blob, not on cleaned text — clean_response_text strips "Label:"
    patterns, which destroys the very markers this parse depends on.

    Returns None on regex match failure (the caller's retry trigger), or a
    dict with keys risk_factor, scenario, mitigation, each individually
    passed through clean_response_text.
    """
    match = _EXPLANATION_LABEL_RE.search(raw)
    if not match:
        return None
    risk_factor, scenario, mitigation = match.groups()
    return {
        "risk_factor": clean_response_text(risk_factor),
        "scenario":    clean_response_text(scenario),
        "mitigation":  clean_response_text(mitigation),
    }


def _generate_explanation(component: dict, risk_score: float, risk_label: str) -> dict:
    """
    Generate a structured risk explanation with three named fields:
    risk_factor, scenario, mitigation.
    Uses semantic cache; on miss, runs LLM with a labeled-output prompt and
    parses the result. Retries once if any field is missing. Falls back to
    putting the full text in risk_factor on persistent failure.
    """
    global total_queries, cache_hits, news_search_failures, news_empty_results
    import json as _json
    total_queries += 1
    t0 = time.time()
    news_grounded = False

    geo_risk = COUNTRY_GEO_RISK.get(component.get("country", ""), 0.35)
    prompt = f"""Component Facts:
- Name: {component['name']}
- Category: {component['category']}
- Tier: {component['tier']} (1=most critical)
- Country of origin: {component['country']}
- Single-source supplier: {'Yes' if component['single_source'] else 'No'}
- Export controlled: {'Yes' if component['export_controlled'] else 'No'}
- Supply lead time: {component['lead_time_days']} days
- Dependencies: {len(component.get('dependencies', []))} upstream components

Key qualitative risk drivers:
- Geographic concentration: {geo_risk:.2f}
- Single-source vulnerability: {'Yes' if component['single_source'] else 'No'}
- Export control exposure: {'Yes' if component['export_controlled'] else 'No'}
- Lead time pressure: {component['lead_time_days']} days

Respond with EXACTLY three labeled lines in this format:
RISK_FACTOR: <one complete sentence, 15-25 words, on the primary qualitative risk driver>
SCENARIO: <one complete sentence, 15-25 words, describing a plausible disruption scenario>
MITIGATION: <one complete sentence, 15-25 words, recommending a concrete mitigation action>"""

    # Namespace by component ID
    namespace = component["id"]

    # 1. Semantic cache lookup — responses are JSON-encoded structured dicts
    try:
        cache = _get_cache()
        transformer = _get_sentence_transformer()
        q_emb = transformer.encode(prompt)
        ttl = get_cache_ttl_hours()
        cached_res, cached_news = cache.get(namespace, prompt, q_emb, ttl)
        if cached_res is not None:
            cache_hits += 1
            latency_ms = round((time.time() - t0) * 1000.0, 2)
            # cached_res may be a JSON dict string (new format) or plain text (old)
            try:
                fields = _json.loads(cached_res)
                if not isinstance(fields, dict):
                    raise ValueError
                fields.setdefault("parse_failed", False)
            except (ValueError, TypeError):
                fields = {"risk_factor": cached_res, "scenario": "", "mitigation": "", "parse_failed": True}
            return {**fields, "source": "cache", "latency_ms": latency_ms, "news_grounded": cached_news}
    except Exception as e:
        print(f"⚠️ Cache read error: {e}")

    # 2. Live news search
    news_context = ""
    search_query = f"{component.get('country', '')} {component.get('name', '')} semiconductor supply chain news"
    headlines = _ddg_search(search_query)
    if headlines:
        news_context = "\nRecent News Context:\n" + "\n".join(f"- {h}" for h in headlines) + "\n"
        news_grounded = True
    else:
        news_grounded = False

    system_msg = (
        "You are a semiconductor supply-chain risk analyst. "
        "Respond with EXACTLY three labeled lines as instructed. "
        "Each line must start with its label (RISK_FACTOR:, SCENARIO:, MITIGATION:) followed by exactly one "
        "complete sentence of 15-25 words — a full clause with a subject and verb, never a short noun phrase "
        "or fragment. Example of the required shape:\n"
        "RISK_FACTOR: Heavy reliance on a single Taiwanese foundry leaves this component exposed to regional "
        "political and natural disaster disruptions.\n"
        "SCENARIO: A cross-strait conflict or major earthquake could halt fabrication for months, stalling "
        "downstream production across multiple product lines.\n"
        "MITIGATION: Qualify a second-source foundry in a different region and begin building buffer inventory "
        "to cover a six-month supply gap.\n"
        "Do not use markdown, bold, lists, or headers. Do not reference numeric risk scores."
    )
    if news_context:
        system_msg += f"\nGround your response in the following real-time news:\n{news_context}"

    # 3. LLM call — with one retry if any field is missing
    def _run_inference() -> str:
        tokenizer, model = _get_llm()
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        _print_rocm_smi("Start of Explanation Generation")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=160,
                do_sample=False,
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        _print_rocm_smi("End of Explanation Generation")
        return raw

    def _fields_score(f: dict[str, str] | None) -> int:
        return sum(1 for v in (f or {}).values() if v)

    try:
        import json as _json
        raw = _run_inference()
        fields = _parse_structured_explanation(raw)

        # Retry once if the regex failed to match or any field came back empty
        if fields is None or not all(fields.values()):
            print("⚠️ Structured parse incomplete, retrying inference once")
            raw2 = _run_inference()
            fields2 = _parse_structured_explanation(raw2)
            if _fields_score(fields2) > _fields_score(fields):
                fields, raw = fields2, raw2

        parse_failed = fields is None or not all(fields.values())
        if parse_failed:
            fields = {
                "risk_factor": clean_response_text(raw),
                "scenario": "",
                "mitigation": "",
            }

        try:
            cache.add(namespace, prompt, q_emb, _json.dumps({**fields, "parse_failed": parse_failed}), news_grounded)
        except Exception as e:
            print(f"⚠️ Failed to add to cache: {e}")

        latency_ms = round((time.time() - t0) * 1000.0, 2)
        _update_latency_stat(latency_ms)
        return {**fields, "parse_failed": parse_failed, "source": "mi300x", "latency_ms": latency_ms, "news_grounded": news_grounded}

    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000.0, 2)
        _update_latency_stat(latency_ms)
        return {
            "risk_factor": f"[LLM inference failed: {e}]",
            "scenario": "",
            "mitigation": "",
            "parse_failed": True,
            "source": "synthetic",
            "latency_ms": latency_ms,
            "news_grounded": False,
        }

# ── Public API ────────────────────────────────────────────────────────────

def analyze_component(component_id: str) -> dict:
    """
    Full risk analysis for a single component.
    """
    suppliers   = _load_suppliers()

    if component_id not in suppliers:
        raise ValueError(f"Unknown component: {component_id}. "
                         f"Valid IDs: {list(suppliers.keys())[:5]}...")

    component = suppliers[component_id]

    final_score  = round(component.get("risk_score", 0.0), 4)
    risk_label   = _risk_label(final_score)

    explanation = _generate_explanation(component, final_score, risk_label)

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

    dependents_map = _load_dependents_map()
    dependent_risks = []
    for dependent_id in dependents_map.get(component_id, []):
        dependent = suppliers[dependent_id]
        dependent_risks.append({
            "id":          dependent_id,
            "name":        dependent["name"],
            "country":     dependent["country"],
            "risk_score":  dependent.get("risk_score", 0.0),
            "single_source": dependent.get("single_source", False),
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
        "risk_label":      risk_label,
        "llm_explanation": explanation,
        "dependency_risks": dep_risks,
        "dependent_risks": dependent_risks,
    }

def answer_query(query: str, context_component_id: str | None = None) -> dict:
    """
    Free-text Q&A about supply-chain risk.
    """
    global total_queries, cache_hits, news_search_failures, news_empty_results
    total_queries += 1
    t0 = time.time()
    news_grounded = None  # None indicates search not attempted (default)
    
    system_msg = (
        "You are a semiconductor supply-chain risk analyst. "
        "Answer in three to four complete sentences of plain prose. "
        "Do not use markdown, bold text, lists, bullets, headers, or labels followed by colons. "
        "Do not begin a sentence with a topic label like 'Single Source:' or 'Export Controls:'. "
        "Write continuous prose only."
    )
    
    context = ""
    component = None
    if context_component_id:
        try:
            suppliers = _load_suppliers()
            if context_component_id in suppliers:
                component = suppliers[context_component_id]
                context = (
                    f"\nContext — Currently viewing component: {component['name']} "
                    f"(Tier {component['tier']}, {component['country']}, "
                    f"single_source={'Yes' if component.get('single_source') else 'No'}, "
                    f"export_controlled={'Yes' if component.get('export_controlled') else 'No'}, "
                    f"lead_time_days={component.get('lead_time_days')}d)\n"
                )
        except Exception:
            pass

    full_prompt = context + query
    namespace = context_component_id if context_component_id else "global"

    # 1. Semantic cache lookup
    try:
        cache = _get_cache()
        transformer = _get_sentence_transformer()
        q_emb = transformer.encode(full_prompt)
        ttl = get_cache_ttl_hours()
        cached_res, cached_news = cache.get(namespace, full_prompt, q_emb, ttl)
        if cached_res is not None:
            cache_hits += 1
            latency_ms = round((time.time() - t0) * 1000.0, 2)
            return {
                "text": cached_res,
                "source": "cache",
                "latency_ms": latency_ms,
                "news_grounded": cached_news
            }
    except Exception as e:
        print(f"⚠️ Cache read error: {e}")

    # 2. News search
    news_context = ""
    if component:
        news_grounded = False
        search_query = f"{component.get('country', '')} {component.get('name', '')} semiconductor supply chain news"
        headlines = _ddg_search(search_query)
        if headlines:
            news_context = "\nRecent News Context:\n" + "\n".join(f"- {h}" for h in headlines) + "\n"
            news_grounded = True

    if news_context:
        system_msg += f"\nGround your response in the following real-time news:\n{news_context}"

    # 3. LLM call
    try:
        tokenizer, model = _get_llm()
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": full_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        _print_rocm_smi("Start of Q&A Generation")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=160,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        _print_rocm_smi("End of Q&A Generation")
        
        cleaned_text = clean_response_text(response_text)
        cleaned_text = truncate_to_last_sentence(cleaned_text)
        
        try:
            cache.add(namespace, full_prompt, q_emb, cleaned_text, news_grounded)
        except Exception as e:
            print(f"⚠️ Failed to add to cache: {e}")

        latency_ms = round((time.time() - t0) * 1000.0, 2)
        _update_latency_stat(latency_ms)
        
        return {
            "text": cleaned_text,
            "source": "mi300x",
            "latency_ms": latency_ms,
            "news_grounded": news_grounded
        }
    except Exception as e:
        fallback_text = f"Supply chain analysis for {component['name'] if component else 'semiconductor supply chain'} is currently offline."
        cleaned_fallback = clean_response_text(fallback_text)
        cleaned_fallback = truncate_to_last_sentence(cleaned_fallback)
        latency_ms = round((time.time() - t0) * 1000.0, 2)
        _update_latency_stat(latency_ms)
        return {
            "text": f"[LLM query failed: {e}] " + cleaned_fallback,
            "source": "synthetic",
            "latency_ms": latency_ms,
            "news_grounded": False
        }

def list_components() -> list[dict]:
    suppliers = _load_suppliers()
    return list(suppliers.values())

if __name__ == "__main__":
    component_id = sys.argv[1] if len(sys.argv) > 1 else "COMP-001"
    print(f"Analyzing {component_id}...")
    init_models()
    result = analyze_component(component_id)
    print(json.dumps(result, indent=2))