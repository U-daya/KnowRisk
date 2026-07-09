#!/usr/bin/env python3
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# Setup paths and environment variables before imports
temp_dir = tempfile.TemporaryDirectory()
cache_path = Path(temp_dir.name) / "test_cache.npz"
os.environ["KNOWRISK_CACHE_PATH"] = str(cache_path)
os.environ["LLM_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"

# Resolve REPO_ROOT and sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import TestClient and backend app
try:
    from fastapi.testclient import TestClient
except ImportError:
    print("TestClient not found, installing fastapi.testclient dependencies...")
    import subprocess
    subprocess.run(["pip", "install", "--break-system-packages", "httpx"], check=True)
    from fastapi.testclient import TestClient

from backend.app import app
import llm_agent.agent as agent

def main():
    print("=============================================================")
    print("🔍 Testing KnowRisk Semantic Caching Integration")
    print("=============================================================")
    
    # Assert cache starts empty
    assert not cache_path.exists(), "Test cache path should not exist initially"
    
    # Mock DDG search so the test never hits the network
    with patch("duckduckgo_search.DDGS.text", return_value=[]):
        # We manually trigger models initialization
        print("Initializing models for test...")
        agent.init_models()
        
        client = TestClient(app)
        
        # Test question
        payload = {
            "query": "Explain the major risk for semiconductor manufacturing single-source dependencies.",
            "component_id": "COMP-001"
        }
        
        print("\n--- Posting Query 1 (Should be mi300x or synthetic) ---")
        response1 = client.post("/api/query", json=payload)
        assert response1.status_code == 200, f"Query 1 failed: {response1.text}"
        r1 = response1.json()
        print(f"R1: {json.dumps(r1, indent=2)}")
        
        print("\n--- Posting Query 2 (Should hit semantic cache) ---")
        response2 = client.post("/api/query", json=payload)
        assert response2.status_code == 200, f"Query 2 failed: {response2.text}"
        r2 = response2.json()
        print(f"R2: {json.dumps(r2, indent=2)}")
        
        # Assertions
        print("\n--- Running Assertions ---")
        
        # 1. r2.source == "cache" (unconditional)
        print(f"Assertion 1: Is R2 source 'cache'? ({r2.get('source')})")
        assert r2.get("source") == "cache", f"Expected r2.source to be 'cache', got '{r2.get('source')}'"
        print("✅ Assertion 1 Passed!")
        
        # 2. r1.source in ("mi300x", "synthetic")
        print(f"Assertion 2: Is R1 source 'mi300x' or 'synthetic'? ({r1.get('source')})")
        assert r1.get("source") in ("mi300x", "synthetic"), f"Expected r1.source to be 'mi300x' or 'synthetic', got '{r1.get('source')}'"
        print("✅ Assertion 2 Passed!")
        
        # 3. Latency comparison
        print("Assertion 3: Latency comparison...")
        if r1.get("source") == "mi300x":
            ratio = r1.get("latency_ms", 0.0) / 10.0
            print(f"  R1 latency: {r1.get('latency_ms')} ms")
            print(f"  R2 latency: {r2.get('latency_ms')} ms (limit: {ratio:.2f} ms)")
            assert r2.get("latency_ms", 0.0) < ratio, f"Expected cache hit latency to be < 10% of real inference, got {r2.get('latency_ms')} vs {r1.get('latency_ms')}"
            print("✅ Assertion 3 Passed!")
        else:
            print(f"👉 SKIP: R1 source was '{r1.get('source')}' (synthetic fallback), not 'mi300x'. Skip latency check.")
            
        # 4. r1.text == r2.text
        print("Assertion 4: Is R1 text identical to R2 text?")
        assert r1.get("text") == r2.get("text"), "Expected r1.text to be identical to r2.text"
        print("✅ Assertion 4 Passed!")
        
        print("\n🎉 ALL TEST ASSERTIONS PASSED SUCCESSFULLY!")
        
    temp_dir.cleanup()

if __name__ == "__main__":
    main()
