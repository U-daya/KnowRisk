#!/usr/bin/env python3
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup environment before imports to ensure isolated cache
temp_dir = tempfile.TemporaryDirectory()
cache_path = Path(temp_dir.name) / "test_collision_cache.npz"
os.environ["KNOWRISK_CACHE_PATH"] = str(cache_path)

# Resolve REPO_ROOT and sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Mock causal model and tokenizer calls during import to avoid GPU usage entirely
sys.modules['transformers.AutoTokenizer'] = MagicMock()
sys.modules['transformers.AutoModelForCausalLM'] = MagicMock()

with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_causal_lm, \
     patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
    
    mock_causal_lm.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()

    from backend.app import app
    import llm_agent.agent as agent

from fastapi.testclient import TestClient

def main():
    print("=============================================================")
    print("🔍 Testing Cache Collision & Prompt Semantic Similarity")
    print("=============================================================")

    # Setup mocks
    mock_tok = MagicMock()
    mock_model = MagicMock()
    
    mock_tok.apply_chat_template.return_value = "chat template text"
    
    # Generate unique output text per call to ensure we can assert fresh generation
    unique_counter = 0
    def mock_batch_decode(*args, **kwargs):
        nonlocal unique_counter
        unique_counter += 1
        return [f"Unique response {unique_counter} for risk assessment"]
        
    mock_tok.batch_decode = mock_batch_decode

    # Initialize MiniLM (local sentence transformer) but mock out the causal LM to run instantly
    with patch("duckduckgo_search.DDGS.text", return_value=[]), \
         patch("llm_agent.agent._get_llm", return_value=(mock_tok, mock_model)), \
         patch("transformers.AutoModelForCausalLM.from_pretrained"), \
         patch("transformers.AutoTokenizer.from_pretrained"):
         
        print("Initializing models...")
        agent.init_models()

        client = TestClient(app)

        print("\n--- GET /api/risk/COMP-001 ---")
        response1 = client.get("/api/risk/COMP-001")
        assert response1.status_code == 200, f"COMP-001 failed: {response1.text}"
        r1 = response1.json()
        print(f"R1: Source={r1['llm_explanation']['source']}, Text='{r1['llm_explanation']['text']}'")

        print("\n--- GET /api/risk/COMP-017 ---")
        response2 = client.get("/api/risk/COMP-017")
        assert response2.status_code == 200, f"COMP-017 failed: {response2.text}"
        r2 = response2.json()
        print(f"R2: Source={r2['llm_explanation']['source']}, Text='{r2['llm_explanation']['text']}'")

        # Extract prompt texts from cache structure to calculate similarity
        assert len(agent._cache.prompts) >= 2, "Expected at least 2 entries in cache prompts"
        p1 = agent._cache.prompts[0]
        p2 = agent._cache.prompts[1]
        
        # Calculate raw cosine similarity between prompt text embeddings using the MiniLM model
        emb1 = agent._sentence_transformer.encode(p1)
        emb2 = agent._sentence_transformer.encode(p2)
        
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"\n👉 Assembled Prompt 1 (COMP-001 length: {len(p1)}):\n[TRUNCATED PROMPT]")
        print(f"👉 Assembled Prompt 2 (COMP-017 length: {len(p2)}):\n[TRUNCATED PROMPT]")
        print(f"\n🔥 Raw Cosine Similarity between prompt embeddings: {cos_sim:.6f}")

        # Assertions
        print("\n--- Running Collision Assertions ---")
        print(f"Asserting R2 source is NOT 'cache' (i.e. is 'mi300x'): {r2['llm_explanation']['source']}")
        assert r2["llm_explanation"]["source"] == "mi300x", f"Collision occurred! Got source '{r2['llm_explanation']['source']}'"
        print("✅ Assertion Passed! Cache partition resolved the collision successfully.")
        
        print("\n🎉 ALL COLLISION TEST ASSERTIONS PASSED SUCCESSFULLY!")

    temp_dir.cleanup()

if __name__ == "__main__":
    main()
