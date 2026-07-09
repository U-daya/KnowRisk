#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Set HF_HOME inside the python process only
os.environ["HF_HOME"] = str(Path("/root/.cache/huggingface"))

def main():
    print("🚀 Warming up models ahead of time...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}. Please ensure requirements are installed.")
        sys.exit(1)
    
    # 1. Warm up MiniLM
    print("Downloading all-MiniLM-L6-v2...")
    try:
        SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ MiniLM warmed up successfully.")
    except Exception as e:
        print(f"❌ Failed to warm up MiniLM: {e}")
        sys.exit(1)
        
    # 2. Warm up LLM
    model_id = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Downloading {model_id}...")
    try:
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print(f"✅ {model_id} warmed up successfully.")
    except Exception as e:
        print(f"❌ Failed to warm up LLM {model_id}: {e}")
        sys.exit(1)
        
    print("🎉 All models warmed up successfully!")

if __name__ == "__main__":
    main()
