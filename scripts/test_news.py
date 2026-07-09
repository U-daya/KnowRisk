#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from ddgs import DDGS
except ImportError:
    print("ddgs not installed.")
    sys.exit(1)

def test_real_search():
    print("=============================================================")
    print("📰 Testing Live DuckDuckGo News Search for COMP-001")
    print("=============================================================")
    
    country = "Taiwan"
    name = "Advanced Logic Die"
    query = f"{country} {name} semiconductor supply chain news"
    print(f"Query: '{query}'")
    
    t0 = time.time()
    results = None
    error = None
    timed_out = False
    
    def _run_search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=3))
            
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_search)
            results = future.result(timeout=3.0)
    except TimeoutError:
        timed_out = True
    except Exception as e:
        error = e
        
    elapsed_ms = (time.time() - t0) * 1000.0
    
    print("\n--- Search Results ---")
    print(f"Elapsed Time: {elapsed_ms:.2f} ms")
    if timed_out:
        print("Status: TIMED OUT (after 3.0 seconds)")
    elif error:
        print(f"Status: RAISED EXCEPTION: {type(error).__name__}: {error}")
    elif results is None or len(results) == 0:
        print("Status: EMPTY RESULTS (Fast empty is a rate-limit/deprecation signature)")
    else:
        print("Status: RETURNED SUCCESSFUL")
        print(f"Number of results: {len(results)}")
        print("\nTitles retrieved:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.get('title')} ({r.get('href')})")

if __name__ == "__main__":
    test_real_search()
