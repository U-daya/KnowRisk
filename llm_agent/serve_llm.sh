#!/usr/bin/env bash
# DEAD: the app runs the model in-process via transformers. Retained for reference.
# serve_llm.sh — Launch vLLM with an OpenAI-compatible API on ROCm
#
# Usage:
#   ./serve_llm.sh [MODEL_NAME] [PORT]
#
# Defaults:
#   MODEL  = meta-llama/Llama-3.1-70B-Instruct
#   PORT   = 8001
#
# Ungated fallbacks (use if Llama is gated/401):
#   ./serve_llm.sh Qwen/Qwen2.5-70B-Instruct 8001
#   ./serve_llm.sh Qwen/Qwen2.5-7B-Instruct 8001

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${2:-8001}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/vllm.log"

echo "=============================================="
echo " KnowRisk LLM Server"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  Log   : ${LOG_FILE}"
echo "=============================================="

# ── ROCm / HIP check ──────────────────────────────────────────────────────
if ! command -v rocm-smi &>/dev/null; then
  echo "❌ rocm-smi not found — ROCm does not appear to be installed."
  exit 1
fi

GPU_INFO=$(rocm-smi --showid 2>/dev/null || true)
echo "✅ ROCm GPU detected:"
echo "$GPU_INFO" | head -n 5

# ── vLLM install check ────────────────────────────────────────────────────
if ! python3 -c "import vllm" 2>/dev/null; then
  echo ""
  echo "📦 vLLM not found — installing (ROCm build)..."
  # Install the ROCm-compatible vLLM wheel
  pip install --break-system-packages \
    "vllm>=0.6.0" \
    2>&1 | tail -5
  echo "✅ vLLM installed"
else
  VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
  echo "✅ vLLM already installed (version: ${VLLM_VER})"
fi

# ── HuggingFace token check ───────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo ""
  echo "⚠️  HF_TOKEN is not set. Gated models (Llama) will fail with 401."
  echo "   If you hit a gated-repo error, re-run with:"
  echo "   HF_TOKEN=<your_token> ./serve_llm.sh ${MODEL} ${PORT}"
  echo "   Or switch to an ungated model:"
  echo "   ./serve_llm.sh Qwen/Qwen2.5-70B-Instruct ${PORT}"
fi

# ── Kill any existing vLLM on this port ──────────────────────────────────
if lsof -ti :"${PORT}" &>/dev/null; then
  echo ""
  echo "⚠️  Port ${PORT} already in use — killing existing process..."
  lsof -ti :"${PORT}" | xargs kill -9 2>/dev/null || true
  sleep 2
fi

# ── Launch vLLM ───────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting vLLM server..."
echo "   Model : ${MODEL}"
echo "   Port  : ${PORT}"
echo "   GPU   : all visible ROCm devices"
echo ""

# Export for child process
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

# MI300X has 192GB VRAM — use tensor parallelism across the single GPU
# max-model-len set conservatively; increase if OOM errors appear
python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  2>&1 | tee "${LOG_FILE}"
