#!/bin/bash
# Build Qwen frozen memory DBs with local vLLM + local Qwen3 embeddings.
#
# Usage:
#   bash scripts/build_memory_qwen.sh --sample-idx all --force
#   bash scripts/build_memory_qwen.sh --sample-idx 0 --force
#
# Useful overrides:
#   CONDA_ENV=llava
#   QWEN_MODEL_PATH=/abs/path/to/Qwen2.5-7B-Instruct
#   QWEN_LLM_MODEL=Qwen/Qwen3-8B
#   QWEN_RUN_SLUG=qwen3_8b
#   EMBEDDING_MODEL_PATH=/abs/path/to/Qwen3-Embedding-0.6B
#   EMBEDDING_DEVICE=auto EMBEDDING_BATCH_SIZE=32
#   MAX_PARALLEL_WORKERS=16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_BASE_URL="${OPENAI_BASE_URL:-http://localhost:${VLLM_PORT:-8000}/v1}"
VLLM_URL="${VLLM_BASE_URL%/}/models"

activate_runtime() {
    if [ -n "${VENV_PATH:-}" ]; then
        source "$VENV_PATH/bin/activate"
        return
    fi

    local env_name="${CONDA_ENV:-llava}"
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env_name"
        return
    fi

    echo "[WARN] Neither VENV_PATH nor conda is available; using current shell." >&2
}

activate_runtime

export OPENAI_API_KEY="${OPENAI_API_KEY:-vllm-local}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-$OPENAI_API_KEY}"
export OPENAI_BASE_URL="$VLLM_BASE_URL"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-$OPENAI_BASE_URL}"
export USE_STREAMING="${USE_STREAMING:-0}"
export QWEN_LLM_MODEL="${QWEN_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-$QWEN_LLM_MODEL}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DEFAULT_EMBEDDING_MODEL_PATH="$ROOT_DIR/models/qwen3-embedding-0.6b"
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$DEFAULT_EMBEDDING_MODEL_PATH}"
if [ ! -f "$EMBEDDING_MODEL_PATH/config.json" ]; then
    echo "[ERROR] EMBEDDING_MODEL_PATH is not a complete model directory: $EMBEDDING_MODEL_PATH" >&2
    echo "[ERROR] Expected config.json. Set EMBEDDING_MODEL_PATH or run the embedding download first." >&2
    exit 1
fi

# On A100, keeping Qwen2.5 in vLLM and Qwen3-0.6B embedding on GPU fits easily.
# Override to cpu if you need maximum vLLM memory for another run.
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"

# These are read by config.py at import time when present.
export MAX_PARALLEL_WORKERS="${MAX_PARALLEL_WORKERS:-16}"
export ENABLE_PARALLEL_PROCESSING="${ENABLE_PARALLEL_PROCESSING:-1}"
export WINDOW_SIZE="${WINDOW_SIZE:-25}"
export OVERLAP_SIZE="${OVERLAP_SIZE:-2}"
export MEMORY_EXTRACTION_MAX_TOKENS="${MEMORY_EXTRACTION_MAX_TOKENS:-3072}"

echo "[OK] OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "[OK] QWEN_RUN_SLUG=${QWEN_RUN_SLUG:-qwen}"
echo "[OK] QWEN_LLM_MODEL=$QWEN_LLM_MODEL"
echo "[OK] USE_STREAMING=$USE_STREAMING"
echo "[OK] EMBEDDING_MODEL_PATH=$EMBEDDING_MODEL_PATH"
echo "[OK] EMBEDDING_DEVICE=$EMBEDDING_DEVICE"
echo "[OK] EMBEDDING_BATCH_SIZE=$EMBEDDING_BATCH_SIZE"
echo "[OK] WINDOW_SIZE=$WINDOW_SIZE"
echo "[OK] MEMORY_EXTRACTION_MAX_TOKENS=$MEMORY_EXTRACTION_MAX_TOKENS"

STARTED_VLLM=0
vllm_serves_requested_model() {
    local model_json
    model_json="$(curl -fsS "$VLLM_URL" 2>/dev/null)" || return 1
    python -c 'import json, sys; data=json.loads(sys.argv[1]); target=sys.argv[2]; ids=[m.get("id") for m in data.get("data", [])]; sys.exit(0 if target in ids else 2)' "$model_json" "$QWEN_SERVED_MODEL_NAME"
}

if vllm_serves_requested_model; then
    echo "[OK] vLLM is already serving $QWEN_SERVED_MODEL_NAME at $VLLM_BASE_URL"
else
    if curl -fsS "$VLLM_URL" >/dev/null 2>&1; then
        echo "[INFO] vLLM is running but not serving $QWEN_SERVED_MODEL_NAME; restarting server..."
    else
        echo "[INFO] vLLM not detected; starting server..."
    fi
    bash "$SCRIPT_DIR/start_vllm.sh" &
    VLLM_PID=$!
    STARTED_VLLM=1

    cleanup() {
        if [ "$STARTED_VLLM" = "1" ]; then
            echo "[INFO] Stopping vLLM PID $VLLM_PID"
            kill "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    echo "[INFO] Waiting for vLLM to be ready..."
    WAIT=0
    until curl -fsS "$VLLM_URL" >/dev/null 2>&1; do
        sleep 5
        WAIT=$((WAIT + 5))
        echo "  ... waited ${WAIT}s"
        if [ "$WAIT" -ge "${VLLM_STARTUP_TIMEOUT:-600}" ]; then
            echo "[ERROR] vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT:-600}s." >&2
            exit 1
        fi
    done
    echo "[OK] vLLM is ready."
fi

echo ""
echo "Starting Qwen memory build..."
echo ""

cd "$ROOT_DIR"
python -u cli/build_memory.py --model qwen "$@"
