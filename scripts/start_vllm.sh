#!/bin/bash
# Start local vLLM server for the configured Qwen model.
#
# Defaults target the current repo/workspace, while keeping all important paths
# configurable through environment variables.
#
# Usage:
#   bash scripts/start_vllm.sh
#   QWEN_MODEL_PATH=/abs/path/to/Qwen2.5-7B-Instruct bash scripts/start_vllm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/torchinductor}"

DEFAULT_QWEN_MODEL_PATH="$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
MODEL="${MODEL:-${QWEN_MODEL_PATH:-$DEFAULT_QWEN_MODEL_PATH}}"
SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-${QWEN_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"

if [ ! -d "$MODEL" ]; then
    echo "[ERROR] Qwen model path not found: $MODEL" >&2
    echo "[ERROR] Set QWEN_MODEL_PATH or MODEL to your local Qwen model directory." >&2
    exit 1
fi

if [ "${STOP_EXISTING_VLLM:-1}" = "1" ]; then
    echo "Checking for existing vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null && echo "Killed existing vLLM instance." || echo "No existing instance found."
    sleep 3
fi

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CC_MAJOR=$(echo "$CC" | cut -d. -f1)
if [ "$CC_MAJOR" -ge 8 ]; then
    DTYPE="${VLLM_DTYPE:-auto}"
    echo "GPU compute capability $CC - using dtype=$DTYPE"
else
    DTYPE="${VLLM_DTYPE:-half}"
    echo "GPU compute capability $CC - using dtype=$DTYPE"
fi

PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

EXTRA_ARGS=()
if [ -n "${VLLM_MAX_NUM_SEQS:-}" ]; then
    EXTRA_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if [ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]; then
    EXTRA_ARGS+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi
if [ -n "${VLLM_GENERATION_CONFIG:-vllm}" ]; then
    EXTRA_ARGS+=(--generation-config "${VLLM_GENERATION_CONFIG:-vllm}")
fi
if [ "${VLLM_DISABLE_LOG_REQUESTS:-1}" = "1" ]; then
    EXTRA_ARGS+=(--disable-log-requests)
fi

echo "Starting vLLM server on http://$HOST:$PORT"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "HF_HOME: $HF_HOME"
echo "VLLM_CACHE_ROOT: $VLLM_CACHE_ROOT"
echo "Press Ctrl+C to stop."
echo ""

vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --host "$HOST" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${EXTRA_ARGS[@]}"
