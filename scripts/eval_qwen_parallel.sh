#!/bin/bash
# Run Qwen T-method eval with question-level concurrency against local vLLM.
#
# Usage:
#   bash scripts/eval_qwen_parallel.sh --samples 0
#   bash scripts/eval_qwen_parallel.sh
#
# Useful overrides:
#   EVAL_MAX_WORKERS=8
#   EMBEDDING_MODEL_PATH=/abs/path/to/Qwen3-Embedding-0.6B
#   QWEN_MODEL_PATH=/abs/path/to/Qwen2.5-7B-Instruct
#   QWEN_LLM_MODEL=Qwen/Qwen3-8B
#   QWEN_RUN_SLUG=qwen3_8b

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
    exit 1
fi

export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
export ANSWER_CONTEXT_MAX_CHARS="${ANSWER_CONTEXT_MAX_CHARS:-20000}"
export EVAL_PARALLEL_QUESTIONS=1
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-8}"
export EVAL_ENABLE_BERT_SCORE="${EVAL_ENABLE_BERT_SCORE:-0}"
export EVAL_ENABLE_SBERT_SIMILARITY="${EVAL_ENABLE_SBERT_SIMILARITY:-0}"
export EVAL_ENABLE_METEOR="${EVAL_ENABLE_METEOR:-0}"

# Reflection costs extra LLM calls per question. Keep planning on, but default
# Qwen eval to a single planned retrieval pass for throughput; override with
# ENABLE_REFLECTION=1 when running quality-focused ablations.
export ENABLE_PLANNING="${ENABLE_PLANNING:-1}"
export ENABLE_REFLECTION="${ENABLE_REFLECTION:-0}"
export MAX_REFLECTION_ROUNDS="${MAX_REFLECTION_ROUNDS:-1}"

echo "[OK] OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "[OK] QWEN_RUN_SLUG=${QWEN_RUN_SLUG:-qwen}"
echo "[OK] QWEN_LLM_MODEL=$QWEN_LLM_MODEL"
echo "[OK] EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS"
echo "[OK] EMBEDDING_MODEL_PATH=$EMBEDDING_MODEL_PATH"
echo "[OK] ANSWER_CONTEXT_MAX_CHARS=$ANSWER_CONTEXT_MAX_CHARS"
echo "[OK] ENABLE_REFLECTION=$ENABLE_REFLECTION"
echo "[OK] EVAL_ENABLE_BERT_SCORE=$EVAL_ENABLE_BERT_SCORE"
echo "[OK] EVAL_ENABLE_METEOR=$EVAL_ENABLE_METEOR"

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

cd "$ROOT_DIR"
python -u cli/ramem_eval.py --model qwen --parallel-questions --eval-workers "$EVAL_MAX_WORKERS" "$@"
