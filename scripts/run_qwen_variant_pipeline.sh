#!/bin/bash
# Run one Qwen local-vLLM full pipeline from scratch:
#   memory build -> 1540 eval
#
# Usage:
#   bash scripts/run_qwen_variant_pipeline.sh qwen25_3b Qwen/Qwen2.5-3B-Instruct
#   bash scripts/run_qwen_variant_pipeline.sh qwen25_15b Qwen/Qwen2.5-1.5B-Instruct
#
# Optional overrides:
#   QWEN_MODEL_PATH=/abs/path/to/local/model
#   AUTO_DOWNLOAD=1
#   FORCE_MEMORY=1
#   EVAL_MAX_WORKERS=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SLUG="${1:?Usage: bash scripts/run_qwen_variant_pipeline.sh <slug> <hf_model_id> [model_path]}"
MODEL_ID="${2:?Usage: bash scripts/run_qwen_variant_pipeline.sh <slug> <hf_model_id> [model_path]}"
CLI_MODEL_PATH="${3:-}"

export CONDA_ENV="${CONDA_ENV:-llava}"

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
export QWEN_RUN_SLUG="$SLUG"
export QWEN_LLM_MODEL="$MODEL_ID"
export QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-$MODEL_ID}"
export QWEN_BACKBONE_LABEL="${QWEN_BACKBONE_LABEL:-${MODEL_ID##*/}}"

model_cache_dir() {
    local model_id="$1"
    printf "%s/models--%s/snapshots" "$HF_HUB_CACHE" "${model_id//\//--}"
}

latest_snapshot() {
    local snapshots_dir="$1"
    if [ -d "$snapshots_dir" ]; then
        find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -1
    fi
}

MODEL_PATH="${CLI_MODEL_PATH:-${QWEN_MODEL_PATH:-}}"
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$(latest_snapshot "$(model_cache_dir "$MODEL_ID")")"
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    if [ "${AUTO_DOWNLOAD:-0}" != "1" ]; then
        echo "[ERROR] Local model not found for $MODEL_ID" >&2
        echo "[ERROR] Set QWEN_MODEL_PATH or rerun with AUTO_DOWNLOAD=1." >&2
        exit 1
    fi

    echo "[INFO] Downloading $MODEL_ID to HF cache..."
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL_ID" --local-dir-use-symlinks False >/dev/null
    else
        python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')" >/dev/null
    fi
    MODEL_PATH="$(latest_snapshot "$(model_cache_dir "$MODEL_ID")")"
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[ERROR] Model download/cache resolution failed for $MODEL_ID" >&2
    exit 1
fi

export QWEN_MODEL_PATH="$MODEL_PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[OK] Variant slug: $QWEN_RUN_SLUG"
echo "[OK] Model id: $QWEN_LLM_MODEL"
echo "[OK] Model path: $QWEN_MODEL_PATH"

BUILD_ARGS=(--sample-idx all)
if [ "${FORCE_MEMORY:-1}" = "1" ]; then
    BUILD_ARGS+=(--force)
fi

echo ""
echo "== Building memory for $QWEN_RUN_SLUG =="
bash "$SCRIPT_DIR/build_memory_qwen.sh" "${BUILD_ARGS[@]}"

echo ""
echo "== Running eval for $QWEN_RUN_SLUG =="
bash "$SCRIPT_DIR/eval_qwen_parallel.sh"

echo ""
echo "[OK] Completed $QWEN_RUN_SLUG"
echo "[OK] Eval: $ROOT_DIR/results/locomo/${QWEN_RUN_SLUG}_eval_1540.json"
echo "[OK] Contexts: $ROOT_DIR/results/locomo_contexts/1540_${QWEN_RUN_SLUG}_t_contexts.json"
