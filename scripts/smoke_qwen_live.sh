#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke_common.sh"

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/smoke_qwen_live.sh" >&2
  exit 1
fi

ROOT_DIR="$(smoke_repo_root)"
cd "$ROOT_DIR"
smoke_activate_venv

TMP_DIR="$(mktemp -d)"
LOG_PATH="$TMP_DIR/qwen_vllm.log"
trap 'bash scripts/stop_vllm.sh >/dev/null 2>&1 || true; rm -rf "$TMP_DIR"' EXIT

export OPENAI_API_KEY="${OPENAI_API_KEY:-vllm-local}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-$OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-$OPENAI_BASE_URL}"
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$ROOT_DIR/models/qwen3-embedding-0.6b}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
export ANSWER_CONTEXT_MAX_CHARS="${ANSWER_CONTEXT_MAX_CHARS:-20000}"

echo "[smoke_qwen_live] starting Qwen vLLM ..."
bash scripts/start_vllm.sh >"$LOG_PATH" 2>&1 &
sleep 2

smoke_wait_for_models "$OPENAI_BASE_URL" 600 >/dev/null
smoke_unset_offline_eval_env
smoke_bert_score_check

echo "[smoke_qwen_live] rebuilding sample 0"
python cli/build_memory.py --model qwen --sample-idx 0 --force

python scripts/smoke_live.py run --model qwen --sample-idx 0

SUBSET_GT="$TMP_DIR/gt_memory_verification_qwen_sample0.json"
smoke_subset_gt "db/qwen/lancedb_data_frozen_qwen_sample" 0 "$SUBSET_GT"

EVAL_SCRIPT="cli/ramem_eval.py"
EVAL_JSON="SSS_results/qwen_eval_s0.json"
CONTEXT_JSON="gt_context_data/1540_qwen_t_contexts_s0.json"

python "$EVAL_SCRIPT" --model qwen --samples 0 --gt-json "$SUBSET_GT" --allow-subset-gt
python scripts/smoke_live.py validate \
  --eval-json "$EVAL_JSON" \
  --context-json "$CONTEXT_JSON" \
  --sample-idx 0

echo "[smoke_qwen_live] PASS"
