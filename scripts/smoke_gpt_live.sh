#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke_common.sh"

OPENAI_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openai-model)
      OPENAI_MODEL="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: bash scripts/smoke_gpt_live.sh --openai-model {gpt-4o-mini|gpt-4.1-mini}" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OPENAI_MODEL" ]]; then
  echo "[smoke_gpt_live] --openai-model is required" >&2
  exit 1
fi

if [[ "$OPENAI_MODEL" != "gpt-4o-mini" && "$OPENAI_MODEL" != "gpt-4.1-mini" ]]; then
  echo "[smoke_gpt_live] unsupported --openai-model: $OPENAI_MODEL" >&2
  exit 1
fi

ROOT_DIR="$(smoke_repo_root)"
cd "$ROOT_DIR"
smoke_activate_venv

if [[ -n "${SMOKE_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$SMOKE_API_KEY"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[smoke_gpt_live] set OPENAI_API_KEY or SMOKE_API_KEY first" >&2
  exit 1
fi

export JUDGE_API_KEY="${JUDGE_API_KEY:-$OPENAI_API_KEY}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"

smoke_unset_offline_eval_env
smoke_bert_score_check

echo "[smoke_gpt_live] rebuilding sample 0 with $OPENAI_MODEL"
python cli/build_memory.py --model gpt --openai-model "$OPENAI_MODEL" --sample-idx 0 --force

python scripts/smoke_live.py run --model gpt --openai-model "$OPENAI_MODEL" --sample-idx 0

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ "$OPENAI_MODEL" == "gpt-4o-mini" ]]; then
  DB_PREFIX="db/gpt/lancedb_data_frozen_sample"
  GT_STEM="gpt"
  EVAL_STEM="gpt"
else
  DB_PREFIX="db/gpt41_mini/lancedb_data_frozen_gpt41_mini_sample"
  GT_STEM="gpt41_mini"
  EVAL_STEM="gpt41_mini"
fi

SUBSET_GT="$TMP_DIR/gt_memory_verification_${GT_STEM}_sample0.json"
smoke_subset_gt "$DB_PREFIX" 0 "$SUBSET_GT"

EVAL_SCRIPT="cli/ramem_eval.py"
EVAL_JSON="SSS_results/${EVAL_STEM}_eval_1540_s0.json"
CONTEXT_JSON="gt_context_data/1540_${EVAL_STEM}_t_contexts_s0.json"

python "$EVAL_SCRIPT" --model gpt --openai-model "$OPENAI_MODEL" --samples 0 --gt-json "$SUBSET_GT" --allow-subset-gt
python scripts/smoke_live.py validate \
  --eval-json "$EVAL_JSON" \
  --context-json "$CONTEXT_JSON" \
  --sample-idx 0

echo "[smoke_gpt_live] PASS ($OPENAI_MODEL)"
