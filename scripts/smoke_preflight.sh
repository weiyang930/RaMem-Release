#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[smoke_preflight] repo root: $ROOT_DIR"

required_paths=(
  "README.md"
  "ramem/config.py"
  "ramem/db_layout.py"
  "ramem/main.py"
  "cli/build_memory.py"
  "cli/ramem_eval.py"
  "model_runs/run_specs.py"
  "model_runs/build_common.py"
  "model_runs/eval_1540_common.py"
  "model_runs/gt_bootstrap.py"
  "model_runs/gt_verification.py"
  "model_runs/locomo.py"
  "core/memory_builder.py"
  "core/hybrid_retriever.py"
  "core/answer_generator.py"
  "database/vector_store.py"
  "models/memory_entry.py"
  "utils/embedding.py"
  "utils/llm_client.py"
  "scripts/smoke_common.sh"
  "scripts/smoke_live.py"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[smoke_preflight] missing required path: $path" >&2
    exit 1
  fi
done

echo "[smoke_preflight] required paths present"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"$PYTHON_BIN" -m py_compile \
  cli/build_memory.py \
  cli/ramem_eval.py \
  cli/run_longmemeval_s.py \
  ramem/config.py \
  ramem/db_layout.py \
  ramem/main.py \
  model_runs/run_specs.py \
  model_runs/build_common.py \
  model_runs/eval_1540_common.py \
  model_runs/gt_bootstrap.py \
  model_runs/gt_verification.py \
  model_runs/locomo.py \
  core/memory_builder.py \
  core/hybrid_retriever.py \
  core/answer_generator.py \
  database/vector_store.py \
  models/memory_entry.py \
  utils/embedding.py \
  utils/llm_client.py \
  scripts/smoke_live.py

echo "[smoke_preflight] py_compile OK"

bash -n \
  scripts/build_memory_qwen.sh \
  scripts/download_qwen_embedding.sh \
  scripts/start_vllm.sh \
  scripts/stop_vllm.sh \
  scripts/eval_qwen_parallel.sh \
  scripts/run_qwen_variant_pipeline.sh \
  scripts/run_qwen25_3b_15b_pipeline.sh \
  scripts/smoke_common.sh \
  scripts/smoke_preflight.sh \
  scripts/smoke_qwen_live.sh \
  scripts/smoke_gpt_live.sh

echo "[smoke_preflight] shell syntax OK"
