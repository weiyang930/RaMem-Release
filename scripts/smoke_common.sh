#!/bin/bash

smoke_repo_root() {
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
}

smoke_activate_venv() {
  if [[ -n "${VENV_PATH:-}" ]]; then
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

  echo "[smoke] VENV_PATH is unset and conda is unavailable; using current shell" >&2
}

smoke_wait_for_models() {
  local base_url="$1"
  local timeout_s="${2:-600}"
  local endpoint="${base_url%/}/models"
  local waited=0

  echo "[smoke] waiting for models endpoint: $endpoint"
  until curl -fsS "$endpoint" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    echo "  ... waited ${waited}s"
    if [[ "$waited" -ge "$timeout_s" ]]; then
      echo "[smoke] timed out waiting for $endpoint" >&2
      return 1
    fi
  done
  curl -fsS "$endpoint"
}

smoke_unset_offline_eval_env() {
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
}

smoke_bert_score_check() {
  python - <<'PY'
from bert_score import score

predictions = ["The meeting was on May 7, 2023."]
references = ["The meeting happened on 7 May 2023."]
_, _, f1 = score(predictions, references, lang="en", verbose=False)
value = float(f1[0])
print(f"[smoke] BERTScore F1={value:.4f}")
if value <= 0.0:
    raise SystemExit("BERTScore smoke returned a non-positive value")
PY
}

smoke_subset_gt() {
  local db_prefix="$1"
  local sample_idx="$2"
  local out_path="$3"
  python -m model_runs.gt_verification \
    --db-prefix "$db_prefix" \
    --samples "$sample_idx" \
    --out "$out_path" \
    --print-limit 0
}
