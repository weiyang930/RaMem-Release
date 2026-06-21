#!/bin/bash
# Download the Qwen embedding model into the configured Hugging Face cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"

find_cached_snapshot() {
    local repo_dir="$HF_HUB_CACHE/models--${MODEL_ID//\//--}"
    if [ -f "$repo_dir/refs/main" ]; then
        local snapshot
        snapshot="$(<"$repo_dir/refs/main")"
        if [ -f "$repo_dir/snapshots/$snapshot/config.json" ]; then
            echo "$repo_dir/snapshots/$snapshot"
            return 0
        fi
    fi

    find "$repo_dir" -type f -path '*/snapshots/*/config.json' 2>/dev/null \
        | sed 's#/config.json$##' \
        | head -n 1
}

echo "[INFO] Checking connectivity to Hugging Face..."
if ! curl -fsS --connect-timeout 10 https://huggingface.co >/dev/null; then
    echo "[ERROR] Could not reach https://huggingface.co from this shell."
    exit 1
fi

echo "[INFO] Downloading $MODEL_ID into $HF_HUB_CACHE"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["HF_HUB_CACHE"]

path = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_files_only=False,
    resume_download=True,
)
print(path)
PY

SNAPSHOT_PATH="$(find_cached_snapshot)"
if [ -z "$SNAPSHOT_PATH" ]; then
    echo "[ERROR] Download finished but no local snapshot path was found."
    exit 1
fi

echo "[OK] Embedding snapshot ready at: $SNAPSHOT_PATH"
