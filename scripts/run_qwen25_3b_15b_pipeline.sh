#!/bin/bash
# Sequential full-pipeline run for Qwen2.5-3B-Instruct and Qwen2.5-1.5B-Instruct.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_qwen_variant_pipeline.sh" qwen25_3b Qwen/Qwen2.5-3B-Instruct
bash "$SCRIPT_DIR/run_qwen_variant_pipeline.sh" qwen25_15b Qwen/Qwen2.5-1.5B-Instruct
