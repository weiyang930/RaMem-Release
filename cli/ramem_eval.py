#!/usr/bin/env python3
"""Unified RaMem LoCoMo evaluation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_runs.eval_1540_common import METHOD_MODE, eval_main
from model_runs.run_specs import (
    DEFAULT_GPT_OPENAI_MODEL,
    GPT4O_OPENAI_MODEL,
    GPT41_MINI_OPENAI_MODEL,
    get_run_spec_for_cli,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RaMem on LoCoMo.")
    parser.add_argument("--model", required=True, choices=["gpt", "qwen", "llama31_8b", "llama32_3b"])
    parser.add_argument(
        "--openai-model",
        choices=[DEFAULT_GPT_OPENAI_MODEL, GPT41_MINI_OPENAI_MODEL, GPT4O_OPENAI_MODEL],
        help="Required when --model gpt.",
    )
    args, remaining = parser.parse_known_args()
    if args.model == "gpt" and not args.openai_model:
        raise SystemExit("--model gpt requires --openai-model")
    spec = get_run_spec_for_cli(args.model, args.openai_model)
    eval_main(spec, METHOD_MODE, argv=remaining)


if __name__ == "__main__":
    main()
