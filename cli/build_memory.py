#!/usr/bin/env python3
"""Primary build entrypoint for GPT, Qwen, and Llama memory DBs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_runs.build_common import build_memory_main
from model_runs.run_specs import DEFAULT_GPT_OPENAI_MODEL, GPT4O_OPENAI_MODEL, GPT41_MINI_OPENAI_MODEL, GPT_SPEC, get_run_spec_for_cli


def main() -> None:
    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        build_memory_main(
            GPT_SPEC,
            argv=sys.argv[1:],
            show_model_flag=True,
            show_openai_model_flag=True,
        )
        return

    parser = argparse.ArgumentParser(
        description="Primary memory-build entrypoint for GPT, Qwen, and Llama families."
    )
    parser.add_argument("--model", required=True, choices=["gpt", "qwen", "llama31_8b", "llama32_3b"])
    parser.add_argument("--openai-model", choices=[DEFAULT_GPT_OPENAI_MODEL, GPT41_MINI_OPENAI_MODEL, GPT4O_OPENAI_MODEL])
    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(2)
    known_args, remaining = parser.parse_known_args()
    if known_args.model == "gpt" and not known_args.openai_model:
        raise SystemExit("--model gpt requires --openai-model (gpt-4o-mini, gpt-4.1-mini, or gpt-4o)")
    spec = get_run_spec_for_cli(known_args.model, known_args.openai_model)
    forwarded_argv = ["--model", known_args.model]
    if known_args.openai_model:
        forwarded_argv.extend(["--openai-model", known_args.openai_model])
    forwarded_argv.extend(remaining)
    build_memory_main(
        spec,
        argv=forwarded_argv,
        show_model_flag=True,
        show_openai_model_flag=True,
    )


if __name__ == "__main__":
    main()
