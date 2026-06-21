from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

from ramem import config
from model_runs.run_specs import RunSpec


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "locomo10.json"

SEP = "=" * 60


def _cleanup_sample_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"  [WARN] CUDA cleanup skipped: {exc}")


def configure_runtime(spec: RunSpec) -> None:
    env_openai_api_key = os.getenv("OPENAI_API_KEY")
    env_judge_api_key = os.getenv("JUDGE_API_KEY")
    env_openai_base_url = os.getenv("OPENAI_BASE_URL")
    env_judge_base_url = os.getenv("JUDGE_BASE_URL")

    if env_openai_api_key:
        config.OPENAI_API_KEY = env_openai_api_key
    elif spec.default_base_url:
        config.OPENAI_API_KEY = "vllm-local"

    config.JUDGE_API_KEY = env_judge_api_key or config.OPENAI_API_KEY
    config.OPENAI_BASE_URL = env_openai_base_url if env_openai_base_url is not None else spec.default_base_url
    config.JUDGE_BASE_URL = env_judge_base_url if env_judge_base_url is not None else config.OPENAI_BASE_URL
    llm_model_override = os.getenv("QWEN_LLM_MODEL") if spec.key == "qwen" else None
    config.LLM_MODEL = llm_model_override or spec.llm_model
    config.JUDGE_MODEL = os.getenv("QWEN_JUDGE_MODEL") if spec.key == "qwen" and os.getenv("QWEN_JUDGE_MODEL") else spec.judge_model
    config.ENABLE_THINKING = False
    config.JUDGE_ENABLE_THINKING = False
    config.EMBEDDING_MODEL = spec.embedding_model
    config.EMBEDDING_DIMENSION = 1024


def _parse_sample_arg(raw_idx: str, n_samples: int) -> list[int]:
    raw_idx = raw_idx.strip().lower()
    if raw_idx == "all":
        return list(range(n_samples))

    try:
        return [int(chunk.strip()) for chunk in raw_idx.split(",") if chunk.strip()]
    except ValueError as exc:
        raise SystemExit(
            f"--sample-idx must be 'all', a single integer, or comma-separated ints. Got: {raw_idx!r}"
        ) from exc


def _build_sample(spec: RunSpec, sample_idx: int, dataset_path: Path, force: bool = False) -> bool:
    from ramem.main import RaMemSystem
    from model_runs.locomo import LoCoMoTester

    frozen_path = spec.frozen_db_path(sample_idx)
    if frozen_path.exists() and not force:
        print(f"  [SKIP] sample {sample_idx}: {frozen_path} already exists (use --force to rebuild)")
        return False

    print()
    print(SEP)
    print(f"  Building memory for sample {sample_idx} [{spec.flow_label}]".center(60))
    print(SEP)
    print("\nInitializing RaMem (clear_db=True)...")

    system = RaMemSystem(clear_db=True, db_path=str(spec.active_db))
    tester = LoCoMoTester(system, str(dataset_path))
    samples = tester.load_dataset()

    if sample_idx >= len(samples):
        print(f"  ERROR: sample_idx {sample_idx} out of range (dataset has {len(samples)} samples)")
        return False

    sample = samples[sample_idx]
    print(f"  Sample {sample_idx}: {sample.conversation.speaker_a} & {sample.conversation.speaker_b}")

    system.vector_store.clear()
    dialogues = tester.convert_to_dialogues(sample)
    print(f"  Adding {len(dialogues)} dialogues to memory...")

    t0 = time.time()
    system.add_dialogues(dialogues)
    system.finalize()
    elapsed = time.time() - t0
    print(f"\n  Memory build complete in {elapsed:.2f}s")

    if frozen_path.exists():
        print(f"  Removing existing frozen DB at {frozen_path.name}/ ...")
        shutil.rmtree(str(frozen_path))
    print(f"  Freezing → {frozen_path.name}/ ...")
    shutil.copytree(str(spec.active_db), str(frozen_path))
    print(f"  Frozen DB saved: {frozen_path}")
    del dialogues
    del samples
    del tester
    del system
    _cleanup_sample_runtime()
    return True


def build_memory_main(
    spec: RunSpec,
    argv: list[str] | None = None,
    show_model_flag: bool = False,
    show_openai_model_flag: bool = False,
) -> None:
    configure_runtime(spec)

    description = (
        "Build and freeze model-backed RaMem LanceDB for LoCoMo10 samples."
        if show_model_flag
        else f"Build and freeze {spec.backbone_label}-backed RaMem LanceDB for LoCoMo10 samples."
    )
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    if show_model_flag:
        parser.add_argument(
            "--model",
            required=True,
            choices=["gpt", "qwen", "llama31_8b", "llama32_3b"],
            help="Model family whose memories should be built",
        )
    if show_openai_model_flag:
        parser.add_argument(
            "--openai-model",
            choices=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
            default=None,
            help="Required when --model gpt. Supported: gpt-4o-mini, gpt-4.1-mini, gpt-4o",
        )
    parser.add_argument(
        "--sample-idx",
        default="all",
        metavar="IDX",
        help="'all' or a single integer 0-9 or comma-separated list (e.g. '0,1,2') (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if frozen DB already exists")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to locomo10.json (default: %(default)s)")
    args = parser.parse_args(argv)

    sample_indices = _parse_sample_arg(args.sample_idx, len(spec.expected_samples))
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    print(SEP)
    print(f" cli/build_memory.py  [{spec.flow_label}]".center(60))
    print(SEP)
    print(f"  Samples to process : {sample_indices}")
    print(f"  Force rebuild      : {args.force}")
    print(f"  Dataset            : {dataset_path.name}")
    print(f"  Output prefix      : {spec.frozen_prefix}{{N}}")
    print()

    for idx in sample_indices:
        frozen_path = spec.frozen_db_path(idx)
        status = "EXISTS" if frozen_path.exists() else "missing"
        skip = " → will skip" if frozen_path.exists() and not args.force else ""
        print(f"    sample {idx:2d}  frozen DB: {status}{skip}")
    print()

    built: list[int] = []
    skipped: list[int] = []
    for sample_idx in sample_indices:
        if _build_sample(spec, sample_idx, dataset_path, force=args.force):
            built.append(sample_idx)
        else:
            skipped.append(sample_idx)

    print()
    print(SEP)
    print(f" Build Summary  [{spec.flow_label}]".center(60))
    print(SEP)
    if built:
        print(f"  Built   : samples {built}")
    if skipped:
        print(f"  Skipped : samples {skipped}  (frozen DB already existed)")
    if not built and not skipped:
        print("  Nothing to do.")
    print()
