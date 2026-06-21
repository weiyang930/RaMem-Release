#!/usr/bin/env python3
"""Live smoke helpers for the RaMem build/eval flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ramem import config
from model_runs.eval_1540_common import (
    METHOD_MODE,
    _apply_mode_config,
    _classify_trigger,
    _derive_conv_window,
    _last_retrieved_entry_ids,
    _preview_entry_ids,
    _safe_bert,
    _safe_f1,
    configure_runtime,
)
from model_runs.run_specs import DEFAULT_GPT_OPENAI_MODEL, RunSpec, get_run_spec_for_cli


def _resolve_db_path(spec: RunSpec, sample_idx: int, raw_db_path: str | None) -> Path:
    if raw_db_path:
        return Path(raw_db_path).expanduser().resolve()
    return spec.frozen_db_path(sample_idx)


def _first_supported_question_idx(sample) -> int:
    for idx, qa in enumerate(sample.qa):
        if qa.category in (1, 2, 3, 4):
            return idx
    raise SystemExit("No category 1-4 question found in selected sample")


def _select_question(sample, question_idx: int | None) -> tuple[int, object]:
    resolved_idx = _first_supported_question_idx(sample) if question_idx is None else question_idx
    if resolved_idx < 0 or resolved_idx >= len(sample.qa):
        raise SystemExit(f"Question index {resolved_idx} is out of range for the sample")
    qa = sample.qa[resolved_idx]
    if qa.category not in (1, 2, 3, 4):
        raise SystemExit(f"Question index {resolved_idx} is category {qa.category}; smoke only supports cat1-4")
    return resolved_idx, qa


def _run_single_question(args: argparse.Namespace) -> None:
    from ramem.main import RaMemSystem
    from model_runs.locomo import LoCoMoTester

    if args.model == "gpt" and not args.openai_model:
        raise SystemExit("--model gpt requires --openai-model (gpt-4o-mini or gpt-4.1-mini)")
    spec = get_run_spec_for_cli(args.model, args.openai_model)
    mode = METHOD_MODE
    configure_runtime(spec)
    config.USE_STREAMING = False
    config.JUDGE_USE_STREAMING = False

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    db_path = _resolve_db_path(spec, args.sample_idx, args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB path not found: {db_path}")

    system = RaMemSystem(clear_db=False, db_path=str(db_path))
    tester = LoCoMoTester(system, str(dataset_path))
    sample = tester.load_dataset()[args.sample_idx]
    question_idx, qa = _select_question(sample, args.question_idx)
    conv_start, conv_end = _derive_conv_window(sample)
    _apply_mode_config(mode, conv_start, conv_end)
    result = tester._process_single_question(qa, question_idx)

    f1 = _safe_f1(result)
    bert = _safe_bert(result)
    retrieved_entry_ids = _last_retrieved_entry_ids()
    if not retrieved_entry_ids:
        raise SystemExit("Smoke failed: no retrieved entry ids were captured")
    answer = result.get("answer", "").strip()
    if not answer:
        raise SystemExit("Smoke failed: empty answer returned")

    print(f"[smoke_live] model={spec.key}")
    print(f"[smoke_live] backbone={spec.backbone_label}")
    print(f"[smoke_live] mode={mode.key}")
    print(f"[smoke_live] db_path={db_path}")
    print(f"[smoke_live] sample_idx={args.sample_idx}")
    print(f"[smoke_live] question_idx={question_idx}")
    print(f"[smoke_live] category={qa.category}")
    print(f"[smoke_live] trigger_type={_classify_trigger(qa.question)}")
    print(f"[smoke_live] question={qa.question}")
    print(f"[smoke_live] answer={answer}")
    print(f"[smoke_live] retrieved_ids={len(retrieved_entry_ids)} {_preview_entry_ids(retrieved_entry_ids)}")
    print(f"[smoke_live] num_retrieved={result.get('num_retrieved', 0)}")
    print(f"[smoke_live] retrieval_time={result.get('retrieval_time', 0.0):.3f}")
    print(f"[smoke_live] f1={f1:.4f}")
    print(f"[smoke_live] bert={bert:.4f}")
    if mode.key == METHOD_MODE.key:
        guard2_window = getattr(config, "_guard2_window", None)
        guard2_buffered = getattr(config, "_guard2_buffered_window", None)
        if _classify_trigger(qa.question) == "none":
            guard_status = "NOT_TRIGGERED"
        elif guard2_window:
            guard_status = "TRIGGERED_G2_FIRED"
        else:
            guard_status = "TRIGGERED_G2_NULL"
        print(f"[smoke_live] guard_status={guard_status}")
        print(f"[smoke_live] guard2_window={guard2_window}")
        print(f"[smoke_live] guard2_buffered={guard2_buffered}")


def _require_fields(record: dict, required_fields: list[str], label: str) -> None:
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise SystemExit(f"{label} is missing required fields: {missing}")


def _validate_outputs(args: argparse.Namespace) -> None:
    mode = METHOD_MODE
    eval_path = Path(args.eval_json).expanduser().resolve()
    context_path = Path(args.context_json).expanduser().resolve()
    if not eval_path.exists():
        raise SystemExit(f"Eval JSON not found: {eval_path}")
    if not context_path.exists():
        raise SystemExit(f"Context JSON not found: {context_path}")

    eval_data = json.loads(eval_path.read_text())
    context_data = json.loads(context_path.read_text())

    eval_questions = eval_data.get("questions")
    context_questions = context_data.get("questions")
    if not isinstance(eval_questions, list) or not eval_questions:
        raise SystemExit("Eval JSON has no questions")
    if not isinstance(context_questions, list) or not context_questions:
        raise SystemExit("Context JSON has no questions")

    eval_metadata = eval_data.get("metadata") or {}
    context_metadata = context_data.get("metadata") or {}
    if eval_metadata.get("mode") != mode.key:
        raise SystemExit(f"Eval JSON mode mismatch: expected {mode.key}, got {eval_metadata.get('mode')}")
    if context_metadata.get("mode") != mode.key:
        raise SystemExit(
            f"Context JSON mode mismatch: expected {mode.key}, got {context_metadata.get('mode')}"
        )

    if args.sample_idx is not None:
        eval_sample_ids = sorted({question.get("sample_idx") for question in eval_questions})
        context_sample_ids = sorted({question.get("sample_idx") for question in context_questions})
        if eval_sample_ids != [args.sample_idx]:
            raise SystemExit(f"Eval JSON sample coverage mismatch: expected [{args.sample_idx}], got {eval_sample_ids}")
        if context_sample_ids != [args.sample_idx]:
            raise SystemExit(
                f"Context JSON sample coverage mismatch: expected [{args.sample_idx}], got {context_sample_ids}"
            )

    eval_required = [
        "sample_idx",
        "question_global_idx",
        "question",
        "category",
        "answer",
        "retrieved_entry_ids",
        "single_gt_entry_id",
        "gt_rank_entry",
        "gt_rank",
        "gt_f1",
        "f1",
        "bert",
        "correct",
    ]
    context_required = [
        "sample_idx",
        "question_global_idx",
        "question",
        "single_gt_entry_id",
        "retrieved_memories",
        "formatted_prompt",
        "answer",
        "f1",
        "bert",
        "correct",
        "gt_new_rank",
        "set_label",
    ]
    if mode.key == METHOD_MODE.key:
        eval_required.extend(["guard_status", "guard2_window", "guard2_buffered"])
        context_required.extend(["guard_status", "guard2_window", "guard2_buffered"])

    for question in eval_questions:
        _require_fields(question, eval_required, "Eval question")
        if not isinstance(question.get("retrieved_entry_ids"), list):
            raise SystemExit("Eval question retrieved_entry_ids must be a list")

    for question in context_questions:
        _require_fields(question, context_required, "Context question")
        if not isinstance(question.get("retrieved_memories"), list):
            raise SystemExit("Context question retrieved_memories must be a list")

    print(f"[smoke_live] validated eval_json={eval_path}")
    print(f"[smoke_live] validated context_json={context_path}")
    print(f"[smoke_live] eval_questions={len(eval_questions)}")
    print(f"[smoke_live] context_questions={len(context_questions)}")
    print(f"[smoke_live] mode={mode.key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live smoke helpers for the cleaned eval/build flow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one live single-question smoke through RaMem",
    )
    run_parser.add_argument("--model", required=True, help="gpt, qwen, llama31_8b, or llama32_3b")
    run_parser.add_argument(
        "--openai-model",
        choices=[DEFAULT_GPT_OPENAI_MODEL, "gpt-4.1-mini"],
        default=None,
        help="Required when --model gpt",
    )
    run_parser.add_argument("--sample-idx", type=int, default=0)
    run_parser.add_argument("--question-idx", type=int, default=None)
    run_parser.add_argument("--db-path", default=None, help="Optional explicit frozen DB path")
    run_parser.add_argument("--dataset", default=str(ROOT / "locomo10.json"))
    run_parser.set_defaults(func=_run_single_question)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the official eval/context JSONs written by a sample smoke eval",
    )
    validate_parser.add_argument("--eval-json", required=True)
    validate_parser.add_argument("--context-json", required=True)
    validate_parser.add_argument("--sample-idx", type=int, default=None)
    validate_parser.set_defaults(func=_validate_outputs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
