from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import dateparser
import lancedb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramem import config
from core.answer_generator import AnswerGenerator
from ramem.db_layout import resolve_prefixed_db_path
from model_runs.gt_bootstrap import ensure_gt_verification
from models.memory_entry import MemoryEntry
from model_runs.run_specs import (
    DEFAULT_METHOD_CATEGORIES,
    EXPECTED_QUESTION_COUNT,
    RunSpec,
    SHARD_DIR,
    SSS_DIR,
)


DATASET_PATH = ROOT / "locomo10.json"
TABLE_NAME = "memory_entries"
FORMATTER = AnswerGenerator(llm_client=None)

SEP = "=" * 80
SEP2 = "─" * 80

_YEAR_ONLY_RE = re.compile(r"\b(19|20)\d{2}\b")
_GUARD1_EXT_RE = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
    r"|\b(summer|winter|spring|fall|autumn)\b"
    r"|\b(first|second|third|last|early|mid|late|beginning|"
    r"end|middle|past|previous)\b.{0,25}\b(week|weekend)\b"
    r"|\b(week|weekend)\b.{0,20}\b(before|after|of)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvalMode:
    key: str
    label: str
    short_tag: str
    entity_aware: bool
    temporal_enabled: bool


METHOD_MODE = EvalMode(
    key="method",
    label="T Method",
    short_tag="t",
    entity_aware=True,
    temporal_enabled=True,
)

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
    judge_model_override = os.getenv("QWEN_JUDGE_MODEL") if spec.key == "qwen" else None
    config.LLM_MODEL = llm_model_override or spec.llm_model
    config.JUDGE_MODEL = judge_model_override or spec.judge_model
    config.ENABLE_THINKING = False
    config.JUDGE_ENABLE_THINKING = False
    config.EMBEDDING_MODEL = spec.embedding_model
    config.EMBEDDING_DIMENSION = 1024
    config.SSS_USE_RRF = True
    config.SSS_ENABLE_PROX_RERANK = True
    config.USE_RRF_BASELINE = False
    config.NORMALIZE_ANSWER_QUOTES = True
    config.STRUCTURED_TOP_K = 25
    config.GUARD1_EXTENDED = True
    config.ENABLE_TEMPORAL = False
    config.CONV_DATE_START = ""
    config.CONV_DATE_END = ""
    config.ENTITY_AWARE_ANSWER = False


def _default_output_stem(spec: RunSpec, mode: EvalMode) -> str:
    return spec.method_eval_stem


def _default_context_out(spec: RunSpec, mode: EvalMode) -> Path:
    return spec.method_context_out


def _parse_categories(raw_categories: list[int] | None) -> list[int]:
    if not raw_categories:
        return list(DEFAULT_METHOD_CATEGORIES)
    categories = sorted(set(raw_categories))
    invalid = [category for category in categories if category not in DEFAULT_METHOD_CATEGORIES]
    if invalid:
        raise SystemExit(f"--categories only supports cat 1-4. Invalid values: {invalid}")
    return categories


def _parse_samples(raw_samples: list[int] | None, spec: RunSpec) -> list[int]:
    if not raw_samples:
        return list(spec.expected_samples)
    invalid = [sample_idx for sample_idx in sorted(set(raw_samples)) if sample_idx not in spec.expected_samples]
    if invalid:
        raise SystemExit(f"--samples contains out-of-range indices: {invalid}")
    return sorted(set(raw_samples))


def _selection_suffix(sample_indices: list[int], categories: list[int], spec: RunSpec) -> str:
    parts: list[str] = []
    if sample_indices != spec.expected_samples:
        parts.append("s" + "".join(str(sample_idx) for sample_idx in sample_indices))
    if categories != DEFAULT_METHOD_CATEGORIES:
        parts.append("c" + "".join(str(category) for category in categories))
    return "_" + "_".join(parts) if parts else ""


def _output_paths(spec: RunSpec, mode: EvalMode, sample_indices: list[int], categories: list[int]) -> tuple[Path, Path]:
    suffix = _selection_suffix(sample_indices, categories, spec)
    eval_path = SSS_DIR / f"{_default_output_stem(spec, mode)}{suffix}.json"

    default_context = _default_context_out(spec, mode)
    context_filename = f"{default_context.stem}{suffix}{default_context.suffix}"
    context_path = default_context.with_name(context_filename)
    return eval_path, context_path


def _shard_paths(spec: RunSpec, mode: EvalMode, stem: str | None = None) -> list[Path]:
    stem = stem or _default_output_stem(spec, mode)
    return [spec.shard_path(stem, sample_idx) for sample_idx in spec.expected_samples]


def _classify_trigger(question: str) -> str:
    if not _GUARD1_EXT_RE.search(question):
        return "none"
    return "year" if _YEAR_ONLY_RE.search(question) else "implicit"


def _safe_f1(result: dict) -> float:
    return result.get("metrics", {}).get("f1", 0.0) or 0.0


def _safe_bert(result: dict) -> float:
    return result.get("metrics", {}).get("bert_f1", 0.0) or 0.0


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: round(float(value), 4)
        for key, value in (metrics or {}).items()
        if isinstance(value, (int, float))
    }


def _record_metrics(record: dict) -> dict[str, float]:
    metrics = _numeric_metrics(record.get("metrics") or {})
    if not metrics:
        metrics = {
            "f1": float(record.get("f1") or 0.0),
            "bert_f1": float(record.get("bert") or 0.0),
        }
    return metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _last_retrieved_entry_ids() -> list[str]:
    runtime_local = getattr(config, "_runtime_local", None)
    if runtime_local and hasattr(runtime_local, "last_retrieved_contexts"):
        entries = getattr(runtime_local, "last_retrieved_contexts", None) or []
    else:
        entries = getattr(config, "_last_retrieved_contexts", None) or []
    return [entry.entry_id for entry in entries if getattr(entry, "entry_id", None)]


def _preview_entry_ids(entry_ids: list[str], limit: int = 5) -> str:
    if not entry_ids:
        return "[]"
    shown = entry_ids[:limit]
    suffix = ", ..." if len(entry_ids) > limit else ""
    return "[" + ", ".join(shown) + suffix + "]"


def _load_gt_index(gt_json_path: Path) -> dict[tuple[int, int], dict]:
    with gt_json_path.open() as handle:
        gt_data = json.load(handle)

    index: dict[tuple[int, int], dict] = {}
    for record in gt_data.get("questions", []):
        sample_idx = record.get("sample_idx")
        question_idx = record.get("question_global_idx")
        if sample_idx is None or question_idx is None:
            continue
        gt_memory = record.get("single_gt_memory") or record.get("single_gt_relaxed_memory")
        index[(sample_idx, question_idx)] = {
            "single_gt_entry_id": gt_memory.get("entry_id") if gt_memory else None,
            "single_gt_status": record.get("single_gt_status"),
        }
    return index


def _rank_entry_id(entry_id: Optional[str], retrieved_entry_ids: list[str]) -> Optional[int]:
    if not entry_id:
        return None
    for rank, retrieved_id in enumerate(retrieved_entry_ids, 1):
        if retrieved_id == entry_id:
            return rank
    return -1


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


GT_RANK_THRESHOLD = 0.10


def _find_gt_rank(reference: str) -> tuple[int, float]:
    runtime_local = getattr(config, "_runtime_local", None)
    if runtime_local and hasattr(runtime_local, "last_retrieved_contexts"):
        entries = getattr(runtime_local, "last_retrieved_contexts", None) or []
    else:
        entries = getattr(config, "_last_retrieved_contexts", None) or []
    if not reference or not entries:
        return -1, 0.0

    best_f1 = 0.0
    best_rank = -1
    for rank, entry in enumerate(entries, 1):
        text = getattr(entry, "lossless_restatement", "") or ""
        f1 = _token_f1(text, reference)
        if f1 > best_f1:
            best_f1 = f1
            best_rank = rank

    if best_f1 < GT_RANK_THRESHOLD:
        return -1, round(best_f1, 4)
    return best_rank, round(best_f1, 4)


def _group_stats(records: list[dict]) -> Optional[dict]:
    if not records:
        return None
    n = len(records)
    metric_names = sorted({name for record in records for name in _record_metrics(record)})
    metric_means = {
        f"mean_{name}": round(_mean([_record_metrics(record).get(name, 0.0) for record in records]), 4)
        for name in metric_names
    }
    gt_found = sum(1 for record in records if (record["gt_rank"] or -1) > 0)
    stats = {
        "n": n,
        "gt_found": gt_found,
        "gt_found_pct": round(100 * gt_found / n, 1),
        "correct_count": sum(1 for record in records if record.get("correct")),
    }
    stats.update(metric_means)
    if "mean_bert_f1" in stats:
        stats["mean_bert"] = stats["mean_bert_f1"]
    elif "mean_bert" not in stats:
        stats["mean_bert"] = round(_mean([float(record.get("bert") or 0.0) for record in records]), 4)
    if "mean_f1" not in stats:
        stats["mean_f1"] = round(_mean([float(record.get("f1") or 0.0) for record in records]), 4)
    return stats


def _derive_conv_window(sample) -> tuple[str, str]:
    parsed_dates = []
    for session in sample.conversation.sessions.values():
        if session.date_time:
            dt = dateparser.parse(
                session.date_time,
                settings={"PREFER_DATES_FROM": "past", "RETURN_AS_TIMEZONE_AWARE": False},
            )
            if dt:
                parsed_dates.append(dt)
    if not parsed_dates:
        raise ValueError("Could not parse any session dates from sample")
    parsed_dates.sort()
    return parsed_dates[0].strftime("%Y-%m-%d"), parsed_dates[-1].strftime("%Y-%m-%d")


def _print_group_table(header: str, groups: dict) -> None:
    print(f"\n  ── {header} {'─' * (60 - len(header))}")
    print(f"  {'Label':>18}  {'N':>4}  {'F1':>7}  {'BERT':>8}  {'GT%':>6}  {'Correct':>8}")
    print(f"  {'─' * 18}  {'─' * 4}  {'─' * 7}  {'─' * 8}  {'─' * 6}  {'─' * 8}")
    for label, stats in groups.items():
        if not stats:
            continue
        print(
            f"  {str(label):>18}  {stats['n']:>4}  {stats['mean_f1']:>7.4f}  {stats['mean_bert']:>8.4f}  "
            f"{stats['gt_found_pct']:>5.1f}%  {stats['correct_count']:>8}"
        )


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)


def _save_eval_json(
    spec: RunSpec,
    mode: EvalMode,
    out_file: Path,
    all_records: list[dict],
    available: list[int],
    missing: list[int],
    gt_json_path: Path,
    gt_indexed_questions: int,
    categories: list[int],
) -> None:
    if not all_records:
        return

    def bucket(key_fn):
        grouped = defaultdict(list)
        for record in all_records:
            grouped[key_fn(record)].append(record)
        return dict(grouped)

    overall_stats = _group_stats(all_records)
    try:
        from model_runs.locomo import aggregate_metrics

        original_metric_summary = aggregate_metrics(
            [_record_metrics(record) for record in all_records],
            [record["category"] for record in all_records],
        )
    except Exception as exc:
        print(f"  WARNING: original metric aggregation failed: {exc}")
        original_metric_summary = {}
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "script": spec.method_eval_script,
            "description": (
                f"{mode.label} on cat1-4 questions using {spec.backbone_label} frozen DBs"
            ),
            "mode": mode.key,
            "backbone": spec.backbone_label,
            "exclude_cat5": True,
            "categories": categories,
            "samples_available": available,
            "samples_missing_db": missing,
            "samples_run": sorted({record["sample_idx"] for record in all_records}),
            "n_questions": len(all_records),
            "sss_use_rrf": config.SSS_USE_RRF,
            "sss_prox_rerank": config.SSS_ENABLE_PROX_RERANK,
            "entity_aware": mode.entity_aware,
            "temporal_enabled": mode.temporal_enabled,
            "context_ablation_variant": os.getenv("CONTEXT_ABLATION_VARIANT", "full"),
            "ablate_disable_cue_guard": getattr(config, "ABLATE_DISABLE_CUE_GUARD", False),
            "ablate_disable_context_aware_ranking": getattr(config, "ABLATE_DISABLE_CONTEXT_AWARE_RANKING", False),
            "ablate_generation_text_only": getattr(config, "ABLATE_GENERATION_TEXT_ONLY", False),
            "normalize_quotes": config.NORMALIZE_ANSWER_QUOTES,
            "gt_rank_threshold": GT_RANK_THRESHOLD,
            "gt_json": str(gt_json_path),
            "gt_indexed_questions": gt_indexed_questions,
            "saved_retrieved_entry_ids": True,
            "saved_entry_rank_fields": ["gt_rank_entry"],
            "db_prefix": spec.frozen_prefix,
        },
        "summary": {
            "overall": overall_stats,
            "by_sample": {str(k): _group_stats(v) for k, v in sorted(bucket(lambda r: r["sample_idx"]).items())},
            "by_category": {str(k): _group_stats(v) for k, v in sorted(bucket(lambda r: r["category"]).items())},
            "by_trigger_type": {k: _group_stats(v) for k, v in sorted(bucket(lambda r: r["trigger_type"]).items())},
            "original_metrics": original_metric_summary,
        },
        "questions": all_records,
    }
    _atomic_write_json(out_file, output)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _question_key(question: dict) -> tuple[int, int]:
    return (question["sample_idx"], question["question_global_idx"])


def _ensure_gt_covers_eval_questions(
    gt_questions: list[dict],
    eval_questions: list[dict],
    require_full_count: bool,
) -> dict[tuple[int, int], dict]:
    gt_by_key = {_question_key(question): question for question in gt_questions}
    missing_keys = [
        _question_key(question)
        for question in eval_questions
        if _question_key(question) not in gt_by_key
    ]
    if missing_keys:
        preview = ", ".join(f"{sample_idx}:{question_idx}" for sample_idx, question_idx in missing_keys[:10])
        suffix = " ..." if len(missing_keys) > 10 else ""
        raise SystemExit(
            "GT verification JSON is missing evaluated questions: "
            f"{preview}{suffix}"
        )
    if require_full_count and len(gt_questions) != EXPECTED_QUESTION_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_QUESTION_COUNT} GT questions, got {len(gt_questions)}"
        )
    return gt_by_key


def _row_to_dict(row: dict) -> dict:
    return {
        "entry_id": row.get("entry_id"),
        "lossless_restatement": row.get("lossless_restatement"),
        "keywords": list(row.get("keywords") or []),
        "timestamp": row.get("timestamp") or None,
        "location": row.get("location") or None,
        "persons": list(row.get("persons") or []),
        "entities": list(row.get("entities") or []),
        "topic": row.get("topic") or None,
        "session_date": row.get("session_date") or None,
        "session_end_date": row.get("session_end_date") or None,
        "mention_date": row.get("mention_date") or None,
    }


def _row_to_entry(row: dict) -> MemoryEntry:
    return MemoryEntry(
        entry_id=row.get("entry_id") or "",
        lossless_restatement=row.get("lossless_restatement") or "",
        keywords=list(row.get("keywords") or []),
        timestamp=row.get("timestamp") or None,
        location=row.get("location") or None,
        persons=list(row.get("persons") or []),
        entities=list(row.get("entities") or []),
        topic=row.get("topic") or None,
        session_date=row.get("session_date") or None,
        session_end_date=row.get("session_end_date") or None,
        mention_date=row.get("mention_date") or None,
    )


def _fetch_entries(spec: RunSpec, sample_idx: int, entry_ids: list[str], db_prefix: str) -> dict[str, dict]:
    if not entry_ids:
        return {}

    db_path = resolve_prefixed_db_path(db_prefix, sample_idx, root=ROOT)
    if not db_path.exists():
        raise SystemExit(f"Missing DB path: {db_path}")

    db = lancedb.connect(str(db_path))
    table = db.open_table(TABLE_NAME)
    ids_str = ", ".join(f"'{entry_id}'" for entry_id in entry_ids)
    rows = (
        table.search()
        .where(f"entry_id IN ({ids_str})", prefilter=True)
        .limit(len(entry_ids) + 20)
        .to_list()
    )
    return {row["entry_id"]: _row_to_dict(row) for row in rows}


def _canonical_gt_source(gt_question: dict, eval_question: dict) -> Optional[dict]:
    return (
        gt_question.get("single_gt_memory")
        or gt_question.get("single_gt_relaxed_memory")
        or (
            {"entry_id": eval_question.get("single_gt_entry_id")}
            if eval_question.get("single_gt_entry_id")
            else None
        )
    )


def _canonical_gt_entry_id(gt_question: dict, eval_question: dict) -> Optional[str]:
    gt_source = _canonical_gt_source(gt_question, eval_question)
    return gt_source.get("entry_id") if gt_source else None


def _fallback_gt_memory_unit(gt_question: dict, eval_question: dict) -> Optional[dict]:
    gt_source = _canonical_gt_source(gt_question, eval_question)
    if not gt_source:
        return None
    return {
        "entry_id": gt_source.get("entry_id"),
        "lossless_restatement": gt_source.get("text"),
        "keywords": list(gt_source.get("keywords") or []),
        "timestamp": gt_source.get("timestamp") or None,
        "location": gt_source.get("location") or None,
        "persons": list(gt_source.get("persons") or []),
        "entities": list(gt_source.get("entities") or []),
        "topic": gt_source.get("topic") or None,
        "session_date": gt_source.get("session_date") or None,
        "session_end_date": gt_source.get("session_end_date") or None,
        "mention_date": gt_source.get("mention_date") or None,
    }


def _gt_memory_text(gt_memory_unit: Optional[dict], gt_question: dict, eval_question: dict) -> Optional[str]:
    if gt_memory_unit and gt_memory_unit.get("lossless_restatement"):
        return gt_memory_unit["lossless_restatement"]

    gt_source = _canonical_gt_source(gt_question, eval_question)
    if gt_source and gt_source.get("text"):
        return gt_source.get("text")
    return None


def _gt_rank(gt_entry_id: Optional[str], retrieved_entry_ids: list[str]) -> Optional[int]:
    if not gt_entry_id:
        return None
    for rank, entry_id in enumerate(retrieved_entry_ids, 1):
        if entry_id == gt_entry_id:
            return rank
    return -1


def _set_label(gt_rank: Optional[int], correct: bool, has_gt: bool) -> str:
    if not has_gt:
        return "no_gt"
    if gt_rank is None or gt_rank == -1:
        return "C_good" if correct else "C_bad"
    if correct:
        return "A"
    if gt_rank == 1:
        return "B_rank1"
    if gt_rank <= 5:
        return "B_rank2_5"
    return "B_rank6plus"


def _build_sample_cache(
    spec: RunSpec,
    eval_questions: list[dict],
    gt_by_key: dict[tuple[int, int], dict],
    db_prefix: str,
) -> dict[int, dict[str, dict]]:
    sample_to_ids: dict[int, set[str]] = defaultdict(set)
    for eval_question in eval_questions:
        key = _question_key(eval_question)
        gt_question = gt_by_key.get(key)
        if gt_question is None:
            raise SystemExit(f"Missing GT verification record for key={key}")
        gt_entry_id = _canonical_gt_entry_id(gt_question, eval_question)
        if gt_entry_id:
            sample_to_ids[eval_question["sample_idx"]].add(gt_entry_id)
        for entry_id in eval_question.get("retrieved_entry_ids") or []:
            sample_to_ids[eval_question["sample_idx"]].add(entry_id)

    sample_cache: dict[int, dict[str, dict]] = {}
    for sample_idx in sorted(sample_to_ids):
        entry_ids = sorted(sample_to_ids[sample_idx])
        print(f"  sample {sample_idx}: fetching {len(entry_ids)} rows ...", end="", flush=True)
        sample_cache[sample_idx] = _fetch_entries(spec, sample_idx, entry_ids, db_prefix)
        print(f" → {len(sample_cache[sample_idx])} rows")
    return sample_cache


def _build_retrieved_memories(
    retrieved_entry_ids: list[str],
    gt_entry_id: Optional[str],
    row_cache: dict[str, dict],
) -> tuple[list[dict], str, int]:
    retrieved_rows = []
    retrieved_memories = []
    missing_count = 0

    for rank, entry_id in enumerate(retrieved_entry_ids, 1):
        memory = row_cache.get(entry_id)
        if memory is None:
            missing_count += 1
            continue
        retrieved_rows.append(memory)
        memory_dict = dict(memory)
        memory_dict["rank"] = rank
        memory_dict["is_gt"] = entry_id == gt_entry_id
        retrieved_memories.append(memory_dict)

    formatted_prompt = FORMATTER._format_contexts([_row_to_entry(row) for row in retrieved_rows]) if retrieved_rows else ""
    return retrieved_memories, formatted_prompt, missing_count


def _build_context_export(
    spec: RunSpec,
    mode: EvalMode,
    merged_eval_path: Path,
    eval_questions: list[dict],
    gt_by_key: dict[tuple[int, int], dict],
    sample_cache: dict[int, dict[str, dict]],
    threshold: float,
) -> dict:
    records = []
    missing_gt_rows = 0
    missing_retrieved_rows = 0

    for eval_question in eval_questions:
        sample_idx = eval_question["sample_idx"]
        key = _question_key(eval_question)
        gt_question = gt_by_key[key]
        row_cache = sample_cache.get(sample_idx, {})

        gt_entry_id = _canonical_gt_entry_id(gt_question, eval_question)
        gt_memory_unit = row_cache.get(gt_entry_id) if gt_entry_id else None
        if gt_entry_id and gt_memory_unit is None:
            missing_gt_rows += 1
        if gt_memory_unit is None:
            gt_memory_unit = _fallback_gt_memory_unit(gt_question, eval_question)
        gt_memory_text = _gt_memory_text(gt_memory_unit, gt_question, eval_question)
        reference_answer = gt_question.get("reference_answer") or eval_question.get("reference")

        retrieved_entry_ids = eval_question.get("retrieved_entry_ids") or []
        retrieved_memories, formatted_prompt, missing_count = _build_retrieved_memories(
            retrieved_entry_ids,
            gt_entry_id,
            row_cache,
        )
        missing_retrieved_rows += missing_count
        correct = float(eval_question.get("f1") or 0.0) >= threshold
        gt_new_rank = _gt_rank(gt_entry_id, retrieved_entry_ids)
        set_label = _set_label(gt_new_rank, correct, gt_entry_id is not None)

        record = {
            "sample_idx": sample_idx,
            "question_global_idx": eval_question["question_global_idx"],
            "category": eval_question.get("category"),
            "question": eval_question.get("question"),
            "reference_answer": reference_answer,
            "single_gt_status": gt_question.get("single_gt_status"),
            "single_gt_entry_id": gt_entry_id,
            "gt_memory_text": gt_memory_text,
            "gt_memory_unit": gt_memory_unit,
            "retrieved_memories": retrieved_memories,
            "formatted_prompt": formatted_prompt,
            "answer": eval_question.get("answer"),
            "f1": eval_question.get("f1"),
            "bert": eval_question.get("bert"),
            "correct": correct,
            "gt_new_rank": gt_new_rank,
            "set_label": set_label,
            "mode": mode.key,
        }
        if mode.key == METHOD_MODE.key:
            record["guard_status"] = eval_question.get("guard_status")
            record["guard2_window"] = eval_question.get("guard2_window")
            record["guard2_buffered"] = eval_question.get("guard2_buffered")
        records.append(record)

    if missing_gt_rows:
        print(f"WARNING: {missing_gt_rows} GT entry_ids were not found in the DBs")
    if missing_retrieved_rows:
        print(f"WARNING: {missing_retrieved_rows} retrieved entry_ids were not found in the DBs")

    status_counts = Counter(
        (gt_by_key[_question_key(question)].get("single_gt_status") or "missing_single_gt_status")
        for question in eval_questions
    )
    samples_present = sorted({question["sample_idx"] for question in eval_questions})
    correct_count = sum(1 for record in records if record["correct"])
    mean_f1 = _mean([float(record.get("f1") or 0.0) for record in records])

    return {
        "metadata": {
            "n": len(records),
            "mode": mode.key,
            "samples_present": samples_present,
            "mean_f1": round(mean_f1, 4),
            "correct_count": correct_count,
            "correct_threshold": threshold,
            "single_gt_status_counts": dict(status_counts.most_common()),
            "source": str(merged_eval_path),
            "description": (
                f"{mode.label} context export. Each record contains the GT memory text/unit, "
                "retrieved memory units, reconstructed prompt, and answer/metric fields."
            ),
        },
        "questions": records,
    }


def _apply_mode_config(mode: EvalMode, conv_start: str, conv_end: str) -> None:
    config._last_retrieved_contexts = None
    config._guard2_window = None
    config._guard2_buffered_window = None
    runtime_local = getattr(config, "_runtime_local", None)
    if runtime_local:
        runtime_local.last_retrieved_contexts = None
        runtime_local.guard2_window = None
        runtime_local.guard2_buffered_window = None
    config.ENTITY_AWARE_ANSWER = mode.entity_aware
    config.ENABLE_TEMPORAL = mode.temporal_enabled
    if mode.temporal_enabled:
        config.GUARD1_EXTENDED = True
        config.CONV_DATE_START = conv_start
        config.CONV_DATE_END = conv_end
    else:
        config.CONV_DATE_START = ""
        config.CONV_DATE_END = ""

    # Optional ablation controls. Defaults preserve the selected eval mode.
    variant = os.getenv("CONTEXT_ABLATION_VARIANT", "full").strip().lower()
    config.ABLATE_DISABLE_CUE_GUARD = False
    config.ABLATE_DISABLE_CONTEXT_AWARE_RANKING = False
    config.ABLATE_GENERATION_TEXT_ONLY = False
    if variant in {"no_session_context", "wo_session_context", "without_session_context"}:
        config.ENABLE_TEMPORAL = False
        config.CONV_DATE_START = ""
        config.CONV_DATE_END = ""
    elif variant in {"no_cue_guard", "wo_cue_guard", "without_cue_guard"}:
        config.ABLATE_DISABLE_CUE_GUARD = True
    elif variant in {"no_context_ranking", "wo_context_ranking", "without_context_aware_ranking"}:
        config.ABLATE_DISABLE_CONTEXT_AWARE_RANKING = True
        config.SSS_USE_RRF = False
        config.SSS_ENABLE_PROX_RERANK = False
        config.USE_RRF_BASELINE = False
    elif variant in {"no_generation_context", "wo_context_preserved_generation", "without_context_preserved_generation"}:
        config.ABLATE_GENERATION_TEXT_ONLY = True


def _evaluate_records(
    spec: RunSpec,
    mode: EvalMode,
    sample_indices: list[int],
    categories: list[int],
    gt_json_path: Path,
    parallel_questions: bool = False,
    eval_workers: int = 1,
) -> tuple[list[dict], list[int], list[int], dict[tuple[int, int], dict]]:
    from ramem.main import RaMemSystem
    from model_runs.locomo import LoCoMoTester

    gt_index = _load_gt_index(gt_json_path)
    available = []
    missing = []
    for sample_idx in sample_indices:
        if spec.frozen_db_path(sample_idx).exists():
            available.append(sample_idx)
        else:
            missing.append(sample_idx)

    print(SEP)
    print(f" {mode.label}  —  {spec.backbone_label}  —  1540 track".center(80))
    print(SEP)
    print(f"  Requested samples  : {sample_indices}")
    print(f"  Requested cats     : {categories}")
    print(f"  Samples available  : {available}")
    if missing:
        print(f"  Samples MISSING DB : {missing}")
    print("  Cat 5              : EXCLUDED")
    print(f"  Backbone           : {spec.backbone_label}")
    print(f"  GT map             : {gt_json_path.name}  ({len(gt_index)} questions)")
    print(f"  Parallel questions : {parallel_questions}  (workers={eval_workers})")
    print(SEP)

    if not available:
        raise SystemExit(f"No {spec.flow_label} frozen DBs found — build at least one sample first.")

    all_records: list[dict] = []
    for sample_idx in available:
        db_path = str(spec.frozen_db_path(sample_idx))
        system = RaMemSystem(clear_db=False, db_path=db_path)
        tester = LoCoMoTester(system, str(DATASET_PATH))
        samples = tester.load_dataset()
        sample = samples[sample_idx]

        try:
            conv_start, conv_end = _derive_conv_window(sample)
        except ValueError as exc:
            print(f"  WARNING: could not derive conv window: {exc} — skipping sample {sample_idx}")
            continue

        n_questions = len(sample.qa)
        n_selected = sum(1 for qa in sample.qa if qa.category in categories)
        print(f"\n{SEP2}\n  SAMPLE {sample_idx}  ({n_questions} total, {n_selected} selected)\n{SEP2}")

        def evaluate_one(q_global_idx: int, qa) -> dict:
            trigger_type = _classify_trigger(qa.question)
            reference = str(qa.final_answer) if qa.final_answer is not None else ""
            gt_info = gt_index.get((sample_idx, q_global_idx), {})
            single_gt_entry_id = gt_info.get("single_gt_entry_id")

            _apply_mode_config(mode, conv_start, conv_end)

            t0 = time.time()
            result = tester._process_single_question(qa, q_global_idx)
            elapsed = time.time() - t0
            metrics = _numeric_metrics(result.get("metrics", {}))
            f1 = _safe_f1(result)
            bert = _safe_bert(result)
            gt_rank, gt_f1 = _find_gt_rank(reference)
            retrieved_entry_ids = _last_retrieved_entry_ids()
            gt_rank_entry = _rank_entry_id(single_gt_entry_id, retrieved_entry_ids)

            print(
                f"  [{mode.short_tag.upper()}][s{sample_idx}][cat{qa.category}][{trigger_type}][q{q_global_idx}] "
                f"{elapsed:.1f}s  F1={f1:.3f}  BERT={bert:.3f}  "
                f"GT_rank={gt_rank if gt_rank > 0 else 'miss'}(f1={gt_f1:.2f})  "
                f"retrieved={result.get('num_retrieved', 0)}"
            )
            print(
                f"      GT_entry={single_gt_entry_id or 'none'}  GT_rank_entry={gt_rank_entry if gt_rank_entry is not None else 'none'}  "
                f"saved_ids={len(retrieved_entry_ids)}  {_preview_entry_ids(retrieved_entry_ids)}"
            )

            record = {
                "sample_idx": sample_idx,
                "question_global_idx": q_global_idx,
                "question": qa.question,
                "category": qa.category,
                "trigger_type": trigger_type,
                "reference": reference,
                "single_gt_entry_id": single_gt_entry_id,
                "answer": result.get("answer", ""),
                "num_retrieved": result.get("num_retrieved", 0),
                "retrieval_time": round(result.get("retrieval_time", elapsed), 3),
                "retrieved_entry_ids": retrieved_entry_ids,
                "gt_rank_entry": gt_rank_entry,
                "gt_rank": gt_rank,
                "gt_f1": gt_f1,
                "metrics": metrics,
                "f1": round(f1, 4),
                "bert": round(bert, 4),
                "correct": f1 >= spec.correct_threshold,
            }
            if mode.key == METHOD_MODE.key:
                runtime_local = getattr(config, "_runtime_local", None)
                if runtime_local:
                    g2_window = getattr(runtime_local, "guard2_window", None)
                    g2_buffered = getattr(runtime_local, "guard2_buffered_window", None)
                else:
                    g2_window = getattr(config, "_guard2_window", None)
                    g2_buffered = getattr(config, "_guard2_buffered_window", None)
                if trigger_type == "none":
                    guard_status = "NOT_TRIGGERED"
                elif g2_window:
                    guard_status = "TRIGGERED_G2_FIRED"
                else:
                    guard_status = "TRIGGERED_G2_NULL"
                record["guard_status"] = guard_status
                record["guard2_window"] = g2_window
                record["guard2_buffered"] = g2_buffered
            return record

        selected_questions = [
            (q_global_idx, qa)
            for q_global_idx, qa in enumerate(sample.qa)
            if qa.category != 5 and qa.category in categories
        ]

        if parallel_questions and eval_workers > 1 and len(selected_questions) > 1:
            print(f"  Running {len(selected_questions)} questions with {eval_workers} eval workers")
            sample_records: list[dict] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
                future_to_question = {
                    executor.submit(evaluate_one, q_global_idx, qa): (q_global_idx, qa)
                    for q_global_idx, qa in selected_questions
                }
                for future in concurrent.futures.as_completed(future_to_question):
                    q_global_idx, qa = future_to_question[future]
                    try:
                        sample_records.append(future.result())
                    except Exception as exc:
                        print(
                            f"  [{mode.short_tag.upper()}][s{sample_idx}][cat{qa.category}][q{q_global_idx}] "
                            f"FAILED: {exc}"
                        )
            sample_records.sort(key=lambda record: record["question_global_idx"])
            all_records.extend(sample_records)
        else:
            for q_global_idx, qa in selected_questions:
                all_records.append(evaluate_one(q_global_idx, qa))

    return all_records, available, missing, gt_index


def _finalize_outputs(
    spec: RunSpec,
    mode: EvalMode,
    records: list[dict],
    available: list[int],
    missing: list[int],
    gt_json_path: Path,
    categories: list[int],
    out_file: Path,
    context_out: Path,
    require_full_gt_count: bool,
) -> None:
    _save_eval_json(
        spec=spec,
        mode=mode,
        out_file=out_file,
        all_records=records,
        available=available,
        missing=missing,
        gt_json_path=gt_json_path,
        gt_indexed_questions=len(_load_gt_index(gt_json_path)),
        categories=categories,
    )
    merged_data = _load_json(out_file)
    gt_data = _load_json(gt_json_path)
    eval_questions = sorted(
        merged_data.get("questions") or [],
        key=lambda question: (question["sample_idx"], question["question_global_idx"]),
    )
    gt_questions = gt_data.get("questions") or []
    gt_by_key = _ensure_gt_covers_eval_questions(
        gt_questions,
        eval_questions,
        require_full_count=require_full_gt_count,
    )
    print()
    print(f"{spec.flow_label} {mode.label} Summary")
    print("────────────────────────")
    print(f"  merged_eval_json : {out_file}")
    print(f"  questions        : {len(eval_questions)}")
    print(f"  samples_present  : {sorted({question['sample_idx'] for question in eval_questions})}")
    print(f"  mean_f1          : {_mean([float(question.get('f1') or 0.0) for question in eval_questions]):.4f}")
    print(f"  mean_bert        : {_mean([float(question.get('bert') or 0.0) for question in eval_questions]):.4f}")
    sample_groups = defaultdict(list)
    category_groups = defaultdict(list)
    trigger_groups = defaultdict(list)
    for record in eval_questions:
        sample_groups[record["sample_idx"]].append(record)
        category_groups[record["category"]].append(record)
        trigger_groups[record["trigger_type"]].append(record)
    _print_group_table("BY SAMPLE", {str(k): _group_stats(v) for k, v in sorted(sample_groups.items())})
    _print_group_table("BY CATEGORY", {str(k): _group_stats(v) for k, v in sorted(category_groups.items())})
    _print_group_table("BY TRIGGER TYPE", {k: _group_stats(v) for k, v in sorted(trigger_groups.items())})

    print("\nBuilding context export sample cache ...")
    sample_cache = _build_sample_cache(spec, eval_questions, gt_by_key, spec.frozen_prefix)
    context_data = _build_context_export(
        spec=spec,
        mode=mode,
        merged_eval_path=out_file,
        eval_questions=eval_questions,
        gt_by_key=gt_by_key,
        sample_cache=sample_cache,
        threshold=spec.correct_threshold,
    )
    _atomic_write_json(context_out, context_data)
    print(f"Saved {mode.label} merged eval → {out_file}")
    print(f"Saved {mode.label} contexts   → {context_out}")


def _load_shard_records(path: Path) -> tuple[list[dict], list[int], list[int], list[int], str]:
    data = _load_json(path)
    metadata = data.get("metadata") or {}
    records = data.get("questions") or []
    categories = metadata.get("categories") or []
    gt_json = metadata.get("gt_json") or ""
    return (
        records,
        metadata.get("samples_available") or [],
        metadata.get("samples_missing_db") or [],
        categories,
        gt_json,
    )


def _finalize_from_shards(
    spec: RunSpec,
    mode: EvalMode,
    gt_json_path: Path,
    require_full_gt_count: bool,
    delete_shards: bool = True,
    output_stem: str | None = None,
    context_out_override: Path | None = None,
) -> bool:
    stem = output_stem or _default_output_stem(spec, mode)
    shard_paths = _shard_paths(spec, mode, stem)
    missing_shards = [path for path in shard_paths if not path.exists()]
    if missing_shards:
        print(f"[shards] Waiting for {len(missing_shards)} shard(s) before finalizing {spec.flow_label} {mode.label}.")
        return False

    lock_path = SHARD_DIR / f"{stem}.finalize.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[shards] Another worker is already finalizing {stem}.")
        return False

    try:
        os.close(lock_fd)
        records: list[dict] = []
        available: list[int] = []
        missing: list[int] = []
        categories = list(DEFAULT_METHOD_CATEGORIES)
        for shard_path in shard_paths:
            shard_records, shard_available, shard_missing, shard_categories, _ = _load_shard_records(shard_path)
            records.extend(shard_records)
            available.extend(shard_available)
            missing.extend(shard_missing)
            if shard_categories:
                categories = shard_categories

        records.sort(key=lambda question: (question["sample_idx"], question["question_global_idx"]))
        available = sorted(set(available))
        missing = sorted(set(missing))
        if output_stem:
            out_file = SSS_DIR / f"{output_stem}.json"
            context_out = context_out_override or _output_paths(spec, mode, spec.expected_samples, categories)[1]
        else:
            out_file, context_out = _output_paths(spec, mode, spec.expected_samples, categories)
        _finalize_outputs(
            spec,
            mode,
            records,
            available,
            missing,
            gt_json_path,
            categories,
            out_file,
            context_out,
            require_full_gt_count=require_full_gt_count,
        )

        if delete_shards:
            for shard_path in shard_paths:
                shard_path.unlink(missing_ok=True)
        return True
    finally:
        lock_path.unlink(missing_ok=True)


def eval_main(spec: RunSpec, mode: EvalMode, argv: list[str] | None = None) -> None:
    configure_runtime(spec)

    parser = argparse.ArgumentParser(
        description=f"{mode.label} eval on the 1540 non-cat5 track for {spec.backbone_label}"
    )
    parser.add_argument("--samples", type=int, nargs="+", metavar="N", help="Run only these sample indices")
    parser.add_argument("--categories", type=int, nargs="+", metavar="CAT", help="Run only these categories (1-4)")
    parser.add_argument(
        "--gt-json",
        default=str(spec.gt_json),
        help=f"{spec.flow_label}-specific GT verification JSON (default: {spec.gt_json.name})",
    )
    parser.add_argument("--allow-subset-gt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--write-shard-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--auto-finalize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-shards", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--parallel-questions",
        action="store_true",
        default=getattr(config, "EVAL_PARALLEL_QUESTIONS", False),
        help="Run questions within each sample concurrently",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=getattr(config, "EVAL_MAX_WORKERS", 4),
        help="Number of per-sample question workers when --parallel-questions is enabled",
    )
    parser.add_argument("--output-stem", help=argparse.SUPPRESS)
    parser.add_argument("--context-out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    sample_indices = _parse_samples(args.samples, spec)
    categories = _parse_categories(args.categories)

    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    SSS_DIR.mkdir(exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    requested_gt_path = Path(args.gt_json).expanduser()
    if args.allow_subset_gt and requested_gt_path.exists():
        gt_json_path = requested_gt_path.resolve()
        print(f"[gt-bootstrap] Reusing subset GT verification for targeted eval: {gt_json_path}")
    else:
        gt_bootstrap = ensure_gt_verification(
            gt_json_path=requested_gt_path,
            db_prefix=spec.frozen_prefix,
            flow_label=spec.flow_label,
        )
        gt_json_path = gt_bootstrap.path

    records, available, missing, _ = _evaluate_records(
        spec,
        mode,
        sample_indices,
        categories,
        gt_json_path,
        parallel_questions=args.parallel_questions,
        eval_workers=max(1, args.eval_workers),
    )
    if not records:
        print("\nNo questions evaluated.")
        return

    if args.write_shard_only:
        if categories != DEFAULT_METHOD_CATEGORIES:
            raise SystemExit("--write-shard-only only supports the default cat1-4 flow")
        if len(sample_indices) != 1:
            raise SystemExit("--write-shard-only requires exactly one sample")
        stem = args.output_stem or _default_output_stem(spec, mode)
        shard_path = spec.shard_path(stem, sample_indices[0])
        _save_eval_json(
            spec=spec,
            mode=mode,
            out_file=shard_path,
            all_records=records,
            available=available,
            missing=missing,
            gt_json_path=gt_json_path,
            gt_indexed_questions=len(_load_gt_index(gt_json_path)),
            categories=categories,
        )
        print(f"Saved shard → {shard_path}")
        if args.auto_finalize:
            context_out_override = None
            if args.context_out:
                context_out_override = Path(args.context_out).expanduser()
                if not context_out_override.is_absolute():
                    context_out_override = ROOT / context_out_override
            _finalize_from_shards(
                spec=spec,
                mode=mode,
                gt_json_path=gt_json_path,
                require_full_gt_count=not args.allow_subset_gt,
                delete_shards=not args.keep_shards,
                output_stem=args.output_stem,
                context_out_override=context_out_override,
            )
        return

    out_file, context_out = _output_paths(spec, mode, sample_indices, categories)
    if args.output_stem:
        suffix = _selection_suffix(sample_indices, categories, spec)
        out_file = SSS_DIR / f"{args.output_stem}{suffix}.json"
    if args.context_out:
        context_out = Path(args.context_out).expanduser()
        if not context_out.is_absolute():
            context_out = ROOT / context_out
    _finalize_outputs(
        spec,
        mode,
        records,
        available,
        missing,
        gt_json_path,
        categories,
        out_file,
        context_out,
        require_full_gt_count=not args.allow_subset_gt,
    )
