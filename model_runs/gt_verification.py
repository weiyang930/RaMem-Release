#!/usr/bin/env python3
"""
Offline ground-truth memory-unit verifier for LoCoMo10.

This script inspects whether the raw LoCoMo evidence turns appear to have
corresponding memory units in the relevant frozen sample DB. It reports a
primary exists/missing/review label, finer evidence-support buckets, and a
canonical single-memory GT status for eval, plus a legacy
reference-answer-overlap candidate for comparison.

No LLM calls are made.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Iterable

try:
    import dateparser
except Exception:  # pragma: no cover - handled at runtime for clearer error
    dateparser = None


ROOT = Path(__file__).resolve().parent.parent  # Simple_mem_Agentic/ root
sys.path.insert(0, str(ROOT))

from ramem.db_layout import GPT_FROZEN_PREFIX

DEFAULT_DATASET = ROOT / "locomo10.json"
DEFAULT_OUT = ROOT / "results/locomo_contexts" / "gt_memory_verification.json"
DEFAULT_DB_PREFIX = GPT_FROZEN_PREFIX
DEFAULT_TABLE_NAME = "memory_entries"


@dataclass(frozen=True)
class TokenOverlap:
    precision: float
    recall: float
    f1: float
    overlap_tokens: list[str]
    prediction_tokens: list[str]
    reference_tokens: list[str]


@dataclass(frozen=True)
class WeightedOverlap:
    precision: float
    recall: float
    f1: float
    overlap_tokens: list[str]
    overlap_weight: float
    prediction_weight: float
    reference_weight: float


def tokenize(text: str) -> list[str]:
    """Plain lowercase word/number tokens for direct reference-answer overlap."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def normalize_for_phrase(text: str) -> str:
    return " ".join(tokenize(text))


def parse_reference_units(reference_answer: str) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for raw in (reference_answer or "").split(","):
        text = raw.strip()
        normalized = normalize_for_phrase(text)
        if normalized:
            units.append({"text": text, "normalized": normalized})
    return units


def phrase_in_normalized_text(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def token_overlap(prediction: str, reference: str) -> TokenOverlap:
    return token_overlap_from_sets(set(tokenize(prediction)), set(tokenize(reference)))


def token_overlap_from_sets(pred_set: set[str], ref_set: set[str]) -> TokenOverlap:
    common = sorted(pred_set & ref_set)

    if not pred_set or not ref_set:
        return TokenOverlap(0.0, 0.0, 0.0, common, sorted(pred_set), sorted(ref_set))

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TokenOverlap(precision, recall, f1, common, sorted(pred_set), sorted(ref_set))


def weighted_overlap_from_sets(
    pred_set: set[str],
    ref_set: set[str],
    token_weights: dict[str, float],
) -> WeightedOverlap:
    common = sorted(pred_set & ref_set)

    def weight(tokens: set[str]) -> float:
        return sum(token_weights.get(tok, 1.0) for tok in tokens)

    pred_weight = weight(pred_set)
    ref_weight = weight(ref_set)
    overlap_weight = weight(set(common))
    if pred_weight <= 0.0 or ref_weight <= 0.0:
        return WeightedOverlap(0.0, 0.0, 0.0, common, overlap_weight, pred_weight, ref_weight)

    precision = overlap_weight / pred_weight
    recall = overlap_weight / ref_weight
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return WeightedOverlap(
        precision,
        recall,
        f1,
        common,
        overlap_weight,
        pred_weight,
        ref_weight,
    )


def pct(value: float) -> float:
    return round(100.0 * value, 2)


def progress(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[gt-verify] {message}", file=sys.stderr, flush=True)


def parse_datetime(text: str | None) -> datetime | None:
    if not text or not dateparser:
        return None
    return dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    if isinstance(text, str):
        iso_candidate = text[:10]
        try:
            return date.fromisoformat(iso_candidate)
        except ValueError:
            pass
    parsed = parse_datetime(text)
    return parsed.date() if parsed else None


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def combine_turn_text(turn: dict[str, Any]) -> str:
    text = turn.get("text") or ""
    caption = turn.get("blip_caption")
    if ("img_url" in turn or caption) and caption:
        image_text = f"[Image: {caption}]"
        return f"{image_text} {text}".strip() if text else image_text
    return text


def normalize_evidence_ids(raw_values: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    Extract D<session>:<turn> ids from normal and mildly malformed evidence fields.

    Examples handled:
      D8:6; D9:17
      D9:1 D4:4 D4:6
      D:11:26
      D30:05
    """
    ids: list[str] = []
    unparsed: list[str] = []
    seen = set()

    patterns = [
        re.compile(r"D\s*(\d+)\s*:\s*(\d+)", re.IGNORECASE),
        re.compile(r"D\s*:\s*(\d+)\s*:\s*(\d+)", re.IGNORECASE),
    ]

    for raw in raw_values or []:
        raw_str = str(raw)
        matches: list[tuple[str, str]] = []
        for pattern in patterns:
            matches.extend(pattern.findall(raw_str))

        if not matches:
            unparsed.append(raw_str)
            continue

        for sess, turn in matches:
            norm = f"D{int(sess)}:{int(turn)}"
            if norm not in seen:
                seen.add(norm)
                ids.append(norm)

    return ids, unparsed


def build_session_index(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    conv = sample.get("conversation") or {}
    sessions: dict[int, dict[str, Any]] = {}

    for key, value in conv.items():
        m = re.fullmatch(r"session_(\d+)", key)
        if not m or not isinstance(value, list):
            continue

        session_id = int(m.group(1))
        raw_date = conv.get(f"session_{session_id}_date_time") or ""
        parsed = parse_datetime(raw_date)
        iso_datetime = parsed.strftime("%Y-%m-%dT%H:%M:%S") if parsed else raw_date
        iso_date = parsed.date().isoformat() if parsed else ""

        turns_by_exact_id: dict[str, dict[str, Any]] = {}
        turns_by_int_id: dict[int, dict[str, Any]] = {}
        for turn in value:
            dia_id = str(turn.get("dia_id") or "")
            turns_by_exact_id[dia_id] = turn
            tm = re.fullmatch(r"D\d+:(\d+)", dia_id)
            if tm:
                turns_by_int_id[int(tm.group(1))] = turn

        sessions[session_id] = {
            "session_id": session_id,
            "raw_date_time": raw_date,
            "iso_datetime": iso_datetime,
            "iso_date": iso_date,
            "turns_by_exact_id": turns_by_exact_id,
            "turns_by_int_id": turns_by_int_id,
        }

    return sessions


def lookup_evidence(
    sample: dict[str, Any],
    raw_evidence: list[str],
) -> dict[str, Any]:
    sessions = build_session_index(sample)
    evidence_ids, unparsed_values = normalize_evidence_ids(raw_evidence)

    turns: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    seen_text_keys = set()

    for evidence_id in evidence_ids:
        m = re.fullmatch(r"D(\d+):(\d+)", evidence_id)
        if not m:
            missing_ids.append(evidence_id)
            continue

        session_id = int(m.group(1))
        turn_num = int(m.group(2))
        session = sessions.get(session_id)
        if not session:
            missing_ids.append(evidence_id)
            continue

        exact_id = f"D{session_id}:{turn_num}"
        turn = session["turns_by_exact_id"].get(exact_id)
        if not turn:
            turn = session["turns_by_int_id"].get(turn_num)
        if not turn:
            missing_ids.append(evidence_id)
            continue

        text = combine_turn_text(turn)
        dedupe_key = (session_id, turn_num, text)
        if dedupe_key in seen_text_keys:
            continue
        seen_text_keys.add(dedupe_key)

        turns.append(
            {
                "evidence_id": evidence_id,
                "session_id": session_id,
                "turn_id": turn.get("dia_id") or evidence_id,
                "speaker": turn.get("speaker") or "",
                "text": text,
                "session_raw_date_time": session["raw_date_time"],
                "session_iso_datetime": session["iso_datetime"],
                "session_iso_date": session["iso_date"],
            }
        )

    return {
        "raw_ids": raw_evidence or [],
        "parsed_ids": evidence_ids,
        "unparsed_values": unparsed_values,
        "missing_ids": missing_ids,
        "turns": turns,
        "text": "\n".join(t["text"] for t in turns),
        "session_dates": sorted({t["session_iso_date"] for t in turns if t["session_iso_date"]}),
    }


def reference_answer_for_qa(qa: dict[str, Any], cat5_reference: str) -> str:
    category = qa.get("category")
    if category == 5:
        if cat5_reference == "not-mentioned":
            return "Not mentioned in the conversation"
        if cat5_reference == "answer":
            return str(qa.get("answer") or "")
        return str(qa.get("adversarial_answer") or qa.get("answer") or "")
    return str(qa.get("answer") or "")


def load_memory_records(
    db_path: Path,
    table_name: str,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load all memory records from a frozen sample DB with the same table path used
    by the working debug scripts.
    """
    db_path = db_path.resolve()
    progress(f"opening memory DB: {db_path}", show_progress)

    try:
        import lancedb
        version = getattr(lancedb, "__version__", "unknown")
    except Exception as exc:
        raise RuntimeError(f"could not import lancedb: {type(exc).__name__}: {exc}") from exc

    listed: Any = "<not listed>"
    try:
        db = lancedb.connect(str(db_path))
        try:
            listed = db.list_tables()
        except Exception:
            try:
                listed = db.table_names()
            except Exception as list_exc:
                listed = f"<could not list tables: {list_exc}>"
        progress(f"lancedb tables: {listed}", show_progress)
        table = db.open_table(table_name)
        rows = table.to_arrow().to_pylist()
        progress(f"loaded {len(rows)} memory rows via lancedb", show_progress)
        return rows, f"lancedb:{version}"
    except Exception as exc:
        error = f"lancedb:{version} tables={listed}: {type(exc).__name__}: {exc}"

    table_dir = db_path / f"{table_name}.lance"
    layout_hint = ""
    if table_dir.exists():
        has_transactions = (table_dir / "_transactions").exists()
        has_versions = (table_dir / "_versions").exists()
        layout_hint = (
            f"\n  table_dir_exists=True _transactions={has_transactions} _versions={has_versions}"
        )
        if has_transactions and not has_versions:
            layout_hint += (
                "\n  hint: this looks like the older LanceDB table layout used by the frozen "
                "DBs. Newer raw-lance readers expect _versions and may fail. Try running "
                "against the frozen DB copy that your working debug scripts use, or restore "
                "the _versions directory for this DB. For example, if the working DBs are in "
                "/Users/brycekan/Downloads/LocoMemTemp_fix/Simple_mem_Agentic, pass that path "
                "with --db-root."
            )
    else:
        layout_hint = f"\n  table_dir_exists=False expected={table_dir}"

    raise RuntimeError(f"could not load memory records from {db_path}{layout_hint}\n  {error}")


def slim_memory_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_id": record.get("entry_id"),
        "text": record.get("lossless_restatement") or "",
        "timestamp": record.get("timestamp") or "",
        "session_date": record.get("session_date") or "",
        "session_end_date": record.get("session_end_date") or "",
        "mention_date": record.get("mention_date") or "",
        "location": record.get("location") or "",
        "persons": list(record.get("persons") or []),
        "entities": list(record.get("entities") or []),
        "topic": record.get("topic") or "",
    }


def prepare_memory_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        slim = slim_memory_record(record)
        cand_start = parse_date(slim.get("session_date"))
        slim["_text_token_set"] = set(tokenize(slim["text"]))
        slim["_normalized_text"] = normalize_for_phrase(slim["text"])
        slim["_session_start_date"] = cand_start
        slim["_session_end_date"] = parse_date(slim.get("session_end_date")) or cand_start
        prepared.append(slim)

    doc_freq: Counter[str] = Counter()
    for record in prepared:
        doc_freq.update(record.get("_text_token_set") or set())
    n_docs = max(len(prepared), 1)
    token_idf = {
        tok: math.log((1 + n_docs) / (1 + df)) + 1.0
        for tok, df in doc_freq.items()
    }
    return prepared, token_idf


def public_memory_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def session_distance_days_from_dates(
    cand_start: date | None,
    cand_end: date | None,
    ev_dates: list[date],
) -> int | None:
    if not ev_dates:
        return None

    if cand_start is None:
        return None

    best: int | None = None
    for ev_date in ev_dates:
        if cand_end and cand_start <= ev_date <= cand_end:
            dist = 0
        elif ev_date < cand_start:
            dist = (cand_start - ev_date).days
        elif cand_end:
            dist = (ev_date - cand_end).days
        else:
            dist = abs((cand_start - ev_date).days)
        best = dist if best is None else min(best, dist)
    return best


def score_candidate(
    record: dict[str, Any],
    reference_token_set: set[str],
    reference_units: list[dict[str, str]],
    evidence_token_set: set[str],
    evidence_dates: list[date],
) -> dict[str, Any]:
    candidate_token_set = record.get("_text_token_set") or set()
    candidate_normalized_text = record.get("_normalized_text") or ""
    ref = token_overlap_from_sets(candidate_token_set, reference_token_set)
    ev = token_overlap_from_sets(candidate_token_set, evidence_token_set)
    reference_unit_hits = [
        unit["text"]
        for unit in reference_units
        if phrase_in_normalized_text(unit["normalized"], candidate_normalized_text)
    ]
    reference_unit_recall = (
        len(reference_unit_hits) / len(reference_units) if reference_units else 0.0
    )
    distance = session_distance_days_from_dates(
        record.get("_session_start_date"),
        record.get("_session_end_date"),
        evidence_dates,
    )

    scored = public_memory_record(record)
    scored.update(
        {
            "reference_precision": round(ref.precision, 6),
            "reference_recall": round(ref.recall, 6),
            "reference_f1": round(ref.f1, 6),
            "reference_precision_pct": pct(ref.precision),
            "reference_recall_pct": pct(ref.recall),
            "reference_f1_pct": pct(ref.f1),
            "reference_overlap_tokens": ref.overlap_tokens,
            "reference_unit_recall": round(reference_unit_recall, 6),
            "reference_unit_recall_pct": pct(reference_unit_recall),
            "reference_unit_hits": reference_unit_hits,
            "reference_unit_count": len(reference_units),
            "evidence_precision": round(ev.precision, 6),
            "evidence_recall": round(ev.recall, 6),
            "evidence_f1": round(ev.f1, 6),
            "evidence_precision_pct": pct(ev.precision),
            "evidence_recall_pct": pct(ev.recall),
            "evidence_f1_pct": pct(ev.f1),
            "evidence_overlap_tokens": ev.overlap_tokens,
            "session_distance_days": distance,
            "session_match": distance == 0 if distance is not None else False,
        }
    )
    return scored


def score_question_candidate(
    record: dict[str, Any],
    question_token_set: set[str],
    evidence_token_set: set[str],
    evidence_dates: list[date],
    token_idf: dict[str, float],
) -> dict[str, Any]:
    candidate_token_set = record.get("_text_token_set") or set()
    question_raw = token_overlap_from_sets(candidate_token_set, question_token_set)
    question_idf = weighted_overlap_from_sets(candidate_token_set, question_token_set, token_idf)
    evidence_raw = token_overlap_from_sets(candidate_token_set, evidence_token_set)
    distance = session_distance_days_from_dates(
        record.get("_session_start_date"),
        record.get("_session_end_date"),
        evidence_dates,
    )

    scored = public_memory_record(record)
    scored.update(
        {
            "question_precision": round(question_raw.precision, 6),
            "question_recall": round(question_raw.recall, 6),
            "question_f1": round(question_raw.f1, 6),
            "question_precision_pct": pct(question_raw.precision),
            "question_recall_pct": pct(question_raw.recall),
            "question_f1_pct": pct(question_raw.f1),
            "question_overlap_tokens": question_raw.overlap_tokens,
            "question_idf_precision": round(question_idf.precision, 6),
            "question_idf_recall": round(question_idf.recall, 6),
            "question_idf_f1": round(question_idf.f1, 6),
            "question_idf_precision_pct": pct(question_idf.precision),
            "question_idf_recall_pct": pct(question_idf.recall),
            "question_idf_f1_pct": pct(question_idf.f1),
            "question_idf_overlap_tokens": question_idf.overlap_tokens,
            "evidence_precision": round(evidence_raw.precision, 6),
            "evidence_recall": round(evidence_raw.recall, 6),
            "evidence_f1": round(evidence_raw.f1, 6),
            "evidence_precision_pct": pct(evidence_raw.precision),
            "evidence_recall_pct": pct(evidence_raw.recall),
            "evidence_f1_pct": pct(evidence_raw.f1),
            "evidence_overlap_tokens": evidence_raw.overlap_tokens,
            "session_distance_days": distance,
            "session_match": distance == 0 if distance is not None else False,
        }
    )
    return scored


def candidate_sort_key(candidate: dict[str, Any]) -> tuple:
    distance = candidate.get("session_distance_days")
    distance_sort = 10**9 if distance is None else distance
    return (
        -candidate.get("reference_unit_recall", 0.0),
        -candidate.get("reference_recall", 0.0),
        -int(bool(candidate.get("session_match"))),
        distance_sort,
        -candidate.get("evidence_f1", 0.0),
        -candidate.get("reference_f1", 0.0),
        candidate.get("entry_id") or "",
    )


def evidence_candidate_sort_key(candidate: dict[str, Any]) -> tuple:
    distance = candidate.get("session_distance_days")
    distance_sort = 10**9 if distance is None else distance
    return (
        -int(bool(candidate.get("session_interval_match"))),
        distance_sort,
        -candidate.get("evidence_turn_idf_f1", 0.0),
        -candidate.get("evidence_turn_idf_recall", 0.0),
        -candidate.get("evidence_turn_f1", 0.0),
        -int(bool(candidate.get("exact_session_datetime_match"))),
        -int(bool(candidate.get("exact_session_date_match"))),
        -int(bool(candidate.get("speaker_match"))),
        -candidate.get("reference_unit_recall", 0.0),
        -candidate.get("reference_recall", 0.0),
        candidate.get("entry_id") or "",
    )


def answer_bearing_sort_key(candidate: dict[str, Any]) -> tuple:
    distance = candidate.get("session_distance_days")
    distance_sort = 10**9 if distance is None else distance
    return (
        -int(bool(candidate.get("session_match"))),
        distance_sort,
        -candidate.get("reference_unit_recall", 0.0),
        -candidate.get("reference_recall", 0.0),
        -candidate.get("reference_f1", 0.0),
        -candidate.get("evidence_f1", 0.0),
        candidate.get("entry_id") or "",
    )


def question_anchor_sort_key(candidate: dict[str, Any]) -> tuple:
    distance = candidate.get("session_distance_days")
    distance_sort = 10**9 if distance is None else distance
    return (
        -int(bool(candidate.get("session_match"))),
        distance_sort,
        -candidate.get("question_idf_f1", 0.0),
        -candidate.get("question_idf_recall", 0.0),
        -candidate.get("question_f1", 0.0),
        -candidate.get("evidence_f1", 0.0),
        candidate.get("entry_id") or "",
    )


def has_contentful_reference_overlap(
    candidate: dict[str, Any],
    token_idf: dict[str, float],
) -> bool:
    if candidate.get("reference_unit_recall", 0.0) > 0.0:
        return True
    overlap_tokens = set(candidate.get("reference_overlap_tokens") or [])
    if not overlap_tokens:
        return False

    # Use the actual memory corpus to decide whether an overlap token is
    # contentful, instead of hard-coding stopwords.
    max_overlap_idf = max(token_idf.get(token, 1.0) for token in overlap_tokens)
    return max_overlap_idf >= 2.0 and candidate.get("reference_recall", 0.0) >= 0.25


def score_evidence_candidate(
    record: dict[str, Any],
    evidence_turn: dict[str, Any],
    evidence_token_set: set[str],
    reference_token_set: set[str],
    reference_units: list[dict[str, str]],
    token_idf: dict[str, float],
) -> dict[str, Any]:
    candidate_token_set = record.get("_text_token_set") or set()
    candidate_normalized_text = record.get("_normalized_text") or ""
    evidence_raw = token_overlap_from_sets(candidate_token_set, evidence_token_set)
    evidence_idf = weighted_overlap_from_sets(candidate_token_set, evidence_token_set, token_idf)
    ref = token_overlap_from_sets(candidate_token_set, reference_token_set)
    reference_unit_hits = [
        unit["text"]
        for unit in reference_units
        if phrase_in_normalized_text(unit["normalized"], candidate_normalized_text)
    ]
    reference_unit_recall = (
        len(reference_unit_hits) / len(reference_units) if reference_units else 0.0
    )

    ev_date = parse_date(evidence_turn.get("session_iso_date"))
    distance = session_distance_days_from_dates(
        record.get("_session_start_date"),
        record.get("_session_end_date"),
        [ev_date] if ev_date is not None else [],
    )
    exact_date = bool(ev_date and record.get("_session_start_date") == ev_date)
    ev_datetime = (evidence_turn.get("session_iso_datetime") or "")[:19]
    record_datetime = (record.get("session_date") or record.get("timestamp") or "")[:19]
    exact_datetime = bool(ev_datetime and ev_datetime == record_datetime)
    speaker = (evidence_turn.get("speaker") or "").strip().lower()
    persons = {str(person).strip().lower() for person in (record.get("persons") or [])}

    scored = public_memory_record(record)
    scored.update(
        {
            "evidence_turn_precision": round(evidence_raw.precision, 6),
            "evidence_turn_recall": round(evidence_raw.recall, 6),
            "evidence_turn_f1": round(evidence_raw.f1, 6),
            "evidence_turn_precision_pct": pct(evidence_raw.precision),
            "evidence_turn_recall_pct": pct(evidence_raw.recall),
            "evidence_turn_f1_pct": pct(evidence_raw.f1),
            "evidence_turn_overlap_tokens": evidence_raw.overlap_tokens,
            "evidence_turn_idf_precision": round(evidence_idf.precision, 6),
            "evidence_turn_idf_recall": round(evidence_idf.recall, 6),
            "evidence_turn_idf_f1": round(evidence_idf.f1, 6),
            "evidence_turn_idf_precision_pct": pct(evidence_idf.precision),
            "evidence_turn_idf_recall_pct": pct(evidence_idf.recall),
            "evidence_turn_idf_f1_pct": pct(evidence_idf.f1),
            "evidence_turn_idf_overlap_tokens": evidence_idf.overlap_tokens,
            "reference_precision": round(ref.precision, 6),
            "reference_recall": round(ref.recall, 6),
            "reference_f1": round(ref.f1, 6),
            "reference_precision_pct": pct(ref.precision),
            "reference_recall_pct": pct(ref.recall),
            "reference_f1_pct": pct(ref.f1),
            "reference_overlap_tokens": ref.overlap_tokens,
            "reference_unit_recall": round(reference_unit_recall, 6),
            "reference_unit_recall_pct": pct(reference_unit_recall),
            "reference_unit_hits": reference_unit_hits,
            "reference_unit_count": len(reference_units),
            "session_distance_days": distance,
            "session_interval_match": distance == 0 if distance is not None else False,
            "exact_session_date_match": exact_date,
            "exact_session_datetime_match": exact_datetime,
            "speaker": evidence_turn.get("speaker") or "",
            "speaker_match": bool(speaker and speaker in persons),
        }
    )
    return scored


def analyze_answer_bearing(
    memory_records: list[dict[str, Any]],
    reference_token_set: set[str],
    reference_units: list[dict[str, str]],
    evidence_token_set: set[str],
    evidence_dates: list[date],
    token_idf: dict[str, float],
    top_n: int,
) -> dict[str, Any]:
    scored = heapq.nsmallest(
        top_n,
        (
            score_candidate(
                record,
                reference_token_set=reference_token_set,
                reference_units=reference_units,
                evidence_token_set=evidence_token_set,
                evidence_dates=evidence_dates,
            )
            for record in memory_records
        ),
        key=answer_bearing_sort_key,
    )
    in_window = [candidate for candidate in scored if candidate.get("session_match")]
    top_candidate = in_window[0] if in_window else (scored[0] if scored else None)
    exact_unit_candidates = [
        candidate for candidate in in_window if candidate.get("reference_unit_recall", 0.0) > 0.0
    ]
    token_candidates = [
        candidate
        for candidate in in_window
        if has_contentful_reference_overlap(candidate, token_idf)
    ]

    if exact_unit_candidates:
        status = "direct_answer_gt"
        selected_answer_candidate = exact_unit_candidates[0]
    elif token_candidates:
        status = "partial_answer_gt"
        selected_answer_candidate = token_candidates[0]
    elif top_candidate and top_candidate.get("session_match"):
        status = "no_literal_answer_in_evidence_window"
        selected_answer_candidate = top_candidate
    elif top_candidate:
        status = "answer_candidate_outside_evidence_window"
        selected_answer_candidate = top_candidate
    else:
        status = "no_answer_candidate"
        selected_answer_candidate = None

    return {
        "answer_bearing_status": status,
        "top_candidate": selected_answer_candidate,
        "top_candidates": scored,
        "in_evidence_window_count": len(in_window),
        "exact_unit_in_evidence_window_count": len(exact_unit_candidates),
        "token_overlap_in_evidence_window_count": len(token_candidates),
    }


def classify_question_anchor(
    top_candidate: dict[str, Any] | None,
) -> str:
    if not top_candidate:
        return "no_question_candidate"

    if not top_candidate.get("session_match"):
        return "question_candidate_outside_evidence_window"

    question_idf_f1 = top_candidate.get("question_idf_f1", 0.0) or 0.0
    question_idf_recall = top_candidate.get("question_idf_recall", 0.0) or 0.0
    evidence_f1 = top_candidate.get("evidence_f1", 0.0) or 0.0

    if question_idf_f1 >= 0.18 or question_idf_recall >= 0.35:
        return "strong_question_match_in_evidence_window"
    if question_idf_f1 >= 0.10 or question_idf_recall >= 0.20 or evidence_f1 >= 0.15:
        return "weak_question_match_in_evidence_window"
    return "question_candidate_in_evidence_window"


def analyze_question_anchor(
    memory_records: list[dict[str, Any]],
    question_token_set: set[str],
    evidence_token_set: set[str],
    evidence_dates: list[date],
    token_idf: dict[str, float],
    top_n: int,
) -> dict[str, Any]:
    scored = heapq.nsmallest(
        top_n,
        (
            score_question_candidate(
                record,
                question_token_set=question_token_set,
                evidence_token_set=evidence_token_set,
                evidence_dates=evidence_dates,
                token_idf=token_idf,
            )
            for record in memory_records
        ),
        key=question_anchor_sort_key,
    )
    in_window = [candidate for candidate in scored if candidate.get("session_match")]
    top_candidate = in_window[0] if in_window else (scored[0] if scored else None)
    return {
        "question_anchor_status": classify_question_anchor(top_candidate),
        "top_candidate": top_candidate,
        "top_candidates": scored,
        "in_evidence_window_count": len(in_window),
    }


def classify_evidence_coverage(
    top_candidate: dict[str, Any] | None,
    has_interval_candidate: bool,
    strong_f1: float,
    likely_f1: float,
) -> str:
    if not top_candidate:
        return "strong_missing"

    top_f1 = top_candidate.get("evidence_turn_idf_f1", 0.0)
    top_recall = top_candidate.get("evidence_turn_idf_recall", 0.0)
    in_interval = bool(top_candidate.get("session_interval_match"))

    if in_interval and top_f1 >= strong_f1:
        return "strong_exists"
    if in_interval and (
        top_f1 >= likely_f1
        or top_recall >= 0.35
        or top_candidate.get("reference_unit_recall", 0.0) > 0.0
    ):
        return "likely_exists"
    if has_interval_candidate:
        return "uncertain"
    if top_f1 >= likely_f1:
        return "likely_missing"
    return "strong_missing"


def analyze_evidence_coverage(
    evidence: dict[str, Any],
    memory_records: list[dict[str, Any]],
    reference_token_set: set[str],
    reference_units: list[dict[str, str]],
    token_idf: dict[str, float],
    strong_f1: float,
    likely_f1: float,
    top_n: int,
) -> dict[str, Any]:
    evidence_items: list[dict[str, Any]] = []
    for evidence_turn in evidence.get("turns") or []:
        evidence_token_set = set(tokenize(evidence_turn.get("text") or ""))
        scored_iter = (
            score_evidence_candidate(
                record,
                evidence_turn=evidence_turn,
                evidence_token_set=evidence_token_set,
                reference_token_set=reference_token_set,
                reference_units=reference_units,
                token_idf=token_idf,
            )
            for record in memory_records
        )
        all_scored = list(scored_iter)
        has_interval_candidate = any(
            c.get("session_interval_match") for c in all_scored
        )
        top_candidates = heapq.nsmallest(
            top_n,
            all_scored,
            key=evidence_candidate_sort_key,
        )
        top_candidate = top_candidates[0] if top_candidates else None
        status = classify_evidence_coverage(
            top_candidate,
            has_interval_candidate=has_interval_candidate,
            strong_f1=strong_f1,
            likely_f1=likely_f1,
        )
        evidence_items.append(
            {
                "evidence_id": evidence_turn.get("evidence_id"),
                "speaker": evidence_turn.get("speaker") or "",
                "session_iso_datetime": evidence_turn.get("session_iso_datetime"),
                "session_iso_date": evidence_turn.get("session_iso_date"),
                "text": evidence_turn.get("text") or "",
                "coverage_status": status,
                "top_candidate": top_candidate,
                "top_candidates": top_candidates,
            }
        )

    statuses = [item["coverage_status"] for item in evidence_items]
    missing_statuses = {"strong_missing", "likely_missing"}
    exists_statuses = {"strong_exists", "likely_exists"}
    if not evidence_items:
        question_status = "no_evidence"
    elif any(status in missing_statuses for status in statuses):
        question_status = "incomplete_or_missing"
    elif any(status == "uncertain" for status in statuses):
        question_status = "needs_review"
    elif all(status == "strong_exists" for status in statuses):
        question_status = "strong_support_exists"
    elif all(status in exists_statuses for status in statuses):
        question_status = "likely_support_exists"
    else:
        question_status = "needs_review"

    gt_memory_set: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in evidence_items:
        if item.get("coverage_status") in missing_statuses:
            continue
        candidate = item.get("top_candidate")
        if not candidate:
            continue
        entry_id = candidate.get("entry_id")
        if not entry_id or entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        gt_memory_set.append(candidate)

    return {
        "question_support_status": question_status,
        "required_evidence_count": len(evidence_items),
        "strong_exists_count": sum(1 for status in statuses if status == "strong_exists"),
        "likely_exists_count": sum(1 for status in statuses if status == "likely_exists"),
        "uncertain_count": sum(1 for status in statuses if status == "uncertain"),
        "likely_missing_count": sum(1 for status in statuses if status == "likely_missing"),
        "strong_missing_count": sum(1 for status in statuses if status == "strong_missing"),
        "covered_evidence_count": sum(1 for status in statuses if status in exists_statuses),
        "evidence_items": evidence_items,
        "candidate_gt_memory_set": gt_memory_set,
    }


def gt_support_binary(question_support_status: str) -> str:
    if question_support_status in {"strong_support_exists", "likely_support_exists"}:
        return "exists"
    if question_support_status in {"incomplete_or_missing", "no_evidence"}:
        return "missing"
    return "review"


TEMPORAL_QUESTION_PREFIXES = (
    "when ",
    "what year ",
    "what month ",
    "what date ",
    "what day ",
    "how long ",
    "how many years ",
    "how many months ",
    "how many weeks ",
    "how many days ",
)


def is_temporal_question(question: str) -> bool:
    normalized = (question or "").strip().lower()
    return normalized.startswith(TEMPORAL_QUESTION_PREFIXES)


def candidate_partial_bucket_for_question(
    single_gt_status: str,
    question_text: str,
    reference_units: list[dict[str, str]],
    answer_top_candidate: dict[str, Any] | None,
) -> str | None:
    if single_gt_status != "candidate_partial":
        return None

    if len(reference_units) > 1:
        return "candidate_partial_multi_unit"

    if is_temporal_question(question_text):
        return "candidate_partial_single_unit_temporal"

    reference_recall = 0.0
    if answer_top_candidate:
        reference_recall = answer_top_candidate.get("reference_recall", 0.0) or 0.0
    if reference_recall >= 0.75:
        return "candidate_partial_single_unit_high_recall"
    return "candidate_partial_single_unit_other"


def single_gt_status_for_question(
    support_binary: str,
    top_candidate: dict[str, Any] | None,
    question_anchor: dict[str, Any] | None = None,
) -> str:
    if support_binary == "missing":
        return "no_support_in_db"

    qa_status = (question_anchor or {}).get("question_anchor_status") or ""
    qa_top = (question_anchor or {}).get("top_candidate")
    qa_strong_in_window = bool(
        qa_top and qa_top.get("session_match")
        and qa_status == "strong_question_match_in_evidence_window"
    )

    if not top_candidate:
        # No answer-bearing candidate; fall back to question-anchor rescue.
        if qa_strong_in_window:
            return "confident_session_anchor"
        return "not_confident_single_gt"

    reference_unit_recall = top_candidate.get("reference_unit_recall", 0.0) or 0.0
    reference_recall = top_candidate.get("reference_recall", 0.0) or 0.0
    evidence_f1 = top_candidate.get("evidence_f1", 0.0) or 0.0
    session_distance = top_candidate.get("session_distance_days")

    # Strict canonical one-memory GT: the single candidate must cover all
    # comma-split reference units.
    if reference_unit_recall >= 1.0:
        return "confident_direct"

    # Relaxed one-memory candidate: useful as an upper bound / review target,
    # but not strict enough to serve as the canonical GT by itself.
    if reference_recall >= 1.0:
        return "candidate_partial"
    if (
        reference_recall >= 0.5
        and evidence_f1 >= 0.2
        and session_distance == 0
    ):
        return "candidate_partial"

    # Session-anchored GT: memory is in the evidence window and topically
    # related to the evidence turns. Covers temporal questions where the
    # answer date lives in session_date metadata rather than the memory text.
    if session_distance == 0 and evidence_f1 >= 0.10:
        return "confident_session_anchor"

    # Fallback: strong question-entity overlap in the evidence window handles
    # sparse evidence turns where evidence_f1 is low but the memory is
    # clearly about the same event as the question.
    if qa_strong_in_window:
        return "confident_session_anchor"

    return "not_confident_single_gt"


def classify_gt_existence(
    support_status: str,
    support_binary: str,
    answer_status: str,
    answer_top_candidate: dict[str, Any] | None,
    question_anchor: dict[str, Any],
) -> tuple[str, str]:
    question_status = question_anchor.get("question_anchor_status") or "no_question_candidate"

    if support_binary == "exists":
        return "exists", "evidence_support_exists"

    answer_session_match = bool(answer_top_candidate and answer_top_candidate.get("session_match"))
    answer_direct = answer_status == "direct_answer_gt"
    answer_partial = answer_status == "partial_answer_gt"

    if support_binary == "missing":
        if support_status != "no_evidence":
            if answer_session_match and (answer_direct or answer_partial):
                return "review", "answer_overlap_with_incomplete_support"
            if question_status == "strong_question_match_in_evidence_window":
                return "review", "question_match_with_incomplete_support"
            return "review", "incomplete_or_missing_support"
        if answer_session_match and (answer_direct or answer_partial):
            return "review", "answer_overlap_but_missing_support"
        if question_status == "strong_question_match_in_evidence_window":
            return "review", "question_match_but_missing_support"
        return "missing", "no_support_in_db"

    if answer_session_match and (answer_direct or answer_partial):
        return "exists", "answer_overlap_in_evidence_window"
    if question_status == "strong_question_match_in_evidence_window":
        return "exists", "question_match_in_evidence_window"
    if question_status == "weak_question_match_in_evidence_window":
        return "review", "weak_question_match_in_evidence_window"
    return "review", "needs_manual_review"


def analyze_question(
    sample_idx: int,
    question_idx: int,
    qa: dict[str, Any],
    sample: dict[str, Any],
    memory_records: list[dict[str, Any]],
    token_idf: dict[str, float],
    cat5_reference: str,
    min_reference_f1: float,
    strong_evidence_f1: float,
    likely_evidence_f1: float,
    evidence_top_n: int,
) -> dict[str, Any]:
    question_text = qa.get("question") or ""
    reference_answer = reference_answer_for_qa(qa, cat5_reference)
    evidence = lookup_evidence(sample, list(qa.get("evidence") or []))
    question_tokens = tokenize(question_text)
    question_token_set = set(question_tokens)
    reference_tokens = tokenize(reference_answer)
    reference_token_set = set(reference_tokens)
    reference_units = parse_reference_units(reference_answer)
    evidence_token_set = set(tokenize(evidence["text"]))
    evidence_dates = [parse_date(d) for d in evidence["session_dates"]]
    evidence_dates = [d for d in evidence_dates if d is not None]
    evidence_coverage = analyze_evidence_coverage(
        evidence=evidence,
        memory_records=memory_records,
        reference_token_set=reference_token_set,
        reference_units=reference_units,
        token_idf=token_idf,
        strong_f1=strong_evidence_f1,
        likely_f1=likely_evidence_f1,
        top_n=evidence_top_n,
    )
    answer_bearing = analyze_answer_bearing(
        memory_records=memory_records,
        reference_token_set=reference_token_set,
        reference_units=reference_units,
        evidence_token_set=evidence_token_set,
        evidence_dates=evidence_dates,
        token_idf=token_idf,
        top_n=max(evidence_top_n, 4),
    )
    question_anchor = analyze_question_anchor(
        memory_records=memory_records,
        question_token_set=question_token_set,
        evidence_token_set=evidence_token_set,
        evidence_dates=evidence_dates,
        token_idf=token_idf,
        top_n=max(evidence_top_n, 4),
    )

    scored = heapq.nsmallest(
        4,
        (
            score_candidate(
                record,
                reference_token_set=reference_token_set,
                reference_units=reference_units,
                evidence_token_set=evidence_token_set,
                evidence_dates=evidence_dates,
            )
            for record in memory_records
        ),
        key=candidate_sort_key,
    )

    selected = scored[0] if scored else None
    alternatives = scored[1:4] if len(scored) > 1 else []
    support_status = evidence_coverage["question_support_status"]
    support_binary = gt_support_binary(support_status)
    answer_status = answer_bearing["answer_bearing_status"]
    answer_top_candidate = answer_bearing.get("top_candidate")
    single_gt_status = single_gt_status_for_question(support_binary, answer_top_candidate, question_anchor)
    candidate_partial_bucket = candidate_partial_bucket_for_question(
        single_gt_status=single_gt_status,
        question_text=question_text,
        reference_units=reference_units,
        answer_top_candidate=answer_top_candidate,
    )
    existence_bucket, existence_reason = classify_gt_existence(
        support_status=support_status,
        support_binary=support_binary,
        answer_status=answer_status,
        answer_top_candidate=answer_top_candidate,
        question_anchor=question_anchor,
    )

    qa_anchor_top = question_anchor.get("top_candidate")
    qa_anchor_strong_in_window = bool(
        qa_anchor_top and qa_anchor_top.get("session_match")
        and question_anchor.get("question_anchor_status") == "strong_question_match_in_evidence_window"
    )

    if single_gt_status == "confident_direct":
        single_gt_memory = answer_top_candidate
    elif single_gt_status == "confident_session_anchor":
        # Prefer question_anchor candidate (ranked by question-entity overlap,
        # more topically relevant for temporal/metadata questions) when strong.
        single_gt_memory = qa_anchor_top if qa_anchor_strong_in_window else answer_top_candidate
    else:
        single_gt_memory = None

    if single_gt_status in {"confident_direct", "candidate_partial"}:
        single_gt_relaxed_memory = answer_top_candidate
    elif single_gt_status == "confident_session_anchor":
        single_gt_relaxed_memory = single_gt_memory
    else:
        single_gt_relaxed_memory = None

    return {
        "sample_idx": sample_idx,
        "question_global_idx": question_idx,
        "category": qa.get("category"),
        "question": question_text,
        "question_tokens": question_tokens,
        "answer": qa.get("answer"),
        "adversarial_answer": qa.get("adversarial_answer"),
        "reference_answer": reference_answer,
        "reference_answer_tokens": reference_tokens,
        "reference_answer_units": [unit["text"] for unit in reference_units],
        "cat5_reference_mode": cat5_reference if qa.get("category") == 5 else None,
        "evidence": evidence,
        "evidence_coverage": evidence_coverage,
        "gt_support_status": support_status,
        "gt_support_binary": support_binary,
        "gt_existence_bucket": existence_bucket,
        "gt_existence_reason": existence_reason,
        "gt_required_evidence_count": evidence_coverage["required_evidence_count"],
        "evidence_source_gt_set": evidence_coverage["candidate_gt_memory_set"],
        "gt_memory_set": evidence_coverage["candidate_gt_memory_set"],
        "answer_bearing_gt": answer_bearing,
        "question_anchor_gt": question_anchor,
        "single_gt_status": single_gt_status,
        "candidate_partial_bucket": candidate_partial_bucket,
        "single_gt_exists": bool(single_gt_memory),
        "single_gt_confident": single_gt_status == "confident_direct",
        "single_gt_memory": single_gt_memory,
        "single_gt_set": [single_gt_memory] if single_gt_memory else [],
        "single_gt_relaxed_exists": bool(single_gt_relaxed_memory),
        "single_gt_relaxed_memory": single_gt_relaxed_memory,
        "single_gt_relaxed_set": [single_gt_relaxed_memory] if single_gt_relaxed_memory else [],
        "answer_bearing_gt_set": (
            [single_gt_relaxed_memory] if single_gt_relaxed_memory else []
        ),
        "candidate_gt_memory_set": evidence_coverage["candidate_gt_memory_set"],
        "gt_answer_status": answer_status,
        "selected_gt": selected,
        "top_alternatives": alternatives,
        "reference_answer_all_units_in_selected_gt": bool(
            selected
            and reference_units
            and selected.get("reference_unit_recall", 0.0) >= 1.0
        ),
        "reference_answer_any_unit_in_selected_gt": bool(
            selected
            and reference_units
            and selected.get("reference_unit_recall", 0.0) > 0.0
        ),
        "reference_answer_any_token_in_selected_gt": bool(
            selected and selected.get("reference_recall", 0.0) > 0.0
        ),
        "gt_exists_confident": bool(
            selected
            and reference_answer
            and selected.get("reference_f1", 0.0) >= min_reference_f1
        ),
    }


def parse_samples(raw_samples: list[str] | None, dataset_len: int) -> list[int] | None:
    if not raw_samples:
        return None
    if any(s.lower() == "all" for s in raw_samples):
        return list(range(dataset_len))

    out: list[int] = []
    for raw in raw_samples:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part)
            except ValueError:
                raise SystemExit(f"invalid sample index: {part!r}")
            if idx < 0 or idx >= dataset_len:
                raise SystemExit(f"sample index out of range: {idx} (dataset has {dataset_len})")
            out.append(idx)
    return sorted(set(out))


def question_key_matches(raw_key: str | None, sample_idx: int, question_idx: int) -> bool:
    if not raw_key:
        return True
    m = re.fullmatch(r"(\d+)\s*[:/,-]\s*(\d+)", raw_key.strip())
    if not m:
        raise SystemExit("--question-key must look like sample_idx:question_idx, e.g. 3:42")
    return sample_idx == int(m.group(1)) and question_idx == int(m.group(2))


def format_score(candidate: dict[str, Any], prefix: str) -> str:
    return (
        f"{prefix} overlap: "
        f"f1={candidate.get(prefix + '_f1_pct', 0.0):.2f}% "
        f"precision={candidate.get(prefix + '_precision_pct', 0.0):.2f}% "
        f"recall={candidate.get(prefix + '_recall_pct', 0.0):.2f}% "
        f"tokens={candidate.get(prefix + '_overlap_tokens', [])}"
    )


def format_reference_unit_score(candidate: dict[str, Any]) -> str:
    return (
        "reference unit overlap: "
        f"recall={candidate.get('reference_unit_recall_pct', 0.0):.2f}% "
        f"hits={candidate.get('reference_unit_hits', [])}"
    )


def format_evidence_turn_score(candidate: dict[str, Any]) -> str:
    return (
        "evidence-turn overlap: "
        f"idf_f1={candidate.get('evidence_turn_idf_f1_pct', 0.0):.2f}% "
        f"idf_precision={candidate.get('evidence_turn_idf_precision_pct', 0.0):.2f}% "
        f"idf_recall={candidate.get('evidence_turn_idf_recall_pct', 0.0):.2f}% "
        f"raw_f1={candidate.get('evidence_turn_f1_pct', 0.0):.2f}% "
        f"tokens={candidate.get('evidence_turn_overlap_tokens', [])}"
    )


def format_question_score(candidate: dict[str, Any]) -> str:
    return (
        "question overlap: "
        f"idf_f1={candidate.get('question_idf_f1_pct', 0.0):.2f}% "
        f"idf_recall={candidate.get('question_idf_recall_pct', 0.0):.2f}% "
        f"raw_f1={candidate.get('question_f1_pct', 0.0):.2f}% "
        f"tokens={candidate.get('question_overlap_tokens', [])}"
    )


def print_candidate(candidate: dict[str, Any], indent: str = "") -> None:
    print(f"{indent}entry_id: {candidate.get('entry_id')}")
    print(
        f"{indent}session: {candidate.get('session_date') or ''}"
        f" -> {candidate.get('session_end_date') or ''}"
    )
    print(f"{indent}{format_score(candidate, 'reference')}")
    print(f"{indent}{format_reference_unit_score(candidate)}")
    if "evidence_turn_f1" in candidate:
        print(f"{indent}{format_evidence_turn_score(candidate)}")
    elif "question_f1" in candidate:
        print(f"{indent}{format_question_score(candidate)}")
        print(f"{indent}{format_score(candidate, 'evidence')}")
    else:
        print(f"{indent}{format_score(candidate, 'evidence')}")
    print(f"{indent}session_distance_days: {candidate.get('session_distance_days')}")
    print(f"{indent}text:")
    print(f"{indent}{candidate.get('text') or ''}")


def print_gt_memory_set(record: dict[str, Any]) -> None:
    candidates = record.get("gt_memory_set") or []
    print("evidence-source candidate gt memory set:")
    if not candidates:
        print("  none")
        return
    for idx, candidate in enumerate(candidates, 1):
        print(f"  candidate {idx}")
        print_candidate(candidate, indent="    ")


def print_evidence_coverage(record: dict[str, Any]) -> None:
    coverage = record.get("evidence_coverage") or {}
    print(
        "gt support status: "
        f"{record.get('gt_support_status')} ({record.get('gt_support_binary')})"
    )
    print(
        "gt existence bucket: "
        f"{record.get('gt_existence_bucket')} ({record.get('gt_existence_reason')})"
    )
    print(
        "evidence coverage counts: "
        f"required={coverage.get('required_evidence_count', 0)} "
        f"strong={coverage.get('strong_exists_count', 0)} "
        f"likely={coverage.get('likely_exists_count', 0)} "
        f"uncertain={coverage.get('uncertain_count', 0)} "
        f"likely_missing={coverage.get('likely_missing_count', 0)} "
        f"strong_missing={coverage.get('strong_missing_count', 0)}"
    )
    items = coverage.get("evidence_items") or []
    if not items:
        return
    print("evidence-turn coverage:")
    for item in items:
        top = item.get("top_candidate") or {}
        print(
            f"  {item.get('evidence_id')}: {item.get('coverage_status')} "
            f"idf_f1={top.get('evidence_turn_idf_f1_pct', 0.0):.2f}% "
            f"raw_f1={top.get('evidence_turn_f1_pct', 0.0):.2f}% "
            f"interval={top.get('session_interval_match')} "
            f"exact_date={top.get('exact_session_date_match')} "
            f"speaker_match={top.get('speaker_match')} "
            f"entry_id={top.get('entry_id')}"
        )
        if top.get("text"):
            print(f"    top memory: {top.get('text')}")


def print_question_anchor(record: dict[str, Any]) -> None:
    question_anchor = record.get("question_anchor_gt") or {}
    top = question_anchor.get("top_candidate") or {}
    print(f"question-anchor status: {question_anchor.get('question_anchor_status')}")
    if not top:
        return
    print(
        "question-anchor top candidate: "
        f"question_idf_f1={top.get('question_idf_f1_pct', 0.0):.2f}% "
        f"question_idf_recall={top.get('question_idf_recall_pct', 0.0):.2f}% "
        f"session_match={top.get('session_match')} "
        f"entry_id={top.get('entry_id')}"
    )
    if top.get("text"):
        print(f"  top question memory: {top.get('text')}")


def print_answer_bearing(record: dict[str, Any]) -> None:
    answer_bearing = record.get("answer_bearing_gt") or {}
    top = answer_bearing.get("top_candidate") or {}
    print(f"answer-bearing status: {record.get('gt_answer_status')}")
    if not top:
        return
    print(
        "answer-bearing top candidate: "
        f"ref_unit={top.get('reference_unit_recall_pct', 0.0):.2f}% "
        f"ref_token={top.get('reference_recall_pct', 0.0):.2f}% "
        f"session_match={top.get('session_match')} "
        f"entry_id={top.get('entry_id')}"
    )
    if top.get("text"):
        print(f"  top answer memory: {top.get('text')}")


def print_single_gt(record: dict[str, Any]) -> None:
    print(f"single-memory gt status: {record.get('single_gt_status')}")
    if record.get("candidate_partial_bucket"):
        print(f"candidate-partial bucket: {record.get('candidate_partial_bucket')}")
    candidate = record.get("single_gt_memory")
    relaxed = record.get("single_gt_relaxed_memory")
    print("single-memory gt (strict):")
    if not candidate:
        print("  none")
    else:
        print_candidate(candidate, indent="  ")
    print("single-memory gt (relaxed candidate):")
    if not relaxed:
        print("  none")
        return
    print_candidate(relaxed, indent="  ")


def print_record(record: dict[str, Any], ordinal: int, total: int | str) -> None:
    print("")
    print(f"record {ordinal} of {total}")
    print(f"sample/question: {record['sample_idx']}:{record['question_global_idx']}")
    print(f"category: {record.get('category')}")
    if record.get("candidate_partial_bucket"):
        print(f"candidate-partial bucket: {record.get('candidate_partial_bucket')}")
    print("question:")
    print(record.get("question") or "")
    print("reference answer:")
    print(record.get("reference_answer") or "")
    print(f"reference answer tokens: {record.get('reference_answer_tokens', [])}")
    print(f"reference answer comma units: {record.get('reference_answer_units', [])}")
    if record.get("answer") is not None:
        print("dataset answer:")
        print(record.get("answer"))
    if record.get("adversarial_answer") is not None:
        print("adversarial answer:")
        print(record.get("adversarial_answer"))

    evidence = record.get("evidence") or {}
    print(f"evidence ids: {evidence.get('raw_ids', [])}")
    print(f"parsed evidence ids: {evidence.get('parsed_ids', [])}")
    if evidence.get("unparsed_values"):
        print(f"unparsed evidence values: {evidence.get('unparsed_values')}")
    if evidence.get("missing_ids"):
        print(f"missing evidence ids: {evidence.get('missing_ids')}")
    print(f"evidence session dates: {evidence.get('session_dates', [])}")
    print("evidence text:")
    print(evidence.get("text") or "")
    print_evidence_coverage(record)
    print_gt_memory_set(record)
    print_single_gt(record)
    print_question_anchor(record)
    print_answer_bearing(record)

    selected = record.get("selected_gt")
    print("reference-overlap candidate (legacy comparator):")
    if selected:
        print_candidate(selected, indent="  ")
    else:
        print("  none")

    alternatives = record.get("top_alternatives") or []
    print("top reference-overlap alternatives:")
    if not alternatives:
        print("  none")
    for idx, candidate in enumerate(alternatives, 1):
        print(f"  alternative {idx}")
        print_candidate(candidate, indent="    ")
    sys.stdout.flush()


def save_json(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def evenly_spaced(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if limit >= len(items):
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]

    selected: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    for idx in range(limit):
        pos = round(idx * (len(items) - 1) / (limit - 1))
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        selected.append(items[pos])
    return selected


def inspection_bucket_for_record(record: dict[str, Any]) -> str:
    status = record.get("single_gt_status") or "unknown"
    if status == "candidate_partial":
        return record.get("candidate_partial_bucket") or "candidate_partial_other"
    return status


def select_inspection_records(
    records: list[dict[str, Any]],
    print_limit: int,
) -> list[dict[str, Any]]:
    quotas = [
        ("confident_direct", 5),
        ("confident_session_anchor", 10),
        ("candidate_partial_single_unit_temporal", 8),
        ("candidate_partial_multi_unit", 8),
        ("candidate_partial_single_unit_high_recall", 5),
        ("candidate_partial_single_unit_other", 5),
        ("not_confident_single_gt", 5),
        ("no_support_in_db", 4),
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[inspection_bucket_for_record(record)].append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def add_records(candidates: list[dict[str, Any]], count: int) -> None:
        for record in evenly_spaced(candidates, count):
            obj_id = id(record)
            if obj_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(obj_id)

    for bucket, quota in quotas:
        add_records(grouped.get(bucket, []), quota)

    if len(selected) < print_limit:
        # Fill remaining slots from the rest of the pool in a stable, inspection-first
        # order so candidate_partial spillover is still visible.
        bucket_priority = [bucket for bucket, _ in quotas] + [
            "candidate_partial_other",
            "unknown",
        ]
        for bucket in bucket_priority:
            for record in grouped.get(bucket, []):
                obj_id = id(record)
                if obj_id in selected_ids:
                    continue
                selected.append(record)
                selected_ids.add(obj_id)
                if len(selected) >= print_limit:
                    return selected[:print_limit]

        for record in records:
            obj_id = id(record)
            if obj_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(obj_id)
            if len(selected) >= print_limit:
                break

    return selected[:print_limit]


def select_print_records(
    records: list[dict[str, Any]],
    print_limit: int,
    print_all: bool,
    print_mode: str,
) -> list[dict[str, Any]]:
    if print_all:
        return records
    if print_limit <= 0:
        return []
    if print_mode == "first":
        return records[:print_limit]
    if print_mode == "inspect":
        return select_inspection_records(records, print_limit)

    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[int(record["sample_idx"])].append(record)

    sample_ids = sorted(by_sample)
    if not sample_ids:
        return []

    base = print_limit // len(sample_ids)
    extra = print_limit % len(sample_ids)
    selected: list[dict[str, Any]] = []
    for idx, sample_idx in enumerate(sample_ids):
        quota = base + (1 if idx < extra else 0)
        selected.extend(evenly_spaced(by_sample[sample_idx], quota))

    if len(selected) < print_limit:
        selected_ids = {id(record) for record in selected}
        for record in records:
            if id(record) not in selected_ids:
                selected.append(record)
                selected_ids.add(id(record))
            if len(selected) >= print_limit:
                break

    return selected[:print_limit]


def concise_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        "entry_id": candidate.get("entry_id"),
        "session_date": candidate.get("session_date"),
        "session_end_date": candidate.get("session_end_date"),
        "text": candidate.get("text"),
        "exact_session_date_match": candidate.get("exact_session_date_match"),
        "exact_session_datetime_match": candidate.get("exact_session_datetime_match"),
        "session_interval_match": candidate.get("session_interval_match"),
        "speaker_match": candidate.get("speaker_match"),
        "session_distance_days": candidate.get("session_distance_days"),
        "evidence_turn_idf_f1_pct": candidate.get("evidence_turn_idf_f1_pct"),
        "evidence_turn_idf_recall_pct": candidate.get("evidence_turn_idf_recall_pct"),
        "evidence_turn_f1_pct": candidate.get("evidence_turn_f1_pct"),
        "question_f1_pct": candidate.get("question_f1_pct"),
        "question_idf_f1_pct": candidate.get("question_idf_f1_pct"),
        "question_idf_recall_pct": candidate.get("question_idf_recall_pct"),
        "question_overlap_tokens": candidate.get("question_overlap_tokens"),
        "reference_recall_pct": candidate.get("reference_recall_pct"),
        "reference_unit_recall_pct": candidate.get("reference_unit_recall_pct"),
        "reference_unit_hits": candidate.get("reference_unit_hits"),
    }


def concise_answer_bearing(answer_bearing: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_bearing_status": answer_bearing.get("answer_bearing_status"),
        "top_candidate": concise_candidate(answer_bearing.get("top_candidate")),
        "top_candidates": [concise_candidate(c) for c in (answer_bearing.get("top_candidates") or [])],
        "in_evidence_window_count": answer_bearing.get("in_evidence_window_count"),
        "exact_unit_in_evidence_window_count": answer_bearing.get(
            "exact_unit_in_evidence_window_count"
        ),
        "token_overlap_in_evidence_window_count": answer_bearing.get(
            "token_overlap_in_evidence_window_count"
        ),
    }


def concise_question_anchor(question_anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_anchor_status": question_anchor.get("question_anchor_status"),
        "top_candidate": concise_candidate(question_anchor.get("top_candidate")),
        "top_candidates": [concise_candidate(c) for c in (question_anchor.get("top_candidates") or [])],
        "in_evidence_window_count": question_anchor.get("in_evidence_window_count"),
    }


def concise_memory_list(candidates: list[dict[str, Any]] | None, limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in (candidates or [])[:limit]:
        concise = concise_candidate(candidate)
        if concise:
            out.append(concise)
    return out


def concise_record_for_review(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence") or {}
    return {
        "sample_idx": record.get("sample_idx"),
        "question_global_idx": record.get("question_global_idx"),
        "category": record.get("category"),
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "reference_answer_units": record.get("reference_answer_units"),
        "gt_support_status": record.get("gt_support_status"),
        "gt_support_binary": record.get("gt_support_binary"),
        "gt_existence_bucket": record.get("gt_existence_bucket"),
        "gt_existence_reason": record.get("gt_existence_reason"),
        "single_gt_status": record.get("single_gt_status"),
        "candidate_partial_bucket": record.get("candidate_partial_bucket"),
        "gt_answer_status": record.get("gt_answer_status"),
        "evidence_ids": evidence.get("parsed_ids") or evidence.get("raw_ids") or [],
        "evidence_session_dates": evidence.get("session_dates") or [],
        "evidence_text": evidence.get("text") or "",
        "evidence_source_gt_set": concise_memory_list(record.get("evidence_source_gt_set"), limit=5),
        "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
        "single_gt_relaxed_memory": concise_candidate(record.get("single_gt_relaxed_memory")),
        "answer_bearing_gt": {
            "answer_bearing_status": (record.get("answer_bearing_gt") or {}).get("answer_bearing_status"),
            "top_candidate": concise_candidate((record.get("answer_bearing_gt") or {}).get("top_candidate")),
            "top_candidates": concise_memory_list((record.get("answer_bearing_gt") or {}).get("top_candidates"), limit=5),
        },
        "question_anchor_gt": concise_question_anchor(record.get("question_anchor_gt") or {}),
        "legacy_reference_overlap_candidate": concise_candidate(record.get("selected_gt")),
        "legacy_reference_overlap_alternatives": concise_memory_list(record.get("top_alternatives"), limit=3),
        "review_label": None,
        "review_confidence": None,
        "review_notes": "",
    }


def build_upper_bound_review_queue(records: list[dict[str, Any]]) -> dict[str, Any]:
    queue = [
        concise_record_for_review(record)
        for record in records
        if record.get("single_gt_status") in {
            "candidate_partial",
            "not_confident_single_gt",
            "no_support_in_db",
        }
    ]
    by_bucket = Counter(
        record.get("candidate_partial_bucket") or record.get("single_gt_status") or "unknown"
        for record in records
        if record.get("single_gt_status") in {
            "candidate_partial",
            "not_confident_single_gt",
            "no_support_in_db",
        }
    )
    return {
        "n_questions": len(queue),
        "by_bucket": dict(sorted(by_bucket.items())),
        "note": (
            "Review queue for adjudicating the approximate one-memory upper bound. "
            "Use review_label to mark one_memory_exists, db_support_but_not_one_memory, "
            "likely_no_db_support, or uncertain."
        ),
        "questions": queue,
    }


def evidence_bucket_example(record: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_idx": record.get("sample_idx"),
        "question_global_idx": record.get("question_global_idx"),
        "category": record.get("category"),
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "gt_support_status": record.get("gt_support_status"),
        "gt_support_binary": record.get("gt_support_binary"),
        "gt_existence_bucket": record.get("gt_existence_bucket"),
        "gt_existence_reason": record.get("gt_existence_reason"),
        "single_gt_status": record.get("single_gt_status"),
        "candidate_partial_bucket": record.get("candidate_partial_bucket"),
        "gt_answer_status": record.get("gt_answer_status"),
        "evidence_id": item.get("evidence_id"),
        "evidence_status": item.get("coverage_status"),
        "evidence_speaker": item.get("speaker"),
        "evidence_session_iso_datetime": item.get("session_iso_datetime"),
        "evidence_text": item.get("text"),
        "top_candidate": concise_candidate(item.get("top_candidate")),
        "top_candidates": [concise_candidate(c) for c in (item.get("top_candidates") or [])],
        "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
        "single_gt_relaxed_memory": concise_candidate(record.get("single_gt_relaxed_memory")),
        "answer_bearing_gt": concise_answer_bearing(record.get("answer_bearing_gt") or {}),
        "question_anchor_gt": concise_question_anchor(record.get("question_anchor_gt") or {}),
    }


def build_evidence_coverage_summary(
    records: list[dict[str, Any]],
    example_limit: int,
) -> dict[str, Any]:
    by_evidence_status: Counter[str] = Counter()
    by_question_status: Counter[str] = Counter()
    by_binary_status: Counter[str] = Counter()
    by_existence_bucket: Counter[str] = Counter()
    by_existence_reason: Counter[str] = Counter()
    by_question_anchor_status: Counter[str] = Counter()
    by_single_gt_status: Counter[str] = Counter()
    by_candidate_partial_bucket: Counter[str] = Counter()
    by_answer_status: Counter[str] = Counter()
    examples_by_evidence_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_question_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_existence_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_single_gt_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_candidate_partial_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_answer_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_question_anchor_status: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        question_status = record.get("gt_support_status") or "unknown"
        binary_status = record.get("gt_support_binary") or gt_support_binary(question_status)
        existence_bucket = record.get("gt_existence_bucket") or "unknown"
        existence_reason = record.get("gt_existence_reason") or "unknown"
        question_anchor = record.get("question_anchor_gt") or {}
        question_anchor_status = question_anchor.get("question_anchor_status") or "unknown"
        single_gt_status = record.get("single_gt_status") or "unknown"
        candidate_partial_bucket = record.get("candidate_partial_bucket") or "not_applicable"
        answer_status = record.get("gt_answer_status") or "unknown"
        by_question_status[question_status] += 1
        by_binary_status[binary_status] += 1
        by_existence_bucket[existence_bucket] += 1
        by_existence_reason[existence_reason] += 1
        by_question_anchor_status[question_anchor_status] += 1
        by_single_gt_status[single_gt_status] += 1
        by_candidate_partial_bucket[candidate_partial_bucket] += 1
        by_answer_status[answer_status] += 1
        coverage = record.get("evidence_coverage") or {}
        items = coverage.get("evidence_items") or []

        if len(examples_by_existence_bucket[existence_bucket]) < example_limit:
            examples_by_existence_bucket[existence_bucket].append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "question_global_idx": record.get("question_global_idx"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "gt_support_status": question_status,
                    "gt_support_binary": binary_status,
                    "gt_existence_bucket": existence_bucket,
                    "gt_existence_reason": existence_reason,
                    "question_anchor_gt": concise_question_anchor(question_anchor),
                    "single_gt_status": single_gt_status,
                    "candidate_partial_bucket": record.get("candidate_partial_bucket"),
                    "gt_answer_status": answer_status,
                    "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
                    "single_gt_relaxed_memory": concise_candidate(
                        record.get("single_gt_relaxed_memory")
                    ),
                    "answer_bearing_gt": concise_answer_bearing(
                        record.get("answer_bearing_gt") or {}
                    ),
                }
            )

        if len(examples_by_single_gt_status[single_gt_status]) < example_limit:
            examples_by_single_gt_status[single_gt_status].append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "question_global_idx": record.get("question_global_idx"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "gt_support_status": question_status,
                    "gt_support_binary": binary_status,
                    "gt_existence_bucket": existence_bucket,
                    "gt_existence_reason": existence_reason,
                    "question_anchor_gt": concise_question_anchor(question_anchor),
                    "single_gt_status": single_gt_status,
                    "gt_answer_status": answer_status,
                    "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
                    "single_gt_relaxed_memory": concise_candidate(
                        record.get("single_gt_relaxed_memory")
                    ),
                    "answer_bearing_gt": concise_answer_bearing(
                        record.get("answer_bearing_gt") or {}
                    ),
                }
            )

        if (
            candidate_partial_bucket != "not_applicable"
            and len(examples_by_candidate_partial_bucket[candidate_partial_bucket]) < example_limit
        ):
            examples_by_candidate_partial_bucket[candidate_partial_bucket].append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "question_global_idx": record.get("question_global_idx"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "gt_support_status": question_status,
                    "gt_support_binary": binary_status,
                    "gt_existence_bucket": existence_bucket,
                    "gt_existence_reason": existence_reason,
                    "question_anchor_gt": concise_question_anchor(question_anchor),
                    "single_gt_status": single_gt_status,
                    "candidate_partial_bucket": record.get("candidate_partial_bucket"),
                    "gt_answer_status": answer_status,
                    "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
                    "single_gt_relaxed_memory": concise_candidate(
                        record.get("single_gt_relaxed_memory")
                    ),
                    "answer_bearing_gt": concise_answer_bearing(
                        record.get("answer_bearing_gt") or {}
                    ),
                }
            )

        if len(examples_by_answer_status[answer_status]) < example_limit:
            examples_by_answer_status[answer_status].append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "question_global_idx": record.get("question_global_idx"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "gt_support_status": question_status,
                    "gt_support_binary": binary_status,
                    "gt_existence_bucket": existence_bucket,
                    "gt_existence_reason": existence_reason,
                    "question_anchor_gt": concise_question_anchor(question_anchor),
                    "single_gt_status": single_gt_status,
                    "candidate_partial_bucket": record.get("candidate_partial_bucket"),
                    "gt_answer_status": answer_status,
                    "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
                    "single_gt_relaxed_memory": concise_candidate(
                        record.get("single_gt_relaxed_memory")
                    ),
                    "answer_bearing_gt": concise_answer_bearing(
                        record.get("answer_bearing_gt") or {}
                    ),
                }
            )

        if len(examples_by_question_status[question_status]) < example_limit:
            examples_by_question_status[question_status].append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "question_global_idx": record.get("question_global_idx"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "gt_support_status": question_status,
                    "gt_support_binary": binary_status,
                    "gt_existence_bucket": existence_bucket,
                    "gt_existence_reason": existence_reason,
                    "question_anchor_gt": concise_question_anchor(question_anchor),
                    "single_gt_status": single_gt_status,
                    "candidate_partial_bucket": record.get("candidate_partial_bucket"),
                    "gt_answer_status": answer_status,
                    "single_gt_memory": concise_candidate(record.get("single_gt_memory")),
                    "single_gt_relaxed_memory": concise_candidate(
                        record.get("single_gt_relaxed_memory")
                    ),
                    "answer_bearing_gt": concise_answer_bearing(
                        record.get("answer_bearing_gt") or {}
                    ),
                    "evidence_statuses": [
                        {
                            "evidence_id": item.get("evidence_id"),
                            "coverage_status": item.get("coverage_status"),
                            "top_candidate": concise_candidate(item.get("top_candidate")),
                        }
                        for item in items
                    ],
                }
            )

        if len(examples_by_question_anchor_status[question_anchor_status]) < example_limit:
            examples_by_question_anchor_status[question_anchor_status].append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "question_global_idx": record.get("question_global_idx"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "gt_support_status": question_status,
                    "gt_support_binary": binary_status,
                    "gt_existence_bucket": existence_bucket,
                    "gt_existence_reason": existence_reason,
                    "question_anchor_gt": concise_question_anchor(question_anchor),
                    "single_gt_status": single_gt_status,
                    "candidate_partial_bucket": record.get("candidate_partial_bucket"),
                    "gt_answer_status": answer_status,
                }
            )

        for item in items:
            status = item.get("coverage_status") or "unknown"
            by_evidence_status[status] += 1
            if len(examples_by_evidence_status[status]) < example_limit:
                examples_by_evidence_status[status].append(evidence_bucket_example(record, item))

    return {
        "by_evidence_status": dict(sorted(by_evidence_status.items())),
        "by_question_status": dict(sorted(by_question_status.items())),
        "by_binary_status": dict(sorted(by_binary_status.items())),
        "by_existence_bucket": dict(sorted(by_existence_bucket.items())),
        "by_existence_reason": dict(sorted(by_existence_reason.items())),
        "by_question_anchor_status": dict(sorted(by_question_anchor_status.items())),
        "by_single_gt_status": dict(sorted(by_single_gt_status.items())),
        "by_candidate_partial_bucket": dict(sorted(by_candidate_partial_bucket.items())),
        "by_answer_status": dict(sorted(by_answer_status.items())),
        "examples_by_evidence_status": {
            status: examples
            for status, examples in sorted(examples_by_evidence_status.items())
        },
        "examples_by_question_status": {
            status: examples
            for status, examples in sorted(examples_by_question_status.items())
        },
        "examples_by_existence_bucket": {
            status: examples
            for status, examples in sorted(examples_by_existence_bucket.items())
        },
        "examples_by_question_anchor_status": {
            status: examples
            for status, examples in sorted(examples_by_question_anchor_status.items())
        },
        "examples_by_single_gt_status": {
            status: examples
            for status, examples in sorted(examples_by_single_gt_status.items())
        },
        "examples_by_candidate_partial_bucket": {
            status: examples
            for status, examples in sorted(examples_by_candidate_partial_bucket.items())
        },
        "examples_by_answer_status": {
            status: examples
            for status, examples in sorted(examples_by_answer_status.items())
        },
    }


def print_evidence_bucket_examples(summary: dict[str, Any]) -> None:
    examples = summary.get("examples_by_evidence_status") or {}
    counts = summary.get("by_evidence_status") or {}
    if not examples:
        return
    print("")
    print("evidence coverage bucket examples")
    for status in ("strong_exists", "likely_exists", "uncertain", "likely_missing", "strong_missing"):
        status_examples = examples.get(status) or []
        if not status_examples:
            continue
        print("")
        print(f"{status} ({counts.get(status, 0)} evidence turns)")
        for example in status_examples:
            top = example.get("top_candidate") or {}
            print(
                f"  sample/question {example.get('sample_idx')}:{example.get('question_global_idx')} "
                f"evidence={example.get('evidence_id')} "
                f"idf_f1={top.get('evidence_turn_idf_f1_pct', 0.0):.2f}% "
                f"interval={top.get('session_interval_match')} "
                f"exact_date={top.get('exact_session_date_match')} "
                f"entry={top.get('entry_id')}"
            )
            print(f"    q: {example.get('question')}")
            print(f"    ref: {example.get('reference_answer')}")
            print(f"    top memory: {top.get('text')}")

    existence_examples = summary.get("examples_by_existence_bucket") or {}
    existence_counts = summary.get("by_existence_bucket") or {}
    if existence_examples:
        print("")
        print("gt existence bucket examples")
        for status in ("exists", "review", "missing"):
            status_examples = existence_examples.get(status) or []
            if not status_examples:
                continue
            print("")
            print(f"{status} ({existence_counts.get(status, 0)} questions)")
            for example in status_examples:
                top = (example.get("question_anchor_gt") or {}).get("top_candidate") or {}
                print(
                    f"  sample/question {example.get('sample_idx')}:{example.get('question_global_idx')} "
                    f"reason={example.get('gt_existence_reason')} "
                    f"question_idf_f1={top.get('question_idf_f1_pct', 0.0):.2f}% "
                    f"entry={top.get('entry_id')}"
                )
                print(f"    q: {example.get('question')}")
                print(f"    ref: {example.get('reference_answer')}")
                print(f"    question-anchor memory: {top.get('text')}")

    answer_examples = summary.get("examples_by_answer_status") or {}
    answer_counts = summary.get("by_answer_status") or {}
    if not answer_examples:
        return
    print("")
    print("answer-bearing bucket examples")
    for status in (
        "direct_answer_gt",
        "partial_answer_gt",
        "no_literal_answer_in_evidence_window",
        "answer_candidate_outside_evidence_window",
        "no_answer_candidate",
    ):
        status_examples = answer_examples.get(status) or []
        if not status_examples:
            continue
        print("")
        print(f"{status} ({answer_counts.get(status, 0)} questions)")
        for example in status_examples:
            top = (example.get("answer_bearing_gt") or {}).get("top_candidate") or {}
            print(
                f"  sample/question {example.get('sample_idx')}:{example.get('question_global_idx')} "
                f"ref_token={top.get('reference_recall_pct', 0.0):.2f}% "
                f"ref_unit={top.get('reference_unit_recall_pct', 0.0):.2f}% "
                f"session_match={top.get('session_distance_days') == 0} "
                f"entry={top.get('entry_id')}"
            )
            print(f"    q: {example.get('question')}")
            print(f"    ref: {example.get('reference_answer')}")
            print(f"    top answer memory: {top.get('text')}")

    single_gt_examples = summary.get("examples_by_single_gt_status") or {}
    single_gt_counts = summary.get("by_single_gt_status") or {}
    if not single_gt_examples:
        return
    print("")
    print("single-memory gt bucket examples")
    for status in (
        "confident_direct",
        "confident_session_anchor",
        "candidate_partial",
        "not_confident_single_gt",
        "no_support_in_db",
    ):
        status_examples = single_gt_examples.get(status) or []
        if not status_examples:
            continue
        print("")
        print(f"{status} ({single_gt_counts.get(status, 0)} questions)")
        for example in status_examples:
            top = example.get("single_gt_memory") or {}
            relaxed = example.get("single_gt_relaxed_memory") or {}
            print(
                f"  sample/question {example.get('sample_idx')}:{example.get('question_global_idx')} "
                f"answer_status={example.get('gt_answer_status')} "
                f"entry={top.get('entry_id') or relaxed.get('entry_id')}"
            )
            print(f"    q: {example.get('question')}")
            print(f"    ref: {example.get('reference_answer')}")
            if example.get("candidate_partial_bucket"):
                print(f"    candidate-partial bucket: {example.get('candidate_partial_bucket')}")
            if top.get("text"):
                print(f"    single gt memory: {top.get('text')}")
            elif relaxed.get("text"):
                print(f"    relaxed single gt memory: {relaxed.get('text')}")

    candidate_partial_examples = summary.get("examples_by_candidate_partial_bucket") or {}
    candidate_partial_counts = summary.get("by_candidate_partial_bucket") or {}
    if candidate_partial_examples:
        print("")
        print("candidate-partial bucket examples")
        for status in (
            "candidate_partial_single_unit_temporal",
            "candidate_partial_single_unit_high_recall",
            "candidate_partial_single_unit_other",
            "candidate_partial_multi_unit",
        ):
            status_examples = candidate_partial_examples.get(status) or []
            if not status_examples:
                continue
            print("")
            print(f"{status} ({candidate_partial_counts.get(status, 0)} questions)")
            for example in status_examples:
                top = (example.get("answer_bearing_gt") or {}).get("top_candidate") or {}
                print(
                    f"  sample/question {example.get('sample_idx')}:{example.get('question_global_idx')} "
                    f"ref_token={top.get('reference_recall_pct', 0.0):.2f}% "
                    f"ref_unit={top.get('reference_unit_recall_pct', 0.0):.2f}% "
                    f"entry={top.get('entry_id')}"
                )
                print(f"    q: {example.get('question')}")
                print(f"    ref: {example.get('reference_answer')}")
                print(f"    top answer memory: {top.get('text')}")

    question_anchor_examples = summary.get("examples_by_question_anchor_status") or {}
    question_anchor_counts = summary.get("by_question_anchor_status") or {}
    if question_anchor_examples:
        print("")
        print("question-anchor bucket examples")
        for status in (
            "strong_question_match_in_evidence_window",
            "weak_question_match_in_evidence_window",
            "question_candidate_in_evidence_window",
            "question_candidate_outside_evidence_window",
            "no_question_candidate",
        ):
            status_examples = question_anchor_examples.get(status) or []
            if not status_examples:
                continue
            print("")
            print(f"{status} ({question_anchor_counts.get(status, 0)} questions)")
            for example in status_examples:
                top = (example.get("question_anchor_gt") or {}).get("top_candidate") or {}
                print(
                    f"  sample/question {example.get('sample_idx')}:{example.get('question_global_idx')} "
                    f"existence={example.get('gt_existence_bucket')} "
                    f"question_idf_f1={top.get('question_idf_f1_pct', 0.0):.2f}% "
                    f"entry={top.get('entry_id')}"
                )
                print(f"    q: {example.get('question')}")
                print(f"    ref: {example.get('reference_answer')}")
                print(f"    question-anchor memory: {top.get('text')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether LoCoMo evidence turns have matching frozen memory units, "
            "with DB-support buckets and canonical single-memory GT labels for manual inspection."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-root", type=Path, default=ROOT)
    parser.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--samples", nargs="+", help="Sample ids, comma lists, or all")
    parser.add_argument("--category", type=int, action="append", help="Filter by category; repeatable")
    parser.add_argument("--question-key", help="Inspect one question, e.g. 3:42")
    parser.add_argument("--start", type=int, default=0, help="Skip this many filtered questions before processing")
    parser.add_argument("--limit", type=int, help="Process at most this many filtered questions")
    parser.add_argument("--print-limit", type=int, default=50, help="Print at most this many records")
    parser.add_argument("--print-all", action="store_true", help="Print every processed record")
    parser.add_argument(
        "--print-mode",
        choices=["balanced", "first", "inspect"],
        default="inspect",
        help="Which processed records to print after saving.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show DB-loading progress logs.")
    parser.add_argument("--include-cat5", action="store_true", help="Include category 5 questions.")
    parser.add_argument("--min-reference-f1", type=float, default=0.10)
    parser.add_argument(
        "--strong-evidence-idf-f1",
        type=float,
        default=0.24,
        help="Calibration threshold for strong evidence-turn coverage.",
    )
    parser.add_argument(
        "--likely-evidence-idf-f1",
        type=float,
        default=0.12,
        help="Calibration threshold for likely evidence-turn coverage.",
    )
    parser.add_argument(
        "--evidence-top-n",
        type=int,
        default=3,
        help="Store this many memory candidates per evidence turn.",
    )
    parser.add_argument(
        "--bucket-example-limit",
        type=int,
        default=3,
        help="Store and print this many examples per evidence coverage bucket.",
    )
    parser.add_argument(
        "--cat5-reference",
        choices=["adversarial", "not-mentioned", "answer"],
        default="adversarial",
        help="Reference text used for category 5 oracle selection.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    show_progress = args.verbose

    if dateparser is None:
        raise SystemExit("dateparser is required to parse LoCoMo session dates")

    progress(f"loading dataset: {args.dataset}", show_progress)
    dataset = load_dataset(args.dataset)
    progress(f"loaded {len(dataset)} samples", show_progress)
    selected_samples = parse_samples(args.samples, len(dataset))
    if selected_samples is None:
        selected_samples = [
            idx
            for idx in range(len(dataset))
            if (args.db_root / f"{args.db_prefix}{idx}").exists()
        ]
    progress(f"selected samples: {selected_samples}", show_progress)
    category_filter = set(args.category or [])
    exclude_cat5 = not args.include_cat5 and 5 not in category_filter

    memory_by_sample: dict[int, list[dict[str, Any]]] = {}
    idf_by_sample: dict[int, dict[str, float]] = {}
    db_loaders: dict[int, str] = {}
    records: list[dict[str, Any]] = []

    filtered_seen = 0
    processed = 0

    for sample_position, sample_idx in enumerate(selected_samples, 1):
        sample = dataset[sample_idx]
        db_path = args.db_root / f"{args.db_prefix}{sample_idx}"
        if not db_path.exists():
            print(f"skipping sample {sample_idx}: db path not found: {db_path}", file=sys.stderr)
            continue

        if sample_idx not in memory_by_sample:
            memory_records, loader = load_memory_records(
                db_path,
                args.table_name,
                show_progress=show_progress,
            )
            progress(f"sample {sample_idx}: preparing {len(memory_records)} memory rows", show_progress)
            prepared_records, token_idf = prepare_memory_records(memory_records)
            memory_by_sample[sample_idx] = prepared_records
            idf_by_sample[sample_idx] = token_idf
            db_loaders[sample_idx] = loader
            progress(
                f"sample {sample_idx}: ready with {len(memory_by_sample[sample_idx])} prepared rows",
                show_progress,
            )

        sample_questions = sample.get("qa") or []
        progress(f"sample {sample_idx}: scanning {len(sample_questions)} questions", show_progress)
        sample_processed = 0
        for question_idx, qa in enumerate(sample_questions):
            category = qa.get("category")
            if exclude_cat5 and category == 5:
                continue
            if category_filter and category not in category_filter:
                continue
            if not question_key_matches(args.question_key, sample_idx, question_idx):
                continue

            if filtered_seen < args.start:
                filtered_seen += 1
                continue
            filtered_seen += 1

            if args.limit is not None and processed >= args.limit:
                break

            record = analyze_question(
                sample_idx=sample_idx,
                question_idx=question_idx,
                qa=qa,
                sample=sample,
                memory_records=memory_by_sample[sample_idx],
                token_idf=idf_by_sample[sample_idx],
                cat5_reference=args.cat5_reference,
                min_reference_f1=args.min_reference_f1,
                strong_evidence_f1=args.strong_evidence_idf_f1,
                likely_evidence_f1=args.likely_evidence_idf_f1,
                evidence_top_n=max(args.evidence_top_n, 1),
            )
            records.append(record)
            processed += 1
            sample_processed += 1

        if args.limit is not None and processed >= args.limit:
            print(
                f"sample {sample_idx} processed "
                f"({sample_position}/{len(selected_samples)}): {sample_processed} questions",
                flush=True,
            )
            break

        print(
            f"sample {sample_idx} processed "
            f"({sample_position}/{len(selected_samples)}): {sample_processed} questions",
            flush=True,
        )

    progress(f"finished scoring {len(records)} questions", show_progress)
    evidence_coverage_summary = build_evidence_coverage_summary(
        records,
        example_limit=max(args.bucket_example_limit, 0),
    )
    upper_bound_review_queue = build_upper_bound_review_queue(records)

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": Path(__file__).name,
            "dataset": str(args.dataset),
            "db_root": str(args.db_root),
            "db_prefix": args.db_prefix,
            "table_name": args.table_name,
            "samples": selected_samples,
            "samples_loaded": sorted(memory_by_sample.keys()),
            "db_loaders": db_loaders,
            "n_questions": len(records),
            "tokenizer": (
                "plain lowercase regex tokens [a-z0-9]+; no stopword removal, "
                "month expansion, or predefined token list"
            ),
            "reference_units": "comma-split reference answer strings, normalized with the same tokenizer",
            "category_filter": sorted(category_filter) if category_filter else None,
            "include_cat5": args.include_cat5,
            "excluded_categories": [5] if exclude_cat5 else [],
            "print_mode": args.print_mode,
            "selection_method": (
                "primary GT support uses raw evidence-turn text anchored to its evidence "
                "date/window: prefer candidates whose session interval contains the evidence "
                "date, then sort by IDF-weighted evidence overlap. The legacy "
                "reference-overlap candidate is still saved for comparison. "
                "A separate question-anchor view scores question-token overlap inside the "
                "evidence window for non-literal or temporal cases. "
                "Canonical single-memory GT uses the answer-bearing view only when the "
                "question is confident_direct; candidate_partial is retained as a "
                "relaxed upper-bound signal."
            ),
            "gt_views": {
                "evidence_source_gt_set": (
                    "deduplicated top evidence-anchored memory candidates for non-missing "
                    "evidence turns"
                ),
                "gt_memory_set": (
                    "alias for evidence_source_gt_set; this is the primary candidate GT set "
                    "for exists/review questions"
                ),
                "answer_bearing_gt_set": (
                    "top memory candidate in the evidence date/window with reference-answer "
                    "unit or IDF-contentful token overlap"
                ),
                "question_anchor_gt": (
                    "top memory candidate in the evidence date/window ranked by "
                    "question-token overlap; useful for event-grounded non-literal cases"
                ),
                "single_gt_memory": (
                    "canonical one-memory GT for later eval; populated when "
                    "single_gt_status is confident_direct or confident_session_anchor. "
                    "For session_anchor cases the question_anchor top candidate is used "
                    "when available (better topical relevance for temporal questions)."
                ),
                "single_gt_set": (
                    "single-element list alias for single_gt_memory, empty when no "
                    "strict canonical one-memory GT exists"
                ),
                "single_gt_relaxed_memory": (
                    "relaxed one-memory candidate for upper-bound analysis; populated "
                    "when single_gt_status is confident_direct, confident_session_anchor, "
                    "or candidate_partial"
                ),
                "single_gt_relaxed_set": (
                    "single-element list alias for single_gt_relaxed_memory, empty when "
                    "no relaxed one-memory candidate exists"
                ),
                "single_gt_status": (
                    "question-level one-memory GT label: confident_direct (all answer "
                    "phrases literally in memory text), confident_session_anchor (memory "
                    "is from the evidence session and topically related — answer comes "
                    "from session_date metadata, common for temporal questions), "
                    "candidate_partial (relaxed token-level match), "
                    "not_confident_single_gt, or no_support_in_db"
                ),
                "candidate_partial_bucket": (
                    "inspection-only split of candidate_partial into temporal single-unit, "
                    "high-recall single-unit, other single-unit, and multi-unit cases"
                ),
                "upper_bound_review_queue": (
                    "dedicated queue of candidate_partial, not_confident_single_gt, and "
                    "no_support_in_db questions for manual/agent adjudication of the "
                    "one-memory upper bound"
                ),
                "gt_existence_bucket": (
                    "practical existence label: exists, review, or missing. This does "
                    "not require a strict one-memory GT."
                ),
                "gt_existence_reason": (
                    "short reason for gt_existence_bucket, such as evidence support, "
                    "answer overlap, question-anchor rescue, or no support"
                ),
                "candidate_gt_memory_set": (
                    "legacy alias for evidence_source_gt_set while this verifier is calibrated"
                ),
            },
            "cat5_reference": args.cat5_reference,
            "min_reference_f1": args.min_reference_f1,
            "evidence_coverage_calibration": {
                "strong_evidence_idf_f1": args.strong_evidence_idf_f1,
                "likely_evidence_idf_f1": args.likely_evidence_idf_f1,
                "evidence_top_n": max(args.evidence_top_n, 1),
                "bucket_example_limit": max(args.bucket_example_limit, 0),
                "note": (
                    "gt_support_binary keeps the raw evidence-support collapse. "
                    "gt_existence_bucket is the practical top-level label for whether "
                    "support likely exists in the DB, with question-anchor rescue for "
                    "event-grounded non-literal cases. Raw conversation evidence is the "
                    "anchor; memory rows remain the candidate GT units. single_gt_status is stricter: "
                    "it asks whether one memory unit is sufficient enough to serve as "
                    "a canonical answer-bearing GT for eval. candidate_partial is kept "
                    "as a relaxed upper-bound signal, not the strict canonical GT."
                ),
            },
        },
        "summary": {
            "n_questions": len(records),
            "gt_exists_confident": sum(1 for r in records if r.get("gt_exists_confident")),
            "reference_answer_all_units_in_selected_gt": sum(
                1 for r in records if r.get("reference_answer_all_units_in_selected_gt")
            ),
            "reference_answer_any_unit_in_selected_gt": sum(
                1 for r in records if r.get("reference_answer_any_unit_in_selected_gt")
            ),
            "reference_answer_any_token_in_selected_gt": sum(
                1 for r in records if r.get("reference_answer_any_token_in_selected_gt")
            ),
            "single_gt_strict_upper_bound": sum(
                1 for r in records if r.get("single_gt_status") in {"confident_direct", "confident_session_anchor"}
            ),
            "single_gt_relaxed_upper_bound": sum(
                1
                for r in records
                if r.get("single_gt_status") in {"confident_direct", "confident_session_anchor", "candidate_partial"}
            ),
            "gt_support_status": evidence_coverage_summary["by_question_status"],
            "gt_support_binary": evidence_coverage_summary["by_binary_status"],
            "gt_existence_bucket": evidence_coverage_summary["by_existence_bucket"],
            "gt_existence_reason": evidence_coverage_summary["by_existence_reason"],
            "question_anchor_status": evidence_coverage_summary["by_question_anchor_status"],
            "single_gt_status": evidence_coverage_summary["by_single_gt_status"],
            "candidate_partial_bucket": evidence_coverage_summary["by_candidate_partial_bucket"],
            "upper_bound_review_queue": {
                "n_questions": upper_bound_review_queue["n_questions"],
                "by_bucket": upper_bound_review_queue["by_bucket"],
            },
            "evidence_coverage_status": evidence_coverage_summary["by_evidence_status"],
            "gt_answer_status": evidence_coverage_summary["by_answer_status"],
            "by_sample": {
                str(sample_idx): sum(1 for r in records if r["sample_idx"] == sample_idx)
                for sample_idx in sorted({r["sample_idx"] for r in records})
            },
            "by_category": {
                str(category): sum(1 for r in records if r.get("category") == category)
                for category in sorted({r.get("category") for r in records if r.get("category") is not None})
            },
        },
        "evidence_coverage_summary": evidence_coverage_summary,
        "upper_bound_review_queue": upper_bound_review_queue,
        "questions": records,
    }

    if not args.no_save:
        save_json(args.out, output)
        print(f"saved json: {args.out}")

    print_records = select_print_records(
        records,
        print_limit=args.print_limit,
        print_all=args.print_all,
        print_mode=args.print_mode,
    )
    for idx, record in enumerate(print_records, 1):
        print_record(record, idx, len(records))
    n_to_print = len(print_records)
    print_evidence_bucket_examples(evidence_coverage_summary)

    if not args.print_all and len(records) > n_to_print:
        print("")
        print(f"printed {n_to_print} of {len(records)} records")
        print("use --print-all or adjust --print-limit to see more")


if __name__ == "__main__":
    main()
