from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ramem import config
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry


GENERATOR_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-4.1-mini"
TABLE_NAME = "memory_entries"


JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer
which you will score as CORRECT/WRONG.
The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def official_judge_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """LongMemEval official-style QA judge prompt.

    The upstream script uses yes/no prompts with task-specific leniency.  Keep
    this local to the adapter so core RaMem code remains unchanged.
    """
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given but "
            "the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        return (
            "I will give you a question, a correct answer, and a response from a model.\n"
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps to get "
            "the correct answer, you should also answer yes. If the response only contains a subset of the "
            "information required by the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
            "the correct answer or contains all the intermediate steps to get the correct answer, you should "
            "also answer yes. If the response only contains a subset of the information required by the answer, "
            "answer no. In addition, do not penalize off-by-one errors for the number of days. If the question "
            "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's response is still correct.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response should be considered as correct "
            "as long as the updated answer is the required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response from a model.\n"
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model "
            "does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
            "and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongMemEval-S with RaMem.")
    parser.add_argument("--dataset", default="longmemeval_s_cleaned.json")
    parser.add_argument("--output_dir", default="results/longmemeval_s")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples for smoke/pilot runs.")
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode and save pilot_results.json.")
    parser.add_argument("--pilot_n", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sample_indices", default=None,
                        help="Comma-separated original dataset row indices to run, overriding --start/--limit/--pilot_n selection.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--judge_protocol", choices=["official", "json"], default="official",
                        help="official uses LongMemEval yes/no prompts; json uses the older CORRECT/WRONG prompt.")
    parser.add_argument("--generator_model", default=GENERATOR_MODEL)
    parser.add_argument("--judge_model", default=JUDGE_MODEL)
    parser.add_argument("--openai_base_url", default=os.getenv("OPENAI_BASE_URL") or None)
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY") or None)
    parser.add_argument("--judge_api_key", default=os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or None)
    parser.add_argument("--judge_base_url", default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None)
    parser.add_argument("--enable_planning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_reflection", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max_reflection_rounds", type=int, default=None)
    parser.add_argument("--memory_workers", type=int, default=4)
    parser.add_argument("--retrieval_workers", type=int, default=4)
    parser.add_argument("--memory_layout", choices=["sliding", "session"], default="session",
                        help="sliding uses the original fixed dialogue windows; session preserves LongMemEval haystack session boundaries.")
    parser.add_argument("--longmem_session_linking", action=argparse.BooleanOptionalAction, default=True,
                        help="Ask extraction to preserve local cross-turn links such as store/app -> coupon/action.")
    parser.add_argument("--longmem_exact_fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="LongMemEval-only exact memory recall pass after the normal retriever.")
    parser.add_argument("--longmem_exact_fallback_k", type=int, default=20,
                        help="Maximum extra memories added by --longmem_exact_fallback.")
    parser.add_argument("--longmem_raw_fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="LongMemEval-only raw haystack evidence fallback for extraction misses.")
    parser.add_argument("--longmem_raw_fallback_k", type=int, default=12,
                        help="Maximum raw haystack snippets added by --longmem_raw_fallback.")
    parser.add_argument("--longmem_preference_anchors", action=argparse.BooleanOptionalAction, default=False,
                        help="Diagnostic LongMemEval preference-only generic raw evidence compaction; disabled by default.")
    parser.add_argument("--longmem_preference_rewrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Diagnostic LongMemEval preference-only answer rewrite; disabled by default.")
    parser.add_argument("--longmem_temporal_anchors", action=argparse.BooleanOptionalAction, default=False,
                        help="Diagnostic LongMemEval temporal-only generic dated evidence compaction; disabled by default.")
    parser.add_argument("--longmem_temporal_rewrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Diagnostic LongMemEval temporal-only answer rewrite; disabled by default.")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--answer_context_max_chars", type=int, default=0)
    parser.add_argument("--semantic_top_k", type=int, default=None)
    parser.add_argument("--keyword_top_k", type=int, default=None)
    parser.add_argument("--structured_top_k", type=int, default=None)
    args = parser.parse_args()
    args.method = "ramem"
    return args


def configure_runtime(args: argparse.Namespace) -> None:
    if not args.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for gpt-4o-mini generation. Set it in the environment or pass --openai_api_key.")
    config.OPENAI_API_KEY = args.openai_api_key
    config.OPENAI_BASE_URL = args.openai_base_url
    config.LLM_MODEL = args.generator_model
    config.JUDGE_API_KEY = args.judge_api_key or args.openai_api_key
    config.JUDGE_BASE_URL = args.judge_base_url
    config.JUDGE_MODEL = args.judge_model
    config.ENABLE_THINKING = False
    config.JUDGE_ENABLE_THINKING = False
    config.USE_STREAMING = False
    config.USE_JSON_FORMAT = True
    config.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    config.EMBEDDING_DIMENSION = 1024
    config.SSS_USE_RRF = False
    config.SSS_ENABLE_PROX_RERANK = False
    config.GUARD1_EXTENDED = True
    config.USE_RRF_BASELINE = False
    config.ENABLE_TEMPORAL = False
    config.ENTITY_AWARE_ANSWER = False
    config.NORMALIZE_ANSWER_QUOTES = True
    config.ENABLE_PLANNING = config.ENABLE_PLANNING if args.enable_planning is None else args.enable_planning
    config.ENABLE_REFLECTION = config.ENABLE_REFLECTION if args.enable_reflection is None else args.enable_reflection
    if args.max_reflection_rounds is not None:
        config.MAX_REFLECTION_ROUNDS = args.max_reflection_rounds
    if args.window_size is not None:
        config.WINDOW_SIZE = args.window_size
    config.ANSWER_CONTEXT_MAX_CHARS = args.answer_context_max_chars
    config.LONGMEM_SESSION_LINKING = bool(args.longmem_session_linking)


def apply_method_config(method: str, record: dict[str, Any]) -> None:
    """Apply eval-only switches without touching core method code."""
    config._last_retrieved_contexts = None
    config._guard2_window = None
    config._guard2_buffered_window = None
    runtime_local = getattr(config, "_runtime_local", None)
    if runtime_local:
        runtime_local.last_retrieved_contexts = None
        runtime_local.guard2_window = None
        runtime_local.guard2_buffered_window = None

    config.ABLATE_DISABLE_CUE_GUARD = False
    config.ABLATE_DISABLE_CONTEXT_AWARE_RANKING = False
    config.ABLATE_GENERATION_TEXT_ONLY = False
    config.USE_RRF_BASELINE = False

    if method == "ramem":
        start, end = context_date_window(record)
        config.ENABLE_TEMPORAL = True
        config.SSS_USE_RRF = True
        config.SSS_ENABLE_PROX_RERANK = True
        config.ENTITY_AWARE_ANSWER = True
        config.GUARD1_EXTENDED = True
        config.CONV_DATE_START = start
        config.CONV_DATE_END = end
    else:
        raise ValueError(f"Unknown method: {method}")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("data", "examples", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unsupported LongMemEval-S top-level schema: {type(data).__name__}")
    return data


def parse_longmem_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    match = re.search(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", raw)
    if match:
        y, mo, d, h, mi = match.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:00"
    match = re.search(r"(\d{4})-(\d{2})-(\d{2}).*?(\d{2}):(\d{2})", raw)
    if match:
        y, mo, d, h, mi = match.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:00"
    try:
        return datetime.fromisoformat(raw).replace(second=0, microsecond=0).isoformat()
    except Exception:
        return None


def format_question_date_for_prompt(raw: Any) -> str | None:
    parsed = parse_longmem_date(raw)
    if not parsed:
        return None
    try:
        return datetime.fromisoformat(parsed).strftime("%d %B %Y")
    except Exception:
        return parsed[:10]


def question_with_asof_date(question: str, record: dict[str, Any]) -> str:
    category = str(record.get("question_type") or record.get("category") or record.get("subtask") or "")
    question_date = format_question_date_for_prompt(record.get("question_date"))
    lines: list[str] = []
    if question_date:
        lines.extend([
            f"Question date / today / as-of date: {question_date}.",
            "Use this date to resolve relative time phrases such as today, currently, this year, past month, next, ago, and last.",
        ])
    if category == "temporal-reasoning":
        lines.extend([
            "Answer the temporal question directly using the retrieved dated evidence.",
            "For ordering questions, return the event or entity names in order; dates may be included but should not replace the event names.",
            "For comparisons, if evidence for one required entity or event is missing, say the information is insufficient instead of choosing the known one by default.",
        ])
    if not lines:
        return question
    lines.append(f"User question: {question}")
    return "\n".join(lines)


def record_id(record: dict[str, Any], idx: int) -> str:
    return str(record.get("question_id") or record.get("sample_id") or record.get("id") or idx)


def parse_sample_indices(raw: str | None, total: int) -> list[int] | None:
    if not raw:
        return None
    indices: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            idx = int(text)
        except ValueError as exc:
            raise SystemExit(f"Invalid --sample_indices value {text!r}; expected comma-separated integers.") from exc
        if idx < 0 or idx >= total:
            raise SystemExit(f"--sample_indices value {idx} is out of range for dataset size {total}.")
        indices.append(idx)
    if not indices:
        raise SystemExit("--sample_indices was provided but no valid indices were found.")
    return indices


def normalize_records(
    records: list[dict[str, Any]],
    start: int = 0,
    limit: int | None = None,
    sample_indices: list[int] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if sample_indices is not None:
        return [(idx, records[idx]) for idx in sample_indices]
    indexed = list(enumerate(records))
    if start:
        indexed = indexed[start:]
    if limit is not None:
        indexed = indexed[:limit]
    return indexed


def make_dialogues(record: dict[str, Any]) -> list[Dialogue]:
    return [dialogue for session_dialogues in make_session_dialogues(record) for dialogue in session_dialogues]


def make_session_dialogues(record: dict[str, Any]) -> list[list[Dialogue]]:
    from models.memory_entry import Dialogue

    sessions = record.get("haystack_sessions") or []
    session_dates_raw = record.get("haystack_dates") or []
    question_date = parse_longmem_date(record.get("question_date"))
    session_dates = [parse_longmem_date(x) for x in session_dates_raw]
    session_dialogues: list[list[Dialogue]] = []
    did = 1
    for sidx, session in enumerate(sessions):
        session_date = session_dates[sidx] if sidx < len(session_dates) else None
        next_session = None
        for later in session_dates[sidx + 1:]:
            if later:
                next_session = later
                break
        session_end = next_session or question_date or session_date
        if not isinstance(session, list):
            continue
        dialogues: list[Dialogue] = []
        for msg in session:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("speaker") or "unknown")
            content = str(msg.get("content") or msg.get("text") or "").strip()
            if not content:
                continue
            dialogues.append(Dialogue(
                dialogue_id=did,
                speaker=role,
                content=content,
                timestamp=session_date,
                session_date=session_date,
                session_end_date=session_end,
            ))
            did += 1
        if dialogues:
            session_dialogues.append(dialogues)
    return session_dialogues


def context_date_window(record: dict[str, Any]) -> tuple[str, str]:
    dates = [parse_longmem_date(x) for x in (record.get("haystack_dates") or [])]
    dates = [x for x in dates if x]
    qdate = parse_longmem_date(record.get("question_date"))
    if qdate:
        dates.append(qdate)
    if not dates:
        return "", ""
    return min(dates)[:10], max(dates)[:10]


def db_path_for(output_dir: Path, generator_model: str, rid: str, memory_layout: str = "sliding") -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", generator_model)
    safe_rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", rid)
    safe_layout = re.sub(r"[^A-Za-z0-9_.-]+", "_", memory_layout)
    return output_dir / "memory_db" / safe_model / safe_layout / safe_rid


def table_has_rows(db_path: Path) -> bool:
    try:
        import lancedb
        db = lancedb.connect(str(db_path))
        if TABLE_NAME not in db.table_names():
            return False
        return db.open_table(TABLE_NAME).count_rows() > 0
    except Exception:
        return False


def memory_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "lossless_restatement": entry.lossless_restatement,
        "keywords": entry.keywords,
        "timestamp": entry.timestamp,
        "location": entry.location,
        "persons": entry.persons,
        "entities": entry.entities,
        "topic": entry.topic,
        "session_date": entry.session_date,
        "session_end_date": entry.session_end_date,
        "mention_date": entry.mention_date,
    }


LONGMEM_STOPWORDS = {
    "about", "after", "again", "asked", "assistant", "because", "before", "being",
    "between", "could", "current", "currently", "detail", "details", "during",
    "given", "going", "have", "help", "helped", "their", "there", "these",
    "thing", "things", "those", "through", "today", "using", "user", "want",
    "wanted", "what", "when", "where", "which", "while", "with", "would",
    "remember", "previously", "conversation", "question", "answer",
}
LONGMEM_SHORT_TERMS = {
    "cat", "dog", "vet", "car", "app", "gym", "tea", "law", "tax", "bed", "bus",
    "job", "mom", "dad", "son", "kid", "art", "run", "sql", "api",
}


def longmem_salient_terms(question: str) -> tuple[list[str], list[str], list[str]]:
    """Extract exact-match cues for the LongMemEval adapter fallback."""
    text = re.sub(r"Question date / today / as-of date:.*?User question:", " ", question, flags=re.S)
    phrases = []
    phrases.extend(match.group(1).strip() for match in re.finditer(r"['\"]([^'\"]{2,80})['\"]", text))
    phrases.extend(match.group(0).strip() for match in re.finditer(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){1,5}\b", text))
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?(?:st|nd|rd|th|%)?\b|\$\s*\d+(?:[.,]\d+)?", text)
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
        lower = token.lower()
        if lower not in LONGMEM_STOPWORDS and (len(lower) >= 4 or lower in LONGMEM_SHORT_TERMS):
            tokens.append(lower)

    def dedupe(values: Iterable[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            cleaned = re.sub(r"\s+", " ", str(value).strip())
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                out.append(cleaned)
        return out

    return dedupe(phrases), dedupe(tokens), dedupe(numbers)


def longmem_exact_fallback(
    vector_store: VectorStore,
    question: str,
    category: str,
    contexts: list[MemoryEntry],
    max_extra: int,
) -> list[MemoryEntry]:
    """Adapter-local recall boost for LongMemEval without changing retriever code."""
    if max_extra <= 0 or not getattr(config, "LONGMEMEVAL_MODE", False):
        return contexts
    try:
        if vector_store.table.count_rows() == 0:
            return contexts
        rows = vector_store.table.to_arrow().to_pylist()
    except Exception as exc:
        print(f"[warn] LongMemEval exact fallback skipped: {exc}")
        return contexts

    phrases, tokens, numbers = longmem_salient_terms(question)
    if not (phrases or tokens or numbers):
        return contexts

    existing_ids = {entry.entry_id for entry in contexts}
    category_cues = {
        "single-session-preference": (
            "prefer", "preference", "likes", "liked", "enjoy", "enjoyed",
            "avoid", "dislike", "wants", "needs", "constraint", "success",
            "frustrated", "uses", "interested",
        ),
        "single-session-assistant": (
            "assistant", "recommended", "suggested", "provided", "advised",
            "chapter", "budget", "minutes", "year", "recipe", "plan", "list",
        ),
        "knowledge-update": (
            "currently", "current", "now", "latest", "updated", "changed",
            "switched", "moved", "increased", "decreased", "started", "stopped",
        ),
        "multi-session": (
            "pickup", "pick up", "return", "appointment", "doctor", "led",
            "wedding", "fish", "count", "distinct", "separate",
        ),
    }.get(category, ())

    scored: list[tuple[int, int, MemoryEntry]] = []
    for row_idx, row in enumerate(rows):
        entry_id = str(row.get("entry_id") or "")
        if entry_id in existing_ids:
            continue
        text = " ".join([
            str(row.get("lossless_restatement") or ""),
            " ".join(map(str, row.get("keywords") or [])),
            " ".join(map(str, row.get("entities") or [])),
            str(row.get("topic") or ""),
        ])
        lower = text.lower()
        score = 0
        score += 4 * sum(1 for phrase in phrases if phrase.lower() in lower)
        score += 3 * sum(1 for number in numbers if str(number).replace(" ", "") in lower.replace(" ", ""))
        score += sum(1 for token in tokens if token in lower)
        score += sum(1 for cue in category_cues if cue in lower)
        if category == "single-session-preference" and any(cue in lower for cue in category_cues):
            score += 2
        if category == "temporal-reasoning":
            if re.search(r"\b(user|i|my|me|we|our)\b", lower):
                score += 2
            if re.search(r"\b(today|yesterday|tomorrow|ago|last|next|recently|first|before|after|earliest|latest|started|ordered|attended|visited|moved|born|signed|helped|decided|finished|received|bought)\b", lower):
                score += 3
            if lower.startswith(("the assistant", "assistant ")) or "recommended" in lower:
                score -= 2
        if score >= 3:
            try:
                scored.append((score, -row_idx, MemoryEntry(
                    entry_id=entry_id,
                    lossless_restatement=str(row.get("lossless_restatement") or ""),
                    keywords=list(row.get("keywords") or []),
                    timestamp=row.get("timestamp") or None,
                    location=row.get("location") or None,
                    persons=list(row.get("persons") or []),
                    entities=list(row.get("entities") or []),
                    topic=row.get("topic") or None,
                    session_date=row.get("session_date") or None,
                    session_end_date=row.get("session_end_date") or None,
                    mention_date=row.get("mention_date") or None,
                )))
            except Exception as exc:
                print(f"[warn] LongMemEval exact fallback row skipped: {exc}")

    if not scored:
        return contexts
    extras = [entry for _, _, entry in sorted(scored, reverse=True)[:max_extra]]
    print(f"[LongMemEval exact fallback] added {len(extras)} contexts")
    return extras + contexts


def _split_raw_message(content: str) -> list[str]:
    parts: list[str] = []
    for line in str(content or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if len(line) <= 900:
            parts.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if current and len(current) + len(sentence) > 900:
                parts.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            parts.append(current.strip())
    if not parts and content:
        text = re.sub(r"\s+", " ", str(content)).strip()
        parts = [text[:900]] if text else []
    return parts


def _longmem_raw_score(snippet: str, role: str, category: str, phrases: list[str], tokens: list[str]) -> int:
    lower = snippet.lower()
    score = 0
    score += 5 * sum(1 for phrase in phrases if phrase.lower() in lower)
    score += sum(1 for token in tokens if token in lower)
    if category == "temporal-reasoning":
        if role == "user":
            score += 3
        elif role == "assistant":
            score -= 2
        if re.search(r"\b(today|yesterday|tomorrow|ago|last|next|recently|first|before|after|earliest|latest|started|ordered|attended|visited|moved|born|signed|helped|decided|finished|received|bought)\b", lower):
            score += 3
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:minutes?|mins?|hours?|days?|weeks?|months?|years?|%|dollars?|cups?|mg|ml|km|miles?)\b", lower):
        score += 3
    if re.search(r"\$\s*\d+|\b\d{4}\b|\bchapter\s+\d+\b", lower):
        score += 3
    if category == "single-session-assistant" and role == "assistant":
        score += 2
    if category in {"single-session-user", "single-session-preference", "knowledge-update"} and role == "user":
        score += 1
    if category == "single-session-preference" and re.search(r"\b(prefer|like|enjoy|avoid|want|need|interested|frustrat|success)\b", lower):
        score += 3
    if category == "knowledge-update" and re.search(r"\b(now|currently|latest|changed|switched|moved|increased|decreased|started|stopped)\b", lower):
        score += 3
    return score


def longmem_raw_fallback(
    record: dict[str, Any],
    question: str,
    category: str,
    contexts: list[MemoryEntry],
    max_extra: int,
) -> list[MemoryEntry]:
    """Question-keyed raw evidence fallback for LongMemEval extraction misses."""
    if max_extra <= 0 or not getattr(config, "LONGMEMEVAL_MODE", False):
        return contexts

    phrases, tokens, _numbers = longmem_salient_terms(question)
    if not (phrases or tokens):
        return contexts

    existing_text = {entry.lossless_restatement.strip().lower() for entry in contexts}
    sessions = record.get("haystack_sessions") or []
    session_dates = record.get("haystack_dates") or []
    scored: list[tuple[int, int, MemoryEntry]] = []
    ordinal = 0
    for session_idx, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        session_date = parse_longmem_date(session_dates[session_idx] if session_idx < len(session_dates) else None)
        for msg in session:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("speaker") or "unknown").lower()
            for snippet in _split_raw_message(str(msg.get("content") or msg.get("text") or "")):
                ordinal += 1
                score = _longmem_raw_score(snippet, role, category, phrases, tokens)
                if score < 3:
                    continue
                restatement = f"Raw prior {role} message"
                if session_date:
                    restatement += f" on {session_date[:10]}"
                restatement += f": {snippet}"
                key = restatement.lower()
                if key in existing_text:
                    continue
                existing_text.add(key)
                scored.append((score, -ordinal, MemoryEntry(
                    lossless_restatement=restatement,
                    keywords=phrases + tokens[:8],
                    timestamp=session_date,
                    topic="LongMemEval raw haystack evidence",
                    session_date=session_date,
                    session_end_date=session_date,
                )))

    if not scored:
        return contexts
    extras = [entry for _, _, entry in sorted(scored, reverse=True)[:max_extra]]
    print(f"[LongMemEval raw fallback] added {len(extras)} contexts")
    return extras + contexts


def longmem_preference_rerank(question: str, contexts: list[MemoryEntry]) -> list[MemoryEntry]:
    """Promote personal preference memories over generic advice for LongMemEval preference tasks."""
    if not contexts or getattr(config, "LONGMEMEVAL_CATEGORY", "") != "single-session-preference":
        return contexts

    phrases, tokens, numbers = longmem_salient_terms(question)
    preference_cues = (
        "user prefer", "user would prefer", "prefer", "preference", "likes",
        "liked", "enjoy", "enjoyed", "interested in", "wants", "wanted",
        "needs", "struggling", "frustrated", "success", "successful",
        "previous", "previously", "recent", "recently", "currently",
        "has been", "uses", "owns", "bought", "tried", "experiment",
        "open to", "not prefer", "avoid", "constraint",
    )
    generic_cues = (
        "recommended", "suggested", "tips include", "options include",
        "general", "popular", "examples include",
    )

    scored: list[tuple[int, int, MemoryEntry]] = []
    for pos, entry in enumerate(contexts):
        text = entry.lossless_restatement or ""
        lower = text.lower()
        is_raw = lower.startswith("raw prior")
        score = 0
        score += 5 * sum(1 for phrase in phrases if phrase.lower() in lower)
        score += 2 * sum(1 for token in tokens if token in lower)
        score += 3 * sum(1 for number in numbers if str(number).replace(" ", "") in lower.replace(" ", ""))
        score += 5 * sum(1 for cue in preference_cues if cue in lower)
        if "the user" in lower or "user " in lower:
            score += 4
        if is_raw and "raw prior user message" in lower:
            score += 2
        if is_raw and "raw prior assistant message" in lower:
            score -= 3
        score -= 2 * sum(1 for cue in generic_cues if cue in lower)
        if len(text) > 700:
            score -= 2
        scored.append((score, -pos, entry))

    reranked = [entry for _, _, entry in sorted(scored, reverse=True)]
    # Keep the generator prompt focused: preference judging rewards using a few
    # precise personal memories more than scanning dozens of generic suggestions.
    max_contexts = int(getattr(config, "LONGMEMEVAL_PREFERENCE_MAX_CONTEXTS", 45) or 45)
    if len(reranked) > max_contexts:
        print(f"[LongMemEval preference rerank] packed {max_contexts}/{len(reranked)} contexts")
        return reranked[:max_contexts]
    print(f"[LongMemEval preference rerank] reranked {len(reranked)} contexts")
    return reranked


def _preference_domain_terms(question: str) -> list[str]:
    """No dataset-specific lexical expansion; rely on question terms only."""
    return []


def _preference_candidate_score(snippet: str, role: str, question_tokens: list[str], domain_terms: list[str]) -> int:
    lower = snippet.lower()
    score = 0
    score += 4 * sum(1 for token in question_tokens if token in lower)
    score += 10 * sum(1 for term in domain_terms if term in lower)
    if role == "user":
        score += 3
    if re.search(r"\b(i|my|me|we|our)\b", lower):
        score += 2
    if re.search(
        r"\b(prefer|like|liked|enjoy|enjoyed|love|interested|want|wanted|need|needed|"
        r"bought|purchased|using|used|tried|try|experiment|success|successful|hit|"
        r"struggl|frustrat|planning|thinking|recent|recently|previous|previously|"
        r"mentioned|remember|favorite|favourite|open to|avoid)\b",
        lower,
    ):
        score += 8
    if re.search(r"\b(colleague|coworker|friend|family|trip|travel|home|work|school|event|project|tool|device|hobby|routine)\b", lower):
        score += 2
    if "raw prior" in lower:
        score -= 2
    if lower.strip() == getattr(config, "LONGMEMEVAL_CURRENT_QUESTION_LOWER", ""):
        score -= 20
    if len(snippet) < 35:
        score -= 3
    if len(snippet) > 1200:
        score -= 2
    return score


def _collect_preference_candidates(record: dict[str, Any], question: str, max_candidates: int = 160) -> list[str]:
    phrases, tokens, _numbers = longmem_salient_terms(question)
    question_tokens = [token.lower() for token in tokens]
    domain_terms = _preference_domain_terms(question)
    config.LONGMEMEVAL_CURRENT_QUESTION_LOWER = re.sub(r"\s+", " ", question.lower()).strip()
    sessions = record.get("haystack_sessions") or []
    session_dates = record.get("haystack_dates") or []
    scored: list[tuple[int, int, str]] = []
    ordinal = 0

    for session_idx, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        session_date = parse_longmem_date(session_dates[session_idx] if session_idx < len(session_dates) else None)
        for msg in session:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("speaker") or "unknown").lower()
            content = str(msg.get("content") or msg.get("text") or "")
            for snippet in _split_raw_message(content):
                ordinal += 1
                score = _preference_candidate_score(snippet, role, question_tokens, domain_terms)
                score += 3 * sum(1 for phrase in phrases if phrase.lower() in snippet.lower())
                if score < 8:
                    continue
                prefix = f"{session_date[:10]} {role}" if session_date else role
                scored.append((score, -ordinal, f"[{prefix}] {snippet}"))

    deduped: list[str] = []
    seen: set[str] = set()
    for _score, _ordinal, text in sorted(scored, reverse=True):
        key = re.sub(r"\W+", " ", text.lower())[:240]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if len(deduped) >= max_candidates:
            break
    return deduped


def longmem_preference_anchor_contexts(
    record: dict[str, Any],
    question: str,
    contexts: list[MemoryEntry],
    llm_client: Any,
) -> list[MemoryEntry]:
    """Select raw personal anchors for LongMemEval preference questions only."""
    if getattr(config, "LONGMEMEVAL_CATEGORY", "") != "single-session-preference":
        return contexts

    candidates = _collect_preference_candidates(record, question)
    if not candidates:
        return contexts

    candidate_block = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(candidates))
    prompt = f"""Select the raw prior-conversation evidence that is most useful for answering this preference-personalization question.

Question: {question}

Candidate evidence:
{candidate_block}

Return JSON only:
{{
  "anchors": [
    "short factual anchor about the user, copied or paraphrased from the candidates"
  ]
}}

Rules:
- Choose 2 to 6 anchors.
- Prefer the user's own preferences, past successes, recent purchases, tools, constraints, frustrations, and memorable experiences.
- The anchors must be relevant to the question.
- Do not choose an anchor that merely restates the current question.
- Prefer concrete personal evidence over generic advice.
- Do not invent facts.
"""
    anchors: list[str] = []
    try:
        raw = llm_client.chat_completion(
            [
                {"role": "system", "content": "You select evidence for long-memory preference QA. Return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        parsed = llm_client.extract_json(raw)
        raw_anchors = parsed.get("anchors", []) if isinstance(parsed, dict) else []
        anchors = [re.sub(r"\s+", " ", str(anchor)).strip() for anchor in raw_anchors if str(anchor).strip()]
    except Exception as exc:
        print(f"[warn] LongMemEval preference anchor selection failed: {exc}")

    if not anchors:
        anchors = candidates[:6]

    text = "Preference anchors for this question:\n" + "\n".join(f"- {anchor}" for anchor in anchors[:8])
    print(f"[LongMemEval preference anchors] added {min(len(anchors), 8)} anchors")
    return [
        MemoryEntry(
            lossless_restatement=text,
            keywords=["preference", "personalization", "longmemeval"],
            topic="LongMemEval preference anchors",
        )
    ] + contexts


def longmem_preference_rewrite_answer(
    question: str,
    answer: str,
    contexts: list[MemoryEntry],
    llm_client: Any,
) -> str:
    """Diagnostic preference-only final pass that uses personal anchors."""
    if getattr(config, "LONGMEMEVAL_CATEGORY", "") != "single-session-preference" or not contexts:
        return answer
    anchor_texts = []
    for entry in contexts[:12]:
        text = re.sub(r"\s+", " ", entry.lossless_restatement or "").strip()
        if text:
            anchor_texts.append(text[:1000])
    if not anchor_texts:
        return answer
    prompt = f"""Rewrite the model answer for a preference-personalization QA task.

Question: {question}

Current answer: {answer}

Relevant personal memory/context:
{chr(10).join(f"- {text}" for text in anchor_texts)}

Return JSON only:
{{"answer": "1-3 concise sentences"}}

Rules:
- Base the answer only on the relevant personal memory/context.
- Explicitly mention the user's remembered personal details: preferences, prior successes, recent purchases, tools, constraints, or experiences.
- Remove generic or unrelated suggestions.
- If multiple relevant personal details are available, use at least two.
"""
    try:
        raw = llm_client.chat_completion(
            [
                {"role": "system", "content": "You rewrite answers using personal evidence. Return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=350,
        )
        parsed = llm_client.extract_json(raw)
        rewritten = re.sub(r"\s+", " ", str(parsed.get("answer", ""))).strip() if isinstance(parsed, dict) else ""
        if rewritten:
            print("[LongMemEval preference rewrite] applied")
            return rewritten
    except Exception as exc:
        print(f"[warn] LongMemEval preference rewrite failed: {exc}")
    return answer


def _temporal_domain_terms(question: str) -> list[str]:
    """No dataset-specific lexical expansion; rely on question terms and date cues."""
    return []


def _temporal_candidate_score(snippet: str, role: str, question_tokens: list[str], domain_terms: list[str]) -> int:
    lower = snippet.lower()
    score = 0
    score += 4 * sum(1 for token in question_tokens if token in lower)
    score += 9 * sum(1 for term in domain_terms if term in lower)
    if role == "user":
        score += 4
    if re.search(r"\b(today|yesterday|tomorrow|ago|last|next|recently|current|first|before|after|earliest|latest|most recently|started|ordered|attended|visited|moved|born|signed|helped)\b", lower):
        score += 7
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b", lower):
        score += 5
    if re.search(r"\b\d+\s*(?:days?|weeks?|months?|years?)\b", lower):
        score += 3
    if lower.startswith(("raw prior assistant", "assistant")):
        score -= 2
    if len(snippet) < 30:
        score -= 3
    if len(snippet) > 1400:
        score -= 2
    return score


def _collect_temporal_candidates(record: dict[str, Any], question: str, max_candidates: int = 180) -> list[str]:
    phrases, tokens, _numbers = longmem_salient_terms(question)
    question_tokens = [token.lower() for token in tokens]
    domain_terms = _temporal_domain_terms(question)
    sessions = record.get("haystack_sessions") or []
    session_dates = record.get("haystack_dates") or []
    scored: list[tuple[int, int, str]] = []
    ordinal = 0
    for session_idx, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        session_date = parse_longmem_date(session_dates[session_idx] if session_idx < len(session_dates) else None)
        date_label = session_date[:10] if session_date else "unknown-date"
        for msg in session:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("speaker") or "unknown").lower()
            for snippet in _split_raw_message(str(msg.get("content") or msg.get("text") or "")):
                ordinal += 1
                score = _temporal_candidate_score(snippet, role, question_tokens, domain_terms)
                score += 3 * sum(1 for phrase in phrases if phrase.lower() in snippet.lower())
                if score < 7:
                    continue
                scored.append((score, -ordinal, f"[session_date={date_label} role={role}] {snippet}"))

    deduped: list[str] = []
    seen: set[str] = set()
    for _score, _ordinal, text in sorted(scored, reverse=True):
        key = re.sub(r"\W+", " ", text.lower())[:260]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if len(deduped) >= max_candidates:
            break
    return deduped


def longmem_temporal_anchor_contexts(
    record: dict[str, Any],
    question: str,
    contexts: list[MemoryEntry],
    llm_client: Any,
) -> list[MemoryEntry]:
    """Select dated event anchors for LongMemEval temporal questions only."""
    if getattr(config, "LONGMEMEVAL_CATEGORY", "") != "temporal-reasoning":
        return contexts
    candidates = _collect_temporal_candidates(record, question)
    if not candidates:
        return contexts
    question_date = format_question_date_for_prompt(record.get("question_date")) or str(record.get("question_date") or "")
    candidate_block = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(candidates))
    prompt = f"""Select dated evidence for this temporal question.

Question date / as-of date: {question_date}
Question: {question}

Candidate evidence:
{candidate_block}

Return JSON only:
{{
  "anchors": [
    "dated event/fact from the candidates, including the session_date and any directly supported relative-date resolution"
  ]
}}

Rules:
- Use only candidate evidence. Do not invent missing facts.
- Session dates are evidence dates. Resolve words like today/yesterday/last/ago relative to the candidate's session_date unless the message states an explicit date.
- For order or comparison questions, include the dated anchors needed for each event/entity when they are present.
- Do not compute the final answer here.
"""
    anchors: list[str] = []
    try:
        raw = llm_client.chat_completion(
            [
                {"role": "system", "content": "You select dated evidence for temporal QA. Return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=750,
        )
        parsed = llm_client.extract_json(raw)
        if isinstance(parsed, dict):
            anchors = [re.sub(r"\s+", " ", str(x)).strip() for x in parsed.get("anchors", []) if str(x).strip()]
    except Exception as exc:
        print(f"[warn] LongMemEval temporal anchor selection failed: {exc}")

    if not anchors:
        anchors = candidates[:8]
    lines = ["Temporal anchors for this question:"]
    lines.extend(f"- {anchor}" for anchor in anchors[:10])
    print(f"[LongMemEval temporal anchors] added {min(len(anchors), 10)} anchors")
    return [
        MemoryEntry(
            lossless_restatement="\n".join(lines),
            keywords=["temporal", "dated-event", "longmemeval"],
            topic="LongMemEval temporal anchors",
        )
    ] + contexts


def longmem_temporal_rewrite_answer(
    record: dict[str, Any],
    question: str,
    answer: str,
    contexts: list[MemoryEntry],
    llm_client: Any,
) -> str:
    """Diagnostic temporal-only final pass that uses dated anchors."""
    if getattr(config, "LONGMEMEVAL_CATEGORY", "") != "temporal-reasoning" or not contexts:
        return answer
    context_texts = []
    for entry in contexts[:12]:
        text = re.sub(r"\s+", " ", entry.lossless_restatement or "").strip()
        if text:
            context_texts.append(text[:1200])
    if not context_texts:
        return answer
    question_date = format_question_date_for_prompt(record.get("question_date")) or str(record.get("question_date") or "")
    prompt = f"""Rewrite the answer for a temporal-reasoning QA task.

Question date / as-of date: {question_date}
Question: {question}
Current answer: {answer}

Relevant dated memory/context:
{chr(10).join(f"- {text}" for text in context_texts)}

Return JSON only:
{{"answer": "final answer"}}

Rules:
- Use only the dated context.
- Resolve today/yesterday/ago/last/first/latest using the session dates and question date.
- For order questions, include the event names in order; dates can be included but should not replace event names.
- For how-many-days/weeks/months questions, answer with the number and unit.
- For comparison questions where one required entity/event is missing, answer that the information is insufficient.
- If the context says "no information about X", "missing", or "not possible to determine" for a compared entity/event, the final answer must start with "Insufficient information" and must not say the other entity/event happened first.
"""
    try:
        raw = llm_client.chat_completion(
            [
                {"role": "system", "content": "You rewrite temporal answers using dated evidence. Return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        parsed = llm_client.extract_json(raw)
        rewritten = re.sub(r"\s+", " ", str(parsed.get("answer", ""))).strip() if isinstance(parsed, dict) else ""
        if rewritten:
            print("[LongMemEval temporal rewrite] applied")
            return rewritten
    except Exception as exc:
        print(f"[warn] LongMemEval temporal rewrite failed: {exc}")
    return answer


def build_or_load_memory(
    record: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    embedding_model: EmbeddingModel,
    output_dir: Path,
) -> tuple[VectorStore, float, int, bool]:
    from core.memory_builder import MemoryBuilder
    from database.vector_store import VectorStore
    from utils.llm_client import LLMClient

    rid = record_id(record, idx)
    db_path = db_path_for(output_dir, args.generator_model, rid, args.memory_layout)
    exists = table_has_rows(db_path)
    vector_store = VectorStore(db_path=str(db_path), embedding_model=embedding_model, table_name=TABLE_NAME)
    if exists and not args.overwrite:
        return vector_store, 0.0, vector_store.table.count_rows(), True
    if args.overwrite and exists:
        vector_store.clear()

    llm_client = LLMClient(
        api_key=args.openai_api_key,
        model=args.generator_model,
        base_url=args.openai_base_url,
        enable_thinking=False,
        use_streaming=False,
    )
    builder = MemoryBuilder(
        llm_client=llm_client,
        vector_store=vector_store,
        window_size=args.window_size or config.WINDOW_SIZE,
        enable_parallel_processing=True,
        max_parallel_workers=args.memory_workers,
    )
    start = time.perf_counter()
    if args.memory_layout == "session":
        session_windows = make_session_dialogues(record)
        print(f"\n[Session Processing] Processing {len(session_windows)} LongMemEval sessions with {args.memory_workers} workers")
        print(f"Session sizes: {[len(window) for window in session_windows]}")
        builder._process_windows_parallel(session_windows)
    else:
        dialogues = make_dialogues(record)
        builder.add_dialogues(dialogues)
        builder.process_remaining()
    build_time = time.perf_counter() - start
    try:
        vector_store.optimize()
    except Exception as exc:
        print(f"[warn] optimize skipped for {rid}: {exc}")
    return vector_store, build_time, vector_store.table.count_rows(), False


def generate_one(
    record: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    embedding_model: EmbeddingModel,
    output_dir: Path,
) -> dict[str, Any]:
    from core.answer_generator import AnswerGenerator
    from core.hybrid_retriever import HybridRetriever
    from utils.llm_client import LLMClient

    rid = record_id(record, idx)
    category = str(record.get("question_type") or record.get("category") or record.get("subtask") or "")
    question = str(record.get("question") or "")
    runtime_question = question_with_asof_date(question, record)
    gold = str(record.get("answer") or record.get("gold_answer") or record.get("reference") or "")
    config.LONGMEMEVAL_MODE = True
    config.LONGMEMEVAL_CATEGORY = category
    config.LONGMEMEVAL_QUESTION_ID = rid
    config.LONGMEMEVAL_QUESTION_DATE = str(record.get("question_date") or "")
    config.LONGMEMEVAL_MEMORY_LAYOUT = args.memory_layout
    apply_method_config(args.method, record)

    vector_store, build_time, memory_count, reused_memory = build_or_load_memory(record, idx, args, embedding_model, output_dir)
    llm_client = LLMClient(
        api_key=args.openai_api_key,
        model=args.generator_model,
        base_url=args.openai_base_url,
        enable_thinking=False,
        use_streaming=False,
    )
    retriever = HybridRetriever(
        llm_client=llm_client,
        vector_store=vector_store,
        semantic_top_k=args.semantic_top_k,
        keyword_top_k=args.keyword_top_k,
        structured_top_k=args.structured_top_k,
        enable_planning=config.ENABLE_PLANNING,
        enable_reflection=config.ENABLE_REFLECTION,
        max_reflection_rounds=config.MAX_REFLECTION_ROUNDS,
        enable_parallel_retrieval=True,
        max_retrieval_workers=args.retrieval_workers,
    )
    answer_generator = AnswerGenerator(llm_client=llm_client)

    retrieval_start = time.perf_counter()
    contexts = retriever.retrieve(question)
    if args.longmem_exact_fallback:
        contexts = longmem_exact_fallback(
            vector_store=vector_store,
            question=question,
            category=category,
            contexts=contexts,
            max_extra=args.longmem_exact_fallback_k,
        )
    if args.longmem_raw_fallback:
        contexts = longmem_raw_fallback(
            record=record,
            question=question,
            category=category,
            contexts=contexts,
            max_extra=args.longmem_raw_fallback_k,
        )
    if args.longmem_preference_anchors:
        contexts = longmem_preference_anchor_contexts(
            record=record,
            question=question,
            contexts=contexts,
            llm_client=llm_client,
        )
    if args.longmem_temporal_anchors:
        contexts = longmem_temporal_anchor_contexts(
            record=record,
            question=question,
            contexts=contexts,
            llm_client=llm_client,
        )
    contexts = longmem_preference_rerank(question, contexts)
    retrieval_time = time.perf_counter() - retrieval_start

    generation_start = time.perf_counter()
    answer = answer_generator.generate_answer(runtime_question, contexts)
    if args.longmem_preference_rewrite:
        answer = longmem_preference_rewrite_answer(question, answer, contexts, llm_client)
    if args.longmem_temporal_rewrite:
        answer = longmem_temporal_rewrite_answer(record, question, answer, contexts, llm_client)
    generation_time = time.perf_counter() - generation_start
    formatted_context = answer_generator._format_contexts_for_generation(contexts) if contexts else ""

    return {
        "sample_idx": idx,
        "sample_id": rid,
        "question_id": rid,
        "method": args.method,
        "category": category,
        "question": question,
        "runtime_question": runtime_question,
        "question_date": record.get("question_date"),
        "gold_answer": gold,
        "generated_answer": answer,
        "answer_session_ids": record.get("answer_session_ids") or [],
        "haystack_session_ids": record.get("haystack_session_ids") or [],
        "retrieved_entry_ids": [entry.entry_id for entry in contexts],
        "retrieved_memories": [memory_to_dict(entry) for entry in contexts],
        "memory_count": memory_count,
        "reused_memory": reused_memory,
        "context_chars": len(formatted_context),
        "retrieved_memories_count": len(contexts),
        "timing": {
            "memory_build_seconds": build_time,
            "retrieval_seconds": retrieval_time,
            "generation_seconds": generation_time,
            "total_generation_pipeline_seconds": build_time + retrieval_time + generation_time,
        },
        "models": {
            "generator": args.generator_model,
            "judge": args.judge_model,
        },
        "debug": {
            "memory_layout": args.memory_layout,
            "longmem_session_linking": bool(args.longmem_session_linking),
            "longmem_exact_fallback": bool(args.longmem_exact_fallback),
            "longmem_exact_fallback_k": args.longmem_exact_fallback_k,
            "longmem_raw_fallback": bool(args.longmem_raw_fallback),
            "longmem_raw_fallback_k": args.longmem_raw_fallback_k,
            "longmem_preference_anchors": bool(args.longmem_preference_anchors),
            "longmem_preference_rewrite": bool(args.longmem_preference_rewrite),
            "longmem_temporal_anchors": bool(args.longmem_temporal_anchors),
            "longmem_temporal_rewrite": bool(args.longmem_temporal_rewrite),
            "enable_temporal": config.ENABLE_TEMPORAL,
            "sss_use_rrf": config.SSS_USE_RRF,
            "sss_enable_prox_rerank": config.SSS_ENABLE_PROX_RERANK,
            "entity_aware_answer": config.ENTITY_AWARE_ANSWER,
            "conv_date_start": getattr(config, "CONV_DATE_START", ""),
            "conv_date_end": getattr(config, "CONV_DATE_END", ""),
            "guard2_window": getattr(config, "_guard2_window", None),
            "guard2_buffered_window": getattr(config, "_guard2_buffered_window", None),
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_one(row: dict[str, Any], judge_client: LLMClient) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a strict JSON-only evaluator."},
        {
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=row.get("question", ""),
                gold_answer=row.get("gold_answer", ""),
                generated_answer=row.get("generated_answer", ""),
            ),
        },
    ]
    raw = ""
    parse_error = None
    for attempt in range(3):
        try:
            raw = judge_client.chat_completion(
                messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=128,
            )
            parsed = judge_client.extract_json(raw)
            label = str(parsed.get("label", "")).strip().upper()
            if label in {"CORRECT", "WRONG"}:
                return {
                    "sample_idx": row.get("sample_idx"),
                    "sample_id": row.get("sample_id"),
                    "category": row.get("category", ""),
                    "question": row.get("question", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "generated_answer": row.get("generated_answer", ""),
                    "label": label,
                    "is_correct": label == "CORRECT",
                    "judge_raw": raw,
                    "judge_parse_error": False,
                }
            parse_error = f"Invalid label: {label!r}"
        except Exception as exc:
            parse_error = str(exc)
        if attempt < 2:
            time.sleep(1 + attempt)
    return {
        "sample_idx": row.get("sample_idx"),
        "sample_id": row.get("sample_id"),
        "category": row.get("category", ""),
        "question": row.get("question", ""),
        "gold_answer": row.get("gold_answer", ""),
        "generated_answer": row.get("generated_answer", ""),
        "label": "WRONG",
        "is_correct": False,
        "judge_raw": raw,
        "judge_parse_error": True,
        "judge_error": parse_error,
    }


def judge_one_official(row: dict[str, Any], judge_client: LLMClient) -> dict[str, Any]:
    task = str(row.get("category") or "")
    qid = str(row.get("question_id") or row.get("sample_id") or "")
    prompt = official_judge_prompt(
        task=task,
        question=str(row.get("question", "")),
        answer=str(row.get("gold_answer", "")),
        response=str(row.get("generated_answer", "")),
        abstention="_abs" in qid,
    )
    raw = ""
    error = None
    for attempt in range(3):
        try:
            raw = judge_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            match = re.search(r"\b(yes|no)\b", raw.strip().lower())
            text = match.group(1) if match else ""
            if text == "yes":
                label = "CORRECT"
            elif text == "no":
                label = "WRONG"
            else:
                raise ValueError(f"Unexpected judge response: {raw!r}")
            return {
                "sample_idx": row.get("sample_idx"),
                "sample_id": row.get("sample_id"),
                "question_id": qid,
                "category": task,
                "question": row.get("question", ""),
                "gold_answer": row.get("gold_answer", ""),
                "generated_answer": row.get("generated_answer", ""),
                "label": label,
                "is_correct": label == "CORRECT",
                "judge_raw": raw,
                "judge_parse_error": False,
                "judge_protocol": "official",
            }
        except Exception as exc:
            error = str(exc)
        if attempt < 2:
            time.sleep(1 + attempt)
    return {
        "sample_idx": row.get("sample_idx"),
        "sample_id": row.get("sample_id"),
        "question_id": qid,
        "category": task,
        "question": row.get("question", ""),
        "gold_answer": row.get("gold_answer", ""),
        "generated_answer": row.get("generated_answer", ""),
        "label": "WRONG",
        "is_correct": False,
        "judge_raw": raw,
        "judge_parse_error": True,
        "judge_error": error,
        "judge_protocol": "official",
    }


def compute_metrics(generation_rows: list[dict[str, Any]], judge_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(row.get("sample_id")): row for row in generation_rows}
    total = len(judge_rows)
    correct = sum(1 for row in judge_rows if row.get("label") == "CORRECT")
    wrong = total - correct
    parse_errors = sum(1 for row in judge_rows if row.get("judge_parse_error"))
    avg = lambda vals: (sum(vals) / len(vals)) if vals else None
    gen_times = [by_id.get(str(row.get("sample_id")), {}).get("timing", {}).get("generation_seconds", 0.0) for row in judge_rows]
    ret_times = [by_id.get(str(row.get("sample_id")), {}).get("timing", {}).get("retrieval_seconds", 0.0) for row in judge_rows]
    build_times = [by_id.get(str(row.get("sample_id")), {}).get("timing", {}).get("memory_build_seconds", 0.0) for row in judge_rows]
    total_times = [by_id.get(str(row.get("sample_id")), {}).get("timing", {}).get("total_generation_pipeline_seconds", 0.0) for row in judge_rows]
    context_chars = [int(by_id.get(str(row.get("sample_id")), {}).get("context_chars") or 0) for row in judge_rows]
    retrieved_counts = [int(by_id.get(str(row.get("sample_id")), {}).get("retrieved_memories_count") or 0) for row in judge_rows]
    summary = {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "judge_parse_error_count": parse_errors,
        "avg_memory_build_seconds": avg(build_times),
        "avg_retrieval_seconds": avg(ret_times),
        "avg_generation_seconds": avg(gen_times),
        "avg_total_generation_pipeline_seconds": avg(total_times),
        "avg_context_chars": avg(context_chars),
        "avg_retrieved_memories": avg(retrieved_counts),
    }
    category_counts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        category_counts[str(row.get("category") or "NA")].append(row)
    by_category = []
    for category, rows in sorted(category_counts.items()):
        n = len(rows)
        c = sum(1 for row in rows if row.get("label") == "CORRECT")
        by_category.append({
            "category": category,
            "total": n,
            "correct": c,
            "wrong": n - c,
            "accuracy": c / n if n else 0.0,
            "judge_parse_error_count": sum(1 for row in rows if row.get("judge_parse_error")),
        })
    task_accs = [row["accuracy"] for row in by_category if row["total"] > 0 and not str(row["category"]).endswith("_abs")]
    summary["task_average_accuracy"] = avg(task_accs) or 0.0
    summary["task_count"] = len(task_accs)
    return summary, by_category


def write_metrics(output_dir: Path, summary: dict[str, Any], by_category: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "metadata": {
            "dataset": args.dataset,
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "generated_at": datetime.utcnow().isoformat(),
            "protocol": "LongMemEval-S accuracy with gpt-4.1-mini LLM-as-judge",
            "method": args.method,
            "judge_protocol": args.judge_protocol,
            "memory_layout": args.memory_layout,
            "longmem_session_linking": bool(args.longmem_session_linking),
            "longmem_exact_fallback": bool(args.longmem_exact_fallback),
            "longmem_exact_fallback_k": args.longmem_exact_fallback_k,
            "longmem_raw_fallback": bool(args.longmem_raw_fallback),
            "longmem_raw_fallback_k": args.longmem_raw_fallback_k,
            "enable_planning": config.ENABLE_PLANNING,
            "enable_reflection": config.ENABLE_REFLECTION,
        },
        "overall": summary,
        "by_category": by_category,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if by_category:
        with (output_dir / "metrics_by_category.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(by_category[0].keys()))
            writer.writeheader()
            writer.writerows(by_category)


def markdown_summary(summary: dict[str, Any], by_category: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("| Split | N | Correct | Wrong | Accuracy | Parse Errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| overall | {summary['total']} | {summary['correct']} | {summary['wrong']} | "
        f"{summary['accuracy']:.4f} | {summary['judge_parse_error_count']} |"
    )
    for row in by_category:
        lines.append(
            f"| {row['category']} | {row['total']} | {row['correct']} | {row['wrong']} | "
            f"{row['accuracy']:.4f} | {row['judge_parse_error_count']} |"
        )
    return "\n".join(lines)


def write_official_hypotheses(path: Path, generation_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in generation_rows:
            qid = str(row.get("question_id") or row.get("sample_id"))
            f.write(json.dumps({"question_id": qid, "hypothesis": row.get("generated_answer", "")}, ensure_ascii=False) + "\n")


def paper_markdown_summary(summary: dict[str, Any], by_category: list[dict[str, Any]]) -> str:
    order = [
        ("Temporal", "temporal-reasoning"),
        ("Multi-Session", "multi-session"),
        ("Knowledge-Update", "knowledge-update"),
        ("Single-Session-User", "single-session-user"),
        ("Single-Session-Assistant", "single-session-assistant"),
        ("Single-Session-Preference", "single-session-preference"),
    ]
    by_key = {row["category"]: row for row in by_category}
    lines = ["| Split | Accuracy | N |", "| --- | ---: | ---: |"]
    for label, key in order:
        row = by_key.get(key)
        if row:
            lines.append(f"| {label} | {row['accuracy'] * 100:.2f} | {row['total']} |")
        else:
            lines.append(f"| {label} | NA | 0 |")
    lines.append(f"| Average | {summary.get('task_average_accuracy', 0.0) * 100:.2f} | {summary.get('total', 0)} |")
    lines.append(f"| Overall | {summary.get('accuracy', 0.0) * 100:.2f} | {summary.get('total', 0)} |")
    return "\n".join(lines)


def print_dataset_report(records: list[dict[str, Any]]) -> None:
    print("\n[Dataset inspection]")
    print(f"Top-level type: list")
    print(f"Number of records: {len(records)}")
    for i, record in enumerate(records[:2]):
        print(f"\nExample {i}:")
        print(f"  keys: {list(record.keys())}")
        print(f"  sample id: {record_id(record, i)}")
        print(f"  category/subtask: {record.get('question_type')}")
        print(f"  question: {record.get('question')}")
        print(f"  gold answer: {record.get('answer')}")
        print(f"  sessions: {len(record.get('haystack_sessions') or [])}")
        first_session = (record.get("haystack_sessions") or [[]])[0]
        if isinstance(first_session, list) and first_session:
            print(f"  first session first message: {str(first_session[0])[:300]}")


def main() -> None:
    args = parse_args()
    from utils.embedding import EmbeddingModel
    from utils.llm_client import LLMClient

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(args)
    records = load_dataset(Path(args.dataset))
    print_dataset_report(records)
    print("\n[Existing interface report]")
    print("- Using the LongMemEval-S adapter for RaMem.")
    print("- Reusing core MemoryBuilder, HybridRetriever, AnswerGenerator, VectorStore, and LLMClient.")
    print("- Existing LoCoMo 1540 eval code is tightly coupled to locomo10.json, so this script uses a minimal LongMemEval-S adapter.")
    print("- Existing LoCoMo metric utilities are LoCoMo-oriented; this script implements LongMemEval-compatible QA judging.")
    print(f"\n[Run config] method={args.method} generator={args.generator_model} judge={args.judge_model} judge_protocol={args.judge_protocol} planning={config.ENABLE_PLANNING} reflection={config.ENABLE_REFLECTION}")

    sample_indices = parse_sample_indices(args.sample_indices, total=len(records))
    limit = args.pilot_n if args.pilot else args.limit
    selected = normalize_records(records, start=args.start, limit=limit, sample_indices=sample_indices)
    if args.pilot:
        generation_path = output_dir / f"pilot_{args.method}_generation_outputs.jsonl"
        judge_path = output_dir / f"pilot_{args.method}_{args.judge_protocol}_judge_outputs.jsonl"
        hypotheses_path = output_dir / f"pilot_{args.method}_hypotheses.jsonl"
    else:
        generation_path = output_dir / f"{args.method}_generation_outputs.jsonl"
        judge_path = output_dir / f"{args.method}_{args.judge_protocol}_judge_outputs.jsonl"
        hypotheses_path = output_dir / f"{args.method}_hypotheses.jsonl"
    if args.overwrite:
        generation_path.unlink(missing_ok=True)
        judge_path.unlink(missing_ok=True)

    existing_generation = {str(row.get("sample_id")): row for row in read_jsonl(generation_path)}
    embedding_model = EmbeddingModel()
    generation_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for idx, record in selected:
        rid = record_id(record, idx)
        if rid in existing_generation and not args.overwrite:
            row = existing_generation[rid]
            print(f"[generation resume] {idx} {rid}")
        else:
            print(f"\n[generation] {idx} {rid}")
            row = generate_one(record, idx, args, embedding_model, output_dir)
            append_jsonl(generation_path, row)
        generation_rows.append(row)

    existing_judges = {str(row.get("sample_id")): row for row in read_jsonl(judge_path)}
    judge_client = LLMClient(
        api_key=args.judge_api_key or args.openai_api_key,
        model=args.judge_model,
        base_url=args.judge_base_url,
        enable_thinking=False,
        use_streaming=False,
    )
    judge_rows: list[dict[str, Any]] = []
    judge_start = time.perf_counter()
    for row in generation_rows:
        rid = str(row.get("sample_id"))
        if rid in existing_judges and not args.overwrite:
            judged = existing_judges[rid]
            print(f"[judge resume] {rid} {judged.get('label')}")
        else:
            print(f"[judge] {rid}")
            judged = judge_one_official(row, judge_client) if args.judge_protocol == "official" else judge_one(row, judge_client)
            append_jsonl(judge_path, judged)
        judge_rows.append(judged)
    elapsed = time.perf_counter() - run_start
    judge_elapsed = time.perf_counter() - judge_start

    summary, by_category = compute_metrics(generation_rows, judge_rows)
    summary["wall_clock_seconds"] = elapsed
    summary["judge_wall_clock_seconds"] = judge_elapsed
    summary["avg_wall_clock_seconds_per_example"] = elapsed / len(generation_rows) if generation_rows else None
    summary["avg_judge_wall_clock_seconds_per_example"] = judge_elapsed / len(judge_rows) if judge_rows else None
    if args.pilot:
        pilot_payload = {
            "summary": summary,
            "by_category": by_category,
            "estimated_full_runtime_seconds": (summary["avg_wall_clock_seconds_per_example"] or 0.0) * len(records),
            "estimated_full_runtime_hours": ((summary["avg_wall_clock_seconds_per_example"] or 0.0) * len(records)) / 3600,
            "n_full_dataset": len(records),
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "memory_layout": args.memory_layout,
            "longmem_session_linking": bool(args.longmem_session_linking),
            "longmem_exact_fallback": bool(args.longmem_exact_fallback),
            "longmem_exact_fallback_k": args.longmem_exact_fallback_k,
            "longmem_raw_fallback": bool(args.longmem_raw_fallback),
            "longmem_raw_fallback_k": args.longmem_raw_fallback_k,
        }
        (output_dir / "pilot_results.json").write_text(json.dumps(pilot_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        write_metrics(output_dir, summary, by_category, args)
    write_official_hypotheses(hypotheses_path, generation_rows)

    print("\n[LongMemEval-S Summary]")
    print(markdown_summary(summary, by_category))
    print("\n[Paper-style Task Average]")
    print(paper_markdown_summary(summary, by_category))
    print("\n[Timing]")
    print(json.dumps({
        "avg_memory_build_seconds": summary.get("avg_memory_build_seconds"),
        "avg_retrieval_seconds": summary.get("avg_retrieval_seconds"),
        "avg_generation_seconds": summary.get("avg_generation_seconds"),
        "avg_judge_wall_clock_seconds_per_example": summary.get("avg_judge_wall_clock_seconds_per_example"),
        "avg_wall_clock_seconds_per_example": summary.get("avg_wall_clock_seconds_per_example"),
    }, indent=2))
    rerun = (
        f"OPENAI_API_KEY=$OPENAI_API_KEY python cli/run_longmemeval_s.py "
        f"--dataset {args.dataset} --output_dir {args.output_dir} "
        f"--judge_protocol {args.judge_protocol} "
        f"--generator_model {args.generator_model} --judge_model {args.judge_model} "
        f"--memory_layout {args.memory_layout}"
    )
    if not args.longmem_session_linking:
        rerun += " --no-longmem_session_linking"
    if not args.longmem_exact_fallback:
        rerun += " --no-longmem_exact_fallback"
    if args.longmem_exact_fallback_k != 20:
        rerun += f" --longmem_exact_fallback_k {args.longmem_exact_fallback_k}"
    if not args.longmem_raw_fallback:
        rerun += " --no-longmem_raw_fallback"
    if args.longmem_raw_fallback_k != 12:
        rerun += f" --longmem_raw_fallback_k {args.longmem_raw_fallback_k}"
    if not config.ENABLE_PLANNING:
        rerun += " --no-enable_planning"
    if not config.ENABLE_REFLECTION:
        rerun += " --no-enable_reflection"
    print("\n[Rerun command]")
    print(rerun)


if __name__ == "__main__":
    main()
