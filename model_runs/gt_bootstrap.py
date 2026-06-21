from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ramem.db_layout import resolve_prefixed_db_path


ROOT = Path(__file__).resolve().parent
VERIFY_SCRIPT = ROOT / "model_runs" / "gt_verification.py"
EXPECTED_GT_QUESTION_COUNT = 1540
EXPECTED_SAMPLE_INDICES = range(10)


@dataclass(frozen=True)
class GTVerificationResult:
    path: Path
    reused_existing: bool
    generated: bool


def infer_gt_flow_label(db_prefix: str, gt_json_path: str | Path) -> str:
    raw_path = str(gt_json_path).lower()
    raw_prefix = str(db_prefix).lower()
    if "qwen" in raw_prefix or "qwen" in raw_path:
        return "Qwen"
    return "standard"


def ensure_gt_verification(
    gt_json_path: str | Path,
    db_prefix: str,
    flow_label: str,
) -> GTVerificationResult:
    resolved_path = Path(gt_json_path).expanduser().resolve()

    existing_count = _gt_question_count(resolved_path)
    if existing_count == EXPECTED_GT_QUESTION_COUNT:
        print(f"[gt-bootstrap] Reusing {flow_label} GT verification: {resolved_path}")
        return GTVerificationResult(path=resolved_path, reused_existing=True, generated=False)
    if resolved_path.exists():
        print(
            f"[gt-bootstrap] Existing {flow_label} GT verification is invalid "
            f"(expected {EXPECTED_GT_QUESTION_COUNT} questions, found {_format_count(existing_count)}); "
            f"regenerating: {resolved_path}"
        )

    if not VERIFY_SCRIPT.exists():
        raise SystemExit(f"GT verifier script not found: {VERIFY_SCRIPT}")

    missing_db_paths = _missing_db_paths(db_prefix)
    if missing_db_paths:
        missing_block = "\n  ".join(str(path) for path in missing_db_paths)
        raise SystemExit(
            f"Cannot generate {flow_label} GT verification because required DB paths are missing:\n"
            f"  {missing_block}"
        )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gt-bootstrap] {flow_label} GT verification missing; generating now: {resolved_path}")
    cmd = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--db-prefix",
        db_prefix,
        "--out",
        str(resolved_path),
        "--print-limit",
        "0",
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Failed to generate {flow_label} GT verification at {resolved_path}"
        ) from exc

    generated_count = _gt_question_count(resolved_path)
    if not resolved_path.exists():
        raise SystemExit(
            f"{flow_label} GT verification command completed but no file was written: {resolved_path}"
        )
    if generated_count != EXPECTED_GT_QUESTION_COUNT:
        raise SystemExit(
            f"{flow_label} GT verification was generated at {resolved_path} but is incomplete "
            f"(expected {EXPECTED_GT_QUESTION_COUNT} questions, found {_format_count(generated_count)})"
        )

    print(f"[gt-bootstrap] Generated {flow_label} GT verification: {resolved_path}")
    return GTVerificationResult(path=resolved_path, reused_existing=False, generated=True)


def _missing_db_paths(db_prefix: str) -> list[Path]:
    missing_paths: list[Path] = []
    for sample_idx in EXPECTED_SAMPLE_INDICES:
        db_path = resolve_prefixed_db_path(db_prefix, sample_idx, root=ROOT)
        if not db_path.exists():
            missing_paths.append(db_path)
    return missing_paths


def _gt_question_count(gt_json_path: Path) -> int | None:
    if not gt_json_path.exists():
        return None

    try:
        with gt_json_path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    questions = data.get("questions")
    if not isinstance(questions, list):
        return None
    return len(questions)


def _format_count(count: int | None) -> str:
    if count is None:
        return "unreadable"
    return str(count)
