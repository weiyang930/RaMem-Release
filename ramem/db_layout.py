from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = REPO_ROOT / "db"
GPT_DB_DIR = DB_ROOT / "gpt"
QWEN_DB_DIR = DB_ROOT / "qwen"
LLAMA31_8B_DB_DIR = DB_ROOT / "llama31_8b"
LLAMA32_3B_DB_DIR = DB_ROOT / "llama32_3b"

GPT_ACTIVE_DB = GPT_DB_DIR / "lancedb_data"
QWEN_ACTIVE_DB = QWEN_DB_DIR / "lancedb_data"
LLAMA31_8B_ACTIVE_DB = LLAMA31_8B_DB_DIR / "lancedb_data"
LLAMA32_3B_ACTIVE_DB = LLAMA32_3B_DB_DIR / "lancedb_data"

GPT_FROZEN_PREFIX = "db/gpt/lancedb_data_frozen_sample"
QWEN_FROZEN_PREFIX = "db/qwen/lancedb_data_frozen_qwen_sample"
LLAMA31_8B_FROZEN_PREFIX = "db/llama31_8b/lancedb_data_frozen_llama31_8b_sample"
LLAMA32_3B_FROZEN_PREFIX = "db/llama32_3b/lancedb_data_frozen_llama32_3b_sample"


def gpt_frozen_db_path(sample_idx: int) -> Path:
    return REPO_ROOT / f"{GPT_FROZEN_PREFIX}{sample_idx}"


def qwen_frozen_db_path(sample_idx: int) -> Path:
    return REPO_ROOT / f"{QWEN_FROZEN_PREFIX}{sample_idx}"


def llama31_8b_frozen_db_path(sample_idx: int) -> Path:
    return REPO_ROOT / f"{LLAMA31_8B_FROZEN_PREFIX}{sample_idx}"


def llama32_3b_frozen_db_path(sample_idx: int) -> Path:
    return REPO_ROOT / f"{LLAMA32_3B_FROZEN_PREFIX}{sample_idx}"


def resolve_prefixed_db_path(prefix: str, sample_idx: int, root: Path | None = None) -> Path:
    raw_path = Path(f"{prefix}{sample_idx}")
    if raw_path.is_absolute():
        return raw_path
    return (root or REPO_ROOT) / raw_path
