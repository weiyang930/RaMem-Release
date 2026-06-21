from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramem.db_layout import (
    GPT_ACTIVE_DB,
    GPT_DB_DIR,
    GPT_FROZEN_PREFIX,
    LLAMA31_8B_ACTIVE_DB,
    LLAMA31_8B_DB_DIR,
    LLAMA31_8B_FROZEN_PREFIX,
    LLAMA32_3B_ACTIVE_DB,
    LLAMA32_3B_DB_DIR,
    LLAMA32_3B_FROZEN_PREFIX,
    QWEN_ACTIVE_DB,
    QWEN_DB_DIR,
    QWEN_FROZEN_PREFIX,
)


ROOT = Path(__file__).resolve().parent.parent
SSS_DIR = ROOT / "SSS_results"
GT_CONTEXT_DIR = ROOT / "gt_context_data"
SHARD_DIR = SSS_DIR / "_internal_shards"

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_MODEL_LEN = 8192
EXPECTED_SAMPLES = list(range(10))
EXPECTED_QUESTION_COUNT = 1540
CORRECT_THRESHOLD = 0.5
DEFAULT_METHOD_CATEGORIES = [1, 2, 3, 4]
DEFAULT_GPT_OPENAI_MODEL = "gpt-4o-mini"
GPT41_MINI_OPENAI_MODEL = "gpt-4.1-mini"
GPT4O_OPENAI_MODEL = "gpt-4o"
GPT_OPENAI_MODEL_TO_SLUG = {
    DEFAULT_GPT_OPENAI_MODEL: "gpt",
    GPT41_MINI_OPENAI_MODEL: "gpt41_mini",
    GPT4O_OPENAI_MODEL: "gpt4o",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    flow_label: str
    backbone_label: str
    llm_model: str
    judge_model: str
    default_base_url: str | None
    db_dir: Path
    active_db: Path
    frozen_prefix: str
    gt_json: Path
    method_eval_script: str
    method_eval_stem: str
    method_context_out: Path
    embedding_model: str = EMBEDDING_MODEL_ID
    max_model_len: int = MAX_MODEL_LEN
    expected_samples: list[int] | None = None
    expected_question_count: int = EXPECTED_QUESTION_COUNT
    correct_threshold: float = CORRECT_THRESHOLD

    def __post_init__(self) -> None:
        if self.expected_samples is None:
            object.__setattr__(self, "expected_samples", EXPECTED_SAMPLES)

    def frozen_db_path(self, sample_idx: int) -> Path:
        return ROOT / f"{self.frozen_prefix}{sample_idx}"

    def shard_path(self, stem: str, sample_idx: int) -> Path:
        return SHARD_DIR / f"{stem}_s{sample_idx}.json"


def _normalize_gpt_openai_model(openai_model: str | None) -> str:
    model = (openai_model or DEFAULT_GPT_OPENAI_MODEL).strip()
    if model not in GPT_OPENAI_MODEL_TO_SLUG:
        options = ", ".join(sorted(GPT_OPENAI_MODEL_TO_SLUG))
        raise SystemExit(f"Unknown GPT --openai-model {model!r}. Expected one of: {options}")
    return model


def _gpt_slug(openai_model: str | None) -> str:
    return GPT_OPENAI_MODEL_TO_SLUG[_normalize_gpt_openai_model(openai_model)]


def make_gpt_spec(openai_model: str | None = None) -> RunSpec:
    model = _normalize_gpt_openai_model(openai_model)
    slug = _gpt_slug(model)

    if slug == "gpt":
        db_dir = GPT_DB_DIR
        active_db = GPT_ACTIVE_DB
        frozen_prefix = GPT_FROZEN_PREFIX
        gt_json = GT_CONTEXT_DIR / "gt_memory_verification_gpt.json"
        method_eval_stem = "gpt_eval"
        method_context_out = GT_CONTEXT_DIR / "1540_gpt_t_contexts.json"
    else:
        db_dir = ROOT / "db" / slug
        active_db = db_dir / "lancedb_data"
        frozen_prefix = f"db/{slug}/lancedb_data_frozen_{slug}_sample"
        gt_json = GT_CONTEXT_DIR / f"gt_memory_verification_{slug}.json"
        method_eval_stem = f"{slug}_eval_1540"
        method_context_out = GT_CONTEXT_DIR / f"1540_{slug}_t_contexts.json"

    return RunSpec(
        key="gpt",
        flow_label="GPT",
        backbone_label=model,
        llm_model=model,
        judge_model=model,
        default_base_url=None,
        db_dir=db_dir,
        active_db=active_db,
        frozen_prefix=frozen_prefix,
        gt_json=gt_json,
        method_eval_script="cli/ramem_eval.py",
        method_eval_stem=method_eval_stem,
        method_context_out=method_context_out,
    )


GPT_SPEC = make_gpt_spec(DEFAULT_GPT_OPENAI_MODEL)

QWEN_SPEC = RunSpec(
    key="qwen",
    flow_label="Qwen",
    backbone_label="Qwen2.5-7B-Instruct",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    judge_model="Qwen/Qwen2.5-7B-Instruct",
    default_base_url="http://localhost:8000/v1",
    db_dir=QWEN_DB_DIR,
    active_db=QWEN_ACTIVE_DB,
    frozen_prefix=QWEN_FROZEN_PREFIX,
    gt_json=GT_CONTEXT_DIR / "gt_memory_verification_qwen.json",
    method_eval_script="cli/ramem_eval.py",
    method_eval_stem="qwen_eval",
    method_context_out=GT_CONTEXT_DIR / "1540_qwen_t_contexts.json",
)


def _qwen_variant_slug() -> str:
    import os

    slug = os.getenv("QWEN_RUN_SLUG", "").strip()
    if not slug:
        return "qwen"
    normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in slug.lower())
    return normalized or "qwen"


def make_qwen_spec() -> RunSpec:
    """Create a Qwen spec, optionally namespaced by QWEN_RUN_SLUG.

    This keeps different local Qwen backbones from overwriting each other's
    frozen DBs, eval JSON, contexts, and GT verification files.
    """
    import os

    slug = _qwen_variant_slug()
    llm_model = os.getenv("QWEN_LLM_MODEL", QWEN_SPEC.llm_model).strip() or QWEN_SPEC.llm_model
    judge_model = os.getenv("QWEN_JUDGE_MODEL", llm_model).strip() or llm_model
    backbone_label = os.getenv("QWEN_BACKBONE_LABEL", llm_model.split("/")[-1]).strip() or llm_model

    if slug == "qwen":
        return RunSpec(
            key="qwen",
            flow_label="Qwen",
            backbone_label=backbone_label,
            llm_model=llm_model,
            judge_model=judge_model,
            default_base_url=QWEN_SPEC.default_base_url,
            db_dir=QWEN_SPEC.db_dir,
            active_db=QWEN_SPEC.active_db,
            frozen_prefix=QWEN_SPEC.frozen_prefix,
            gt_json=QWEN_SPEC.gt_json,
            method_eval_script=QWEN_SPEC.method_eval_script,
            method_eval_stem=QWEN_SPEC.method_eval_stem,
            method_context_out=QWEN_SPEC.method_context_out,
        )

    db_dir = ROOT / "db" / slug
    return RunSpec(
        key="qwen",
        flow_label=f"Qwen ({slug})",
        backbone_label=backbone_label,
        llm_model=llm_model,
        judge_model=judge_model,
        default_base_url=QWEN_SPEC.default_base_url,
        db_dir=db_dir,
        active_db=db_dir / "lancedb_data",
        frozen_prefix=f"db/{slug}/lancedb_data_frozen_{slug}_sample",
        gt_json=GT_CONTEXT_DIR / f"gt_memory_verification_{slug}.json",
        method_eval_script=QWEN_SPEC.method_eval_script,
        method_eval_stem=f"{slug}_eval_1540",
        method_context_out=GT_CONTEXT_DIR / f"1540_{slug}_t_contexts.json",
    )

LLAMA31_8B_SPEC = RunSpec(
    key="llama31_8b",
    flow_label="Llama-3.1-8B",
    backbone_label="Llama-3.1-8B-Instruct",
    llm_model="meta-llama/Llama-3.1-8B-Instruct",
    judge_model="meta-llama/Llama-3.1-8B-Instruct",
    default_base_url="http://localhost:8000/v1",
    db_dir=LLAMA31_8B_DB_DIR,
    active_db=LLAMA31_8B_ACTIVE_DB,
    frozen_prefix=LLAMA31_8B_FROZEN_PREFIX,
    gt_json=GT_CONTEXT_DIR / "gt_memory_verification_llama31_8b.json",
    method_eval_script="cli/ramem_eval.py",
    method_eval_stem="llama31_8b_eval",
    method_context_out=GT_CONTEXT_DIR / "1540_llama31_8b_t_contexts.json",
)

LLAMA32_3B_SPEC = RunSpec(
    key="llama32_3b",
    flow_label="Llama-3.2-3B",
    backbone_label="Llama-3.2-3B-Instruct",
    llm_model="meta-llama/Llama-3.2-3B-Instruct",
    judge_model="meta-llama/Llama-3.2-3B-Instruct",
    default_base_url="http://localhost:8000/v1",
    db_dir=LLAMA32_3B_DB_DIR,
    active_db=LLAMA32_3B_ACTIVE_DB,
    frozen_prefix=LLAMA32_3B_FROZEN_PREFIX,
    gt_json=GT_CONTEXT_DIR / "gt_memory_verification_llama32_3b.json",
    method_eval_script="cli/ramem_eval.py",
    method_eval_stem="llama32_3b_eval",
    method_context_out=GT_CONTEXT_DIR / "1540_llama32_3b_t_contexts.json",
)

SPECS_BY_KEY: dict[str, RunSpec] = {
    GPT_SPEC.key: GPT_SPEC,
    QWEN_SPEC.key: QWEN_SPEC,
    LLAMA31_8B_SPEC.key: LLAMA31_8B_SPEC,
    LLAMA32_3B_SPEC.key: LLAMA32_3B_SPEC,
}


def get_run_spec(model_key: str) -> RunSpec:
    normalized = model_key.strip().lower()
    if normalized == "gpt":
        return GPT_SPEC
    try:
        return SPECS_BY_KEY[normalized]
    except KeyError as exc:
        options = ", ".join(sorted(SPECS_BY_KEY))
        raise SystemExit(f"Unknown --model {model_key!r}. Expected one of: {options}") from exc


def get_run_spec_for_cli(model_key: str, openai_model: str | None = None) -> RunSpec:
    normalized = model_key.strip().lower()
    if normalized == "gpt":
        return make_gpt_spec(openai_model)
    if normalized == "qwen":
        return make_qwen_spec()
    return get_run_spec(normalized)
