"""RaMem runtime configuration.

Secrets are read from environment variables. Do not hard-code API keys in this
file.
"""

import os
import threading


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


_runtime_local = threading.local()

# ============================================================================
# LLM Configuration
# ============================================================================

# OpenAI-compatible API key. Local vLLM servers usually accept any placeholder.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "vllm-local")

# Custom OpenAI Base URL (optional)
# Set to None to use default OpenAI endpoint
# Examples:
#   - Qwen/Alibaba: "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   - Azure OpenAI: "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT"
#   - Local server: "http://localhost:8000/v1"
#   - OpenAI (default): None
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

# LLM Model name
# Examples: "gpt-4.1-mini", "gpt-4.1", "qwen3-max", "qwen-plus-2025-07-28"
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# LLM Seed for reproducibility (OpenAI supports this; set to None to disable)
LLM_SEED = 42

# Embedding model (local, no API needed)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024  # For Qwen3: up to 1024, supports 32-1024
EMBEDDING_CONTEXT_LENGTH = 32768  # Qwen3 supports 32k context


# ============================================================================
# Advanced LLM Features
# ============================================================================

# Enable deep thinking mode (for Qwen and compatible models)
# Adds extra_body={"enable_thinking": True} to API calls
# Set to False for OpenAI models (they don't support this)
ENABLE_THINKING = False

# Enable streaming responses (outputs content as it's generated)
USE_STREAMING = _env_bool("USE_STREAMING", True)

# Enable JSON format mode (ensures LLM outputs valid JSON)
# Adds response_format={"type": "json_object"} to API calls
# Helps prevent parsing failures from extra text like ```json
USE_JSON_FORMAT = _env_bool("USE_JSON_FORMAT", False)

# Per-call output cap for memory extraction. 0 lets the backend choose.
# Local Qwen models benefit from a finite cap because runaway JSON generation
# can block parallel memory building for minutes.
MEMORY_EXTRACTION_MAX_TOKENS = _env_int("MEMORY_EXTRACTION_MAX_TOKENS", 4096)


# ============================================================================
# Memory Building Parameters
# ============================================================================

# Number of dialogues per window (for locomo; for other dataset, please finetune it)
WINDOW_SIZE = _env_int("WINDOW_SIZE", 40)

# Window overlap size (for context continuity)
OVERLAP_SIZE = _env_int("OVERLAP_SIZE", 2)


# ============================================================================
# Retrieval Parameters (can be adjusted to balance between token usage and performance)
# ============================================================================

# Max entries returned by semantic search (vector similarity)
SEMANTIC_TOP_K = _env_int("SEMANTIC_TOP_K", 25)

# Max entries returned by keyword search (BM25 matching)
KEYWORD_TOP_K = _env_int("KEYWORD_TOP_K", 5)

# Max entries returned by structured search (metadata filtering)
STRUCTURED_TOP_K = _env_int("STRUCTURED_TOP_K", 25)

# SSS ranker: True = BM25/FTS (keyword_search_with_filter), False = semantic (original)
# BM25 improves Cat 5 precision (mean_rank 3.7 vs 7.4) but has higher miss rate on Cat 2.
# Set True to evaluate SSS-BM25-k25; leave False for original SSS-Sem behavior.
SSS_USE_BM25 = False

# SSS_USE_RRF: RRF fusion of sem-filtered + BM25-filtered lists within the session window.
# Combines semantic coverage with exact-entity BM25 recall.
# Overrides SSS_USE_BM25 when True.
SSS_USE_RRF = False

# SSS_ENABLE_PROX_RERANK: after RRF (or sem), proximity-sort entries so those whose
# session overlaps the Guard2 window float to the top (exploits primacy bias).
# Only active when SSS_USE_RRF=True (or SSS_USE_BM25=False).
SSS_ENABLE_PROX_RERANK = False

# Enable temporal session-overlap retrieval.
# When True and CONV_DATE_START/END are set, the conversation date window is automatically
# injected into the Guard 2 prompt so the LLM can resolve implicit time references
# (e.g. "last August", "this summer") without an explicit year.
ENABLE_TEMPORAL = True

# Conversation date window — set from sample metadata before calling ask().
# Leave empty ("") to disable window injection (Guard 2 prompt reverts to original).
CONV_DATE_START = ""   # e.g. "2023-01-15"
CONV_DATE_END   = ""   # e.g. "2024-03-10"

# Guard 1 mode:
#   True  = extended regex (year OR month name OR season OR week/weekend reference)
#           Catches implicit time-window questions like "plans for the summer", "in October".
#           Pair with CONV_DATE_START/END so Guard 2 can resolve month/season to a year.
#   False = year-only fallback (\b20\d\d\b), original Guard 1 behaviour
GUARD1_EXTENDED = True

# Session date buffer (days) applied symmetrically around the G2 window.
# Accounts for the lag between when an event occurs (what G2 extracts) and
# when that event is discussed/stored in the session DB (typically 1–5 days later).
# Set to 0 to disable.  5 is enough to capture all observed gaps in LoCoMo10.
TEMPORAL_WINDOW_BUFFER_DAYS = 5


# ============================================================================
# Database Configuration
# ============================================================================

# Path to the default standard/GPT LanceDB working directory
LANCEDB_PATH = "./db/gpt/lancedb_data"

# Memory table name
MEMORY_TABLE_NAME = "memory_entries"



# ============================================================================
# Parallel Processing Configuration
# ============================================================================

# Memory Building Parallel Processing
ENABLE_PARALLEL_PROCESSING = _env_bool("ENABLE_PARALLEL_PROCESSING", True)
MAX_PARALLEL_WORKERS = _env_int("MAX_PARALLEL_WORKERS", 16)  # Number of parallel workers for memory building

# Retrieval Parallel Processing  
ENABLE_PARALLEL_RETRIEVAL = _env_bool("ENABLE_PARALLEL_RETRIEVAL", True)
MAX_RETRIEVAL_WORKERS = _env_int("MAX_RETRIEVAL_WORKERS", 8)  # Number of parallel workers for retrieval queries

# Planning and Reflection Configuration
ENABLE_PLANNING = _env_bool("ENABLE_PLANNING", True)
ENABLE_REFLECTION = _env_bool("ENABLE_REFLECTION", True)
MAX_REFLECTION_ROUNDS = _env_int("MAX_REFLECTION_ROUNDS", 2)

# Eval question-level parallelism. This is separate from retrieval parallelism:
# it runs multiple full QA evaluations concurrently so a local vLLM server can
# dynamically batch planning/answer/judge requests.
EVAL_PARALLEL_QUESTIONS = _env_bool("EVAL_PARALLEL_QUESTIONS", False)
EVAL_MAX_WORKERS = _env_int("EVAL_MAX_WORKERS", 4)

# Answer rewriting: after initial answer generation, run a concision pass that strips
# verbose sentence wrappers for entity/name/place questions (e.g. "They visited Tokyo" → "Tokyo").
# Preserves complex answers that need a sentence to be meaningful.
ENABLE_ANSWER_REWRITE = False

# Entity-aware answer generation: adds an explicit instruction to _build_answer_prompt
# telling the LLM to return ONLY the entity for entity-type questions (city, game, person,
# food, instrument, etc.) and short yes/no for yes/no questions. Keeps full phrases for
# descriptive/event/feeling questions. No extra LLM call — baked into the generation prompt.
ENTITY_AWARE_ANSWER = False

# Quote normalization: strip typographic/curly quote characters (unicode Pi/Pf categories)
# from every LLM answer before returning it. Fixes cases where the LLM wraps titles in
# curly quotes ('Xenoblade Chronicles') that the eval tokenizer can't strip. Always safe.
NORMALIZE_ANSWER_QUOTES = True

# Generator prompt packing. 0 disables packing and sends every retrieved memory.
# Qwen/vLLM eval wrappers set this to a bounded value so long retrieved contexts
# do not exceed the model context window and so answer generation stays fast.
ANSWER_CONTEXT_MAX_CHARS = _env_int("ANSWER_CONTEXT_MAX_CHARS", 0)

# LongMemEval adapter state. These defaults are inert for normal RaMem
# use; run_longmemeval_s.py sets them per example.
LONGMEMEVAL_MODE = False
LONGMEMEVAL_CATEGORY = ""
LONGMEMEVAL_QUESTION_ID = ""
LONGMEMEVAL_QUESTION_DATE = ""
LONGMEMEVAL_MEMORY_LAYOUT = ""

# Ablation-only runtime switches. Defaults preserve the full method.
# These are intentionally environment-controlled so experiment runners can
# isolate variants without editing the main pipeline code.
ABLATE_DISABLE_CUE_GUARD = _env_bool("ABLATE_DISABLE_CUE_GUARD", False)
ABLATE_DISABLE_CONTEXT_AWARE_RANKING = _env_bool("ABLATE_DISABLE_CONTEXT_AWARE_RANKING", False)
ABLATE_GENERATION_TEXT_ONLY = _env_bool("ABLATE_GENERATION_TEXT_ONLY", False)


# ============================================================================
# LLM-as-Judge Configuration (not used yet)
# ============================================================================

# Judge LLM API key. Defaults to OPENAI_API_KEY.
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", OPENAI_API_KEY)

# Judge LLM Base URL (optional - if None, uses OPENAI_BASE_URL)
# Example: Use cheaper endpoint for evaluation
# JUDGE_BASE_URL = "https://api.openai.com/v1/"  # OpenAI (commented out for Qwen)
JUDGE_BASE_URL = "http://localhost:8000/v1"

# Judge LLM Model (optional - if None, uses LLM_MODEL)
# JUDGE_MODEL = "gpt-4o-mini"  # OpenAI (commented out for Qwen)
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Judge specific settings
JUDGE_ENABLE_THINKING = False  # Usually false for evaluation tasks
JUDGE_USE_STREAMING = False    # Usually false for evaluation
JUDGE_TEMPERATURE = 0.3        

# Example configurations:
# 1. Use cheaper model for judge evaluation:
#    JUDGE_MODEL = "gpt-4o-mini"
#
# 2. Use different API provider for judge:
#    JUDGE_API_KEY = "your-judge-api-key"
#    JUDGE_BASE_URL = "https://api.different-provider.com/v1"
#    JUDGE_MODEL = "different-provider-model"
#
# 3. Use Qwen for judge (if available):
#    JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#    JUDGE_MODEL = "qwen-plus-2025-09-11"
