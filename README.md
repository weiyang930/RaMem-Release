<div align="center">

# RaMem: Contextual Reinstatement for Long-term Agentic Memory

[![Framework](https://img.shields.io/badge/Framework-RaMem-success?style=for-the-badge)](#)
[![Memory](https://img.shields.io/badge/Task-Long--term%20Agentic%20Memory-orange?style=for-the-badge)](#)
[![Retrieval](https://img.shields.io/badge/Focus-Contextual%20Evidence-blueviolet?style=for-the-badge)](#)
[![Code](https://img.shields.io/badge/Code-Open%20Source-black?style=for-the-badge)](https://anonymous.4open.science/r/RaMem-7BE0/)

<p align="center">
  <b>RaMem</b>: A contextual reinstatement framework that turns retrieved long-term memories into contextually verifiable evidence for agentic reasoning.
</p>

<p align="center">
  <a href="https://anonymous.4open.science/r/RaMem-7BE0/"><b>Code</b></a> •
  <a href="#overview"><b>Overview</b></a> •
  <a href="#abstract"><b>Abstract</b></a> •
  <a href="#key-features"><b>Key Features</b></a> •
  <a href="#datasets"><b>Datasets</b></a> •
  <a href="#environment-setup"><b>Setup</b></a> •
  <a href="#running-the-code"><b>Run</b></a> •
  <a href="#citation"><b>Citation</b></a>
</p>

</div>

---

## Overview

Long-term memory is becoming a central component of LLM agents, enabling them to preserve useful information across extended interactions, evolving user states, and multi-session task contexts. Existing memory systems have made important progress in storing, compressing, indexing, and retrieving past experiences. However, retrieval alone does not guarantee that a memory is valid evidence for the current query.

We identify this failure mode as **context collapse**. When past experiences are compressed into reusable memory fragments, they may lose the surrounding episodic conditions that determine when they should be used. As a result, memories from different sessions, time periods, participants, or user states can appear equally relevant, even though only one of them actually supports the current query.

To address this problem, we introduce **RaMem**, a contextual reinstatement framework for long-term agentic memory. Rather than treating retrieved memories as context-free snippets, RaMem restores the conditions under which each memory was formed and checks whether those conditions match the current query. In this way, RaMem shifts long-term memory from simple relevance-based retrieval toward **contextual evidence verification**.

RaMem operates through four coordinated stages:

- **Episodic Memory Anchoring** grounds each memory in its original episodic conditions, such as event time, mention time, session span, participants, locations, entities, and topics.
- **Recall Condition Induction** decomposes the query into an information need and a contextual recall frame, specifying what valid evidence should satisfy.
- **Validity-Aware Memory Retrieval** promotes memories whose episodic context matches the query while demoting related but context-conflicting distractors.
- **Context-Preserved Evidence Synthesis** passes selected memories to the generator with their structured context intact, enabling answer generation from verifiable evidence.

Experiments on long-term memory benchmarks show that RaMem consistently improves answer quality, ground-truth memory retrieval, and context efficiency over strong memory baselines across multiple LLM backbones.

---

## Abstract

> Long-term memory is increasingly important for LLM agents operating across extended interactions and evolving task contexts. Existing memory systems make past experiences persistent, compact, and retrievable, but they often assume that retrieved memories can be directly used as evidence. This assumption breaks down when memories from different situations appear similar because they share recurring entities, topics, or user states. We call this failure **context collapse**, where memories lose the surrounding context needed to determine whether they are valid for the current query.
>
> We propose **RaMem**, a contextual reinstatement framework that turns retrieved memory fragments into contextually verifiable evidence. RaMem anchors each memory to its original episodic conditions, induces recall conditions from the query, retrieves memories with validity-aware context matching, and preserves structured context during answer synthesis. Experiments on long-term memory benchmarks show that RaMem consistently improves performance over strong memory baselines, with substantial gains across multiple backbone models.

---

## Key Features

- **From memory retrieval to evidence verification**  
  RaMem reframes long-term agent memory as an evidence identification problem: a retrieved memory should not only be relevant to the query, but also valid under the current episodic context.

- **Context collapse as a first-class failure mode**  
  RaMem addresses a common weakness of memory systems, where compressed memories from different sessions, time periods, or user states become indistinguishable and lead agents to use plausible but invalid evidence.

- **Contextual reinstatement for agentic memory**  
  Instead of passing flattened memory snippets directly to the generator, RaMem restores the original conditions under which each memory was formed and uses them to determine whether the memory can support the current query.

- **Validity-aware retrieval under ambiguous long-term histories**  
  RaMem combines broad content-based recall with context compatibility, allowing the system to keep useful candidates while suppressing related but context-conflicting distractors.

- **Context-preserved generation from structured evidence**  
  Retrieved memories remain linked to their episodic coordinates during answer synthesis, helping the generator reason over not only what was remembered, but also when, where, and under which interaction context it was remembered.

- **Support for multiple backbones and evaluation settings**  
  This repository supports OpenAI-compatible APIs, local vLLM backends, GPT-family models, Qwen-family models, LLaMA-family models, LoCoMo evaluation, and LongMemEval-S evaluation.

---

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── cli/                    # Python command-line entrypoints
│   ├── build_memory.py      # Build RaMem memory databases
│   ├── ramem_eval.py        # Evaluate RaMem on LoCoMo
│   └── run_longmemeval_s.py # Evaluate RaMem on LongMemEval-S
├── ramem/                  # Runtime configuration and system wiring
├── core/                   # Memory building, retrieval, and answer generation
├── database/               # LanceDB vector-store wrapper
├── model_runs/             # Shared benchmark build/evaluation helpers
├── models/                 # Pydantic data models
├── scripts/                # Shell helpers for vLLM, Qwen, and smoke tests
├── test_ref/               # Reference metric utilities
└── utils/                  # LLM and embedding clients
```

Generated outputs are intentionally excluded from the repository. Local runs will create directories such as `db/`, `results/locomo/`, `results/locomo_contexts/`, and `results/longmemeval_s/`.

---

## Datasets

Due to licensing and redistribution restrictions, benchmark datasets are not bundled with this repository. Please obtain the official datasets and place them under the project root or pass explicit paths through the CLI arguments.

### LoCoMo

For LoCoMo experiments, prepare a local file named:

```text
locomo10.json
```

By default, the LoCoMo memory-building script expects this file at the repository root:

```bash
python cli/build_memory.py --model qwen --dataset locomo10.json
```

You may also pass an absolute path:

```bash
python cli/build_memory.py --model qwen --dataset /path/to/locomo10.json
```

### LongMemEval-S

For LongMemEval-S experiments, prepare a local file named:

```text
longmemeval_s_cleaned.json
```

The LongMemEval-S script defaults to this filename, or you can pass a custom path:

```bash
python cli/run_longmemeval_s.py --dataset /path/to/longmemeval_s_cleaned.json
```

---

## Environment Setup

We recommend using Python 3.10 or later in a clean virtual environment.

### Option 1: Conda

```bash
conda create -n ramem python=3.10 -y
conda activate ramem
pip install -r requirements.txt
```

### Option 2: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Environment Variables

Copy the example environment file if you want a local template:

```bash
cp .env.example .env
```

Then export the variables in your shell or load the file with your preferred environment manager before running experiments.

For OpenAI-hosted models:

```bash
export OPENAI_API_KEY="your-api-key"
unset OPENAI_BASE_URL
export LLM_MODEL="gpt-4.1-mini"
```

For a local OpenAI-compatible vLLM server:

```bash
export OPENAI_API_KEY="vllm-local"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="Qwen/Qwen3-8B"
```

RaMem uses `Qwen/Qwen3-Embedding-0.6B` as the default embedding model. You can either let Hugging Face download it through `sentence-transformers`, or point the code to a local model directory:

```bash
export EMBEDDING_MODEL_PATH=/path/to/qwen3-embedding-0.6b
```

For the provided Qwen shell helpers, the default local embedding path is:

```text
models/qwen3-embedding-0.6b
```

You can download the embedding model with:

```bash
bash scripts/download_qwen_embedding.sh
```

---

## Running the Code

Run all commands from the repository root.

### 1. Smoke Check

Before launching expensive runs, verify that required files are present and the Python/shell entrypoints are syntactically valid:

```bash
bash scripts/smoke_preflight.sh
```

This does not run a full benchmark. It is intended as a quick repository sanity check.

### 2. Build LoCoMo Memories

RaMem first builds and freezes one LanceDB memory database per LoCoMo sample.

For Qwen or another local OpenAI-compatible backend:

```bash
export OPENAI_API_KEY="vllm-local"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export QWEN_LLM_MODEL="Qwen/Qwen3-8B"
export QWEN_RUN_SLUG="qwen3_8b"
export QWEN_BACKBONE_LABEL="Qwen3-8B"
export EMBEDDING_MODEL_PATH=/path/to/qwen3-embedding-0.6b

python cli/build_memory.py \
  --model qwen \
  --sample-idx all \
  --dataset locomo10.json \
  --force
```

For GPT-family models:

```bash
export OPENAI_API_KEY="your-api-key"

python cli/build_memory.py \
  --model gpt \
  --openai-model gpt-4.1-mini \
  --sample-idx all \
  --dataset locomo10.json \
  --force
```

Useful options:

```bash
--sample-idx all        # build all LoCoMo samples
--sample-idx 0          # build one sample
--sample-idx 0,1,2      # build selected samples
--force                 # rebuild even if frozen DBs already exist
--dataset PATH          # path to locomo10.json
```

### 3. Evaluate on LoCoMo

After memory construction, run RaMem evaluation:

```bash
python cli/ramem_eval.py \
  --model qwen \
  --samples 0 1 2 3 4 5 6 7 8 9 \
  --parallel-questions \
  --eval-workers 8
```

For GPT-family models:

```bash
python cli/ramem_eval.py \
  --model gpt \
  --openai-model gpt-4.1-mini
```

For LLaMA-family local backbones served through an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="vllm-local"
export OPENAI_BASE_URL="http://localhost:8000/v1"

python cli/build_memory.py --model llama31_8b --sample-idx all --force
python cli/ramem_eval.py --model llama31_8b
```

Supported `--model` values are:

```text
gpt
qwen
llama31_8b
llama32_3b
```

LoCoMo outputs are written to:

```text
results/locomo/
results/locomo_contexts/
db/
```

### 4. Run Local Qwen with vLLM

The repository includes shell helpers for local Qwen experiments. These scripts can start vLLM, build memories, and evaluate with the same RaMem pipeline.

Configure paths:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
export QWEN_LLM_MODEL="Qwen/Qwen3-8B"
export QWEN_SERVED_MODEL_NAME="Qwen/Qwen3-8B"
export QWEN_RUN_SLUG="qwen3_8b"
export QWEN_BACKBONE_LABEL="Qwen3-8B"
export EMBEDDING_MODEL_PATH=/path/to/qwen3-embedding-0.6b
```

Start vLLM manually:

```bash
bash scripts/start_vllm.sh
```

In another terminal, build and evaluate:

```bash
bash scripts/build_memory_qwen.sh --sample-idx all --force
bash scripts/eval_qwen_parallel.sh
```

You can also run a full Qwen variant pipeline:

```bash
bash scripts/run_qwen_variant_pipeline.sh qwen3_8b Qwen/Qwen3-8B /path/to/Qwen3-8B
```

### 5. Evaluate on LongMemEval-S

Run a small pilot first:

```bash
export OPENAI_API_KEY="your-api-key"

python cli/run_longmemeval_s.py \
  --dataset longmemeval_s_cleaned.json \
  --output_dir results/longmemeval_s/ramem_pilot \
  --generator_model gpt-4.1-mini \
  --judge_model gpt-4.1-mini \
  --judge_protocol official \
  --pilot \
  --pilot_n 5 \
  --overwrite
```

Run the full LongMemEval-S evaluation:

```bash
python cli/run_longmemeval_s.py \
  --dataset longmemeval_s_cleaned.json \
  --output_dir results/longmemeval_s/ramem_gpt41mini \
  --generator_model gpt-4.1-mini \
  --judge_model gpt-4.1-mini \
  --judge_protocol official \
  --memory_layout session \
  --memory_workers 4 \
  --retrieval_workers 4 \
  --no-enable_reflection \
  --overwrite
```

Useful LongMemEval-S options:

```bash
--limit N                         # run only N examples
--start N                         # start from dataset row N
--sample_indices 1,5,9            # run specific dataset row indices
--memory_layout session           # preserve LongMemEval session boundaries
--memory_layout sliding           # use fixed sliding windows
--judge_protocol official         # official LongMemEval-style yes/no judging
--answer_context_max_chars N      # cap generation context length
--semantic_top_k N
--keyword_top_k N
--structured_top_k N
```

LongMemEval-S outputs include generation records, judge records, summary metrics, paper-style task averages, and an official-style hypothesis JSONL file under `--output_dir`.

---

## Runtime Switches

The most commonly used environment variables are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `vllm-local` for local runs | API key for OpenAI-compatible calls |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` for local Qwen/LLaMA | OpenAI-compatible endpoint |
| `JUDGE_API_KEY` | `OPENAI_API_KEY` | Optional separate judge API key |
| `JUDGE_BASE_URL` | `OPENAI_BASE_URL` | Optional separate judge endpoint |
| `LLM_MODEL` | backend-specific | Generator/planner model |
| `QWEN_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Qwen model name for local runs |
| `QWEN_RUN_SLUG` | `qwen` | Namespaces Qwen outputs and frozen DBs |
| `EMBEDDING_MODEL_PATH` | unset | Local embedding model directory |
| `ENABLE_PLANNING` | `true` | Enable recall condition induction |
| `ENABLE_REFLECTION` | `true` | Enable retrieval reflection |
| `ENABLE_TEMPORAL` | `true` | Enable temporal/session-aware retrieval |
| `ANSWER_CONTEXT_MAX_CHARS` | `0` | Cap generated-answer context length; `0` means no cap |
| `EVAL_MAX_WORKERS` | `4` or script-specific | Question-level evaluation workers |

---

## Reproducibility Notes

- Keep raw benchmark datasets outside Git unless redistribution is permitted.
- Do not commit generated databases, result files, model checkpoints, caches, or API keys.
- Record the model name, endpoint, `QWEN_RUN_SLUG`, embedding model, dataset version, and runtime switches for each run.
- For local vLLM experiments, ensure the served model name matches `QWEN_SERVED_MODEL_NAME` or the model name used by your OpenAI-compatible client.
- For expensive evaluations, start with `--sample-idx 0`, `--samples 0`, or LongMemEval-S `--pilot` before launching full runs.

---

## Citation

If you use RaMem in your research, please cite our paper:

```bibtex
@misc{ramem,
  title  = {RaMem: Contextual Reinstatement for Long-term Agentic Memory},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Under review}
}
```
