# Korean Business Embedding Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a committed Python project that prepares Korean business-name and industry-name data for BGE-M3 similarity fine-tuning.

**Architecture:** A small Python package handles normalization, ingestion, and pair generation. CLI scripts prepare parquet datasets, train BGE-M3 with SentenceTransformers, evaluate cosine similarity, and upload artifacts to Hugging Face.

**Tech Stack:** Python 3.9+, pandas, pyarrow, sentence-transformers, datasets, huggingface-hub, pytest.

---

### Task 1: Core Package

**Files:**
- Create: `src/bizembed/normalize.py`
- Create: `src/bizembed/ingest.py`
- Create: `src/bizembed/pairs.py`
- Test: `tests/test_normalize.py`
- Test: `tests/test_ingest.py`
- Test: `tests/test_pairs.py`

- [x] Write tests for text normalization, source column mapping, same-industry positives, and same-name hard negatives.
- [x] Run `python3 -m pytest -q` and verify tests fail because `bizembed` does not exist.
- [x] Implement normalization, ingestion, and pair generation.
- [x] Run `python3 -m pytest -q` and verify tests pass.

### Task 2: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `docs/superpowers/specs/2026-05-29-business-embedding-design.md`

- [x] Define package metadata and dependencies.
- [x] Document data sources, local setup, data preparation, training, and Hugging Face upload.
- [x] Save the project design.

### Task 3: CLI Scripts

**Files:**
- Create: `scripts/prepare_dataset.py`
- Create: `scripts/train_bge_m3.py`
- Create: `scripts/evaluate_similarity.py`
- Create: `scripts/upload_to_hf.py`

- [x] Add dataset preparation from local CSV/XLSX/Parquet/ZIP inputs.
- [x] Add BGE-M3 training script using `MultipleNegativesRankingLoss`.
- [x] Add pair similarity evaluation.
- [x] Add Hugging Face Hub upload script.

### Task 4: Verification and Commit

**Files:**
- All project files.

- [x] Run the full test suite.
- [x] Inspect `git diff --check`.
- [x] Commit all project files.

### Task 5: Public Data Acquisition Layer

**Files:**
- Create: `src/bizembed/sources.py`
- Create: `src/bizembed/collect.py`
- Create: `scripts/list_sources.py`
- Create: `scripts/download_openapi.py`
- Create: `scripts/download_sbiz_by_industry.py`
- Create: `scripts/download_sbiz_all.py`
- Test: `tests/test_sources.py`
- Test: `tests/test_collect.py`

- [x] Write tests for source catalog and paginated data.go.kr request helpers.
- [x] Run tests and verify they fail because acquisition modules do not exist.
- [x] Implement source catalog, generic OpenAPI downloader, SBIZ industry downloader, and SBIZ all-industry downloader.
- [x] Run full verification and commit the acquisition layer.
