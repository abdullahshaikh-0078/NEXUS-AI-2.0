# Phase 16B - Experimentation and Metric Tracking

## What This Phase Adds

Phase 16B expands the evaluation system into an experimentation workflow:

- baseline comparison across BM25, dense, and hybrid retrieval
- automatic improvement tracking versus a chosen baseline
- timestamped benchmark run directories for reproducibility
- export-ready summary tables and benchmark plots

## Key Files

- `backend/src/rag_service/evaluation/models.py`
- `backend/src/rag_service/evaluation/reporting.py`
- `data/evaluation/README.md`
- `data/evaluation/sample_eval_dataset.jsonl`
- `backend/config/base.yaml`
- `backend/.env.example`

## Why It Matters

This phase makes the repository useful for interview storytelling and research discussion because each run now produces defendable benchmark artifacts rather than ad hoc screenshots.

## Suggested Commit Message

`feat(phase-16b): add experiment tracking baseline comparisons and benchmark artifacts`


