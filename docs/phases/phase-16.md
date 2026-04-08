# Phase 16 - Evaluation System

## What This Phase Adds

Phase 16 introduces a reproducible evaluation layer for the RAG stack:

- JSONL evaluation dataset support
- retrieval metrics including Recall@K, MRR, and NDCG@K
- answer-grounding metrics including faithfulness and hallucination rate
- full benchmark artifacts written as JSON, CSV, Markdown, and SVG

## Key Files

- `src/rag_service/evaluation/dataset.py`
- `src/rag_service/evaluation/metrics.py`
- `src/rag_service/evaluation/runner.py`
- `src/rag_service/evaluation/reporting.py`
- `scripts/generate_eval_dataset.py`
- `scripts/run_evaluation.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_runner.py`

## Why It Matters

This phase closes the loop between system design and measurable quality. It gives the project recruiter-grade evidence instead of just feature claims.

## Suggested Commit Message

`feat(phase-16): add reproducible rag evaluation metrics and benchmark runner`
