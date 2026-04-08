# Evaluation Dataset Format

This directory contains example evaluation datasets for the phase 16 benchmarking pipeline.

Preferred dataset shape:

```json
{"query_id": "q1", "query": "What is hybrid retrieval?", "relevant_document_ids": ["doc-id-1"], "expected_answer": "Optional reference answer"}
```

You can also use chunk-level labels:

```json
{"query_id": "q2", "query": "What is RRF?", "relevant_chunk_ids": ["chunk-id-4"]}
```

To create a runnable dataset from your real corpus:

```powershell
.\.venv\Scripts\python scripts\generate_eval_dataset.py --chunks-file data/processed/chunks.jsonl --output-file data/evaluation/generated_eval_dataset.jsonl
```
