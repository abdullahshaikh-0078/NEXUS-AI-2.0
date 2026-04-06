# Phase 4: Query Pipeline

## What This Phase Includes

1. Query cleaning with whitespace normalization, Unicode cleanup, and control-character stripping
2. Query normalization and tokenization for downstream retrieval compatibility
3. Domain-aware query expansion for RAG, retrieval, latency, caching, and indexing terminology
4. Rule-based query rewriting that converts terse keyword input into retrieval-friendly questions
5. Optional OpenAI rewrite fallback for ambiguous or very short queries
6. Runnable query-processing CLI that outputs structured JSON for local inspection
7. Tests covering cleaning, expansion, rule-based rewriting, and fallback behavior

## Example Usage

```powershell
.\.venv\Scripts\python scripts/run_query_pipeline.py "rag bm25 latency"
```

## Recommended Commit Message

```text
feat(phase-04): add modular query processing and rewriting pipeline
```

## What This Demonstrates To Recruiters

- Strong understanding of the online half of a production RAG system before retrieval even begins
- Practical query-engineering design with deterministic behavior, config control, and optional model assistance
- Clean abstraction boundaries that prepare the system for hybrid retrieval and API integration in later phases
- Testing discipline around both heuristic behavior and graceful fallback paths

## Independently Runnable Scope

Phase 4 runs without retrieval execution and produces a structured processed-query artifact:

- input: raw user query string
- output: cleaned query, normalized query, expansions, rewrite strategy, and rewritten query
