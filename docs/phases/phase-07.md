# Phase 7: Context Engineering

## What This Phase Includes

1. Context assembly layer that consumes phase 6 reranked candidates
2. Relevance filtering based on rerank score thresholds
3. Deduplication by document and normalized text
4. Token budget management across selected evidence blocks
5. Extractive compression for prompt-ready evidence snippets
6. Structured context package with metadata-rich headers and final context text
7. Runnable context CLI and tests covering budget enforcement and deduplication

## Example Usage

```powershell
.\.venv\Scripts\python backend/scripts/pipelines/run_context_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

## Recommended Commit Message

```text
feat(phase-07): add context engineering pipeline with budgeting and compression
```

## What This Demonstrates To Recruiters

- Strong understanding of prompt construction as a systems problem, not just a template-writing task
- Practical tradeoff management across recall, precision, and token budget constraints
- Clean context-packaging abstractions that prepare the system for grounded generation and citations
- Attention to production concerns like deterministic filtering, compression, and provenance retention

## Independently Runnable Scope

Phase 7 consumes the reranked candidates from phase 6 and returns a bounded, compressed context package:

- input: raw user query plus phase 3 manifest path
- output: selected context blocks, omitted candidates, token counts, and final prompt-ready context text


