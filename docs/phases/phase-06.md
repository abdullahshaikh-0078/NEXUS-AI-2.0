# Phase 6: Reranking

## What This Phase Includes

1. Dedicated reranking layer that consumes phase 5 hybrid retrieval candidates
2. Cross-encoder reranker interface using `sentence-transformers`
3. Deterministic heuristic fallback reranker for local development and CI
4. Configurable top-K selection and candidate limit controls
5. Structured reranking result with retrieval provenance retained per candidate
6. Runnable reranking CLI that chains retrieval and second-stage ranking
7. Tests covering rerank ordering, top-K trimming, and reuse of existing retrieval results

## Example Usage

```powershell
.\.venv\Scripts\python backend/scripts/pipelines/run_reranking_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

## Recommended Commit Message

```text
feat(phase-06): add modular reranking pipeline with cross-encoder fallback
```

## What This Demonstrates To Recruiters

- Real multi-stage retrieval architecture with explicit separation between recall and precision layers
- Production-minded fallback strategy that keeps the system runnable even when heavyweight ranking models are unavailable
- Practical understanding of search quality engineering through candidate pruning and second-stage scoring
- Clean interfaces that set up context construction and answer generation without later rewrites

## Independently Runnable Scope

Phase 6 consumes the hybrid retrieval output from phase 5 and returns the final top-K ranked candidates:

- input: raw user query plus phase 3 manifest path
- output: reranked candidate list with rerank score, retrieval provenance, and top-K selection


