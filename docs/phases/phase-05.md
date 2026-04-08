# Phase 5: Hybrid Retrieval

## What This Phase Includes

1. Dense retrieval over the phase 3 vector index
2. Sparse BM25 retrieval over the phase 3 lexical index
3. Query pipeline integration so retrieval consumes the rewritten query from phase 4
4. Reciprocal Rank Fusion (RRF) for hybrid ranking
5. Candidate pool construction with configurable dense, sparse, and fused cutoffs
6. Runnable hybrid retrieval CLI that returns structured JSON hits
7. Tests covering end-to-end retrieval and fusion behavior

## Example Usage

```powershell
.\.venv\Scripts\python backend/scripts/pipelines/run_retrieval_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

## Recommended Commit Message

```text
feat(phase-05): add hybrid retrieval with dense sparse search and rrf fusion
```

## What This Demonstrates To Recruiters

- Practical hybrid retrieval design rather than a single-mode toy search stack
- Strong systems thinking across offline and online RAG layers through shared manifests and query-state reuse
- Retrieval engineering fundamentals including candidate pooling, ranking fusion, and reproducible pipeline boundaries
- Production-minded modularity that prepares the system for reranking and context assembly in later phases

## Independently Runnable Scope

Phase 5 consumes the indexing manifest from phase 3 and a processed query from phase 4 to produce hybrid retrieval candidates:

- input: raw user query plus phase 3 manifest path
- output: dense hits, sparse hits, and fused RRF candidates with chunk text and metadata


