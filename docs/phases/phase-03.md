# Phase 3: Embeddings And Indexing

## What This Phase Includes

1. Batch embedding pipeline with configurable BGE provider and deterministic local fallback
2. Versioned embedding artifacts stored as JSONL for auditability and reproducibility
3. Dense index builder with FAISS support and a native cosine-search fallback for dependency-light runs
4. Sparse BM25 index builder with Whoosh support and a native BM25 fallback
5. ID mapping store that links row positions back to chunk and document metadata
6. Build manifest capturing embedding version, backend choices, and artifact paths
7. Runnable indexing CLI and tests covering end-to-end artifact generation and searchability

## Example Usage

```powershell
.\.venv\Scripts\python scripts/run_indexing.py --chunks-file data/processed/chunks.jsonl --version phase3-local --embedding-provider hash --dense-backend native --sparse-backend native
```

## Recommended Commit Message

```text
feat(phase-03): add modular embeddings and indexing pipeline with versioned artifacts
```

## What This Demonstrates To Recruiters

- Clear separation between offline preprocessing, embedding generation, and index materialization
- Production-minded artifact management through manifests, version labels, and explicit ID mapping
- Practical handling of heavyweight ML dependencies with graceful fallbacks for local development and CI
- Retrieval infrastructure that is ready for hybrid retrieval and evaluation in later phases

## Independently Runnable Scope

Phase 3 consumes the chunk artifact from phase 2 and produces versioned retrieval assets:

- input: JSONL chunk records from the ingestion pipeline
- output: embeddings, dense index, sparse index, ID mapping, and manifest files under a versioned index directory
