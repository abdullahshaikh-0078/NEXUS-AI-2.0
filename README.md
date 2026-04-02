# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in recruiter-friendly phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1 through Phase 7 are implemented:

- FastAPI service skeleton with health endpoint
- YAML + environment configuration
- Structured logging and lifecycle hooks
- Offline ingestion pipeline for TXT, HTML, and PDF sources
- Parsing, cleaning, adaptive chunking, and metadata enrichment
- Batch embeddings with configurable BGE provider and deterministic fallback
- Dense and sparse indexing with versioned artifacts, ID mapping, and manifests
- Query cleaning, normalization, domain-aware expansion, and rewriting with optional OpenAI fallback
- Hybrid retrieval with dense search, sparse search, candidate pooling, and reciprocal rank fusion
- Modular reranking with cross-encoder support and a deterministic fallback scorer
- Context engineering with filtering, deduplication, token budgeting, and extractive compression
- Runnable ingestion, indexing, query, retrieval, reranking, and context CLIs with test coverage

## Repository Structure

```text
.
|-- config/
|   `-- base.yaml
|-- docs/
|   `-- phases/
|       |-- phase-01.md
|       |-- phase-02.md
|       |-- phase-03.md
|       |-- phase-04.md
|       |-- phase-05.md
|       |-- phase-06.md
|       `-- phase-07.md
|-- scripts/
|   |-- run_api.py
|   |-- run_indexing.py
|   |-- run_ingestion.py
|   |-- run_query_pipeline.py
|   |-- run_retrieval_pipeline.py
|   |-- run_reranking_pipeline.py
|   `-- run_context_pipeline.py
|-- src/
|   `-- rag_service/
|       |-- api/
|       |-- context/
|       |   |-- compression.py
|       |   |-- models.py
|       |   `-- pipeline.py
|       |-- core/
|       |-- indexing/
|       |   |-- dense.py
|       |   |-- embedders.py
|       |   |-- loaders.py
|       |   |-- models.py
|       |   |-- pipeline.py
|       |   `-- sparse.py
|       |-- ingestion/
|       |   |-- chunkers.py
|       |   |-- cleaners.py
|       |   |-- models.py
|       |   |-- parsers.py
|       |   `-- pipeline.py
|       |-- query/
|       |   |-- cleaners.py
|       |   |-- expanders.py
|       |   |-- models.py
|       |   |-- pipeline.py
|       |   `-- rewriters.py
|       |-- retrieval/
|       |   |-- fusion.py
|       |   |-- loaders.py
|       |   |-- models.py
|       |   `-- pipeline.py
|       |-- reranking/
|       |   |-- models.py
|       |   |-- pipeline.py
|       |   `-- scorers.py
|       `-- main.py
|-- tests/
|   |-- conftest.py
|   |-- test_context_pipeline.py
|   |-- test_health.py
|   |-- test_indexing_pipeline.py
|   |-- test_ingestion_pipeline.py
|   |-- test_query_pipeline.py
|   |-- test_reranking_pipeline.py
|   `-- test_retrieval_pipeline.py
|-- .env.example
|-- .gitignore
`-- pyproject.toml
```

## Quickstart

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
```

2. Copy environment defaults:

```powershell
Copy-Item .env.example .env
```

3. Run the API:

```powershell
.\.venv\Scripts\python scripts/run_api.py
```

4. Run offline ingestion:

```powershell
.\.venv\Scripts\python scripts/run_ingestion.py --input-dir data/raw --output-file data/processed/chunks.jsonl --strategy structure_aware
```

5. Build embeddings and indexes:

```powershell
.\.venv\Scripts\python scripts/run_indexing.py --chunks-file data/processed/chunks.jsonl --version v1 --embedding-provider hash --dense-backend native --sparse-backend native
```

6. Process a query before retrieval:

```powershell
.\.venv\Scripts\python scripts/run_query_pipeline.py "rag bm25 latency"
```

7. Run hybrid retrieval:

```powershell
.\.venv\Scripts\python scripts/run_retrieval_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

8. Run reranking:

```powershell
.\.venv\Scripts\python scripts/run_reranking_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

9. Build grounded context:

```powershell
.\.venv\Scripts\python scripts/run_context_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

10. Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

## Configuration

Configuration is layered in this order:

1. `config/base.yaml`
2. Environment variables prefixed with `RAG_`

Example overrides:

```powershell
$env:RAG_APP__ENV="production"
$env:RAG_LOGGING__LEVEL="DEBUG"
$env:RAG_INGESTION__DEFAULT_STRATEGY="semantic"
$env:RAG_INDEXING__EMBEDDING_PROVIDER="sentence_transformers"
$env:RAG_INDEXING__DENSE_BACKEND="faiss"
$env:RAG_QUERY__ENABLE_LLM_FALLBACK="true"
$env:RAG_RETRIEVAL__MANIFEST_PATH="data/indexes/v1/manifest.json"
$env:RAG_RERANKING__PROVIDER="heuristic"
$env:RAG_CONTEXT__MAX_CONTEXT_TOKENS="700"
.\.venv\Scripts\python scripts/run_ingestion.py
.\.venv\Scripts\python scripts/run_indexing.py
.\.venv\Scripts\python scripts/run_query_pipeline.py "hybrid retrieval latency"
.\.venv\Scripts\python scripts/run_retrieval_pipeline.py "hybrid retrieval latency"
.\.venv\Scripts\python scripts/run_reranking_pipeline.py "hybrid retrieval latency"
.\.venv\Scripts\python scripts/run_context_pipeline.py "hybrid retrieval latency"
```

## Phase Deliverables

- [Phase 1 notes](docs/phases/phase-01.md)
- [Phase 2 notes](docs/phases/phase-02.md)
- [Phase 3 notes](docs/phases/phase-03.md)
- [Phase 4 notes](docs/phases/phase-04.md)
- [Phase 5 notes](docs/phases/phase-05.md)
- [Phase 6 notes](docs/phases/phase-06.md)
- [Phase 7 notes](docs/phases/phase-07.md)

## Next Phases

The structure is now prepared for:

- answer generation and citation injection
- evaluation and production deployment layers
