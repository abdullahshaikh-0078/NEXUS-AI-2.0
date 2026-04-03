# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in recruiter-friendly phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1 through Phase 8 are implemented, and the browser-based website scaffold is available with mocked responses:

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
- Grounded generation with prompt templates, inline citations, and OpenAI-to-heuristic fallback
- Browser-based HTML, CSS, and JavaScript website with chat, markdown answers, citations, debug panel, and mocked API integration
- Runnable ingestion, indexing, query, retrieval, reranking, context, and generation CLIs with test coverage

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
|       |-- phase-07.md
|       `-- phase-08.md
|-- frontend/
|   |-- assets/
|   |   |-- css/
|   |   |   `-- main.css
|   |   `-- js/
|   |       |-- api.js
|   |       |-- app.js
|   |       |-- markdown.js
|   |       |-- mock-data.js
|   |       `-- ui.js
|   `-- index.html
|-- scripts/
|   |-- run_api.py
|   |-- run_context_pipeline.py
|   |-- run_generation_pipeline.py
|   |-- run_indexing.py
|   |-- run_ingestion.py
|   |-- run_query_pipeline.py
|   |-- run_reranking_pipeline.py
|   `-- run_retrieval_pipeline.py
|-- src/
|   `-- rag_service/
|       |-- api/
|       |-- context/
|       |-- core/
|       |-- generation/
|       |   |-- generators.py
|       |   |-- models.py
|       |   |-- pipeline.py
|       |   `-- prompts.py
|       |-- indexing/
|       |-- ingestion/
|       |-- query/
|       |-- reranking/
|       |-- retrieval/
|       `-- main.py
|-- tests/
|   |-- conftest.py
|   |-- test_context_pipeline.py
|   |-- test_generation_pipeline.py
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

## Backend Quickstart

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
```

2. Copy environment defaults:

```powershell
Copy-Item .env.example .env
```

3. Run the API skeleton:

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

10. Generate a grounded answer:

```powershell
.\.venv\Scripts\python scripts/run_generation_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

11. Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

## Website Quickstart

The website lives in `frontend/` as a plain HTML, CSS, and JavaScript frontend. It is intentionally decoupled from the backend until phase 10 adds the production `/query` endpoint.

1. Open it with any static web server. For example, from the repository root:

```powershell
python -m http.server 5500
```

2. Open the site at `http://127.0.0.1:5500/frontend/`

You can also use VS Code Live Server with `frontend/index.html`.

The frontend currently runs against mocked responses by default. When phase 10 step 1 lands, update `frontend/assets/js/api.js` to switch from mock mode to the real backend URL.

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
$env:RAG_GENERATION__PROVIDER="heuristic"
.\.venv\Scripts\python scripts/run_generation_pipeline.py "hybrid retrieval latency"
```

## Phase Deliverables

- [Phase 1 notes](docs/phases/phase-01.md)
- [Phase 2 notes](docs/phases/phase-02.md)
- [Phase 3 notes](docs/phases/phase-03.md)
- [Phase 4 notes](docs/phases/phase-04.md)
- [Phase 5 notes](docs/phases/phase-05.md)
- [Phase 6 notes](docs/phases/phase-06.md)
- [Phase 7 notes](docs/phases/phase-07.md)
- [Phase 8 notes](docs/phases/phase-08.md)

## Next Phases

The structure is now prepared for:

- post-processing and confidence modeling
- production `/query` API integration for the website
- caching, monitoring, deployment, and evaluation layers
