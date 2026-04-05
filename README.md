# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in recruiter-friendly phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1 through Phase 10 are implemented, and the browser-based website scaffold is ready to connect to the backend query API:

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
- Structured post-processing with confidence scoring, citation formatting, and metadata packaging
- Production API layer with `POST /api/v1/query`, async retrieval orchestration, cache, metrics, and deployment assets
- Browser-based HTML, CSS, and JavaScript website with chat, markdown answers, citations, debug panel, and mock/live API integration
- Runnable ingestion, indexing, query, retrieval, reranking, context, generation, post-processing, and API tests

## Repository Structure

```text
.
|-- config/
|   `-- base.yaml
|-- deploy/
|   `-- kubernetes/
|       |-- deployment.yaml
|       `-- service.yaml
|-- docs/
|   `-- phases/
|       |-- phase-01.md
|       |-- phase-02.md
|       |-- phase-03.md
|       |-- phase-04.md
|       |-- phase-05.md
|       |-- phase-06.md
|       |-- phase-07.md
|       |-- phase-08.md
|       |-- phase-09.md
|       `-- phase-10.md
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
|   |-- run_postprocessing_pipeline.py
|   |-- run_query_pipeline.py
|   |-- run_reranking_pipeline.py
|   `-- run_retrieval_pipeline.py
|-- src/
|   `-- rag_service/
|       |-- api/
|       |   |-- routes/
|       |   |-- app.py
|       |   `-- schemas.py
|       |-- context/
|       |-- core/
|       |   |-- cache.py
|       |   |-- config.py
|       |   |-- lifecycle.py
|       |   |-- logging.py
|       |   `-- metrics.py
|       |-- generation/
|       |-- indexing/
|       |-- ingestion/
|       |-- postprocessing/
|       |-- query/
|       |-- reranking/
|       |-- retrieval/
|       |-- services/
|       |   `-- query_service.py
|       `-- main.py
|-- tests/
|   |-- conftest.py
|   |-- test_context_pipeline.py
|   |-- test_generation_pipeline.py
|   |-- test_health.py
|   |-- test_indexing_pipeline.py
|   |-- test_ingestion_pipeline.py
|   |-- test_postprocessing_pipeline.py
|   |-- test_query_api.py
|   |-- test_query_pipeline.py
|   |-- test_query_service.py
|   |-- test_reranking_pipeline.py
|   `-- test_retrieval_pipeline.py
|-- .env.example
|-- .gitignore
|-- docker-compose.yml
|-- Dockerfile
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

3. Run the API:

```powershell
.\.venv\Scripts\python scripts/run_api.py
```

4. Query the service:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/query" -ContentType "application/json" -Body '{"query":"What is RAG?"}'
```

5. Inspect runtime metrics:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/metrics"
```

6. Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

## Pipeline CLIs

The earlier phases remain independently runnable:

```powershell
.\.venv\Scripts\python scripts/run_ingestion.py --input-dir data/raw --output-file data/processed/chunks.jsonl --strategy structure_aware
.\.venv\Scripts\python scripts/run_indexing.py --chunks-file data/processed/chunks.jsonl --version v1 --embedding-provider hash --dense-backend native --sparse-backend native
.\.venv\Scripts\python scripts/run_query_pipeline.py "rag bm25 latency"
.\.venv\Scripts\python scripts/run_retrieval_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
.\.venv\Scripts\python scripts/run_reranking_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
.\.venv\Scripts\python scripts/run_context_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
.\.venv\Scripts\python scripts/run_generation_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
.\.venv\Scripts\python scripts/run_postprocessing_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

## Website Quickstart

The website lives in `frontend/` as a plain HTML, CSS, and JavaScript frontend.

1. Open it with any static web server. For example, from the repository root:

```powershell
python -m http.server 5500
```

2. Open the site at `http://127.0.0.1:5500/frontend/`

3. Use the `Switch to live` control in the UI when the backend API is running.

The frontend currently supports both mocked responses and the real `/api/v1/query` endpoint.

## Deployment

Docker:

```powershell
docker compose up --build
```

Kubernetes manifests:

- `deploy/kubernetes/deployment.yaml`
- `deploy/kubernetes/service.yaml`

## Configuration

Configuration is layered in this order:

1. `config/base.yaml`
2. Environment variables prefixed with `RAG_`

Example overrides:

```powershell
$env:RAG_APP__ENV="production"
$env:RAG_CACHE__PROVIDER="redis"
$env:RAG_CACHE__REDIS_URL="redis://localhost:6379/0"
$env:RAG_RETRIEVAL__MANIFEST_PATH="data/indexes/v1/manifest.json"
$env:RAG_GENERATION__PROVIDER="heuristic"
$env:RAG_POSTPROCESSING__HIGH_CONFIDENCE_THRESHOLD="0.8"
.\.venv\Scripts\python scripts/run_api.py
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
- [Phase 9 notes](docs/phases/phase-09.md)
- [Phase 10 notes](docs/phases/phase-10.md)

## Next Phases

The structure is now prepared for:

- latency optimization and streaming responses
- deeper scaling and failure-handling policies
- evaluation, experimentation, and production hardening
