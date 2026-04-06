# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1 through Phase 15 are implemented, and the browser-based website is prepared to consume both the standard and streaming query APIs:

- FastAPI service skeleton with health endpoint and lifecycle management
- YAML + environment configuration with production-oriented phase controls
- Structured logging, caching, metrics, and deployment assets
- Offline ingestion for TXT, HTML, and PDF sources
- Parsing, cleaning, adaptive chunking, metadata enrichment, and versioned indexing
- Query cleaning, expansion, rewriting, hybrid retrieval, reranking, and context engineering
- Grounded generation, structured post-processing, citations, and confidence scoring
- Production API layer with `POST /api/v1/query`, `POST /api/v1/query/stream`, and `GET /api/v1/metrics`
- Latency optimization with adaptive retrieval plans and streaming-ready chunk emission
- Scaling controls with concurrency admission, multi-worker startup, and Kubernetes HPA manifests
- Failure handling with retry logic, circuit breakers, retrieval degradation, and graceful fallback
- Cost controls that trim generation context and skip expensive LLM calls when evidence is already strong
- Security hardening with API-key auth, prompt-injection filtering, and response hardening headers
- Browser-based HTML, CSS, and JavaScript website with chat, markdown answers, citations, debug panel, and mock/live API integration
- Runnable unit and integration tests across retrieval, generation, API, and production concerns

## Repository Structure

```text
.
|-- config/
|   `-- base.yaml
|-- deploy/
|   `-- kubernetes/
|       |-- deployment.yaml
|       |-- hpa.yaml
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
|       |-- phase-10.md
|       |-- phase-11.md
|       |-- phase-12.md
|       |-- phase-13.md
|       |-- phase-14.md
|       `-- phase-15.md
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
|       |   |-- costing.py
|       |   |-- exceptions.py
|       |   |-- lifecycle.py
|       |   |-- logging.py
|       |   |-- metrics.py
|       |   |-- optimization.py
|       |   |-- resilience.py
|       |   |-- scaling.py
|       |   `-- security.py
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
|   |-- test_cost_controls.py
|   |-- test_failure_handling.py
|   |-- test_generation_pipeline.py
|   |-- test_health.py
|   |-- test_indexing_pipeline.py
|   |-- test_ingestion_pipeline.py
|   |-- test_latency_optimization.py
|   |-- test_postprocessing_pipeline.py
|   |-- test_query_api.py
|   |-- test_query_pipeline.py
|   |-- test_query_service.py
|   |-- test_reranking_pipeline.py
|   |-- test_retrieval_pipeline.py
|   |-- test_scaling_controls.py
|   `-- test_security_controls.py
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

5. Stream a response:

```powershell
Invoke-WebRequest -Method Post -Uri "http://127.0.0.1:8000/api/v1/query/stream" -ContentType "application/json" -Body '{"query":"What is RAG?"}'
```

6. Inspect runtime metrics:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/metrics"
```

7. Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

## Website Quickstart

The website lives in `frontend/` as a plain HTML, CSS, and JavaScript frontend.

1. Start a simple static server from the repository root:

```powershell
python -m http.server 5500
```

2. Open the site at `http://127.0.0.1:5500/frontend/`

3. Use the `Switch to live` control in the UI when the backend API is running.

## Deployment

Docker:

```powershell
docker compose up --build
```

Kubernetes manifests:

- `deploy/kubernetes/deployment.yaml`
- `deploy/kubernetes/service.yaml`
- `deploy/kubernetes/hpa.yaml`

## Configuration

Configuration is layered in this order:

1. `config/base.yaml`
2. Environment variables prefixed with `RAG_`

Example overrides:

```powershell
$env:RAG_APP__ENV="production"
$env:RAG_CACHE__PROVIDER="redis"
$env:RAG_SCALING__MAX_CONCURRENT_QUERIES="16"
$env:RAG_RESILIENCE__ALLOW_PARTIAL_RETRIEVAL="true"
$env:RAG_COST__SKIP_LLM_FOR_HIGH_CONFIDENCE="true"
$env:RAG_SECURITY__REQUIRE_API_KEY="true"
$env:RAG_SECURITY__API_KEYS='["change-me"]'
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
- [Phase 11 notes](docs/phases/phase-11.md)
- [Phase 12 notes](docs/phases/phase-12.md)
- [Phase 13 notes](docs/phases/phase-13.md)
- [Phase 14 notes](docs/phases/phase-14.md)
- [Phase 15 notes](docs/phases/phase-15.md)

## Next Phases

The structure is now prepared for:

- evaluation and experimentation pipelines
- benchmark reporting and reproducibility artifacts
- end-to-end UI connection to the live and streaming backend in production mode
