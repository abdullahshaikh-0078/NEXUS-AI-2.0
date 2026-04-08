# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1 through Phase 16B are implemented. The project now covers the full RAG lifecycle from ingestion and indexing to online serving, security hardening, and reproducible benchmarking:

- FastAPI service with standard and streaming query endpoints
- YAML + environment configuration with production-focused controls
- Ingestion, chunking, metadata enrichment, embeddings, and indexing
- Query processing, hybrid retrieval, reranking, context construction, and grounded generation
- Structured post-processing with confidence scoring and citations
- Caching, monitoring, scaling, resilience, cost controls, and API security
- Browser-based website with mock/live mode, streaming-ready UX, citations, debug evidence, and API key support
- Evaluation and experimentation pipeline comparing BM25, dense, and hybrid retrieval with Recall@K, MRR, NDCG, faithfulness, hallucination rate, latency breakdowns, JSON/CSV/Markdown/SVG artifacts, and baseline improvement tracking

## Repository Structure

```text
.
|-- config/
|   `-- base.yaml
|-- data/
|   `-- evaluation/
|       |-- README.md
|       `-- sample_eval_dataset.jsonl
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
|       |-- phase-15.md
|       |-- phase-16.md
|       `-- phase-16b.md
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
|   |-- generate_eval_dataset.py
|   |-- run_api.py
|   |-- run_context_pipeline.py
|   |-- run_evaluation.py
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
|       |-- context/
|       |-- core/
|       |-- evaluation/
|       |-- generation/
|       |-- indexing/
|       |-- ingestion/
|       |-- postprocessing/
|       |-- query/
|       |-- reranking/
|       |-- retrieval/
|       |-- services/
|       `-- main.py
|-- tests/
|   |-- test_context_pipeline.py
|   |-- test_cost_controls.py
|   |-- test_evaluation_metrics.py
|   |-- test_evaluation_runner.py
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

7. Run the full test suite:

```powershell
.\.venv\Scripts\python -m pytest
```

## Evaluation Quickstart

1. Generate a starter evaluation dataset from your chunk file:

```powershell
.\.venv\Scripts\python scripts/generate_eval_dataset.py --chunks-file data/processed/chunks.jsonl --output-file data/evaluation/generated_eval_dataset.jsonl --max-documents 5
```

2. Run benchmark experiments:

```powershell
.\.venv\Scripts\python scripts/run_evaluation.py --dataset-path data/evaluation/generated_eval_dataset.jsonl --manifest-path data/indexes/v1/manifest.json --systems bm25 dense hybrid
```

3. Inspect the generated artifacts under `artifacts/evaluation/<timestamp>/`

## Website Quickstart

The website lives in `frontend/` as a plain HTML, CSS, and JavaScript frontend.

1. Start a simple static server from the repository root:

```powershell
python -m http.server 5500
```

2. Open the site at `http://127.0.0.1:5500/frontend/`

3. Use `Switch to live` when the backend is running.

4. If API auth is enabled, enter your API key in the website header and save it before sending a live query.

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
$env:RAG_SECURITY__REQUIRE_API_KEY="true"
$env:RAG_SECURITY__API_KEYS='["change-me"]'
$env:RAG_EVALUATION__DATASET_PATH="data/evaluation/generated_eval_dataset.jsonl"
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
- [Phase 16 notes](docs/phases/phase-16.md)
- [Phase 16B notes](docs/phases/phase-16b.md)

## Final Notes

The repository now has both the product-facing path and the research-facing path:

- product path: ingest -> index -> query API -> website demo
- research path: dataset -> benchmark runner -> comparison table -> reproducible artifacts
