# Phase 10: Production Infrastructure

## What This Phase Includes

1. FastAPI query API with `POST /api/v1/query` and structured validation
2. Async service orchestration with parallel dense and sparse retrieval execution
3. Cache layer with in-memory support, Redis-ready adapter, and TTL-based namespaces for query, retrieval, and embedding caches
4. Runtime metrics endpoint with latency, cache hit, confidence, and stage-level timing summaries
5. Deployment assets including backend/Dockerfile, Docker Compose, and Kubernetes manifests
6. API and service tests covering query execution, validation, caching, and metrics reporting

## Example Usage

```powershell
.\.venv\Scripts\python backend/scripts/api/run_api.py
```

Then query the service:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/query" -ContentType "application/json" -Body '{"query":"What is RAG?"}'
```

Inspect runtime metrics:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/metrics"
```

## Recommended Commit Message

```text
feat(phase-10): add production api layer async retrieval caching metrics and deployment assets
```

## What This Demonstrates To Recruiters

- End-to-end systems thinking from offline indexing through online serving
- Async orchestration and infrastructure-aware design, not just model-side logic
- Practical production features such as caching, validation, observability, and deployment manifests
- A backend shape that is ready for real browser integration and operational hardening

## Independently Runnable Scope

Phase 10 turns the earlier pipelines into a real service layer:

- input: `POST /api/v1/query` with a user query
- output: grounded answer, citations, confidence, debug evidence, stage latencies, and cache-aware metrics


