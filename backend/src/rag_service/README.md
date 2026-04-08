# Application Package

The backend application is organized by capability instead of framework layer.

- `api/`
  FastAPI application, routes, and schemas.
- `core/`
  Shared infrastructure concerns such as config, logging, caching, metrics, resilience, security, and scaling.
- `ingestion/`, `indexing/`, `query/`, `retrieval/`, `reranking/`, `context/`, `generation/`, `postprocessing/`
  The RAG pipeline stages from offline processing through answer construction.
- `evaluation/`
  Experimentation, metric computation, and reporting.
- `services/`
  Higher-level orchestration used by the API layer.
