# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in recruiter-friendly phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1, Phase 2, and Phase 3 are implemented:

- FastAPI service skeleton with health endpoint
- YAML + environment configuration
- Structured logging and lifecycle hooks
- Offline ingestion pipeline for TXT, HTML, and PDF sources
- Parsing, cleaning, adaptive chunking, and metadata enrichment
- Batch embeddings with configurable BGE provider and deterministic fallback
- Dense and sparse indexing with versioned artifacts, ID mapping, and manifests
- Runnable ingestion and indexing CLIs with test coverage

## Repository Structure

```text
.
|-- config/
|   `-- base.yaml
|-- docs/
|   `-- phases/
|       |-- phase-01.md
|       |-- phase-02.md
|       `-- phase-03.md
|-- scripts/
|   |-- run_api.py
|   |-- run_indexing.py
|   `-- run_ingestion.py
|-- src/
|   `-- rag_service/
|       |-- api/
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
|       `-- main.py
|-- tests/
|   |-- conftest.py
|   |-- test_health.py
|   |-- test_indexing_pipeline.py
|   `-- test_ingestion_pipeline.py
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
.\.venv\Scripts\python scripts/run_indexing.py --chunks-file data/processed/chunks.jsonl --version phase3-local --embedding-provider hash --dense-backend native --sparse-backend native
```

6. Run tests:

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
.\.venv\Scripts\python scripts/run_ingestion.py
.\.venv\Scripts\python scripts/run_indexing.py
```

## Phase Deliverables

- [Phase 1 notes](docs/phases/phase-01.md)
- [Phase 2 notes](docs/phases/phase-02.md)
- [Phase 3 notes](docs/phases/phase-03.md)

## Next Phases

The structure is now prepared for:

- hybrid retrieval orchestration
- reranking and context construction
- evaluation and production deployment layers
