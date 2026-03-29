# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation (RAG) system built in recruiter-friendly phases. Each phase is independently runnable, tested, and documented so the repository demonstrates both systems engineering discipline and applied ML platform design.

## Current Status

Phase 1 is implemented:

- Clean production repository structure
- YAML + environment-based configuration
- Structured logging with request lifecycle hooks
- Async FastAPI application skeleton
- Health endpoint
- Test suite for core startup and config behavior

## Repository Structure

```text
.
|-- config/
|   `-- base.yaml
|-- docs/
|   `-- phases/
|       `-- phase-01.md
|-- scripts/
|   `-- run_api.py
|-- src/
|   `-- rag_service/
|       |-- api/
|       |   |-- routes/
|       |   |   `-- health.py
|       |   `-- app.py
|       |-- core/
|       |   |-- config.py
|       |   |-- lifecycle.py
|       |   `-- logging.py
|       `-- main.py
|-- tests/
|   |-- conftest.py
|   `-- test_health.py
|-- .env.example
|-- .gitignore
`-- pyproject.toml
```

## Quickstart

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

2. Copy environment defaults:

```powershell
Copy-Item .env.example .env
```

3. Run the API:

```powershell
python scripts/run_api.py
```

4. Verify the service:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

5. Run tests:

```powershell
pytest
```

## Configuration

Configuration is layered in this order:

1. `config/base.yaml`
2. Environment variables prefixed with `RAG_`

Example overrides:

```powershell
$env:RAG_APP__ENV="production"
$env:RAG_LOGGING__LEVEL="DEBUG"
python scripts/run_api.py
```

## Phase 1 Deliverable

See [Phase 1 notes](docs/phases/phase-01.md) for:

- code delivered
- example usage
- recommended commit message
- what this phase demonstrates to recruiters

## Next Phases

The structure is intentionally prepared for:

- offline ingestion pipelines
- retrieval and ranking services
- evaluation workflows
- production deployment and observability
