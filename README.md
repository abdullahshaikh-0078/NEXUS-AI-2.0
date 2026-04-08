# NEXUS AI 2.0

Production-grade, modular Retrieval-Augmented Generation platform with a clean full-stack repository layout.

## Structure

```text
.
|-- backend/
|   |-- .env.example
|   |-- Dockerfile
|   |-- README.md
|   |-- config/
|   |-- deploy/
|   |-- scripts/
|   |   |-- api/
|   |   |-- evaluation/
|   |   |-- pipelines/
|   |   `-- shared/
|   |-- src/
|   `-- tests/
|-- data/
|   `-- evaluation/
|-- docs/
|   `-- phases/
|-- frontend/
|   |-- README.md
|   |-- assets/
|   |   |-- css/
|   |   `-- js/
|   |       |-- app/
|   |       |-- fixtures/
|   |       |-- renderers/
|   |       |-- services/
|   |       `-- utils/
|   `-- index.html
|-- docker-compose.yml
|-- pyproject.toml
`-- README.md
```

## What Lives Where

- `backend/`
  Python service code, API, evaluation system, backend config, deployment assets, concern-grouped scripts, and backend tests.
- `frontend/`
  Browser website assets only: HTML, CSS, and JavaScript, grouped by app state, rendering, services, utilities, and fixtures.
- `data/`
  Shared corpus and evaluation datasets.
- `docs/`
  Phase notes and project documentation.
- root
  Workspace-level files such as `pyproject.toml`, `docker-compose.yml`, and this overview.

## Backend Quickstart

1. Install the backend in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
```

2. Create the backend env file:

```powershell
Copy-Item backend/.env.example backend/.env
```

3. Run the API:

```powershell
.\.venv\Scripts\python backend/scripts/api/run_api.py
```

4. Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

## Frontend Quickstart

```powershell
python -m http.server 5500
```

Then open `http://127.0.0.1:5500/frontend/`

## Evaluation Quickstart

Generate a starter dataset:

```powershell
.\.venv\Scripts\python backend/scripts/evaluation/generate_eval_dataset.py --chunks-file data/processed/chunks.jsonl --output-file data/evaluation/generated_eval_dataset.jsonl --max-documents 5
```

Run experiments:

```powershell
.\.venv\Scripts\python backend/scripts/evaluation/run_evaluation.py --dataset-path data/evaluation/generated_eval_dataset.jsonl --manifest-path data/indexes/v1/manifest.json --systems bm25 dense hybrid
```

## Deployment

```powershell
docker compose up --build
```

The backend image is built from [backend/Dockerfile](backend/Dockerfile).

