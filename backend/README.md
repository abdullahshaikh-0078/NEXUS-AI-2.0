# Backend

This folder contains the Python backend for NEXUS AI 2.0.

## Layout

- `config/`
  Runtime configuration files.
- `deploy/`
  Docker and Kubernetes deployment assets used by the backend service.
- `scripts/`
  Runnable entrypoints grouped into `api/`, `pipelines/`, `evaluation/`, and `shared/`.
- `src/`
  Application source code following the `src/` layout.
- `tests/`
  Backend unit and integration tests.

## Common Commands

```powershell
Copy-Item backend/.env.example backend/.env
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python backend/scripts/api/run_api.py
.\.venv\Scripts\python -m pytest
```

