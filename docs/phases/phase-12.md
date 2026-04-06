# Phase 12 - Scaling

## What This Phase Adds

Phase 12 introduces scaling controls at runtime and deployment time:

- concurrency admission controller for request backpressure
- multi-worker API startup in non-development environments
- Kubernetes HPA manifest and stronger deployment tuning
- runtime scaling snapshot exposed through metrics

## Key Files

- `src/rag_service/core/scaling.py`
- `src/rag_service/core/lifecycle.py`
- `src/rag_service/main.py`
- `deploy/kubernetes/deployment.yaml`
- `deploy/kubernetes/hpa.yaml`
- `tests/test_scaling_controls.py`

## Why It Matters

This phase demonstrates that the service can be reasoned about under load, not only on a single developer laptop.

## Suggested Commit Message

`feat(phase-12): add admission control worker tuning and horizontal scaling assets`
