# Phase 11 - Latency Optimization

## What This Phase Adds

Phase 11 focuses on request latency without breaking correctness:

- adaptive retrieval plans for short queries
- reduced candidate pools before reranking
- streaming-ready query endpoint at `POST /api/v1/query/stream`
- chunked answer emission so the frontend can render partial output later

## Key Files

- `src/rag_service/core/optimization.py`
- `src/rag_service/services/query_service.py`
- `src/rag_service/api/routes/query.py`
- `tests/test_latency_optimization.py`

## Why It Matters

This phase shows that the system is not only accurate, but also engineered to reduce user-perceived latency and prepare for token streaming in a production UI.

## Suggested Commit Message

`feat(phase-11): add adaptive retrieval planning and streaming-ready query responses`
