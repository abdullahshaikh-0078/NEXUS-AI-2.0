# Phase 13 - Failure Handling

## What This Phase Adds

Phase 13 hardens the online path against dependency failures:

- retry loops around dense, sparse, and generation backends
- circuit breakers for repeated dependency failures
- partial retrieval fallback when one backend is down
- graceful answer degradation instead of total request collapse

## Key Files

- `backend/src/rag_service/core/resilience.py`
- `backend/src/rag_service/services/query_service.py`
- `backend/tests/test_failure_handling.py`

## Why It Matters

This phase shows production maturity: the system keeps serving useful answers under degraded conditions.

## Suggested Commit Message

`feat(phase-13): add retries circuit breakers and graceful retrieval degradation`


