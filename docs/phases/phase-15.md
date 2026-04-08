# Phase 15 - Security

## What This Phase Adds

Phase 15 hardens the public API surface:

- API-key protection for query endpoints
- prompt-injection filtering with sanitize or block modes
- stricter input sanitation and query length enforcement
- security response headers added through middleware

## Key Files

- `backend/src/rag_service/core/security.py`
- `backend/src/rag_service/core/exceptions.py`
- `backend/src/rag_service/api/app.py`
- `backend/src/rag_service/api/routes/query.py`
- `backend/tests/test_security_controls.py`

## Why It Matters

This phase shows that the project is designed for hostile environments and real deployment, not just offline demos.

## Suggested Commit Message

`feat(phase-15): add api authentication and prompt-injection defenses`


