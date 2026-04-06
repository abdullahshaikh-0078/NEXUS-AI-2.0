# Phase 14 - Cost Optimization

## What This Phase Adds

Phase 14 reduces unnecessary model spend:

- skip expensive LLM calls when retrieval confidence is already high
- trim generation context before prompt construction
- preserve cache-first behavior for repeated queries
- emit warnings when low-cost fallbacks are intentionally used

## Key Files

- `src/rag_service/core/costing.py`
- `src/rag_service/services/query_service.py`
- `tests/test_cost_controls.py`

## Why It Matters

This phase demonstrates awareness of real operating cost, not just model quality.

## Suggested Commit Message

`feat(phase-14): add confidence-aware llm skipping and context cost controls`
