# Phase 8: LLM Generation

## What This Phase Includes

1. Modular generation layer that consumes the phase 7 context package
2. Prompt templates that enforce grounded answers and inline citation markers
3. Deterministic heuristic generator for offline development and tests
4. Optional OpenAI-backed generator with automatic fallback to the heuristic path
5. Citation packaging that preserves source metadata for later API and UI use
6. Runnable generation CLI and tests covering prompt construction and grounded output

## Example Usage

```powershell
.\.venv\Scripts\python scripts/run_generation_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

## Recommended Commit Message

```text
feat(phase-08): add grounded generation pipeline with prompt templates and citations
```

## What This Demonstrates To Recruiters

- Clear separation between retrieval, context engineering, and answer synthesis
- Production-minded fallback behavior when the external LLM path is unavailable
- Evidence-aware prompt design that keeps citations traceable through the generation layer
- Backend contracts that are ready for a future `/query` API and browser-based UI

## Independently Runnable Scope

Phase 8 consumes the grounded context produced in phase 7 and returns a grounded answer object:

- input: raw user query plus phase 3 manifest path
- output: prompt bundle, grounded markdown answer, inline citations, and selected evidence metadata
