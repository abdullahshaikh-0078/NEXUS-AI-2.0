# Phase 9: Post-Processing

## What This Phase Includes

1. Dedicated post-processing layer that converts grounded answers into structured response objects
2. Deterministic confidence scoring based on citation strength, coverage, context breadth, and fallback usage
3. Citation formatting that prepares reference-friendly metadata and markdown reference lists
4. Metadata packaging for provider, model, token usage, and context block counts
5. Runnable post-processing CLI and tests covering confidence scoring and formatted output

## Example Usage

```powershell
.\.venv\Scripts\python backend/scripts/pipelines/run_postprocessing_pipeline.py "hybrid retrieval metadata" --manifest-path data/indexes/v1/manifest.json
```

## Recommended Commit Message

```text
feat(phase-09): add structured post-processing with confidence scoring and citation formatting
```

## What This Demonstrates To Recruiters

- Mature response-shaping beyond raw LLM text generation
- Deterministic confidence heuristics that are explainable and production-friendly
- Clear separation between generation and API-ready output contracts
- System design awareness around provenance, metadata packaging, and trust signals

## Independently Runnable Scope

Phase 9 consumes the grounded answer from phase 8 and returns a structured response object:

- input: raw user query plus phase 3 manifest path
- output: answer text, formatted citations, confidence assessment, references markdown, and response metadata


