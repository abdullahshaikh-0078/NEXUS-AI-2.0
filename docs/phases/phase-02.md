# Phase 2: Ingestion Pipeline

## What This Phase Includes

1. Multi-format ingestion for TXT, HTML, and PDF documents
2. Parsing and text cleaning with normalization of whitespace and layout noise
3. Adaptive chunking strategies:
   - fixed
   - semantic
   - structure-aware
4. Metadata enrichment at chunk level for downstream indexing
5. Runnable offline pipeline that writes JSONL chunk artifacts
6. Tests covering parsers, chunkers, and end-to-end ingestion output

## Example Usage

```powershell
.\.venv\Scripts\python scripts/run_ingestion.py --input-dir data/raw --output-file data/processed/chunks.jsonl --strategy structure_aware
```

## Recommended Commit Message

```text
feat(phase-02): add modular offline ingestion pipeline with adaptive chunking
```

## What This Demonstrates To Recruiters

- Real document processing engineering beyond toy RAG demos
- Separation of concerns between parsing, chunking, and orchestration layers
- Extensible design that can plug into embeddings and indexing without rework
- Practical handling of heterogeneous enterprise document formats

## Independently Runnable Scope

Phase 2 runs without retrieval dependencies and produces normalized chunk artifacts suitable for the next phase:

- input: directory of raw `.txt`, `.html`, and `.pdf` files
- output: JSONL file of chunk records with enriched metadata
