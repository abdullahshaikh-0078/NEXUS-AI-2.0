from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.indexing.models import SearchHit, SparseIndexManifest

logger = get_logger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class SparseBackend(Protocol):
    backend_name: str

    def build(
        self,
        texts: list[str],
        chunk_ids: list[str],
        output_path: Path,
        settings: Settings,
    ) -> SparseIndexManifest:
        ...


class NativeSparseBackend:
    backend_name = "native"

    def build(
        self,
        texts: list[str],
        chunk_ids: list[str],
        output_path: Path,
        settings: Settings,
    ) -> SparseIndexManifest:
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        doc_lengths: list[int] = []

        for row_id, text in enumerate(texts):
            tokens = _tokenize(text)
            term_counts = Counter(tokens)
            doc_lengths.append(len(tokens))
            for term, count in term_counts.items():
                postings[term][str(row_id)] = count

        average_document_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        payload = {
            "chunk_ids": chunk_ids,
            "postings": postings,
            "doc_lengths": doc_lengths,
            "avg_doc_length": average_document_length,
            "k1": settings.indexing.bm25_k1,
            "b": settings.indexing.bm25_b,
            "total_docs": len(texts),
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")

        return SparseIndexManifest(
            backend=self.backend_name,
            index_path=str(output_path),
            documents_indexed=len(texts),
            average_document_length=average_document_length,
            k1=settings.indexing.bm25_k1,
            b=settings.indexing.bm25_b,
        )


class WhooshSparseBackend:
    backend_name = "whoosh"

    def build(
        self,
        texts: list[str],
        chunk_ids: list[str],
        output_path: Path,
        settings: Settings,
    ) -> SparseIndexManifest:
        try:
            from whoosh import fields, index
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Whoosh is required for the configured sparse backend") from exc

        output_path.mkdir(parents=True, exist_ok=True)
        schema = fields.Schema(
            row_id=fields.NUMERIC(stored=True, unique=True),
            chunk_id=fields.ID(stored=True),
            content=fields.TEXT(stored=False),
        )
        whoosh_index = index.create_in(str(output_path), schema=schema)
        writer = whoosh_index.writer()
        for row_id, (chunk_id, text) in enumerate(zip(chunk_ids, texts, strict=False)):
            writer.add_document(row_id=row_id, chunk_id=chunk_id, content=text)
        writer.commit()

        return SparseIndexManifest(
            backend=self.backend_name,
            index_path=str(output_path),
            documents_indexed=len(texts),
            average_document_length=0.0,
            k1=settings.indexing.bm25_k1,
            b=settings.indexing.bm25_b,
        )


def create_sparse_backend(settings: Settings) -> SparseBackend:
    backend = settings.indexing.sparse_backend
    if backend == "whoosh":
        try:
            import whoosh  # noqa: F401
        except ImportError as exc:
            fallback = settings.indexing.sparse_fallback_backend
            logger.warning(
                "sparse_backend_fallback",
                preferred=backend,
                fallback=fallback,
                reason=str(exc),
            )
            backend = fallback

    if backend == "whoosh":
        return WhooshSparseBackend()
    if backend == "native":
        return NativeSparseBackend()

    raise ValueError(f"Unsupported sparse backend: {backend}")


def search_sparse_index(
    manifest: SparseIndexManifest,
    query: str,
    top_k: int = 5,
) -> list[SearchHit]:
    if manifest.backend == "whoosh":
        return _search_whoosh_index(Path(manifest.index_path), query, top_k)
    if manifest.backend == "native":
        return _search_native_index(Path(manifest.index_path), query, top_k)
    raise ValueError(f"Unsupported sparse manifest backend: {manifest.backend}")


def _search_native_index(path: Path, query: str, top_k: int) -> list[SearchHit]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    postings: dict[str, dict[str, int]] = payload["postings"]
    doc_lengths: list[int] = payload["doc_lengths"]
    chunk_ids: list[str] = payload["chunk_ids"]
    total_docs: int = payload["total_docs"]
    average_document_length: float = payload["avg_doc_length"]
    k1: float = payload["k1"]
    b: float = payload["b"]

    scores: dict[int, float] = defaultdict(float)
    for term in _tokenize(query):
        term_postings = postings.get(term)
        if not term_postings:
            continue
        document_frequency = len(term_postings)
        inverse_document_frequency = math.log(
            1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for row_id_text, term_frequency in term_postings.items():
            row_id = int(row_id_text)
            document_length = doc_lengths[row_id] or 1
            denominator = term_frequency + k1 * (
                1 - b + b * (document_length / max(average_document_length, 1.0))
            )
            scores[row_id] += inverse_document_frequency * (term_frequency * (k1 + 1) / denominator)

    hits = [
        SearchHit(row_id=row_id, chunk_id=chunk_ids[row_id], score=score)
        for row_id, score in scores.items()
    ]
    return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


def _search_whoosh_index(path: Path, query: str, top_k: int) -> list[SearchHit]:
    try:
        from whoosh import index, qparser, scoring
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Whoosh is required to search the Whoosh sparse index") from exc

    whoosh_index = index.open_dir(str(path))
    parser = qparser.QueryParser("content", schema=whoosh_index.schema)
    parsed_query = parser.parse(query)

    with whoosh_index.searcher(weighting=scoring.BM25F()) as searcher:
        results = searcher.search(parsed_query, limit=top_k)
        return [
            SearchHit(row_id=int(result["row_id"]), chunk_id=result["chunk_id"], score=float(result.score))
            for result in results
        ]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
