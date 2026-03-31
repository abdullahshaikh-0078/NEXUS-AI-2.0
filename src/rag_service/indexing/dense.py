from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.indexing.models import DenseIndexManifest, SearchHit

logger = get_logger(__name__)


class DenseBackend(Protocol):
    backend_name: str

    def build(
        self,
        embeddings: list[list[float]],
        chunk_ids: list[str],
        output_path: Path,
        settings: Settings,
    ) -> DenseIndexManifest:
        ...


class NativeDenseBackend:
    backend_name = "native"

    def build(
        self,
        embeddings: list[list[float]],
        chunk_ids: list[str],
        output_path: Path,
        settings: Settings,
    ) -> DenseIndexManifest:
        payload = {"chunk_ids": chunk_ids, "vectors": embeddings}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return DenseIndexManifest(
            backend=self.backend_name,
            index_type="bruteforce",
            metric="cosine",
            index_path=str(output_path),
            vectors_indexed=len(embeddings),
            dimensions=len(embeddings[0]) if embeddings else 0,
        )


class FaissDenseBackend:
    backend_name = "faiss"

    def build(
        self,
        embeddings: list[list[float]],
        chunk_ids: list[str],
        output_path: Path,
        settings: Settings,
    ) -> DenseIndexManifest:
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("faiss-cpu and numpy are required for the FAISS dense backend") from exc

        if not embeddings:
            raise ValueError("Cannot build a dense index from zero embeddings")

        matrix = np.array(embeddings, dtype="float32")
        dimensions = matrix.shape[1]

        if settings.indexing.dense_index_type == "hnsw":
            index = faiss.IndexHNSWFlat(
                dimensions,
                settings.indexing.dense_hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = settings.indexing.dense_hnsw_ef_construction
        else:
            index = faiss.IndexFlatIP(dimensions)

        index.add(matrix)
        faiss.write_index(index, str(output_path))
        output_path.with_suffix(".meta.json").write_text(
            json.dumps({"chunk_ids": chunk_ids}),
            encoding="utf-8",
        )

        return DenseIndexManifest(
            backend=self.backend_name,
            index_type=settings.indexing.dense_index_type,
            metric="cosine",
            index_path=str(output_path),
            vectors_indexed=len(chunk_ids),
            dimensions=dimensions,
        )


def create_dense_backend(settings: Settings) -> DenseBackend:
    backend = settings.indexing.dense_backend
    if backend == "faiss":
        try:
            import faiss  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            fallback = settings.indexing.dense_fallback_backend
            logger.warning(
                "dense_backend_fallback",
                preferred=backend,
                fallback=fallback,
                reason=str(exc),
            )
            backend = fallback

    if backend == "faiss":
        return FaissDenseBackend()
    if backend == "native":
        return NativeDenseBackend()

    raise ValueError(f"Unsupported dense backend: {backend}")


def search_dense_index(
    manifest: DenseIndexManifest,
    query_vector: list[float],
    top_k: int = 5,
) -> list[SearchHit]:
    if manifest.backend == "faiss":
        return _search_faiss_index(Path(manifest.index_path), query_vector, top_k)
    if manifest.backend == "native":
        return _search_native_index(Path(manifest.index_path), query_vector, top_k)
    raise ValueError(f"Unsupported dense manifest backend: {manifest.backend}")


def _search_native_index(path: Path, query_vector: list[float], top_k: int) -> list[SearchHit]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunk_ids: list[str] = payload["chunk_ids"]
    vectors: list[list[float]] = payload["vectors"]
    scored_hits = [
        SearchHit(row_id=index, chunk_id=chunk_ids[index], score=_cosine_similarity(query_vector, vector))
        for index, vector in enumerate(vectors)
    ]
    return sorted(scored_hits, key=lambda item: item.score, reverse=True)[:top_k]


def _search_faiss_index(path: Path, query_vector: list[float], top_k: int) -> list[SearchHit]:
    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("faiss-cpu and numpy are required to search the FAISS dense index") from exc

    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    chunk_ids: list[str] = metadata["chunk_ids"]
    index = faiss.read_index(str(path))
    scores, row_ids = index.search(np.array([query_vector], dtype="float32"), top_k)

    hits: list[SearchHit] = []
    for raw_score, row_id in zip(scores[0], row_ids[0], strict=False):
        if row_id < 0:
            continue
        hits.append(SearchHit(row_id=int(row_id), chunk_id=chunk_ids[int(row_id)], score=float(raw_score)))
    return hits


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(l_value * r_value for l_value, r_value in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
