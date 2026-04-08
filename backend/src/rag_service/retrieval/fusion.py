from __future__ import annotations

from rag_service.indexing.models import SearchHit


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    rrf_k: int,
    top_k: int,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    row_ids: dict[str, int] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        row_ids[hit.chunk_id] = hit.row_id

    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        row_ids[hit.chunk_id] = hit.row_id

    fused = [
        SearchHit(row_id=row_ids[chunk_id], chunk_id=chunk_id, score=score)
        for chunk_id, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]
