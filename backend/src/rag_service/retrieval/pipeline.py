from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.indexing.dense import search_dense_index
from rag_service.indexing.embedders import create_embedder
from rag_service.indexing.models import SearchHit
from rag_service.indexing.sparse import search_sparse_index
from rag_service.query.models import ProcessedQuery
from rag_service.query.pipeline import process_query
from rag_service.retrieval.fusion import reciprocal_rank_fusion
from rag_service.retrieval.loaders import load_retrieval_artifacts
from rag_service.retrieval.models import HybridRetrievalResult, RetrievedChunk

logger = get_logger(__name__)


def hybrid_retrieve(
    query: str,
    settings: Settings,
    manifest_path: Path | None = None,
    processed_query: ProcessedQuery | None = None,
) -> HybridRetrievalResult:
    query_state = processed_query or process_query(query, settings=settings)
    active_manifest_path = manifest_path or Path(settings.retrieval.manifest_path)
    manifest, chunk_lookup = load_retrieval_artifacts(active_manifest_path)

    embedder = create_embedder(settings)
    query_vector = embedder.embed_texts([query_state.rewritten_query])[0]

    dense_hits = search_dense_index(
        manifest.dense,
        query_vector=query_vector,
        top_k=settings.retrieval.dense_top_k,
    )
    sparse_hits = search_sparse_index(
        manifest.sparse,
        query=query_state.rewritten_query,
        top_k=settings.retrieval.sparse_top_k,
    )
    fused_hits = reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        rrf_k=settings.retrieval.rrf_k,
        top_k=settings.retrieval.candidate_pool_size,
    )

    dense_lookup = {hit.chunk_id: hit for hit in dense_hits}
    sparse_lookup = {hit.chunk_id: hit for hit in sparse_hits}

    logger.info(
        "hybrid_retrieval_completed",
        query=query_state.rewritten_query,
        dense_hits=len(dense_hits),
        sparse_hits=len(sparse_hits),
        fused_hits=len(fused_hits),
        manifest_path=str(active_manifest_path),
    )

    return HybridRetrievalResult(
        processed_query=query_state,
        dense_hits=_materialize_hits(dense_hits, chunk_lookup, source="dense", dense_lookup=dense_lookup, sparse_lookup=sparse_lookup),
        sparse_hits=_materialize_hits(sparse_hits, chunk_lookup, source="sparse", dense_lookup=dense_lookup, sparse_lookup=sparse_lookup),
        fused_hits=_materialize_hits(fused_hits[: settings.retrieval.fused_top_k], chunk_lookup, source="rrf", dense_lookup=dense_lookup, sparse_lookup=sparse_lookup),
        dense_raw_hits=dense_hits,
        sparse_raw_hits=sparse_hits,
    )


def _materialize_hits(
    hits: list[SearchHit],
    chunk_lookup: dict[str, object],
    *,
    source: str,
    dense_lookup: dict[str, SearchHit],
    sparse_lookup: dict[str, SearchHit],
) -> list[RetrievedChunk]:
    dense_ranks = {hit.chunk_id: index for index, hit in enumerate(dense_lookup.values(), start=1)}
    sparse_ranks = {hit.chunk_id: index for index, hit in enumerate(sparse_lookup.values(), start=1)}

    materialized: list[RetrievedChunk] = []
    for hit in hits:
        chunk = chunk_lookup[hit.chunk_id]
        dense_hit = dense_lookup.get(hit.chunk_id)
        sparse_hit = sparse_lookup.get(hit.chunk_id)
        materialized.append(
            RetrievedChunk(
                row_id=hit.row_id,
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=chunk.text,
                metadata=chunk.metadata,
                source=source,
                dense_rank=dense_ranks.get(hit.chunk_id),
                sparse_rank=sparse_ranks.get(hit.chunk_id),
                dense_score=dense_hit.score if dense_hit else None,
                sparse_score=sparse_hit.score if sparse_hit else None,
            )
        )
    return materialized
