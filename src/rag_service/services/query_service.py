from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from urllib.parse import quote_plus

from rag_service.api.schemas import QueryCitation, QueryResponse, RetrievedDocument
from rag_service.context.pipeline import build_context
from rag_service.core.cache import CacheBackend, CacheNamespace
from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.core.metrics import MetricsRegistry
from rag_service.generation.pipeline import generate_grounded_answer
from rag_service.indexing.dense import search_dense_index
from rag_service.indexing.embedders import TextEmbedder, create_embedder
from rag_service.indexing.models import SearchHit
from rag_service.indexing.sparse import search_sparse_index
from rag_service.postprocessing.pipeline import postprocess_grounded_answer
from rag_service.query.models import ProcessedQuery
from rag_service.query.pipeline import process_query
from rag_service.reranking.pipeline import rerank_candidates
from rag_service.retrieval.fusion import reciprocal_rank_fusion
from rag_service.retrieval.loaders import load_retrieval_artifacts
from rag_service.retrieval.models import HybridRetrievalResult, RetrievedChunk

logger = get_logger(__name__)


class QueryService:
    def __init__(
        self,
        settings: Settings,
        cache_backend: CacheBackend,
        metrics: MetricsRegistry,
    ) -> None:
        self._settings = settings
        self._metrics = metrics
        self._query_cache = CacheNamespace(cache_backend, "query", settings.cache.query_ttl_seconds)
        self._retrieval_cache = CacheNamespace(cache_backend, "retrieval", settings.cache.retrieval_ttl_seconds)
        self._embedding_cache = CacheNamespace(cache_backend, "embedding", settings.cache.embedding_ttl_seconds)
        self._embedder: TextEmbedder = create_embedder(settings)

    async def answer(self, query: str) -> QueryResponse:
        start_time = perf_counter()
        cache_key = quote_plus(query.strip().lower())
        cached_payload = await self._query_cache.get(cache_key)
        if cached_payload is not None:
            response = QueryResponse.model_validate(cached_payload)
            response.cache_hit = True
            self._metrics.record_success(
                total_latency_ms=response.latency_ms,
                confidence_score=response.confidence_score,
                citation_count=len(response.citations),
                cache_hit=True,
            )
            return response

        stage_latencies: dict[str, float] = {}
        try:
            processed_query = await self._run_stage(
                "query_processing",
                stage_latencies,
                asyncio.to_thread(process_query, query, self._settings),
            )
            retrieval = await self._run_stage(
                "retrieval",
                stage_latencies,
                self._hybrid_retrieve_async(processed_query),
            )
            reranking = await self._run_stage(
                "reranking",
                stage_latencies,
                asyncio.to_thread(rerank_candidates, query, self._settings, retrieval),
            )
            context = await self._run_stage(
                "context",
                stage_latencies,
                asyncio.to_thread(build_context, query, self._settings, reranking),
            )
            grounded_answer = await self._run_stage(
                "generation",
                stage_latencies,
                asyncio.to_thread(generate_grounded_answer, query, self._settings, context),
            )
            structured = await self._run_stage(
                "postprocessing",
                stage_latencies,
                asyncio.to_thread(postprocess_grounded_answer, query, self._settings, grounded_answer),
            )

            total_latency_ms = round((perf_counter() - start_time) * 1000, 2)
            response = self._build_response(structured, stage_latencies, total_latency_ms)
            await self._query_cache.set(cache_key, response.model_dump(mode="json"))
            self._metrics.record_success(
                total_latency_ms=total_latency_ms,
                confidence_score=response.confidence_score,
                citation_count=len(response.citations),
                cache_hit=False,
            )
            return response
        except Exception:
            total_latency_ms = round((perf_counter() - start_time) * 1000, 2)
            self._metrics.record_failure(total_latency_ms)
            raise

    async def _run_stage(self, stage_name: str, stage_latencies: dict[str, float], awaitable):
        start_time = perf_counter()
        result = await awaitable
        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        stage_latencies[stage_name] = duration_ms
        self._metrics.record_stage(stage_name, duration_ms)
        return result

    async def _hybrid_retrieve_async(self, processed_query: ProcessedQuery) -> HybridRetrievalResult:
        manifest_path = Path(self._settings.retrieval.manifest_path)
        retrieval_key = quote_plus(f"{processed_query.rewritten_query}|{manifest_path}")
        cached_payload = await self._retrieval_cache.get(retrieval_key)
        if cached_payload is not None:
            return HybridRetrievalResult.model_validate(cached_payload)

        manifest, chunk_lookup = await asyncio.to_thread(load_retrieval_artifacts, manifest_path)
        query_vector = await self._get_query_vector(processed_query.rewritten_query)

        dense_hits, sparse_hits = await asyncio.gather(
            asyncio.to_thread(
                search_dense_index,
                manifest.dense,
                query_vector=query_vector,
                top_k=self._settings.retrieval.dense_top_k,
            ),
            asyncio.to_thread(
                search_sparse_index,
                manifest.sparse,
                query=processed_query.rewritten_query,
                top_k=self._settings.retrieval.sparse_top_k,
            ),
        )

        fused_hits = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            rrf_k=self._settings.retrieval.rrf_k,
            top_k=self._settings.retrieval.candidate_pool_size,
        )
        dense_lookup = {hit.chunk_id: hit for hit in dense_hits}
        sparse_lookup = {hit.chunk_id: hit for hit in sparse_hits}

        result = HybridRetrievalResult(
            processed_query=processed_query,
            dense_hits=_materialize_hits(
                dense_hits,
                chunk_lookup,
                source="dense",
                dense_lookup=dense_lookup,
                sparse_lookup=sparse_lookup,
            ),
            sparse_hits=_materialize_hits(
                sparse_hits,
                chunk_lookup,
                source="sparse",
                dense_lookup=dense_lookup,
                sparse_lookup=sparse_lookup,
            ),
            fused_hits=_materialize_hits(
                fused_hits[: self._settings.retrieval.fused_top_k],
                chunk_lookup,
                source="rrf",
                dense_lookup=dense_lookup,
                sparse_lookup=sparse_lookup,
            ),
            dense_raw_hits=dense_hits,
            sparse_raw_hits=sparse_hits,
        )
        await self._retrieval_cache.set(retrieval_key, result.model_dump(mode="json"))
        logger.info(
            "async_hybrid_retrieval_completed",
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            fused_hits=len(result.fused_hits),
        )
        return result

    async def _get_query_vector(self, rewritten_query: str) -> list[float]:
        embedding_key = quote_plus(rewritten_query)
        cached_payload = await self._embedding_cache.get(embedding_key)
        if cached_payload is not None:
            return [float(value) for value in cached_payload]

        vector = await asyncio.to_thread(self._embedder.embed_texts, [rewritten_query])
        query_vector = vector[0]
        await self._embedding_cache.set(embedding_key, query_vector)
        return query_vector

    def _build_response(self, structured, stage_latencies: dict[str, float], total_latency_ms: float) -> QueryResponse:
        debug_documents = [
            RetrievedDocument(
                chunk_id=block.chunk_id,
                text=block.compressed_text,
                source=Path(block.source_path or block.title).name or block.title,
                score=block.rerank_score,
                title=block.title,
                section_title=block.section_title,
                source_path=block.source_path,
            )
            for block in structured.grounded_answer.context.selected_blocks
        ]

        return QueryResponse(
            answer=structured.answer,
            citations=[
                QueryCitation(
                    id=citation.id,
                    text=citation.text,
                    source=citation.source,
                    score=citation.score,
                    title=citation.title,
                    section_title=citation.section_title,
                    source_path=citation.source_path,
                    reference_text=citation.reference_text,
                )
                for citation in structured.citations
            ],
            latency_ms=total_latency_ms,
            confidence_score=structured.confidence.score,
            confidence_label=structured.confidence.label,
            references_markdown=structured.references_markdown,
            stage_latencies_ms=stage_latencies,
            debug_documents=debug_documents,
            cache_hit=False,
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
