from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from urllib.parse import quote_plus

from rag_service.api.schemas import QueryCitation, QueryResponse, RetrievedDocument
from rag_service.context.models import ContextPackage
from rag_service.context.pipeline import build_context
from rag_service.core.cache import CacheBackend, CacheNamespace
from rag_service.core.config import Settings
from rag_service.core.costing import choose_generation_provider, optimize_context_for_cost
from rag_service.core.logging import get_logger
from rag_service.core.metrics import MetricsRegistry
from rag_service.core.optimization import build_retrieval_plan, iter_answer_chunks
from rag_service.core.resilience import CircuitBreaker, retry_async
from rag_service.core.scaling import QueryAdmissionController
from rag_service.core.security import sanitize_query
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
        admission_controller: QueryAdmissionController,
    ) -> None:
        self._settings = settings
        self._metrics = metrics
        self._admission_controller = admission_controller
        self._query_cache = CacheNamespace(cache_backend, "query", settings.cache.query_ttl_seconds)
        self._retrieval_cache = CacheNamespace(cache_backend, "retrieval", settings.cache.retrieval_ttl_seconds)
        self._embedding_cache = CacheNamespace(cache_backend, "embedding", settings.cache.embedding_ttl_seconds)
        self._embedder: TextEmbedder = create_embedder(settings)
        self._dense_breaker = CircuitBreaker(
            failure_threshold=settings.resilience.dense_failure_threshold,
            reset_timeout_seconds=settings.resilience.circuit_reset_seconds,
        )
        self._sparse_breaker = CircuitBreaker(
            failure_threshold=settings.resilience.sparse_failure_threshold,
            reset_timeout_seconds=settings.resilience.circuit_reset_seconds,
        )
        self._generation_breaker = CircuitBreaker(
            failure_threshold=settings.resilience.generation_failure_threshold,
            reset_timeout_seconds=settings.resilience.circuit_reset_seconds,
        )

    async def answer(self, query: str) -> QueryResponse:
        security_assessment = sanitize_query(query, self._settings)
        sanitized_query = security_assessment.sanitized_query
        warnings = list(security_assessment.warnings)

        start_time = perf_counter()
        cache_key = quote_plus(sanitized_query.strip().lower())
        cached_payload = await self._query_cache.get(cache_key)
        if cached_payload is not None and self._settings.cost.prefer_cached_answers:
            response = QueryResponse.model_validate(cached_payload)
            response.cache_hit = True
            response.warnings = warnings + response.warnings
            self._metrics.record_success(
                total_latency_ms=response.latency_ms,
                confidence_score=response.confidence_score,
                citation_count=len(response.citations),
                cache_hit=True,
            )
            self._metrics.record_event("query_cache_hit")
            return response

        async with self._admission_controller.acquire():
            stage_latencies: dict[str, float] = {}
            try:
                processed_query = await self._run_stage(
                    "query_processing",
                    stage_latencies,
                    asyncio.to_thread(process_query, sanitized_query, self._settings),
                )
                retrieval = await self._run_stage(
                    "retrieval",
                    stage_latencies,
                    self._hybrid_retrieve_async(processed_query, warnings),
                )
                reranking = await self._run_stage(
                    "reranking",
                    stage_latencies,
                    asyncio.to_thread(
                        rerank_candidates,
                        sanitized_query,
                        self._reranking_settings(processed_query),
                        retrieval,
                    ),
                )
                context = await self._run_stage(
                    "context",
                    stage_latencies,
                    asyncio.to_thread(build_context, sanitized_query, self._settings, reranking),
                )
                optimized_context = self._apply_cost_controls(context, warnings)
                grounded_answer = await self._run_stage(
                    "generation",
                    stage_latencies,
                    self._generate_answer_async(sanitized_query, optimized_context, warnings),
                )
                structured = await self._run_stage(
                    "postprocessing",
                    stage_latencies,
                    asyncio.to_thread(postprocess_grounded_answer, sanitized_query, self._settings, grounded_answer),
                )

                total_latency_ms = round((perf_counter() - start_time) * 1000, 2)
                response = self._build_response(structured, stage_latencies, total_latency_ms, warnings)
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

    async def stream_answer(self, query: str):
        response = await self.answer(query)
        yield {"type": "start", "cache_hit": response.cache_hit}
        for chunk in iter_answer_chunks(response.answer, self._settings.latency.stream_chunk_size):
            yield {"type": "token", "chunk": chunk}
        yield {"type": "final", "response": response.model_dump(mode="json")}

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "circuits": {
                "dense": self._dense_breaker.snapshot(),
                "sparse": self._sparse_breaker.snapshot(),
                "generation": self._generation_breaker.snapshot(),
            }
        }

    async def _run_stage(self, stage_name: str, stage_latencies: dict[str, float], awaitable):
        start_time = perf_counter()
        result = await awaitable
        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        stage_latencies[stage_name] = duration_ms
        self._metrics.record_stage(stage_name, duration_ms)
        return result

    async def _hybrid_retrieve_async(
        self,
        processed_query: ProcessedQuery,
        warnings: list[str],
    ) -> HybridRetrievalResult:
        manifest_path = Path(self._settings.retrieval.manifest_path)
        plan = build_retrieval_plan(processed_query, self._settings)
        retrieval_key = quote_plus(
            f"{processed_query.rewritten_query}|{manifest_path}|{plan.dense_top_k}|{plan.sparse_top_k}|{plan.candidate_pool_size}"
        )
        cached_payload = await self._retrieval_cache.get(retrieval_key)
        if cached_payload is not None:
            self._metrics.record_event("retrieval_cache_hit")
            return HybridRetrievalResult.model_validate(cached_payload)

        manifest, chunk_lookup = await asyncio.to_thread(load_retrieval_artifacts, manifest_path)
        query_vector = await self._get_query_vector(processed_query.rewritten_query)

        dense_task = self._run_dense_search(manifest.dense, query_vector, plan.dense_top_k)
        sparse_task = self._run_sparse_search(manifest.sparse, processed_query.rewritten_query, plan.sparse_top_k)
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)

        dense_error = dense_hits if isinstance(dense_hits, Exception) else None
        sparse_error = sparse_hits if isinstance(sparse_hits, Exception) else None
        dense_results = [] if dense_error else dense_hits
        sparse_results = [] if sparse_error else sparse_hits

        if dense_error:
            warnings.append("Dense retrieval was unavailable; sparse retrieval results were used.")
            self._metrics.record_event("dense_retrieval_fallback")
        if sparse_error:
            warnings.append("Sparse retrieval was unavailable; dense retrieval results were used.")
            self._metrics.record_event("sparse_retrieval_fallback")
        if dense_error and sparse_error:
            raise RuntimeError("All retrieval backends failed.")
        if (dense_error or sparse_error) and not self._settings.resilience.allow_partial_retrieval:
            raise RuntimeError("Partial retrieval is disabled.")

        fused_hits = reciprocal_rank_fusion(
            dense_hits=dense_results,
            sparse_hits=sparse_results,
            rrf_k=self._settings.retrieval.rrf_k,
            top_k=plan.candidate_pool_size,
        )
        dense_lookup = {hit.chunk_id: hit for hit in dense_results}
        sparse_lookup = {hit.chunk_id: hit for hit in sparse_results}

        result = HybridRetrievalResult(
            processed_query=processed_query,
            dense_hits=_materialize_hits(
                dense_results,
                chunk_lookup,
                source="dense",
                dense_lookup=dense_lookup,
                sparse_lookup=sparse_lookup,
            ),
            sparse_hits=_materialize_hits(
                sparse_results,
                chunk_lookup,
                source="sparse",
                dense_lookup=dense_lookup,
                sparse_lookup=sparse_lookup,
            ),
            fused_hits=_materialize_hits(
                fused_hits[: plan.fused_top_k],
                chunk_lookup,
                source="rrf",
                dense_lookup=dense_lookup,
                sparse_lookup=sparse_lookup,
            ),
            dense_raw_hits=dense_results,
            sparse_raw_hits=sparse_results,
        )
        await self._retrieval_cache.set(retrieval_key, result.model_dump(mode="json"))
        logger.info(
            "async_hybrid_retrieval_completed",
            dense_hits=len(dense_results),
            sparse_hits=len(sparse_results),
            fused_hits=len(result.fused_hits),
        )
        return result

    async def _run_dense_search(self, dense_manifest, query_vector: list[float], top_k: int) -> list[SearchHit]:
        async def operation() -> list[SearchHit]:
            return await asyncio.to_thread(
                search_dense_index,
                dense_manifest,
                query_vector=query_vector,
                top_k=top_k,
            )

        return await self._dense_breaker.call(
            lambda: retry_async(
                operation,
                attempts=self._settings.resilience.retry_attempts,
                base_delay_seconds=self._settings.resilience.retry_backoff_seconds,
            )
        )

    async def _run_sparse_search(self, sparse_manifest, query: str, top_k: int) -> list[SearchHit]:
        async def operation() -> list[SearchHit]:
            return await asyncio.to_thread(
                search_sparse_index,
                sparse_manifest,
                query=query,
                top_k=top_k,
            )

        return await self._sparse_breaker.call(
            lambda: retry_async(
                operation,
                attempts=self._settings.resilience.retry_attempts,
                base_delay_seconds=self._settings.resilience.retry_backoff_seconds,
            )
        )

    async def _get_query_vector(self, rewritten_query: str) -> list[float]:
        embedding_key = quote_plus(rewritten_query)
        cached_payload = await self._embedding_cache.get(embedding_key)
        if cached_payload is not None:
            self._metrics.record_event("embedding_cache_hit")
            return [float(value) for value in cached_payload]

        vector = await asyncio.to_thread(self._embedder.embed_texts, [rewritten_query])
        query_vector = vector[0]
        await self._embedding_cache.set(embedding_key, query_vector)
        return query_vector

    async def _generate_answer_async(
        self,
        query: str,
        context: ContextPackage,
        warnings: list[str],
    ):
        decision = choose_generation_provider(query, context, self._settings)
        if decision.provider != self._settings.generation.provider.lower():
            warnings.append("Skipped the external LLM call because the retrieved context already had strong evidence.")
            self._metrics.record_event("llm_call_skipped")
            return await asyncio.to_thread(
                generate_grounded_answer,
                query,
                self._settings,
                context,
                provider_override=decision.provider,
            )

        async def operation():
            return await asyncio.to_thread(
                generate_grounded_answer,
                query,
                self._settings,
                context,
                provider_override=decision.provider,
                allow_fallback=False,
            )

        try:
            return await self._generation_breaker.call(
                lambda: retry_async(
                    operation,
                    attempts=self._settings.resilience.retry_attempts,
                    base_delay_seconds=self._settings.resilience.retry_backoff_seconds,
                )
            )
        except Exception as exc:
            logger.warning("generation_degraded_to_fallback", error=str(exc))
            warnings.append("Primary generation backend was unavailable; a heuristic fallback answer was used.")
            self._metrics.record_event("generation_fallback")
            return await asyncio.to_thread(
                generate_grounded_answer,
                query,
                self._settings,
                context,
                provider_override=self._settings.generation.fallback_provider,
            )

    def _apply_cost_controls(self, context: ContextPackage, warnings: list[str]) -> ContextPackage:
        optimized = optimize_context_for_cost(context, self._settings)
        if optimized is not context:
            warnings.append("Generation context was trimmed to reduce token and model cost.")
            self._metrics.record_event("context_trimmed_for_cost")
        return optimized

    def _reranking_settings(self, processed_query: ProcessedQuery) -> Settings:
        plan = build_retrieval_plan(processed_query, self._settings)
        if plan.rerank_candidate_limit == self._settings.reranking.candidate_limit:
            return self._settings

        tuned = self._settings.model_copy(deep=True)
        tuned.reranking.candidate_limit = plan.rerank_candidate_limit
        return tuned

    def _build_response(self, structured, stage_latencies: dict[str, float], total_latency_ms: float, warnings: list[str]) -> QueryResponse:
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
            warnings=warnings,
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
