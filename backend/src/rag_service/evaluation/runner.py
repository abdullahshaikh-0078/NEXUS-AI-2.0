from __future__ import annotations

from pathlib import Path
from time import perf_counter
from uuid import uuid4

from rag_service.context.pipeline import build_context
from rag_service.core.config import Settings
from rag_service.evaluation.dataset import load_evaluation_dataset
from rag_service.evaluation.metrics import average, faithfulness_score, hallucination_rate, ndcg_at_k, recall_at_k, reciprocal_rank
from rag_service.evaluation.models import ExperimentRunResult, QueryEvaluationResult, RetrievalSystem, SystemBenchmark
from rag_service.evaluation.reporting import build_benchmark_table, create_run_dir, write_experiment_artifacts
from rag_service.generation.pipeline import generate_grounded_answer
from rag_service.indexing.dense import search_dense_index
from rag_service.indexing.embedders import create_embedder
from rag_service.indexing.models import SearchHit
from rag_service.indexing.sparse import search_sparse_index
from rag_service.postprocessing.pipeline import postprocess_grounded_answer
from rag_service.query.pipeline import process_query
from rag_service.reranking.pipeline import rerank_candidates
from rag_service.retrieval.fusion import reciprocal_rank_fusion
from rag_service.retrieval.loaders import load_retrieval_artifacts
from rag_service.retrieval.models import HybridRetrievalResult, RetrievedChunk


def run_experiment_suite(
    settings: Settings,
    *,
    dataset_path: Path | None = None,
    manifest_path: Path | None = None,
    systems: list[RetrievalSystem] | None = None,
) -> ExperimentRunResult:
    active_dataset_path = Path(dataset_path or settings.evaluation.dataset_path)
    active_manifest_path = Path(manifest_path or settings.evaluation.manifest_path or settings.retrieval.manifest_path)
    active_systems = systems or settings.evaluation.systems
    dataset = load_evaluation_dataset(active_dataset_path)
    manifest, chunk_lookup = load_retrieval_artifacts(active_manifest_path)
    embedder = create_embedder(settings)

    query_results: list[QueryEvaluationResult] = []
    for sample in dataset:
        processed_query = process_query(sample.query, settings=settings)
        query_vector = embedder.embed_texts([processed_query.rewritten_query])[0]
        for system in active_systems:
            query_results.append(
                _evaluate_sample(
                    sample=sample,
                    system=system,
                    settings=settings,
                    manifest=manifest,
                    chunk_lookup=chunk_lookup,
                    processed_query=processed_query,
                    query_vector=query_vector,
                )
            )

    benchmarks = _aggregate_results(
        query_results,
        systems=active_systems,
        k_values=settings.evaluation.k_values,
        primary_k=settings.evaluation.primary_k,
        baseline_system=settings.evaluation.baseline_system,
    )
    benchmark_table = build_benchmark_table(benchmarks, settings.evaluation.primary_k)

    run_result = ExperimentRunResult(
        run_id=uuid4().hex[:12],
        dataset_path=str(active_dataset_path),
        manifest_path=str(active_manifest_path),
        baseline_system=settings.evaluation.baseline_system,
        systems=active_systems,
        k_values=settings.evaluation.k_values,
        primary_k=settings.evaluation.primary_k,
        query_results=query_results,
        benchmarks=benchmarks,
        benchmark_table=benchmark_table,
    )

    output_base = Path(settings.evaluation.output_dir)
    artifacts = write_experiment_artifacts(
        run_result,
        create_run_dir(output_base),
        generate_plot=settings.evaluation.generate_plots,
    )
    run_result.artifacts = artifacts
    return run_result


def _evaluate_sample(
    *,
    sample,
    system: RetrievalSystem,
    settings: Settings,
    manifest,
    chunk_lookup,
    processed_query,
    query_vector: list[float],
) -> QueryEvaluationResult:
    total_start = perf_counter()

    retrieval_start = perf_counter()
    retrieval_result = _retrieve_for_system(
        system=system,
        settings=settings,
        manifest=manifest,
        chunk_lookup=chunk_lookup,
        processed_query=processed_query,
        query_vector=query_vector,
    )
    retrieval_latency_ms = round((perf_counter() - retrieval_start) * 1000, 2)

    rerank_start = perf_counter()
    reranking = rerank_candidates(sample.query, settings=settings, retrieval=retrieval_result)
    rerank_latency_ms = round((perf_counter() - rerank_start) * 1000, 2)

    context = build_context(sample.query, settings=settings, reranking=reranking)

    llm_start = perf_counter()
    grounded_answer = generate_grounded_answer(sample.query, settings=settings, context=context)
    structured = postprocess_grounded_answer(sample.query, settings=settings, grounded_answer=grounded_answer)
    llm_latency_ms = round((perf_counter() - llm_start) * 1000, 2)

    total_latency_ms = round((perf_counter() - total_start) * 1000, 2)

    ranked_ids = _ranked_ids(retrieval_result, system=system, target=sample.target)
    relevant_ids = set(sample.relevant_ids)
    matched_ids = [candidate for candidate in ranked_ids if candidate in relevant_ids]
    recall_scores = {str(k): recall_at_k(relevant_ids, ranked_ids, k) for k in settings.evaluation.k_values}
    ndcg_scores = {str(k): ndcg_at_k(relevant_ids, ranked_ids, k) for k in settings.evaluation.k_values}
    mrr_score = reciprocal_rank(relevant_ids, ranked_ids)
    evidence_texts = [citation.text for citation in structured.citations]
    faithfulness = faithfulness_score(structured.answer, evidence_texts, overlap_threshold=settings.evaluation.faithfulness_overlap_threshold)
    hallucination = hallucination_rate(structured.answer, evidence_texts, overlap_threshold=settings.evaluation.faithfulness_overlap_threshold)

    return QueryEvaluationResult(
        query_id=sample.query_id,
        query=sample.query,
        system=system,
        target=sample.target,
        relevant_ids=sample.relevant_ids,
        retrieved_ids=ranked_ids,
        matched_ids=matched_ids,
        recall_at_k=recall_scores,
        ndcg_at_k=ndcg_scores,
        mrr=mrr_score,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
        faithfulness=faithfulness,
        hallucination_rate=hallucination,
        answer=structured.answer,
        citations=[citation.source for citation in structured.citations],
    )


def _retrieve_for_system(*, system: RetrievalSystem, settings: Settings, manifest, chunk_lookup, processed_query, query_vector: list[float]) -> HybridRetrievalResult:
    dense_hits: list[SearchHit] = []
    sparse_hits: list[SearchHit] = []
    if system in {"dense", "hybrid"}:
        dense_hits = search_dense_index(manifest.dense, query_vector=query_vector, top_k=settings.retrieval.dense_top_k)
    if system in {"bm25", "hybrid"}:
        sparse_hits = search_sparse_index(manifest.sparse, query=processed_query.rewritten_query, top_k=settings.retrieval.sparse_top_k)

    if system == "hybrid":
        fused_hits = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            rrf_k=settings.retrieval.rrf_k,
            top_k=settings.retrieval.candidate_pool_size,
        )[: settings.retrieval.fused_top_k]
        source = "rrf"
    elif system == "dense":
        fused_hits = dense_hits[: settings.retrieval.fused_top_k]
        source = "dense"
    else:
        fused_hits = sparse_hits[: settings.retrieval.fused_top_k]
        source = "sparse"

    dense_lookup = {hit.chunk_id: hit for hit in dense_hits}
    sparse_lookup = {hit.chunk_id: hit for hit in sparse_hits}
    return HybridRetrievalResult(
        processed_query=processed_query,
        dense_hits=_materialize_hits(dense_hits, chunk_lookup, source="dense", dense_lookup=dense_lookup, sparse_lookup=sparse_lookup),
        sparse_hits=_materialize_hits(sparse_hits, chunk_lookup, source="sparse", dense_lookup=dense_lookup, sparse_lookup=sparse_lookup),
        fused_hits=_materialize_hits(fused_hits, chunk_lookup, source=source, dense_lookup=dense_lookup, sparse_lookup=sparse_lookup),
        dense_raw_hits=dense_hits,
        sparse_raw_hits=sparse_hits,
    )


def _aggregate_results(
    results: list[QueryEvaluationResult],
    *,
    systems: list[RetrievalSystem],
    k_values: list[int],
    primary_k: int,
    baseline_system: RetrievalSystem,
) -> list[SystemBenchmark]:
    benchmarks: list[SystemBenchmark] = []
    by_system = {system: [result for result in results if result.system == system] for system in systems}

    for system in systems:
        system_results = by_system[system]
        recall_scores = {
            str(k): round(sum(result.recall_at_k[str(k)] for result in system_results) / max(1, len(system_results)), 4)
            for k in k_values
        }
        ndcg_scores = {
            str(k): round(sum(result.ndcg_at_k[str(k)] for result in system_results) / max(1, len(system_results)), 4)
            for k in k_values
        }
        benchmark = SystemBenchmark(
            system=system,
            query_count=len(system_results),
            recall_at_k=recall_scores,
            ndcg_at_k=ndcg_scores,
            mrr=round(sum(result.mrr for result in system_results) / max(1, len(system_results)), 4),
            faithfulness=average([result.faithfulness for result in system_results]),
            hallucination_rate=average([result.hallucination_rate for result in system_results]),
            retrieval_latency_ms=round(sum(result.retrieval_latency_ms for result in system_results) / max(1, len(system_results)), 2),
            rerank_latency_ms=round(sum(result.rerank_latency_ms for result in system_results) / max(1, len(system_results)), 2),
            llm_latency_ms=round(sum(result.llm_latency_ms for result in system_results) / max(1, len(system_results)), 2),
            total_latency_ms=round(sum(result.total_latency_ms for result in system_results) / max(1, len(system_results)), 2),
        )
        benchmarks.append(benchmark)

    baseline = next((item for item in benchmarks if item.system == baseline_system), None)
    if baseline is not None:
        for benchmark in benchmarks:
            benchmark.improvements_vs_baseline = {
                f"recall@{primary_k}": _percent_change(benchmark.recall_at_k.get(str(primary_k), 0.0), baseline.recall_at_k.get(str(primary_k), 0.0)),
                "mrr": _percent_change(benchmark.mrr, baseline.mrr),
                f"ndcg@{primary_k}": _percent_change(benchmark.ndcg_at_k.get(str(primary_k), 0.0), baseline.ndcg_at_k.get(str(primary_k), 0.0)),
                "latency_reduction": _percent_change(baseline.total_latency_ms, benchmark.total_latency_ms),
            }
    return benchmarks


def _ranked_ids(retrieval_result: HybridRetrievalResult, *, system: RetrievalSystem, target: str) -> list[str]:
    if system == "dense":
        hits = retrieval_result.dense_hits
    elif system == "bm25":
        hits = retrieval_result.sparse_hits
    else:
        hits = retrieval_result.fused_hits

    ranked_ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        candidate_id = hit.chunk_id if target == "chunk" else hit.metadata.document_id
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        ranked_ids.append(candidate_id)
    return ranked_ids


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


def _percent_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return round(((current - baseline) / baseline) * 100.0, 2)
