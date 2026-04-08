from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvaluationTarget = Literal["chunk", "document"]
RetrievalSystem = Literal["bm25", "dense", "hybrid"]


class EvaluationSample(BaseModel):
    query_id: str
    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    target: EvaluationTarget = "document"
    expected_answer: str = ""
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QueryEvaluationResult(BaseModel):
    query_id: str
    query: str
    system: RetrievalSystem
    target: EvaluationTarget
    relevant_ids: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    matched_ids: list[str] = Field(default_factory=list)
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    faithfulness: float | None = None
    hallucination_rate: float | None = None
    answer: str = ""
    citations: list[str] = Field(default_factory=list)


class SystemBenchmark(BaseModel):
    system: RetrievalSystem
    query_count: int
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float = 0.0
    faithfulness: float | None = None
    hallucination_rate: float | None = None
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    improvements_vs_baseline: dict[str, float] = Field(default_factory=dict)


class ExperimentArtifacts(BaseModel):
    run_dir: str
    aggregate_json: str
    summary_csv: str
    per_query_csv: str
    benchmark_table_md: str
    benchmark_plot_svg: str | None = None


class ExperimentRunResult(BaseModel):
    run_id: str
    dataset_path: str
    manifest_path: str
    baseline_system: RetrievalSystem
    systems: list[RetrievalSystem] = Field(default_factory=list)
    k_values: list[int] = Field(default_factory=list)
    primary_k: int = 10
    query_results: list[QueryEvaluationResult] = Field(default_factory=list)
    benchmarks: list[SystemBenchmark] = Field(default_factory=list)
    benchmark_table: str = ""
    artifacts: ExperimentArtifacts | None = None
