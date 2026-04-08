from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "base.yaml"


class AppConfig(BaseModel):
    name: str = "nexus-rag-platform"
    version: str = "0.1.0"
    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000


class ApiConfig(BaseModel):
    prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = 30.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True


class OpenAIConfig(BaseModel):
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    timeout_seconds: float = 10.0


class IngestionConfig(BaseModel):
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
    default_strategy: str = "structure_aware"
    chunk_size: int = 900
    chunk_overlap: int = 150
    semantic_similarity_threshold: float = 0.18


class IndexingConfig(BaseModel):
    input_chunk_file: str = "data/processed/chunks.jsonl"
    output_dir: str = "data/indexes"
    embedding_provider: str = "sentence_transformers"
    embedding_fallback_provider: str = "hash"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 16
    embedding_version: str = "v1"
    dense_backend: str = "faiss"
    dense_fallback_backend: str = "native"
    dense_index_type: str = "hnsw"
    dense_hnsw_m: int = 32
    dense_hnsw_ef_construction: int = 80
    sparse_backend: str = "whoosh"
    sparse_fallback_backend: str = "native"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class QueryConfig(BaseModel):
    enable_expansion: bool = True
    expansion_terms_per_token: int = 3
    enable_rule_rewrite: bool = True
    enable_llm_fallback: bool = True
    llm_fallback_min_tokens: int = 4
    max_expanded_terms: int = 12
    preserve_original_case: bool = False


class RetrievalConfig(BaseModel):
    manifest_path: str = "data/indexes/v1/manifest.json"
    dense_top_k: int = 8
    sparse_top_k: int = 8
    fused_top_k: int = 10
    candidate_pool_size: int = 12
    rrf_k: int = 60


class RerankingConfig(BaseModel):
    provider: str = "cross_encoder"
    fallback_provider: str = "heuristic"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 8
    top_k: int = 5
    candidate_limit: int = 10
    normalize_scores: bool = True


class ContextConfig(BaseModel):
    max_context_tokens: int = 700
    max_chunks: int = 4
    min_rerank_score: float = 0.1
    per_chunk_token_limit: int = 180
    deduplicate_by_document: bool = True
    deduplicate_by_text: bool = True
    compression_strategy: str = "extractive"
    include_metadata_headers: bool = True


class GenerationConfig(BaseModel):
    provider: str = "openai"
    fallback_provider: str = "heuristic"
    temperature: float = 0.0
    max_output_tokens: int = 400
    max_citations: int = 4


class PostProcessingConfig(BaseModel):
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.55
    target_citation_count: int = 3
    target_context_blocks: int = 3
    citation_score_weight: float = 0.55
    citation_coverage_weight: float = 0.25
    context_coverage_weight: float = 0.20
    fallback_penalty: float = 0.12


class CacheConfig(BaseModel):
    provider: str = "memory"
    redis_url: str = "redis://redis:6379/0"
    default_ttl_seconds: int = 300
    query_ttl_seconds: int = 180
    retrieval_ttl_seconds: int = 120
    embedding_ttl_seconds: int = 600


class MonitoringConfig(BaseModel):
    enabled: bool = True


class LatencyConfig(BaseModel):
    enable_adaptive_retrieval: bool = True
    short_query_token_threshold: int = 4
    short_query_dense_top_k: int = 4
    short_query_sparse_top_k: int = 4
    short_query_candidate_pool_size: int = 6
    short_query_fused_top_k: int = 6
    stream_chunk_size: int = 80
    stream_media_type: str = "application/x-ndjson"


class ScalingConfig(BaseModel):
    max_concurrent_queries: int = 8
    acquire_timeout_seconds: float = 1.5
    worker_processes: int = 2
    replica_hint: int = 2
    enable_backpressure: bool = True


class ResilienceConfig(BaseModel):
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.2
    dense_failure_threshold: int = 3
    sparse_failure_threshold: int = 3
    generation_failure_threshold: int = 3
    circuit_reset_seconds: float = 30.0
    allow_partial_retrieval: bool = True


class CostConfig(BaseModel):
    skip_llm_for_high_confidence: bool = True
    confidence_rerank_threshold: float = 0.82
    minimum_context_blocks_for_skip: int = 2
    max_generation_context_blocks: int = 3
    prefer_cached_answers: bool = True


class SecurityConfig(BaseModel):
    require_api_key: bool = False
    api_keys: list[str] = Field(default_factory=list)
    prompt_injection_action: str = "sanitize"
    max_query_characters: int = 2000
    auth_header_name: str = "x-api-key"


class EvaluationConfig(BaseModel):
    dataset_path: str = "data/evaluation/generated_eval_dataset.jsonl"
    output_dir: str = "artifacts/evaluation"
    manifest_path: str = ""
    systems: list[str] = Field(default_factory=lambda: ["bm25", "dense", "hybrid"])
    baseline_system: str = "bm25"
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 10])
    primary_k: int = 10
    faithfulness_overlap_threshold: float = 0.35
    generate_plots: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    postprocessing: PostProcessingConfig = Field(default_factory=PostProcessingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Settings:
        config_path = path or DEFAULT_CONFIG_PATH
        payload: dict[str, Any] = {}
        if config_path.exists():
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(**payload)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    yaml_settings = Settings.from_yaml()
    env_settings = Settings().model_dump(exclude_unset=True)
    merged = _deep_merge(yaml_settings.model_dump(), env_settings)
    return Settings(**merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
