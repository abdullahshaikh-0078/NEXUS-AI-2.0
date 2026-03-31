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


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True


class OpenAIConfig(BaseModel):
    api_key: str = ""


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)

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
