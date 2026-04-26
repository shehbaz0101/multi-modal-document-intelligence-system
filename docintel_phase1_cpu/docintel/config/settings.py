"""Typed, environment-driven configuration."""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")
    backend: str = "local"
    root: str = "./data"
    page_dpi: int = 150


class ParserSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PARSER_", extra="ignore")
    primary: str = "cpu"
    fallback: str | None = None
    low_confidence_threshold: float = 0.6
    enable_table_structure: bool = True
    enable_ocr: bool = True


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")
    text_model: str = "all-MiniLM-L6-v2"
    text_dim: int = 384
    text_batch_size: int = 64
    visual_model: str = "vidore/colqwen2-v1.0"
    visual_enabled: bool = False
    caption_model: str = "gemini-2.5-flash-lite"
    caption_max_tokens: int = 200


class IndexSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INDEX_", extra="ignore")
    vector_backend: str = "qdrant"
    vector_url: str = "http://localhost:6333"
    vector_api_key: str | None = None
    text_collection: str = "blocks_text"
    visual_collection: str = "pages_visual"
    sparse_backend: str = "opensearch"
    sparse_url: str = "http://localhost:9200"
    sparse_index: str = "blocks_sparse"
    metadata_db_url: str = "postgresql://docintel:docintel@localhost:5432/docintel"


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")
    rrf_k: int = 60
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20


class VisualRerankSettings(BaseSettings):
    """Phase 2 — Gemini visual reranker."""
    model_config = SettingsConfigDict(env_prefix="VISUAL_RERANK_", extra="ignore")
    enabled: bool = False
    model: str = "gemini-2.5-flash-lite"
    max_pages: int = 10
    score_weight: float = 0.7  # 0 = pure text ranking, 1 = pure visual


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENERATION_", extra="ignore")
    model: str = "gemini-2.5-flash-lite"
    max_tokens: int = 1500
    temperature: float = 0.1
    api_key: str | None = None
    consistency_runs: int = 1


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBS_", extra="ignore")
    tracing_backend: str = "none"
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    env: str = Field(default="dev")
    log_level: str = "INFO"
    storage: StorageSettings = Field(default_factory=StorageSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    visual_rerank: VisualRerankSettings = Field(default_factory=VisualRerankSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()