"""Central configuration. Every value can be overridden with an environment variable
(or a `.env` file), e.g. EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".docx")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_document_prefix: str = ""
    embedding_batch_size: int = 32
    embedding_device: str | None = None
    preload_model: bool = True

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    tokens_per_word: float = 1.3  # rough English average, avoids needing a tokenizer

    # Search
    default_top_k: int = 5
    max_top_k: int = 50
    search_min_score: float = 0.2

    # Storage / uploads
    data_dir: Path = BASE_DIR / "data"
    max_upload_mb: int = 25

    # Misc
    log_level: str = "INFO"
    cors_origins: str = ""

    @field_validator("embedding_device", mode="before")
    @classmethod
    def _empty_device_is_none(cls, v):
        return v or None

    @model_validator(mode="after")
    def _check_chunking(self):
        if self.chunk_size_tokens < 20:
            raise ValueError("CHUNK_SIZE_TOKENS must be at least 20")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be >= 0 and < CHUNK_SIZE_TOKENS")
        return self

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for d in (self.documents_dir, self.processed_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
