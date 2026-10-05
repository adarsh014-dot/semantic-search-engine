"""Pydantic models for API requests/responses and internal records."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DocStatus = Literal["uploaded", "processing", "indexed", "failed"]


# ---------- documents ----------
class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    file_hash: str
    pages: int | None = None  # None for TXT/DOCX (no fixed pages)
    chunks: int = 0
    status: DocStatus = "uploaded"
    error: str | None = None
    created_at: str
    indexed_at: str | None = None


class DocumentRecord(DocumentInfo):
    """Internal record: adds the on-disk name, which is never sent to clients."""
    stored_name: str


class UploadItemResult(BaseModel):
    filename: str
    status: Literal["uploaded", "duplicate", "rejected"]
    message: str
    document: DocumentInfo | None = None


class UploadResponse(BaseModel):
    results: list[UploadItemResult]


class IndexRequest(BaseModel):
    document_ids: list[str] | None = Field(
        default=None, description="Documents to process. Omit to process every pending/failed one."
    )


class JobResponse(BaseModel):
    accepted: list[str]
    message: str


class ChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    page: int | None
    text: str


class ChunkContext(BaseModel):
    document: DocumentInfo
    chunk: ChunkOut
    previous: ChunkOut | None = None
    next: ChunkOut | None = None


# ---------- search ----------
class SearchRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Query must not be empty.")
        return v


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str
    score: float = Field(description="Cosine similarity in [-1, 1]; higher is more similar.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]
    count: int
    model: str
    metric: str = "cosine_similarity"
    took_ms: float
    embed_ms: float
    search_ms: float


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    model_loaded: bool
    dimension: int | None
    indexed_chunks: int
    documents: int
    metric: str
    max_upload_mb: int
    allowed_extensions: list[str]
    default_top_k: int
