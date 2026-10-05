"""Ingestion pipeline: extract -> clean -> chunk -> embed -> store in the vector index."""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict

import numpy as np

from backend.config import Settings
from backend.services.chunker import Chunker
from backend.services.document_processor import DocumentProcessor
from backend.services.document_registry import DocumentRegistry
from backend.services.embedding_service import BaseEmbedder
from backend.services.vector_store import FaissVectorStore
from backend.utils.errors import AppError, DocumentProcessingError, EmbeddingError, VectorStoreError
from backend.utils.helpers import atomic_write_json, utc_now_iso

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(self, settings: Settings, registry: DocumentRegistry, processor: DocumentProcessor,
                 chunker: Chunker, embedder: BaseEmbedder, store: FaissVectorStore):
        self.settings, self.registry, self.processor = settings, registry, processor
        self.chunker, self.embedder, self.store = chunker, embedder, store
        self._job_lock = threading.Lock()  # one indexing job at a time (CPU heavy)

    def processed_path(self, doc_id: str):
        return self.settings.processed_dir / f"{doc_id}.json"

    def ensure_model_consistent(self) -> None:
        """An index only makes sense for the model that produced it."""
        if self.store.count and self.store.model_name != self.embedder.model_name:
            raise AppError(409, "index_model_mismatch",
                           f"The index was built with '{self.store.model_name}' but the configured "
                           f"model is '{self.embedder.model_name}'. Run a full re-index.")

    def run(self, doc_ids: list[str], rebuild_all: bool = False) -> None:
        """Entry point for background tasks. Never raises; failures are stored per document."""
        with self._job_lock:
            if rebuild_all:
                self.store.reset()
            for doc_id in doc_ids:
                self._index_one(doc_id)

    def _index_one(self, doc_id: str) -> None:
        rec = self.registry.get(doc_id)
        if rec is None:
            return
        self.registry.update(doc_id, status="processing", error=None)
        try:
            self.ensure_model_consistent()
            path = self.settings.documents_dir / rec.stored_name
            if not path.exists():
                raise DocumentProcessingError("The stored file is missing. Please upload it again.")
            pages = self.processor.extract(path, rec.filename)
            chunks = self.chunker.chunk_pages(pages, doc_id)
            if not chunks:
                raise DocumentProcessingError("The document contains no usable text.")
            vectors = self.embedder.embed_documents([c.text for c in chunks])  # batched inside
            records = [c.to_dict() for c in chunks]

            self.store.delete_document(doc_id)  # re-indexing replaces old vectors, no duplicates
            self.store.add(np.asarray(vectors), records, self.embedder.model_name)
            self.store.save()
            atomic_write_json(self.processed_path(doc_id), {
                "document_id": doc_id, "source": rec.filename, "chunks": [asdict(c) for c in chunks]})

            numbered = [p.page for p in pages if p.page is not None]
            self.registry.update(doc_id, status="indexed", chunks=len(chunks),
                                 pages=len(numbered) or None, indexed_at=utc_now_iso(), error=None)
            logger.info("Indexed %s: %d chunks", rec.filename, len(chunks))
        except (DocumentProcessingError, EmbeddingError, VectorStoreError, AppError) as exc:
            message = exc.message if isinstance(exc, AppError) else str(exc)
            logger.warning("Indexing failed for %s: %s", rec.filename, message)
            self.registry.update(doc_id, status="failed", error=message, chunks=0)
        except Exception:
            logger.exception("Unexpected error while indexing %s", rec.filename)
            self.registry.update(doc_id, status="failed", chunks=0,
                                 error="Unexpected error while processing this document.")

    def remove(self, doc_id: str) -> None:
        with self._job_lock:
            if self.store.delete_document(doc_id):
                self.store.save()
            self.processed_path(doc_id).unlink(missing_ok=True)
