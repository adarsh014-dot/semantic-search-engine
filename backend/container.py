"""Builds and wires all services. Tests (and future projects) can inject their own embedder."""
from __future__ import annotations

from dataclasses import dataclass

from backend.config import Settings
from backend.services.chunker import Chunker
from backend.services.document_processor import DocumentProcessor
from backend.services.document_registry import DocumentRegistry
from backend.services.embedding_service import BaseEmbedder, create_embedder
from backend.services.indexer import IndexingService
from backend.services.search_service import SearchService
from backend.services.vector_store import FaissVectorStore


@dataclass
class Services:
    settings: Settings
    embedder: BaseEmbedder
    store: FaissVectorStore
    registry: DocumentRegistry
    indexer: IndexingService
    search: SearchService


def build_services(settings: Settings, embedder: BaseEmbedder | None = None) -> Services:
    settings.ensure_dirs()
    embedder = embedder or create_embedder(settings)
    store = FaissVectorStore(settings.index_dir)  # loads the persisted index if present
    registry = DocumentRegistry(settings.processed_dir / "documents.json")
    # Reconcile the document list with what is really in the index (crash / manual deletion).
    for doc in registry.list():
        if doc.status == "processing":
            registry.update(doc.id, status="failed", error="Processing was interrupted. Re-index it.")
        elif doc.status == "indexed" and not store.get_records(doc.id):
            registry.update(doc.id, status="uploaded", chunks=0,
                            error="The search index no longer contains this document. Re-index it.")
    chunker = Chunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens,
                      settings.tokens_per_word)
    indexer = IndexingService(settings, registry, DocumentProcessor(), chunker, embedder, store)
    return Services(settings, embedder, store, registry, indexer,
                    SearchService(settings, embedder, store))
