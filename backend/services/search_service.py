"""Query -> embedding -> FAISS similarity search -> metadata -> results."""
from __future__ import annotations

import time

from backend.config import Settings
from backend.models.schemas import SearchResponse, SearchResult
from backend.services.embedding_service import BaseEmbedder
from backend.services.vector_store import VectorStore
from backend.utils.errors import AppError


class SearchService:
    def __init__(self, settings: Settings, embedder: BaseEmbedder, store: VectorStore):
        self.settings, self.embedder, self.store = settings, embedder, store

    def search(self, query: str, top_k: int) -> SearchResponse:
        if self.store.count == 0:
            raise AppError(409, "index_empty",
                           "No indexed documents are available. Process a document first.")
        if self.store.model_name != self.embedder.model_name:
            raise AppError(409, "index_model_mismatch",
                           f"The index was built with '{self.store.model_name}' but the configured "
                           f"model is '{self.embedder.model_name}'. Run a full re-index.")
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        hits = self.store.search(qvec, top_k)
        t2 = time.perf_counter()

        results = []
        for score, rec in hits:
            if score < self.settings.search_min_score:
                continue  # hits come sorted, so this just trims the tail of weak matches
            results.append(SearchResult(
                chunk_id=rec["chunk_id"], document_id=rec["document_id"], source=rec["source"],
                page=rec.get("page"), chunk_index=rec["chunk_index"], text=rec["text"],
                score=round(score, 4), metadata=rec.get("extra", {})))
        return SearchResponse(
            query=query, top_k=top_k, results=results, count=len(results),
            model=self.embedder.model_name, took_ms=round((t2 - t0) * 1000, 2),
            embed_ms=round((t1 - t0) * 1000, 2), search_ms=round((t2 - t1) * 1000, 2))
