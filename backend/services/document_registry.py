"""Persistent list of uploaded documents (status, page/chunk counts, hash)."""
from __future__ import annotations

import threading
from pathlib import Path

from backend.models.schemas import DocumentRecord
from backend.utils.helpers import atomic_write_json, read_json


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._docs: dict[str, DocumentRecord] = {}
        for raw in read_json(self.path, []):
            rec = DocumentRecord.model_validate(raw)
            self._docs[rec.id] = rec

    def _persist(self) -> None:
        atomic_write_json(self.path, [d.model_dump() for d in self._docs.values()])

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._docs.get(doc_id)

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with self._lock:
            return next((d for d in self._docs.values() if d.file_hash == file_hash), None)

    def add(self, rec: DocumentRecord) -> None:
        with self._lock:
            self._docs[rec.id] = rec
            self._persist()

    def update(self, doc_id: str, **fields) -> DocumentRecord:
        with self._lock:
            rec = self._docs[doc_id].model_copy(update=fields)
            self._docs[doc_id] = rec
            self._persist()
            return rec

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            self._persist()

    def mark_all_unindexed(self, reason: str) -> None:
        """Used when the vector index had to be discarded: documents need re-indexing."""
        with self._lock:
            for d in self._docs.values():
                if d.status in ("indexed", "processing"):
                    self._docs[d.id] = d.model_copy(
                        update={"status": "uploaded", "chunks": 0, "error": reason})
            self._persist()
