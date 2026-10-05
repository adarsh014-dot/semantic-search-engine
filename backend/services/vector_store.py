"""Vector storage behind a small interface, with a FAISS implementation.

Similarity vs distance
----------------------
We use `IndexFlatIP` (exact inner-product search) on L2-normalised vectors.
For unit vectors  a . b = |a||b|cos(theta) = cos(theta),  so the inner product IS the
cosine similarity. FAISS returns it as a *score where higher = more similar* (range
[-1, 1]). That is a similarity, not a distance. With an L2 index FAISS would return a
squared distance where LOWER = more similar; we never mix the two.

Persistence
-----------
FAISS stores only vectors. Each vector gets an int64 id (IndexIDMap2), and
`chunks.json` maps that id -> chunk metadata. Both files are written atomically and
checked against each other on load.
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from backend.utils.errors import VectorStoreError
from backend.utils.helpers import atomic_write_json, read_json

logger = logging.getLogger(__name__)

Record = dict[str, Any]


class VectorStore(ABC):
    """Interface the rest of the app depends on. Implement it to swap FAISS for
    Chroma / Qdrant / Milvus / Pinecone / Weaviate."""

    model_name: str | None
    recovered_from_corruption: bool

    @abstractmethod
    def add(self, vectors: np.ndarray, records: list[Record], model_name: str) -> list[int]: ...
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[float, Record]]: ...
    @abstractmethod
    def delete_document(self, document_id: str) -> int: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def save(self) -> None: ...
    @abstractmethod
    def load(self) -> None: ...
    @property
    @abstractmethod
    def count(self) -> int: ...
    @property
    @abstractmethod
    def dimension(self) -> int | None: ...


class FaissVectorStore(VectorStore):
    METRIC = "inner_product (cosine similarity on normalised vectors)"

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "chunks.json"
        self._lock = threading.RLock()
        self._index = None
        self._records: dict[int, Record] = {}
        self._next_id = 0
        self.model_name = None
        self.recovered_from_corruption = False
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    # ---- state ----
    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def dimension(self) -> int | None:
        with self._lock:
            return int(self._index.d) if self._index is not None else None

    # ---- create ----
    def _create_index(self, dim: int):
        import faiss
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def reset(self) -> None:
        with self._lock:
            self._index, self._records, self._next_id, self.model_name = None, {}, 0, None
            self.save()

    # ---- add ----
    def add(self, vectors: np.ndarray, records: list[Record], model_name: str) -> list[int]:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(records):
            raise VectorStoreError("Vectors and metadata records must have the same length.")
        if len(records) == 0:
            return []
        with self._lock:
            if self._index is None or (not self._records and self._index.d != vectors.shape[1]):
                self._create_index(vectors.shape[1])
            elif vectors.shape[1] != self._index.d:
                raise VectorStoreError(
                    f"Embedding dimension {vectors.shape[1]} does not match index dimension "
                    f"{self._index.d}. Re-index all documents after changing the model.")
            ids = np.arange(self._next_id, self._next_id + len(records), dtype=np.int64)
            try:
                self._index.add_with_ids(vectors, ids)
            except Exception as exc:
                logger.exception("FAISS add failed")
                raise VectorStoreError("Failed to add vectors to the index.") from exc
            for i, rec in zip(ids.tolist(), records):
                self._records[i] = rec
            self._next_id += len(records)
            self.model_name = model_name
            return ids.tolist()

    # ---- search ----
    def search(self, query: np.ndarray, k: int) -> list[tuple[float, Record]]:
        with self._lock:
            if self._index is None or not self._records:
                return []
            q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
            if q.shape[1] != self._index.d:
                raise VectorStoreError("Query dimension does not match the index dimension.")
            try:
                scores, ids = self._index.search(q, min(k, len(self._records)))
            except Exception as exc:
                logger.exception("FAISS search failed")
                raise VectorStoreError("The vector index failed to answer the query.") from exc
            out = []
            for score, vid in zip(scores[0].tolist(), ids[0].tolist()):
                rec = self._records.get(vid)
                if vid == -1 or rec is None:  # -1 = FAISS padding; None = missing metadata
                    if vid != -1:
                        logger.warning("Vector id %s has no metadata; skipping", vid)
                    continue
                out.append((float(score), rec))
            return out

    # ---- delete ----
    def delete_document(self, document_id: str) -> int:
        import faiss
        with self._lock:
            ids = [i for i, r in self._records.items() if r.get("document_id") == document_id]
            if not ids:
                return 0
            self._index.remove_ids(np.array(ids, dtype=np.int64))
            for i in ids:
                del self._records[i]
            if not self._records:
                self._index, self.model_name = None, None  # empty index: forget old model/dim
            return len(ids)

    def get_records(self, document_id: str) -> list[Record]:
        with self._lock:
            return sorted((r for r in self._records.values() if r.get("document_id") == document_id),
                          key=lambda r: r.get("chunk_index", 0))

    # ---- persistence ----
    def save(self) -> None:
        import faiss
        with self._lock:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            if self._index is None:
                for p in (self.index_path, self.meta_path):
                    p.unlink(missing_ok=True)
                return
            tmp = self.index_path.with_name(self.index_path.name + ".tmp")
            faiss.write_index(self._index, str(tmp))
            os.replace(tmp, self.index_path)
            atomic_write_json(self.meta_path, {
                "version": 1, "model_name": self.model_name, "dimension": int(self._index.d),
                "metric": self.METRIC, "ntotal": len(self._records), "next_id": self._next_id,
                "records": {str(i): r for i, r in self._records.items()},
            })

    def load(self) -> None:
        import faiss
        with self._lock:
            if not self.index_path.exists() or not self.meta_path.exists():
                self._index, self._records, self._next_id, self.model_name = None, {}, 0, None
                return
            try:
                index = faiss.read_index(str(self.index_path))
                meta = read_json(self.meta_path, {})
                records = {int(k): v for k, v in meta["records"].items()}
                if index.ntotal != len(records):
                    raise VectorStoreError(
                        f"index has {index.ntotal} vectors but metadata has {len(records)} records")
            except Exception:
                logger.exception("Index files are unreadable or inconsistent; starting empty. "
                                 "Documents must be re-indexed.")
                for p in (self.index_path, self.meta_path):
                    if p.exists():
                        os.replace(p, p.with_name(p.name + ".corrupt"))
                self._index, self._records, self._next_id, self.model_name = None, {}, 0, None
                self.recovered_from_corruption = True
                return
            self._index, self._records = index, records
            self._next_id = int(meta.get("next_id", max(records, default=-1) + 1))
            self.model_name = meta.get("model_name")
            logger.info("Loaded index: %d vectors, model=%s", index.ntotal, self.model_name)
