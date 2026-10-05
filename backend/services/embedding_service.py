"""Embedding layer. The rest of the app only talks to `BaseEmbedder`.

To add another backend (OpenAI, Ollama, a CLIP model, ...) subclass BaseEmbedder and
return it from `create_embedder`. The model is chosen purely by configuration.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

from backend.config import Settings
from backend.utils.errors import EmbeddingError

logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so that inner product == cosine similarity."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class BaseEmbedder(ABC):
    model_name: str

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def warm_up(self) -> None:
        """Load the model eagerly (no-op for lightweight embedders)."""

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray: ...

    # Prefixes are applied here so callers never think about model-specific quirks.
    document_prefix: str = ""
    query_prefix: str = ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed many passages (batched). Returns float32 array (n, dim), L2-normalised."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _l2_normalize(self._encode([self.document_prefix + t for t in texts]))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query. Returns float32 vector (dim,), L2-normalised."""
        return _l2_normalize(self._encode([self.query_prefix + query]))[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text with no query/document prefix."""
        return _l2_normalize(self._encode([text]))[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Pretrained Sentence Transformers model, loaded once and reused for all requests."""

    def __init__(self, model_name: str, query_prefix: str = "", document_prefix: str = "",
                 batch_size: int = 32, device: str | None = None):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:  # make sure concurrent requests don't load it twice
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading embedding model %s ...", self.model_name)
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    self._dimension = int(self._model.get_sentence_embedding_dimension())
                    logger.info("Model ready (dimension=%d)", self._dimension)
                except Exception as exc:
                    logger.exception("Could not load embedding model %s", self.model_name)
                    raise EmbeddingError(
                        f"The embedding model '{self.model_name}' could not be loaded. "
                        "Check the model name and your internet connection (first run downloads it)."
                    ) from exc
        return self._model

    def warm_up(self) -> None:
        self._load()

    @property
    def dimension(self) -> int:
        self._load()
        return int(self._dimension)  # type: ignore[arg-type]

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        try:
            return model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            logger.exception("Embedding failed")
            raise EmbeddingError("Failed to generate embeddings.") from exc


def create_embedder(settings: Settings) -> BaseEmbedder:
    """Factory: swap models (or whole backends) via configuration only."""
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )
