"""Shared fixtures.

IMPORTANT: `HashingEmbedder` is a TEST-ONLY stand-in that lets the API/vector-store tests
run offline and instantly. It hashes words into a vector, so it only captures word overlap.
It is never used by the application. Real semantic behaviour is tested in test_embeddings.py
(which needs the real model and skips itself if it cannot be downloaded).
"""
import re
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.services.embedding_service import BaseEmbedder

DIM = 256


class HashingEmbedder(BaseEmbedder):
    model_name = "test-hashing-embedder"

    @property
    def dimension(self) -> int:
        return DIM

    def _encode(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % DIM] += 1.0
        return out


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", preload_model=False, search_min_score=0.0,
                    chunk_size_tokens=60, chunk_overlap_tokens=10, max_upload_mb=1,
                    _env_file=None)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, HashingEmbedder())) as c:
        yield c


def make_client(settings) -> TestClient:
    """A fresh app on the same data dir == a server restart."""
    return TestClient(create_app(settings, HashingEmbedder()))
