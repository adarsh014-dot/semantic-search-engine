"""Embedding tests.

* Contract tests run against the offline test embedder (shape / normalisation / batching).
* `real_model` tests load the configured Sentence Transformers model and are SKIPPED when it
  cannot be loaded (e.g. no internet on first run). Run them with network access once.
"""
import numpy as np
import pytest

from backend.config import Settings
from backend.services.embedding_service import create_embedder
from tests.conftest import DIM, HashingEmbedder


def test_contract_shapes_and_normalisation():
    e = HashingEmbedder()
    docs = e.embed_documents(["tcp handshake", "dns lookup", "photosynthesis"])
    assert docs.shape == (3, DIM) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert e.embed_query("tcp").shape == (DIM,)
    assert e.embed_documents([]).shape == (0, DIM)


def test_identical_text_gives_identical_vectors():
    e = HashingEmbedder()
    assert np.allclose(e.embed_text("same words"), e.embed_text("same words"))


@pytest.fixture(scope="module")
def real_model():
    emb = create_embedder(Settings(_env_file=None))
    try:
        emb.warm_up()
    except Exception as exc:  # no network / model not cached
        pytest.skip(f"real embedding model unavailable: {exc}")
    return emb


def test_real_model_dimension_and_unit_norm(real_model):
    vecs = real_model.embed_documents(["hello world", "another sentence"])
    assert vecs.shape == (2, real_model.dimension)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_real_model_is_semantic_not_lexical(real_model):
    q = real_model.embed_query("How does TCP start communication?")
    related = real_model.embed_documents(["TCP establishes communication using a three-way handshake."])[0]
    unrelated = real_model.embed_documents(["Photosynthesis converts sunlight into glucose in plants."])[0]
    assert float(q @ related) > float(q @ unrelated)


def test_real_model_identical_text_scores_one(real_model):
    v = real_model.embed_documents(["the quick brown fox"])[0]
    assert float(v @ v) == pytest.approx(1.0, abs=1e-4)
