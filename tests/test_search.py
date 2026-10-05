"""Vector store tests (FAISS): insert, search, metadata mapping, delete, persistence."""
import numpy as np
import pytest

from backend.services.vector_store import FaissVectorStore
from backend.utils.errors import VectorStoreError


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def rec(i, doc="d1"):
    return {"chunk_id": f"{doc}-{i}", "document_id": doc, "source": f"{doc}.txt",
            "page": i, "chunk_index": i, "text": f"text {i}"}


@pytest.fixture
def store(tmp_path):
    s = FaissVectorStore(tmp_path / "index")
    vecs = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0)])
    s.add(vecs, [rec(0), rec(1), rec(2, "d2")], "m")
    return s


def test_insert_and_count(store):
    assert store.count == 3 and store.dimension == 3


def test_search_scores_are_cosine_similarity_and_sorted(store):
    hits = store.search(unit(1, 0, 0), 3)
    assert [r["chunk_id"] for _, r in hits] == ["d1-0", "d2-2", "d1-1"]
    assert hits[0][0] == pytest.approx(1.0, abs=1e-6)             # identical direction
    assert hits[1][0] == pytest.approx(np.cos(np.pi / 4), abs=1e-6)  # 45 degrees
    assert hits[2][0] == pytest.approx(0.0, abs=1e-6)             # orthogonal


def test_metadata_maps_to_the_right_vector(store):
    _, r = store.search(unit(0, 1, 0), 1)[0]
    assert (r["chunk_id"], r["page"], r["text"]) == ("d1-1", 1, "text 1")


def test_k_larger_than_index_is_fine(store):
    assert len(store.search(unit(1, 0, 0), 50)) == 3


def test_delete_document_removes_only_its_vectors(store):
    assert store.delete_document("d1") == 2
    hits = store.search(unit(1, 0, 0), 5)
    assert [r["chunk_id"] for _, r in hits] == ["d2-2"]


def test_dimension_mismatch_is_rejected(store):
    with pytest.raises(VectorStoreError):
        store.add(np.ones((1, 5), dtype=np.float32), [rec(9)], "m")


def test_persistence_roundtrip(store, tmp_path):
    store.save()
    reloaded = FaissVectorStore(tmp_path / "index")
    assert reloaded.count == 3 and reloaded.model_name == "m"
    assert reloaded.search(unit(0, 1, 0), 1)[0][1]["chunk_id"] == "d1-1"
    reloaded.add(np.stack([unit(0, 0, 1)]), [rec(5, "d3")], "m")  # ids keep increasing
    assert reloaded.search(unit(0, 0, 1), 1)[0][1]["chunk_id"] == "d3-5"


def test_corrupt_index_recovers_empty(store, tmp_path):
    store.save()
    (tmp_path / "index" / "chunks.json").write_text("{not json")
    s = FaissVectorStore(tmp_path / "index")
    assert s.count == 0 and s.recovered_from_corruption


def test_empty_store_search_returns_nothing(tmp_path):
    assert FaissVectorStore(tmp_path / "i").search(unit(1, 0), 3) == []
