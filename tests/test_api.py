"""End-to-end API tests (upload -> index -> search -> restart -> delete).
They use the offline HashingEmbedder from conftest; see the note there."""
import io
import time
from pathlib import Path

from docx import Document

from tests.conftest import make_client

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"


def upload(client, name, data, ctype="application/octet-stream"):
    return client.post("/api/documents/upload", files=[("files", (name, data, ctype))])


def wait_until_idle(client, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        docs = client.get("/api/documents").json()
        if all(d["status"] != "processing" for d in docs):
            return docs
        time.sleep(0.05)
    raise AssertionError("indexing did not finish")


def upload_and_index_samples(client):
    for f in ("networking_basics.txt", "algorithms_notes.docx", "biology_primer.pdf"):
        r = upload(client, f, (SAMPLES / f).read_bytes()).json()["results"][0]
        assert r["status"] == "uploaded", r
    assert client.post("/api/documents/index", json={}).status_code == 202
    return wait_until_idle(client)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["indexed_chunks"] == 0


def test_full_workflow_pdf_txt_docx(client):
    docs = upload_and_index_samples(client)
    assert {d["file_type"] for d in docs} == {"pdf", "txt", "docx"}
    assert all(d["status"] == "indexed" and d["chunks"] > 0 for d in docs), docs
    pdf = next(d for d in docs if d["file_type"] == "pdf")
    assert pdf["pages"] == 2

    r = client.post("/api/search", json={"query": "How does TCP establish a connection?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    top = body["results"][0]
    assert top["source"] == "networking_basics.txt"
    # (the offline test embedder is purely lexical, so check the top 3 rather than rank 1)
    assert any("three-way handshake" in x["text"] for x in body["results"])
    assert 0 < top["score"] <= 1
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)

    # PDF results carry real page numbers
    r = client.post("/api/search", json={"query": "mitochondria glucose ATP respiration"}).json()
    assert r["results"][0]["source"] == "biology_primer.pdf" and r["results"][0]["page"] == 2


def test_chunk_context_endpoint(client):
    upload_and_index_samples(client)
    hit = client.post("/api/search", json={"query": "three-way handshake SYN"}).json()["results"][0]
    r = client.get(f"/api/documents/{hit['document_id']}/chunks/{hit['chunk_index']}")
    assert r.status_code == 200 and r.json()["chunk"]["text"] == hit["text"]


def test_search_before_indexing_is_a_clear_409(client):
    r = client.post("/api/search", json={"query": "anything"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "index_empty"
    assert "Process a document first" in r.json()["error"]["message"]


def test_invalid_search_requests(client):
    upload_and_index_samples(client)
    assert client.post("/api/search", json={"query": "   "}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/search", json={}).status_code == 422
    r = client.post("/api/search", content="not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "Traceback" not in r.text


def test_unsupported_and_fake_files_are_rejected(client):
    r = upload(client, "evil.exe", b"MZ....").json()["results"][0]
    assert r["status"] == "rejected" and "Unsupported file type" in r["message"]
    r = upload(client, "fake.pdf", b"this is not a pdf").json()["results"][0]
    assert r["status"] == "rejected"
    r = upload(client, "empty.txt", b"").json()["results"][0]
    assert r["status"] == "rejected"
    r = upload(client, "bin.txt", b"\x00\x01\x02binary").json()["results"][0]
    assert r["status"] == "rejected"


def test_oversized_file_rejected(client):  # settings fixture sets a 1 MB limit
    r = upload(client, "big.txt", b"a" * (1024 * 1024 + 10)).json()["results"][0]
    assert r["status"] == "rejected" and "limit" in r["message"]


def test_path_traversal_filename_is_neutralised(client, settings):
    r = upload(client, "../../etc/passwd.txt", b"hello world text").json()["results"][0]
    assert r["status"] == "uploaded" and r["document"]["filename"] == "passwd.txt"
    stored = [p.name for p in settings.documents_dir.iterdir()]
    assert len(stored) == 1 and stored[0].endswith(".txt") and "passwd" not in stored[0]
    assert not (settings.data_dir.parent / "etc").exists()


def test_duplicate_upload_detected(client):
    data = b"Some identical content for hashing."
    assert upload(client, "a.txt", data).json()["results"][0]["status"] == "uploaded"
    r = upload(client, "renamed.txt", data).json()["results"][0]
    assert r["status"] == "duplicate" and r["message"] == "Document already exists."
    assert len(client.get("/api/documents").json()) == 1


def test_empty_docx_and_corrupt_pdf_fail_gracefully(client):
    buf = io.BytesIO(); Document().save(buf)
    upload(client, "blank.docx", buf.getvalue())
    pdf = b"%PDF-1.4\n" + b"garbage" * 50
    upload(client, "broken.pdf", pdf)
    client.post("/api/documents/index", json={})
    docs = {d["filename"]: d for d in wait_until_idle(client)}
    assert docs["blank.docx"]["status"] == "failed" and docs["blank.docx"]["error"]
    assert docs["broken.pdf"]["status"] == "failed"
    assert "Traceback" not in (docs["broken.pdf"]["error"] or "")


def test_restart_restores_index_without_reupload(client, settings):
    upload_and_index_samples(client)
    before = client.post("/api/search", json={"query": "TCP three-way handshake"}).json()

    restarted = make_client(settings)  # new app instance, same data directory
    assert restarted.get("/api/health").json()["indexed_chunks"] > 0
    after = restarted.post("/api/search", json={"query": "TCP three-way handshake"}).json()
    assert [r["chunk_id"] for r in after["results"]] == [r["chunk_id"] for r in before["results"]]
    assert [r["score"] for r in after["results"]] == [r["score"] for r in before["results"]]
    assert all(d["status"] == "indexed" for d in restarted.get("/api/documents").json())


def test_delete_removes_chunks_from_search(client):
    docs = upload_and_index_samples(client)
    txt = next(d for d in docs if d["file_type"] == "txt")
    assert client.delete(f"/api/documents/{txt['id']}").status_code == 200
    results = client.post("/api/search", json={"query": "TCP three-way handshake", "top_k": 10}).json()["results"]
    assert all(r["document_id"] != txt["id"] for r in results)
    assert client.get(f"/api/documents/{txt['id']}").status_code == 404
    assert client.delete(f"/api/documents/{txt['id']}").status_code == 404


def test_reindex_all_rebuilds_index(client):
    upload_and_index_samples(client)
    n = client.get("/api/health").json()["indexed_chunks"]
    assert client.post("/api/documents/reindex", json={}).status_code == 202
    wait_until_idle(client)
    assert client.get("/api/health").json()["indexed_chunks"] == n  # no duplicate vectors


def test_reindex_single_document_does_not_duplicate(client):
    docs = upload_and_index_samples(client)
    n = client.get("/api/health").json()["indexed_chunks"]
    client.post("/api/documents/reindex", json={"document_ids": [docs[0]["id"]]})
    wait_until_idle(client)
    assert client.get("/api/health").json()["indexed_chunks"] == n


def test_unknown_document_and_unknown_route(client):
    assert client.get("/api/documents/nope").status_code == 404
    assert client.post("/api/documents/index", json={"document_ids": ["nope"]}).status_code == 404
    assert client.get("/api/does-not-exist").json()["error"]["message"]
