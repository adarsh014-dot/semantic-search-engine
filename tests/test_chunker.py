from backend.services.chunker import Chunker
from backend.services.document_processor import ExtractedPage
from backend.services.text_cleaner import clean_text

SENTENCES = [f"Sentence number {i} talks about topic {i} in some detail." for i in range(40)]
TEXT = " ".join(SENTENCES)


def test_chunks_are_generated_and_respect_size():
    chunker = Chunker(chunk_size_tokens=65, overlap_tokens=13, tokens_per_word=1.3)  # 50 / 10 words
    pieces = chunker.split_text(TEXT)
    assert len(pieces) > 1
    assert all(len(p.split()) <= chunker.max_words for p in pieces)


def test_overlap_is_present_between_neighbours():
    chunker = Chunker(65, 26, 1.3)  # 50-word chunks, 20-word overlap
    pieces = chunker.split_text(TEXT)
    for a, b in zip(pieces, pieces[1:]):
        assert a.split()[-1] in b.split(), "last words of a chunk should reappear in the next one"
        assert b.split()[0] in a.split(), "next chunk should start with text already seen"


def test_zero_overlap_means_no_repetition():
    pieces = Chunker(65, 0, 1.3).split_text(TEXT)
    assert " ".join(pieces).split() == TEXT.split()


def test_metadata_ids_pages_and_order_are_preserved():
    pages = [ExtractedPage(TEXT, "notes.pdf", 1), ExtractedPage(TEXT, "notes.pdf", 2)]
    chunks = Chunker(65, 13, 1.3).chunk_pages(pages, document_id="doc1")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "notes.pdf" and c.document_id == "doc1" and c.created_at for c in chunks)
    pages_in_order = [c.page for c in chunks]
    assert pages_in_order == sorted(pages_in_order)  # page 1 chunks come before page 2


def test_overlong_sentence_is_hard_split():
    pieces = Chunker(65, 0, 1.3).split_text("word " * 200)
    assert len(pieces) == 4 and all(len(p.split()) <= 50 for p in pieces)


def test_empty_text_gives_no_chunks():
    assert Chunker().split_text("") == []


def test_cleaner_fixes_artifacts_but_keeps_content():
    raw = "Intro  line\r\ncontinues here.\n\n\n12\nSecond para with connec-\ntion and ﬁne print.\x00\n"
    cleaned = clean_text(raw)
    assert cleaned == "Intro line continues here.\n\nSecond para with connection and fine print."
