from travelrag.chunking import chunk_text, count_tokens


def test_short_text_is_one_chunk():
    assert chunk_text("A short paragraph.", size=400, overlap=50) == ["A short paragraph."]


def test_chunks_respect_size_and_cover_all_text():
    paragraphs = [f"Paragraph {i} " + "word " * 60 for i in range(12)]
    chunks = chunk_text("\n\n".join(paragraphs), size=200, overlap=40)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 200 + 40 for c in chunks)  # a chunk may carry up to `overlap` extra
    joined = "\n\n".join(chunks)
    assert all(f"Paragraph {i} " in joined for i in range(12))


def test_consecutive_chunks_overlap():
    paragraphs = [f"Unique marker {i}. " + "filler " * 40 for i in range(10)]
    chunks = chunk_text("\n\n".join(paragraphs), size=150, overlap=80)
    assert any(chunks[i].split("\n\n")[-1] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_oversized_paragraph_is_split():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = chunk_text(text, size=100, overlap=10)
    assert len(chunks) > 3
    assert all(count_tokens(c) <= 110 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("  \n\n  ", size=400, overlap=50) == []
