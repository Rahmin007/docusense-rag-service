from app.chunking import recursive_split


def test_chunking_is_deterministic_and_overlapped():
    text = "A" * 500 + "\n\n" + "B" * 500
    first = recursive_split(text, chunk_size=300, overlap=50)
    second = recursive_split(text, chunk_size=300, overlap=50)
    assert first == second
    assert len(first) >= 2
    assert len(first[0]) <= 300


def test_short_document_stays_as_one_chunk():
    assert recursive_split("hello world", 100, 10) == ["hello world"]
