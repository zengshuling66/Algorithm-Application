import pytest
from chunking import build_chunks, split_text


def test_split_text_keeps_overlap():
    chunks = split_text(
        "ABCDEFGHIJKL",
        chunk_size=5,
        overlap=2,
    )

    assert chunks == [
        "ABCDE",
        "DEFGH",
        "GHIJK",
        "JKL",
    ]


def test_split_text_handles_short_and_empty_text():
    assert split_text("ABCDE", chunk_size=5, overlap=2) == [
        "ABCDE"
    ]
    assert split_text("", chunk_size=5, overlap=2) == []


def test_build_chunks_keeps_each_pages_metadata():
    page_records = [
        {"text": "ABCDEFG", "source": "manual.pdf", "source_id": "manual_a","content_hash": "h1","page": 1},
        {"text": "XYZ", "source": "manual.pdf", "source_id": "manual_a","content_hash": "h1","page": 2},
    ]

    chunks = build_chunks(
        page_records,
        chunk_size=5,
        overlap=2,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "ABCDE",
        "DEFG",
        "XYZ",
    ]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "manual_a:h1:s5:o2:p1:c1",
        "manual_a:h1:s5:o2:p1:c2",
        "manual_a:h1:s5:o2:p2:c1",
    ]
    assert [chunk["page"] for chunk in chunks] == [1, 1, 2]
    assert [chunk["char_count"] for chunk in chunks] == [5, 4, 3]
    assert [chunk["source_id"] for chunk in chunks] == [
        "manual_a", "manual_a", "manual_a"
    ]
    assert [chunk["content_hash"] for chunk in chunks] == [
        "h1", "h1", "h1"
    ]


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_text("ABC", chunk_size=5, overlap=5)

def test_chunk_id_tracks_content_and_settings():
    page = {
        "text": "ABCDEFG",
        "source": "manual.pdf",
        "source_id": "manual_a",
        "content_hash": "h1",
        "page": 1,
    }

    original = build_chunks(
        [page], chunk_size=5, overlap=2
    )[0]["chunk_id"]
    repeated = build_chunks(
        [page], chunk_size=5, overlap=2
    )[0]["chunk_id"]

    page["content_hash"] = "h2"
    updated = build_chunks(
        [page], chunk_size=5, overlap=2
    )[0]["chunk_id"]
    reconfigured = build_chunks(
        [page], chunk_size=6, overlap=2
    )[0]["chunk_id"]

    assert repeated == original
    assert updated != original
    assert reconfigured != updated

#运行方式：python -m pytest -q .\projects\smart_cockpit_rag\test_chunking.py