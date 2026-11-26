import pytest
import numpy as np
import vector_store as store
from bm25_retrieval import build_bm25_search


def test_store_is_persistent_idempotent_and_scoped(tmp_path, monkeypatch): #函数名以 test_ 开头，pytest 会发现并调用它
                                                #参数 tmp_path 和 monkeypatch 由 pytest 自动提供
    monkeypatch.setattr(store, "STORAGE_PATH", tmp_path / ".qdrant") #monkeypatch.setattr() 临时把 store.STORAGE_PATH 改成测试目录，测试结束后恢复原值
    chunks = [
        {"chunk_id": "a:h1", "source_id": "a", "content_hash": "h1",
         "page": 1, "text": "A version 1"},
        {"chunk_id": "a:h2", "source_id": "a", "content_hash": "h2",
         "page": 2, "text": "A version 2"},
        {"chunk_id": "b:h1", "source_id": "b", "content_hash": "h1",
         "page": 3, "text": "B version 1"},
    ]
    vectors = np.zeros((3, 512), dtype=np.float32)
    vectors[:, 0] = 1.0 #逗号前的 : 选全部行，后面的 0 选第一列，因此三行都变成 [1, 0, …, 0]，每行长度为 512，L2 范数为 1
    points = store.build_points(chunks, vectors)

    assert store.save_points(points, dimension=512) == (0, 3)
    assert store.save_points(points, dimension=512) == (3, 3)
    results = store.search_points(vectors[0], "a", "h1", top_k=3)
    assert len(results) == 1
    for key in ("chunk_id", "text", "page", "source_id", "content_hash"):
        assert results[0][key] == chunks[0][key]
    assert store.search_points(vectors[0], "missing", "h1") == []
    assert store.search_points(vectors[0], "a", "missing") == []


def test_upsert_keeps_omitted_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORAGE_PATH", tmp_path / ".qdrant")
    chunks = [
        {"chunk_id": "demo:h1:c1", "source_id": "demo", "content_hash": "h1", "text": "first"},
        {"chunk_id": "demo:h1:c2", "source_id": "demo", "content_hash": "h1", "text": "second"},
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    points = store.build_points(chunks, vectors)
    assert store.save_points(points, dimension=2) == (0, 2)
    assert store.save_points(points[:1], dimension=2) == (2, 2)
    stored = store.load_stored_chunks("demo", "h1")
    assert {row["chunk_id"] for row in stored} == {
        "demo:h1:c1", "demo:h1:c2",
    }


def test_delete_version_is_scoped_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORAGE_PATH", tmp_path / ".qdrant")
    assert store.delete_stored_version("a", "h1") == (0, 0)
    chunks = [
        {"chunk_id": "a:h1", "source_id": "a", "content_hash": "h1", "text": "old"},
        {"chunk_id": "a:h2", "source_id": "a", "content_hash": "h2", "text": "new"},
        {"chunk_id": "b:h1", "source_id": "b", "content_hash": "h1", "text": "other"},
    ]
    points = store.build_points(chunks, np.ones((3, 2), dtype=np.float32))
    assert store.save_points(points, dimension=2) == (0, 3)
    for source_id, content_hash in (("", "h1"), ("a", ""), (" ", "h1"), ("a", None)):
        with pytest.raises(ValueError):
            store.delete_stored_version(source_id, content_hash)
    assert store.delete_stored_version("a", "h1") == (1, 0)
    assert store.search_points(np.ones(2, dtype=np.float32), "a", "h1") == []
    assert len(store.load_stored_chunks("a", "h2")) == 1
    assert len(store.load_stored_chunks("b", "h1")) == 1
    assert store.delete_stored_version("a", "h1") == (0, 0)


def test_prune_and_bm25_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORAGE_PATH", tmp_path / ".qdrant")
    chunks = [
        {"chunk_id": "a:h1:c1", "source_id": "a", "content_hash": "h1", "text": "keep"},
        {"chunk_id": "a:h1:c2", "source_id": "a", "content_hash": "h1", "text": "obsolete"},
        {"chunk_id": "a:h2:c1", "source_id": "a", "content_hash": "h2", "text": "version"},
        {"chunk_id": "b:h1:c1", "source_id": "b", "content_hash": "h1", "text": "other"},
    ]
    store.save_points(store.build_points(chunks, np.ones((4, 2), dtype=np.float32)), 2)
    old_search = build_bm25_search(store.load_stored_chunks("a", "h1"))
    new = [{**chunks[0], "text": "fresh"}]
    store.save_points(store.build_points(new, np.ones((1, 2), dtype=np.float32)), 2)
    for keep in ([], ["missing"]):
        with pytest.raises(ValueError):
            store.prune_stored_chunks("a", "h1", keep)
    assert store.prune_stored_chunks("a", "h1", [new[0]["chunk_id"]]) == 1
    assert store.load_stored_chunks("a", "h1") == new
    assert store.prune_stored_chunks("a", "h1", [new[0]["chunk_id"]]) == 0
    assert len(store.load_stored_chunks("a", "h2")) == len(store.load_stored_chunks("b", "h1")) == 1
    assert old_search("obsolete")
    fresh_search = build_bm25_search(store.load_stored_chunks("a", "h1"))
    assert fresh_search("obsolete") == []