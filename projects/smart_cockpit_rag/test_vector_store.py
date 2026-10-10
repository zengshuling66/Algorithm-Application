import numpy as np
import vector_store as store


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