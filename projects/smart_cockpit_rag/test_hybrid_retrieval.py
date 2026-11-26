from hybrid_retrieval import rrf_fuse

def test_rrf_contract():
    vector = [{"chunk_id": key, "score": 0.9} for key in ["A", "B", "C"]]
    bm25 = [{"chunk_id": key, "score": 20.0} for key in ["C", "A", "D"]]
    result = rrf_fuse({"vector": vector, "bm25": bm25})
    assert [item["chunk_id"] for item in result] == ["A", "C", "B"]
    assert result[0]["score"] == 1 / 61 + 1 / 62
    assert result[0]["retrieval_ranks"] == {"vector": 1, "bm25": 2}
    assert result[0]["retrieval_scores"] == {"vector": 0.9, "bm25": 20.0}
    assert vector[0]["score"] == 0.9
    assert rrf_fuse({"vector": [], "bm25": []}) == []