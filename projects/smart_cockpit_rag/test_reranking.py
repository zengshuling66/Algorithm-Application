from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from reranking import rerank_results

def test_rerank_contract():
    rows = [{"chunk_id": str(i), "text": str(i), "score": 0.03, "source_id": "s", "content_hash": "h"} for i in range(3)]
    before = deepcopy(rows)
    model = SimpleNamespace(predict=lambda *args, **kwargs: np.array([0.0, 1.0, 1.0]))
    result = rerank_results("q", rows, model)
    assert [item["chunk_id"] for item in result] == ["1", "2", "0"]
    assert result[0] == {**rows[1], "rrf_rank": 2, "rerank_score": 1.0}
    assert rows == before
    assert rerank_results("q", [], None) == []

@pytest.mark.parametrize("scores", [[1.0], [np.nan] * 3, [np.inf] * 3])
def test_rerank_invalid_scores(scores):
    model = SimpleNamespace(predict=lambda *args, **kwargs: np.asarray(scores))
    with pytest.raises(ValueError):
        rerank_results("q", [{"chunk_id": str(i), "text": "x"} for i in range(3)], model)