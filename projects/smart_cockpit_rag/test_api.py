from unittest.mock import MagicMock, Mock
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import api

@pytest.fixture
def offline_api(monkeypatch):
    doc = Document(id="demo-chunk", page_content="Demo evidence", metadata={
        "source": "demo.pdf", "source_id": "demo", "content_hash": "test-hash", "page": 1,
    })
    sdk = MagicMock()
    sdk.__enter__.return_value = sdk
    generate = Mock(return_value={
        "answer": '{"answer_status":"answered","answer":"API_OK","evidence_ids":["E1"]}',
        "model": "offline-model", "usage": None,
    })
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.setattr(api, "SentenceTransformer", Mock())
    monkeypatch.setattr(api, "build_client", lambda: sdk)
    monkeypatch.setattr(api, "generate_with_client", generate)
    monkeypatch.setattr(api, "build_rerank_retriever",
                        lambda *args: RunnableLambda(lambda q: [] if q == "none" else [doc]))
    with TestClient(api.app) as client:
        yield client, generate
    sdk.__exit__.assert_called_once()

def test_qa_returns_citation_without_internal_state(offline_api):
    client, generate = offline_api
    response = client.post("/qa", json={"question": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "API_OK" and data["citations"][0]["page"] == 1
    assert {"messages", "documents", "evidence_map"}.isdisjoint(data)
    generate.assert_called_once()

@pytest.mark.parametrize("question, code", [(" ", 422), ("none", 200)])
def test_qa_does_not_generate_for_invalid_or_empty_evidence(offline_api, question, code):
    client, generate = offline_api
    response = client.post("/qa", json={"question": question})
    assert response.status_code == code
    if code == 200:
        assert response.json()["status"] == "no_evidence"
        assert response.json()["answer_status"] == "refused"
    generate.assert_not_called()