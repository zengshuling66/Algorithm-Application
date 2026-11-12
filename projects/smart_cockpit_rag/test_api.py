from unittest.mock import MagicMock, Mock
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import api
import json
import asyncio

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

@pytest.fixture
def offline_stream(offline_api, monkeypatch):
    client, _ = offline_api
    raw = '{"answer_status":"answered","answer":"API_OK","evidence_ids":["E1"]}'
    def generate(*args, **kwargs):
        yield {"type": "delta", "text": raw}
        yield {"type": "generation_done", "answer": raw,
               "model": "offline-model", "usage": None}
    stream = Mock(side_effect=generate)
    monkeypatch.setattr(api, "stream_with_client", stream)
    return client, stream

@pytest.mark.parametrize("question, code", [("test", 200), ("none", 200), (" ", 422)])
def test_stream_response(offline_stream, question, code):
    client, stream = offline_stream
    response = client.post("/qa/stream", json={"question": question})
    assert response.status_code == code
    assert not api.app.state.qa_lock.locked()
    if code == 422:
        stream.assert_not_called()
        return
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ")]
    if question == "test":
        assert [event["type"] for event in events] == ["delta", "final"]
        assert events[0]["provisional"] is True
        assert events[-1]["data"]["citations"][0]["page"] == 1
        stream.assert_called_once()
    else:
        assert [event["type"] for event in events] == ["final"]
        assert events[0]["data"]["status"] == "no_evidence"
        assert events[0]["data"]["answer_status"] == "refused"
        stream.assert_not_called()

def test_stream_failure_is_error_not_final(offline_stream):
    client, stream = offline_stream
    stream.side_effect = ValueError("private error")
    response = client.post("/qa/stream", json={"question": "test"})
    assert response.status_code == 200
    assert "event: error" in response.text and "event: final" not in response.text
    assert "private error" not in response.text
    assert not api.app.state.qa_lock.locked()

def test_stream_busy_and_disconnect_cleanup(offline_stream):
    client, stream = offline_stream
    with api.app.state.qa_lock:
        response = client.post("/qa/stream", json={"question": "test"})
        assert response.status_code == 503
        stream.assert_not_called()
    closed = Mock()
    def generate(*args, **kwargs):
        try:
            while True:
                yield {"type": "delta", "text": "partial"}
        finally:
            closed()
    stream.side_effect = generate
    async def run_request():
        disconnected = asyncio.Event()
        delivered = False
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b'{"question":"test"}',
                        "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if (message["type"] == "http.response.body"
                    and b"event: delta" in message.get("body", b"")):
                disconnected.set()
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "path": "/qa/stream",
            "query_string": b"", "headers": [(b"content-type", b"application/json")],
        }
        await asyncio.wait_for(api.app(scope, receive, send), timeout=3)
        assert disconnected.is_set()
    asyncio.run(run_request())
    closed.assert_called_once()
    assert not api.app.state.qa_lock.locked()
    assert client.post("/qa", json={"question": "test"}).status_code == 200