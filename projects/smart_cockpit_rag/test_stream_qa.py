from unittest.mock import Mock
from langchain_core.runnables import RunnableLambda
from langchain_query import stream_qa
import json
import pytest
from langchain_core.documents import Document

def test_empty_evidence_does_not_generate():
    retriever = RunnableLambda(lambda question: [])
    generator = Mock(side_effect=AssertionError("Must not generate"))
    events = list(stream_qa("test", retriever, generator))
    assert len(events) == 1
    assert events[0]["type"] == "final"
    assert events[0]["data"]["status"] == "no_evidence"
    assert events[0]["data"]["answer_status"] == "refused"
    generator.assert_not_called()

def make_case(evidence_id="E1", complete=True):
    doc = Document(id="demo-chunk", page_content="Demo evidence", metadata={
        "source": "demo.pdf", "source_id": "demo", "content_hash": "hash", "page": 1,
    })
    retriever = RunnableLambda(lambda question: [doc])
    closed = Mock()
    raw = json.dumps({"answer_status": "answered", "answer": "API_OK",
                      "evidence_ids": [evidence_id]})
    def generator(messages):
        try:
            yield {"type": "delta", "text": raw}
            if complete:
                yield {"type": "generation_done", "answer": raw,
                       "model": "offline-model", "usage": None}
        finally:
            closed()
    return retriever, generator, closed

@pytest.mark.parametrize("evidence_id, complete, rejected", [
    ("E1", True, False), ("E99", True, True), ("E1", False, True),
])
def test_final_requires_completion_and_valid_citation(evidence_id, complete, rejected):
    retriever, generator, closed = make_case(evidence_id, complete)
    seen = []
    events = stream_qa("test", retriever, generator)
    if rejected:
        with pytest.raises(ValueError):
            seen.extend(events)
        assert all(event["type"] != "final" for event in seen)
    else:
        seen.extend(events)
        assert [event["type"] for event in seen] == ["delta", "final"]
        citation = seen[-1]["data"]["citations"][0]
        assert citation["evidence_id"] == "E1" and citation["source"] == "demo.pdf"
    assert seen[0]["provisional"] is True
    closed.assert_called_once()

def test_closing_qa_closes_upstream():
    retriever, generator, closed = make_case()
    events = stream_qa("test", retriever, generator)
    assert next(events)["type"] == "delta"
    events.close()
    closed.assert_called_once()