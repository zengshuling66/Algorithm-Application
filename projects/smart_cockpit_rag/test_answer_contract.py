import json
import pytest
from answer_contract import parse_answer

EVIDENCE = {"E1": {"chunk_id": "demo-p43-c1", "source": "demo.pdf", "source_id": "demo", "content_hash": "demo-hash", "page": 43, "text": "Single: current door."}}
BASE = {"answer_status": "answered", "answer": " current door ", "evidence_ids": ["E1"]}

def test_answer_and_refusal():
    result = parse_answer(json.dumps(BASE), EVIDENCE)
    assert result["answer"] == "current door"
    assert result["citations"] == [{"evidence_id": "E1", **EVIDENCE["E1"]}]
    refused = {**BASE, "answer_status": "refused", "answer": "Insufficient evidence", "evidence_ids": []}
    assert parse_answer(json.dumps(refused), EVIDENCE)["citations"] == []

@pytest.mark.parametrize("bad", [
    "not json", "[]", "{}",
    {"answer_status": "other"}, {"answer": " "}, {"answer": 123},
    {"evidence_ids": "E1"}, {"evidence_ids": [[]]}, {"evidence_ids": []},
    {"evidence_ids": ["E99"]}, {"evidence_ids": ["E1", "E1"]}, {"source": "forged.pdf"},
])
def test_reject_invalid_answer(bad):
    raw = bad if isinstance(bad, str) else json.dumps({**BASE, **bad})
    with pytest.raises(ValueError):
        parse_answer(raw, EVIDENCE)