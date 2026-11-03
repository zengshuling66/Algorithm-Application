import json
import pytest
from evaluation import evaluate_qa_row, write_report, summarize_qa_review

def test_empty_context_does_not_generate():
    def forbidden(messages):
        raise AssertionError("Generator must not run")
    row = {"question": "demo", "results": []}
    result = evaluate_qa_row(row, generator=forbidden)
    assert result["status"] == "no_evidence"
    assert result["answer_status"] == "refused"

ROW = {"question": "demo", "results": [
    {"chunk_id": "demo-1", "source": "demo.pdf", "source_id": "demo",
     "content_hash": "demo-hash", "page": 1, "text": "Current door."}
]}
RAW = json.dumps({"answer_status": "answered", "answer": "Current door.", "evidence_ids": ["E1"]})

@pytest.mark.parametrize("raw,status", [(RAW, "generated"), ("not json", "error")])
def test_generated_and_invalid_json(raw, status):
    def fake(messages):
        return {"answer": raw, "model": "fake", "usage": {"total_tokens": 7}}
    result = evaluate_qa_row(ROW, generator=fake)
    assert result["status"] == result["generation_status"] == status
    assert result["raw_answer"] == raw and result["usage"]["total_tokens"] == 7
    if status == "generated":
        assert result["citations"][0]["chunk_id"] == "demo-1"
    else:
        assert result["error_type"] == "JSONDecodeError"
        assert "answer_status" not in result

def test_transport_error_does_not_leak():
    def broken(messages):
        raise RuntimeError("private-detail")
    result = evaluate_qa_row(ROW, generator=broken)
    assert result["status"] == "error" and result["error_type"] == "RuntimeError"
    assert "private-detail" not in json.dumps(result)

def test_report_refuses_overwrite(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, {"saved": True})
    with pytest.raises(FileExistsError):
        write_report(path, {"saved": False})
    assert json.loads(path.read_text(encoding="utf-8"))["saved"] is True

def test_review_summary_counts_and_preserves_input():
    report = {"report_type": "qa_review", "cases": [
        {"method": "vector", "review": {
            "task_pass": False, "faithful": True,
            "citation_pass": None, "refusal_appropriate": True,
            "failure_types": ["retrieval", "retrieval"]}},
        {"method": "vector", "review": {
            "task_pass": True, "faithful": True,
            "citation_pass": True, "refusal_appropriate": None,
            "failure_types": []}},
    ]}
    before = json.dumps(report, sort_keys=True)
    summary = summarize_qa_review(report)
    vector = summary["vector"]
    assert vector["case_count"] == 2
    assert vector["task_pass"] == {"n": 2, "value": 0.5}
    assert vector["faithful"] == {"n": 2, "value": 1.0}
    assert vector["citation_pass"] == {"n": 1, "value": 1.0}
    assert vector["refusal_appropriate"] == {"n": 1, "value": 1.0}
    assert vector["failure_counts"] == {"retrieval": 1}
    assert summary["bm25"]["task_pass"] == {"n": 0, "value": None}
    assert json.dumps(report, sort_keys=True) == before

@pytest.mark.parametrize("field,value", [
    ("task_pass", None), ("task_pass", "true"), ("faithful", 1),
])
def test_review_summary_rejects_invalid_scores(field, value):
    review = {
        "task_pass": True, "faithful": True,
        "citation_pass": True, "refusal_appropriate": None,
        "failure_types": [],
    }
    review[field] = value
    report = {"report_type": "qa_review",
              "cases": [{"method": "vector", "review": review}]}
    with pytest.raises(ValueError):
        summarize_qa_review(report)