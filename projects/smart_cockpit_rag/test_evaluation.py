import json
import pytest
from evaluation import evaluate_qa_row, write_report, summarize_qa_review, score_required_blocks, add_block_scores, compare_retrieval, run_retrieval_evaluation, prepare_qa_review, score_review_metrics

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

def test_required_blocks_supports_chunk_id_and_legacy_default():
    results = [
        {"block_id": "body", "chunk_id": "other"},
        {"block_id": "body", "chunk_id": "wanted"},
    ]
    score = score_required_blocks(
        ["wanted", "missing"], results, id_field="chunk_id"
    )
    assert score == {
        "required_block_recall": 0.5,
        "required_block_rr": 0.5,
        "all_required_blocks_hit": False,
    }
    assert score_required_blocks(["body"], results)["all_required_blocks_hit"]

@pytest.mark.parametrize("required,alternatives,ids,expected", [
    (["A"], [["B"]], ["B"], (1.0, 1.0, True, 1)),
    (["A"], [["B"]], ["X", "B"], (1.0, 0.5, True, 1)),
    (["A", "C"], [["B", "D"]], ["A", "B", "D"], (1.0, 0.5, True, 1)),
    (["A", "C"], [["B", "D"]], ["A", "D"], (0.5, 1.0, False, 0)),
    (["A"], [], ["A"], (1.0, 1.0, True, 0)),
    (["A"], [["B"]], [], (0.0, 0.0, False, None)),
    (["A", "C"], [["A", "D"]], ["A", "C", "D"], (1.0, 1.0, True, 0)),
])
def test_alternative_evidence_sets(required, alternatives, ids, expected):
    identity = {
        "source_id": "demo", "content_hash": "hash", "pipeline_version": "v1",
    }
    gold = {
        **identity, "collection": "demo", "id_field": "chunk_id",
        "cases": {"q": required},
        "alternative_evidence_sets": {"q": alternatives},
    }
    row = {
        "case_id": "q", "answerable": True,
        "results": [
            {**identity, "chunk_id": cid, "block_id": "body"} for cid in ids
        ],
    }
    report = {**identity, "collection": "demo", "cases": [row]}
    scored = add_block_scores(report, gold)["cases"][0]
    keys = ("required_block_recall", "required_block_rr",
            "all_required_blocks_hit", "selected_evidence_set")
    assert tuple(scored[key] for key in keys) == expected

@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("split", ["dev", "test"])
def test_compare_topk_keeps_candidates(monkeypatch, k, split):
    rows = [{"chunk_id": str(i), "page": i, "source_id": "demo"} for i in range(1, 11)]
    monkeypatch.setattr("evaluation.load_stored_chunks", lambda s, h: rows)
    monkeypatch.setattr("evaluation.build_bm25_search",
                        lambda chunks: lambda q, top_k: chunks[:top_k])
    monkeypatch.setattr("evaluation.build_reranker", lambda: None)
    monkeypatch.setattr("evaluation.search_stored_chunks",
                        lambda q, m, s, h, top_k: rows[:top_k])
    monkeypatch.setattr("evaluation.rrf_fuse", lambda groups, top_k: rows[:top_k])
    monkeypatch.setattr("evaluation.rerank_results",
                        lambda q, items, model, top_k: items[:top_k])
    cases = [{"id": "q", "question": "demo", "split": split, "answerable": False}]
    report = compare_retrieval(cases, None, "demo", "hash", top_k=k, expected_split=split)
    assert report["split"] == split
    assert report["top_k"] == k
    assert len(report["cases"]) == 4
    for row in report["cases"]:
        assert len(row["candidate_results"]) == 10
        assert row["results"] == row["candidate_results"][:k]

@pytest.mark.parametrize("top_k", [0, 11, True, 1.5, "5"])
def test_runner_rejects_invalid_topk(tmp_path, monkeypatch, top_k):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"top_k": top_k}), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Model must not load")
    monkeypatch.setattr("evaluation.SentenceTransformer", forbidden)
    with pytest.raises(ValueError, match="top_k"):
        run_retrieval_evaluation(config, tmp_path / "out.json")

def test_prepare_review_marks_type_and_preserves_source(tmp_path):
    source = tmp_path / "qa.json"
    data = {
        "report_type": "qa", "run_status": "complete", "expected_records": 1,
        "cases": [{"case_id": "q", "answerable": False, "results": []}],
    }
    source.write_text(json.dumps(data), encoding="utf-8")
    result = prepare_qa_review(source, tmp_path / "review.json")
    assert result["report_type"] == "qa_review"
    assert result["review_schema_version"] == "qa-review-v2"
    assert result["qa_report"] == str(source.resolve())
    assert json.loads(source.read_text(encoding="utf-8")) == data

@pytest.mark.parametrize("flags,expected", [
    ([True, True], (0.4, 1.0)), ([True, False], (0.2, 0.5)),
    ([False, False], (0.0, 0.0)), ([None, False], (None, None)),
    ([1, False], ValueError), ([True], ValueError),
])
def test_score_review_metrics(flags, expected):
    row = {
        "answerable": True, "results": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        "reference_facts": [{"id": "F1"}, {"id": "F2"}],
        "review": {
            "chunk_relevance": [{"chunk_id": cid, "relevant": v}
                                for cid, v in zip(("a", "b"), flags)],
            "fact_checks": [{"fact_id": fid, "covered_correctly": v}
                            for fid, v in zip(("F1", "F2"), flags)],
        },
    }
    if expected is ValueError:
        with pytest.raises(ValueError):
            score_review_metrics(row, 5)
    else:
        result = score_review_metrics(row, 5)
        assert (result["precision_at_k"], result["key_fact_coverage"]) == expected

def test_review_summary_macro_and_pending(monkeypatch):
    base = {"task_pass": None, "faithful": None, "citation_pass": None,
            "refusal_appropriate": None, "failure_types": []}
    rows = []
    for answerable, facts, p, f in [
        (True, [{}], 0.4, 1.0), (True, [{}, {}, {}], 0.0, 1 / 3),
        (True, [{}], None, None), (False, [], 1.0, 1.0),
    ]:
        rows.append({"method": "vector", "answerable": answerable,
                     "reference_facts": facts, "review": dict(base),
                     "scores": {"precision_at_k": p, "key_fact_coverage": f}})
    monkeypatch.setattr("evaluation.score_review_metrics", lambda row, k: row["scores"])
    report = {"report_type": "qa_review", "review_schema_version": "qa-review-v2",
              "top_k": 5, "cases": rows}
    result = summarize_qa_review(report)
    assert result["vector"]["precision_at_k"] == {
        "k": 5, "applicable_n": 3, "n": 2, "pending_n": 1, "value": 0.2}
    assert result["vector"]["key_fact_coverage"]["value"] == pytest.approx(2 / 3)
    assert result["vector"]["task_pending_n"] == 4
    assert result["bm25"]["precision_at_k"]["value"] is None

@pytest.mark.parametrize("methods", [None, ["rerank"]])
@pytest.mark.parametrize("split", ["dev", "test"])
def test_saved_facts_reach_review(tmp_path, monkeypatch, split, methods):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({} if methods is None else {"qa_methods": methods}), encoding="utf-8")
    from evaluation import run_saved_qa
    facts = [{"id": "F1", "text": "Current door."}]
    case = {
        "id": "q", "question": "demo", "split": split, "answerable": True,
        "reference_answer": "Current door.",
        "evidence": {"source_id": "demo", "page": 1, "anchor": "Current"},
        "reference_facts": facts,
    }
    rows = [
        {**ROW, "case_id": "q", "method": m, "answerable": True,
         "generation_status": "not_run"}
        for m in ("vector", "bm25", "hybrid", "rerank")
    ]
    def fake_load(path, expected_split="dev"):
        assert expected_split == split
        return [case], {}, {}
    monkeypatch.setattr("evaluation.load_evaluation_inputs", fake_load)
    monkeypatch.setattr("evaluation.add_block_scores", lambda report, gold: report)
    source, output = tmp_path / "source.json", tmp_path / "qa.json"
    gold, review_path = tmp_path / "gold.json", tmp_path / "review.json"
    source.write_text(json.dumps({"top_k": 3, "split": split, "cases": rows}), encoding="utf-8")
    gold.write_text("{}", encoding="utf-8")
    fake = lambda messages: {"answer": RAW, "model": "fake", "usage": None}
    run_saved_qa(config, source, "all", output, generator=fake, expected_split=split)
    assert not output.exists()
    report = run_saved_qa(config, source, "all", output,
                          run=True, generator=fake, expected_split=split)
    assert all(r["split"] == case["split"] for r in report["cases"])
    assert all(r["status"] == "generated" and r["reference_facts"] == facts
               for r in report["cases"])
    expected_methods = {"vector", "bm25", "hybrid", "rerank"} if methods is None else set(methods)
    assert {r["method"] for r in report["cases"]} == expected_methods
    assert len(report["cases"]) == len(expected_methods)
    before = output.read_bytes()
    result = prepare_qa_review(output, review_path, gold)
    assert all(r["review"]["fact_checks"] == [{"fact_id": "F1", "covered_correctly": None}]
               for r in result["cases"])
    assert output.read_bytes() == before

def test_retrieval_diagnostics_keep_final_results():
    from evaluation import summarize_retrieval
    identity = {"source_id": "demo", "content_hash": "hash", "pipeline_version": "v1"}
    chunks = [{**identity, "chunk_id": cid} for cid in ("X", "B", "Y", "D")]
    gold = {**identity, "collection": "demo", "id_field": "chunk_id",
            "cases": {"q": ["B", "D"]}}
    row = {"case_id": "q", "method": "vector", "answerable": True,
           "results": chunks[:3], "candidate_results": chunks,
           "evidence_page_hit": True, "annotated_page_rr": 0.5}
    report = {**identity, "collection": "demo", "top_k": 3, "candidate_k": 10,
              "required_block_gold": gold, "cases": [row]}
    add_block_scores(report, gold)
    before = json.dumps(report, sort_keys=True)
    result = summarize_retrieval(report)["vector"]
    assert result["required_block_recall"] == {"n": 1, "value": 0.5}
    assert result["candidate_required_block_recall"] == {"n": 1, "value": 1.0, "k": 10}
    assert result["annotated_page_hit_at_1"] == {"n": 1, "value": 0.0}
    assert json.dumps(report, sort_keys=True) == before

def test_answerability_summary():
    from evaluation import summarize_answerability
    specs = [(True, "generated", "refused", False),
             (True, "error", None, False),
             (False, "generated", "refused", None)]
    rows = [{"answerable": a, "status": s, "answer_status": d,
             "review": {"task_pass": p}} for a, s, d, p in specs]
    result = summarize_answerability(rows)
    assert result["answerable"]["refusal"] == {
        "refused_n": 1, "n": 1, "no_decision_n": 1, "value": 1.0}
    assert result["answerable"]["task_pass"]["value"] == 0.0
    assert result["answerable"]["error_n"] == 1
    assert result["unanswerable"]["task_pass"]["pending_n"] == 1
    assert result["unanswerable"]["task_pass"]["value"] is None
    assert summarize_answerability([])["answerable"]["refusal"]["value"] is None

@pytest.mark.parametrize("name,metric,expected", [
    ("task_pass", {"value": 0.0, "n": 1}, "0.00%"),
    ("required_block_mrr", {"value": 0.5, "n": 1}, "0.5000"),
    ("task_pass", {"value": None, "n": 0, "pending_n": 2}, "待复核"),
    ("task_pass", {"value": None, "n": 0, "pending_n": 0}, "暂无值"),
])
def test_metric_report_rendering(name, metric, expected):
    from evaluation import render_metric_table, METRIC_NOTES
    text = render_metric_table({"vector": {name: metric}}, (name,))
    assert f"| vector | {name} | {expected} | {metric['n']} |" in text
    assert f"| {metric.get('pending_n', '未单列')} |" in text
    assert METRIC_NOTES[name] in text

from evaluation import render_run_details
@pytest.mark.parametrize("usage,expected", [(None, "暂无记录"), ({"total_tokens": 0}, "0")])
def test_run_details_preserves_unknown_and_zero(usage, expected):
    row = {"status": "generated", "qa_seconds": None, "usage": usage, "review": {"task_pass": None}}
    md = render_run_details([row])
    assert f"`total_tokens`：已知合计{expected}；" in md
    assert "暂无有效记录" in md and "### " not in md