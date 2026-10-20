import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from retrieval import DEFAULT_TOP_K, MODEL_NAME, search_stored_chunks

def validate_case_semantics(case: dict, row_number: int) -> None:
    answerable = case.get("answerable") #.get(...) 从题目中取字段，.get 在字段缺失时返回 None，便于我们给出明确的校验错误，而不是立刻产生 KeyError。
    reference_answer = case.get("reference_answer")
    evidence = case.get("evidence")
    split = case.get("split")
    question_origin = case.get("question_origin")

    if not isinstance(answerable, bool):
        raise ValueError(f"第{row_number}条的 answerable 必须是布尔值")
    if split not in ("dev", "test"):
        raise ValueError(f"第{row_number}条的 split 必须是 dev 或 test") #split 只能是开发集 dev 或测试集 test
    if not isinstance(question_origin, str) or not question_origin.strip():
        raise ValueError(f"第{row_number}条缺少题目来源")

    if answerable: #把可回答题单独处理：它必须有非空参考答案，以及一个证据字典
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(f"第{row_number}条可回答题缺少参考答案")
        if not isinstance(evidence, dict):
            raise ValueError(f"第{row_number}条可回答题缺少证据")

        source_id = evidence.get("source_id")
        page = evidence.get("page")
        anchor = evidence.get("anchor")

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"第{row_number}条证据缺少 source_id")
        if type(page) is not int or page < 1:
            raise ValueError(f"第{row_number}条证据页码必须是正整数")
        if not isinstance(anchor, str) or not anchor.strip():
            raise ValueError(f"第{row_number}条证据缺少原文定位词")
    elif reference_answer is not None or evidence is not None: #elif处理不可回答题。它要求参考答案和证据都为 null，对应 Python 的 None。
        raise ValueError(f"第{row_number}条无答案题不应填写答案或证据")


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as case_file: #encoding="utf-8" 让中文按明确编码读取
        cases = json.load(case_file)

    if not isinstance(cases, list) or not cases: #isinstance 检查实际类型；not cases 还拒绝空列表，避免随后 cases[0] 越界
        raise ValueError("评测文件必须是非空列表")

    seen_ids = set() #创建集合，用来记住已出现的题号

    for row_number, case in enumerate(cases, start=1): #这里让报错能指出第几条样本
        if not isinstance(case, dict):
            raise ValueError(f"第{row_number}条不是对象")

        case_id = case.get("id")
        question = case.get("question")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"第{row_number}条缺少有效id")
        if case_id in seen_ids:
            raise ValueError(f"重复的样本id: {case_id}")
        seen_ids.add(case_id)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"第{row_number}条缺少有效问题")

        validate_case_semantics(case, row_number)

    return cases #保留完整数据


#先执行向量检索，得到 results → 再拿 results 与人工标注进行比较 → 记录 True / False
#检索只限定资料身份和文件哈希，再按相似度选取结果；它不知道人工期望页码。
def inspect_retrieval(case, model, source_id, content_hash, top_k=DEFAULT_TOP_K):
    results = search_stored_chunks(
        case["question"], model, source_id, content_hash, top_k=top_k,
    )
    evidence = case["evidence"]
    evidence_page_hit = None
    if case["answerable"]:
        evidence_page_hit = any( #any(...) 表示多个检索结果中，只要至少一条同时满足“来源相同”和“页码相同”，就返回 True，不要求所有结果都命中。
            item["source_id"] == evidence["source_id"]
            and item["page"] == evidence["page"] #and表示两个条件都要满足，才能算命中
            for item in results
        )
    record = {
        "case_id": case["id"],
        "retrieved_pages": [item["page"] for item in results],
        "evidence_page_hit": evidence_page_hit,
    }
    return record, results


def main():
    if len(sys.argv) not in (2, 5): #只传题集路径时长度为 2，仍然只校验题集；再传来源、哈希和报告路径时长度为 5，才进入批量检索。
        raise SystemExit("Usage: python evaluation.py <cases.json> [<source_id> <content_hash> <report.json>]")
    cases = load_cases(Path(sys.argv[1]))
    if len(sys.argv) == 2:
        print(f"样本数: {len(cases)}")
        print(f"首题: {cases[0]['question']}")
        return #防止校验模式继续加载模型
    source_id, content_hash, output_path = sys.argv[2:] #取出后三个参数，分别交给三个变量
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    top_k = DEFAULT_TOP_K
    records = []
    for case in cases:
        record, results = inspect_retrieval(case, model, source_id, content_hash, top_k=top_k)
        record["question"] = case["question"] #record 已有题号、命中页和页码检查结果；现在补上问题、可回答标注和分数
        record["answerable"] = case["answerable"]
        record["scores"] = [item["score"] for item in results] #分数列表与命中页按相同排名对应，便于分析
        record["generation_status"] = "not_run" #明确表示这次批量任务没有调用生成模型，不是生成失败
        records.append(record)
    report = {
        "embedding_model": MODEL_NAME,
        "source_id": source_id,
        "content_hash": content_hash,
        "top_k": top_k,
        "cases": records,
    }
    with Path(output_path).open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2) #json.dump 与此前的 json.load 相反，负责写出 JSON
        #ensure_ascii=False 保留可读中文，indent=2 使用两空格展开。with 会关闭文件，"w" 会覆盖同名文件，保留下一轮结果时应换一个报告文件名。
    print(f"Recorded {len(records)} cases: {output_path}")


if __name__ == "__main__":
    main()

#python .\projects\smart_cockpit_rag\evaluation.py .\projects\smart_cockpit_rag\eval_cases.json
#或切换到子文件夹：cd .\projects\smart_cockpit_rag 运行：python evaluation.py eval_cases.json