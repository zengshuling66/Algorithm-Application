import json
import sys
from pathlib import Path


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


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法: python evaluation.py <评测JSON路径>")

    cases = load_cases(Path(sys.argv[1]))
    print(f"样本数: {len(cases)}")
    print(f"首题: {cases[0]['question']}")


if __name__ == "__main__":
    main()

#python .\projects\smart_cockpit_rag\evaluation.py .\projects\smart_cockpit_rag\eval_cases.json
#或切换到子文件夹：cd .\projects\smart_cockpit_rag 运行：python evaluation.py eval_cases.json