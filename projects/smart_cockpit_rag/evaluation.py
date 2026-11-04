import json
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from retrieval import DEFAULT_TOP_K, MODEL_NAME, search_stored_chunks
from vector_store import load_stored_chunks, COLLECTION_NAME
from bm25_retrieval import build_bm25_search
from hybrid_retrieval import rrf_fuse, DEFAULT_CANDIDATE_K
from reranking import build_reranker, rerank_results, RERANK_MODEL_NAME, RERANK_MODEL_REVISION
from datetime import datetime, timezone
from time import perf_counter
from langchain_core.runnables import RunnableLambda
from langchain_adapter import results_to_documents
from langchain_query import build_qa_chain, generate_json_answer
from langchain_retrieval import build_hybrid_retriever, build_rerank_retriever

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
#检索只限定资料身份和文件哈希，再按相似度选取结果；它不知道人工期望页码
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


def score_retrieval(case, results):
    page_hit = None
    page_rr = None
    if case["answerable"]:
        evidence = case["evidence"]
        page_hit = False
        page_rr = 0.0
        for rank, item in enumerate(results, start=1):
        #上面inspect_retrieval的 any 只告诉我们是否命中；现在还要知道首次命中的排名，因此使用 enumerate(..., start=1)，让 rank 从1开始。
            if (item["source_id"] == evidence["source_id"]
                    and item["page"] == evidence["page"]):
                page_hit = True
                page_rr = 1.0 / rank #1.0 / rank 计算倒数排名
                break #停止循环，避免后面的重复页覆盖首次命中
            #可回答但未命中是 False、0.0；可回答但结果为空也一样；不可回答题是 None、None
    return {
        "case_id": case["id"],
        "retrieved_pages": [item["page"] for item in results],
        "retrieved_chunk_ids": [item["chunk_id"] for item in results],
        "evidence_page_hit": page_hit,
        "annotated_page_rr": page_rr,
    }


def compare_retrieval(cases, model, source_id, content_hash):
    if not cases or any(case["split"] != "dev" for case in cases):
        raise ValueError("Use a nonempty dev set")
    #分别复用库中正文读取、BM25、RRF及候选上限、精排及模型版本
    lexical = build_bm25_search(load_stored_chunks(source_id, content_hash))
    reranker = build_reranker()
    records = []
    for case in cases:
        question = case["question"]
        #向量与 BM25 各取最多10条，再融合取最多10条
        vector = search_stored_chunks(question, model, source_id, content_hash,
                                      top_k=DEFAULT_CANDIDATE_K)
        bm25 = lexical(question, top_k=DEFAULT_CANDIDATE_K)
        fused = rrf_fuse({"vector": vector, "bm25": bm25}, top_k=DEFAULT_CANDIDATE_K)
        #四种输出
        outputs = {
            "vector": vector[:DEFAULT_TOP_K],
            "bm25": bm25[:DEFAULT_TOP_K],
            "hybrid": fused[:DEFAULT_TOP_K],
            "rerank": rerank_results(question, fused, reranker, top_k=DEFAULT_TOP_K),
        }
        for method, results in outputs.items(): #.items() 每次取策略名和结果列表
            #评分后，update 补入问题、策略、完整结果等字段，append 收集记录
            record = score_retrieval(case, results) #计算标注页是否命中及首次命中的倒数排名
            record.update(method=method, question=question, answerable=case["answerable"],
                          results=results, generation_status="not_run")
            records.append(record)
    return {
        "source_id": source_id, "content_hash": content_hash,
        "collection": COLLECTION_NAME, "embedding_model": MODEL_NAME,
        "reranker_model": RERANK_MODEL_NAME, "reranker_revision": RERANK_MODEL_REVISION,
        "top_k": DEFAULT_TOP_K, "candidate_k": DEFAULT_CANDIDATE_K, "cases": records,
    }


#它只负责“拿必需行与实际结果比较”，不检索、不生成
def score_required_blocks(required_ids, results):
    required = set(required_ids) #set 去重
    if not required or len(required) != len(required_ids): #输入是非空、不重复的必需行ID列表和有序结果列表
        raise ValueError("Required block IDs must be nonempty and unique")
    found = {item["block_id"] for item in results}
    ranks = [rank for rank, item in enumerate(results, start=1) #ranks 收集必需行的排名，取第一个计算RR，完全未找到则为0
             if item["block_id"] in required]
    return {
        "required_block_recall": len(required & found) / len(required), #& 取交集
        "required_block_rr": 1.0 / ranks[0] if ranks else 0.0,
        "all_required_blocks_hit": required <= found, #集合的 <= 表示“左侧是否全部包含在右侧”，不是比较数量
    }


#读取两份题集并合并，分别是20道普通题+10道表格题，要求题号不重复、都是 dev、都来自同一资料和同一哈希。
def load_evaluation_inputs(config_path):
    config_path = Path(config_path).resolve() #Path(...) 接受路径字符串，.resolve() 转为绝对路径
    root = config_path.parent #.parent 取得配置所在文件夹
    config = json.loads(config_path.read_text(encoding="utf-8")) #read_text(...)先得到字符串，json.loads(...)再把字符串解析为Python对象

    cases = []
    seen_ids = set()
    for filename in config["case_files"]:
        for case in load_cases(root / filename): #root / filename 是拼接路径，因此配置内的文件名始终相对于配置所在目录解释
            if case["id"] in seen_ids:
                raise ValueError(f"Duplicate case ID: {case['id']}")
            if case["split"] != "dev":
                raise ValueError("Use dev cases only; keep test frozen")
            seen_ids.add(case["id"]) #这里的 seen_ids 跨文件共享，可以发现“两份文件分别合法，但题号相互重复”的情况
            cases.append(case) #append 保留完整题目，包括参考答案，不只保留问题

    if not cases:
        raise ValueError("At least one dev case is required")

    manual = json.loads(
        (root / config["manual_metadata"]).read_text(encoding="utf-8")
    )
    gold = json.loads(
        (root / config["required_block_gold"]).read_text(encoding="utf-8")
    )
    #循环分别比较资料身份、文件哈希，再检查标注集合是否等于当前查询集合，避免拿另一版本的标注评分。
    for key in ("source_id", "content_hash"):
        if not manual.get(key) or manual[key] != gold[key]:
            raise ValueError(f"Manual and block gold differ: {key}")
    if gold["collection"] != COLLECTION_NAME:
        raise ValueError("Block gold does not match the current collection")

    return cases, manual, gold

#四策略重新检索，组织完整流程，负责读取配置、加载模型、检索、评分，最后调用保存函数
# run_retrieval_evaluation
# ├─ load_evaluation_inputs → load_cases → validate_case_semantics
# ├─ compare_retrieval → score_retrieval
# ├─ add_block_scores → score_required_blocks
# ├─ summarize_retrieval
# └─ write_report
def run_retrieval_evaluation(config_path, output_path):
    output_path = Path(output_path)
    if output_path.exists(): #exists() 提前发现重名，避免白跑模型
        raise FileExistsError(f"Report already exists: {output_path}")

    cases, manual, gold = load_evaluation_inputs(config_path)
    model = SentenceTransformer(
        MODEL_NAME, device="cpu", local_files_only=True
    )
    #compare_retrieval内部建立 BM25、加载精排，并调用原向量/BM25/RRF/精排函数。
    report = compare_retrieval(cases, model, manual["source_id"], manual["content_hash"])
    #add_block_scores负责整份报告的范围核对和逐条补分；遇到有目标行标注的题，才调用score_required_blocks，计算行覆盖率、首次目标行RR及是否全部找齐
    report = add_block_scores(report, gold)
    #新增配置路径与 UTC 时间字符串，这里记录的是检索完成附近的时间，不是耗时。
    report["config_file"] = str(Path(config_path).resolve())
    report["run_at_utc"] = datetime.now(timezone.utc).isoformat()

    report["summary"] = summarize_retrieval(report) #把逐条评分按策略取平均
    write_report(output_path, report)

    print(f"Recorded {len(report['cases'])} rows: {output_path}")
    return report


#只负责把传入的报告写入文件，检索报告、评分报告等都可以复用它
def write_report(path, report, exclusive=True): #后续用 exclusive=False 更新这份文件
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as file:
    #写文件时的 "x" 表示“只创建新文件”，即使检查后出现同名文件，也不会覆盖
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        #ensure_ascii=False保留中文；indent=2指定缩进的空格数，便于阅读；allow_nan=False遇到NaN/Infinity会报错，避免写入不标准的JSON


#接收报告字典和标注字典，原地补充评分并返回报告
def add_block_scores(report, gold):
    for key in ("collection", "source_id", "content_hash"):
        if report[key] != gold[key]:
            raise ValueError(f"Report and block gold differ: {key}")

    for row in report["cases"]:
        if any(item[key] != gold[key] for item in row["results"] #内部 any 遍历“每条结果的三个字段”，任一不符就中止
               for key in ("source_id", "content_hash", "pipeline_version")):
            raise ValueError("Retrieved context does not match block gold")
        row.update(required_block_recall=None, required_block_rr=None,
                   all_required_blocks_hit=None) #没有标注的记录为 None，不是零分；先清空旧值，是防止重新评分时残留过期分数
        required = gold["cases"].get(row["case_id"]) #用 case_id 查标注，查到才调用已有的 score_required_blocks
        if required is not None:
            if not row["answerable"]:
                raise ValueError("Required blocks must label answerable cases")
            row.update(score_required_blocks(required, row["results"]))

    report["required_block_gold"] = gold
    return report


def summarize_retrieval(report):
    metrics = {
        "annotated_page_hit_rate": "evidence_page_hit", #标注页命中率：命中标注资料身份和页码的比例
        "annotated_page_mrr": "annotated_page_rr", #标注页 MRR：首次命中标注页的倒数排名，未命中为0
        "required_block_recall": "required_block_recall", #必需行召回率：检索结果中命中标注必需行的比例
        "required_block_mrr": "required_block_rr", #必需行 MRR：首次命中标注必需行的倒数排名，未命中为0
        "all_required_blocks_hit_rate": "all_required_blocks_hit", #必需行全命中率：检索结果中是否包含标注的所有必需行
    }
    summary = {}
    for method in ("vector", "bm25", "hybrid", "rerank"): #外层循环选出某种策略的 30 条记录
        rows = [row for row in report["cases"] if row["method"] == method]
        summary[method] = {"case_count": len(rows)}
        for name, key in metrics.items(): #内层 .items() 逐项取得指标名称和字段名
            values = [row[key] for row in rows if row.get(key) is not None] #关键是is not None：必须保留False和0.0，它们表示有效失败，不能排除
            summary[method][name] = {
                "n": len(values),
                "value": sum(values) / len(values) if values else None, #sum 会把布尔值按 1/0 累加
            }
    return summary


def evaluate_qa_row(row, generator=generate_json_answer):
    generated = {}

    def replay(question): #把保存的 results 转为 Document
        if question != row["question"]:
            raise ValueError("Question differs from saved context")
        return results_to_documents(row["results"])

    def capture(messages): #调用原生成函数，同时记住原始响应
        response = generator(messages)
        generated.update(response)
        return response

    started = perf_counter()
    try:
        #复用build_qa_chain执行问答，返回答案、引用、状态、原始响应和耗时
        result = build_qa_chain(RunnableLambda(replay), generator=capture).invoke(row["question"])
        #keep 只选可保存的字段，不保存整套 Document 对象
        keep = ("prompt_version", "status", "answer_status", "answer",
                "evidence_ids", "citations", "evidence_map", "model", "usage")
        record = {key: result[key] for key in keep}
    except Exception as exc: #异常记录为 error，不是拒答
        record = {"status": "error", "error_type": type(exc).__name__, #type(exc).__name__ 取异常类型
                  "http_status": getattr(exc, "status_code", None), #getattr(..., None) 在没有 HTTP 状态码时返回 None
                  "model": generated.get("model"), "usage": generated.get("usage")}
    return {**row, **record, "generation_status": record["status"],
            "raw_answer": generated.get("answer"),
            "qa_seconds": perf_counter() - started} #perf_counter() 的差值包括适配、Prompt、生成和校验，不包含此前检索


#负责读取、校验、选择和保存报告，调用 evaluate_qa_row 生成答案并记录结果
def run_saved_qa(config_path, source_path, case_id, output_path, run=False, generator=generate_json_answer):
    #复用输入函数阻止冻结题混入，复用 add_block_scores 核对资料及解析范围
    cases, _, gold = load_evaluation_inputs(config_path)
    saved = json.loads(Path(source_path).read_text(encoding="utf-8"))
    saved = add_block_scores(saved, gold)
    case_map = {case["id"]: case for case in cases} #case_map 按题号查题；wanted 表示选一题或全部开发题
    wanted = set(case_map) if case_id == "all" else {case_id}
    rows = [row for row in saved["cases"] if row["case_id"] in wanted]
    expected = {(cid, method) for cid in wanted for method in ("vector", "bm25", "hybrid", "rerank")}
    #expected 是“选中题号 × 四种策略”的组合集合
    if len(rows) != len(expected) or {(r["case_id"], r["method"]) for r in rows} != expected:
        raise ValueError("Expected four unique methods for each selected dev case")
    for row in rows:
        case = case_map[row["case_id"]]
        if (row["question"] != case["question"] or row["answerable"] != case["answerable"]
                or row["generation_status"] != "not_run"):
            raise ValueError("Input must match the dev cases and contain retrieval results only")
    print("records:", len(rows), "max_api_calls:", sum(bool(row["results"]) for row in rows))
    if not run:
        return
    if generator is generate_json_answer and not all(os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")):
        raise RuntimeError("Configure all three LLM environment variables first")
    report = {**saved, "report_type": "qa", "run_status": "running", "cases": [],
              "qa_run_at_utc": datetime.now(timezone.utc).isoformat(),
              "base_url": os.environ.get("LLM_BASE_URL"), "requested_model": os.environ.get("LLM_MODEL"),
              "retrieval_report": str(Path(source_path).resolve()), "expected_records": len(rows)}
    report.pop("summary", None)
    write_report(output_path, report)
    for row in rows:
        record = evaluate_qa_row(row, generator=generator)
        case = case_map[row["case_id"]]
        record.update(split="dev", reference_answer=case["reference_answer"], reference_evidence=case["evidence"])
        report["cases"].append(record)
        write_report(output_path, report, exclusive=False)
        print(row["case_id"], row["method"], record["status"])
    report["run_status"] = "complete"
    write_report(output_path, report, exclusive=False)
    return report


#它接收两个路径，读取完整问答报告，给每条记录加入空的review字段，再保存新副本；它准备评分表，评分仍需人工填写。
def prepare_qa_review(source_path, output_path):
    report = json.loads(Path(source_path).read_text(encoding="utf-8")) #json.loads 创建内存对象
    if (report.get("report_type") != "qa"
            or report.get("run_status") != "complete"
            or len(report["cases"]) != report["expected_records"]):
        raise ValueError("Use a complete QA report")
    for row in report["cases"]: #循环给每条记录新增 review，None 写入 JSON 后是 null
        row["review"] = {
            "task_pass": None, "faithful": None,
            "citation_pass": None, "refusal_appropriate": None,
            "failure_types": [], "notes": "",
        }
        #task_pass表示整题是否完成：可回答题完整正确；无答案题正确拒答
        #faithful表示实际陈述的事实是否忠实于提供的上下文
        #citation_pass表示事实是否得到所引证据支持，且没有漏引关键事实
        #refusal_appropriate表示该拒答时是否拒答、实际拒答是否合理；正常可回答且已回答时不适用
        #failure_types表示失败原因标签及具体解释
    report["report_type"] = "qa_review"
    report["qa_report"] = str(Path(source_path).resolve())
    write_report(output_path, report)
    print("Review rows:", len(report["cases"]))
    return report


#读取已填写的review，计算任务通过率、忠实度、引用与拒答指标，以及失败标签计数
def summarize_qa_review(report):
    if report.get("report_type") != "qa_review":
        raise ValueError("Use a QA review report")
    metrics = ("task_pass", "faithful", "citation_pass", "refusal_appropriate")
    if any(type(row["review"]["task_pass"]) is not bool for row in report["cases"]):
        raise ValueError("Finish task_pass review for every row")
    summary = {}
    for method in ("vector", "bm25", "hybrid", "rerank"):
        rows = [row for row in report["cases"] if row["method"] == method]
        item = {"case_count": len(rows)}
        for key in metrics:
            values = [row["review"][key] for row in rows
                      if row["review"][key] is not None]
            if any(type(value) is not bool for value in values):
                raise ValueError("Review scores must be true, false or null")
            item[key] = {"n": len(values),
                         "value": sum(values) / len(values) if values else None}
                        #布尔值求和时按 True=1、False=0，因此平均值就是通过比例，n 是适用记录数
        counts = {}
        for row in rows:
            for tag in set(row["review"]["failure_types"]): #set(...) 让同一条记录的同类失败只计一次
                counts[tag] = counts.get(tag, 0) + 1 #counts.get(tag, 0) + 1 在首次出现时从零开始累加
        item["failure_counts"] = counts #多标签可以同时计数
        summary[method] = item
    return summary


def benchmark_retrieval(config_path, output_path):
    if Path(output_path).exists():
        raise FileExistsError(f"Report already exists: {output_path}")
    cases, manual, _ = load_evaluation_inputs(config_path)
    started = perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    setup_seconds = {"embedding": perf_counter() - started} #记录模型加载时间
    retrievers = {}
    for method, builder in (("hybrid", build_hybrid_retriever),
                            ("rerank", build_rerank_retriever)):
        started = perf_counter()
        retrievers[method] = builder(model, manual["source_id"], manual["content_hash"])
        setup_seconds[method] = perf_counter() - started #记录检索器构建时间
    for retriever in retrievers.values():
        retriever.invoke(cases[0]["question"]) #各执行一次不计时的检索作为预热
    report = {
        "report_type": "retrieval_benchmark", "collection": COLLECTION_NAME,
        "embedding_model": MODEL_NAME, "reranker_model": RERANK_MODEL_NAME,
        "reranker_revision": RERANK_MODEL_REVISION, "top_k": DEFAULT_TOP_K,
        "candidate_k": DEFAULT_CANDIDATE_K, "device": "cpu", "repeats": 3,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "setup_seconds": setup_seconds, "cases": [],
    }
    for repeat in range(1, 4): #左闭右开
        for case in cases:
            for method, retriever in retrievers.items():
                started = perf_counter()
                documents = retriever.invoke(case["question"])
                seconds = perf_counter() - started
                #正式 seconds 覆盖问题编码、BM25、RRF、文档转换和可选精排，不含加载、预热、LLM 或 HTTP，不是整个问答系统的延迟
                report["cases"].append({
                    "case_id": case["id"], "method": method,
                    "repeat": repeat, "seconds": seconds,
                    "chunk_ids": [doc.id for doc in documents],
                    "pages": [doc.metadata["page"] for doc in documents],
                })
    write_report(output_path, report)
    print(f"Benchmark rows: {len(report['cases'])}")
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark": #本地检索计时，不生成答案、不评价答案正确性
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python evaluation.py benchmark <config.json> <report.json>")
        benchmark_retrieval(sys.argv[2], sys.argv[3])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "review": #创建待填写review的新副本，不自动判答案
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python evaluation.py review <qa.json> <output.json>")
        prepare_qa_review(sys.argv[2], sys.argv[3])
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "qa": #回放已有上下文；加--run才真实生成
        if len(sys.argv) not in (6, 7) or (len(sys.argv) == 7 and sys.argv[6] != "--run"):
            raise SystemExit("Usage: python evaluation.py qa <config> <retrieval> <case_id|all> <output> [--run]")
            #四个位置参数依次是配置、检索报告、题号、输出路径，只有最后显式添加 --run，才将 run 设为 True
        run_saved_qa(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], run=len(sys.argv) == 7)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "summary": #汇总已有评分，不重新检索或生成
        if len(sys.argv) != 3:
            raise SystemExit("Usage: python evaluation.py summary <report.json>")
        report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        if report.get("report_type") == "qa_review":
            summary = summarize_qa_review(report)
        else:
            summary = summarize_retrieval(report)
        print(json.dumps(summary, ensure_ascii=False, indent=2)) #json.dumps 转为便于打印的字符串
        return

    if len(sys.argv) > 1 and sys.argv[1] == "score": #给已有检索报告补目标行评分，不重新检索
        if len(sys.argv) != 5:
            raise SystemExit(
                "Usage: python evaluation.py score <config.json> <input.json> <output.json>"
            )
        _, _, gold = load_evaluation_inputs(sys.argv[2])
        report = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        report = add_block_scores(report, gold)
        report["summary"] = summarize_retrieval(report)
        write_report(sys.argv[4], report)
        count = sum(row["required_block_recall"] is not None for row in report["cases"]) #sum 把每个“评分非空”的 True 计作 1
        print("Rows with block labels:", count)
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare": #四策略重新检索、评分、保存报告
        if len(sys.argv) != 4:
            raise SystemExit(
                "Usage: python evaluation.py compare <config.json> <report.json>"
            )
        run_retrieval_evaluation(sys.argv[2], sys.argv[3])
        return
    
    if len(sys.argv) not in (2, 5): #只传题集路径时长度为 2，仍然只校验题集；再传来源、哈希和报告路径时长度为 5，才进入批量检索。
        raise SystemExit("Usage: python evaluation.py <cases.json> [<source_id> <content_hash> <report.json>]")
    cases = load_cases(Path(sys.argv[1])) #校验单个题集，或执行旧单路检索

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