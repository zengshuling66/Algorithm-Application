import json
import sys
import os
import numpy as np
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
from hashlib import sha256
from langchain_prompt import STRUCTURED_PROMPT_VERSION

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


#从候选截取结果、重新计算页指标
def select_retrieval_top_k(report, cases, top_k):
    limit = report.get("candidate_k")
    if type(limit) is not int or type(top_k) is not int or not 0 < top_k <= limit:
        raise ValueError("top_k must be an integer from 1 to saved candidate_k")
    case_map = {case["id"]: case for case in cases}
    expected = {(cid, m) for cid in case_map for m in ("vector", "bm25", "hybrid", "rerank")}
    actual = [(row["case_id"], row["method"]) for row in report["cases"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Saved rows must contain every dev case and method exactly once")
    selected = {**report, "top_k": top_k, "cases": []}
    for row in report["cases"]:
        case = case_map[row["case_id"]]
        if row["question"] != case["question"] or row["answerable"] != case["answerable"] or row.get("generation_status") != "not_run":
            raise ValueError("Use matching retrieval rows without generation")
        candidates = row.get("candidate_results")
        if not isinstance(candidates, list):
            raise ValueError("Saved candidate_results required; rerun compare")
        if len(candidates) > limit or len({item["chunk_id"] for item in candidates}) != len(candidates):
            raise ValueError("Invalid saved candidate list")
        results = candidates[:top_k]
        record = score_retrieval(case, results)
        record.update(method=row["method"], question=case["question"], answerable=case["answerable"],
                      candidate_results=candidates, results=results, generation_status="not_run")
        selected["cases"].append(record)
    selected.pop("summary", None)
    selected["config_snapshot"] = {**report.get("config_snapshot", {}), "top_k": top_k}
    return selected


#每路取10条 → 融合取10条 → 精排保留10条 → 保存到 candidate_results → 截取前K条到 results
def compare_retrieval(cases, model, source_id, content_hash, top_k=DEFAULT_TOP_K, expected_split="dev"):
    if (expected_split not in ("dev", "test") or not cases
            or any(case["split"] != expected_split for case in cases)):
        raise ValueError("Use a nonempty set matching expected_split")
    if type(top_k) is not int or not 0 < top_k <= DEFAULT_CANDIDATE_K:
        raise ValueError("top_k must be an integer from 1 to candidate_k")
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
            "vector": vector,
            "bm25": bm25,
            "hybrid": fused,
            "rerank": rerank_results(question, fused, reranker, top_k=DEFAULT_CANDIDATE_K),
        }
        for method, candidates in outputs.items(): #.items() 每次取策略名和结果列表
            #评分后，update 补入问题、策略、完整结果等字段，append 收集记录
            results = candidates[:top_k]
            record = score_retrieval(case, results) #计算标注页是否命中及首次命中的倒数排名
            record.update(method=method, question=question, answerable=case["answerable"],
                          candidate_results=candidates, results=results, generation_status="not_run")
            records.append(record)
    return {
        "source_id": source_id, "content_hash": content_hash,
        "collection": COLLECTION_NAME, "embedding_model": MODEL_NAME,
        "reranker_model": RERANK_MODEL_NAME, "reranker_revision": RERANK_MODEL_REVISION,
        "top_k": top_k, "candidate_k": DEFAULT_CANDIDATE_K, "cases": records, "split": expected_split,
    }


#它只负责“拿必需行与实际结果比较”，不检索、不生成
def score_required_blocks(required_ids, results, id_field="block_id"):
    if id_field not in ("block_id", "chunk_id"):
        raise ValueError("Use block_id or chunk_id")
    required = set(required_ids) #set 去重
    if not required or len(required) != len(required_ids): #输入是非空、不重复的必需行ID列表和有序结果列表
        raise ValueError("Required block IDs must be nonempty and unique")
    found = {item[id_field] for item in results}
    ranks = [rank for rank, item in enumerate(results, start=1) #ranks 收集必需行的排名，取第一个计算RR，完全未找到则为0
             if item[id_field] in required]
    return {
        "required_block_recall": len(required & found) / len(required), #& 取交集计算覆盖比例
        "required_block_rr": 1.0 / ranks[0] if ranks else 0.0,
        "all_required_blocks_hit": required <= found, #集合的 <= 表示“左侧是否全部包含在右侧”，不是比较数量
    }


#按配置读取题集，要求题号不重复、都来自同一资料和同一哈希。
def load_evaluation_inputs(config_path, expected_split="dev"):
    config_path = Path(config_path).resolve() #Path(...) 接受路径字符串，.resolve() 转为绝对路径
    root = config_path.parent #.parent 取得配置所在文件夹
    config = json.loads(config_path.read_text(encoding="utf-8")) #read_text(...)先得到字符串，json.loads(...)再把字符串解析为Python对象
    if expected_split not in ("dev", "test"):
        raise ValueError("expected_split must be dev or test")
#expected_split 是调用者要求读取的集合类型；配置中的 split 是文件声明，两者必须一致；有 split 就取实际值，没有这个键才返回 "dev"
    if config.get("split", "dev") != expected_split: 
        raise ValueError("Config split differs from expected_split")

    cases = []
    seen_ids = set()
    for filename in config["case_files"]:
        for case in load_cases(root / filename): #root / filename 是拼接路径，因此配置内的文件名始终相对于配置所在目录解释
            if case["id"] in seen_ids:
                raise ValueError(f"Duplicate case ID: {case['id']}")
            if case["split"] != expected_split:
                raise ValueError(f"Expected {expected_split} cases only")
            seen_ids.add(case["id"]) #这里的 seen_ids 跨文件共享，可以发现“两份文件分别合法，但题号相互重复”的情况
            cases.append(case) #append 保留完整题目，包括参考答案，不只保留问题

    if not cases:
        raise ValueError("At least one case is required")

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

    #依次核对：题集文件名一致、文件字节指纹一致、金标题号恰好覆盖全部可回答题
    if expected_split == "test":
        if config["case_files"] != [gold["case_file"]]:
            raise ValueError("Test case file differs from block gold")
        case_bytes = (root / gold["case_file"]).read_bytes()
        if sha256(case_bytes).hexdigest() != gold["case_file_sha256"]:
            raise ValueError("Test case file fingerprint differs from block gold")
        positive_ids = {case["id"] for case in cases if case["answerable"]}
        if set(gold["cases"]) != positive_ids:
            raise ValueError("Test block gold must cover every answerable case")

    return cases, manual, gold

#四策略重新检索，组织完整流程：配置读取 K → 提前校验 → 传入 compare_retrieval → 按 K 截取 → 使用新金标评分 → 保存
# run_retrieval_evaluation
# ├─ load_evaluation_inputs → load_cases → validate_case_semantics
# ├─ compare_retrieval → score_retrieval
# ├─ add_block_scores → score_required_blocks
# ├─ summarize_retrieval
# └─ write_report
def run_retrieval_evaluation(config_path, output_path, expected_split="dev"):
    output_path = Path(output_path)
    if output_path.exists(): #exists() 提前发现重名，避免白跑模型
        raise FileExistsError(f"Report already exists: {output_path}")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    top_k = config.get("top_k", DEFAULT_TOP_K)
    if type(top_k) is not int or not 0 < top_k <= DEFAULT_CANDIDATE_K:
        raise ValueError("top_k must be an integer from 1 to candidate_k")

    cases, manual, gold = load_evaluation_inputs(config_path, expected_split=expected_split)
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    #compare_retrieval内部建立 BM25、加载精排，并调用原向量/BM25/RRF/精排函数。
    report = compare_retrieval(cases, model, manual["source_id"], manual["content_hash"], top_k=top_k, expected_split=expected_split)
    #add_block_scores负责整份报告的范围核对和逐条补分；遇到有目标行标注的题，才调用score_required_blocks，计算行覆盖率、首次目标行RR及是否全部找齐
    report = add_block_scores(report, gold)
    #新增配置路径与 UTC 时间字符串，这里记录的是检索完成附近的时间，不是耗时。
    report["config_file"] = str(Path(config_path).resolve())
    report["config_snapshot"] = config #保存当时的配置内容，避免以后修改配置文件后无法追溯
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
        #没有标注的记录为 None，不是零分；先清空旧值，是防止重新评分时残留过期分数
        row.update(required_block_recall=None, required_block_rr=None,
                   all_required_blocks_hit=None, selected_evidence_set=None)
        required = gold["cases"].get(row["case_id"]) #用 case_id 查标注，查到才调用已有的 score_required_blocks
        if required is not None:
            if not row["answerable"]:
                raise ValueError("Required blocks must label answerable cases")
            alternatives = gold.get("alternative_evidence_sets", {}).get(row["case_id"], [])
            options = [required] + alternatives #options 是“标注的必需行 + 备选必需行”，每个都是列表
            scores = [ #scores 保存逐套评分
                score_required_blocks(ids, row["results"], id_field=gold.get("id_field", "block_id"))
                for ids in options
            ]
            best = max(range(len(scores)), key=lambda i: ( #best 选出方案编号：先比较覆盖率，再比较首次命中 RR，完全相同时用 -i 优先选择编号较小的方案
                scores[i]["required_block_recall"], scores[i]["required_block_rr"], -i,
            ))
            row.update(scores[best]) #将选出方案的整组指标写回记录
            row["selected_evidence_set"] = best if scores[best]["required_block_recall"] > 0 else None
            #`selected_evidence_set` 中，0表示主方案，1表示第一套替代方案；完全未命中或没有金标时为 None

    report["evidence_scoring_policy"] = "best_set_recall_rr_v1"
    report["required_block_gold"] = gold
    return report


def summarize_retrieval(report):
    records = report["cases"]
    if "required_block_gold" in report and all("candidate_results" in r for r in records):
        limit = report.get("candidate_k")
        if type(limit) is not int or limit <= 0 or any(len(r["candidate_results"]) > limit for r in records):
            raise ValueError("Invalid candidate_k or candidate_results")
        view = {**report, "cases": [{**r, "results": r["candidate_results"]} for r in records]}
        add_block_scores(view, report["required_block_gold"])
        records = [
            {**r, "candidate_required_block_recall": c["required_block_recall"]}
            for r, c in zip(records, view["cases"])
        ]
    # 汇总指标名 -> 单题记录字段；各指标仅对非 None 的记录取平均。
    # TopK 指最终检索结果；RR = 首次命中排名的倒数，未命中为 0。
    metrics = {
        # 页级诊断：匹配资料身份和标注页，不代表答案证据已经找齐。
        "annotated_page_hit_rate": "evidence_page_hit",  # TopK 至少命中一次标注页的题目比例
        "annotated_page_hit_at_1": "annotated_page_rr",  # 第一条结果命中标注页的题目比例，由 RR == 1 判断
        "annotated_page_mrr": "annotated_page_rr",  # 各题标注页 RR 的平均值，反映首次命中位置

        # 块级评估：必要证据块包含正文块或表格行，沿用金标主/替代方案的计分规则。
        "required_block_recall": "required_block_recall",  # 各题“命中必要块数 / 金标必要块数”的平均值
        "all_required_blocks_hit_rate": "all_required_blocks_hit",  # TopK 找齐全部必要证据块的题目比例
        "required_block_mrr": "required_block_rr",  # 各题必要证据块 RR 的平均值，只看首次命中位置

        # 候选阶段诊断：观察最终截取 TopK 之前，候选结果中的必要证据覆盖情况。
        "candidate_required_block_recall": "candidate_required_block_recall",  # 候选必要块召回率的题目平均值
    }
    summary = {}
    for method in ("vector", "bm25", "hybrid", "rerank"): #外层循环选出某种策略的 30 条记录
        rows = [row for row in records if row["method"] == method]
        summary[method] = {"case_count": len(rows)}
        for name, key in metrics.items(): #内层 .items() 逐项取得指标名称和字段名
            values = [row[key] for row in rows if row.get(key) is not None] #关键是is not None：必须保留False和0.0，它们表示有效失败，不能排除
            if name == "annotated_page_hit_at_1":
                values = [value == 1.0 for value in values]
            summary[method][name] = {
                "n": len(values),
                "value": sum(values) / len(values) if values else None, #sum 会把布尔值按 1/0 累加
            }
            if name == "candidate_required_block_recall":
                summary[method][name]["k"] = report.get("candidate_k")
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
def run_saved_qa(config_path, source_path, case_id, output_path, run=False, generator=generate_json_answer, expected_split="dev"):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    supported = ["vector", "bm25", "hybrid", "rerank"]
    methods = config.get("qa_methods", supported)
    if (not isinstance(methods, list) or not methods
            or any(m not in supported for m in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("qa_methods must be a nonempty list of unique supported methods")
    #复用输入函数阻止冻结题混入，复用 add_block_scores 核对资料及解析范围
    cases, _, gold = load_evaluation_inputs(config_path, expected_split=expected_split)
    saved = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if saved.get("split", "dev") != expected_split:
        raise ValueError("Retrieval split differs from expected_split")
    saved = add_block_scores(saved, gold)
    case_map = {case["id"]: case for case in cases} #case_map 按题号查题；wanted 表示选一题或当前题集全部题
    wanted = set(case_map) if case_id == "all" else {case_id}
    rows = [row for row in saved["cases"]
            if row["case_id"] in wanted and row["method"] in methods]
    #expected 是“选中题号 × 配置指定策略”的组合集合
    expected = {(cid, method) for cid in wanted for method in methods}
    if len(rows) != len(expected) or {(r["case_id"], r["method"]) for r in rows} != expected:
        raise ValueError("Expected configured methods exactly once for each selected case")
    for row in rows:
        case = case_map[row["case_id"]]
        if (row["question"] != case["question"] or row["answerable"] != case["answerable"]
                or row["generation_status"] != "not_run"):
            raise ValueError("Input must match the cases and contain retrieval results only")
    print("records:", len(rows), "max_api_calls:", sum(bool(row["results"]) for row in rows))
    if not run:
        return
    if generator is generate_json_answer and not all(os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")):
        raise RuntimeError("Configure all three LLM environment variables first")
    report = {**saved, "report_type": "qa", "run_status": "running", "cases": [],
              "qa_run_at_utc": datetime.now(timezone.utc).isoformat(),
              "base_url": os.environ.get("LLM_BASE_URL"), "requested_model": os.environ.get("LLM_MODEL"),
              "retrieval_report": str(Path(source_path).resolve()), "expected_records": len(rows)}
    report["qa_methods"] = methods
    report.pop("summary", None)
    write_report(output_path, report)
    for row in rows:
        record = evaluate_qa_row(row, generator=generator)
        case = case_map[row["case_id"]]
        record.update(split=case["split"], reference_answer=case["reference_answer"], reference_evidence=case["evidence"], reference_facts=case.get("reference_facts", []))
        report["cases"].append(record)
        write_report(output_path, report, exclusive=False)
        print(row["case_id"], row["method"], record["status"])
    report["run_status"] = "complete"
    write_report(output_path, report, exclusive=False)
    return report


#读取完整QA与可选事实金标，创建待复核副本，不自动评分
#核对QA完整性及金标来源 → 按题复制事实、建立空检查项 → 保存新副本与金标指纹
def prepare_qa_review(source_path, output_path, gold_path=None):
    report = json.loads(Path(source_path).read_text(encoding="utf-8")) #json.loads 创建内存对象
    if (report.get("report_type") != "qa"
            or report.get("run_status") != "complete"
            or len(report["cases"]) != report["expected_records"]):
        raise ValueError("Use a complete QA report")
    raw_gold = Path(gold_path).read_bytes() if gold_path is not None else None
    gold = json.loads(raw_gold) if raw_gold is not None else {}
    identity = ("collection", "source_id", "content_hash")
    for key in identity + ("pipeline_version", "case_files_sha256", "case_file_sha256"):
        origin = report if key in identity else report.get("required_block_gold", {})
        if key in gold and origin.get(key) != gold[key]:
            raise ValueError(f"QA and review gold differ: {key}")
    fact_map = gold.get("reference_facts", {})
    for row in report["cases"]: #循环给每条记录新增 review，None 写入 JSON 后是 null
        #优先取独立金标中该题的事实；没有该题时，取QA报告保存的事实；两处都没有才取空列表
        facts = fact_map.get(row["case_id"], row.get("reference_facts", []))
        if row["answerable"] and raw_gold is not None and not facts:
            raise ValueError(f"Missing reference facts: {row['case_id']}")
        if not row["answerable"] and facts:
            raise ValueError("Unanswerable cases must not have positive reference facts")
        if len({fact["id"] for fact in facts}) != len(facts):
            raise ValueError("Duplicate reference fact IDs")
        row["reference_facts"] = [dict(fact) for fact in facts]
        row["review"] = {
            "task_pass": None, "faithful": None,
            "citation_pass": None, "refusal_appropriate": None,
            "failure_types": [], "notes": "",
            "chunk_relevance": [{"chunk_id": item["chunk_id"], "relevant": None}
                                for item in row["results"]] if row["answerable"] else None,
            "fact_checks": [{"fact_id": fact["id"], "covered_correctly": None}
                            for fact in facts] if row["answerable"] and facts else None,
        }
        #task_pass表示整题是否完成：可回答题完整正确；无答案题正确拒答
        #faithful表示实际陈述的事实是否忠实于提供的上下文
        #citation_pass表示事实是否得到所引证据支持，且没有漏引关键事实
        #refusal_appropriate表示该拒答时是否拒答、实际拒答是否合理；正常可回答且已回答时不适用
        #failure_types表示失败原因标签及具体解释
        #chunk_relevance判断每块是否有助于回答问题
        #fact_checks判断答案是否正确覆盖必要事实，不是判断检索文本里有没有这些事实
    report.update(report_type="qa_review", review_schema_version="qa-review-v2",
                  qa_report=str(Path(source_path).resolve()))
    if raw_gold is not None:
        report["review_gold_file"] = str(Path(gold_path).resolve())
        report["review_gold_sha256"] = sha256(raw_gold).hexdigest()
    report.pop("summary", None) #移除旧汇总
    write_report(output_path, report)
    print("Review rows:", len(report["cases"]))
    return report


#读取已经填写的标签，计算单题Precision@K和事实覆盖率
def score_review_metrics(row, top_k):
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not row["answerable"]:
        return {"precision_at_k": None, "key_fact_coverage": None}
    def fraction(checks, expected_ids, id_key, value_key, denominator):
        expected = set(expected_ids)
        if len(expected) != len(expected_ids) or len(expected) > denominator:
            raise ValueError("Invalid expected IDs or denominator")
        if not isinstance(checks, list) or any(
                not isinstance(c, dict) or id_key not in c or value_key not in c for c in checks):
            raise ValueError("Use a complete check list")
        ids = [c[id_key] for c in checks]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError("Check IDs must match exactly without duplicates")
        values = [c[value_key] for c in checks]
        if any(v is not None and type(v) is not bool for v in values):
            raise ValueError("Check values must be true, false or null")
        return None if any(v is None for v in values) else sum(values) / denominator
    review = row["review"]
    #Precision@K：相关块数÷K。例如K=5，只返回2块且都相关，得分是40%。
    precision = fraction(review["chunk_relevance"], [r["chunk_id"] for r in row["results"]],
                         "chunk_id", "relevant", top_k)
    #关键事实覆盖率：正确覆盖事实数÷必要事实数。例如2项覆盖1项，得分是50%。
    facts = row.get("reference_facts", [])
    coverage = fraction(review["fact_checks"], [f["id"] for f in facts],
                        "fact_id", "covered_correctly", len(facts)) if facts else None
    #内部 fraction 是两项指标共用的比例计算函数：先检查评分表的ID有没有漏项、重复或错项，再检查值是否为布尔值或 None，最后计算比例。
    return {"precision_at_k": precision, "key_fact_coverage": coverage}


def summarize_answerability(rows):
    summary = {}
    for name, answerable in (("answerable", True), ("unanswerable", False)):
        group = [r for r in rows if r["answerable"] == answerable]
        values = [r["review"]["task_pass"] for r in group
                  if r["review"]["task_pass"] is not None] #values保留有效的`False`，只排除`None`
        decisions = [r.get("answer_status") for r in group
                     if r.get("status") in ("generated", "no_evidence") #`no_evidence`表示正常缺证据拒答，因此可以计入有效决策
                     and r.get("answer_status") in ("answered", "refused")]
        refused = decisions.count("refused") #统计拒答次数
        summary[name] = {
            "case_count": len(group),
            "task_pass": { #`task_pass.n`是已复核题数
                "passed_n": sum(values), "n": len(values),
                "pending_n": len(group) - len(values),
                "value": sum(values) / len(values) if values else None,
            },
            "refusal": { #`refusal.n`是有效决策数
                "refused_n": refused, "n": len(decisions),
                "no_decision_n": len(group) - len(decisions),
                "value": refused / len(decisions) if decisions else None,
            },
            "error_n": sum(r.get("status") == "error" for r in group),
        }
    return summary


#读取已填写的review，计算任务通过率、忠实度、引用与拒答指标，以及失败标签计数
def summarize_qa_review(report):
    if report.get("report_type") != "qa_review":
        raise ValueError("Use a QA review report")
    schema = report.get("review_schema_version")
    if schema not in (None, "qa-review-v2"):
        raise ValueError("Unsupported review schema")
    is_v2 = schema == "qa-review-v2"
    if is_v2 and (type(report.get("top_k")) is not int or report["top_k"] <= 0):
        raise ValueError("top_k must be a positive integer")
    metrics = ("task_pass", "faithful", "citation_pass", "refusal_appropriate")
    if not is_v2 and any(type(row["review"]["task_pass"]) is not bool for row in report["cases"]):
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
        if is_v2:
            item["answerability"] = summarize_answerability(rows)
            item["task_pending_n"] = sum(r["review"]["task_pass"] is None for r in rows)
            positives = [r for r in rows if r["answerable"]]
            scored = [score_review_metrics(r, report["top_k"]) for r in positives]
            for name in ("precision_at_k", "key_fact_coverage"):
                applicable = len(positives) if name == "precision_at_k" else sum(
                    bool(r.get("reference_facts")) for r in positives)
                values = [s[name] for s in scored if s[name] is not None]
                item[name] = {
                    "applicable_n": applicable, "n": len(values), #applicable_n 是适用题数, n是已完整复核的题数
                    "pending_n": applicable - len(values), #pending_n 是适用题数减去已复核题数，表示还有多少题待复核
                    "value": sum(values) / len(values) if values else None, #`value` 只对已评题取平均，保留有效的0分
                }
            item["precision_at_k"]["k"] = report["top_k"]
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


#证据预览
def preview_case_chunks(cases_path, case_id):
    cases_path = Path(cases_path).resolve()
    cases = load_cases(cases_path) #继续负责原有基础校验
    selected = [case for case in cases if case["id"] == case_id] #列表推导式筛出指定题号
    if not selected:
        raise ValueError(f"Unknown case: {case_id}")
    case = selected[0] #`selected[0]`取出题目字典
    if not case["answerable"]: #不可回答题不在这里建立正向证据金标，因此提前拒绝；这不代表它在库中没有相关片段
        raise ValueError("Use an answerable case")
    manual_path = cases_path.parent / "manual_metadata.json" #`.parent / "manual_metadata.json"取得题集同目录的配置
    manual = json.loads(manual_path.read_text(encoding="utf-8")) #read_text()`读字符串，`json.loads()`转字典
    if case["evidence"]["source_id"] != manual["source_id"]:
        raise ValueError("Case and manual source_id differ")
    pages = set(case["evidence_pages"]) #`pages`是证据页集合
    chunks = load_stored_chunks(manual["source_id"], manual["content_hash"])
    #`load_stored_chunks()`按资料身份和哈希读取当前集合的payload
    rows = [chunk for chunk in chunks if chunk["page"] in pages]
    if not pages or pages - {row["page"] for row in rows}:
    #`pages - 已读到的页集合`是差集；有差集说明某些标注页没有记录，应先排查，不能默默继续
        raise ValueError("Some evidence pages are missing")
    rows.sort(key=lambda row: (row["page"], row["chunk_id"])) #按页码和块号排序，便于人工检查
    print(json.dumps(rows, ensure_ascii=False, indent=2)) #`json.dumps()`将列表转成可读JSON，保留中文
    return rows


METRIC_NOTES = {
    "annotated_page_hit_rate": "最终TopK至少命中一次原标注页的题目比例。",
    "annotated_page_hit_at_1": "第一条结果命中原标注页的题目比例。",
    "annotated_page_mrr": "各题首次命中原标注页的RR均值；未命中为0。",
    "required_block_recall": "必要块命中数除以金标必要块数，再按题平均。",
    "all_required_blocks_hit_rate": "最终TopK找齐全部必要证据块的题目比例。",
    "required_block_mrr": "各题首次命中必要证据块的RR均值；未命中为0。",
    "candidate_required_block_recall": "最终截取TopK前，候选必要块召回率的题目平均值。",
    "task_pass": "已复核题中的任务通过比例；不可回答题要求正确拒答。",
    "faithful": "适用且已复核的答案中，实际陈述忠于上下文的通过比例。",
    "citation_pass": "适用且已复核的答案中，引用支持相应陈述的通过比例。",
    "refusal_appropriate": "适用且已复核记录中的拒答合理性通过比例。",
    "precision_at_k": "相关返回块数除以K，再按可回答题平均；缺位不算相关。",
    "key_fact_coverage": "答案正确覆盖事实数除以金标事实数，再按题平均。",
}


def format_metric_value(name, metric):
    value = metric["value"]
    if value is None:
        return "待复核" if metric.get("pending_n", 0) > 0 else "暂无值"
    return f"{value:.4f}" if name.endswith("_mrr") else f"{value:.2%}"


def render_metric_table(summary, names):
    lines = ["| 策略 | 指标 | 数值 | 已评题数 | 待评题数 |",
             "| --- | --- | --- | --- | --- |"]
    for method, scores in summary.items():
        for name in names:
            metric = scores[name]
            value = format_metric_value(name, metric)
            pending = metric.get("pending_n", "未单列")
            lines.append(f"| {method} | {name} | {value} | {metric['n']} | {pending} |")
    lines.extend(["", "指标定义："])
    lines.extend(f"- `{name}`：{METRIC_NOTES[name]}" for name in names)
    return "\n".join(lines) + "\n"


#运行统计和失败详情
def render_run_details(rows): #`rows` 已由调用者筛选策略并确保非空
    total = len(rows)
    errors = sum(r.get("status") == "error" for r in rows)
    times = [r["qa_seconds"] for r in rows
             if type(r.get("qa_seconds")) in (int, float)
             and np.isfinite(r["qa_seconds"]) and r["qa_seconds"] >= 0]
    lines = ["## 运行统计", f"- 工程错误率：{errors}/{total}（{errors / total:.2%}）。"]
    if times:
        p50, p95 = np.percentile(times, [50, 95]) #`percentile` 返回两个分位值，分别赋给 p50、p95
        lines.append(f"- QA耗时：P50={p50:.3f}s；P95={p95:.3f}s。")
    else:
        lines.append("- QA耗时：暂无有效记录。")
    lines.append(f"- 有效耗时：{len(times)}/{total}条；缺失或无效{total - len(times)}条。")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(r.get("usage") or {}).get(name) for r in rows]
        known = [v for v in values if type(v) is int and v >= 0]
        value = str(sum(known)) if known else "暂无记录"
        lines.append(f"- `{name}`：已知合计{value}；已知{len(known)}/{total}条，缺失或无效{total - len(known)}条。")
    lines.append("QA计时含失败尝试，不含此前检索和HTTP传输；P50为中位数，P95为95分位估计，非压测结果。缺失用量不按0计算，Token数不等于费用。")
    lines.extend(["", "## 失败详情"])
    failed = [r for r in rows if r.get("status") == "error" or r["review"]["task_pass"] is False]
    if not failed:
        lines.append("没有已确认的任务失败或工程错误；未复核记录不算通过。")
    for row in failed:
        lines.extend(["", f"### {row['case_id']}"])
        for label, value in (
            ("问题", row["question"]),
            ("模型输出", row.get("answer") or row.get("raw_answer") or "无有效答案"),
            ("参考答案", row["reference_answer"]),
            ("引用页码", sorted({c["page"] for c in row.get("citations", [])})),
            ("错误类型", row.get("error_type")),
            ("失败标签", row["review"]["failure_types"]), ("复核说明", row["review"]["notes"])):
            lines.append(f"- {label}：{value}")
    return "\n".join(lines) + "\n"


#导出汇总报告
def export_final_report(source_path, output_path, method="rerank"): #输入复核文件、输出MD文件、检索策略
    labels = {"vector": "向量检索", "bm25": "BM25",
              "hybrid": "混合检索", "rerank": "混合检索＋精排"}
    if method not in labels:
        raise ValueError(f"Unsupported method: {method}")
    source = Path(source_path)
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("report_type") != "qa_review" or report.get("review_schema_version") != "qa-review-v2":
        raise ValueError("Use a qa-review-v2 report") #`qa-review-v2`是文件结构版本
    if report.get("run_status") != "complete" or len(report["cases"]) != report.get("expected_records"):
        raise ValueError("Use a complete QA run")
    rows = [row for row in report["cases"] if row["method"] == method]
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Expected one selected-method record per case")
    split = rows[0].get("split")
    if (split not in ("dev", "test") or report.get("split", split) != split
            or any(row.get("split") != split for row in rows)):
        raise ValueError("Selected rows and report must share one dev/test split")
    retrieval = summarize_retrieval(report)[method]
    qa = summarize_qa_review(report)[method]
    summary = {**retrieval, **qa} #两个`summarize_*`继续负责计分汇总
    names = ("precision_at_k", "required_block_recall", "all_required_blocks_hit_rate",
             "task_pass", "key_fact_coverage", "faithful", "citation_pass",
             "candidate_required_block_recall", "required_block_mrr") #按顺序列出指标
    metadata = {key: report.get(key) for key in (
        "collection", "source_id", "content_hash", "top_k", "candidate_k", "embedding_model",
        "reranker_model", "reranker_revision", "requested_model", "review_gold_file", "review_gold_sha256")}
    metadata.update(review_file=str(source.resolve()), review_sha256=sha256(raw).hexdigest(),
                    selected_method=method, split=split, report_cases=len(rows),
                    source_records=len(report["cases"]),
                    actual_models=sorted({row.get("model") or "未记录" for row in rows}),
                    prompt_versions=sorted({row.get("prompt_version") or "未记录" for row in rows}))
    #标题按策略、数据集、TopK自动生成
    lines = [f"# RAG评测报告：{labels[method]} / {split} / Top{report['top_k']}",
             "", "## 来源与已记录配置", "```json",
             json.dumps(metadata, ensure_ascii=False, indent=2), "```",
             "", "## 指标", render_metric_table({method: summary}, names), "## 回答与拒答"]
    for label, key in (("可回答题", "answerable"), ("不可回答题", "unanswerable")):
        group = qa["answerability"][key]
        task, refusal = group["task_pass"], group["refusal"]
        lines.append(f"- {label}：共{group['case_count']}题；已复核{task['n']}题，通过{task['passed_n']}题，待复核{task['pending_n']}题；有效决策{refusal['n']}次，拒答{refusal['refused_n']}次；工程错误{group['error_n']}次。")
    lines.append(render_run_details(rows))
    lines.append("说明：n是各指标实际计分题数；未单列不等于0；拒答比例不等于拒答合理性。分数仅描述该次运行与该题集，不代表泛化能力。")
    with Path(output_path).open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print("Saved report:", Path(output_path).resolve())


def freeze_evaluation(config_path, output_path, check=False):
    load_evaluation_inputs(config_path, expected_split="test")
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm = {key: os.environ.get(key) for key in ("LLM_BASE_URL", "LLM_MODEL")}
    if not all(llm.values()):
        raise RuntimeError("Configure LLM_BASE_URL and LLM_MODEL first")
    names = config["case_files"] + [
        config_path.name, config["manual_metadata"], config["required_block_gold"], "requirements.txt"]
    names += [path.name for path in root.glob("*.py")]
    snapshot = {
        "config": config, "llm": llm, "prompt_version": STRUCTURED_PROMPT_VERSION,
        "collection": COLLECTION_NAME, "embedding_model": MODEL_NAME,
        "reranker_model": RERANK_MODEL_NAME, "reranker_revision": RERANK_MODEL_REVISION,
        "candidate_k": DEFAULT_CANDIDATE_K,
        "files_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                         for name in sorted(set(names))},
    }
    if check:
        frozen = json.loads(Path(output_path).read_text(encoding="utf-8"))
        if frozen.get("report_type") != "evaluation_freeze" or frozen["snapshot"] != snapshot:
            raise ValueError("Frozen inputs changed; inspect before evaluation")
        print("Freeze check passed")
    else:
        write_report(output_path, {"report_type": "evaluation_freeze",
                     "created_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot})
        print("Saved freeze:", Path(output_path).resolve())


def replay_rerank(config_path, source_path, output_path, *, expected_split):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(Path(config_path).read_bytes())
    cases, _, gold = load_evaluation_inputs(config_path, expected_split=expected_split)
    saved = json.loads(Path(source_path).read_bytes())
    route_k, fusion_k, top_k = saved["candidate_k"], config["fusion_k"], config["top_k"]
    if any(type(k) is not int for k in (fusion_k, top_k)) or not 0 < top_k <= fusion_k <= 2 * route_k:
        raise ValueError("Use 0 < top_k <= fusion_k <= 2 * route_k")
    if (saved.get("split", "dev"), saved["reranker_model"], saved["reranker_revision"]) != (
        expected_split, RERANK_MODEL_NAME, RERANK_MODEL_REVISION
    ):
        raise ValueError("Saved split or reranker differs")

    baseline = select_retrieval_top_k(saved, cases, saved["top_k"])
    lanes = {(r["case_id"], r["method"]): r["candidate_results"] for r in baseline["cases"]}
    model = build_reranker()
    rows = []
    for case in cases:
        qid, question = case["id"], case["question"]
        pair = {m: lanes[qid, m] for m in ("vector", "bm25")}
        fused = rrf_fuse(pair, top_k=fusion_k)
        ranked = rerank_results(question, fused, model, top_k=fusion_k)
        results = ranked[:top_k]
        rows.append({**score_retrieval(case, results), "method": "rerank",
                     "question": question, "answerable": case["answerable"],
                     "candidate_results": ranked, "results": results,
                     "generation_status": "not_run"})

    report = {**saved, "split": expected_split, "cases": rows, "top_k": top_k, "candidate_k": fusion_k,
              "per_route_k": route_k, "source_report": str(Path(source_path).resolve()),
              "config_file": str(Path(config_path).resolve()), "config_snapshot": config,
              "run_at_utc": datetime.now(timezone.utc).isoformat()}
    report = add_block_scores(report, gold)
    report["summary"] = summarize_retrieval(report)
    write_report(output, report)
    print("Rows:", len(rows), "full evidence:", sum(r["all_required_blocks_hit"] is True for r in rows))
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("replay-rerank-dev", "replay-rerank-test"):
        if len(sys.argv) != 5:
            raise SystemExit("Usage: python evaluation.py <replay-rerank-dev|replay-rerank-test> <config> <source> <output>")
        split = "dev" if sys.argv[1] == "replay-rerank-dev" else "test"
        replay_rerank(*sys.argv[2:], expected_split=split)
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("freeze", "check-freeze"):
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python evaluation.py <freeze|check-freeze> <config.json> <freeze.json>")
        freeze_evaluation(sys.argv[2], sys.argv[3], check=sys.argv[1] == "check-freeze")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        if len(sys.argv) not in (4, 5):
            raise SystemExit("Usage: python evaluation.py export <review.json> <output.md> [method]")
        export_final_report(*sys.argv[2:]) #*sys.argv[2:]将命令参数展开，两项使用默认策略，三项使用指定策略
        return

    if len(sys.argv) > 1 and sys.argv[1] == "check-test":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: python evaluation.py check-test <config.json>")
        cases, _, gold = load_evaluation_inputs(sys.argv[2], expected_split="test")
        print("cases:", len(cases), "answerable:", sum(c["answerable"] for c in cases)) #布尔值求和统计可回答题
        print("gold_cases:", len(gold["cases"]))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "evidence":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python evaluation.py evidence <cases.json> <case_id>")
        preview_case_chunks(sys.argv[2], sys.argv[3])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark": #本地检索计时，不生成答案、不评价答案正确性
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python evaluation.py benchmark <config.json> <report.json>")
        benchmark_retrieval(sys.argv[2], sys.argv[3])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "review": #创建待填写review的新副本，不自动判答案
        if len(sys.argv) not in (4, 5):
            raise SystemExit("Usage: python evaluation.py review <qa.json> <output.json> [gold.json]")
        prepare_qa_review(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else None)
        return
    
    if len(sys.argv) > 1 and sys.argv[1] in ("qa", "qa-test"): #回放已有上下文；加--run才真实生成
        if len(sys.argv) not in (6, 7) or (len(sys.argv) == 7 and sys.argv[6] != "--run"):
            raise SystemExit("Usage: python evaluation.py <qa|qa-test> <config> <retrieval> <case_id|all> <output> [--run]")
            #四个位置参数依次是配置、检索报告、题号、输出路径，只有最后显式添加 --run，才将 run 设为 True
        run_saved_qa(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], run=len(sys.argv) == 7, expected_split="test" if sys.argv[1] == "qa-test" else "dev")
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
        if len(sys.argv) not in (5, 6):
            raise SystemExit(
                "Usage: python evaluation.py score <config.json> <input.json> <output.json> [top_k]"
            )
        cases, _, gold = load_evaluation_inputs(sys.argv[2])
        report = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        if len(sys.argv) == 6:
            report = select_retrieval_top_k(report, cases, int(sys.argv[5]))
            report["source_report"] = str(Path(sys.argv[3]).resolve())
        report = add_block_scores(report, gold)
        report["summary"] = summarize_retrieval(report)
        write_report(sys.argv[4], report)
        count = sum(row["required_block_recall"] is not None for row in report["cases"]) #sum 把每个“评分非空”的 True 计作 1
        print("Rows with block labels:", count)
        return
    
    if len(sys.argv) > 1 and sys.argv[1] in ("compare", "compare-test"): #四策略重新检索、评分、保存报告
        if len(sys.argv) != 4:
            raise SystemExit(
                "Usage: python evaluation.py <compare|compare-test> <config.json> <report.json>"
            )
        run_retrieval_evaluation(sys.argv[2], sys.argv[3], expected_split="test" if sys.argv[1] == "compare-test" else "dev")
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