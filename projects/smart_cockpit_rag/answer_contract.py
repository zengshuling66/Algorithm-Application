import json

#建立证据映射
def build_evidence_map(documents):
    evidence_map = {}
    for rank, doc in enumerate(documents, start=1): #按输入顺序同时取出编号和Document
        metadata = doc.metadata
        evidence_map[f"E{rank}"] = { #f"E{rank}"生成字典键；右侧字典保存这条证据的信息
            "chunk_id": doc.id,
            "source": metadata["source"],
            "source_id": metadata["source_id"],
            "content_hash": metadata["content_hash"],
            "page": metadata["page"],
            "text": doc.page_content,
        }
    return evidence_map


#解析、检查、组装真实引用
#接收生成函数返回的JSON字符串和本次证据映射，成功时返回含可信citations的新字典，失败时抛出异常。
def parse_answer(raw_answer, evidence_map):
    data = json.loads(raw_answer)
    required = {"answer_status", "answer", "evidence_ids"} #要求恰好等于三个规定字段，因此缺字段或多出模型编造的source/page都会失败
    if not isinstance(data, dict) or set(data) != required: #set(data)取得字典键；or短路保证非字典不会继续执行后面的键检查
        raise ValueError("Invalid answer fields")
    status, answer, ids = data["answer_status"], data["answer"], data["evidence_ids"] #一次取出三个字段；随后分别检查状态、非空答案、ID列表及元素类型
    if status not in ("answered", "refused"):
        raise ValueError("Invalid answer status")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a nonempty string")
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids): #先检查元素都是字符串，再调用set(ids)，避免列表等不可哈希对象引发意外错误
        raise ValueError("Evidence IDs must be a list of strings")
    if len(ids) != len(set(ids)) or any(item not in evidence_map for item in ids): #set(ids)去重后数量变少，说明出现重复ID
        raise ValueError("Duplicate or unknown evidence ID") #any(...)表示只要有一个ID不在本次映射中，就拒绝
    if status == "answered" and not ids:
        raise ValueError("Answered output requires evidence")
    return {
        **data, "answer": answer.strip(), #**data保留已校验字段，覆盖清理后的答案
        "citations": [{"evidence_id": eid, **evidence_map[eid]} for eid in ids],
    }