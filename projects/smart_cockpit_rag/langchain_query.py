import sys
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
from langchain_retrieval import build_hybrid_retriever
from langchain_prompt import build_structured_messages, STRUCTURED_PROMPT_VERSION
from generation import generate_answer
from retrieval import MODEL_NAME
from answer_contract import build_evidence_map, parse_answer

def generate_json_answer(messages): #generate_json_answer只负责打开JSON开关，仍允许测试时注入假生成器。
    return generate_answer(messages, json_mode=True)


#组装而不执行，输入检索链，返回顺序链；.invoke(question) 才是启动流程
def build_qa_chain(retriever, generator=generate_json_answer):
    #根据本次最终Document建立映射；输入问题，执行检索，输出含问题、Document 列表、空消息列表和 Prompt 版本的字典。
    #prepare 返回的字典就是 state，它负责把本次问题、命中记录和证据映射传给下一步。
    def prepare(question):
        documents = retriever.invoke(question) #检索出的 documents
        return {"question": question, "documents": documents,
                "evidence_map": build_evidence_map(documents), "messages": [], "prompt_version": STRUCTURED_PROMPT_VERSION}
                #build_evidence_map(documents) → 建立 evidence_map
    #answer先处理空证据，否则构建消息、生成、解析。
    def answer(state):
        if not state["documents"]:
            return {**state, "status": "no_evidence", "answer_status": "refused",
                    "answer": "根据现有资料无法确定。", "evidence_ids": [], "citations": [],
                    "model": None, "usage": None}
            #{**state, ...} 是字典展开。把原状态各字段放入新字典，再补新字段；同名字段以后面的值为准。
        messages = build_structured_messages(state["question"], state["evidence_map"]) #组织模型输入
        generated = generator(messages) #调用模型，得到原始 JSON 字符串
        parsed = parse_answer(generated["answer"], state["evidence_map"]) #解析、检查、组装真实引用
        return {**state, "messages": messages, **generated, **parsed, "status": "generated"}
        #**generated 因此保留原函数返回的答案、实际响应模型和完整 usage，不只留下答案字符串。
    return RunnableLambda(prepare) | RunnableLambda(answer)

def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python langchain_query.py <source_id> <content_hash> <query>")
    source_id, content_hash, question = sys.argv[1:]
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    retriever = build_hybrid_retriever(model, source_id, content_hash)
    result = build_qa_chain(retriever).invoke(question)
    print("prompt version:", result["prompt_version"])
    print("status:", result["status"])
    print(result["answer"])
    print("answer status:", result["answer_status"])
    print("citations:", result["citations"])
    for doc in result["documents"]:
        print("evidence:", doc.id, "PDF page:", doc.metadata["page"])
        print(doc.page_content)
    print("model:", result["model"])
    print("usage:", result["usage"])

if __name__ == "__main__":
    main()