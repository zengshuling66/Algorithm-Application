import sys
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
from langchain_retrieval import build_retriever
from langchain_prompt import build_langchain_messages
from generation import generate_answer
from retrieval import MODEL_NAME
from prompt_builder import PROMPT_VERSION

#组装而不执行，输入检索链，返回顺序链
def build_qa_chain(retriever, generator=generate_answer):
    #输入问题，执行检索，输出含问题、Document 列表、空消息列表和 Prompt 版本的字典。
    def prepare(question):
        documents = retriever.invoke(question)
        return {"question": question, "documents": documents,
                "messages": [], "prompt_version": PROMPT_VERSION}
    #answer处理两条路径。无证据直接返回程序级保守回答，不调用模型，故模型和用量都是 None；有证据则构建消息，再调用生成函数。
    def answer(state):
        if not state["documents"]:
            return {**state, "status": "no_evidence",
                    "answer": "根据现有资料无法确定。",
                    "model": None, "usage": None}
            #{**state, ...} 是字典展开。把原状态各字段放入新字典，再补新字段；同名字段以后面的值为准。
        messages = build_langchain_messages(state["question"], state["documents"])
        generated = generator(messages)
        return {**state, "messages": messages, **generated, "status": "generated"}
        #**generated 因此保留原函数返回的答案、实际响应模型和完整 usage，不只留下答案字符串。
    return RunnableLambda(prepare) | RunnableLambda(answer)

def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python langchain_query.py <source_id> <content_hash> <query>")
    source_id, content_hash, question = sys.argv[1:]
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    retriever = build_retriever(model, source_id, content_hash)
    result = build_qa_chain(retriever).invoke(question)
    print("prompt version:", result["prompt_version"])
    print("status:", result["status"])
    print(result["answer"])
    for doc in result["documents"]:
        print("evidence:", doc.id, "PDF page:", doc.metadata["page"])
        print(doc.page_content)
    print("model:", result["model"])
    print("usage:", result["usage"])

if __name__ == "__main__":
    main()