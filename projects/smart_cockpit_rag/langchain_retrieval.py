import sys
from langchain_core.runnables import RunnableLambda #RunnableLambda 提供函数包装
from sentence_transformers import SentenceTransformer #加载现有 BGE Embedding 模型
from langchain_adapter import results_to_documents
from retrieval import DEFAULT_TOP_K, MODEL_NAME, search_stored_chunks

#返回处理链，不返回检索结果；它接收已加载模型、资料范围和数量。
def build_retriever(model, source_id, content_hash, top_k=DEFAULT_TOP_K):
    def retrieve(query): #内部retrieve只接收问题，却能使用外层的模型及范围，这叫闭包：内部函数保留对外层变量的引用，因此不必每次传问题时再重复传这些配置。
        return search_stored_chunks(
            query, model, source_id, content_hash, top_k=top_k,
        )
    return RunnableLambda(retrieve) | RunnableLambda(results_to_documents)
    #用 A | B 连接时，这里的 | 是 LangChain 定义的顺序连接，A 的输出直接成为 B 的输入，调用 .invoke(输入) 才执行。
    #RunnableLambda(retrieve) 传入的是函数本身，留待执行；写成 retrieve() 就会立即调用，而且缺少问题参数，右侧同理。

def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python langchain_retrieval.py <source_id> <content_hash> <query>"
        )
    source_id, content_hash, question = sys.argv[1:]
    model = SentenceTransformer(
        MODEL_NAME, device="cpu", local_files_only=True, #local_files_only=True表示只读已有模型缓存，缺缓存就报错，不自动下载。
    )
    retriever = build_retriever(model, source_id, content_hash)
    documents = retriever.invoke(question) #模型在链外加载，同一进程中多次 .invoke() 可以复用。
    #它先调用内部 retrieve(question)，再将返回的整个列表交给转换函数。最终返回列表，不是答案。
    print("documents:", len(documents))
    for doc in documents:
        metadata = doc.metadata
        print(
            f"rank={metadata['retrieval_rank']} "
            f"page={metadata['page']} score={metadata['score']:.4f}"
        )
        print("chunk_id:", doc.id)
        print("scope:", metadata["source_id"], metadata["content_hash"])
        print(doc.page_content)

if __name__ == "__main__":
    main()