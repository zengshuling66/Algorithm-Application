import sys #用于读取启动参数
from sentence_transformers import SentenceTransformer #用于加载 Embedding 模型
from retrieval import MODEL_NAME, search_stored_chunks
from prompt_builder import PROMPT_VERSION, build_messages
from generation import generate_answer


def main():
    if len(sys.argv) != 4: #检查启动参数数量
        raise SystemExit("Usage: python query.py <source_id> <content_hash> <query>")

    source_id = sys.argv[1]
    content_hash = sys.argv[2]
    query = sys.argv[3]

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    results = search_stored_chunks( #只编码这个问题，不重新编码数据库里的 469 个资料块
        query, model, source_id, content_hash, top_k=3,
    )

    if not results:
        print("No matching evidence found.")
        return

    # print(format_context(results))

    # for rank, result in enumerate(results, start=1):
    #     print(
    #         f"{rank}. PDF page {result['page']}, "
    #         f"score={result['score']:.4f}"
    #     )
    #     print(result["text"])

    messages = build_messages(query, results)
    generated = generate_answer(messages)
    print("prompt version:", PROMPT_VERSION)
    print(generated["answer"])
    print("model:", generated["model"])
    print("usage:", generated["usage"])


if __name__ == "__main__":
    main()