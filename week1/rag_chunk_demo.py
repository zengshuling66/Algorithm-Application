chunks = [
    {
        "text": "RAG 是 Retrieval-Augmented Generation，也就是检索增强生成。",
        "source": "大模型应用讲义.pdf",
        "page": 12,
        "score": 0.92,
    },
    {
        "text": "LangChain 可以用来编排大模型应用，包括 Prompt、Retriever 和 Chain。",
        "source": "LangChain 入门.pdf",
        "page": 5,
        "score": 0.81,
    },
    {
        "text": "卷积神经网络常用于图像识别任务。",
        "source": "深度学习基础.pdf",
        "page": 33,
        "score": 0.42,
    },
]

threshold = 0.75

def filter_chunks(chunks, threshold): #filter_chunks 过滤低分 chunk
    filtered_chunks = []

    for chunk in chunks:
        if chunk["score"] >= threshold:
            filtered_chunks.append(chunk)
    
    return filtered_chunks

def sort_chunks_by_score(chunks): #sort_chunks_by_score 按 score 排序
    return sorted(chunks, key=lambda x: x["score"], reverse=True)

def print_chunks(chunks): #print_chunks 打印结果
    print("检索到的相关资料：")

    for chunk in chunks:
        print(f"文本: {chunk['text']}")
        print(f"来源: {chunk['source']} (第 {chunk['page']} 页)")
        print(f"相关度得分: {chunk['score']:.2f}")
        print("-" * 40)

def main():
    filtered_chunks = filter_chunks(chunks, threshold)
    sorted_chunks = sort_chunks_by_score(filtered_chunks)
    print_chunks(sorted_chunks)

if __name__ == "__main__":
    main()