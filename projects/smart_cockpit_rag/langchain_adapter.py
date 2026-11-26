from langchain_core.documents import Document

#负责格式转换
def results_to_documents(results):
    documents = []
    for rank, result in enumerate(results, start=1): #enumerate(..., start=1) 按现有顺序编号
        metadata = result.copy()
        text = metadata.pop("text") #从新字典中取出正文并删除该字段
        metadata["retrieval_rank"] = rank
        documents.append(Document(
            id=result["chunk_id"], page_content=text, metadata=metadata,
        ))
    return documents

if __name__ == "__main__":
    sample = [{
        "chunk_id": "demo-43", "text": "单门模式仅解锁当前车门。",
        "source_id": "training_manual_a", "content_hash": "demo-hash",
        "page": 43, "score": 0.8,
    }]
    documents = results_to_documents(sample)
    doc = documents[0]
    assert doc.page_content == sample[0]["text"]
    assert doc.metadata["content_hash"] == sample[0]["content_hash"]
    assert "text" in sample[0]
    assert results_to_documents([]) == []
    print(doc.id, doc.metadata["page"], doc.metadata["retrieval_rank"])
    print("adapter checks passed")