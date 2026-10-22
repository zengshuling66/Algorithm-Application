import pytest
from langchain_adapter import results_to_documents
from langchain_prompt import build_langchain_messages
from prompt_builder import build_messages

def test_prompt_matches_original():
    question = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
    results = [{"chunk_id": "demo-43", "text": "测试正文 {literal}",
                "source": "train_a.pdf", "page": 43}]
    documents = results_to_documents(results)
    assert build_langchain_messages(question, documents) == build_messages(question, results)

def test_prompt_rejects_empty_documents():
    with pytest.raises(ValueError):
        build_langchain_messages("无钥匙解锁如何设置？", [])