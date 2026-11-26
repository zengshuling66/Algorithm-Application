import torch #torch 提供下面的激活函数
import numpy as np
from vector_store import DEFAULT_TOP_K
from sentence_transformers import CrossEncoder #CrossEncoder 提供加载与推理接口

RERANK_MODEL_NAME = "BAAI/bge-reranker-base" #模型名称
RERANK_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70" #权重版本

#接收是否只读缓存，返回模型对象，尚未执行打分
def build_reranker(local_files_only=True): #默认 True 防止日常运行意外联网，首次准备权重时显式传 False
    return CrossEncoder(
        RERANK_MODEL_NAME, revision=RERANK_MODEL_REVISION,
        device="cpu", max_length=512, #max_length=512 限制的是问题、资料和特殊标记合计的 token 数，过长输入会被截断
        activation_fn=torch.nn.Identity(), #Identity() 表示输出保持不变，保留原始 logits，不把它压缩为0到1
        local_files_only=local_files_only,
    )

#接收问题、已按RRF排序的候选字典列表、已加载的精排模型，返回最多 top_k 条新字典
def rerank_results(question, results, reranker, top_k=DEFAULT_TOP_K):
    if not question.strip() or top_k <= 0:
        raise ValueError("Use a nonempty question and positive top_k")
    if not results:
        return []
    pairs = [(question, item["text"]) for item in results]
    scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
    if scores.shape != (len(results),) or not np.isfinite(scores).all(): #np.isfinite(scores)逐项检查是否为有限数，.all()要求全部通过
        raise ValueError("Invalid reranker scores")
    ranked = [
        {**item, "rrf_rank": rank, "rerank_score": float(score)} #{**item, ...}创建新字典，保留正文、身份、哈希及两路信息
        for rank, (item, score) in enumerate(zip(results, scores), start=1)
        #enumerate(zip(...), start=1) 同时取得原RRF名次、记录和对应分数
    ]
    return sorted(
        ranked, key=lambda item: (-item["rerank_score"], item["rrf_rank"]), #排序先按精排分降序，同分按原RRF名次升序，最后才截取数量
    )[:top_k]


if __name__ == "__main__":
    question = "单门解锁模式下，触摸前门把手内部传感器会解锁哪些车门？"
    texts = [
        "单门模式仅解锁当前车门。",
        "全车模式解锁全车车门。",
        "前排座椅通风分为低、中、高三档。",
    ]
    pairs = [(question, text) for text in texts] #pairs 的列表推导式得到三组 (同一个问题, 不同文本)
    model = build_reranker()
    scores = model.predict(pairs, batch_size=4, show_progress_bar=False) #predict 给每组输入打分；batch_size=4 表示每批最多4组；show_progress_bar=False 关闭推理进度条
    assert scores.shape == (len(texts),)
    print("shape:", scores.shape, "dtype:", scores.dtype)
    for text, score in zip(texts, scores): #zip 逐一配对原文和分数
        print(f"{float(score):.4f}", text) #predict 按输入顺序返回，不会自动排序