from vector_store import DEFAULT_TOP_K
from bm25_retrieval import build_bm25_search
from retrieval import search_stored_chunks
from vector_store import load_stored_chunks

DEFAULT_CANDIDATE_K = 10 #向量和 BM25 分别找前10条，为融合提供候选

#RRF改用名次：每出现一路，贡献 1 / (60 + 名次)；未出现则贡献0，名次从1开始，两路等权。
#输入是 {"vector": 向量结果列表, "bm25": BM25结果列表}，输出仍是检索字典列表，因此后续可以复用原来的 Document 转换函数。
def rrf_fuse(ranked_results, top_k=DEFAULT_TOP_K, rank_constant=60):
    if top_k <= 0 or rank_constant <= 0:
        raise ValueError("top_k and rank_constant must be positive")
    fused = {}
    for method, results in ranked_results.items(): #.items() 逐路取出名称和列表
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            if chunk_id not in fused: #fused 跨两路共享，以 chunk_id 为键：首次出现复制完整记录，再累计贡献。
                fused[chunk_id] = {
                    **item, "score": 0.0,
                    "retrieval_ranks": {}, "retrieval_scores": {},
                }
            entry = fused[chunk_id] #entry 指向 fused 中的字典，所以修改它就是更新共享记录。
            entry["score"] += 1.0 / (rank_constant + rank) #当前 chunk 的 RRF 总分
            entry["retrieval_ranks"][method] = rank #method 标识当前是哪一路，记录当前 chunk 在该路的名次
            entry["retrieval_scores"][method] = float(item["score"]) #记录当前 chunk 在该路的原始分数
    return sorted(
        #fused.values()取出所有融合记录；排序键先用负分数实现降序，同分再按 ID 稳定排序，最后截取3条，此时 score 表示 RRF分数。
        fused.values(), key=lambda item: (-item["score"], item["chunk_id"]),
    )[:top_k]


#外层读取语料、建立一次 BM25 统计，返回可重复调用的 search；内层每次接收问题，依次执行两路检索，再交给 RRF。
def build_hybrid_search(model, source_id, content_hash, candidate_k=DEFAULT_CANDIDATE_K):
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    bm25_search = build_bm25_search(load_stored_chunks(source_id, content_hash))
    def search(query, top_k=DEFAULT_TOP_K):
        if not 0 < top_k <= candidate_k:
            raise ValueError("Use 0 < top_k <= candidate_k")
        vector = search_stored_chunks(query, model, source_id, content_hash, top_k=candidate_k)
        lexical = bm25_search(query, top_k=candidate_k)
        return rrf_fuse({"vector": vector, "bm25": lexical}, top_k=top_k)
    return search