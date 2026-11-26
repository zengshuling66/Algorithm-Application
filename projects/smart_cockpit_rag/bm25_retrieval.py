import sys
import jieba #负责中文分词
from rank_bm25 import BM25Okapi #负责BM25计分
from vector_store import DEFAULT_TOP_K, load_stored_chunks

def tokenize(text): #tokenize() 接收字符串，返回词列表
    return [word.lower() for word in jieba.lcut(text) if word.isalnum()]
    #lcut()切词，isalnum()保留由字母、数字或汉字组成的词、去掉空白标点，lower()统一英文大小写，使 FCTA 与 fcta 一致

#外层构建、内层执行：build_bm25_search() 接收 chunks，建立词频等统计，返回 search 函数；不是立即返回命中结果。
def build_bm25_search(chunks):
    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus, k1=1.5, b=0.75) if any(corpus) else None #BM25Okapi()根据corpus统计词频、文档频率和长度，不调用大模型
    #k1 控制词频收益的饱和程度；b 控制长度归一化强度。any(corpus) 检查是否至少有一个非空词列表，避免空语料建索引。
    def search(query, top_k=DEFAULT_TOP_K):
        if not query.strip() or top_k <= 0:
            raise ValueError("Use a nonempty query and positive top_k")
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens) #get_scores() 返回每篇文本的分数
        positions = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        #key=lambda i: scores[i] 指定按分数排序，reverse=True 表示从高到低
        return [
            {**chunks[i], "score": float(scores[i])} #{**chunks[i], ...} 创建新字典，保留全部来源字段、不改原 payload
            for i in positions if any(token in corpus[i] for token in tokens) #内层any(...)检查候选是否至少包含一个查询词，排除完全无词项重合的文本。
        ][:top_k]
    return search

if __name__ == "__main__":
    source_id, content_hash, query = sys.argv[1:]
    chunks = load_stored_chunks(source_id, content_hash) #数据库读取
    search = build_bm25_search(chunks) #BM25 分词计分
    print("scoped chunks:", len(chunks))
    for item in search(query):
        print(item["page"], item["score"], item["chunk_id"], item["text"])