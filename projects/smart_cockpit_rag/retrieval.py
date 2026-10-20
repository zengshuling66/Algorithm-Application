import sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from ingestion import load_pdf_pages
from chunking import build_chunks
from vector_store import DEFAULT_TOP_K, build_points, save_points, search_points

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

def build_faiss_index(vectors: np.ndarray): #标明参数预期是 NumPy 数组；类型标注便于阅读，实际检查由下一行执行
    if vectors.ndim != 2 or vectors.shape[0] == 0: #vectors.ndim是数组的维度数量；vectors.shape[0] 是行数，这里为 3；条件拒绝一维数组或没有任何向量的空矩阵。
        raise ValueError("vectors must be a nonempty 2D matrix")

    dimension = vectors.shape[1] #取列数，得到 512。索引必须知道每条向量有多少个数字；从数组中读取，比把 512 写死更容易适配其他模型。
    index = faiss.IndexFlatIP(dimension) #创建一个空索引。Flat 表示逐条比较保存的向量，得到精确搜索结果；IP 是 Inner Product，即内积，它会优先返回内积较大的向量。
    index.add(vectors) #这时才真正写入三行向量。只创建索引、没调用 add() 时，里面没有任何候选数据。
    return index

#负责“问题编码 + FAISS 查询”，后面由Qdrant代替
def search_chunks(query, model, index, chunks, top_k=DEFAULT_TOP_K):
    query = query.strip() #字符串清理：去掉两端空白
    if not query: #拒绝空问题
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    actual_top_k = min(top_k, len(chunks)) #防止要求的结果数超过现有数量
    query_vector = model.encode(
        [QUERY_INSTRUCTION + query], #把检索任务说明接到问题前面，一同交给 Embedding 模型
        convert_to_numpy=True,
        normalize_embeddings=True, #问题和资料都需要归一化
        show_progress_bar=False,
    )
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
    scores, positions = index.search(query_vector, actual_top_k) #用问题向量搜索索引，找出分数最高的actual_top_k个资料块，同时返回它们的分数和存储位置。
    #FAISS 支持一次搜索多个问题，所以结果统一组织成二维数组：[问题数量, 每个问题返回的命中数量]

    results = []
    for score, position in zip(scores[0], positions[0]): #第一个问题的全部三个分数和命中位置，zip 会把两个数组中相同顺序的元素配成一组。scores[0][0]：第一个问题的第一名分数；positions[0][1]：第一个问题的第二名位置。
        result = chunks[int(position)].copy() #int(position) 把 NumPy 整数转成 Python 整数，用它读取 chunks；.copy() 创建一个新字典，保留正文和所有来源字段
        result["score"] = float(score) #转成普通 Python 浮点数，方便后面保存 JSON
        results.append(result)
    return results


def search_stored_chunks(query, model, source_id, content_hash, top_k=DEFAULT_TOP_K): #source_id 和 content_hash 决定查询哪份资料、哪个文件版本
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_batch = model.encode(
        [QUERY_INSTRUCTION + query],
        convert_to_numpy=True,
        normalize_embeddings=True, #将向量长度归一化到约 1
        show_progress_bar=False,
    )
    query_vector = np.ascontiguousarray(query_batch[0], dtype=np.float32) #query_batch[0] 取出第一个问题的完整向量，形状变成 (512,)
    return search_points(query_vector, source_id, content_hash, top_k=top_k)
    #把向量、来源、哈希和返回数量交给它，再直接返回它的结果。这里不需要自己读取 chunks[position]，因为数据库已经返回保存的正文和页码。

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python retrieval.py <pdf_path> <source_id>")

    pdf_path = Path(sys.argv[1])
    source_id = sys.argv[2]
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    page_records, _, _, _ = load_pdf_pages(pdf_path, source_id)
    chunks = build_chunks(page_records)
    if not chunks:
        raise ValueError("No text chunks found in PDF")

    model = SentenceTransformer(MODEL_NAME) #当前 BGE 模型把每段文本表示成 512 个数字。
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode( #chunks[i] 对应 vectors[i]
        texts,
        batch_size=16,
        convert_to_numpy=True, #让结果回到 CPU 上的 NumPy 数组
        normalize_embeddings=True, #把每行向量的 L2 范数调到约 1：模型先生成向量，再把每个分量除以这条向量原来的长度。
        show_progress_bar=False, #仅关闭进度显示
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32) #确保后续交给 FAISS 的矩阵采用连续内存和 32 位浮点数
    #FAISS 是一个搜索相似向量的库。它接收模型生成的数字向量，建立索引，再根据问题向量查找最相似的几条。

    points = build_points(chunks, vectors)
    print("points:", len(points))
    print("vector length:", len(points[0].vector)) #points[0].vector 是首条记录的向量列表
    print("page:", points[0].payload["page"])
    repeated = build_points(chunks[:1], vectors[:1])
    print("stable ID:", points[0].id == repeated[0].id)
    before, after = save_points(points, dimension=vectors.shape[1])
    _, after_repeat = save_points(points, dimension=vectors.shape[1]) #_ 表示不用第二次的写入前数量，只保留重复写入后的数量
    print("counts:", before, after, after_repeat)

    index = build_faiss_index(vectors)

    self_scores, self_indices = index.search( #第一个参数是查询向量矩阵，第二个参数 1 表示每个查询只取最相似的一条。我们暂时拿已有的第一条向量查询自身，因此使用 vectors[:1]。
        vectors[:1],
        1,
    )
    #self_scores：每个查询命中结果的内积分数；self_indices：命中向量在索引中的下标

    print("index dimension:", index.d) #索引要求的向量维度，应该为 512
    print("indexed vectors:", index.ntotal) #已写入的向量数量，应该为len(chunks)
    print("self scores:", self_scores)
    print("self indices:", self_indices)

    print("chunks:", len(chunks))
    print("vectors:", vectors.shape, vectors.dtype)
    print("first:", chunks[0]["chunk_id"])
    print("first norm:", np.linalg.norm(vectors[0])) #向量长度检查，预期接近 1，而不是必须逐位等于 1

    query = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
    # results = search_chunks(query, model, index, chunks, top_k=3)
    results = search_stored_chunks(query, model, source_id, content_hash=chunks[0]["content_hash"], top_k=DEFAULT_TOP_K)
    #这里chunks[0]["content_hash"]是从第一个块中读取当前 PDF 的文件哈希。同一文件的全部块都有这个哈希，所以不是只检索第一块，而是限定在该文件版本的全部块中检索。

    print("\nquery:", query)
    for rank, result in enumerate(results, start=1): #enumerate 每次提供“编号＋列表中的元素”，start=1 表示编号从 1 开始
        print(
            f"{rank}. PDF第{result['page']}页，"
            f"score={result['score']:.4f}"
        )
        print(result["text"])
        print("metadata:", result["brand"], result["section"], result["model_year"])


if __name__ == "__main__":
    main()