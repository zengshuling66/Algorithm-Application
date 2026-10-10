import numpy as np
from uuid import NAMESPACE_URL, uuid5
#NAMESPACE_URL是固定的命名空间标记；uuid5() 根据“命名空间＋字符串”确定性生成 UUID，同一个命名空间、同一个 chunk ID，得到同一个 UUID。
from qdrant_client import QdrantClient, models #models 是 Qdrant 的数据结构模块；QdrantClient 负责数据库操作
from pathlib import Path

STORAGE_PATH = Path(__file__).resolve().parent / ".qdrant"
#__file__ 是当前脚本路径，resolve() 转成绝对路径，parent 取项目目录，/ ".qdrant" 拼接存储目录。这让终端切换目录时仍使用同一个数据库。
COLLECTION_NAME = "cockpit_manual_bge_small_zh_v1_5" #COLLECTION_NAME 是集合名称，包含当前模型标识

# 1. 文档向量矩阵 (469, 512)
# 它有 469 行，每行是一个文本块的向量；每条向量有 512 个数字：
# vectors.shape       → (469, 512)
# vectors.shape[0]    → 469：行数，也是 chunk 数量
# vectors.shape[1]    → 512：列数，也是每条向量的维度

# 2. 一个问题的向量批次 (1, 512)
# 我们把 [query] 交给模型，一次只编码一个问题，因此：
# query_batch.shape       → (1, 512)
# query_batch.shape[0]    → 1：这一批有几个问题
# query_batch.shape[1]    → 512：每个问题的向量维度
# query_vector.shape[0]   → 512：单个问题的向量维度

# 模型按批次输出问题向量
#           ↓
# query_batch：(1, 512)
#           ↓ 取出第一个问题，query_batch[0]
# query_vector：(512,)
#           ↓
# 交给接收单个问题向量的 search_points

def build_points(chunks: list[dict], vectors: np.ndarray):
    if vectors.ndim != 2: #先确认二维
        raise ValueError("vectors must be a 2D matrix")
    if vectors.shape[0] != len(chunks): #再读取行数，行数必须等于块数
        raise ValueError("chunk count and vector count must match")

    points = []
    for chunk, vector in zip(chunks, vectors): #zip(chunks, vectors) 按相同位置配对
        point_id = str(uuid5(NAMESPACE_URL, chunk["chunk_id"])) #uuid5(...)返回 UUID 对象，外层 str() 转成数据库接受的 UUID 字符串。它用于数据库定位；原来的可读 chunk ID 用于追踪来源。
        point = models.PointStruct( #PointStruct(...) 构造一条记录
            id=point_id, #id 接收稳定 ID
            vector=vector.tolist(), #vector.tolist() 把 NumPy 一维数组转为普通 Python 数字列表，长度仍是 512
            payload=chunk.copy(), #保存正文、页码、来源等字段的字典副本
        )
        points.append(point)
    return points


#单独负责数据库保存，便于后续独立入库入口复用，也能测试重复导入行为
def save_points(points, dimension: int): #输入是记录列表和向量维度，输出是写入前、写入后的数量
    client = QdrantClient(path=str(STORAGE_PATH))
    try: #try 包围数据库操作；finally 无论正常返回还是中途异常都会执行
        if not client.collection_exists(COLLECTION_NAME): #collection_exists()返回布尔值，not表示不存在时才创建；不能每次启动都删除重建集合，否则会丢数据
            client.create_collection(
                collection_name=COLLECTION_NAME, #collection_name 指定集合；vectors_config 指定向量规则
                vectors_config=models.VectorParams( #内层 VectorParams(...) 先构造配置，再传给外层 create_collection()
                    size=dimension, #每条向量的长度，此处为 512
                    distance=models.Distance.COSINE, #Distance.COSINE 指定余弦相似度规则
                ),
            )
        config = client.get_collection(COLLECTION_NAME).config.params.vectors #get_collection() 返回集合信息对象，.config.params.vectors 逐层取出向量配置
        if not isinstance(config, models.VectorParams): #isinstance() 检查类型，确保 config 是 VectorParams 类型
            raise ValueError("Collection vector configuration is not VectorParams")
        if config.size != dimension or config.distance != models.Distance.COSINE: #随后检查维度和距离规则，任一不匹配就报错
            raise ValueError("Collection vector configuration does not match")
        #before 在写入前取得，after 必须在写入后取得
        before = client.count(collection_name=COLLECTION_NAME, exact=True).count #count()返回计数结果对象，最后的 .count 才取出整数；exact=True 要求精确计数。
        #upsert()接收集合名称和整批记录，wait=True表示等待修改完成后再继续统计。已有 ID 会被覆盖，而不是另添一条；当前 469 条可以整批处理，资料明显增多后再分批。
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        after = client.count(collection_name=COLLECTION_NAME, exact=True).count
        return before, after
    finally:
        client.close() #close() 释放资源和本地锁，不删除数据库


def search_points(query_vector, source_id, content_hash, top_k=3):
    if query_vector.ndim != 1 or top_k <= 0:
        raise ValueError("Use a 1D query vector and positive top_k")
    if not source_id or not content_hash:
        raise ValueError("Explicit source_id and content_hash are required")

    scope = models.Filter( #Filter 是过滤规则对象
        must=[ #must 是条件列表，含义是列表里的条件全部满足，相当于逻辑 AND
            models.FieldCondition( #FieldCondition 描述某个字段的要求
                key="source_id",
                match=models.MatchValue(value=source_id),
            ),
            models.FieldCondition(
                key="content_hash",
                match=models.MatchValue(value=content_hash),
            ),
        ],
    )
    client = QdrantClient(path=str(STORAGE_PATH))
    try:
        config = client.get_collection(COLLECTION_NAME).config.params.vectors
        if not isinstance(config, models.VectorParams): #单向量类型检查
            raise ValueError("This collection must use a single unnamed vector")
        if config.size != query_vector.shape[0] or config.distance != models.Distance.COSINE:
        #单向量类型检查通过后，query_vector.shape[0] 才代表向量长度 512，用它核对集合维度，并确认余弦规则
            raise ValueError("Query vector configuration does not match")

        response = client.query_points( #执行相似度检索
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            query_filter=scope,
            limit=top_k, #指定最多返回几条，如果匹配记录少于 top_k，就返回实际可用数量
            with_payload=True, #要求附带正文和来源
            with_vectors=False, #不返回每条文档的 512 个原始数字，因为组织引用只需要正文、来源和分数
        )
        results = []
        for hit in response.points: #每个 hit 是带属性的命中对象
            result = hit.payload.copy() #.payload 包含入库时的 chunk 字典
            result["score"] = float(hit.score) #.score 是相似度分数
            results.append(result)
        return results
    finally:
        client.close()