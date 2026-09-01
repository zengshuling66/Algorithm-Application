import torch
from sentence_transformers import SentenceTransformer
# SentenceTransformer 是一个类，它让我们直接从文本得到句向量，封装了：
# Tokenizer
# + 预训练 Transformer
# + pooling
# + sentence embedding 输出

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
#这是 Hugging Face 上的模型标识，不是本地文件路径。第一次运行会下载模型并缓存，以后通常直接读取缓存。使用全大写表示它是程序配置常量。

QUERY_INSTRUCTION = ("为这个句子生成表示以用于检索相关文章：")

class EmbeddingRetriever:
    def __init__(self, documents, model_name=MODEL_NAME):
        if not documents:
            raise ValueError("documents不能为空")

        self.documents = documents

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = SentenceTransformer( #加载预训练模型
            model_name,
            device=self.device,
        )

        document_texts = [ #提取需要编码的文本
            document["text"]
            for document in self.documents
        ]

        #批量生成文档向量
        self.document_embeddings = self.model.encode(
            document_texts, #4 个字符串，属于一个 batch
            convert_to_tensor=True, #返回一个整体 PyTorch Tensor，而不是 NumPy 数组。
            normalize_embeddings=True, #把每个向量进行 L2 归一化：向量长度是sqrt(x₁² + x₂² + ... + x₅₁₂²)，每个向量都除以它的长度得到归一化后的大小
            show_progress_bar=False, #只有 4 条文本，不显示进度条。
        )
        #BGE 官方实现使用第一个 [CLS] token 的最后隐藏状态作为句向量，而不是我们手写的 mean pooling。

    def search(self, query, top_k=2):
        query = query.strip() #strip() 删除字符串两端的空格和换行

        if not query:
            raise ValueError("query不能为空")

        if top_k <= 0: #top_k 表示最多返回多少条结果。返回 0 条或负数没有业务意义，所以提前报错
            raise ValueError("top_k必须大于0")

        query_text = QUERY_INSTRUCTION + query #生成查询向量

        query_embedding = self.model.encode(
            query_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # 这里传入的是一个字符串，不是字符串列表，因此返回：
        # shape  = [512]
        # dtype  = torch.float32
        # device = cuda:0

        scores = torch.matmul( #批量计算所有相似度
            self.document_embeddings,
            query_embedding,
        )
        # 运算前：document_embeddings：[4, 512]；query_embedding：[512]
        # [4, 512] @ [512] -> [4]，四组点积计算出四个数字，每个数字表示一篇文档与查询的相似度
        # 由于查询向量和文档向量都已归一化：矩阵乘法结果就是余弦相似度。

        # 两个向量 a 和 b 的点积是：a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
        # 从几何角度，点积还可以写成：a · b = ||a|| × ||b|| × cosθ
        # 因此：cosθ = (a · b) / (||a|| × ||b||)
        # 归一化后的向量长度等于 1：||a|| = 1；||b|| = 1
        # cosθ = (a · b) / (1 × 1) = a · b
        # 因此：单位向量的点积 = 余弦相似度

        # 文档向量：d = [3, 4]；查询向量：q = [4, 3]
        # 两个向量的长度都是：||d|| = sqrt(3² + 4²) = 5；||q|| = sqrt(4² + 3²) = 5
        # 现在先归一化：d_normalized = [3/5, 4/5] = [0.6, 0.8]；q_normalized = [4/5, 3/5] = [0.8, 0.6]
        # 归一化后的点积：0.6×0.8 + 0.8×0.6 = 0.48 + 0.48 = 0.96，它恰好等于余弦相似度。

        actual_top_k = min( #防止 top_k 超出文档数量
            top_k,
            len(self.documents),
        )

        top_scores, top_indices = torch.topk( #取最高分和对应下标
            scores,
            k=actual_top_k,
        )

        #把 GPU Tensor 转成 Python 数据
        score_values = top_scores.cpu().tolist()
        index_values = top_indices.cpu().tolist()

        results = []

        #恢复原始文档和 metadata
        for score, index in zip(
            score_values,
            index_values,
        ):
            document = self.documents[index]

            result = document.copy() #copy() 创建一个新的浅拷贝字典：原始文档：保持不变；检索结果：额外拥有 score
            result["score"] = round(score, 4)

            results.append(result)

        return results


def main():
    documents = [
        {
            "text": "驾驶员可以通过语音指令调节空调温度。",
            "source": "智能座舱用户手册",
            "page": 12,
        },
        {
            "text": "方向盘上的语音按键可以唤醒语音助手。",
            "source": "智能座舱用户手册",
            "page": 15,
        },
        {
            "text": "导航系统可以规划路线并播报道路信息。",
            "source": "车载导航说明书",
            "page": 8,
        },
        {
            "text": "胎压异常时，仪表盘会显示报警信息。",
            "source": "车辆安全手册",
            "page": 27,
        },
    ]

    retriever = EmbeddingRetriever(documents=documents)

    #检查向量长度
    embedding_norms = torch.linalg.vector_norm( #这是 PyTorch 中计算向量范数，也就是向量长度的函数。
        retriever.document_embeddings, #输入形状[4, 512]
        ord=2, #表示范数类型，2 表示计算 L2 范数，即向量长度：sqrt(x₁² + x₂² + ... + x₅₁₂²)
        dim=1, #沿每一行的 512 个元素计算。
    )

    query = "怎么调节车内温度？"

    results = retriever.search(
        query=query,
        top_k=2,
    )

    print("\nquery:", query)
    print("retrieval results:")

    for rank, result in enumerate(
        results,
        start=1, #start=1 只是让排名从 1 开始显示，不会影响列表下标
    ):
        print(
            f"{rank}. "
            f"{result['text']} "
            f"score={result['score']}"
        )
        print(
            f"   source={result['source']}, "
            f"page={result['page']}"
        )

    print("device:", retriever.document_embeddings.device)
    print("dtype:", retriever.document_embeddings.dtype)
    print("shape:", retriever.document_embeddings.shape)
    print("norms:", embedding_norms)


if __name__ == "__main__":
    main()