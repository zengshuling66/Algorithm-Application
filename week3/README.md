# Week 3：PyTorch 与大模型基础

本目录用于学习 PyTorch 张量、自动求导和神经网络训练的基础机制，并为后续 Embedding、Attention 和 Transformer 学习做准备。

## 当前进度

### Day 1：Tensor 与自动求导

- Tensor 的 `shape`、`dtype`、`device`、`ndim` 和 `numel`
- 张量索引、切片、聚合、广播和形状变换
- `reshape`、`unsqueeze`、`squeeze`、`transpose` 和 `permute`
- 逐元素运算、点积、矩阵乘法与 `Q @ K^T` 的形状变化
- `requires_grad`、计算图、反向传播和梯度累加
- `torch.no_grad()`、梯度清零和梯度下降
- 手写一元线性模型训练循环，使参数收敛到 `weight=2`、`bias=1`

### Day 2：标准训练框架与数据加载

- 使用 `nn.Module` 和 `nn.Linear` 定义模型
- 使用 `nn.MSELoss` 和 `torch.optim.SGD` 完成标准训练循环
- 理解 `zero_grad → forward → loss → backward → step`
- 使用 `TensorDataset` 和 `DataLoader` 进行mini-batch训练
- 区分epoch、batch和step
- 划分训练集与验证集
- 使用 `model.train()`、`model.eval()` 和 `torch.no_grad()`
- 通过训练损失和验证损失判断过拟合

### Day 3：Tokenizer 与 Embedding

- 实现字符级 Tokenizer，并建立 `token_to_id` 和 `id_to_token`
- 理解 `[PAD]`、`[UNK]`、Padding、Truncation 和 Attention Mask
- 将多条文本整理成 `[batch_size, sequence_length]` 的批量张量
- 使用 `nn.Embedding` 将 Token ID 转换为 Token Embedding
- 使用 Masked Mean Pooling 得到固定维度的文本向量
- 理解 Tokenizer、Token ID、Token Embedding 和 Text Embedding 的区别

### Day 4：Sentence Embedding 与最小语义检索

- 使用 `BAAI/bge-small-zh-v1.5` 生成 512 维句向量
- 理解 L2 归一化、点积与余弦相似度
- 使用矩阵乘法批量计算查询和文档相似度
- 使用 `torch.topk` 返回高分文档
- 保留并返回 `text`、`source`、`page` 和 `score`
- 实现最小可复用 `EmbeddingRetriever`

## 文件说明

```text
week3/
├── tensor_demo.py      # Tensor、维度操作和矩阵乘法
├── autograd_demo.py    # 自动求导、梯度下降和线性模型训练
├── nn_training_demo.py
├── dataloader_training_demo.py
├── tokenizer_embedding_demo.py
├── embedding_retriever.py
└── README.md
```

## 运行环境

- Python 3.9
- PyTorch 2.8.0
- CUDA 可用时自动使用 GPU

激活环境：

```powershell
conda activate pytorch
```