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

## 文件说明

```text
week3/
├── tensor_demo.py      # Tensor、维度操作和矩阵乘法
├── autograd_demo.py    # 自动求导、梯度下降和线性模型训练
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

## 运行方式

在 `week3` 目录执行：

```powershell
python tensor_demo.py
python autograd_demo.py
```

`autograd_demo.py` 的训练结果应表现为损失持续下降，最终参数接近：

```text
weight = 2
bias = 1
```
