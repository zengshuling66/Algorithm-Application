# 资料索引生成器 v0.2

这个项目用于扫描本地学习资料目录，提取文件路径、文件名、后缀、大小、所在目录等 metadata，并生成 Markdown 资料索引报告。

它是后续 RAG 系统的数据准备模块。真实 RAG 项目在进行文档解析、chunk 切分、embedding 和向量检索之前，需要先把原始资料整理成结构化数据。

## 当前功能

- 递归扫描指定资料目录
- 统计文件数量和文件夹数量
- 统计不同文件后缀的数量
- 记录每个文件的基础 metadata
- 按文件大小找出最大的若干个文件
- 生成 Markdown 报告
- 将输出结果保存到 output 目录

## 项目结构

```text
week2/
├── material_indexer.py
├── README.md
├── requirements.txt
├── output/
│   └── 资料索引报告.md
└── notes/