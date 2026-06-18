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
- 生成 JSON 数据
- 将输出结果保存到 output 目录
- 处理单个文件访问异常

## 项目结构

```text
week2/
├── api.py
├── material_indexer.py
├── README.md
├── requirements.txt
└── output/
    ├── 资料索引报告.md
    └── 资料索引数据.json
```

## 运行方式

生成资料索引报告：

```powershell
python -X utf8 .\material_indexer.py
```

启动 FastAPI 服务：

```powershell
python -m uvicorn api:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/summary
http://127.0.0.1:8000/files
http://127.0.0.1:8000/files?limit=3
```

生成最大的 5 个文件报告：

```powershell
python -X utf8 .\material_indexer.py --top-n 5
```