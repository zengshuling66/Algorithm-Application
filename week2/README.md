# 资料索引 API v0.2

## 简介

这个项目用于扫描本地资料目录，提取文件路径、文件名、后缀、大小、所在目录等元数据，并生成 Markdown 报告和 JSON 数据。

在此基础上，项目通过 FastAPI 提供查询接口，支持按数量限制、按后缀筛选、按最小文件大小过滤，以及查看最大文件和文件类型统计。

它对应后续 RAG 项目中的“资料整理与元数据构建”阶段。

## 当前功能

- 递归扫描资料目录
- 统计文件数、文件夹数和后缀分布
- 记录每个文件的基础 metadata
- 生成 Markdown 报告和 JSON 数据
- 提供 FastAPI 查询接口
- 使用 Pydantic 约束响应结构
- 使用 Query 校验接口参数

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

在 `week2` 目录下执行：

```powershell
python -X utf8 .\material_indexer.py
python -X utf8 .\material_indexer.py --top-n 5
python -m uvicorn api:app --reload
```

## 接口

```text
GET /health
GET /summary
GET /files
GET /files/by-suffix
GET /largest-files
GET /extensions
GET /docs
```

常用示例：

```text
http://127.0.0.1:8000/files?limit=3
http://127.0.0.1:8000/files?limit=5&min_size=1048576
http://127.0.0.1:8000/files/by-suffix?suffix=pdf&limit=5
http://127.0.0.1:8000/files/by-suffix?suffix=pdf&limit=5&min_size=1048576
http://127.0.0.1:8000/largest-files?limit=5&min_size=104857600
http://127.0.0.1:8000/extensions
http://127.0.0.1:8000/docs
```

## 输出文件

运行索引脚本后，会在 `output` 目录下生成：

- `资料索引报告.md`
- `资料索引数据.json`

## 后续方向

- 文档内容解析
- chunk 切分
- embedding 生成
- 向量检索
- 检索结果重排
- 基于检索结果生成答案