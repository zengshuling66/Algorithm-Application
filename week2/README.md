# 资料索引 API v0.3

## 简介

这个项目用于扫描本地资料目录，提取文件路径、文件名、后缀、大小、所在目录等元数据，并生成 Markdown 报告和 JSON 数据。

在此基础上，项目通过 FastAPI 提供查询接口，支持按数量限制、按后缀筛选、按最小文件大小过滤、查看最大文件和文件类型统计，以及通过 SSE 流式返回文件事件。扫描任务摘要使用 SQLite 持久化，可创建和查询扫描历史。

它对应后续 RAG 项目中的“资料整理与元数据构建”阶段。

## 当前功能

- 递归扫描资料目录
- 统计文件数、文件夹数和后缀分布
- 记录每个文件的基础 metadata
- 生成 Markdown 报告和 JSON 数据
- 提供 FastAPI 查询接口
- 使用 Pydantic 约束响应结构
- 使用 Query 校验接口参数
- 使用 HTTPException 转换文件系统异常
- 使用生成器逐个产生文件事件
- 使用 SSE 流式返回文件事件，并检测客户端断开
- 使用 SQLite、参数化 SQL 和事务保存扫描历史
- 使用 pytest 和 TestClient 验证正常路径、错误路径和流式接口

## 项目结构

```text
week2/
├── api.py
├── database.py
├── material_indexer.py
├── test_api.py
├── test_database.py
├── README.md
├── requirements.txt
├── data/
│   └── scan_history.db
└── output/
    ├── 资料索引报告.md
    └── 资料索引数据.json
```

`data/*.db` 是本地运行数据，由 `.gitignore` 排除，不提交到仓库。

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
GET /stream/files
POST /scans
GET /scans
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

PowerShell 中验证 SSE 流式输出：

```powershell
curl.exe -N "http://127.0.0.1:8000/stream/files?limit=3&delay=0.2"
```

流式接口依次发送 `file` 事件，并在结束时发送一个 `done` 事件。响应类型为 `text/event-stream`。

创建并查询扫描历史：

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/scans"
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/scans?limit=3"
```

`POST /scans` 扫描资料目录并写入一条 SQLite 记录；`GET /scans` 按时间倒序返回最近的扫描历史。

## 测试

```powershell
python -m pytest -q
```

当前测试覆盖数据库写入与回滚、API 参数校验、异常转换和 SSE 流式响应，预期结果为 `13 passed`。

错误状态码：

- `422`：请求参数校验失败
- `500`：服务端扫描目录配置错误
- `503`：文件系统或数据库暂时不可用

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
