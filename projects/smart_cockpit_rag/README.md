# 智能座舱手册 RAG

面向车主手册的可溯源问答原型。系统从 PDF 正文和原生文本表格检索证据，返回答案、原文片段和 PDF 物理页码；证据不足或适用范围不明时，允许拒答。

## 功能

- **手册入库**：区分正文与表格行，保存资料身份、PDF 版本、页码和文本块。
- **混合检索**：本地 BGE 向量召回与 BM25 关键词召回，经 RRF 融合和 Cross-Encoder 精排。
- **问答服务**：提供命令行、FastAPI 普通问答和 SSE 流式问答，由服务端把模型返回的证据 ID 还原为出处。
- **评测与复现**：可保存检索快照、独立复核答案、导出 Markdown 报告；支持在 Linux 或 WSL 2 中使用 Docker 运行。

## 架构

~~~text
入库  PDF → pdfplumber 解析 → 正文块 / 表格行 → BGE 编码 → 本地 Qdrant
问答  问题 → 向量召回 + BM25 → RRF → 精排 → 证据与问题 → LLM → 校验与引用还原
~~~

FastAPI 默认每路召回最多 10 块，融合后最多送 10 块给精排，最终给生成模型最多 3 块。LangChain Core 用于文档适配、提示模板和问答链编排。向量编码和精排在本机运行，生成通过兼容 Chat Completions 的外部服务。

## 快速开始

以下为 **Windows PowerShell + Python 3.11** 的本机流程。在项目目录执行。Linux / WSL 2 用户可直接使用 [Docker 复现流程](docs/PROJECT_REVIEW.md#docker-复现与日常操作)。

### 1. 安装依赖和准备权重

~~~powershell
conda create -n cockpit-rag python=3.11 -y
conda activate cockpit-rag
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python -m pip check
~~~

首次准备权重需要访问 Hugging Face。下载后，API 启动阶段只从已有缓存加载模型。

~~~powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
python -B -c "from sentence_transformers import SentenceTransformer; from retrieval import MODEL_NAME; from reranking import build_reranker; SentenceTransformer(MODEL_NAME, device='cpu'); build_reranker(local_files_only=False)"
$env:HF_HUB_OFFLINE = "1"
~~~

### 2. 导入手册

自备有权使用的 PDF。本仓库不分发原始汽车手册。[manual_metadata.json](manual_metadata.json) 绑定服务启动时使用的资料身份和文件版本。先确认 PDF 的 SHA-256 与配置中的 `content_hash` 一致：

~~~powershell
$pdf = (Resolve-Path -LiteralPath 'C:\path\to\train_a.pdf').Path
$manual = Get-Content -Raw -Encoding UTF8 .\manual_metadata.json | ConvertFrom-Json
$hash = (Get-FileHash -LiteralPath $pdf -Algorithm SHA256).Hash
if ($hash -ne $manual.content_hash) { throw "PDF hash does not match manual_metadata.json" }
python -X utf8 -B retrieval.py $pdf $manual.source_id
~~~

换成自己的手册时，先修改 `manual_metadata.json` 的 `source_id`、`content_hash` 和适用元数据，再入库。当前版式规则针对示例手册；其他 PDF 先抽查正文阅读顺序、表格和页眉，再进行问答。入库会写入 `.qdrant/`，并清理同一资料版本中不属于本次完整产出的旧块。**不要在 API 运行时对同一目录执行入库。**

### 3. 启动 API

以下是已验证的 DeepSeek 官方配置。密钥只在当前进程环境中设置，不写入项目文件。发起问答前，请确认允许将问题及检索出的片段发送给生成服务。

~~~powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
$key = Read-Host "DeepSeek API Key" -AsSecureString
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
python -X utf8 -B -m uvicorn api:app --host 127.0.0.1 --port 8001 --workers 1
~~~

在另一个终端检查服务，或打开 [交互式 API 文档](http://127.0.0.1:8001/docs)：

~~~powershell
curl.exe -i http://127.0.0.1:8001/health
~~~

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 查看就绪状态和已加载的资料身份，不调用生成模型 |
| `POST /qa` | 接收 `{"question":"你的问题"}`，返回答案、引用和模型信息 |
| `POST /qa/stream` | 接收相同请求体，返回 SSE；以 `final` 为最终结果，`delta` 只是临时片段 |

也可在**服务未运行时**单独检查检索，不需要生成密钥：

~~~powershell
$manual = Get-Content -Raw -Encoding UTF8 .\manual_metadata.json | ConvertFrom-Json
python -X utf8 -B langchain_retrieval.py $manual.source_id $manual.content_hash "如何设置无钥匙解锁？" rerank
~~~

## 评测与测试

- 离线回归展示配置为融合上限 **20**、精排 **Top 5**，与上述 API 默认配置不同。同一批 100 道回归题最近三轮整题通过数为 **95、96、96**。第 2 轮的指标、分母和失败案例见 [评测报告](final_eval_v2.6_rrf20_top5.md)。这些题曾用于分析和优化，不是独立盲测或线上准确率。
- 不重新调用模型，可从已有复核数据重新导出报告：

~~~powershell
python -X utf8 -B evaluation.py export final_qa_rerank_v2.6_rrf20_top5_review_r2.json final_eval_local.md rerank
~~~

- 运行代码测试：`python -B -m pytest -q`。离线测试通过不等于问答质量通过。

## 代码入口

| 入口 | 职责 |
| --- | --- |
| [retrieval.py](retrieval.py) | 解析、切块、编码与入库 |
| [langchain_retrieval.py](langchain_retrieval.py) | 查看向量、混合或精排检索 |
| [langchain_query.py](langchain_query.py) | 结构化问答链；CLI 默认混合检索，不走 API 默认精排 |
| [api.py](api.py) / [api_schemas.py](api_schemas.py) | FastAPI 服务与输入输出模型 |
| [evaluation.py](evaluation.py) | 检索快照、生成回放、复核和报告导出 |
| [Dockerfile](Dockerfile) | CPU 容器镜像 |

完整调用链、技术选型、排障和面试问答见 [项目复盘](docs/PROJECT_REVIEW.md)。

## 使用边界

- 本项目验证的是单手册、单进程的本地原型，不提供生产级并发、多租户权限或通用 OCR。扫描页、复杂表格和双栏文本仍可能漏读或误读。
- 程序能检查输出结构和引用身份，不能自动证明原文支持答案中的全部结论。安全相关操作以适用车型的官方手册为准。
- 对外发布仓库前，单独核对代码许可证，以及评测题、原文摘录、报告、缓存和密钥等内容是否适合分发。不要将本地评测材料默认视为可公开发布的数据集。
