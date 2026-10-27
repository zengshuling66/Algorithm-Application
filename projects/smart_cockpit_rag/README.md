# 智能座舱汽车知识大脑 RAG

基于车主手册的检索增强问答项目。将 PDF 解析为带来源信息的文本块，使用本地 BGE 编码、Qdrant 持久化检索和兼容 Chat Completions 的模型接口生成回答，并展示资料引用。

当前提供 LangChain 编排的命令行问答、直接函数调用基线与初版检索评测，定位为单资料工程化原型，不宣称生产上线或完整质量验收。

## 功能与架构

- 使用 pdfplumber 分离正文与检测到的原生文本表格，按单元格几何关系展开合并格，并为表格行保留列标签及相邻文字。
- 正文采用字符窗口切块，表格行不再按字符窗口拆分；保留文件名、物理页码、内容类型和解析版本。
- 文件内容哈希、稳定 chunk ID 和幂等写入，支持按资料身份及文件版本过滤。
- 通过绑定资料身份与哈希的配置读取品牌、未知车型/年款及页眉区域，提取章节并写入数据库 payload。
- 本地 BGE 向量检索、BM25 关键词检索及 RRF 融合，两路使用同一资料身份和文件版本范围。
- 仅检索入口可选预训练 cross-encoder 精排，对 RRF 前10候选重新打分后取3条；问答默认仍为不精排的混合检索。
- 结构化答案与引用契约：模型只返回回答状态、答案和证据 ID，程序检查 JSON、字段及 ID，再从本次可信证据映射组装来源与页码；不等于自动验证答案事实或引用支持关系。
- 使用 LangChain Document、Runnable 和 ChatPromptTemplate 组织原有检索及生成模块，保留证据身份、排名和运行状态。
- 开发集与冻结测试集分离，支持题集字段校验、批量向量检索观察，以及切块、向量库、Prompt、RRF、精排与答案契约自动测试。

```text
资料入库：PDF + source_id + Metadata 配置 -> 正文/表格记录 -> chunks -> BGE 向量 -> Qdrant
在线问答：问题 + source_id + content_hash -> 同范围向量/BM25 各取候选 -> RRF -> Top3 Document -> 本次证据 ID 映射 -> JSON Prompt -> 兼容 SDK/LLM -> JSON 与 ID 校验 -> 可信引用
可选精排检索：同范围两路候选 -> RRF Top10 -> cross-encoder -> Top3 Document（尚未接入问答入口）
检索评测：题集 -> 向量检索 -> 对照标注来源与页码 -> JSON 报告
```

| 模块 | 职责 |
| --- | --- |
| [table_parser.py](table_parser.py) | 定位表格、展开合并格、渲染带列标签的数据行，分离正文与邻文 |
| [manual_metadata.json](manual_metadata.json) | 当前资料的身份、哈希、品牌、未知车型/年款及页眉位置配置 |
| [ingestion.py](ingestion.py) | 计算 SHA-256、生成正文/表格记录、提取章节、记录无文本页 |
| [chunking.py](chunking.py) | 正文窗口切分、表格行整体保留，传递 metadata 并生成稳定 chunk ID |
| [retrieval.py](retrieval.py) | 文本与问题编码、入库入口、持久化检索及 FAISS 基线 |
| [vector_store.py](vector_store.py) | Qdrant 集合检查、写入、过滤检索、同范围 payload 分批读取与资源关闭 |
| [bm25_retrieval.py](bm25_retrieval.py) | 中文分词、构建范围内 BM25 统计、按关键词召回，不读取原 PDF 或调用生成模型 |
| [hybrid_retrieval.py](hybrid_retrieval.py) | 两路候选召回、按 chunk ID 合并、RRF 排序，保留各路名次与原始分数 |
| [reranking.py](reranking.py) | 加载固定版本的预训练 cross-encoder，对问题与候选正文打分，保留原记录并按精排分排序 |
| [prompt_builder.py](prompt_builder.py) | 编号证据，组织回答规则、来源与页码 |
| [generation.py](generation.py) | 调用生成接口，检查结束标记与非空内容，返回答案、模型及用量 |
| [answer_contract.py](answer_contract.py) | 为最终 Document 建立本次证据映射，解析模型 JSON、检查字段及引用 ID，并从映射回取完整来源 |
| [query.py](query.py) | 保留直接函数调用的问答基线 |
| [langchain_adapter.py](langchain_adapter.py) | 将检索字典转换为 Document，保留正文、chunk ID、metadata 和检索排名 |
| [langchain_retrieval.py](langchain_retrieval.py) | 用 Runnable 包装向量、混合或混合后精排及 Document 转换，提供可选择模式的仅检索入口 |
| [langchain_prompt.py](langchain_prompt.py) | 保留原文本 Prompt，另用 ChatPromptTemplate 构建只向模型提供证据 ID/正文的结构化消息 |
| [langchain_query.py](langchain_query.py) | 默认混合检索，通过 prepare/answer 串联证据映射、结构化生成及校验，保留运行状态、回答状态、可信引用、模型和用量 |
| [evaluation.py](evaluation.py) | 校验题集，记录逐题或批量向量检索结果，不调用生成 API |
| [test_chunking.py](test_chunking.py)、[test_vector_store.py](test_vector_store.py) | 切块边界、metadata、持久化、幂等写入与范围隔离测试 |
| [test_langchain_prompt.py](test_langchain_prompt.py) | 新旧 Prompt 消息一致性与空证据校验测试 |
| [test_hybrid_retrieval.py](test_hybrid_retrieval.py) | RRF 合并顺序、总分、各路名次与原始分数、输入不变及空列表检查 |
| [test_reranking.py](test_reranking.py) | 使用假模型检查精排排序、同分顺序、原记录不变、空输入及非法分数，不评估真实模型质量 |
| [test_answer_contract.py](test_answer_contract.py) | 检查合法回答/拒答及 JSON、字段、状态、类型、重复/未知 ID 和无引用回答的拒绝路径 |

Qdrant 是当前持久化检索后端，使用本地模式，无需额外启动服务器。FAISS `IndexFlatIP` 保留为精确检索基线；入库入口仍执行其自检，但独立问答不重建 FAISS 索引，也不重新解析或编码整份 PDF。

LangChain 只负责数据适配与流程编排，实际召回由 BGE/Qdrant 和 BM25 完成，融合由独立函数执行，生成仍调用现有 SDK 适配函数；索引和最终 top-k 未因框架接入而改变。结构化 Prompt 和引用契约由项目代码显式实现，不是框架自动赋予的能力。当前使用 `langchain-core`，未引入 LangGraph、自动引用语义评测或真实流式输出。

BM25 从当前 Qdrant 集合按 `source_id` 和 `content_hash` 读取已入库的 payload，不重新解析 PDF 或编码整份资料。构建检索器时建立一次分词和词频统计，同一进程内可复用；语料更新后须重新构建检索器。向量路仅编码当前问题，两路目前顺序执行，不宣称并行或具备大规模稀疏索引服务能力。

精排直接读取候选正文，不修改 BGE 向量或数据库。构建精排检索器时加载一次模型，同一链的多次调用可复用；新启命令仍重新加载。精排只能重排已召回候选，不能补回未进入候选集的证据，也不保证领域表格排序一定改善。

## 快速开始

以下示例使用 Windows PowerShell。进入项目目录后，入库、问答和评测命令均在同一目录执行。

在新开的终端中设置以下编码，尤其用于把 Python 输出通过管道交给 `Tee-Object` 时；同一会话中无需逐条命令重复设置：

```powershell
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
```

`$OutputEncoding` 控制 PowerShell 传给外部程序的文本编码，控制台输出编码也须与 Python 输出匹配；仅启用 Python 的 `-X utf8` 不会替 PowerShell 设置读取编码。Windows PowerShell 5.1 的 `Tee-Object` 日志文件可能使用 UTF-16，这与文本已发生错误解码是两回事。jieba 的普通启动信息写到 stderr，可能被 PowerShell 标红；须结合 traceback、退出码和后续结果判断，不能单凭红色文本认定调用失败。

### 1. 准备环境

当前项目环境为 Windows、Conda 环境 `cockpit-rag`、Python 3.11.17、PyTorch `2.8.0+cpu`、FAISS CPU 1.11.0、NumPy 1.26.4 和 `langchain-core==1.6.13`；主要依赖版本见 [requirements.txt](requirements.txt)。原 `pytorch` / Python 3.9 环境仅保留用于历史基线，不作为当前依赖的安装目标。

从仓库根目录执行以下参考安装步骤；整套流程尚未在全新环境中验证：

```powershell
conda create -n cockpit-rag python=3.11
conda activate cockpit-rag
cd .\projects\smart_cockpit_rag
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

本机环境已配置完成，直接激活即可，不要再次创建同名环境。上述安装步骤面向新机器；现有环境的依赖检查、缓存模型和测试已通过，但完整的干净环境复现仍待最终验收。当前使用 CPU，不要求安装 CUDA；导入 `retrieval.py` 仍会导入 FAISS，因此该依赖暂不能移除。

首次使用 `BAAI/bge-small-zh-v1.5` 需要下载权重或准备可用的本地缓存；安装 Python 包不等于模型权重已就绪。问答与批量评测明确使用 CPU 编码。

本地缓存完整时，可在当前 PowerShell 中设置 `$env:HF_HUB_OFFLINE = "1"`，避免 Hugging Face 联网检查和重试。首次下载其他模型前执行 `Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue`，然后重新运行脚本；仅不再输入设置命令不会清除当前会话中的已有值。该设置不替代模型缓存，也不关闭其他供应商的生成 API。

LangChain 检索和问答入口另外设置了 `local_files_only=True`，因此即使清除离线环境变量，也不会通过这两个入口自动下载权重；首次准备当前 BGE 缓存可在允许联网时使用下面的入库入口。

精排另需 `BAAI/bge-reranker-base` 权重，代码固定 revision 为 `2cfc18c9415c912f9d8155881c133215df768a70`，不能用 BGE Embedding 缓存代替。现有机器已缓存；新机器首次准备时，先解除上述离线设置，再由使用者在允许联网时执行 `python -B -c "from reranking import build_reranker; build_reranker(local_files_only=False)"`。日常入口默认只读缓存，缺少权重会报错；精排不需要生成 API Key。

### 2. 导入资料

准备有权使用的 PDF。以下路径对应本地训练营样本，项目不附带原始手册。

当前 [manual_metadata.json](manual_metadata.json) 只对应 `training_manual_a` 及其中登记的文件哈希。解析入口先比较配置与实际输入，不匹配会停止，防止把已有资料的品牌和版式信息误用于其他文件。处理新资料或新文件版本前，须核对并更新对应配置；当前尚无多资料配置注册表或任意版式自动识别。`header_y=[95,120]` 是本手册页眉的 PDF 坐标，章节从该区域的实际文字提取，不按页码硬编码；无页眉时保留 `None`。

```powershell
$pdf = "..\..\..\阶段12：智能座舱汽车知识大脑\丁师兄综合项目实战项目\综合项目实战项目一\data\train_a.pdf"
$sourceId = "training_manual_a"
python -B retrieval.py $pdf $sourceId
$contentHash = (Get-FileHash -LiteralPath $pdf -Algorithm SHA256).Hash.ToLowerInvariant()
```

该入口完成解析、切块、向量编码与写入，并输出数量及检索自检结果。同一份资料的文件内容更新后沿用其 `source_id`，先同步 Metadata 配置，再重新入库并使用新文件哈希查询；已有资料的日常问答无需再次入库。

本次仅新增 Metadata 时，正文和 chunk ID 不变，相同 ID 的 upsert 会更新原记录，当前数量仍为604。仅修改配置或内存记录不会自动刷新数据库；现有入口仍会重复编码文本，并非已经实现仅更新 payload。原始 PDF 哈希标识文件字节，`pipeline_version` 标识解析方案；当前新旧流程使用不同集合，旧库保留作基线。

仅检查解析或切块、不写数据库时，可分别执行：

```powershell
python -B ingestion.py $pdf $sourceId
python -B chunking.py $pdf $sourceId
```

### 3. 配置生成接口

`generation.py` 从当前进程环境变量读取以下配置，未实现自动加载 `.env`：

| 环境变量 | 含义 |
| --- | --- |
| `LLM_API_KEY` | 所选接口的密钥，不写入代码、日志或 Git |
| `LLM_BASE_URL` | 供应商允许用于当前任务的 API 根地址 |
| `LLM_MODEL` | 该地址下可调用的模型 ID |

先单独执行密钥输入命令，等待提示后粘贴真实密钥并回车：

```powershell
$secureKey = Read-Host "Paste API Key here" -AsSecureString
```

再在同一个终端配置当前使用的 DeepSeek 官方接口，密钥必须来自该供应商，不能沿用方舟密钥：

```powershell
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
```

这些变量仅对当前终端及其子进程生效。输入隐藏不代表密钥全程加密：转换后密钥以普通字符串保存在进程环境中；不要打印或截图密钥。

2026-10-22已用上述配置验证 LangChain 单题真实问答，实际响应模型为 `deepseek-flash`。接口地址和模型名参见 [DeepSeek 官方接入文档](https://api-docs.deepseek.com/)。更换供应商不需要重新编码同一份资料，但前后质量对照须固定生成条件，不能把供应商变化归为检索算法收益；模型别名也不保证底层权重永久不变。

当前请求使用 `max_tokens=1024`、非流式输出、60 秒网络操作超时、零自动重试，以及通过 `extra_body` 传入 `thinking.type=disabled` 关闭思考，参见 [DeepSeek 思考模式配置](https://api-docs.deepseek.com/guides/thinking_mode/)。更换供应商时须确认这些字段兼容；仅支持 Chat Completions 协议不代表支持所有扩展参数。

LangChain 问答通过 `generate_json_answer()` 开启 `response_format={"type":"json_object"}`，Prompt 包含 JSON 要求及示例，参见 [DeepSeek JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)。`generate_answer()` 的 `json_mode` 默认关闭，保留直接问答基线。JSON Output 不等于严格 JSON Schema 或事实保证；截断和空内容由生成函数报错，非法 JSON 或引用契约由解析器报错，不自动修复或伪装成正常拒答。

### 4. 运行问答

已有资料无需再次入库。在配置好生成接口的同一个终端，从当前资料配置读取身份和文件哈希，运行默认混合检索的 LangChain 问答入口：

```powershell
$manual = Get-Content -Raw -Encoding UTF8 .\manual_metadata.json | ConvertFrom-Json
$sourceId = $manual.source_id
$contentHash = $manual.content_hash
$question = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
python -B langchain_query.py $sourceId $contentHash $question
```

输出包含 Prompt 版本、运行层 `status`、答案、回答层 `answer_status`、程序组装的 `citations`、每条候选证据的 chunk ID/物理页码/正文，以及实际响应模型和 token 用量。`generated` 表示生成和 JSON/字段/ID 检查完成，不表示答案事实或引用支持关系已自动验证；模型可在此状态下返回 `answered` 或 `refused`。`no_evidence` 表示检索为空，程序直接返回 `refused`，模型和用量为 `None`，不会发起生成请求。

最终 Document 按顺序映射为本次的 `E1`、`E2` 等编号，不是 PDF 页码或跨请求稳定身份。模型只返回 `answer_status`、非空 `answer` 和无重复字符串 `evidence_ids`；`answered` 至少引用一条本次证据，`refused` 可引用相关证据或返回空列表。来源文件、资料身份、文件哈希、页码、chunk ID 和正文由程序从映射表回取，不采信模型另写的来源字段。合法 ID 不保证结论由该证据支持。

入口逐条打印的 `evidence:` 是全部候选上下文，不代表模型引用了全部候选。返回状态保留 Document、证据映射和消息，但当前 CLI 不自动保存机器可读生成报告；终端验收脚本保存的混合日志也不冒充完整评测报告。直接 `query.py` 仍使用旧文本 Prompt 及模型生成的引用，未同步为结构化契约。

仅查看检索、不调用生成模型或配置 API Key，可分别运行：

```powershell
python -B langchain_retrieval.py $sourceId $contentHash $question vector
python -B bm25_retrieval.py $sourceId $contentHash $question
python -B langchain_retrieval.py $sourceId $contentHash $question hybrid
python -B langchain_retrieval.py $sourceId $contentHash $question rerank
```

`langchain_retrieval.py` 省略最后的模式参数时仍默认 `vector`；这与 `langchain_query.py` 的混合问答默认值不同，后者尚不接受精排模式。混合输出的 `score` 是 RRF 总分，`retrieval_ranks` 保存两路名次，`retrieval_scores` 保存两路原始分数；精排额外保留 `rrf_rank` 和 `rerank_score`，不覆盖原 `score`。Document 的 `retrieval_rank` 是当前模式输出的最终名次，因此精排后的 `score` 不一定降序。精排分数为原始 logit，可为负数，不是概率；不同分数不能直接比较或套用同一个拒答阈值。

从题集选取单题时，Windows PowerShell 5.1 应先用括号完成 JSON 转换，再逐题过滤，避免把数组整体交给筛选器。例如：`$question = ((Get-Content -Raw -Encoding UTF8 .\table_eval_cases_v1.json | ConvertFrom-Json) | Where-Object { $_.id -eq "table_07" }).question`。

原向量问答基线仍可通过 `python -B query.py $sourceId $contentHash $question` 运行；其空结果提示为 `No matching evidence found.`，不输出上述链状态。

存在检索结果时，两个问答入口都会发起真实生成请求并产生供应商对应费用；回答为空或 `finish_reason` 不是 `stop` 时抛出异常。空检索拦截和 Prompt 拒答规则不等于可靠的证据充分性判断。模型在链外加载，可在同一进程中复用；每次重新运行命令仍会重新创建进程并加载模型。

用完密钥后可清除当前终端变量：

```powershell
Remove-Item Env:LLM_API_KEY -ErrorAction SilentlyContinue
Remove-Variable secureKey -ErrorAction SilentlyContinue
```

## 数据与索引约定

| 项目 | 当前实现 |
| --- | --- |
| Embedding | 本地 `BAAI/bge-small-zh-v1.5`，512 维，资料与问题均归一化 |
| 切块 | 正文为页内500字符窗口、100字符重叠；每条表格数据行及其上下文整体保留 |
| 向量召回 | Qdrant COSINE；当前客户端本地模式按向量距离计算排序，未启用 HNSW 近似检索 |
| BM25 召回 | `jieba` 分词、英文小写化；`BM25Okapi(k1=1.5, b=0.75)`，排除无查询词重合的候选 |
| 混合候选 | `hybrid_retrieval.py` 中 `DEFAULT_CANDIDATE_K=10`，两路分别最多10条；这是初始值，不是已调优结论 |
| 融合 | 按 `chunk_id` 合并，以各路 `1 / (60 + rank)` 累加排序，同分按 ID 排序；每路输入约定 ID 唯一 |
| 精排 | `BAAI/bge-reranker-base`，固定上述 revision，CPU / float32，问题与正文配对输入，`max_length=512`、`batch_size=4`、Identity 输出原始分数；同分按原RRF名次排序 |
| 最终数量 | `vector_store.py` 中 `DEFAULT_TOP_K=3`；混合直接取最终3条，可选精排先保留RRF前10再取3条，向量基线与评测仍引用此值 |
| 持久化 | 本项目目录下 `.qdrant/`，已通过 `.gitignore` 忽略 |
| 当前集合 | `cockpit_manual_bge_small_zh_v1_5_tables_v1`，604条 |
| 历史基线集合 | `cockpit_manual_bge_small_zh_v1_5`，469条，不因切换常量而自动重建或删除 |
| 解析与 Metadata | `pipeline_version=tables-v1`；品牌、车型、年款、章节与来源字段随 payload 保存，未知值保留 `None` |
| Prompt | LangChain 结构化问答使用 `cockpit-rag-json-v1`；直接文本问答及旧消息一致性基线保留 `cockpit-rag-v1` |

上述切块及检索值是当前源码默认值，不是已开放的命令行配置项。

每条数据库记录包含稳定 UUID、向量与 payload；payload 保存 `chunk_id`、`text`、`source`、`source_id`、`content_hash`、`page`、`chunk_index`、`char_count`，以及 `pipeline_version`、`block_id`、`kind`、`brand`、`vehicle_model`、`model_year`、`section`。导入时必须保持 `chunks[i]` 与 `vectors[i]` 对应，查询时由数据库返回正文与 metadata。

- `source_id` 是资料身份，`content_hash` 是文件字节的 SHA-256 指纹，两者不是已核实的车型、年款或官方版本号。查询要求两者同时匹配，不根据哈希反查 PDF 路径或自动选择最新版本。
- `chunk_id` 包含资料身份、文件哈希、切块参数、物理页码、块序号，以及当前解析版本和块身份，再通过 UUID5 映射为数据库 ID。版本与块身份避免同一页多条表格记录的序号碰撞。相同 ID 的 upsert 更新原记录，不增加重复条目；当前入口仍重复解析和编码，并进行重复写入自检。
- 只修改独立 Metadata，且实际编码文本、模型和编码配置均不变时，可以复用向量；当前尚无仅更新 payload 的专用入口。字段已入库不等于已实现按车型/章节过滤或自动适用性判断。
- 文件内容变化会产生新 ID，旧版本不会自动删除。更换 Embedding 模型或编码配置时须重新编码资料并建立匹配索引；向量维度相同不代表语义空间兼容。
- 页码为从 1 开始的 PDF 物理页序号，不保证与印刷页码一致。原始资料的获取和再分发权限须另行确认，不将版权不明的原文上传公开仓库。

## 评测与测试

### 检索观察

题集按文件分离，当前共40条：

| 文件 | 划分 | 可回答 / 不可回答 | 用途 |
| --- | --- | --- | --- |
| [eval_cases.json](eval_cases.json) | dev，20条 | 16 / 4 | 普通问答、改写、无答案及资料适用范围回归 |
| [table_eval_cases_v1.json](table_eval_cases_v1.json) | dev，10条 | 8 / 2 | 表格解析与检索回归 |
| [eval_test_v1.json](eval_test_v1.json) | test，10条 | 8 / 2 | 冻结的问题留出集，用于配置固定后的最终评测 |

样本包含唯一 `id`、问题、`answerable`、题目来源和 `split`；可回答题还包含参考答案及来源、页码、定位词，不可回答题的答案与证据必须为 `null`。不可回答题的 `review_notes` 保存相关章节缺失信息的人工核验说明，当前保留但不参与评分。

参考答案和标注证据只用于检索后的核对，不传入被测 RAG 的过滤条件或生成 Prompt。字段校验不替代原文核验；当前脚本读取传入文件中的全部题目，不自动根据 `split` 分组。

```powershell
python -B evaluation.py eval_cases.json
python -B evaluation.py eval_cases.json $sourceId $contentHash retrieval_report_v2.json
```

第一条命令只校验字段，不加载 Embedding 权重或调用生成 API。第二条加载一次 BGE 后复用，保存逐题记录；它会覆盖同名报告，应为不同实验使用不同文件名。当前 `evaluation.py` 仍调用向量检索，不会随 LangChain 问答入口切换而自动评测 BM25 或 RRF；完整多策略报告尚未接入。

表格开发集可替换上述命令的题集参数使用；历史表格真实问答前后对照复用 `query.py`，不由 `evaluation.py` 自动执行生成或评分。

`eval_test_v1.json` 已核对原文、完成字段校验，并于2026-10-22登记冻结基线；与30条dev无重复题号或完全相同问题，按问题族复核，不将同题改写移入test。当前仅为同一手册的问题留出集，不代表跨资料泛化评测。开发与方案选择只用dev；test尚未运行检索或生成，待最终配置固定后使用，不根据test结果筛题或调参数。标注确需纠错时记录原因与新版本，不静默覆盖。

冻结时题集文件的字节 SHA-256 为 `4b2e8373a12a7f66b7f3d2563bbeef2a619d96c977b6f0b57c5a41cac3e1c226`，可用 `Get-FileHash -LiteralPath eval_test_v1.json -Algorithm SHA256` 核对。它不是PDF的 `content_hash`，格式化也会改变该指纹；冻结是使用与版本约定，不是文件只读锁定。测试集字段校验通过不表示检索、生成或拒答质量通过，当前未报告test准确率。

| 报告字段 | 含义 |
| --- | --- |
| `retrieved_pages`、`scores` | 按相同检索排名对应的命中页和相似度分数 |
| `evidence_page_hit` | 可回答题有任一结果同时命中标注身份及页码为 `true`，否则为 `false`；不可回答题为 `null` |
| `generation_status` | 固定为 `not_run`，表示该批量任务没有运行生成，不是生成失败或拒答通过 |

### 验证快照

核对范围：表格前后对照、Metadata落库、LangChain接入与历史单题问答、混合检索和可选精排的有限dev观察，以及2026-10-27结构化问答四场景验收和26项自动测试。当前真实语料仍只有 `train_a.pdf`。

| 项目 | 已核验结果 |
| --- | --- |
| 页面 | 共354页，334页有提取文本；20个无文本页经渲染及原PDF检查仅见空白和裁切标记，可排除 |
| 结构化记录 | 476条正文/表格记录，切块604条，其中142条表格记录对应的chunk |
| 表格范围 | 全书检测到27个候选区域，不等于任意表格完整覆盖；未实现OCR或跨页表格自动拼接 |
| 新旧数据库 | 当前tables-v1集合604条，历史集合469条，均已实际读取计数 |
| Metadata | 新库604/604条均有新增四字段，品牌为领克，车型/年款保留未知；抽查43/197/243/351页章节与原文一致 |

章节提取依赖当前手册的固定页眉区域。原文未确认具体车型/年款，PDF创建于2023年不能作为2023款车型依据。20页空白核验不证明其他页面的图片语义或所有表格均已正确解析。

固定10道表格开发题在同一BGE、top-3、Prompt及方舟DeepSeek基线下的人工对照如下；采用整题完整正确口径：

| 检查项 | 历史流程 | tables-v1 |
| --- | --- | --- |
| 8道可回答题命中标注页 | 8/8 | 8/8 |
| 8道可回答题完整回答正确 | 5/8，62.5% | 7/8，87.5% |
| 2道不可回答题按本组定义拒答 | 2/2 | 2/2 |

完整回答提高25个百分点，主要修复座椅210/50毫米及轮胎空载/满载的条件对应错误；不是单凭换库、提高相似度分数或命中页得出的结论。该历史向量问答保留一项回归：ISOFIX题所需的D行已在新库，但排第4，top-3仅含B/A/B1，因此D部分无法完整回答。后续混合检索已补入D行，见下面的检索观察；尚无新的完整生成核验，不据此改写本表的7/8。仅此小开发集单轮结果，不代表全库准确率或稳定拒答能力。

检索前后报告分别为 `table_retrieval_before_v1.json`、`table_retrieval_after_v1.json`；本地真实问答日志为 `table_qa_before_20261015_174625.txt`、`table_qa_after_20261016_155502.txt`，保留实际文本、回答、模型和用量。日志包含手册原文，提交前须检查资料授权及敏感信息，不默认随公开仓库分发。Metadata更新后的本次核验没有再次调用生成或产生新质量指标。

历史20题报告见 [retrieval_report_w4d7_v1.json](retrieval_report_w4d7_v1.json)：旧库16道可回答题中14道在top-3内命中标注来源和页码，4道不可回答题不做页码命中判断。**14/16是历史标注页命中比例，不是答案准确率或完整证据Recall@k**；不能与表格开发题的7/8回答正确率混用。

| 历史未命中案例 | 标注物理页 | 旧库 top-3 物理页 |
| --- | --- | --- |
| FCTA 提醒场景 | 197 | 200、198、211 |
| 空调风量简短改写 | 243 | 250、244、251 |

2026-10-16只读复查新库时，FCTA仍返回200/198/211页，未命中197页；风量简短改写已返回250/244/243页，命中243页。该复查没有生成回答或全量重测20题，不能更新整体指标，也不能把旧库风量失败继续写成新库现状。

三条端到端人工抽查使用 Prompt `cockpit-rag-v1`，响应模型均为 `deepseek-v4-1-flash`，思考 tokens 均为 0：

| 问题 | 人工核验结果 | 总 tokens |
| --- | --- | --- |
| 前排座椅通风档位与切换 | 三级及关-低-中-高循环正确，引用对应物理第 116、117 页 | 743 |
| 扶手箱 Type C 的 USB PD 版本 | 正确说明无法确定，第 132、133 页的电压、电流不足以推出协议版本 | 1021 |
| 未提供车型、年款时是否一定配有通风 | 正确保留无法确定的边界，但正文引用 `[1][2]`，依据列表另列 `[3]` | 793 |

这些结果仅有人工核验截图，尚未写入机器可读生成报告，也不会更新检索报告中的 `not_run`。少量成功样例不代表整体准确率、稳定拒答能力或模型对比结论；模型别名也不保证底层权重永久不变。

此前 LangChain 向量问答入口已用 DeepSeek 官方 `deepseek-flash` 验证无钥匙单门/全车开发题：候选物理页为43/44/41，答案与43页两种解锁规则一致，引用 `[1]` 正确；Prompt仍为 `cockpit-rag-v1`，本次输入785、输出51、合计836 tokens。该结果依据学习者提供的真实终端输出核验，仅覆盖一个dev问题，不冒充切换混合后的真实生成结果。上面的表格对照与三题快照来自历史方舟模型，不等于当前官方接口的整套质量回归，不能跨供应商直接归因或合并指标。

同一tables-v1集合、BGE和最终top-3下，2026-10-23使用当前已保存代码复查三条dev问题，未调用生成API：

| 开发题 | 向量前三条 | RRF融合前三条 | 观察 |
| --- | --- | --- | --- |
| FCTA 提醒场景 | 200、198、211页 | 200、198、197页 | 标注197页进入最终上下文 |
| `table_07` ISOFIX的D与B行 | `p125:t2:r7`、`r9`、`r8` | `p125:t2:r7`、`r5`、`r9` | D行`r5`由向量第4进入融合第2，与B行`r7`同时保留 |
| 无钥匙单门/全车 | 43、44、41页 | 43、44、41页 | 本题证据页未变 |

这些只是三条dev的检索观察，尤其表格必须核对行内容而非仅看125页命中；不代表全开发集、生成准确率或冻结test成绩。还以临时离线生成桩检查了混合Document到Prompt/问答链的传递及空证据不调用生成器，但该检查不属于持久化pytest套件，也不是供应商真实回答。切换后的完整生成质量仍待验证。

可选精排使用同一次 RRF 前10候选，保持资料、BGE、最终 top-3 和 Prompt 不变，有限 dev 观察如下，未调用生成 API：

| 开发题 | 不精排 | 精排后 | 判断边界 |
| --- | --- | --- | --- |
| FCTA 提醒场景 | 197页在第3名 | 197页升至第1名 | 仅该题首条证据排名改善 |
| `table_07` ISOFIX的D与B行 | B、D、A行 | 普通座椅表U/L行、D行、B行 | D/B仍完整保留，但干扰行由RRF第10升至第1，不是全面改善 |
| 无钥匙单门/全车 | 43、44、41页 | 43、44、41页 | 顺序未变 |

上述精排函数的同候选对照属于有限观察；另已运行保存后的 `rerank` CLI，确认 `table_07` 返回3条，最终名次1/2/3对应RRF名次10/2/1，身份、正文、各路记录与两类分数均保留。没有据此更新完整dev、生成或冻结test指标，尚不能判定精排应作为问答默认。

CPU / float32 下，使用 `table_07` 同一10候选预热一次，再计时5轮，学习者实际运行的精排平均耗时为0.39秒；最长配对输入207 tokens，超过512的数量为0；模型参数体积1060.655 MiB。计时不含模型加载、召回和长度统计，不是端到端延迟、P95或稳定性能承诺；参数体积不是进程RSS或峰值内存。只说明本题无截断风险，不推广到全部资料或dev。完整质量、截断与资源对照仍待统一评测。

### 结构化问答验收

本地日志 `qa_contract_dev_v2_20261027_163724.txt` 记录三条既有dev和一条内存模拟冲突，使用 `cockpit-rag-json-v1`、DeepSeek官方接口，实际响应模型均为 `deepseek-flash`。前三题走默认混合检索，最后一题用临时 Document 替代检索输出，后续仍调用同一映射、Prompt、生成和校验链，不向正式数据库写入模拟数据。

| 场景 | 已核验行为 |
| --- | --- |
| 无钥匙单门/全车 | `generated/answered`，正确区分当前车门和全车车门，引用 `E1` 对应物理第43页 |
| USB PD版本缺失 | `generated/refused`，说明资料未提供协议版本，引用为空，不根据电压或接口外形猜测 |
| 车型/年款未知 | `generated/refused`，不把通风操作说明推断为用户车辆一定配备，引用为空 |
| 相同条件的模拟矛盾资料 | `generated/refused`，指出两种结果冲突，引用 `E1/E2`；适用范围措辞略宽泛，不能由此推断实际车型差异 |

日志中文及JSON可正常读取，前三题的9条上下文身份、哈希、页码和正文均与Qdrant payload一致，返回的citations与本次映射对应。四次合计2850 tokens；这仅是小范围功能验收，不是全开发集准确率、稳定拒答保证或冻结test成绩。日志含手册正文，提交或公开前须核对资料权限和敏感信息。此前乱码日志仅保留为编码故障记录，不用于答案质量验收；本节结果不与旧供应商、旧Prompt的历史比例合并。

### 自动测试

```powershell
python -B -m pytest -q
python -m pip check
```

2026-10-27复跑26项测试通过：5项切块测试覆盖窗口与重叠、空文本、跨页metadata、非法参数和ID变化；1项临时数据库集成测试覆盖持久化、重复写入、来源/版本过滤、payload保留与空范围结果；2项旧文本Prompt测试检查新旧消息完全一致（含正文中的字面花括号）及空Document列表报错；1项RRF测试检查合并排序、总分、两路名次与原始分数、原输入未改写及空结果；4项精排契约测试检查排序与同分顺序、原记录和RRF分不变、原名次、空结果，以及分数数量、NaN和Inf异常；13项答案契约测试检查合法回答/拒答及非法JSON、字段、状态、类型、重复/未知ID和无引用回答。精排单元测试使用假模型，不下载权重或衡量模型效果；同日 `pip check` 无依赖冲突。

这些测试不调用付费API，不运行冻结test的问答。人工构造的向量用于验证存储行为，不评估Embedding语义质量；Prompt一致性也不证明答案正确或引用支持结论。完整链的空检索分支、供应商异常、生成质量与引用语义尚未纳入持久化自动回归套件，metadata正确性仍需原文抽查。

## 已知限制与后续方向

- **资料质量：**只验证一份真实PDF，支持当前可检测的原生文本表格，不保证所有表格均被发现。表头识别存在启发式边界，相邻文字只作为归属待核验的上下文，未实现OCR、跨页表格拼接或完整版面理解；部分双栏正文仍有交错，字符切块也可能打断语义。
- **适用范围：**品牌与章节字段已入库，具体车型/年款仍未知；固定页眉区域及单资料配置不直接推广到其他手册，尚无车型过滤、自动适用性判断或已核实的官方资料版本映射。
- **检索质量：**已接入BM25、向量、RRF及仅检索入口的可选cross-encoder；LangChain问答仍默认混合，尚无完整dev四策略对照。精排存在把同页另一张表的干扰行提到首位的案例，不能用页命中、分数变高或模型接通代替质量判定。每路输入ID唯一是RRF前提，当前未保留同路重复容错；扩展召回源时须维持此约定。10条冻结test尚未终测，现有质量数字来自小开发集，不能作为独立测试成绩。
- **答案与安全：**结构化路径已检查字段和引用ID，并从可信映射组装来源；直接文本基线仍有正文与引用列表不一致的历史案例。两条路径均未实现自动引用语义支持核验或可靠的证据充分性判定；小范围拒答验收不等于覆盖所有冲突、错误拒答或Prompt注入对抗场景。
- **索引生命周期：**尚无资料更新/删除后的旧索引清理、完整处理版本记录或跳过重复编码机制，不能声称增量维护已经完成。
- **服务与可靠性：**尚无 FastAPI、流式输出、统一错误响应、请求追踪或生产重试；本地 Qdrant 未验证多进程并发、大规模检索或故障恢复。
- **复现范围：**整套新环境安装与最终回归仍待验证。本地 BGE 是当前 Embedding 实现，不宣称已调用 Embedding API；密钥、数据库和原始资料不随代码提交。

后续使用同一开发集完成向量、BM25、RRF、RRF加精排四种策略对照，再决定问答是否采用精排；扩展引用语义评测与持久化全链回归，完成服务化、索引生命周期、冻结test终测与干净环境复现。只记录可核验结果，不将框架接入等同于质量改善，也不将计划中的能力写成已实现功能。

## 技术参考

- [BGE 模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [BGE 精排模型说明](https://huggingface.co/BAAI/bge-reranker-base)
- [Sentence Transformers 召回与精排](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [pdfplumber 文本与表格解析](https://github.com/jsvine/pdfplumber)
- [Hugging Face 离线与缓存配置](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)
- [Qdrant 客户端本地模式及版本说明](https://github.com/qdrant/qdrant-client/blob/v1.15.1/README.md)
- [Qdrant point 与幂等写入](https://qdrant.tech/documentation/manage-data/points/)
- [RRF 公式、名次与候选窗口说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [FAISS 相似度与距离规则](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [LangChain Runnable 顺序组合](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)
- [LangChain 聊天 Prompt 模板](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [DeepSeek 官方 API](https://api-docs.deepseek.com/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)
