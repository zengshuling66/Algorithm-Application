# 智能座舱汽车知识大脑 RAG

基于车主手册的检索增强问答项目。将 PDF 解析为带来源信息的文本块，使用本地 BGE 编码、Qdrant 持久化检索和兼容 Chat Completions 的模型接口生成回答，并展示资料引用。

当前提供 LangChain 编排的命令行问答、直接函数调用基线与初版检索评测，定位为单资料工程化原型，不宣称生产上线或完整质量验收。

## 功能与架构

- 使用 pdfplumber 分离正文与检测到的原生文本表格，按单元格几何关系展开合并格，并为表格行保留列标签及相邻文字。
- 正文采用字符窗口切块，表格行不再按字符窗口拆分；保留文件名、物理页码、内容类型和解析版本。
- 文件内容哈希、稳定 chunk ID 和幂等写入，支持按资料身份及文件版本过滤。
- 通过绑定资料身份与哈希的配置读取品牌、未知车型/年款及页眉区域，提取章节并写入数据库 payload。
- 本地向量检索、结构化证据 Prompt、答案与引用展示、模型及 token 用量输出。
- 使用 LangChain Document、Runnable 和 ChatPromptTemplate 组织原有检索及生成模块，保留证据身份、排名和运行状态。
- 开发集与冻结测试集分离，支持题集字段校验、批量检索观察，以及切块、向量库和 Prompt 自动测试。

```text
资料入库：PDF + source_id + Metadata 配置 -> 正文/表格记录 -> chunks -> BGE 向量 -> Qdrant
在线问答：问题 + source_id + content_hash -> BGE -> 过滤检索 -> Document -> 证据 Prompt -> 兼容 SDK -> LLM
检索评测：题集 -> 检索 -> 对照标注来源与页码 -> JSON 报告
```

| 模块 | 职责 |
| --- | --- |
| [table_parser.py](table_parser.py) | 定位表格、展开合并格、渲染带列标签的数据行，分离正文与邻文 |
| [manual_metadata.json](manual_metadata.json) | 当前资料的身份、哈希、品牌、未知车型/年款及页眉位置配置 |
| [ingestion.py](ingestion.py) | 计算 SHA-256、生成正文/表格记录、提取章节、记录无文本页 |
| [chunking.py](chunking.py) | 正文窗口切分、表格行整体保留，传递 metadata 并生成稳定 chunk ID |
| [retrieval.py](retrieval.py) | 文本与问题编码、入库入口、持久化检索及 FAISS 基线 |
| [vector_store.py](vector_store.py) | Qdrant 集合检查、写入、过滤检索与资源关闭 |
| [prompt_builder.py](prompt_builder.py) | 编号证据，组织回答规则、来源与页码 |
| [generation.py](generation.py) | 调用生成接口，检查结束标记与非空内容，返回答案、模型及用量 |
| [query.py](query.py) | 保留直接函数调用的问答基线 |
| [langchain_adapter.py](langchain_adapter.py) | 将检索字典转换为 Document，保留正文、chunk ID、metadata 和检索排名 |
| [langchain_retrieval.py](langchain_retrieval.py) | 用 Runnable 包装原检索函数与 Document 转换，提供仅检索入口 |
| [langchain_prompt.py](langchain_prompt.py) | 复用原 Prompt 规则和证据格式，通过 ChatPromptTemplate 生成 SDK 消息 |
| [langchain_query.py](langchain_query.py) | 组装并执行问答链，保留状态、证据、Prompt 版本、实际模型和用量 |
| [evaluation.py](evaluation.py) | 校验题集，记录逐题或批量检索结果，不调用生成 API |
| [test_chunking.py](test_chunking.py)、[test_vector_store.py](test_vector_store.py) | 切块边界、metadata、持久化、幂等写入与范围隔离测试 |
| [test_langchain_prompt.py](test_langchain_prompt.py) | 新旧 Prompt 消息一致性与空证据校验测试 |

Qdrant 是当前持久化检索后端，使用本地模式，无需额外启动服务器。FAISS `IndexFlatIP` 保留为精确检索基线；入库入口仍执行其自检，但独立问答不重建 FAISS 索引，也不重新解析或编码整份 PDF。

LangChain 只负责数据适配与流程编排，实际检索仍由 BGE/Qdrant 完成，生成仍调用现有 SDK 适配函数；没有因引入框架更换索引、提高 top-k 或修改 Prompt 规则。当前使用 `langchain-core`，未引入 LangGraph，也没有自动获得检索质量、引用校验或真实流式能力。

## 快速开始

以下示例使用 Windows PowerShell。进入项目目录后，入库、问答和评测命令均在同一目录执行。

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

### 4. 运行问答

沿用入库步骤中的资料身份和文件哈希，在配置好生成接口的同一个终端运行 LangChain 问答入口：

```powershell
$question = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
python -B langchain_query.py $sourceId $contentHash $question
```

输出包含 Prompt 版本、`status`、答案与模型生成的引用、每条候选证据的 chunk ID/物理页码/正文，以及实际响应模型和 token 用量。`generated` 只表示生成完成，不表示事实或引用已自动校验；`no_evidence` 表示检索为空，程序返回保守回答，模型和用量为 `None`，不会发起生成请求。

引用编号按本次检索排名生成，`[1]` 不是 PDF 第1页；文件名和物理页码来自检索记录。入口逐条打印的 `evidence:` 是候选上下文，不代表模型在答案中引用了全部候选。返回状态保留原始 Document 和消息，但当前命令行不会自动保存机器可读生成报告。

仅查看检索、不调用生成模型时使用 `python -B langchain_retrieval.py $sourceId $contentHash $question`。原问答基线仍可通过 `python -B query.py $sourceId $contentHash $question` 运行；其空结果提示为 `No matching evidence found.`，不输出上述链状态。

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
| 检索 | Qdrant COSINE 距离，`vector_store.py` 中 `DEFAULT_TOP_K=3`，检索/问答/评测统一引用 |
| 持久化 | 本项目目录下 `.qdrant/`，已通过 `.gitignore` 忽略 |
| 当前集合 | `cockpit_manual_bge_small_zh_v1_5_tables_v1`，604条 |
| 历史基线集合 | `cockpit_manual_bge_small_zh_v1_5`，469条，不因切换常量而自动重建或删除 |
| 解析与 Metadata | `pipeline_version=tables-v1`；品牌、车型、年款、章节与来源字段随 payload 保存，未知值保留 `None` |
| Prompt | `cockpit-rag-v1`，约束证据范围、引用及无依据拒答 |

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

第一条命令只校验字段，不加载 Embedding 权重或调用生成 API。第二条加载一次 BGE 后复用，保存逐题记录；它会覆盖同名报告，应为不同实验使用不同文件名。

表格开发集可替换上述命令的题集参数使用；历史表格真实问答前后对照复用 `query.py`，不由 `evaluation.py` 自动执行生成或评分。

`eval_test_v1.json` 已核对原文、完成字段校验，并于2026-10-22登记冻结基线；与30条dev无重复题号或完全相同问题，按问题族复核，不将同题改写移入test。当前仅为同一手册的问题留出集，不代表跨资料泛化评测。开发与方案选择只用dev；test尚未运行检索或生成，待最终配置固定后使用，不根据test结果筛题或调参数。标注确需纠错时记录原因与新版本，不静默覆盖。

冻结时题集文件的字节 SHA-256 为 `4b2e8373a12a7f66b7f3d2563bbeef2a619d96c977b6f0b57c5a41cac3e1c226`，可用 `Get-FileHash -LiteralPath eval_test_v1.json -Algorithm SHA256` 核对。它不是PDF的 `content_hash`，格式化也会改变该指纹；冻结是使用与版本约定，不是文件只读锁定。测试集字段校验通过不表示检索、生成或拒答质量通过，当前未报告test准确率。

| 报告字段 | 含义 |
| --- | --- |
| `retrieved_pages`、`scores` | 按相同检索排名对应的命中页和相似度分数 |
| `evidence_page_hit` | 可回答题有任一结果同时命中标注身份及页码为 `true`，否则为 `false`；不可回答题为 `null` |
| `generation_status` | 固定为 `not_run`，表示该批量任务没有运行生成，不是生成失败或拒答通过 |

### 验证快照

核对范围：2026-10-16表格前后对照、2026-10-19 Metadata落库，以及2026-10-22 LangChain接入、单题问答、题集和自动测试。当前真实语料仍只有 `train_a.pdf`。

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

完整回答提高25个百分点，主要修复座椅210/50毫米及轮胎空载/满载的条件对应错误；不是单凭换库、提高相似度分数或命中页得出的结论。保留一项回归：ISOFIX题所需的D行已在新库，但排第4；top-3仅含B/A/B1，因此D部分无法完整回答。当前保持top-3，不把该题标为通过，待混合检索/重排序后用原题复测。仅此小开发集单轮结果，不代表全库准确率或稳定拒答能力。

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

当前 LangChain 入口已用 DeepSeek 官方 `deepseek-flash` 验证无钥匙单门/全车开发题：候选物理页为43/44/41，答案与43页两种解锁规则一致，引用 `[1]` 正确；Prompt仍为 `cockpit-rag-v1`，本次输入785、输出51、合计836 tokens。该结果依据学习者提供的真实终端输出核验，仅覆盖一个dev问题。上面的表格对照与三题快照来自历史方舟模型，不等于当前官方接口的整套质量回归，不能跨供应商直接归因或合并指标。

### 自动测试

```powershell
python -B -m pytest -q test_chunking.py test_vector_store.py test_langchain_prompt.py
python -m pip check
```

2026-10-22复跑8项测试通过，依赖检查无冲突：5项切块测试覆盖窗口与重叠、空文本、跨页metadata、非法参数和ID变化；1项临时数据库集成测试覆盖持久化、重复写入、来源/版本过滤、payload保留与空范围结果；2项Prompt测试检查新旧消息完全一致（含正文中的字面花括号）及空Document列表报错。

这些测试不调用付费API，不运行冻结test的问答。人工构造的向量用于验证存储行为，不评估Embedding语义质量；Prompt一致性也不证明答案正确或引用支持结论。完整链的空检索分支、供应商异常、生成质量与引用语义尚未纳入持久化自动回归套件，metadata正确性仍需原文抽查。

## 已知限制与后续方向

- **资料质量：**只验证一份真实PDF，支持当前可检测的原生文本表格，不保证所有表格均被发现。表头识别存在启发式边界，相邻文字只作为归属待核验的上下文，未实现OCR、跨页表格拼接或完整版面理解；部分双栏正文仍有交错，字符切块也可能打断语义。
- **适用范围：**品牌与章节字段已入库，具体车型/年款仍未知；固定页眉区域及单资料配置不直接推广到其他手册，尚无车型过滤、自动适用性判断或已核实的官方资料版本映射。
- **检索质量：**目前为单路向量检索，尚无 BM25、混合融合、重排或系统性参数对照。10条冻结test尚未终测，现有质量数字来自小开发集，不能作为独立测试成绩。
- **答案与安全：**引用列表由模型生成，存在与正文不一致的已知案例；尚无自动引用支持核验、可靠的证据充分性判断或 Prompt 注入对抗验证。
- **索引生命周期：**尚无资料更新/删除后的旧索引清理、完整处理版本记录或跳过重复编码机制，不能声称增量维护已经完成。
- **服务与可靠性：**尚无 FastAPI、流式输出、统一错误响应、请求追踪或生产重试；本地 Qdrant 未验证多进程并发、大规模检索或故障恢复。
- **复现范围：**整套新环境安装与最终回归仍待验证。本地 BGE 是当前 Embedding 实现，不宣称已调用 Embedding API；密钥、数据库和原始资料不随代码提交。

后续使用开发集比较BM25、向量、混合检索及重排，完善引用契约与分层评测，再完成服务化、索引生命周期、冻结test终测与干净环境复现。只记录可核验结果，不将框架接入等同于质量改善，也不将计划中的能力写成已实现功能。

## 技术参考

- [BGE 模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [pdfplumber 文本与表格解析](https://github.com/jsvine/pdfplumber)
- [Hugging Face 离线与缓存配置](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)
- [Qdrant 客户端本地模式及版本说明](https://github.com/qdrant/qdrant-client/blob/v1.15.1/README.md)
- [Qdrant point 与幂等写入](https://qdrant.tech/documentation/manage-data/points/)
- [FAISS 相似度与距离规则](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [LangChain Runnable 顺序组合](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)
- [LangChain 聊天 Prompt 模板](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [DeepSeek 官方 API](https://api-docs.deepseek.com/)
