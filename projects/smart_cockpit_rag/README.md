# 智能座舱汽车知识大脑 RAG

基于车主手册的检索增强问答原型：解析 PDF 正文与原生文本表格，使用本地 BGE、Qdrant 和 BM25 检索证据，通过兼容 Chat Completions 的接口生成结构化答案，并由程序组装可信来源引用。

当前提供命令行问答、四策略开发集评测、人工复核和本地检索基准。仅验证一份真实手册；尚无 FastAPI、真实流式输出、Docker 或完整生产验收。

## 架构与当前状态

~~~text
入库：PDF + source_id + Metadata → 正文/表格行 → chunks → BGE 编码 → Qdrant
问答：问题 → 同范围向量/BM25 → RRF → Top3 Document → 证据映射
      → JSON Prompt → LLM → JSON/字段/ID校验 → 答案与可信引用
精排：两路候选 → RRF Top10 → cross-encoder → Top3（可选检索与评测路径）
评测：配置/题集 → 四路检索快照 → 保存结果回放生成 → 人工review → 汇总
~~~

- 正文使用 pdfplumber 的 text-flow 候选方案改善部分双栏顺序；检测表格按几何单元格展开合并格，再渲染为带列标签的数据行。
- 正文按页内字符窗口切块；表格数据行保留表头、条件和相邻文字，不再按字符窗口拆分。
- LangChain 的 Document、Runnable、ChatPromptTemplate 负责适配和编排；实际检索、融合、生成、契约校验仍由独立模块实现，未使用 LangGraph。
- 日常 `langchain_query.py` 仍默认不精排的混合检索；混合加精排已在检索及保存结果的问答评测中验证，作为质量优先候选，尚未切换日常入口。
- 自动检查只保证 JSON 结构及引用身份合法，不保证答案事实、引用支持关系或证据充分性。正文编号等输出规范仍可能由人工复核发现。

### 核心配置

| 项目 | 当前值 |
| --- | --- |
| Embedding | 本地 `BAAI/bge-small-zh-v1.5`，CPU，512维，归一化 |
| 切块 | 正文500字符、重叠100字符；每条表格数据行整体保留 |
| 向量库 | `qdrant-client 1.15.1` Python本地模式，COSINE；目录 `.qdrant/` |
| BM25 | jieba分词、英文小写化，BM25Okapi：k1=1.5、b=0.75 |
| 两路候选 | 各最多10条，`hybrid_retrieval.DEFAULT_CANDIDATE_K` |
| RRF | 按chunk_id合并，累计各路 `1/(60+rank)`，同分按ID排序 |
| 最终数量 | `vector_store.DEFAULT_TOP_K=3` |
| 精排 | `BAAI/bge-reranker-base`，revision `2cfc18c9415c912f9d8155881c133215df768a70` |
| 精排输入/输出 | CPU/float32，配对长度上限512 tokens，batch_size=4，Identity保留原始logit |
| 当前解析版本 | `tables-textflow-v1` |
| 当前集合 | `cockpit_manual_bge_small_zh_v1_5_tables_textflow_v1`，604条 |
| 保留基线 | `cockpit_manual_bge_small_zh_v1_5_tables_v1`：604条；更早无表格规范化集合：469条 |
| Prompt | 结构化路径 `cockpit-rag-json-v2`；直接文本基线保留 `cockpit-rag-v1` |

这些值是源码默认值，不是全部开放为CLI参数。混合的 `score` 是RRF总分；精排另保留 `rerank_score` 与 `rrf_rank`，不覆盖原分数。各路分数不是同一种量，原始精排logit也不是概率。

两路各最多10条，按ID合并后最多20个不同候选；混合直接取RRF Top3，精排路径先取RRF Top10再重排为Top3。当前向量查询没有设置 `score_threshold`，不能对余弦、BM25、RRF和精排分数套用同一阈值。

上述已核验版本的Python本地稠密检索通过NumPy计算距离并排序，不使用HNSW。服务端HNSW参数不是当前路径的调优开关；精确近邻排序也不等于答案证据完整。是否迁移后端应依据数据规模与性能实测，不因容器化而强制迁移。

## 快速开始

以下命令在 `WEEK/projects/smart_cockpit_rag` 目录、Windows PowerShell 中执行。

### 1. 环境与模型缓存

本机使用Conda环境 `cockpit-rag`、Python 3.11和CPU PyTorch。主要依赖版本见 [requirements.txt](requirements.txt)；原 `pytorch` 环境独立保留，不需要迁移或升级。

~~~powershell
conda activate cockpit-rag
python -m pip check
~~~

新机器可创建Python 3.11环境、按CPU官方安装源准备PyTorch，再安装 `requirements.txt`；整套干净环境复现尚未验收。当前导入检索模块仍依赖FAISS，不应直接移除该依赖。

BGE与精排模型均需已有权重缓存。日常可设置：
~~~powershell
$env:HF_HUB_OFFLINE = "1"
~~~

首次下载新模型前须清除当前会话变量：
~~~powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
~~~

清除该变量不会解除代码中的 `local_files_only=True`。新机器准备精排缓存时，可在允许联网的情况下执行 `python -B -c "from reranking import build_reranker; build_reranker(local_files_only=False)"`。安装包不等于模型权重已准备好。

新终端如需传递中文脚本或把输出经PowerShell管道处理，先统一编码；同一会话无需反复设置：
~~~powershell
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
~~~

Python的 `-X utf8` 不会替PowerShell设置管道编码。jieba普通启动日志写入stderr可能标红，不能仅凭颜色认定失败；应结合traceback和退出码判断。评测JSON由Python直接按UTF-8保存，不依赖终端日志作为正式结果。

### 2. 导入或检查资料

[manual_metadata.json](manual_metadata.json) 绑定当前 `training_manual_a` 与PDF哈希。更换资料前须核对配置，当前没有多资料注册表或任意版式自动识别。品牌已确认，具体车型/年款保留未知；章节来自当前手册页眉区域 `header_y=[95,120]`，无文字时保留空值。

~~~powershell
$pdf = "../../../阶段12：智能座舱汽车知识大脑/丁师兄综合项目实战项目/综合项目实战项目一/data/train_a.pdf"
$manual = Get-Content -Raw -Encoding UTF8 ./manual_metadata.json | ConvertFrom-Json
python -X utf8 -B chunking.py $pdf $manual.source_id
python -X utf8 -B retrieval.py $pdf $manual.source_id
~~~

第一条Python命令只解析、切块；第二条执行BGE编码、`build_points`、`save_points`、同范围多余块清理和检索自检，实际修改当前集合。必须使用该资料版本的完整解析产出：写入成功后，`prune_stored_chunks`会移除同资料/哈希范围内不在本轮完整ID集中的记录。现有资料日常问答不需要再次入库。

仅比较某一物理页的两种正文提取方式：
~~~powershell
python -X utf8 -B table_parser.py $pdf 244
~~~

入库入口显式设置 `use_text_flow=True`；`parse_page` 默认仍为False，以便进行新旧对照。该选项改变字符阅读顺序，不是OCR，也不保证所有页面的逻辑关系正确。

### 3. 配置生成接口

[generation.py](generation.py) 只读取当前进程环境变量，不自动加载 `.env`。密钥不写入代码、报告或Git。

以下地址和请求模型对应本地已保存实验；更换接口须同时核对地址、模型与该供应商密钥。连通记录不构成套餐权限或业务授权确认，实际使用应遵循供应商授权范围。

~~~powershell
$env:LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
$env:LLM_MODEL = "deepseek-v4.1-flash"
$secureKey = Read-Host "输入对应供应商密钥" -AsSecureString
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
~~~

该实验实际响应模型为 `deepseek-v4-1-flash`。历史DeepSeek官方实验单独保存，不与方舟结果混合计算。切换生成供应商不需要重建同一份资料的向量，但不能把供应商变化归因于检索收益。

请求当前使用 `max_tokens=1024`、非流式、60秒超时、零自动重试、`thinking.type=disabled`；结构化路径开启 `response_format=json_object`。更换供应商前应核对扩展字段兼容性。密钥转换后是进程环境中的普通字符串，不应打印或截图。

### 4. 问答与仅检索

~~~powershell
$manual = Get-Content -Raw -Encoding UTF8 ./manual_metadata.json | ConvertFrom-Json
$question = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
python -X utf8 -B langchain_query.py $manual.source_id $manual.content_hash $question
~~~

这会真实调用生成API。入口打印状态、答案、证据、可信引用、模型和用量，但不自动保存批量QA报告。直接函数调用基线仍可通过 `query.py` 运行，未同步为结构化契约。

只查看检索结果，不需要生成密钥：
~~~powershell
python -X utf8 -B langchain_retrieval.py $manual.source_id $manual.content_hash $question hybrid
python -X utf8 -B langchain_retrieval.py $manual.source_id $manual.content_hash $question rerank
~~~

仅检索入口省略模式时默认vector；日常问答入口默认hybrid，尚不接受rerank模式参数。BM25另有 `bm25_retrieval.py` 入口。模型在检索器构建时加载，同一进程内复用；重启命令仍重新构建。

用完后可清除密钥：
~~~powershell
Remove-Item Env:LLM_API_KEY -ErrorAction SilentlyContinue
Remove-Variable secureKey -ErrorAction SilentlyContinue
~~~

## 数据身份与答案契约

每条Qdrant记录包含UUID、向量和payload。payload保留正文、文件名、资料身份、文件哈希、物理页码、解析版本、block_id、内容类型、chunk_id及车型/章节Metadata。

- `source_id` 是资料身份，`content_hash` 是原始PDF字节的SHA-256；二者同时过滤，不是车型年款或厂家版本证明，也不会按哈希自动找回PDF路径。
- `pipeline_version` 标识派生解析方案。PDF未变但正文提取顺序变化时，哈希不变，解析版本与集合另行区分，重新切块和编码；当前入口尚未实现只复用未变表格向量的增量优化。
- `chunk_id` 包含来源、哈希、切块参数、页码、块序号、解析版本与块身份。稳定UUID用于幂等upsert；相同ID覆盖不等于自动删除已消失的旧块。同范围清理由显式函数处理，解析方案对照继续使用独立集合。
- 原始文件版本变化时，先同步Metadata配置，再入库并使用新文件哈希查询。只改JSON或内存记录不会刷新数据库。
- 表格相邻文字只是归属待核验的上下文。当前目标行标注已结合原表核对，不保证任意表格邻文都属于该行。

### 资料更新与清理

[vector_store.py](vector_store.py) 已提供以下函数，并用临时数据库验证；它们不自动决定历史版本何时退役，也不删除本地PDF。

| 函数 | 作用与返回值 |
| --- | --- |
| `save_points(points, dimension)` | 按稳定ID新增或覆盖，返回整个集合写入前后的数量；不会自动删除本轮未提交的旧ID |
| `delete_stored_version(source_id, content_hash)` | 在当前集合内按资料身份与文件哈希的AND条件删除，返回该范围删除前后的数量；重复删除返回`(0,0)` |
| `prune_stored_chunks(source_id, content_hash, keep_chunk_ids)` | 同范围已有ID减去完整保留ID，只删除多余块，返回删除数量；重复清理返回0 |

同范围更新顺序是：完整产出与编码 -> `build_points` -> `save_points`成功 -> 传入本轮完整ID集执行`prune_stored_chunks` -> 核验剩余ID -> 重新构建BM25/混合检索器。不能只传新增批次的ID；空保留列表或预期ID尚未写齐时，清理函数拒绝执行。集合中的解析版本目前由集合隔离，而非单独的payload过滤条件。

`retrieval.main()`已在首次完整`save_points`成功后接入`prune_stored_chunks`，再执行重复写入和检索自检；输出`counts (before/upsert/repeat)`与`pruned`，前者是整个集合的数量，后者是本次范围内的删除数量。写入异常会停止，不继续清理。新文件哈希的清理不会自动删除旧哈希；确需退役旧版本时再明确调用版本删除函数。保留的历史集合用于对照/回滚，不视作活动库中的脏数据。日常混合检索CLI新进程会从数据库重新构建BM25，已存在的检索器实例则需显式重建，不是入库函数自动刷新所有实例。

`delete`删除选中的记录，`delete_collection`才删除集合；若集合中全部记录都匹配指定身份/哈希，版本删除会清空记录但留下集合。因此只在明确退役后操作，不用正式资料演示删除。当前能力限单进程离线作业，写入与清理不是原子事务；写入失败不得继续清理，部分写入不保证自动回滚。已构建的BM25持有内存语料，不会因Qdrant删除而自动刷新。

### 结构化答案与引用

结构化链只把本次最终Document映射为 `E1/E2/E3`，这些编号不是页码，也不跨请求稳定。模型只返回 `answer_status`、`answer`、`evidence_ids`；程序从可信映射回取来源、哈希、块ID、页码与正文，不信任模型另写的来源字段。

| 状态 | 含义 |
| --- | --- |
| `generated` | 已生成并通过JSON/字段/ID检查，不等于人工任务、事实或引用语义通过 |
| `no_evidence` | 无Document，直接返回refused，不调用生成API |
| `error` | 评测入口捕获API或解析异常，保留已取得的raw_answer/model/usage，不伪装成正常拒答 |
| `answered/refused` | 回答层状态；正常生成也可以拒答，合理拒答不等于完成一条原本可回答的任务 |

`answered` 至少引用一条本次证据；`refused` 可引用相关证据或为空。JSON-object模式不保证严格Schema，当前正文中的E编号等规范也未全部自动检查；可信ID只证明引用身份真实，不能证明推理正确。

## 可复用评测流程

### 配置与数据划分

- `eval_cases.json`：20道dev；`table_eval_cases_v1.json`：10道dev，合计24道可回答、6道不可回答。
- `eval_test_v1.json`：冻结的10题，尚未运行检索或生成；最终配置固定后终测，不用于调参。
- `evaluation_config_textflow_v1.json` 指向原有dev题集、Metadata和新版表格目标行标注；旧配置与旧标注均保留。
- 配置文件不直接选择数据库，实际集合由 `vector_store.COLLECTION_NAME` 指定；入口核对标注与报告的来源、哈希、集合和解析版本，避免混评。
- 冻结题集字节SHA-256为 `4b2e8373a12a7f66b7f3d2563bbeef2a619d96c977b6f0b57c5a41cac3e1c226`，不是PDF的content_hash。格式化也会改变该指纹。

### 1. 检索、评分与保存

~~~powershell
python -X utf8 -B evaluation.py compare evaluation_config_textflow_v1.json dev_retrieval_textflow_run.json
python -X utf8 -B evaluation.py summary dev_retrieval_textflow_run.json
~~~

`compare` 复用四路检索、目标行评分和写报告函数，生成30题×4策略共120条最终Top3，全部 `generation_status=not_run`。同时保存完整上下文、模型配置及汇总；不调用生成API。输出文件存在时拒绝覆盖。

| 指标 | 本项目定义与适用范围 |
| --- | --- |
| 标注页命中率 | 命中标注来源及页码的题目比例；24道可回答题 |
| 标注页MRR | 首次命中标注页的倒数排名均值，未命中为0；24题 |
| 目标行Recall | 每题命中必需block_id的比例，再作题目平均；8道有行标注的表格题 |
| 目标行MRR | 首次命中任一必需行的倒数排名均值；8题，不证明全部证据齐全 |
| 目标行全齐率 | 每题是否找到全部必需行，再计算比例；8题，是覆盖率等于1的派生判断 |

替代页也可能有有效证据，单一标注页不是完整相关集合。不可回答或无行标注时相应指标为null；False与0是有效失败，不能从分母排除。

### 2. 用保存上下文生成

先预检，命令没有 `--run` 时不调用API、也不创建QA文件：
~~~powershell
python -X utf8 -B evaluation.py qa evaluation_config_textflow_v1.json dev_retrieval_textflow_run.json all qa_dev_textflow_run.json
~~~

确认供应商配置及费用后，在同一命令末尾加 `--run` 才真实生成。参数 `all` 可换为一个dev题号，选单题时运行四策略；全量最多120次调用。

该入口将已保存结果适配为Document，复用原问答链，不重新检索。参考答案在生成后附入报告，不进入Prompt。逐条保存原始响应、状态、引用、模型与用量；`run_status=complete` 只表示循环结束，不代表全对。失败也保留，当前无断点续跑、原子检查点或自动重试。

### 3. 人工复核与汇总

~~~powershell
python -X utf8 -B evaluation.py review qa_dev_textflow_run.json qa_dev_textflow_review.json
~~~

在新副本填写每条review后，才汇总答案质量：
~~~powershell
python -X utf8 -B evaluation.py summary qa_dev_textflow_review.json
~~~

review包含整题通过、忠实度、引用支持、拒答合理性、失败类型及说明。只有task_pass必须全为布尔；其余不适用项可为null。单条多标签允许重叠，不能把标签计数直接相加当作失败题数。原始生成报告不改写，也不静默修复模型失败响应。

### 4. 本地检索基准

~~~powershell
python -X utf8 -B evaluation.py benchmark evaluation_config_textflow_v1.json retrieval_benchmark_textflow_run.json
~~~

模型/检索器构建单独记录，各路径预热一次；30题×两策略×3轮保存180条耗时、有序块ID和页码。仅衡量CPU热检索，不含生成或HTTP，不证明LLM回答稳定。该报告类型当前不接原 `summary` 的质量汇总分支，不应用它计算答案通过率。

所有报告示例使用新文件名。正式实验应保留原始输出与独立review；报告含手册正文，不默认随公开仓库分发。

清理时保留正式检索基线、原始QA、独立review和性能报告；仅在核对内容覆盖及引用后删除冗余中间文件。`__pycache__/`、`.pytest_cache/`可重新生成，`.qdrant/`则是持久化数据库，不作为缓存删除。

## 已验证结果

工程状态核验于2026-11-05；以下质量与性能数字沿用所列实验报告，不表示当日重新生成。均为单手册、小开发集结果，不是独立测试集或生产成绩。

### 数据与解析

354个物理页，334页有提取文本，20页经历史渲染核验仅见空白/裁切标记。切块604条，其中142条表格行；检测到27个表格区域，不等于所有视觉表格均被识别。

活动库按现有流程重新解析、切块后只读核对：实际ID与预期ID均为604，两个方向差集为0；此前完整payload对齐也无差异，当前没有需清理的多余记录。该核对没有调用正式清理函数，也不证明所有页面语义解析正确。

新旧解析全书对照中，每页非空白字符及出现次数一致，表格记录含邻文保持一致，305页正文输出改变。197/244页部分段落归属更连贯，但页码/标题位置也改变，不能由字符未丢或块数相同推出全书阅读逻辑正确。

### 完整问答对照

每个单元格为同30道dev问题的整题完整输出通过数，包含回答状态和输出规范；不把不同Prompt或供应商混成一轮。

| 解析方案 | Prompt | 向量 | BM25 | 混合 | 混合加精排 |
| --- | --- | --- | --- | --- | --- |
| tables-v1 | json-v1 | 28/30 | 29/30 | 30/30 | 28/30 |
| tables-v1 | json-v2 | 27/30 | 29/30 | 28/30 | 29/30 |
| tables-textflow-v1 | json-v2 | 26/30 | 29/30 | 29/30 | 30/30 |

对应原始报告为 `qa_dev_ark_v41_v1.json`、`qa_dev_ark_v41_prompt_v2_full.json`、`qa_dev_ark_v41_prompt_v2_textflow_v1.json`，各有独立review。三轮均请求同一方舟模型别名；别名、未固定采样及运行时间不能保证后端或随机性完全相同，不宣称严格因果或统计显著性。

新解析的完整v2共120条，全部generated，使用100109 tokens；旧解析v2为119条generated加1条契约error，使用101175 tokens，错误保留在通过率分母。新旧同v2有5处改善、4处退化，合计113/120→114/120，不能称为全面提升。

- 面板操作题由四路均未通过变为四路通过，旋钮操作与屏幕点击不再混用。
- 新轮五条可回答任务因最终缺证据而合理拒答：向量FCTA/风量口语/ISOFIX、BM25 AutoHold、混合FCTA。拒答合理仍不能记整题成功。
- 向量SOH正文写E1/E2导致完整输出不通过；向量FCTA同时有同类格式问题，与漏召回重叠。不是电池事实错误或未知引用。
- FCTA的197页是BM25第1，但未进入向量前10，融合后为第5而被最终Top3截掉；精排可将它提回第1。
- 精排仍把目录4页排在加热题第1，把普通座椅U/L行排在ISOFIX第1；本轮正确引用后续证据，不代表干扰消失。

### 当前检索指标

| 策略 | 标注页命中（24题） | 标注页MRR | 目标行Recall（8题） | 目标行MRR | 目标行全齐 |
| --- | --- | --- | --- | --- | --- |
| 向量 | 22/24 | 0.847222 | 0.9375 | 0.9375 | 7/8 |
| BM25 | 22/24 | 0.854167 | 1 | 1 | 8/8 |
| 混合 | 22/24 | 0.854167 | 1 | 1 | 8/8 |
| 混合加精排 | 24/24 | 0.979167 | 1 | 0.8125 | 8/8 |

来源为 `dev_retrieval_textflow_v1.json`；旧基线 `dev_retrieval_v3.json` 保留。命中页、完整证据、生成质量分别检查，不把上述数字称作五个彼此独立的通用能力。

### CPU热检索基准

| 路径 | 次数 | 均值 | 中位数 | 最近秩P95 |
| --- | --- | --- | --- | --- |
| 混合 | 90 | 0.153秒 | 0.155秒 | 0.171秒 |
| 混合加精排 | 90 | 0.821秒 | 0.826秒 | 1.050秒 |

来源为 `retrieval_benchmark_textflow_v1.json`。两路各30题的三轮有序块ID全部一致，也与保存的检索快照一致；平均差约0.668秒。P95在各90条排序后取第86条。构建和预热另计，不是进程冷启动、HTTP端到端延迟、并发容量或线上SLA。

当前以新解析加精排作为后续质量优先候选；日常问答源码仍默认混合，最终配置固定后的冻结终测尚未执行。

## 自动测试

~~~powershell
python -X utf8 -B -m pytest -q -p no:cacheprovider
python -m pip check
~~~

2026-11-05实跑38项通过，pip check无依赖冲突。

| 覆盖范围 | 项数 |
| --- | --- |
| 切块边界、重叠、Metadata、ID与非法参数 | 5 |
| 临时Qdrant持久化、upsert边界、精确删除、缩块清理、范围隔离与BM25重建 | 4 |
| 旧文本Prompt一致性与空输入 | 2 |
| RRF排序、两路信息与输入不变 | 1 |
| 假精排排序、同分及非法分数 | 4 |
| 结构化JSON/字段/状态/引用身份契约 | 13 |
| 问答空证据、生成桩、异常、防覆盖与人工复核汇总 | 9 |

测试不调用付费API，不运行Embedding/精排推理或冻结test问答。临时库与人工向量只测工程行为，不评估语义质量。存储测试覆盖重复写入不增量但遗漏块仍在、删除后的空查询与其他范围保留、非法身份、空/缺失保留ID、2块缩为1块、重复清理，以及旧BM25与重建实例的区别。汇总用例覆盖False保留、null分母、标签去重、输入不变和严格评分类型；测试通过不等于38道真实问题正确。

另用假解析结果、2维单位向量和临时数据库运行当前`retrieval.main()`，保留真实切块、Qdrant写入/清理、FAISS自检及向量查询，验证首次清理1条、重复导入清理0条、其他资料/哈希保留，以及写入失败时不清理。该临时入口检查不计入上述38项持久化pytest，也不等于真实PDF/BGE全量入库、并发更新或故障恢复验收；没有重新写入或删除正式604条记录。

## 代码导航

| 模块 | 主要职责 |
| --- | --- |
| [table_parser.py](table_parser.py)、[ingestion.py](ingestion.py) | 正文/表格解析、可选页对照、哈希与Metadata |
| [chunking.py](chunking.py)、[retrieval.py](retrieval.py) | 切块、编码、入库及FAISS原理基线 |
| [vector_store.py](vector_store.py) | Qdrant写入、来源/版本过滤、分批读取payload、精确版本删除与多余块清理 |
| [bm25_retrieval.py](bm25_retrieval.py)、[hybrid_retrieval.py](hybrid_retrieval.py) | 同范围关键词召回与RRF |
| [reranking.py](reranking.py)、[langchain_retrieval.py](langchain_retrieval.py) | 精排及三种检索器适配 |
| [answer_contract.py](answer_contract.py)、[langchain_prompt.py](langchain_prompt.py) | 本次证据映射、结构化消息、JSON/引用身份校验 |
| [generation.py](generation.py)、[langchain_query.py](langchain_query.py) | 供应商调用与结构化问答链 |
| [prompt_builder.py](prompt_builder.py)、[query.py](query.py) | 直接文本基线 |
| [evaluation.py](evaluation.py)、[test_evaluation.py](test_evaluation.py) | compare/score/qa/review/summary/benchmark及离线测试 |

### 评测入口与函数关系

`main()`按子命令选择分支并返回，不按函数定义顺序执行全部流程。

| 子命令 | 主调度与职责 | 调用生成API |
| --- | --- | --- |
| `compare` | `run_retrieval_evaluation`组织四路检索、评分、汇总及保存 | 否 |
| `score` | `main`复用 `add_block_scores`、`summarize_retrieval`、`write_report`，只给已有报告补分 | 否 |
| `qa` | `run_saved_qa`逐行调用 `evaluate_qa_row`，后者回放Document并复用 `build_qa_chain` | 仅显式加 `--run` 后 |
| `review` | `prepare_qa_review`创建带空review的新副本，需人工填写 | 否 |
| `summary` | `summarize_retrieval`或 `summarize_qa_review`汇总已有分数，不自动判断答案 | 否 |
| `benchmark` | `benchmark_retrieval`构建两检索器，预热后重复检索计时 | 否 |

共用的 `load_evaluation_inputs`合并dev题集、读取Metadata/目标行标注并核对范围；`score_retrieval`和 `score_required_blocks`只比较已有结果；`write_report`只保存报告。旧位置参数入口仍由 `inspect_retrieval`执行单路检索，并保留 `"w"`覆盖写入，不具备新配置流程的dev限制和新文件防覆盖保护，不作为常规四策略评测入口。

## 限制与后续方向

- 只有单份真实资料；原生表格检测与表头处理有启发式边界，未实现OCR、跨页表格拼接或完整版面理解。text-flow不能保证任意PDF阅读顺序。
- 车型/年款未知，页眉坐标和Metadata配置针对当前手册；尚无可靠的车型过滤或自动适用性判断。
- 只有dev方案比较，冻结10题尚未终测；精排和目录干扰仍有风险，不能凭本轮30/30承诺泛化。
- JSON/可信ID校验不等于自动引用语义评测；正文格式、证据充分性和拒答仍需回归，旧error行可能缺Prompt版本等元信息。
- 保存QA逐条落盘但非原子检查点，无断点续跑、生产重试或请求追踪；API错误保留，不自动改写为正常拒答。
- 入库CLI已接入同范围旧块清理，仅按单进程离线作业验证；尚无自动版本切换、写入/清理原子事务、并发更新或失败回滚，也未实现仅更新payload、跳过重复编码。保留旧集合不等于自动版本管理。
- 尚无FastAPI、SSE、Docker和CI；本地Qdrant多进程、故障恢复、并发容量及新环境完整复现仍待验证。
- 密钥、原始PDF和数据库不随代码提交；生成报告含原文，公开或提交前须确认资料授权及敏感信息。

## 技术参考

- [pdfplumber文字与表格提取](https://github.com/jsvine/pdfplumber)
- [Sentence Transformers召回与精排](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [BGE精排模型说明](https://huggingface.co/BAAI/bge-reranker-base)
- [RRF公式与候选窗口](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [Qdrant points与upsert](https://qdrant.tech/documentation/concepts/points/)
- [Qdrant删除记录与按ID/过滤器选择（v1.15）](https://api.qdrant.tech/v-1-15-x/api-reference/points/delete-points)
- [Qdrant Python本地实现（v1.15.1）](https://github.com/qdrant/qdrant-client/blob/v1.15.1/qdrant_client/local/local_collection.py)
- [LangChain RunnableSequence](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)
