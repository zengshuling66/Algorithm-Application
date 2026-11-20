# 智能座舱汽车知识大脑 RAG

基于车主手册的检索增强问答原型：解析 PDF 正文与原生文本表格，使用本地 BGE、Qdrant 和 BM25 检索证据，通过兼容 Chat Completions 的接口生成结构化答案，并由程序组装可信来源引用。

当前提供命令行问答、FastAPI 普通/真实 SSE 问答、请求关联与阶段日志、四策略开发集对照、冻结终测、独立复核和Markdown报告导出。100题终测采用混合召回加精排、Top3，本轮任务通过92/100；结果仅代表这一份手册、题集、模型运行和既定复核口径，不是生产准确率。Docker与干净环境复现仍待验收。

## 架构与当前状态

~~~text
入库：PDF + source_id + Metadata → 正文/表格行 → chunks → BGE 编码 → Qdrant
问答：CLI / POST /qa / POST /qa/stream → 同范围向量/BM25 → RRF → CLI直接取Top3 / API精排后取Top3
      → Document → 证据映射
      → JSON Prompt → LLM → JSON/字段/ID校验 → 答案与可信引用
流式：SDK delta → stream_qa临时事件 → SSE delta；正常生成结束且契约校验通过 → SSE final
精排：两路候选 → RRF Top10 → cross-encoder → Top3（API默认，检索与评测可选）
评测：题集/金标/配置冻结 → 四路检索快照 → 按qa_methods回放生成 → 独立review → 汇总/Markdown
~~~

- 正文使用 pdfplumber 的 text-flow 候选方案改善部分双栏顺序；检测表格按几何单元格展开合并格，再渲染为带列标签的数据行。
- 正文按页内字符窗口切块；表格数据行保留表头、条件和相邻文字，不再按字符窗口拆分。
- LangChain 的 Document、Runnable、ChatPromptTemplate 负责适配和编排；实际检索、融合、生成、契约校验仍由独立模块实现，未使用 LangGraph。
- `langchain_query.py` 仍默认不精排的混合检索，保留为对照；FastAPI 复用现有问答链，采用混合召回加精排，最终传入最多3条证据。
- 普通与流式接口复用启动时构建的检索器、模型和 SDK 客户端；流式路径复用证据映射、Prompt 和引用解析，不把临时 JSON 片段作为最终答案。
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
| 最终数量 | 默认 `vector_store.DEFAULT_TOP_K=3`；批量评测优先读取配置的 `top_k`，终测固定为3 |
| 生成评测策略 | 配置 `qa_methods`；缺省为四路，当前终测为 `["rerank"]` |
| 精排 | `BAAI/bge-reranker-base`，revision `2cfc18c9415c912f9d8155881c133215df768a70` |
| 精排输入/输出 | CPU/float32，配对长度上限512 tokens，batch_size=4，Identity保留原始logit |
| 当前解析版本 | `tables-textflow-v1` |
| 当前集合 | `cockpit_manual_bge_small_zh_v1_5_tables_textflow_v1`，604条 |
| 保留基线 | `cockpit_manual_bge_small_zh_v1_5_tables_v1`：604条；更早无表格规范化集合：469条 |
| Prompt | 结构化路径 `cockpit-rag-json-v2`；直接文本基线保留 `cockpit-rag-v1` |

除评测配置中的 `top_k`、`qa_methods` 外，这些值不是全部开放为CLI参数。混合的 `score` 是RRF总分；精排另保留 `rerank_score` 与 `rrf_rank`，不覆盖原分数。各路分数不是同一种量，原始精排logit也不是概率。

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

生成参数集中在 `generation.py` 的 `LLM_MAX_TOKENS=1024`、`LLM_TIMEOUT_SECONDS=60.0`、`LLM_MAX_RETRIES=0`，普通与流式路径共用。SDK超时是网络操作等待限制，不是整次HTTP请求的硬截止，也不限制本地检索或排队时长；流持续返回数据时总时长可能超过60秒。保持零自动重试，先定位问题再由调用者决定是否重发，避免流式重复输出或重复计费。

两条路径均使用 `thinking.type=disabled`；普通路径 `stream=False`，流式路径 `stream=True`，结构化路径开启 `response_format=json_object`。更换供应商前应核对扩展字段兼容性。密钥转换后是进程环境中的普通字符串，不应打印或截图。

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

### 5. 启动本地 HTTP 服务

在项目目录、已配置第3节环境变量的同一个 `cockpit-rag` 终端中启动：

~~~powershell
$env:HF_HUB_OFFLINE = "1"
python -B -m uvicorn api:app --host 127.0.0.1 --port 8001 --workers 1
~~~

启动需要本地 BGE、精排权重缓存和当前资料的 Qdrant 集合。离线设置仅作用于 Hugging Face，避免缓存加载期间的远端元数据请求，不阻止问答调用生成 API。新终端需重新设置会话环境变量。

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 检查初始化状态及当前资料身份/哈希；不调用生成 API |
| `POST /qa` | 接收 JSON 对象 `{"question": "问题文本"}`，返回状态、答案、引用、模型和用量；不返回内部 Prompt 或完整检索上下文 |
| `POST /qa/stream` | 接收相同请求体，以 `text/event-stream` 发送临时片段、最终结果或错误事件 |

可在本机的 [接口文档](http://127.0.0.1:8001/docs) 中执行请求。非法输入返回 HTTP 422；正常回答或业务拒答可以返回 HTTP 200，应同时查看 `status` 和 `answer_status`。有证据的真实问答会调用生成 API，可能产生费用。用完在启动服务的终端按 `Ctrl+C` 正常停机。

普通接口已验收一题真实开发集请求：单门仅解锁当前车门，全车解锁全车车门，引用指向 `train_a.pdf` 的物理第43页。该结果不代表冻结测试集或并发验收。

### 6. 验证真实流式与断开恢复

直接验证供应商 JSON 流，不经过检索或本地 HTTP 服务：
~~~powershell
python -X utf8 -B stream_probe.py
~~~

这会真实调用生成 API；预期完整 JSON 为 `{"answer":"API_OK"}`、状态为 `completed`，并打印正文片段数量、实际模型与用量。生成层仅接受 `finish_reason=stop` 和非空正文；usage 可以来自完成块或无 choices 的用量块，缺失时保留 `None`，不自行估算。

验证完整 RAG 经 HTTP 逐段返回时，保持第5节服务运行，在项目目录的另一个 `cockpit-rag` 终端依次执行：
~~~powershell
python -X utf8 -B http_stream_probe.py
python -X utf8 -B http_stream_probe.py --disconnect
python -X utf8 -B http_stream_probe.py
~~~

探针复用 `load_cases`，读取 `eval_cases.json` 的开发题 `q_unlock_scope_001`；只发送问题，不发送金标或参考答案。它访问固定本地地址 `127.0.0.1:8001/qa/stream`，打印HTTP状态、响应头中的请求ID、事件到达时间及最终结果；客户端无需配置生成密钥，供应商配置仍由服务进程持有。三次请求都可能产生费用。

| SSE 事件 | 数据与使用边界 |
| --- | --- |
| `delta` | `type/text/provisional=true`；正文可能是半截 JSON，尚未通过完整引用契约 |
| `final` | `type=final`，`data` 为通过答案契约及 `QuestionResponse` 校验的结果；不等于语义质量自动通过 |
| `error` | `type=error/code=stream_failed`；生成或校验异常后结束，不返回异常原文，不伪装为正常 final 或业务拒答 |

外层 SSE 的 `event` 名与 JSON 内的 `type` 对应。无证据时只发一次拒答 `final`，不调用模型；正常有证据时先发临时片段，完整生成结束后再校验并发 `final`。HTTP 200 只说明响应已开始，客户端还须收到 final；连接断开或仅有 delta 不能记为问答完成。探针仅处理当前框架的单行 JSON 数据格式，不是通用 SSE 解析器。

`lifespan` 管理服务级资源，`Depends(..., scope="request")` 管理单次流的迭代器与锁。普通 `/qa` 等待共享锁；`/qa/stream` 非阻塞取锁，忙碌时在响应开始前返回 HTTP 503，非法输入仍为422。流式开始后的异常通过 error 事件报告，不能改写已经发出的状态码。

断开清理等待当前同步读取退出，再关闭上游迭代器并释放锁；AnyIO 取消保护用于完成退出清理，不会强杀线程或保证供应商立即停止生成/计费。SDK 客户端继续供其他请求复用，仅在正常停机时关闭。提前断开后若暂时503，应等待清理，不能自动重试计费请求。

本机单题验收中，两次正常请求均返回17个 delta，首个片段/最终结果分别约3.62/4.01秒和1.76/2.15秒；答案均正确区分单门与全车，引用物理第43页，响应模型为 `deepseek-v4-1-flash`，用量分别916和917 tokens。中间请求收到首个 delta 后主动关闭，随后正常请求成功。片段数不是 token 数；后一次响应报告命中768个缓存 token，不能把单题耗时变化归因于某项优化，也不视为并发或稳定性能基准。提前关闭没有最终用量，不能补算其成本。

用完后可清除密钥：
~~~powershell
Remove-Item Env:LLM_API_KEY -ErrorAction SilentlyContinue
Remove-Variable secureKey -ErrorAction SilentlyContinue
~~~

### 7. 请求追踪与排障

[request_logging.py](request_logging.py) 使用纯ASGI中间件，在每次HTTP请求进入时生成UUID4编号，写入 `request.state.request_id`，并随响应头 `X-Request-ID` 返回。普通链通过 `RunnableConfig.metadata` 传递编号，SSE通过显式参数传递；它不进入模型Prompt，也不存入共享模型或Qdrant。旧CLI未提供请求编号时相应日志字段为 `null`。

日志消息经现有 `uvicorn.error` 通道输出，`INFO`不表示错误；当前默认在启动Uvicorn的服务端终端查看，没有自动保存为日志文件。curl和探针显示的是客户端响应，不会自动显示服务端日志。

| 事件 | 记录时机与字段 |
| --- | --- |
| `service_config` | 初始化完成、进入服务前记录一次请求模型、SDK超时、重试次数及输出上限，不记录密钥 |
| `stage_finished` | `measure_stage`退出时记录请求ID、阶段、completed/error/interrupted、异常类型、可取得的上游HTTP状态和耗时 |
| `generation_usage` | 完整生成后、答案契约前记录实际响应模型及prompt/completion/total token；用量缺失保留null，0保留为0 |
| `request_finished` | 下游ASGI调用退出后记录请求ID、方法、路径、观察到的HTTP状态及服务端总耗时，覆盖流式生命周期和退出清理 |

阶段边界为：`retrieval`覆盖当前混合召回、融合及精排；`generation`覆盖普通生成调用或流式完整迭代及完成事件检查；`answer_contract`覆盖 `parse_answer` 的JSON、字段与引用映射检查。证据映射、Prompt组装、排队、响应序列化等未逐项单独计时，三段之和不要求等于总耗时。流式阶段跨越yield暂停，可能包含下游等待，不是纯供应商推理耗时。

正常有证据时，按同一请求ID依次查找检索、生成、用量、答案契约及请求结束日志；无证据只有检索阶段和请求结束，输入422没有业务阶段。生成失败不继续契约阶段；契约失败仍保留此前已取得的用量。截断、断开或其他异常可能没有用量事件，不代表未计费，不自行补算成本。

定位错误时联合查看 `stage`、`error_type`、`upstream_http_status`；后者是供应商响应状态，不是本地HTTP状态。没有上游响应时保留null；SSE头已发出后只能通过error事件报告业务失败。响应头尚未产生就退出时，中间件观察到的 `http_status` 也可能为null；不能将request_finished、阶段completed或HTTP200直接当作答案正确。

自定义日志不主动记录问题、Prompt、证据正文或异常原文；完整模型响应、底层库及部署日志仍需独立审查，当前不宣称全链路自动脱敏。批量评测报告与HTTP日志尚未打通，未配置外部日志平台、日志轮转或跨服务分布式追踪。

Windows本机先按进程、端口、HTTP、业务日志的顺序排查，不在未确认归属时终止进程：
~~~powershell
$listener = Get-NetTCPConnection -LocalPort 8001 -State Listen
Get-Process -Id $listener.OwningProcess
curl.exe -i http://127.0.0.1:8001/health
~~~

服务端Ctrl+C正常停机后，`curl.exe --max-time 3 http://127.0.0.1:8001/health` 应连接失败；启动前先确认没有其他服务占用该端口。Linux/容器复现仍需单独验收，不由Windows检查替代。

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
- `eval_test_v3.json`：100道test，80道可回答、20道拒答或澄清题，197条参考事实；从历史10题扩展，不将各版本数量相加。`eval_test_v1.json`及v2保留为历史版本，不再作为当前终测入口。
- `evaluation_config_dev_v3.json`、`dev_retrieval_gold_v3.json`：当前开发集配置及完整chunk_id/替代证据、参考事实金标。`evaluation_config_test_v3.json`、`test_retrieval_gold_v3.json`：当前测试配置和80道可回答题的精确块金标。测试事实来自v3题集，复核入口优先使用独立金标的事实映射，未提供时保留题集事实。
- 配置文件不直接选择数据库，实际集合由 `vector_store.COLLECTION_NAME` 指定；入口核对标注与报告的来源、哈希、集合和解析版本，避免混评。
- `load_evaluation_inputs`默认要求dev；`compare-test`、`qa-test`和`check-test`显式要求test，防止误混。参考答案与事实清单只用于评测，不进入被测Prompt。
- 测试题集v3的SHA-256为 `1253735b8ed9954fd7f8e96e958aa0820665cdf6247851d04963a43593e2bd9e`，不是PDF的content_hash；格式化也会改变指纹。冻结清单见 [test_freeze_v3.json](test_freeze_v3.json)。

### 1. 检索、评分与保存

~~~powershell
python -X utf8 -B evaluation.py compare evaluation_config_dev_v3.json dev_retrieval_run.json
python -X utf8 -B evaluation.py summary dev_retrieval_run.json
~~~

`compare`生成30题×4策略共120条检索记录；`compare-test`使用相同链路读取test，100题生成400条检索记录。每路保存候选Top10至 `candidate_results`，再以 `results = candidates[:top_k]`得到最终上下文，同时保存来源、模型和配置快照。均不调用生成API；输出已存在时拒绝覆盖。

| 指标 | 本项目定义与适用范围 |
| --- | --- |
| 标注页Hit/MRR | 任一结果命中标注资料与页码的题目比例/首次命中倒数排名均值；Hit@1另查首条结果 |
| 必要块Recall@K | 每题命中必要证据块的比例，再按题平均；旧block_id标注仍兼容，当前完整金标使用chunk_id |
| 必要块MRR/全齐率 | 首次命中任一必要块的RR均值/整题找齐全部必要块的题目比例；首个命中不等于全部找齐 |
| 候选必要块Recall@10 | 对最终截取之前的候选计算必要块覆盖，区分候选漏召回与最后截断损失 |
| Precision@K | 人工标注相关返回块数除以配置K，再按可回答题平均；返回不足K时缺位不算相关 |
| 关键事实覆盖率 | 答案正确覆盖的必要事实数除以该题事实数，再按可回答题平均，不是检索文本包含事实的比例 |
| 任务/忠实度/引用/拒答 | 分别检查是否完成题目、陈述是否受上下文支持、引用是否支持相应陈述，以及是否应拒答；均为本项目逐题复核通过率，不等同Ragas模型裁判分数 |

检索及相关性/事实指标只在适用的可回答题上统计，当前dev为24题、test为80题。替代证据按已核验的候选集合评分，先比较覆盖率，再比较首次命中RR，最后按方案顺序破同分；不临时按模型答案改金标。单一标注页不穷尽有效证据。False与0是有效失败，不得从分母排除；未评和不适用不自动算0分或通过。

### 2. 用保存上下文生成

先预检，命令没有 `--run` 时不调用API、也不创建QA文件：
~~~powershell
python -X utf8 -B evaluation.py qa evaluation_config_dev_v3.json dev_retrieval_run.json all dev_qa_run.json
~~~

确认供应商配置及费用后，在同一命令末尾加 `--run` 才真实生成。`all`可换为单个题号。`qa_methods`缺省为四路，dev全量最多120次；当前test配置为 `["rerank"]`，只生成100条精排记录。该字段不改变四路检索快照。

该入口将已保存的 `results`适配为Document，不重新检索或按当前配置重新截取K。更改TopK须在dev中重新检索或对已保存候选重新截取/评分，不能只改配置后回放旧上下文。逐条保存原始输出、状态、可信引用、模型、用量和 `qa_seconds`；`run_status=complete`只表示循环结束，不代表全对。失败保留，当前无断点续跑、原子检查点或自动重试。

### 3. 人工复核与汇总

~~~powershell
python -X utf8 -B evaluation.py review dev_qa_run.json dev_qa_review_run.json dev_retrieval_gold_v3.json
~~~

在独立副本填写 `review`后汇总；可在未完成时查看已评数与待评数，但不能冒称全量成绩：
~~~powershell
python -X utf8 -B evaluation.py summary dev_qa_review_run.json
python -X utf8 -B evaluation.py export dev_qa_review_run.json dev_rerank_summary_run.md
~~~

`qa-review-v2`在原有通过/忠实度/引用/拒答标签、失败类型及说明之外，增加 `chunk_relevance`逐块相关性与 `fact_checks`逐事实覆盖。未填为null，完成后适用项使用布尔值；正例逐块/逐事实检查不全时，相应分数显示待评。负例这两项不适用。单条多标签可重叠，标签数不等于失败题数。

`export <review.json> <output.md> [method]`默认选择rerank，也可显式选择vector/bm25/hybrid。标题、题数、K、配置、金标及review指纹从输入读取；复用已有汇总函数，输出指标定义和分母、回答/拒答分层、耗时/Token及任务失败详情，不重新调用模型。纯拒答且无正向事实时引用可不适用，不能把null填成True凑通过率。

QA耗时包含保存上下文适配、Prompt、SDK生成和契约校验，不含此前检索、HTTP传输或并发排队；P50/P95使用NumPy默认线性分位数。缺失用量单列，不按0计算，Token数不直接代表费用。失败详情目前只列工程error或task_pass=False；例如test_91的拒答任务通过、附加建议证据不足，详细标记保留在review，阅读时仍须同时看忠实度与引用指标。

### 4. 冻结与正式终测

以下为新运行示例，使用未存在的输出名；已完成的v3终测不要覆盖或为提高分数反复重跑。

~~~powershell
python -X utf8 -B evaluation.py check-test evaluation_config_test_v3.json
python -X utf8 -B evaluation.py freeze evaluation_config_test_v3.json test_freeze_run.json
python -X utf8 -B evaluation.py check-freeze evaluation_config_test_v3.json test_freeze_run.json
python -X utf8 -B evaluation.py compare-test evaluation_config_test_v3.json test_retrieval_run.json
python -X utf8 -B evaluation.py qa-test evaluation_config_test_v3.json test_retrieval_run.json all test_qa_rerank_run.json
~~~

最后一条只是100条精排记录的计费预检；真正生成前再次执行 `check-freeze`，并仅在该 `qa-test`命令末尾加 `--run`。完成后：

~~~powershell
python -X utf8 -B evaluation.py review test_qa_rerank_run.json test_qa_rerank_review_run.json test_retrieval_gold_v3.json
# 逐题填写新副本的review后再导出
python -X utf8 -B evaluation.py export test_qa_rerank_review_run.json test_rerank_summary_run.md
~~~

冻结清单记录题集、金标、Metadata、配置、requirements及项目顶层Python文件的指纹，以及模型地址/名称、Prompt与检索常量；不记录密钥。校验是主动短命令，尚未自动挂入检索/生成入口，不锁数据库或远端模型，也不把文件设为只读。改注释/格式也可能触发漂移。后续修复另建版本；分析过的test可用于回归，但不能再冒充未见过的独立终测。

### 5. 本地检索基准

~~~powershell
python -X utf8 -B evaluation.py benchmark evaluation_config_textflow_v1.json retrieval_benchmark_textflow_run.json
~~~

模型/检索器构建单独记录，各路径预热一次；30题×两策略×3轮保存180条耗时、有序块ID和页码。仅衡量CPU热检索，不含生成或HTTP，不证明LLM回答稳定。该报告类型当前不接原 `summary` 的质量汇总分支，不应用它计算答案通过率。

所有报告示例使用新文件名。正式实验应保留原始输出与独立review；报告含手册正文，不默认随公开仓库分发。

清理时保留正式检索基线、原始QA、独立review和性能报告；仅在核对内容覆盖及引用后删除冗余中间文件。`__pycache__/`、`.pytest_cache/`可重新生成，`.qdrant/`则是持久化数据库，不作为缓存删除。

## 已验证结果

工程状态及当前终测报告核验于2026-11-20。各表按所列题集、金标和运行分别统计；历史开发集对照不与当前终测混算，单手册结果不代表生产流量或跨文档泛化。

### 数据与解析

354个物理页，334页有提取文本，20页经历史渲染核验仅见空白/裁切标记。切块604条，其中142条表格行；检测到27个表格区域，不等于所有视觉表格均被识别。

活动库按现有流程重新解析、切块后只读核对：实际ID与预期ID均为604，两个方向差集为0；此前完整payload对齐也无差异，当前没有需清理的多余记录。该核对没有调用正式清理函数，也不证明所有页面语义解析正确。

新旧解析全书对照中，每页非空白字符及出现次数一致，表格记录含邻文保持一致，305页正文输出改变。197/244页部分段落归属更连贯，但页码/标题位置也改变，不能由字符未丢或块数相同推出全书阅读逻辑正确。

### 历史解析与Prompt对照

下表保留历史原评审口径，每格为同30道dev的整题完整输出通过数。后续事实金标修订后的TopK比较单列，不直接按两张表的差值归因于模型改善。

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

### 开发集TopK选择

同30道dev、相同生成配置下比较精排Top3/Top5；使用v3事实金标与v4复核副本：

| 精排配置 | 任务通过 | 可回答题关键事实覆盖率 | Precision@K |
| --- | --- | --- | --- |
| Top3 | 30/30 | 100%（24题平均） | 56.94% |
| Top5 | 30/30 | 100%（24题平均） | 45.00% |

来源为 `dev_qa_top3_review_v4.json`、`dev_qa_top5_review_v4.json`；参考事实按问题必需范围修订，沿用原始 `dev_qa_top3_v2.json`、`dev_qa_top5_v2.json`答案，未因重评再次生成。复核口径修订不是模型提升。开发集未观察到Top5任务质量收益，因此选择Top3后冻结，不按test成绩反选K。

### 100题冻结终测

四路检索保存400条记录；仅rerank回放生成100条，80道可回答、20道不可回答或应澄清。模型响应均为 `deepseek-v4-1-flash`、Prompt均为 `cockpit-rag-json-v2`，100条generated且无工程error。generated只证明生成和程序契约通过，不证明事实正确。

| 指标 | 本轮结果 | 分母/口径 |
| --- | --- | --- |
| 任务通过 | 92/100，92.00% | 全100题复核 |
| 可回答题通过 | 72/80，90.00% | 可回答题；其中5题拒答 |
| 不可回答题正确拒答 | 20/20，100% | 不可回答/应澄清题 |
| 必要块Recall@3 | 97.60% | 80题的单题覆盖率均值 |
| 必要块全齐率 | 75/80，93.75% | 找齐全部必要证据的题数 |
| 候选必要块Recall@10 | 98.85% | 80题的候选覆盖率均值 |
| 必要块MRR | 0.9688 | 80题首次命中RR均值 |
| Precision@3 | 59.58% | 80题逐块相关性评分；143/240个返回块相关 |
| 关键事实覆盖率 | 95.21% | 80题的单题事实覆盖率均值；共197条事实，不能把197当题数 |
| 忠实度 | 98/100，98.00% | 实际陈述是否受最终上下文支持 |
| 引用通过 | 93/95，97.89% | 适用95题；5条纯拒答不适用，不记为通过 |

原始检索为 [test_retrieval_v3.json](test_retrieval_v3.json)，原始QA为 [test_qa_rerank_v3.json](test_qa_rerank_v3.json)，独立复核为 [test_qa_rerank_review_v3.json](test_qa_rerank_review_v3.json)，完整定义、配置指纹和案例见 [test_rerank_summary_v3.md](test_rerank_summary_v3.md)。复核填写100条任务标签、240个上下文相关性标签和197条事实检查；仅修改review，原始响应和金标保持不变。这是助手逐题复核的单轮结果，不宣称独立多人评审一致性。

本轮8道任务失败及边界：

- `test_07`：候选Top10已含两项必要证据，最终Top3只保留一项，属于最终排序/截断损失；模型合理拒答，系统任务仍失败。
- `test_70/71/72`：必要术语在候选Top10中也未齐，不能只靠把最终K从3改为5解决。test_72还要求4个独立行块，Top3存在容量限制；test_70另把未召回条目说成全表不存在，属于不受证据支持的断言。
- `test_32/78/80`：必要证据已齐，但遗漏逐窗初始化、鼓风机前提或充电安全条件；属于生成完整性问题，不是检索失败。
- `test_62`：原PDF车型单元格存在 `MR6453DCH-`换行`EV02`写法，模型未确认与题干无连字符写法同一归属；按既定金标未通过，保留原始输入歧义，不简单认定编造。
- `test_91`：拒绝虚构实时胎压的任务通过，但额外的刷新后查看建议仅引用到截断前提，忠实度/引用另扣分；它不在当前导出器的任务失败列表中。

QA耗时P50为1.431秒、P95为1.946秒；已报告输入84,797、输出6,268、合计91,065 Tokens，100条用量均有记录。该耗时不含检索或HTTP，不与历史检索P95相加冒充端到端P95，也不作为生产SLA。

后续dev修复方向为候选覆盖、证据组合/截断、已知安全条件完整性，以及有来源核验的车型文本规范化。这里仅记录方向，不修改冻结题集、金标或本轮分数；修复后的同题测试属于回归，不再声称独立未见终测。

### 历史检索基线

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

API已采用新解析加混合召回/精排，CLI保留不精排的混合基线；冻结终测采用的也是混合加精排，但运行方式为保存上下文回放，不是100次HTTP端到端测试。

## 自动测试

~~~powershell
python -X utf8 -B -m pytest -q -p no:cacheprovider
python -m pip check
~~~

2026-11-20实跑122项通过，pip check无依赖冲突。本轮仅使用已有依赖，无需因评测和导出新增requirements条目；这不等于干净机器已能完整复现。

| 覆盖范围 | 项数 |
| --- | --- |
| 切块边界、重叠、Metadata、ID与非法参数 | 5 |
| 临时Qdrant持久化、upsert边界、精确删除、缩块清理、范围隔离与BM25重建 | 4 |
| 文本/结构化Prompt、空输入与不可信资料边界 | 4 |
| RRF排序、两路信息与输入不变 | 1 |
| 假精排排序、同分及非法分数 | 4 |
| 结构化JSON/字段/状态/引用身份契约及语义校验边界 | 14 |
| 题集语义、split保护、ID/替代证据、候选截断、问答回放、复核指标及报告渲染 | 46 |
| API 请求模型：去空白、类型、长度与额外字段校验 | 7 |
| 生成客户端复用、关闭与响应边界 | 6 |
| 供应商流式：片段拼接、两类usage位置、截断/空输出拒收及响应流关闭 | 6 |
| 流式问答编排：空证据、完成事件、合法/未知引用及关闭向上传递 | 5 |
| 离线 HTTP：普通/流式引用、输入与错误事件、忙碌503、断开清理/恢复、请求ID及阶段/用量日志 | 17 |
| 阶段计时工具：正常/异常记录、异常向外传递、上游状态及不记录异常原文 | 3 |

测试不调用付费API，不运行Embedding/精排推理或冻结test问答。临时库与人工向量只测工程行为，不评估语义质量。存储测试覆盖重复写入不增量但遗漏块仍在、删除后的空查询与其他范围保留、非法身份、空/缺失保留ID、2块缩为1块、重复清理，以及旧BM25与重建实例的区别。汇总用例覆盖False保留、null分母、标签去重、输入不变和严格评分类型；离线 HTTP 使用检索和生成替身，ASGI模拟首个delta后断开并检查上游关闭、锁释放和下一次问答。TestClient缓冲不能证明真实事件到达时间，模拟断开也不代替真实网络或并发测试。

冻结创建、校验、防覆盖、文件/模型漂移和缺失环境变量另经临时目录检查，不计入122项pytest；正式v3清单已校验通过。历史另用假解析结果、2维单位向量和临时库验证`retrieval.main()`写入/清理接线，亦不计入122项；两者不等于真实PDF全量重新入库、并发更新或故障恢复验收，没有改写或删除正式604条记录。

## 代码导航

| 模块 | 主要职责 |
| --- | --- |
| [table_parser.py](table_parser.py)、[ingestion.py](ingestion.py) | 正文/表格解析、可选页对照、哈希与Metadata |
| [chunking.py](chunking.py)、[retrieval.py](retrieval.py) | 切块、编码、入库及FAISS原理基线 |
| [vector_store.py](vector_store.py) | Qdrant写入、来源/版本过滤、分批读取payload、精确版本删除与多余块清理 |
| [bm25_retrieval.py](bm25_retrieval.py)、[hybrid_retrieval.py](hybrid_retrieval.py) | 同范围关键词召回与RRF |
| [reranking.py](reranking.py)、[langchain_retrieval.py](langchain_retrieval.py) | 精排及三种检索器适配 |
| [answer_contract.py](answer_contract.py)、[langchain_prompt.py](langchain_prompt.py) | 本次证据映射、结构化消息、JSON/引用身份校验 |
| [generation.py](generation.py)、[langchain_query.py](langchain_query.py) | 普通/流式供应商调用、结构化问答链与临时/最终事件编排 |
| [api.py](api.py)、[api_schemas.py](api_schemas.py)、[test_api.py](test_api.py) | 本地服务入口、请求与响应模型、离线接口测试 |
| [request_logging.py](request_logging.py)、[test_request_logging.py](test_request_logging.py) | 请求关联、完整生命周期日志、阶段计时、用量白名单及工具回归 |
| [stream_probe.py](stream_probe.py)、[http_stream_probe.py](http_stream_probe.py) | 供应商JSON流探针、HTTP流式及主动断开恢复验收 |
| [test_stream_generation.py](test_stream_generation.py)、[test_stream_qa.py](test_stream_qa.py) | 生成层与问答编排层的离线流式回归 |
| [prompt_builder.py](prompt_builder.py)、[query.py](query.py) | 直接文本基线 |
| [evaluation.py](evaluation.py)、[test_evaluation.py](test_evaluation.py) | dev/test输入、检索/生成、替代证据与事实评分、冻结/漂移检查、复核/Markdown导出及离线测试 |

### 评测入口与函数关系

`main()`按子命令选择分支并返回，不按函数定义顺序执行全部流程。

| 子命令 | 主调度与职责 | 调用生成API |
| --- | --- | --- |
| `compare` / `compare-test` | `run_retrieval_evaluation`组织对应split的四路检索、评分、汇总及保存 | 否 |
| `score` | `main`复用 `add_block_scores`、`summarize_retrieval`、`write_report`，只给已有报告补分 | 否 |
| `qa` / `qa-test` | `run_saved_qa`按qa_methods筛行并校验完整组合，`evaluate_qa_row`回放Document及原问答链 | 仅显式加 `--run` 后 |
| `review` | `prepare_qa_review`绑定事实金标/指纹并创建空review副本，需逐题填写 | 否 |
| `summary` | `summarize_retrieval`或 `summarize_qa_review`汇总已有分数，不自动判断答案 | 否 |
| `export` | `export_final_report`复用两类汇总与渲染函数，将选中策略输出为新Markdown | 否 |
| `check-test` | `load_evaluation_inputs(..., expected_split="test")`校验题集/范围/指纹及金标覆盖 | 否 |
| `freeze` / `check-freeze` | `freeze_evaluation`创建或比较配置/文件/模型状态快照，复用write_report防覆盖 | 否 |
| `benchmark` | `benchmark_retrieval`构建两检索器，预热后重复检索计时 | 否 |

共用的 `load_evaluation_inputs`默认只读dev，test需显式指定；读取题集、Metadata与金标并核对身份、split及指纹。`score_retrieval`和 `score_required_blocks`只比较已有结果，`add_block_scores`负责范围核对与主/替代证据选择，`write_report`只保存报告。旧位置参数入口仍由 `inspect_retrieval`执行单路检索，保留 `"w"`覆盖，不作为常规批量评测入口。

## 限制与后续方向

- 只有单份真实资料；原生表格检测与表头处理有启发式边界，未实现OCR、跨页表格拼接或完整版面理解。text-flow不能保证任意PDF阅读顺序。
- 车型/年款未知，页眉坐标和Metadata配置针对当前手册；尚无可靠的车型过滤或自动适用性判断。
- 已完成30题dev对照和100题固定配置终测，但只有一份手册、单轮供应商运行及助手逐题复核；数据代表性、标注一致性和统计不确定性尚未充分量化。终测已被分析，未来同题结果应标作回归，精排与目录干扰也未消失。
- JSON/可信ID校验不等于自动引用语义评测；正文格式、证据充分性和拒答仍需回归，旧error行可能缺Prompt版本等元信息。
- 保存QA逐条落盘但非原子检查点，无断点续跑或自动重试；批量评测报告尚未关联HTTP请求日志，API错误保留，不自动改写为正常拒答。
- 入库CLI已接入同范围旧块清理，仅按单进程离线作业验证；尚无自动版本切换、写入/清理原子事务、并发更新或失败回滚，也未实现仅更新payload、跳过重复编码。保留旧集合不等于自动版本管理。
- FastAPI 仅按本机单 worker 验证；进程内共享锁串行保护普通与流式问答，流式忙碌直接503。服务运行期间不要另起进程对同一 Qdrant 目录入库或清理。本地多进程、并发容量、故障恢复及新环境完整复现未验收；尚无 Docker 和 CI。
- SSE已验证本机单题与主动断开后恢复，已有请求ID与三个粗阶段的本地日志；未实现断点续传、自动重连、前端流式界面或分布式追踪。流式降低等待完整输出前的可见延迟，不保证总耗时更短；取消不保证供应商即时停止或完整用量回传。
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
- [FastAPI原生SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events/)
- [FastAPI请求级yield依赖](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/)
- [AnyIO取消与清理保护](https://anyio.readthedocs.io/en/stable/cancellation.html)
- [ASGI消息规范](https://asgi.readthedocs.io/en/stable/specs/www.html)
- [LangChain运行配置](https://reference.langchain.com/python/langchain-core/runnables/config/RunnableConfig)
- [HTTPX超时边界](https://www.python-httpx.org/advanced/timeouts/)
- [pytest日志捕获](https://docs.pytest.org/en/stable/how-to/logging.html)
