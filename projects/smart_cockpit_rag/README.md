# 智能座舱汽车知识大脑 RAG

当前进度：Week4 Day6 命令行最小 RAG 闭环。已完成 PDF 解析、文本切块、来源与内容哈希传递、初版评测题集校验、BGE Embedding + FAISS 精确检索基线，以及 Qdrant 本地持久化、稳定 ID 幂等写入、来源与文件版本过滤；新增结构化证据 Prompt 和 OpenAI-compatible 生成适配层，由 `query.py` 串联检索、生成和引用展示。

一条座舱问题已使用豆包完成真实回答与人工引用核对。后续生成测试固定使用火山方舟 `deepseek-v4.1-flash`；当前尚未收到该配置的完整 RAG 输出，不将其记录为已通过的端到端结果。完整拒答判断、自动引用支持核验和题集质量评测尚未实现；`evaluation.py` 仍只校验题集字段。

## 当前数据流程

```text
PDF 文件 + source_id（同一资料跨版本保持不变）
  -> ingestion.py: 计算文件 SHA-256，逐页提取文字，记录无文本页
  -> 页面记录: 正文 + 文件名 + source_id + 内容哈希 + PDF 物理页码
  -> chunking.py: 在每页内按字符窗口切块
  -> chunk 记录: 正文 + 页面来源信息 + 切块参数 + 稳定 chunk_id
  -> retrieval.py: BGE 将正文编码为归一化的 512 维向量
  -> vector_store.build_points: 组装稳定 UUID、向量和 chunk payload
  -> vector_store.save_points: upsert 到项目目录下的 .qdrant/，重复 ID 更新原记录
```

```text
query.py: source_id + content_hash + 问题
  -> retrieval.search_stored_chunks: 编码问题，得到 (1, 512) 的向量批次
  -> query_batch[0]: 取出形状为 (512,) 的单个问题向量
  -> vector_store.search_points: 同时过滤 source_id 和 content_hash，执行余弦检索
  -> 检索结果: payload 中的正文、来源、页码等字段 + 相似度分数
  -> prompt_builder.build_messages: 编号证据、来源、物理页码 + system/user 消息
  -> generation.generate_answer: 读取模型配置，调用生成 API，检查完整性与非空回答
  -> 输出: Prompt 版本、答案和模型生成的引用、响应模型标识、token 用量
```

导入时 `chunks[i]` 与 `vectors[i]` 必须一一对应，`build_points()` 按相同顺序配对，将正文及来源字段保存到对应向量的 payload；独立查询时由数据库返回 payload，不再依赖原进程中的 `chunks` 列表。

FAISS 的 `build_faiss_index()` 和 `search_chunks()` 保留为 Day4 学习基线：通过命中位置读取 `chunks[position]`，不能单独打乱 chunk 顺序。当前 `retrieval.py` 的 `main()` 仍会建立 FAISS 索引并做自检，但真实问题已走 Qdrant 查询；`query.py` 不调用这个 `main()`，不重新解析 PDF、编码资料或建立 FAISS 索引。FAISS 支持保存索引，但本项目基线尚未实现索引保存，不将其作为第二套最终存储后端。

## 模块职责

| 文件 | 当前职责 |
| --- | --- |
| `ingestion.py` | 计算文件哈希、逐页提取文本、保留 PDF 物理页码及来源 |
| `chunking.py` | 按页切块，传递 metadata，生成可读的稳定 chunk ID |
| `retrieval.py` | 加载 BGE、编码文本、保留 FAISS 基线，并提供持久化检索函数 |
| `vector_store.py` | 将 chunk ID 映射为 UUID，创建或检查集合、写入记录、过滤检索、关闭客户端 |
| `prompt_builder.py` | 组织证据与 system/user 消息，要求依据资料回答、引用及无依据拒答，记录 Prompt 版本 |
| `generation.py` | 读取密钥、地址和模型环境变量，调用兼容 API，检查回答完整性，返回回答及用量 |
| `query.py` | 读取来源、文件哈希及问题，串联持久化检索、Prompt 组装、生成和终端展示 |
| `evaluation.py` | 校验评测题集字段，目前不执行检索或生成评测 |
| `test_chunking.py`、`test_vector_store.py` | 检查切块边界、来源传递、幂等写入、持久化及范围隔离 |

## 环境与运行

当前实测环境为 Python 3.9.23、`pypdf==6.19.0`、`pytest==8.4.2`、`numpy==1.26.4`、`sentence-transformers==5.1.2`、`torch==2.8.0+cu129`、`qdrant-client==1.15.1`、`openai==2.48.0` 和 CPU FAISS 1.11.0。Embedding 模型为 `BAAI/bge-small-zh-v1.5`。导入时模型编码可使用 GPU，编码结果转为 CPU 上的连续 NumPy `float32` 数组；独立查询入口明确使用 CPU。

`requirements.txt` 已记录 PDF 解析、测试、NumPy、PyTorch、Sentence Transformers、Qdrant 客户端和 OpenAI SDK 的版本；FAISS 按本机实际安装方式单独记录为 Conda 依赖。这里使用 OpenAI SDK 作为兼容协议客户端，不代表模型请求发送到 OpenAI 服务。当前代码仍通过 `retrieval.py` 导入 `faiss`，暂不能移除该依赖。现有环境已安装这些依赖，不需要重复安装；以下命令从本项目目录执行，尚未在全新环境中验证整套安装流程：

```powershell
python -m pip install -r requirements.txt
conda install -c pytorch faiss-cpu=1.11.0 numpy=1.26.4
```

本机 FAISS 来自 PyTorch Conda channel 的 Windows/Python 3.9 CPU 构建，不将这个安装步骤写成已验证的通用 pip 安装方式。PyTorch 的 CUDA 构建需按硬件和官方安装说明选择；上述版本记录不表示任意设备都应安装 CUDA 12.9 构建。首次加载 BGE 需要下载模型或准备可用的本地缓存，安装 Python 包不等于模型权重已经可用。

Qdrant 当前使用 `QdrantClient(path=...)` 本地模式，无需启动服务器或 Docker。存储目录由 `vector_store.py` 的位置确定，因此切换终端工作目录不会切换数据库。该目录已通过 `.gitignore` 忽略，不提交数据库文件。本地模式适用于开发、原型和测试，不把它等同于已验证的服务端并发或大规模检索方案；参考 [qdrant-client 1.15.1 官方说明](https://github.com/qdrant/qdrant-client/blob/v1.15.1/README.md)。

从本项目目录运行，注意不要在 `WEEK` 根目录直接执行 `python retrieval.py`：

```powershell
$pdf = "..\..\..\阶段12：智能座舱汽车知识大脑\丁师兄综合项目实战项目\综合项目实战项目一\data\train_a.pdf"
python ingestion.py $pdf training_manual_a
python chunking.py $pdf training_manual_a
python retrieval.py $pdf training_manual_a
python -m pytest -q test_chunking.py test_vector_store.py
```

若终端当前位于 `WEEK` 根目录，使用下面这一组路径，不与上一组相对路径混用：

```powershell
$pdf = "..\阶段12：智能座舱汽车知识大脑\丁师兄综合项目实战项目\综合项目实战项目一\data\train_a.pdf"
python .\projects\smart_cockpit_rag\retrieval.py $pdf training_manual_a
python -m pytest -q .\projects\smart_cockpit_rag
```

上面的 `$pdf` 是训练营目录中的本地样本路径。使用其他有权处理的 PDF 时，把它替换为对应文件路径，并为不同资料分配不同的 `source_id`；同一资料更新后沿用原 `source_id`。原始车主手册不随本项目代码提交；使用前应确认资料的使用和再分发权限。

### 生成模型配置

后续学习测试固定使用火山方舟 DeepSeek-V4.1-Flash，不继续进行豆包与 DeepSeek 官方接口的来回切换。当前用户选择的 Coding Plan 连接配置为：

| 环境变量 | 内容 |
| --- | --- |
| `LLM_API_KEY` | 用户自己的火山 Coding Plan API Key，不写入仓库或日志 |
| `LLM_BASE_URL` | `https://ark.cn-beijing.volces.com/api/coding/v3` |
| `LLM_MODEL` | `deepseek-v4.1-flash` |

适用范围：上述配置仅记录当前学习环境，不表示普通业务调用获得了套餐授权。官方说明 Coding Plan 额度用于支持的 AI 编程工具，不能用于普通 API 调用；正式业务应改用允许业务调用的普通方舟 API 及其对应密钥、地址和 Model ID。普通 API 费用不计入 Coding Plan 套餐，不应混用两类地址和凭证。参见 [套餐使用范围](https://docs.volcengine.com/docs/ark/coding-plan-personal-plan-overview?lang=zh) 和 [Coding Plan 模型与地址](https://docs.volcengine.com/docs/ark/coding-plan-personal-ai-deepseek-harness?lang=zh)。

密钥输入命令单独执行，等待提示后再粘贴真实密钥；引号内是提示文字，不是密钥：

```powershell
$secureKey = Read-Host "Paste your Volcengine API Key" -AsSecureString
```

输入完成后，在同一个终端配置其余环境变量；以下只记录当前学习环境的连接参数，不将模型写死在业务代码中：

```powershell
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
$env:LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
$env:LLM_MODEL = "deepseek-v4.1-flash"
```

这些变量仅在当前终端及其子进程生效，新终端需要重新配置。当前代码没有自动加载 `.env`，不能只创建该文件就假设配置已被读取。更换生成模型不会重建 BGE 文档向量；更换 Embedding 模型或编码方式则需要重新考虑向量与集合的兼容性。配置别名不等于永久固定底层权重，回归时应记录实际返回的 `model`、Prompt 版本和日期。

`generation.py` 当前固定 `max_tokens=1024`、`stream=False`、关闭思考、60 秒网络操作超时和 `max_retries=0`。`finish_reason` 不是 `stop` 或回答为空时会抛出异常；配置缺失、鉴权、余额、超时及服务端错误目前仍直接暴露，尚无统一错误响应或生产重试策略。具体字段支持范围以所选模型和供应商接口为准。

### 查询已有数据库

先成功执行一次 `retrieval.py` 导入资料，再运行独立查询入口。以下参数对应当前本地样本；替换 PDF 后应使用新文件的实际哈希，不能沿用示例哈希。

从本项目目录执行：

```powershell
$sourceId = "training_manual_a"
$contentHash = "d6509e2e55a17e9145b9af6883ef4629c380e7213944b9fcd8ee3b050ed7d2a1"
$question = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
python query.py $sourceId $contentHash $question
```

若终端位于 `WEEK` 根目录，用同样的三个参数执行：

```powershell
$sourceId = "training_manual_a"
$contentHash = "d6509e2e55a17e9145b9af6883ef4629c380e7213944b9fcd8ee3b050ed7d2a1"
$question = "无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？"
python .\projects\smart_cockpit_rag\query.py $sourceId $contentHash $question
```

查询入口只编码问题，不需要传 PDF 路径或重新生成文档向量。有证据时会发送一次真实生成请求，使用前确认环境变量、接口使用范围及计费方式；无匹配记录时打印 `No matching evidence found.` 并提前返回，不调用生成模型。空检索拦截和 Prompt 中的拒答要求不等于已经实现完整的证据充分性判断。数据库或集合尚不存在、模型不可用或参数非法时，当前脚本仍可能直接报错，尚无统一错误处理。

## 当前验证结果

对本地 `train_a.pdf` 的一次运行结果：PDF 共 354 页，其中 334 页提取到文本、20 页未提取到文本，最终得到 469 个 chunk；首块 ID 由 `training_manual_a`、完整文件哈希、`s500:o100:p1:c1` 组成。`test_chunking.py` 的 5 个测试均通过，覆盖重叠窗口、短文本与空文本、跨页元数据归属、非法重叠参数，以及内容哈希或切块参数变化时的 ID 变化。

抽样渲染核对的 PDF 第 2 页为空白页；其余无文本页尚未逐页核验。无文本页不能仅凭提取结果认定为空白页，也可能需要 OCR 或其他解析方式。

### Day4 检索基线

完整 PDF 编码后得到 `(469, 512)` 的 `float32` 矩阵，FAISS 索引维度为 512、向量数量为 469。首行向量 L2 范数实测为 1.0；查询首行向量自身时，返回位置 0、内积分数 1.0。这是索引自检，不是 RAG 质量评测。

真实问题示例：`无钥匙解锁设为单门和全车时，触摸前门把手内部传感器区域有什么区别？`

| 命中排名 | 索引位置 | PDF 物理页码 | 相似度分数（显示四位小数） |
| --- | --- | --- | --- |
| 1 | 80 | 43 | 0.8007 |
| 2 | 81 | 44 | 0.6841 |
| 3 | 78 | 41 | 0.6371 |

第一条命中正文明确说明单门模式仅解锁当前车门，全车模式解锁全车车门。后两条主要涉及闭锁和两步解锁，主题相关但不能同等直接支持这个问题的答案。该结果只记录一条问题的实测，不代表题集上的整体召回率或答案正确率。

问题编码为 `(1, 512)`，本次搜索返回的 `scores` 和 `positions` 均为 `(1, 3)`；第一维表示问题批次，第二维表示命中排名。`scores[0]` 与 `positions[0]` 是同一个问题的三个分数和位置，`zip` 按排名配对后取回原始 chunk。

资料向量和问题向量均已归一化，因此 `IndexFlatIP` 的内积对应余弦相似度，分数越高排名越靠前。分数不是答案正确率；top-k 返回结果也不代表资料中一定存在答案。原理参考 [FAISS 相似度说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) 和 [BGE 官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)。

Day4 示例问题保留在 `retrieval.py` 的 `main()` 中；Day5 新增的 `query.py` 接收命令行问题参数。FAISS 基线 `search_chunks()` 会拒绝空问题和非正数 `top_k`，并将返回条数限制在 chunk 总数以内。Day4 采用索引自检、真实问题核对和切块回归测试，尚未新增 FAISS 检索映射自动化测试。

### Day5 持久化与范围隔离

- 本地集合名为 `cockpit_manual_bge_small_zh_v1_5`，向量维度为 512，距离规则为 COSINE。保存和查询都会核对集合的向量类型、维度和距离规则。
- `build_points()` 使用 `uuid5(NAMESPACE_URL, chunk_id)` 生成稳定 point ID，同时保留可读 chunk ID 和完整 chunk payload。相同 ID 的 upsert 更新已有记录，不增加重复条目；相同文件与处理配置下，这一点已由测试及真实导入验证。
- 真实资料库当前为 469 条记录；重复写入同一批记录后仍为 469 条。这个结果不代表已经实现新版本替换、旧记录删除或跳过重复编码。
- 查询要求明确提供 `source_id` 和 `content_hash`；两个条件在 `must` 中同时满足，避免检索混入其他资料或文件版本。哈希仅标识文件字节内容，不是已核实的车型或官方版本号。
- 两个独立查询进程均返回 PDF 第 43、44、41 页，分数显示为 0.8007、0.6841、0.6371，验证退出原进程后仍可查询；它们与 Day4 的示例结果一致，不据此推断所有问题上的等价性或质量提升。
- 项目测试共 6 项通过：5 项切块测试和 1 项向量库集成测试。后者使用临时数据库、三条人工记录及相同方向的测试向量，检查重新打开客户端后的查询、重复写入数量、来源与版本隔离、payload 字段保留和范围无匹配时返回空列表；它不评估 Embedding 的语义质量。

实现原则参考 [Qdrant point 与幂等写入说明](https://qdrant.tech/documentation/manage-data/points/)；测试成功不代表所有非法输入、崩溃恢复或生产并发路径均已覆盖。

### Day6 最小 RAG 与调用边界

- Prompt 版本为 `cockpit-rag-v1`。检索结果按本次排名编号，`[1]` 是第一条证据，不是 PDF 第 1 页；编号通过 payload 映射到真实文件名和物理页码。
- `build_messages()` 返回 system/user 两条消息，包含回答规则、检索正文、来源、页码和用户问题，不读取或附带评测集参考答案。资料中的指令被要求视为数据，但目前没有经过对抗测试，不能宣称已防御 Prompt 注入。
- 用户提供的真实豆包输出：单门模式仅解锁当前车门，全车模式解锁全车车门，引用 `[1]` 指向 `train_a.pdf` 物理第 43 页；人工核对与该页证据一致。响应模型为 `doubao-seed-2-1-pro`，输入 666 tokens、输出 58 tokens、总计 724，思考 tokens 为 0。仅验证这一条问题，不推断整体正确率或模型优劣。
- DeepSeek 官方 `deepseek-flash` 已完成用户的 `API_OK` 基础连通性验证，但不作为后续项目基线，也不将它算作完整 RAG 验证。后续基线是火山方舟 `deepseek-v4.1-flash`，其完整问答结果待学习者提供。
- 原有 6 项项目测试通过；另在进程内检查了真实检索与 Prompt 组装、空问题/空证据拒绝、空检索不调用生成，以及模拟 SDK 的回答/用量解析、截断和空回答拒绝。这些额外检查未保存为测试文件，不宣称已纳入持久化自动回归套件；模拟响应不代表真实模型质量。
- 排查时发现密钥误输入为一个控制字符，真实只读接口返回空响应的 400；正常输入密钥后基础调用成功。密钥存在不等于密钥有效，也不能只凭 400 认定模型不支持或供应商故障。

## 初版评测题集

`eval_cases.json` 目前包含 14 条题目：13 条可回答、1 条不可回答，全部暂列入开发集（`split: "dev"`）。每条题目记录问题、是否可回答及题目来源；可回答题还记录参考答案、资料 `source_id`、PDF 物理页码和原文定位词。参考答案与证据仅供后续评测使用，不应传入被测 RAG 的生成 Prompt。

从本目录检查题集格式：

```powershell
python evaluation.py eval_cases.json
```

当前程序会检查题目列表、唯一 ID、非空问题、`dev`/`test` 字段，以及答案与证据字段是否符合可回答状态；通过后打印题目数量和首题。它**不会**调用检索或生成模型，也不会计算 Recall@k、答案正确性或引用忠实度。字段校验通过不等于原文确实支持答案，证据内容仍需核对。

后续需要补充无答案、自然语言改写和车型/版本不明确等题型，扩充至本周要求的至少 20 条初版题目；在评测方案确定后，再划分冻结测试集并实现实际计分。

## 已知限制与后续工作

- 当前按字符位置切分，可能切断句子或表格；重叠会增加重复内容，尚未用问答集评估切块效果。
- `source_id` 是人为分配的资料身份，不能据此推断未核实的车型或版本。文件哈希标识字节内容版本；当前 `chunk_id` 还包含切块参数、页码和页内序号。解析器或切块算法实现变化时，仍需要进一步考虑文本哈希或处理版本。
- 相同输入及处理配置下的稳定 ID、UUID 映射和幂等写入已实现；文件哈希变化会产生新记录，旧版本仍留在数据库中，尚无完整增量更新、替换与清理流程。
- 导入入口 `retrieval.py` 每次仍重新解析并编码全部资料，FAISS 基线也仍在该入口重建；独立查询入口 `query.py` 已避免这些工作，但每次启动仍加载一次 Embedding 模型。
- 当前只验证单份真实 PDF，以及人工测试数据上的来源/文件哈希过滤；尚无核实后的车型、年款、官方手册版本 metadata、用户权限过滤或服务接口。
- Qdrant 当前为本地模式，客户端用完即关闭以释放资源和本地锁；尚未验证多进程并发读写、服务端部署、HNSW 性能、负载或故障恢复。
- 当前已实现命令行答案生成与模型生成的引用展示，尚无程序化引用映射展示、自动引用支持核验、可靠的证据不足拒答或端到端题集质量评测；一条人工核验样例不能替代回归评测。
- 更换 Embedding 模型或编码配置后，需要重新生成资料向量并重建匹配索引，不能只替换问题编码模型。集合名包含模型标识，但尚无完整模型/处理配置清单；维度相同并不能证明两个模型的向量兼容。
- 后续按学习计划补充资料解析抽查、可核实的车型/版本 metadata、生成与拒答、检索质量评测以及更新/删除流程，不把当前原型的结果当作最终项目效果或真实生产上线。
