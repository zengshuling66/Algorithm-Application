# Python 并发编程面试复习

用途：大模型应用算法、RAG Agent、AI 应用开发实习面试前速查。

定位：这不是高并发后端专家笔记，而是为了能在实习面试中讲清楚“Python 并发怎么选、RAG/Agent 项目里怎么用、常见坑怎么避开”。

## 一页知识地图

| 概念 | 面试回答 |
| --- | --- |
| 并发 | 多个任务在一段时间内交替推进，不一定同一时刻同时执行。 |
| 并行 | 多个任务在同一时刻真正同时运行，需要多核 CPU 或多设备。 |
| 同步 | 调用后一直等待结果返回，简单但容易阻塞。 |
| 异步 | 发起任务后让出执行权，结果好了再处理，适合 I/O 等待。 |
| 阻塞 | 当前任务等待期间不能继续执行其他任务。 |
| 非阻塞 | 等待期间可以切换处理其他任务。 |
| I/O 密集型 | 时间主要花在网络、磁盘、数据库、API 等等待上。 |
| CPU 密集型 | 时间主要花在计算上。 |

选择原则：

| 任务类型 | 推荐方式 | 例子 |
| --- | --- | --- |
| I/O 密集型 | 多线程或协程 | 请求模型 API、读写文件、访问数据库、调用向量库。 |
| CPU 密集型 | 多进程 | 复杂文本清洗、图片处理、CPU 上的大量计算。 |
| 高并发网络 I/O | 协程 | 批量 embedding API、Agent 并发工具调用、FastAPI 异步接口。 |
| GPU 推理 | 模型服务化和 batching | vLLM、SGLang，不要每个进程重复加载模型。 |

## GIL 必背

GIL 是 CPython 的全局解释器锁。同一时刻通常只有一个线程执行 Python 字节码，所以 Python 多线程不适合纯 Python CPU 密集型任务。

但多线程仍然适合 I/O 密集型任务，因为线程等待网络、磁盘、数据库时可以切换。多进程可以绕过 GIL，因为每个进程有独立解释器和独立 GIL。

面试回答：

```text
Python 多线程适合 I/O 密集型任务，不适合纯 Python CPU 密集型任务。
原因是 CPython 有 GIL，同一时刻通常只有一个线程执行 Python 字节码。
如果要利用多核 CPU 跑计算密集任务，通常使用 multiprocessing 或 ProcessPoolExecutor。
```

## 语法速查

### 线程和进程对照

| 语法条目 | 多线程 | 多进程 |
| --- | --- | --- |
| 引入 | `from threading import Thread` | `from multiprocessing import Process` |
| 创建 | `t = Thread(target=func, args=(100,))` | `p = Process(target=func, args=("bob",))` |
| 启动 | `t.start()` | `p.start()` |
| 等待 | `t.join()` | `p.join()` |
| 队列 | `queue.Queue()` | `multiprocessing.Queue()` |
| 锁 | `threading.Lock()` | `multiprocessing.Lock()` |
| 池化 | `ThreadPoolExecutor` | `ProcessPoolExecutor` |

线程共享同一进程内存，常用 `queue.Queue` 做线程安全通信。进程内存隔离，常用 `multiprocessing.Queue`、`Pipe`、`Manager` 或共享内存通信，成本更高。

Windows 下写多进程必须加：

```python
if __name__ == "__main__":
    ...
```

原因是 Windows 默认用 `spawn` 创建子进程，子进程会重新导入主模块；没有保护可能递归创建进程。

### Queue 和 Lock 最小模板

线程队列：

```python
import queue

q = queue.Queue()
q.put("task")
item = q.get()
```

进程队列：

```python
from multiprocessing import Queue

q = Queue()
q.put(["task", 1])
item = q.get()
```

加锁模板：

```python
from threading import Lock

lock = Lock()

with lock:
    # 修改共享数据
    pass
```

面试记忆：

```text
Queue 用来在线程/进程之间传递任务或结果。
Lock 用来保护共享数据，避免多个任务同时修改导致结果错误。
```

### map 和 submit

| 写法 | 特点 | 适合场景 |
| --- | --- | --- |
| `executor.map(func, items)` | 简洁，结果顺序和输入顺序一致。 | 批量处理同类任务。 |
| `executor.submit(func, item)` | 返回 `Future`，可单独控制异常、超时、结果。 | 需要精细控制任务状态。 |

面试回答：

```text
map 更像批量提交，适合简单批处理。
submit 会返回 Future，适合单独处理任务状态、异常、超时和结果。
```

## 三个必会模板

### 线程池：I/O 密集型

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


def handle_file(path):
    return str(path)


paths = ["a.pdf", "b.pdf", "c.pdf"]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(handle_file, path) for path in paths]

    for future in as_completed(futures):
        result = future.result()
        print(result)
```

### 进程池：CPU 密集型

```python
from concurrent.futures import ProcessPoolExecutor


def clean_text(text):
    return text.strip().lower()


if __name__ == "__main__":
    texts = [" A ", " B ", " C "]

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(clean_text, texts)
        print(list(results))
```

### 协程限流：API 并发

```python
import asyncio

semaphore = asyncio.Semaphore(5)


async def embed_chunk(chunk):
    async with semaphore:
        await asyncio.sleep(1)
        return f"embedding for {chunk}"


async def main():
    chunks = ["chunk1", "chunk2", "chunk3"]
    tasks = [embed_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    print(results)


asyncio.run(main())
```

## 大模型应用里怎么用

### RAG 入库

流程：

```text
扫描文件 -> 解析文档 -> 文本清洗 -> chunk 切分 -> embedding -> 写入向量库
```

并发设计：

| 阶段 | 推荐方式 | 原因 |
| --- | --- | --- |
| 扫描文件 | 普通循环或线程池 | I/O 较轻，先简单实现。 |
| 文档解析 | 线程池或进程池 | PDF/OCR/复杂解析可能偏 CPU。 |
| 文本清洗 | 进程池可选 | 大规模正则和清洗可能消耗 CPU。 |
| embedding | 协程 + Semaphore + batch | API 调用是 I/O，必须限流。 |
| 写向量库 | 分批写入 + 限流 + 重试 | 避免写入压力过大。 |

项目表达：

```text
我会把 RAG 入库拆成流水线，并按阶段选择并发方式。
embedding API 会用异步并发加 Semaphore 控制 QPS，失败任务记录日志并重试。
大文件和失败文件单独记录，避免一个文件失败影响整个入库流程。
```

### Agent 工具调用

如果多个工具没有依赖关系，可以并发调用，例如同时查知识库、查数据库、查搜索 API。  
如果工具之间有依赖，比如先检索再计算，就应该用 LangGraph/workflow 显式编排顺序。

项目表达：

```text
独立工具可以用 asyncio.gather 并发执行；
依赖工具通过 workflow 控制顺序；
每个工具都设置 timeout、异常处理和 fallback。
```

### FastAPI 服务

规则：

- 接口内部主要是模型 API、数据库、向量库、外部服务请求，可以用 `async def`。
- 接口内部是 CPU 密集型计算，`async def` 不会自动变快，可能阻塞事件循环。
- CPU 重任务应放进进程池、后台任务队列或独立服务。

面试回答：

```text
async 的优势是减少 I/O 等待，不是让 CPU 计算变快。
RAG 查询接口适合异步化；PDF 解析和复杂清洗这类重任务更适合后台处理。
```

### GPU 推理并发

不要盲目多进程加载模型。每个进程可能加载一份模型，显存很快爆掉。

更合理的方式：

- 用 vLLM/SGLang 等模型服务。
- 应用层通过 HTTP/OpenAI-compatible API 调用模型服务。
- 依赖 batching、KV cache、PagedAttention 等机制提高吞吐。

### 高并发工程补充

实习面试知道这些即可，不需要深挖源码：

| 知识点 | 面试要点 |
| --- | --- |
| 连接池 | 复用 HTTP/数据库连接，减少创建连接开销。 |
| 超时和重试 | 外部 API 必须设置 timeout、retry、backoff。 |
| 限流和背压 | 防止 API、向量库、数据库、内存被打爆。 |
| batch | embedding 和推理常用 batch 提高吞吐，但过大会增加延迟和显存压力。 |
| 缓存 | 高频查询、embedding、检索结果可缓存。 |
| 任务队列 | 文档入库、批量解析不应阻塞在线请求。 |
| 压测和监控 | 关注 QPS、P95 延迟、错误率、队列长度、API latency。 |
| SSE/WebSocket | 大模型流式输出需要处理长连接、断开、超时和异常。 |

## 高频面试题

### 1. 进程、线程、协程有什么区别？

进程是资源分配单位，内存隔离，创建和通信成本高，适合 CPU 密集型。  
线程是进程内执行单元，共享内存，适合 I/O 密集型。  
协程是用户态轻量任务，通过 `async/await` 主动让出执行权，适合高并发 I/O。

### 2. ThreadPoolExecutor 和 ProcessPoolExecutor 怎么选？

I/O 密集型任务选 `ThreadPoolExecutor`。CPU 密集型任务选 `ProcessPoolExecutor`。  
如果是大量异步网络 I/O，也可以用 `asyncio`。

### 3. asyncio 里为什么不能直接调用阻塞函数？

阻塞函数会堵住事件循环，导致其他协程无法调度。  
如果必须调用阻塞函数，可以用 `asyncio.to_thread()`、executor、进程池或后台任务。

### 4. 如何控制异步并发数？

用 `asyncio.Semaphore`。它可以限制同时执行的协程数量，避免 API 限流、连接数过高或内存暴涨。

```python
semaphore = asyncio.Semaphore(10)

async with semaphore:
    await do_request()
```

### 5. RAG 入库怎么用并发优化？

把入库拆成扫描、解析、清洗、切分、embedding、写向量库。  
I/O 阶段用线程或协程，CPU 重阶段用进程池，embedding API 用异步并发加限流，写入向量库分批并重试。

### 6. 如果要处理 1 万个文档，你怎么设计？

不一次性读入内存，而是分批处理。  
用队列或流水线解耦扫描、解析、embedding、写入。  
控制并发数和 batch size，记录失败任务，支持断点续跑。

### 7. embedding API 有 QPS 限制怎么办？

使用异步并发加 Semaphore 限流，chunk 按 batch 分组。  
失败时做重试、指数退避和日志记录。永久失败任务落盘，后续单独重跑。

### 8. Agent 同时调用多个工具时怎么提升速度？

没有依赖关系的工具可以并发调用，例如知识库、数据库、搜索 API。  
有依赖关系的工具用 workflow/LangGraph 控制顺序。  
每个工具要设置 timeout、异常处理和 fallback。

### 9. FastAPI 中什么时候用 async def？

接口内部主要是 I/O 操作时用 `async def`，例如模型 API、数据库、向量库、外部服务。  
CPU 密集型任务不应直接放在 async 接口里，应该放进进程池、后台任务或独立服务。

### 10. async 为什么不一定让程序变快？

async 只是在 I/O 等待时提高并发利用率，不能让 CPU 计算本身变快。  
如果任务是 CPU 密集型，async 可能阻塞事件循环，反而变慢。

### 11. 本地 GPU 推理为什么不要盲目开多进程？

多个进程可能各自加载一份模型，显存会迅速爆掉。  
更合理的是把模型作为独立服务，用 vLLM/SGLang 等做 batching 和请求调度，应用层通过 API 调用。

### 12. RAG 文档解析适合线程池还是进程池？

看瓶颈。文件 I/O 或轻量解析可以用线程池。复杂 PDF 版面分析、OCR、图片处理、重文本清洗偏 CPU，可考虑进程池。实际项目要用日志和压测判断。

### 13. 如何避免并发任务导致内存爆掉？

限制并发数，分批处理，流式读取文件，中间结果及时写入数据库或向量库。  
大文件和异常文件单独处理，不把所有文档、chunk、embedding 一次性放进内存。

### 14. 如果并发任务中某个任务失败，怎么办？

捕获异常，记录失败输入、错误类型和堆栈。  
网络超时、限流等可恢复错误做重试；不可恢复错误跳过并进入失败报告，不能让整个批次无记录地失败。

### 15. SSE 或流式输出和并发有什么关系？

大模型流式输出需要服务端持续返回 token 或片段。  
如果接口被阻塞，用户看不到实时输出，也会影响其他请求。  
实现时要注意异步 I/O、连接断开、超时、异常处理和日志记录。

## 项目表达模板

面试讲项目时可以这样说：

```text
在 RAG 入库阶段，我把流程拆成文档扫描、解析、chunk、embedding 和向量库写入。
其中 embedding 是外部 API 调用，属于 I/O 密集型，所以我用异步并发加 Semaphore 控制 QPS；
向量库写入采用 batch，并对失败任务记录日志和重试。
在线查询接口主要访问向量库和模型 API，所以适合 async；
但 PDF 解析、复杂清洗这种 CPU 重任务不会直接阻塞在线接口，而是放到后台任务或进程池。
```

## 复习检查清单

- [ ] 能解释并发和并行。
- [ ] 能解释进程、线程、协程。
- [ ] 能解释 GIL 对多线程的影响。
- [ ] 能判断 I/O 密集型和 CPU 密集型任务。
- [ ] 能写出线程池、进程池、协程限流模板。
- [ ] 能解释 `map` 和 `submit` 的区别。
- [ ] 能解释 `queue.Queue` 和 `multiprocessing.Queue` 的区别。
- [ ] 能解释 Windows 下多进程为什么要加 `if __name__ == "__main__"`。
- [ ] 能讲清 RAG 入库如何并发优化。
- [ ] 能讲清 embedding API 如何限流和重试。
- [ ] 能讲清 Agent 工具调用如何并发。
- [ ] 能讲清 FastAPI 中 async 的边界。
- [ ] 能讲清为什么 GPU 推理不要乱开多进程。

## 一句话总结

```text
Python 并发不是盲目开很多任务，而是先判断瓶颈：
I/O 用线程或协程，CPU 用进程，RAG 入库要分批限流，
Agent 工具调用可并发但要看依赖关系，
GPU 推理优先服务化和 batching，不要多进程重复加载模型。
```
