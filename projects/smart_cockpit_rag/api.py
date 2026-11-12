import json
import os
import anyio #一个异步并发库，提供任务管理、取消控制、超时和工作线程等能力
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException
from sentence_transformers import SentenceTransformer
from generation import build_client, generate_with_client, stream_with_client
from langchain_query import build_qa_chain, stream_qa
from langchain_retrieval import build_rerank_retriever
from retrieval import MODEL_NAME
from threading import Lock
from api_schemas import QuestionRequest, QuestionResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.concurrency import run_in_threadpool

#lifespan 管整个服务的共享资源，open_qa_stream 管单次流式请求；先安全关闭这次请求的流，再释放锁
@asynccontextmanager #装饰器 @asynccontextmanager 将 lifespan 转成异步上下文管理器，让框架能够进入和退出这个资源范围。
async def lifespan(app: FastAPI): #应用启动和关闭时，lifespan 负责启动和关闭的准备工作
    path = Path(__file__).resolve().with_name("manual_metadata.json")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    model_name = os.environ["LLM_MODEL"]
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    with build_client() as client:
        retriever = build_rerank_retriever(
            model, metadata["source_id"], metadata["content_hash"]) #检索器仍使用原来的资料身份＋哈希过滤
        def generator(messages): #内部 generator 是闭包，保留外层的客户端和模型名
            return generate_with_client(messages, client, model_name, json_mode=True)
        def stream_generator(messages):
            return stream_with_client(messages, client, model_name, json_mode=True)
        #app.state 是应用保存共享资源的位置；链中保留了模型、检索器和客户端的引用，因此后续请求可以复用，不需要重新加载
        app.state.retriever = retriever
        app.state.stream_generator = stream_generator
        app.state.qa_chain = build_qa_chain(retriever, generator=generator)
        app.state.source_id = metadata["source_id"]
        app.state.content_hash = metadata["content_hash"]
        app.state.qa_lock = Lock() #锁必须在启动时创建并共享，不能在每次请求里各建一把，这样先控制本地模型并发和Qdrant重复打开冲突
        yield #yield 是暂停，不是结束。启动代码执行到这里，服务开始处理请求；停机后恢复执行并退出外面的 with，关闭 SDK 客户端

app = FastAPI(title="Smart Cockpit RAG", lifespan=lifespan)
#lifespan=lifespan 把它交给 FastAPI；定义应用时不会立刻加载模型，服务器启动应用时才执行启动部分

@app.get("/health") #注册健康检查路径
def health(request: Request):
    #ready 仅表示初始化完成；request.app 找到当前应用，再读取其 state
    return {"status": "ready", "source_id": request.app.state.source_id,
            "content_hash": request.app.state.content_hash}

#注册提交问答的接口。请求体仍是 {"question":"问题"}；`payload` 自动校验为 QuestionRequest，`request` 由 FastAPI 提供，不是客户端另传的字段。
@app.post("/qa", response_model=QuestionResponse)
def ask(payload: QuestionRequest, request: Request): #QuestionRequest进行输入校验，通过后再调用 ask()
    with request.app.state.qa_lock: #Lock 让同一时间只有一个线程执行 with 内的问答链，其他请求等待；离开或发生异常都会释放。
        result = request.app.state.qa_chain.invoke(payload.question)
        #.invoke(payload.question) 执行原来的混合召回、精排、生成和引用检查
    return result #return result 后，FastAPI 根据 response_model 校验、过滤并序列化结果，只返回声明字段

async def open_qa_stream(payload: QuestionRequest, request: Request):
    state = request.app.state
    if not state.qa_lock.acquire(blocking=False):
    #acquire(blocking=False)：立即尝试获取锁；忙碌则返回 HTTP 503，不排队等待，这是新流式接口的策略，旧 /qa 不变
        raise HTTPException(status_code=503, detail="QA service is busy")
    events = stream_qa(payload.question, state.retriever, state.stream_generator)
    try:
        yield events #把迭代器交给接口使用，自己暂停；等响应结束后，再回来执行清理
    finally: #同步迭代由框架在线程池运行；关闭操作也放入线程池。
        try:
            #CancelScope表示一个“取消控制范围”；`shield=True` 表示：暂时屏蔽来自外层取消范围的取消，让里面的清理有机会完成
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(events.close) #await：当前异步任务等待关闭完成，等待期间允许其他任务运行
        finally: #最外层 finally 即使关闭操作抛出异常，也会尝试释放锁
            state.qa_lock.release()

@app.post("/qa/stream", response_class=EventSourceResponse)
def ask_stream(events=Depends(open_qa_stream, scope="request")):
#Depends(...)：让 FastAPI 先执行 open_qa_stream，把它 yield 出来的迭代器交给 ask_stream 的 events
#scope="request"：让依赖一直存活到响应结束，再执行 finally，否则锁可能在真正流式输出前就被释放
    try:
        for event in events:
            if event["type"] == "final":
                data = QuestionResponse.model_validate(event["data"]) #model_validate → model_dump：最终结果仍经过已有响应模型检查，再转回可发送的字典
                event = {**event, "data": data.model_dump(mode="json")} #model_dump()将整个响应模型转换为 JSON 兼容字典
            yield ServerSentEvent(event=event["type"], data=event)
            #ServerSentEvent(...)：将字典编码成 JSON，并添加事件名和 SSE 格式；`delta` 是临时片段，`final` 才是完成结果
    except Exception:
        yield ServerSentEvent(
            event="error", data={"type": "error", "code": "stream_failed"})