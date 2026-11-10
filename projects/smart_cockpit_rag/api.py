import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
from generation import build_client, generate_with_client
from langchain_query import build_qa_chain
from langchain_retrieval import build_rerank_retriever
from retrieval import MODEL_NAME
from threading import Lock
from api_schemas import QuestionRequest, QuestionResponse

@asynccontextmanager #将含 yield 的异步函数变成异步上下文管理器
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
        #app.state 是应用保存共享资源的位置；链中保留了模型、检索器和客户端的引用，因此后续请求可以复用，不需要重新加载
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