import json
import logging #`logging` 是标准库日志模块
from time import perf_counter #计时函数
from uuid import uuid4 #生成随机UUID的函数，用于给请求编号
from starlette.datastructures import MutableHeaders #“可修改HTTP头”的工具，避免我们手工处理底层响应头格式
from contextlib import contextmanager

logger = logging.getLogger("uvicorn.error") #取得一个名为 uvicorn.error 的日志记录器，保存到变量 logger，它不是异常对象

#给请求编号、把编号放进响应头、处理结束后记录耗时
#整体顺序是：`创建中间件对象 → 收到请求 → 编号并计时 → 交给原应用 → 包装响应消息 → 原应用退出 → 记录日志`
class RequestLoggingMiddleware:
    def __init__(self, app): #__init__ 是对象初始化方法；`self` 代表当前这个中间件对象；参数 app 是下游处理程序
        self.app = app #`self.app = app` 将它保存起来，方便之后继续交给下游处理

    async def __call__(self, scope, receive, send): #对象定义了 __call__，就可以像函数一样被调用
    #`scope` 是请求基本信息字典；`receive` 是接收请求消息的函数；`send` 是发送响应消息的函数；它们由上游传进来
        if scope["type"] != "http": #如果不是 "http"，例如服务启动/停机消息，就原样交给下游，再 return；因此不会执行后面的HTTP日志逻辑
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex #生成随机UUID，再通过 .hex 得到32位十六进制字符串
        scope.setdefault("state", {})["request_id"] = request_id
        #`setdefault("state", {})` 在没有 "state" 时创建空字典，有时就使用已有字典；然后将编号写入这个字典的 "request_id" 字段
        started, status_code = perf_counter(), None #`status_code` 暂为 None，因为此时响应还没有产生，尚不知道状态码

        async def send_with_id(message): #发送前检查消息，必要时补响应头
            nonlocal status_code #声明后面的赋值要修改外层本次请求的 status_code；没有它，赋值只会创建内部函数的局部变量，外层仍是 None
            if message["type"] == "http.response.start": #响应开始消息，里面有状态码和响应头
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id #修改这条响应消息中的头信息，加入请求编号
            await send(message) #响应正文消息也会经过包装函数，但不满足 http.response.start 条件，所以直接转发，不修改正文

        try:
            await self.app(scope, receive, send_with_id) #send_with_id：下游以后需要发送响应时，先调用这个包装函数
            #await self.app(...) 等待下游应用调用退出
        finally:
            logger.info(json.dumps({ #记录退出日志；`.info()` 写普通信息；`json.dumps()` 将字典转为JSON消息
                "event": "request_finished", "request_id": request_id, #`request_id` 为本次编号
                "method": scope["method"], "path": scope["path"],
                "http_status": status_code, #`http_status` 为观察到的响应状态码
                "total_seconds": round(perf_counter() - started, 4), #这个值是本中间件观察到的服务端处理耗时，不是完整客户端往返耗时，也不是纯模型耗时
            }, ensure_ascii=False)) #`ensure_ascii=False` 保留中文

@contextmanager
def measure_stage(request_id, stage): #记录同一个请求的不同阶段耗时；`request_id` 是本次请求编号，`stage` 是阶段名称
    started = perf_counter()
    upstream_http_status = None
    outcome, error_type = "interrupted", None #默认先设为 interrupted，因为生成器关闭等结束信号不一定属于 Exception，不能误记完成
    try:
        yield
    except Exception as exc: #取得异常对象
        outcome, error_type = "error", type(exc).__name__ #`type(exc).__name__` 取得异常类型的名字，例如 ValueError、RuntimeError 等
        upstream_http_status = getattr(exc, "status_code", None) #getattr(对象, "属性名", 默认值) 表示读取属性，没有则返回默认值
        raise #将原异常继续抛出
    else:
        outcome = "completed"
    finally: #`finally` 最后统一写日志；不记录异常原文，避免其中夹带敏感数据
        logger.info(json.dumps({
            "event": "stage_finished", "request_id": request_id,
            "stage": stage, "outcome": outcome, "error_type": error_type,
            "upstream_http_status": upstream_http_status,
            "seconds": round(perf_counter() - started, 4),
        }, ensure_ascii=False))

def log_generation_usage(request_id, generated):
    usage = generated["usage"]
    counts = None if usage is None else { #用量缺失就保留 None，写成JSON时为 null
        key: usage.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens") #三个字段分别是输入、输出、总token数量，输入不只是用户问题，还包含Prompt和证据
    }
    logger.info(json.dumps({
        "event": "generation_usage", "request_id": request_id,
        "model": generated["model"], "usage": counts,
    }, ensure_ascii=False))