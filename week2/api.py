import logging
import asyncio
import json
import sqlite3
from typing import Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
# Pydantic 在 FastAPI 里用于定义数据模型，约束接口输入输出，并自动生成接口文档。
# BaseModel：Pydantic 所有数据模型的基类，你自己定义的模型类都继承它。
# Field：给字段增加说明、示例、约束，用于文档和校验。
from material_indexer import build_config, scan_folder, validate_root
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from database import (
    init_database,
    list_scan_history,
    save_scan_history,
)
from cache import TTLCache

# 路由函数负责 HTTP 请求处理
# 业务函数负责业务逻辑
# 底层函数负责文件、数据库或模型调用

@asynccontextmanager
async def lifespan(_: FastAPI): #lifespan中文可以理解成“应用生命周期”
    init_database() # yield 前：服务启动时执行
    yield           # 服务运行期间；我们只需要启动时初始化数据库，因此 yield 后暂时没有代码。
                    # yield 后：服务关闭时执行

#创建一个 API 应用
app = FastAPI(
    title="资料索引 API",
    version="0.1.0",
    description="扫描训练营资料目录，并提供文件查询、后缀筛选和大小过滤接口",
    lifespan=lifespan
)

logger = logging.getLogger(__name__)

SCAN_REPORT_CACHE_KEY = "scan_report"
scan_report_cache = TTLCache(default_ttl=30.0)

class FileInfo(BaseModel): #定义一个叫 FileInfo 的数据模型，它继承 BaseModel，所以 Pydantic 能识别它、校验它、把它展示到 /docs 里
    path: str = Field(description="文件完整路径")
    name: str = Field(description="文件名")
    suffix: str = Field(description="文件后缀")
    size: int = Field(description="文件大小，单位为字节")
    parent_dir: str = Field(description="文件所在目录名")

class FileListResponse(BaseModel):
    limit: int = Field(description="本次返回的文件数量上限")
    min_size: int = Field(description="最小文件大小，单位为字节")
    total: int = Field(description="符合条件的文件总数")
    files: list[FileInfo] = Field(description="文件列表") #files 是一个列表，列表里的每个元素都应该符合 FileInfo 结构

class FilesBySuffixResponse(BaseModel):
    suffix: str = Field(description="归一化后的文件后缀")
    limit: int = Field(description="本次返回的文件数量上限")
    min_size: int = Field(description="最小文件大小，单位为字节")
    total: int = Field(description="匹配该后缀且符合大小条件的文件总数")
    files: list[FileInfo] = Field(description="匹配该后缀的文件列表")

class ExtensionStatsResponse(BaseModel):
    root: str = Field(description="扫描根目录")
    file_count: int = Field(description="文件总数")
    folder_count: int = Field(description="文件夹总数")
    extension_count: dict[str, int] = Field(description="不同文件后缀的数量统计")

#→ 创建一条扫描记录后的响应
class ScanCreatedResponse(BaseModel):
    id: int = Field(description="本次扫描记录 ID")
    root: str = Field(description="扫描根目录")
    file_count: int = Field(description="扫描到的文件数量")
    folder_count: int = Field(description="扫描到的文件夹数量")
    error_count: int = Field(description="扫描过程中跳过的异常数量")

#→ 一条完整的历史记录
class ScanHistoryItem(BaseModel):
    id: int = Field(description="扫描记录 ID")
    root: str = Field(description="扫描根目录")
    file_count: int = Field(description="文件数量")
    folder_count: int = Field(description="文件夹数量")
    error_count: int = Field(description="异常数量")
    created_at: str = Field(description="扫描记录创建时间")

#→ 多条历史记录组成的列表响应
class ScanHistoryListResponse(BaseModel):
    limit: int = Field(description="最多返回多少条记录")
    count: int = Field(description="本次实际返回的记录数")
    scans: list[ScanHistoryItem] = Field(description="扫描历史列表")

class CacheStatsResponse(BaseModel):
    hits: int = Field(description="缓存命中次数")
    misses: int = Field(description="缓存未命中次数")
    size: int = Field(description="当前缓存条目数量")
    hit_rate: float = Field(description="缓存命中率，范围为 0 到 1")


def filter_files_by_min_size(files: list[dict], min_size: int) -> list[dict]:
    filtered_files = []

    for file_info in files:
        if file_info["size"] >= min_size:
            filtered_files.append(file_info)

    return filtered_files
    
def load_scan_report(
    force_refresh: bool = False, #默认允许使用缓存
) -> dict[str, Any]:
    """优先读取缓存，未命中时扫描目录，并把底层文件系统异常转换成 HTTP 异常。"""

    if not force_refresh:
        cached_report = scan_report_cache.get(
            SCAN_REPORT_CACHE_KEY
        )

        if cached_report is not None:
            logger.info("扫描报告缓存命中")
            return cached_report #缓存命中时直接 return，后面的目录扫描不会执行

        logger.info("扫描报告缓存未命中")
    else:
        logger.info("强制刷新扫描报告")

    config = build_config() #复用资料索引器里的配置

    try:
        validate_root(config.root) #检查扫描目录是否存在
        report = scan_folder(config.root) #重新扫描资料目录，拿到结构化报告，FastAPI 会自动把 Python 的 list/dict 转成 JSON

        #扫描成功后使用 set()，否则下一次请求仍然会重复扫描
        scan_report_cache.set(
            SCAN_REPORT_CACHE_KEY,
            report,
        )

        return report

    #扫描异常时不会写入缓存，原来的异常处理仍然有效
    except (FileNotFoundError, NotADirectoryError) as error:
        logger.exception("扫描根目录配置错误") #它只能在 except 中使用，会自动记录当前异常的 traceback

        #表示发生异常，立即中断当前执行流程
        raise HTTPException(
            status_code=500, #服务端代码或配置错误
            detail={
                "code": "SCAN_ROOT_INVALID",
                "message": "服务端扫描目录配置不可用",
            },
        ) from error #表示新的 HTTP 异常是由原来的文件系统异常导致的，这样 traceback 会保留完整因果链，debug 时能看到真正根因

    except OSError as error:
        logger.exception("扫描资料目录失败")

        raise HTTPException(
            status_code=503, #依赖暂时不可用
            detail={
                "code": "SCAN_SERVICE_UNAVAILABLE",
                "message": "资料目录暂时无法访问",
            },
        ) from error

# 状态码 含义	                当前项目场景
# 200	请求成功	           查询完成，结果可以为空
# 201   资源创建成功            创建扫描历史
# 400	请求业务含义不合理	    传入不支持的参数组合     #参数合法，但业务逻辑发现不合理，主动返回400
# 404	客户端请求的资源不存在	请求错误路径或不存在的记录
# 422	参数格式或校验不通过	limit=0、缺少 suffix    #请求还没进入路由函数，就被 FastAPI/Pydantic的参数校验拦住了
# 500	服务端代码或配置错误	配置的扫描根目录不存在/response_model 缺少字段，服务端返回结构错误
# 503	外部依赖暂时不可用	    文件系统、数据库或模型服务故障

#生成器 Generator
#含有 yield 的函数叫生成器函数，调用它时，函数体不会立刻执行，而是返回生成器对象，然后一次next只产生一个值
#生成器是特殊的迭代器。所有生成器都是迭代器，但迭代器不一定由生成器函数创建。
def generate_file_events(files: list[dict], limit: int) -> Iterator[dict[str, Any]]: #返回一个能够逐个产生字典的迭代器Iterator
    """逐个产生文件事件，供后续流式接口消费。"""

    total = min(len(files), limit)

    for index, file_info in enumerate(files[:total], start=1): #取前 total 个文件
        #每次产生一个事件后暂停。第二次调用 next() 时，从暂停位置继续
        yield {
            "event": "file",
            "index": index,
            "total": total,
            "name": file_info["name"],
            "suffix": file_info["suffix"],
        }

    #最后循环结束后继续执行：
    yield {
        "event": "done",
        "total": total,
    }
    #再下一次调用 next()，函数真正结束并触发 StopIteration

async def generate_file_sse(
    files: list[dict],
    limit: int,
    delay: float,
    request: Request,
) -> AsyncIterator[str]:
    """把结构化文件事件转换成 SSE 文本事件。"""
    #SSE全称 Server-Sent Events，中文通常叫“服务器发送事件”，它允许一次 HTTP 请求建立长连接后，服务器不断向客户端发送文本事件。

    for event in generate_file_events(files, limit): #消费前面写好的同步生成器，每次拿到一个字典。
        if await request.is_disconnected(): #检查客户端是否已经关闭页面或取消请求
            logger.info("客户端已断开文件流")
            break

        payload = json.dumps(event, ensure_ascii=False) #把 Python 字典转换为 JSON 字符串，ensure_ascii=False 可以让中文保持中文

        yield (
            f"event: {event['event']}\n" #event:：事件类型，这里是file/done；第一个 \n：结束当前字段
            f"data: {payload}\n\n" #data:：事件数据；第二个 \n：结束整个SSE事件
        )
        #产生一个完整 SSE 事件，StreamingResponse 获取到它后就能立即发送

        if delay > 0 and event["event"] != "done":
            await asyncio.sleep(delay) #模拟大模型生成下一段内容所需的时间；done 后不再等待，因为已经没有下一条事件。
            #time.sleep()阻塞线程，await asyncio.sleep() 会把控制权交回事件循环
            #SSE等待下一条事件时使用 await asyncio.sleep()，其他请求仍然可以被 FastAPI 处理

# @app.get("/xxx")是路由装饰器，它把普通 Python函数注册为指定路径的 GET接口。
# 请求到达时，FastAPI 根据路由表调用函数，并负责参数解析、校验和响应序列化。
@app.get("/health") #表示把下面这个函数注册成一个 GET 接口，路径是 /health
def health():
    return {"status": "ok"} #FastAPI 会自动把 Python 字典转换成 JSON


@app.get("/summary")
def summary():
    return {
        "project": "material_indexer",
        "description": "scan local files and generate metadata",
    }


@app.get("/files", response_model=FileListResponse) #普通文件列表，这个接口最终返回的数据，应该符合 FileListResponse，同时可以按模型过滤输出
def list_files(limit: int = Query(default=10, ge=1, le=100, description="最多返回多少个文件"), min_size: int = Query(default=0, ge=0, description="只返回大于等于该字节数的文件"),):
    #URL里可以传query参数并进行类型转换，用来控制接口层返回结果，/files?limit=3，如果不传，默认返回 10 个
    #Query 用来声明和校验 URL 查询参数，它可以设置默认值、必填参数、最大值、最小值、字符串长度、描述信息等
    #Query参数校验：ge=1：greater than or equal，大于等于 1；le=100：less than or equal，小于等于 100。避免一次返回过多文件，也避免负数这种无意义请求
    #response_model约束接口输出结构，只返回模型里定义的字段。return中多余的字段不会返回给前端，FastAPI 会根据 response_model 过滤掉多余字段；return中如果缺少了模型里定义的字段，FastAPI 会报错500 Internal Server Error
    report = load_scan_report()
    
    filtered_files = filter_files_by_min_size(report["files"], min_size)

    return {
        "limit": limit, #按 limit 截断返回
        "min_size": min_size, #按 min_size 过滤
        "total": len(filtered_files), #统计过滤后的总数 total
        "files": filtered_files[:limit], #不排序，返回符合字节数要求的前limit个文件
    }


@app.get("/files/by-suffix", response_model=FilesBySuffixResponse) #按后缀筛选接口
def files_by_suffix(
    suffix: str = Query(..., min_length=1, description="文件后缀，例如 pdf 或 .pdf"),
    #Query(...) 里的 ... 很重要，它表示这个参数是必填的
    limit: int = Query(default=10, ge=1, le=100, description="最多返回多少个文件"),
    min_size: int = Query(default=0, ge=0, description="只返回大于等于该字节数的文件"),):
    
    report = load_scan_report()

    suffix = suffix.strip().lower()
    
    if not suffix.startswith(".") and suffix != "no_suffix":
        suffix = "." + suffix #输入归一化：用户输入的后缀如果没有点，自动加上点；如果用户输入 no_suffix，表示没有后缀的文件，不需要加点

    matched_files = []

    for file_info in report["files"]:
        if file_info["suffix"] == suffix:
            matched_files.append(file_info)

    filtered_files = filter_files_by_min_size(matched_files, min_size)

    return {
        "suffix": suffix,
        "limit": limit,
        "min_size": min_size,
        "total": len(filtered_files),
        "files": filtered_files[:limit],
    }
# http://127.0.0.1:8000/files/by-suffix?suffix=PDF
# http://127.0.0.1:8000/files/by-suffix?suffix=.pdf
# http://127.0.0.1:8000/files/by-suffix?suffix=pdf&limit=5
# http://127.0.0.1:8000/files?limit=5&min_size=1048576
# http://127.0.0.1:8000/largest-files?limit=5&min_size=104857600
# http://127.0.0.1:8000/files/by-suffix?suffix=pdf&limit=5&min_size=1048576


@app.get("/largest-files", response_model=FileListResponse) #最大文件列表
def largest_files(limit: int = Query(default=10, ge=1, le=100, description="最多返回多少个文件"), min_size: int = Query(default=0, ge=0, description="只返回大于等于该字节数的文件"),): 
    report = load_scan_report()

    filtered_files = filter_files_by_min_size(report["files"], min_size)
    sorted_files = sorted(filtered_files, key=lambda x: x["size"], reverse=True)
    #先过滤，再排序，再截断
    return {
        "limit": limit,
        "min_size": min_size,
        "total": len(filtered_files),
        "files": sorted_files[:limit], #排序，返回最大的前limit个文件
    }


@app.get("/extensions", response_model=ExtensionStatsResponse)
def extension_stats():
    report = load_scan_report()

    return {
        "root": report["root"],
        "file_count": report["file_count"],
        "folder_count": report["folder_count"],
        "extension_count": report["extension_count"],
    }

#添加流式路由
@app.get("/stream/files") #SSE返回的是多个连续文本片段，不是一个完整 JSON 对象，因此不能直接使用普通 Pydantic response_model
async def stream_files(
    request: Request,
    limit: int = Query(
        default=5,
        ge=1,
        le=100,
        description="最多流式返回多少个文件",
    ),
    delay: float = Query(
        default=0.2,
        ge=0,
        le=2,
        description="相邻事件之间的模拟延迟，单位为秒",
    ),
):
    report = load_scan_report()

    event_stream = generate_file_sse(
        files=report["files"],
        limit=limit,
        delay=delay,
        request=request,
    )

    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
#load_scan_report()仍会先完成整个目录扫描，然后才开始流式返回文件事件。
#所以它目前是“扫描结果流”，不是严格意义上的“实时扫描进度”。真正的扫描进度需要把 scan_folder()的遍历过程本身改造成生成器。

#不同于GET用于读取资源，正常情况下不改变服务端状态；POST用于创建资源，改变服务端状态。
@app.post(
    "/scans",
    response_model=ScanCreatedResponse,
    status_code=201, #201 Created：成功创建了新资源
)
def create_scan():
    report = load_scan_report(force_refresh=True) #POST/scans：明确要求创建一次新的扫描记录，跳过旧缓存
    #强制刷新完成后，新报告仍会写入缓存，因此后续 GET 可以直接复用最新结果

    try:
        scan_id = save_scan_history(report) #开启 SQLite 事务，参数化 SQL 写入 scan_history

    except sqlite3.Error as error:
        logger.exception("保存扫描历史失败")

        raise HTTPException(
            status_code=503,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "扫描完成，但扫描历史暂时无法保存",
            },
        ) from error

    return {
        "id": scan_id,
        "root": report["root"],
        "file_count": report["file_count"],
        "folder_count": report["folder_count"],
        "error_count": report["error_count"],
    }

@app.get( #这个接口不创建新数据，只读取已有记录，所以使用 GET
    "/scans",
    response_model=ScanHistoryListResponse,
)
def get_scan_history(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="最多返回多少条扫描历史",
    ),
):
    try:
        scans = list_scan_history(limit=limit)

    except sqlite3.Error as error:
        logger.exception("查询扫描历史失败")

        raise HTTPException(
            status_code=503,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "扫描历史暂时无法查询",
            },
        ) from error

    return {
        "limit": limit,
        "count": len(scans), #表示本次实际返回多少条，不是数据库历史总数
        "scans": scans,
    }

@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
)
def get_cache_stats():
    stats = scan_report_cache.stats()

    total_requests = stats["hits"] + stats["misses"]

    if total_requests == 0:
        hit_rate = 0.0
    else:
        hit_rate = stats["hits"] / total_requests

    return {
        "hits": stats["hits"],
        "misses": stats["misses"],
        "size": stats["size"],
        "hit_rate": round(hit_rate, 4),
    }


# GET 请求完整数据流
# 访问：GET /files?limit=5&min_size=1024
# 代码依次经历：
# 1. Uvicorn 接收 HTTP 请求
# 2. FastAPI 根据 @app.get("/files") 找到 list_files()
# 3. Query 校验 limit 和 min_size                       请求校验：发生在进入路由函数之前
# 4. list_files() 调用 load_scan_report()
# 5. load_scan_report() 查询 TTL 缓存
# 6. 缓存命中则直接返回，未命中才扫描目录
# 7. filter_files_by_min_size() 过滤小文件
# 8. 按 limit 截断结果
# 9. FileListResponse 校验并过滤响应字段                 响应校验：发生在路由函数 return 之后
# 10. FastAPI 把 Python dict/list 序列化为 JSON

# POST 请求完整数据流
# 访问：POST /scans
# 依次发生：
# 1. FastAPI 调用 create_scan()
# 2. load_scan_report(force_refresh=True) 跳过旧缓存
# 3. 重新扫描文件夹
# 4. 使用最新扫描报告覆盖缓存
# 5. save_scan_history(report) 开启 SQLite 事务
# 6. 参数化 SQL 写入 scan_history
# 7. 成功后提交事务，返回新记录 ID
# 8. ScanCreatedResponse 校验返回结果
# 9. FastAPI 返回 201 Created

# SSE 数据流
# 访问：GET /stream/files?limit=3&delay=0.2
# 流程是：
# load_scan_report()
# → generate_file_events()
# → generate_file_sse()
# → StreamingResponse
# → 客户端逐条收到 file 和 done 事件
# generate_file_events() 是同步生成器，逐个产生 Python 字典。
# generate_file_sse() 是异步生成器，把字典转换为 SSE 文本，并使用：await asyncio.sleep(delay)
# 等待期间会把执行权交还事件循环，因此其他请求仍然可以被处理。



#week2中启动后端：python -m uvicorn api:app --reload
# http://127.0.0.1:8000/docs FastAPI 会自动生成接口文档