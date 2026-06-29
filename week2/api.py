from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
# Pydantic 在 FastAPI 里用于定义数据模型，约束接口输入输出，并自动生成接口文档。
# BaseModel：Pydantic 所有数据模型的基类，你自己定义的模型类都继承它。
# Field：给字段增加说明、示例、约束，用于文档和校验。
from material_indexer import build_config, scan_folder, validate_root

#创建一个 API 应用
app = FastAPI(
    title="资料索引 API",
    version="0.1.0",
    description="扫描训练营资料目录，并提供文件查询、后缀筛选和大小过滤接口"
)

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

def filter_files_by_min_size(files: list[dict], min_size: int) -> list[dict]:
    filtered_files = []

    for file_info in files:
        if file_info["size"] >= min_size:
            filtered_files.append(file_info)

    return filtered_files

# @app.get("/xxx")：把 Python 函数注册成 HTTP GET 接口
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
    config = build_config() #复用资料索引器里的配置
    validate_root(config.root) #检查扫描目录是否存在

    report = scan_folder(config.root) #重新扫描资料目录，拿到结构化报告，FastAPI 会自动把 Python 的 list/dict 转成 JSON
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
    config = build_config()
    validate_root(config.root)

    report = scan_folder(config.root)

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
    config = build_config()
    validate_root(config.root)

    report = scan_folder(config.root)
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
    config = build_config()
    validate_root(config.root)

    report = scan_folder(config.root)

    return {
        "root": report["root"],
        "file_count": report["file_count"],
        "folder_count": report["folder_count"],
        "extension_count": report["extension_count"],
    }

#week2中启动后端：python -m uvicorn api:app --reload
# http://127.0.0.1:8000/docs FastAPI 会自动生成接口文档