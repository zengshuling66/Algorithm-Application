from fastapi import FastAPI, Query
from material_indexer import build_config, scan_folder, validate_root

app = FastAPI(title="资料索引 API") #创建一个 API 应用

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


@app.get("/files") #普通文件列表
def list_files(limit: int = Query(default=10, ge=1, le=100)):
    #URL里可以传query参数并进行类型转换，用来控制接口层返回结果，/files?limit=3，如果不传，默认返回 10 个
    #Query参数校验：ge=1：greater than or equal，大于等于 1；le=100：less than or equal，小于等于 100。避免一次返回过多文件，也避免负数这种无意义请求
    config = build_config() #复用资料索引器里的配置
    validate_root(config.root) #检查扫描目录是否存在

    report = scan_folder(config.root) #重新扫描资料目录，拿到结构化报告，FastAPI 会自动把 Python 的 list/dict 转成 JSON

    return {
        "limit": limit,
        "files": report["files"][:limit], #不排序，返回扫描到的前limit个文件
    }


@app.get("/files/by-suffix") #按后缀筛选接口
def files_by_suffix(suffix: str, limit: int = Query(default=10, ge=1, le=100)):
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

    return {
        "suffix": suffix,
        "limit": limit,
        "total": len(matched_files),
        "files": matched_files[:limit],
    }
# http://127.0.0.1:8000/files/by-suffix?suffix=PDF
# http://127.0.0.1:8000/files/by-suffix?suffix=.pdf
# http://127.0.0.1:8000/files/by-suffix?suffix=pdf&limit=5


@app.get("/largest-files") #最大文件列表
def largest_files(limit: int = Query(default=10, ge=1, le=100)): 
    config = build_config()
    validate_root(config.root)

    report = scan_folder(config.root)
    sorted_files = sorted(report["files"], key=lambda x: x["size"], reverse=True)

    return {
        "limit": limit,
        "files": sorted_files[:limit], #排序，返回最大的前limit个文件
    }


@app.get("/extensions")
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


#http://127.0.0.1:8000/docs FastAPI 会自动生成接口文档