from fastapi import FastAPI
from material_indexer import build_config, scan_folder, validate_root

app = FastAPI(title="资料索引 API") #创建一个 API 应用


@app.get("/health") #表示把下面这个函数注册成一个 GET 接口，路径是 /health
def health():
    return {"status": "ok"} #FastAPI 会自动把 Python 字典转换成 JSON


@app.get("/summary")
def summary():
    return {
        "project": "material_indexer",
        "description": "scan local files and generate metadata",
    }


@app.get("/files")
def list_files(limit: int = 10): #URL 里可以传参数，/files?limit=3，如果不传，默认返回 10 个
    config = build_config() #复用资料索引器里的配置
    validate_root(config.root) #检查扫描目录是否存在

    report = scan_folder(config.root) #重新扫描资料目录，拿到结构化报告，FastAPI 会自动把 Python 的 list/dict 转成 JSON
    sorted_files = sorted(report["files"], key=lambda x: x["size"], reverse=True)

    return sorted_files[:limit] #只返回前 limit 个文件