import json
from pathlib import Path

def validate_root(root): #raise：主动抛出异常
    if not root.exists():
        raise FileNotFoundError(f"路径不存在：{root}")

    if not root.is_dir():
        raise NotADirectoryError(f"路径不是文件夹：{root}")
    

def scan_folder(root):
    file_count = 0
    folder_count = 0
    extension_count = {}
    files = []
    errors = []

    for path in root.rglob("*"):
        try:
            if path.is_file():
                file_count += 1

                suffix = path.suffix.lower()
                if suffix == "":
                    suffix = "no_suffix"

                if suffix not in extension_count:
                    extension_count[suffix] = 0
                extension_count[suffix] += 1

                file_info = {
                    "path": str(path),
                    "name": path.name, #文件名
                    "suffix": suffix,
                    "size": path.stat().st_size,
                    "parent_dir": path.parent.name,
                }
                files.append(file_info)

            elif path.is_dir():
                folder_count += 1

        except OSError as error: #某个文件访问失败时，不让整个扫描任务失败，而是记录这个坏文件，继续处理后面的文件
            errors.append({
                "path": str(path),
                "error_type": type(error).__name__,
                "message": str(error),
            })

    return {
        "root": str(root),
        "file_count": file_count,
        "folder_count": folder_count,
        "extension_count": extension_count,
        "files": files,
        "errors": errors,
        "error_count": len(errors),
        }


def generate_markdown_report(report, top_n=10):
    lines = []

    lines.append("# 资料索引报告")
    lines.append("")
    lines.append(f"扫描目录：{report['root']}")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 文件数量：{report['file_count']}")
    lines.append(f"- 文件夹数量：{report['folder_count']}")
    lines.append(f"- 跳过文件数量：{report['error_count']}")
    lines.append("")

    lines.append("## 文件类型统计")
    lines.append("")
    lines.append("| 文件后缀 | 数量 |")
    lines.append("| --- | ---: |")

    for suffix, count in report["extension_count"].items():
        lines.append(f"| {suffix} | {count} |")

    lines.append("")
    lines.append(f"## 最大的{top_n}个文件")
    lines.append("")
    lines.append("| 文件名 | 所在目录 | 大小 (MB) |")
    lines.append("| --- | --- | ---: |")

    sorted_files = sorted(report["files"], key=lambda x: x["size"], reverse=True)

    for file_info in sorted_files[:top_n]:
        size_mb = file_info["size"] / (1024 * 1024)
        file_name = file_info["name"].replace("|", "\\|")
        parent_dir = file_info["parent_dir"].replace("|", "\\|")
        lines.append(f"| {file_name} | {parent_dir} | {round(size_mb, 2)} |")

    lines.append("")
    lines.append("## 扫描异常")
    lines.append("")

    if report["error_count"] == 0:
        lines.append("本次扫描没有遇到文件访问异常。")
    else:
        lines.append("| 路径 | 错误类型 | 错误信息 |")
        lines.append("| --- | --- | --- |")

        for error in report["errors"][:10]: #只展示前 10 个错误
            error_path = error["path"].replace("|", "\\|")
            error_type = error["error_type"].replace("|", "\\|")
            message = error["message"].replace("|", "\\|")
            lines.append(f"| {error_path} | {error_type} | {message} |")

    return "\n".join(lines)


def save_report(content, output_path):
    output_path.write_text(content, encoding="utf-8") #encoding="utf-8" 是为了支持中文


def save_json_report(report, output_path):
    json_content = json.dumps(report, ensure_ascii=False, indent=2) #把 Python 字典 report 转成 JSON 字符串
    #ensure_ascii=False表示中文不要转义；indent=2表示格式化输出，缩进2个空格
    output_path.write_text(json_content, encoding="utf-8")

#把 Python 对象直接写入文件
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(report, f, ensure_ascii=False, indent=2)

# 函数	         输入	           输出	            记忆方式
# json.dumps()	Python对象	      JSON 字符串	    s表示 string
# json.dump()	Python对象+文件	  写入 JSON 文件	直接 dump 到文件
# json.loads()	JSON字符串	      Python 对象	   从 string 读回来
# json.load()	JSON文件	      Python 对象	   从文件读回来
# 带 s：处理字符串
# 不带 s：处理文件


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    output_dir.mkdir(exist_ok=True)

    root = Path(r"C:\Users\Administrator\Desktop\OneDrive - 南方科技大学\丁师兄训练营")
    output_path = output_dir / "资料索引报告.md"
    json_output_path = output_dir / "资料索引数据.json"

    try:
        validate_root(root)

        report = scan_folder(root)
        markdown_content = generate_markdown_report(report, top_n=10)
        save_report(markdown_content, output_path)
        save_json_report(report, json_output_path)

        print("报告已生成：", output_path)
        print("JSON数据已生成：", json_output_path)

    except FileNotFoundError as error:
        print("输入路径错误：", error)

    except NotADirectoryError as error:
        print("输入路径错误：", error)

    except OSError as error:
        print("文件系统错误：", error)

if __name__ == "__main__":
    main()