import json
import logging #logging 可以区分 INFO、WARNING、ERROR，支持日志级别和统一格式
import argparse #支持命令行参数
from pathlib import Path
from typing import Any #typing 让函数输入输出更清楚，便于维护、协作和排错

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成训练营资料索引报告") #创建一个命令行参数解析器
    parser.add_argument(
        "--top-n", #声明程序支持一个参数，名字叫 --top-n
        type=int, #表示这个参数必须转成整数
        default=10, #如果用户不传，默认是 10
        help="Markdown 报告中展示最大的前 N 个文件",
    )
    return parser.parse_args() #真正读取命令行输入


logging.basicConfig(
    level=logging.INFO, #表示 INFO 及以上级别的日志都会显示
    format="%(levelname)s | %(message)s", #表示日志格式
)

logger = logging.getLogger(__name__) #创建当前文件专用的 logger


class IndexerConfig: #配置参数，定义一个类，也就是“对象的模板”
    def __init__(self, root: Path, output_dir: Path, top_n: int = 10): #__init__ 是初始化方法，创建对象时会自动执行
        self.root = root
        self.output_dir = output_dir
        self.top_n = top_n
        self.markdown_output_path = output_dir / "资料索引报告.md"
        self.json_output_path = output_dir / "资料索引数据.json"


def validate_root(root: Path) -> None: #输入 root 是 Path 类型，函数没有返回值
    if not root.exists():
        raise FileNotFoundError(f"路径不存在：{root}") #raise：主动抛出异常

    if not root.is_dir():
        raise NotADirectoryError(f"路径不是文件夹：{root}")
    

def scan_folder(root: Path) -> dict[str, Any]: #输入是 Path，返回是一个字典，Any 表示里面的值类型比较复杂，可能是字符串、数字、列表、字典等
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
            logger.warning("跳过无法访问的路径：%s", path)
            
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


def generate_markdown_report(report: dict[str, Any], top_n: int = 10) -> str: #输入 report 是字典，top_n 是整数，返回 Markdown 字符串
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


def save_report(content: str, output_path: Path) -> None: #输入是字符串内容和路径，负责写文件，不返回东西
    output_path.write_text(content, encoding="utf-8") #encoding="utf-8" 是为了支持中文


def save_json_report(report: dict[str, Any], output_path: Path) -> None:
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


def build_config(top_n: int = 10) -> IndexerConfig: #创建配置对象
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    root = Path(r"C:\Users\Administrator\Desktop\OneDrive - 南方科技大学\丁师兄训练营")

    return IndexerConfig(root=root, output_dir=output_dir, top_n=top_n)


def main() -> None:
    args = parse_args()
    config = build_config(top_n=args.top_n)
    config.output_dir.mkdir(exist_ok=True)

    try:
        validate_root(config.root)

        report = scan_folder(config.root)
        markdown_content = generate_markdown_report(report, top_n=config.top_n)
        save_report(markdown_content, config.markdown_output_path)
        save_json_report(report, config.json_output_path)

        logger.info("报告已生成：%s", config.markdown_output_path)
        logger.info("JSON数据已生成：%s", config.json_output_path)

    except FileNotFoundError as error:
        logger.error("输入路径错误： %s", error)

    except NotADirectoryError as error:
        logger.error("输入路径错误： %s", error)

    except OSError as error:
        logger.error("文件系统错误： %s", error)

if __name__ == "__main__":
    main()