from scan_folder import scan_folder
from pathlib import Path

def generate_markdown_report(report, top_n=10):
    lines = [] #lines 是一个 list，用来一行一行保存 Markdown 内容

    lines.append("# 训练营资料扫描报告") #append 是往列表里加一行
    lines.append("") #空行
    lines.append(f"扫描目录：{report['root']}") #f"...{...}..." 是 Python 的 f-string 语法，可以在字符串里直接插入变量的值
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 文件数量：{report['file_count']}")
    lines.append(f"- 文件夹数量：{report['folder_count']}")
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
    lines.append("| 文件名 | 大小 (MB) |")
    lines.append("| --- | ---: |")

    sorted_files = sorted(report["files"], key=lambda x: x["size"], reverse=True)

    for file_info in sorted_files[:top_n]:
        size_mb = file_info["size"] / (1024 * 1024)
        file_name = file_info["name"].replace("|", "\\|") #如果文件名里有 |，会破坏 Markdown 表格格式，所以要转义一下
        lines.append(f"| {file_name} | {round(size_mb, 2)} |") #把 size_mb 保留 2 位小数

    return "\n".join(lines) #把列表里的每一项用换行符 \n 拼起来，变成完整 Markdown 文档

def save_report(content, output_path):
    output_path.write_text(content, encoding="utf-8") #encoding="utf-8" 是为了支持中文

def main():
    root = Path(r"C:\Users\Administrator\Desktop\OneDrive - 南方科技大学\丁师兄训练营") #r是raw string，主要是为了避免 Windows 路径里的反斜杠 \ 被当成转义字符
    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / "训练营资料扫描报告.md"

    if not root.exists():
        print("路径不存在：", root)
        return

    if not root.is_dir():
        print("路径不是一个文件夹：", root)
        return

    report = scan_folder(root)
    markdown_content = generate_markdown_report(report, top_n=15)
    save_report(markdown_content, output_path)

    print("报告已生成：", output_path)


if __name__ == "__main__":
    main()