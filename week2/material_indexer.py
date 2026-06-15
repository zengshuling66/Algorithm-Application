from pathlib import Path

def scan_folder(root):
    file_count = 0
    folder_count = 0
    extension_count = {}
    files = []

    for path in root.rglob("*"):
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

    return {
        "root": str(root),
        "file_count": file_count,
        "folder_count": folder_count,
        "extension_count": extension_count,
        "files": files,
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

    return "\n".join(lines)


def save_report(content, output_path):
    output_path.write_text(content, encoding="utf-8") #encoding="utf-8" 是为了支持中文


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    output_dir.mkdir(exist_ok=True)

    root = Path(r"C:\Users\Administrator\Desktop\OneDrive - 南方科技大学\丁师兄训练营")
    output_path = output_dir / "资料索引报告.md"

    if not root.exists():
        print("路径不存在：", root)
        return

    if not root.is_dir():
        print("路径不是文件夹：", root)
        return

    report = scan_folder(root)
    markdown_content = generate_markdown_report(report, top_n=10)
    save_report(markdown_content, output_path)

    print("报告已生成：", output_path)


if __name__ == "__main__":
    main()