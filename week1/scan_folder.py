from pathlib import Path

def scan_folder(root):
    file_count = 0
    folder_count = 0
    extension_count = {} #字典
    files = [] #列表，本身没有“行/列”的概念，它只是按顺序保存一堆元素

    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1

            suffix = path.suffix.lower() #转成小写，避免同一类型的文件因为大小写不同而被统计成不同的类型
            if suffix == "":
                suffix = "no_suffix"
            
            if suffix not in extension_count:
                extension_count[suffix] = 0

            extension_count[suffix] += 1

            file_info = {
                "path": str(path),
                "name": path.name,
                "suffix": suffix,
                "size": path.stat().st_size,
            }
            files.append(file_info) #用 dict 表达一个对象，用 list 保存多个对象

        elif path.is_dir():
            folder_count += 1

    report = {
            "root": str(root),
            "file_count": file_count,
            "folder_count": folder_count,
            "extension_count": extension_count,
            "files": files,
        }

    return report

def print_extension_count(extension_count):
    print("\n文件类型统计：")
    for suffix, count in extension_count.items():
        print(suffix, ":", count)

def print_largest_files(files, top_n):
    sorted_files = sorted(files, key=lambda x: x["size"], reverse=True) #按每个文件的 size 从大到小排序
    print(f"\n最大的{top_n}个文件：")

    for file_info in sorted_files[:top_n]:
        size_mb = file_info["size"] / (1024 * 1024)
        print(file_info["name"], round(size_mb,2), "MB")

def main():
    root = Path(r"C:\Users\Administrator\Desktop\OneDrive - 南方科技大学\丁师兄训练营")

    if not root.exists():
        print("路径不存在：", root)
        return
    
    if not root.is_dir():
        print("路径不是一个文件夹：", root)
        return
    
    report = scan_folder(root)

    print("扫描目录：", report["root"])
    print("文件数量：", report["file_count"])
    print("文件夹数量：", report["folder_count"])
    
    print_extension_count(report["extension_count"])
    print_largest_files(report["files"], top_n=10)

if __name__ == "__main__":
    main()