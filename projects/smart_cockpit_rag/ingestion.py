import sys
import json
import pdfplumber
from pathlib import Path
# from pypdf import PdfReader
from table_parser import parse_page
from hashlib import sha256 #导入的是 SHA-256 计算器的创建入口


#内容哈希：这份文件的字节内容现在是什么版本？字节变化，哈希通常随之变化。
#source_id 用于跟踪“同一份资料”；哈希用于判断“内容是否变化”。
def hash_file(file_path: Path) -> str:
    digest = sha256() #先创建一个尚未喂入文件数据的计算对象，放在 digest 里

    with file_path.open("rb") as source_file:
        while True: #while True表示循环本身没有预先规定次数；退出条件写在循环内部。
            block = source_file.read(1024 * 1024) #每轮 read(1024 * 1024) 最多读取 1,048,576 字节，也就是 1 MiB
            #最后一次读到文件末尾后，再读会得到空的 bytes，即 b""；if not block: break 正是利用这个“读到末尾得到空字节串”的结果退出

            if not block:
                break

            digest.update(block) #把本轮字节接到同一次哈希计算中

    return digest.hexdigest() #把内部计算结果转成便于打印和保存的 64 字符字符串


# def load_pdf_pages(pdf_path: Path, source_id: str):
#     content_hash = hash_file(pdf_path) #文件先被哈希函数按字节读一遍，再由 PdfReader 读一遍
#     records = []
#     empty_pages = []

#     #open("rb") 中的 r 是只读，b 是按二进制读取 PDF；with 保证使用完毕后或中途出错时自动关闭文件
#     with pdf_path.open("rb") as pdf_file:
#         reader = PdfReader(pdf_file) #创建读取器
#         total_pages = len(reader.pages) #reader.pages 提供页面

#         for page_number, page in enumerate(reader.pages, start=1):
#             text = (page.extract_text() or "").strip() #page.extract_text() 可能拿不到文字，or "" 把这种结果变成空字符串，才能安全调用 .strip()
#             #.strip() 仅去掉两端空白，不会删除正文

#             if not text:
#                 empty_pages.append(page_number) #若结果为空，empty_pages.append(...) 记下页码，continue 直接进入下一页
#                 continue

#             records.append({
#                 "text": text,
#                 "source": pdf_path.name,
#                 "source_id": source_id,
#                 "content_hash": content_hash,
#                 "page": page_number, #这里的 page 指 PDF 物理页序号，不一定等于纸面印刷的页码
#             })

#     return records, empty_pages, total_pages, content_hash
#     #返回要带content_hash是因为：即使某份 PDF 一页文字都没提取到，调用者仍能知道它的文件指纹；不能依赖 records[0] 去取，否则空列表会报错


def load_pdf_pages(pdf_path: Path, source_id: str):
    content_hash = hash_file(pdf_path)
    metadata_path = Path(__file__).resolve().with_name("manual_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) #read_text() 读成字符串，json.loads() 转成字典。
    if (metadata["source_id"], metadata["content_hash"]) != (source_id, content_hash):
        raise ValueError("Metadata does not match this PDF version")
    top, bottom = metadata["header_y"] #header_y 是本手册页眉的上下边界，单位是 PDF 点，不是字符数。
    records, empty_pages = [], []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            body, tables = parse_page(page)
            header = page.crop((0, top, page.width, bottom)).extract_text() or "" #crop() 接收“左、上、右、下”，这里取整页宽度内的页眉带。
            #提取失败先用 ""；去掉首尾空白后仍为空，则保存 None，不猜章节。
            section = header.strip() or None #section来自每一页 PDF 的页眉文字
            base = {
                "source": pdf_path.name, "source_id": source_id,
                "content_hash": content_hash, "page": page.page_number,
                "pipeline_version": "tables-v1",
                "brand": metadata["brand"],
                "vehicle_model": metadata["vehicle_model"],
                "model_year": metadata["model_year"], "section": section,
            }
            if not body and not tables:
                empty_pages.append(page.page_number)
            if body:
                records.append({**base, "block_id": "body", "kind": "text", "text": body}) #**base把共同来源字段展开到每条字典中
                #block_id 区分同页不同内容，kind 区分正文和表格，
            for table in tables:
                for number, row_text in enumerate(table["row_texts"], start=1):
                    text = (f"表格数据：\n{row_text}\n"
                            f"相邻上文（归属待核验）：\n{table['context_before']}\n"
                            f"相邻下文（归属待核验）：\n{table['context_after']}")
                    records.append({**base, "block_id": f"{table['block_id']}:r{number}",
                                    "kind": "table", "text": text})
            page.close()
    return records, empty_pages, total_pages, content_hash


if __name__ == "__main__":
    #Python的sys库是标准库中用于访问解释器运行环境的重要模块，提供命令行参数（如 sys.argv）、解释器路径（如 sys.path）、字节序（sys.byteorder）等系统级信息。
    #sys.argv 是命令行参数列表：第 0 项是脚本名，第 1 项才是 PDF 路径，第 2 项是 source_id
    if len(sys.argv) != 3:
        raise SystemExit("用法: python ingestion.py <PDF路径> <source_id>")

    pdf_path = Path(sys.argv[1])
    source_id = sys.argv[2]
    if not pdf_path.is_file(): #.is_file() 先确认目标确实是文件
        raise FileNotFoundError(pdf_path)

    records, empty_pages, total_pages, content_hash = load_pdf_pages(pdf_path, source_id)

    print(f"SHA-256: {content_hash}")
    print(f"PDF总页数: {total_pages}")
    print(f"有提取文本的页数: {len({r['page'] for r in records})}")
    print(f"无提取文本的页数: {len(empty_pages)}")
    print(f"无提取文本的全部页码: {empty_pages}")

    #if records 先确认列表非空，才访问 records[0]，否则全书都没提取到文字时会越界
    if records:
        first = records[0]
        print(f"首条记录: {first['source']}，PDF第{first['page']}页")
        print(first["text"][:80]) #[:80] 只是控制终端预览长度，不改变保存的数据

    for record in records:
        if record["page"] == 351:
            preview = record.copy() #创建字典的浅拷贝
            print(preview["text"])
            print("section:", preview["section"])
            print("original_has_section:", "section" in record) #"section" in record 检查的是字典的键，结果为布尔值
            #字段是否进入真实记录？看 original_has_section: True；是否落库？看重新查询后的 metadata: 输出。
            break

#运行方式：（代码固定，输入资料可替换）
#python ".\projects\smart_cockpit_rag\ingestion.py" "..\阶段12：智能座舱汽车知识大脑\丁师兄综合项目实战项目\综合项目实战项目一\data\train_a.pdf" training_manual_a