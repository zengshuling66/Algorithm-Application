import sys
from pathlib import Path
from ingestion import load_pdf_pages


#全大写是 Python 中“常量”的命名约定，表示这些值是默认配置，不应在程序运行过程中随意修改
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

#一个字符串 → 按固定窗口切分 → 多个字符串
# 第一段：0 ---------------- 499
# 第二段：          400 ---------------- 899
# 重叠部分：        400 ----- 499
# 每次向前移动的距离 = chunk_size - overlap=500-100=400
def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap必须大于等于0且小于chunk_size"
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min( #min() 的作用是防止最后一个窗口越过字符串结尾。Python 切片即使越界通常也不会报错，但明确计算 end 后，我们能判断是否已经处理到末尾。
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks

#先看旧逻辑：外层 for 逐页处理；split_text() 对这一页的正文切块；内层 enumerate(..., start=1) 给这一页的块从 1 编号；append() 把块字典放入最终列表。
def build_chunks(
    page_records: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    chunks = []

    for page_record in page_records:
        page_chunks = split_text(
            text=page_record["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            chunk_id = (
                f"{page_record['source_id']}:"
                f"{page_record['content_hash']}:"
                f"s{chunk_size}:o{overlap}:"
                f"p{page_record['page']}:c{chunk_index}"
            ) #chunk_id 依次包含：资料身份、文件内容哈希、切块参数、PDF 页码、页内块序号

            chunks.append(
                {
                    "chunk_id": chunk_id, #唯一标识
                    "text": chunk_text,
                    "source": page_record["source"],
                    "source_id": page_record["source_id"],
                    "content_hash": page_record["content_hash"],
                    "page": page_record["page"],
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text), #检查切分长度是否符合预期
                }
            )

    return chunks


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "用法: python chunking.py <PDF路径> <source_id>"
        )

    pdf_path = Path(sys.argv[1])
    source_id = sys.argv[2]

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    page_records, empty_pages, total_pages, content_hash = load_pdf_pages(pdf_path, source_id)

    chunks = build_chunks(page_records)

    print(f"PDF总页数: {total_pages}")
    print(f"有文本的页面数: {len(page_records)}")
    print(f"无文本的页面数: {len(empty_pages)}")
    print(f"chunk总数: {len(chunks)}")

    if chunks:
        first_chunk = chunks[0]

        print("\n首个chunk:")
        print(f"资料ID: {first_chunk['source_id']}")
        print(f"id: {first_chunk['chunk_id']}")
        print(
            f"来源: {first_chunk['source']}，"
            f"PDF第{first_chunk['page']}页"
        )
        print(f"字符数: {first_chunk['char_count']}")
        print(first_chunk["text"][:100])


if __name__ == "__main__":
    main()