import sys
import pdfplumber

#把一张表变成二维列表
def expand_merged_cells(table):
    xs = sorted({x for cell in table.cells for x in (cell[0], cell[2])})
    ys = sorted({y for cell in table.cells for y in (cell[1], cell[3])})
    grid = [[None] * (len(xs) - 1) for _ in range(len(ys) - 1)] #建立空二维列表
    for row, texts in zip(table.rows, table.extract()):
        for box, text in zip(row.cells, texts):
            if box is None:
                continue
            left, top, right, bottom = box
            c0, c1 = xs.index(left), xs.index(right)
            r0, r1 = ys.index(top), ys.index(bottom)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if grid[r][c] is not None:
                        raise ValueError("Overlapping cells")
                    grid[r][c] = (text or "").replace("\n", " ")
    if any(value is None for row in grid for value in row):
        raise ValueError("Incomplete table grid")
    return grid


#结合多级表头生成行文本
def render_table_rows(table, rows):
    ys = sorted({y for cell in table.cells for y in (cell[1], cell[3])})
    header_bottom = max(cell[3] for cell in table.rows[0].cells if cell is not None)
    header_count = ys.index(header_bottom)
    if rows[0][0].strip().isdigit():
        header_count = 0
    if header_count >= len(rows):
        raise ValueError("Cannot separate table headers from data")

    headers = []
    for column in range(len(rows[0])):
        parts = []
        for row in rows[:header_count]:
            value = row[column]
            if value and value not in parts:
                parts.append(value)
        headers.append(" / ".join(parts) or f"column_{column + 1}")

    texts = []
    for row in rows[header_count:]:
        fields = []
        for header, value in zip(headers, row):
            fields.append(f"{header}: {value or '[empty]'}")
        texts.append("\n".join(fields))
    return texts


#正文去重、表格记录及邻文候选
def parse_page(page, context_margin=60):
    tables = sorted(page.find_tables(), key=lambda item: (item.bbox[1], item.bbox[0]))
    #page.find_tables() 返回表格对象列表，bbox 是 (左, 上, 右, 下)
    body_page = page
    for table in tables:
        body_page = body_page.outside_bbox(table.bbox) #每次 outside_bbox() 返回一个排除指定区域后的页面视图，再让 body_page 指向它

    records = []
    for index, table in enumerate(tables, start=1):
        left, top, right, bottom = table.bbox
        above_box = (left, max(page.bbox[1], top - context_margin), right, top)
        below_box = (left, bottom, right, min(page.bbox[3], bottom + context_margin))
        rows = expand_merged_cells(table) #从原表还原二维网格
        records.append({ #每个字典保存页码、表格边界、二维表格和邻近文字
            "block_id": f"p{page.page_number}:t{index}", #block_id如 p351:t4，只表示页内表格身份，后续还要结合资料身份与解析版本构造 chunk ID
            "page": page.page_number,
            "bbox": table.bbox,
            "rows": rows,
            "row_texts": render_table_rows(table, rows), #结合表头生成行文本
            "context_before": body_page.crop(above_box).extract_text() or "",
            "context_after": body_page.crop(below_box).extract_text() or "",
        })
    return (body_page.extract_text() or "").strip(), records
    #第一项提取剩余文字；or "" 处理没有文字的情况，.strip() 去掉首尾空白；第二项就是四张表的字典列表。


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python table_parser.py <pdf_path>")
    with pdfplumber.open(sys.argv[1]) as pdf:
        for page in pdf.pages:
            body_text, table_records = parse_page(page) #body_text 接正文字符串，table_records 接表格列表
            if table_records:
                print(f"page={page.page_number}, tables={len(table_records)}")
                print("BODY:", body_text)
                for record in table_records:
                    for text in record["row_texts"]:
                        print(text)
            page.close()