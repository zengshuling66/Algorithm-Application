import sqlite3
from pathlib import Path
from typing import Any

# 数据库文件的路径
DB_PATH = (
    Path(__file__).resolve().parent # 当前文件的父目录，使用绝对路径，避免相对路径会受到终端当前目录影响
    / "data"
    / "scan_history.db"
)


def connect_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    #SQLite能创建 .db 文件，但不能创建不存在的 data 文件夹
    db_path.parent.mkdir(parents=True, exist_ok=True) #parents=True：上层目录不存在时一起创建；exist_ok=True：目录已经存在也不报错。

    #connect()会：数据库不存在：创建数据库文件；数据库存在：打开数据库；返回一个数据库连接对象。
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row #设置 row_factory 后，可以按列名读取

    return connection


def init_database(db_path: Path = DB_PATH) -> None:
    connection = connect_database(db_path)

    try:
        connection.execute( #execute()可以理解为“提出修改”
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                folder_count INTEGER NOT NULL,
                error_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_scan_history_created_at
            ON scan_history(created_at DESC, id DESC)
            """
        )

        connection.commit() #commit()可以理解为“确认保存”

    except sqlite3.Error:
        connection.rollback()
        raise

    finally:
        connection.close()

#保存扫描历史
def save_scan_history(report: dict[str, Any], db_path: Path = DB_PATH) -> int:
    connection = connect_database(db_path)

    try:
        #execute()执行 SQL后会返回一个游标对象 cursor；游标可以理解为数据库执行这条 SQL 后，留给 Python 的结果操作入口。
        #扫描目录和统计值通过问号占位符写入数据库，没有直接拼接进 SQL 字符串，而是通过参数化的方式传递。
        #参数化 SQL 将 SQL 结构与参数值分开处理，能够避免拼接错误和 SQL 注入。
        cursor = connection.execute(
            """
            INSERT INTO scan_history (
                root,
                file_count,
                folder_count,
                error_count
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                report["root"],
                report["file_count"],
                report["folder_count"],
                report["error_count"],
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)

    except sqlite3.Error:
        connection.rollback()
        raise

    finally:
        connection.close()

#查询最近记录
def list_scan_history(limit: int = 10, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit 必须大于等于 1")

    connection = connect_database(db_path)

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                root,
                file_count,
                folder_count,
                error_count,
                created_at
            FROM scan_history
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,), #这是只有一个元素的元组，如果不加逗号只是普通整数；数据库要求参数以序列形式传入，所以单个参数也要写成单元素元组。
        ).fetchall() #表示把查询到的所有行取回来

        return [dict(row) for row in rows]

    finally:
        connection.close()


def main() -> None:
    init_database()
    print("数据库初始化完成：", DB_PATH)


if __name__ == "__main__":
    main()