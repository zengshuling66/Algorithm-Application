import pytest
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app, generate_file_events


client = TestClient(app) #TestClient：不启动浏览器和 Uvicorn，直接模拟 HTTP 请求
#请求参数校验 → 路由函数 → load_scan_report → HTTPException → JSON响应

# Arrange：准备条件
# Act：调用被测试接口
# Assert：检查结果是否符合预期

def test_invalid_limit_returns_422():
    response = client.get("/files", params={"limit": 0})

    assert response.status_code == 422


def test_missing_suffix_returns_422():
    response = client.get("/files/by-suffix")

    assert response.status_code == 422


def test_invalid_root_returns_500():
    missing_root = Path(__file__).parent / "__missing_scan_root__"
    fake_config = SimpleNamespace(root=missing_root) #SimpleNamespace(root=...)：快速创建一个带 .root 属性的假配置对象

    with patch("api.build_config", return_value=fake_config): #临时替换 api.py 命名空间中的函数，离开 with 后自动恢复
        response = client.get("/files")

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "SCAN_ROOT_INVALID"


def test_scan_os_error_returns_503(tmp_path): #tmp_path：pytest 自动提供的临时目录，测试后自动清理
    fake_config = SimpleNamespace(root=tmp_path)

    with (
        patch("api.build_config", return_value=fake_config),
        patch("api.scan_folder", side_effect=OSError("模拟文件系统故障")), #side_effect=OSError(...)：调用假函数时主动抛出异常
    ):
        response = client.get("/files")

    assert response.status_code == 503 #assert：声明预期结果；不成立时测试失败
    assert response.json()["detail"]["code"] == "SCAN_SERVICE_UNAVAILABLE"

def test_generate_file_events_yields_one_by_one():
    files = [
        {"name": "guide.pdf", "suffix": ".pdf"},
        {"name": "notes.txt", "suffix": ".txt"},
    ]

    events = generate_file_events(files, limit=1)

    assert iter(events) is events #生成器本身就是迭代器

    first_event = next(events) #得到文件事件
    done_event = next(events) #limit=1：第二个文件没有被产生；得到完成事件

    assert first_event == {
        "event": "file",
        "index": 1,
        "total": 1,
        "name": "guide.pdf",
        "suffix": ".pdf",
    }

    assert done_event == {
        "event": "done",
        "total": 1,
    }

    with pytest.raises(StopIteration):
        next(events) #第三次next调用时，生成器耗尽，抛出 StopIteration

def test_stream_files_returns_sse():
    fake_report = {
        "files": [
            {"name": "guide.pdf", "suffix": ".pdf"},
            {"name": "notes.txt", "suffix": ".txt"},
        ]
    }

    with patch("api.load_scan_report", return_value=fake_report):
        with client.stream(
            "GET",
            "/stream/files",
            params={"limit": 1, "delay": 0},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert "event: file\n" in body
    assert '"name": "guide.pdf"' in body
    assert "notes.txt" not in body
    assert "event: done\n" in body

def test_stream_files_rejects_negative_delay():
    response = client.get(
        "/stream/files",
        params={"delay": -0.1},
    )

    assert response.status_code == 422

def test_create_scan_saves_history():
    fake_report = {
        "root": "C:/materials",
        "file_count": 20,
        "folder_count": 5,
        "error_count": 0,
    }

    with (
        patch("api.load_scan_report", return_value=fake_report),
        patch("api.save_scan_history", return_value=7) as mocked_save,
    ):
        response = client.post("/scans")

    assert response.status_code == 201
    assert response.json() == {
        "id": 7,
        "root": "C:/materials",
        "file_count": 20,
        "folder_count": 5,
        "error_count": 0,
    }
    mocked_save.assert_called_once_with(fake_report)


def test_list_scan_history_returns_records():
    fake_scans = [
        {
            "id": 2,
            "root": "C:/materials",
            "file_count": 30,
            "folder_count": 8,
            "error_count": 1,
            "created_at": "2026-07-23 08:00:00",
        }
    ]

    with patch(
        "api.list_scan_history",
        return_value=fake_scans,
    ) as mocked_list:
        response = client.get(
            "/scans",
            params={"limit": 1},
        )

    assert response.status_code == 200
    assert response.json() == {
        "limit": 1,
        "count": 1,
        "scans": fake_scans,
    }
    mocked_list.assert_called_once_with(limit=1)


def test_scan_history_rejects_invalid_limit():
    response = client.get(
        "/scans",
        params={"limit": 0},
    )

    assert response.status_code == 422


def test_create_scan_database_error_returns_503():
    fake_report = {
        "root": "C:/materials",
        "file_count": 20,
        "folder_count": 5,
        "error_count": 0,
    }

    with (
        patch("api.load_scan_report", return_value=fake_report),
        patch(
            "api.save_scan_history",
            side_effect=sqlite3.OperationalError(
                "模拟数据库不可用"
            ),
        ),
    ):
        response = client.post("/scans")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "DATABASE_UNAVAILABLE"
    )

#运行python -m pytest -q进行测试