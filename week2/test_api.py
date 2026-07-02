from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app


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