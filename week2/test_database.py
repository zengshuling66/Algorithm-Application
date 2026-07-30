import sqlite3
import pytest

from database import (
    init_database,
    list_scan_history,
    save_scan_history,
)


def test_save_and_list_scan_history(tmp_path):
    db_path = tmp_path / "test.db"
    init_database(db_path)

    first_report = {
        "root": "C:/materials/first",
        "file_count": 20,
        "folder_count": 5,
        "error_count": 0,
    }
    second_report = {
        "root": "C:/materials/second",
        "file_count": 30,
        "folder_count": 8,
        "error_count": 1,
    }

    first_id = save_scan_history(first_report, db_path)
    second_id = save_scan_history(second_report, db_path)

    histories = list_scan_history(
        limit=1,
        db_path=db_path,
    )

    assert first_id == 1
    assert second_id == 2
    assert len(histories) == 1
    assert histories[0]["root"] == "C:/materials/second"
    assert histories[0]["file_count"] == 30
    assert histories[0]["created_at"] is not None


def test_invalid_report_is_rolled_back(tmp_path):
    db_path = tmp_path / "test.db"
    init_database(db_path)

    invalid_report = {
        "root": None,
        "file_count": 20,
        "folder_count": 5,
        "error_count": 0,
    }

    with pytest.raises(sqlite3.IntegrityError):
        save_scan_history(invalid_report, db_path)

    histories = list_scan_history(db_path=db_path)

    assert histories == []