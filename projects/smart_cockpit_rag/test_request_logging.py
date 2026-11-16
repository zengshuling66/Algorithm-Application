import json
import logging
import pytest
from request_logging import measure_stage

@pytest.mark.parametrize("fail", [False, True])
def test_measure_stage(caplog, fail):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        if fail:
            with pytest.raises(ValueError):
                with measure_stage("offline-id", "retrieval"):
                    raise ValueError("private message")
        else:
            with measure_stage("offline-id", "retrieval"):
                pass
    row = json.loads(caplog.records[-1].getMessage())
    assert row["event"] == "stage_finished" and row["stage"] == "retrieval"
    assert row["request_id"] == "offline-id" and row["seconds"] >= 0
    assert row["outcome"] == ("error" if fail else "completed")
    assert row["error_type"] == ("ValueError" if fail else None)
    assert "private message" not in caplog.text

def test_stage_records_upstream_status(caplog):
    class UpstreamFailure(Exception):
        status_code = 429
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with pytest.raises(UpstreamFailure):
            with measure_stage("offline-id", "generation"):
                raise UpstreamFailure("private message")
    row = json.loads(caplog.records[-1].getMessage())
    assert row["stage"] == "generation" and row["outcome"] == "error"
    assert row["upstream_http_status"] == 429
    assert "private message" not in caplog.text