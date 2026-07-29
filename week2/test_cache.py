from unittest.mock import patch
import pytest
from cache import TTLCache


def test_cache_hit_returns_value():
    cache = TTLCache(default_ttl=10)
    report = {"file_count": 20}

    with patch(
        "cache.time.monotonic",
        side_effect=[100.0, 105.0],
    ):
        cache.set("scan_report", report)
        result = cache.get("scan_report")

    assert result == report
    assert cache.stats() == {
        "hits": 1,
        "misses": 0,
        "size": 1,
    }

    cache.clear()
    assert cache.stats()["size"] == 0


def test_cache_miss_returns_none():
    cache = TTLCache(default_ttl=10)

    result = cache.get("missing_key")

    assert result is None
    assert cache.stats()["misses"] == 1


def test_expired_cache_returns_none():
    cache = TTLCache(default_ttl=10)

    with patch(
        "cache.time.monotonic",
        side_effect=[100.0, 111.0],
    ):
        cache.set(
            "scan_report",
            {"file_count": 20},
        )
        result = cache.get("scan_report")

    assert result is None
    assert cache.stats() == {
        "hits": 0,
        "misses": 1,
        "size": 0,
    }


def test_cache_rejects_invalid_ttl():
    with pytest.raises(ValueError):
        TTLCache(default_ttl=0)

    cache = TTLCache()

    with pytest.raises(ValueError):
        cache.set(
            "scan_report",
            {},
            ttl=0,
        )