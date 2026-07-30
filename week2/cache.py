import time
from typing import Any


class TTLCache:
    def __init__(
        self,
        default_ttl: float = 30.0,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError("default_ttl 必须大于 0")

        self.default_ttl = default_ttl

        # key -> (value, expires_at)
        self._store: dict[str, tuple[Any, float]] = {}

        self.hits = 0 #缓存命中次数
        self.misses = 0 #缓存未命中次数，包括key根本不存在或key存在但已经过期
        #我们可以监控命中率，因为命中率过低可能说明缓存没有发挥作用

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        if ttl is None:
            effective_ttl = self.default_ttl
        else:
            effective_ttl = ttl

        if effective_ttl <= 0:
            raise ValueError("ttl 必须大于 0")

        expires_at = time.monotonic() + effective_ttl #time.monotonic() 是单调递增计时器，只用于计算经过时间，不会倒退
        #缓存关心：过去了多少秒；数据库 created_at 关心：具体是哪年哪月哪日
        #TTL 计算      → time.monotonic()
        #数据库创建时间 → CURRENT_TIMESTAMP

        self._store[key] = (
            value,
            expires_at,
        )

    def get(
        self,
        key: str,
    ) -> Any | None:
        item = self._store.get(key)

        if item is None:
            self.misses += 1
            return None

        value, expires_at = item #元组解包

        if time.monotonic() >= expires_at:
            self._store.pop(key, None) #pop() 会删除指定 key，这种“读取时才删除过期数据”的方式叫惰性删除
            self.misses += 1
            return None

        self.hits += 1
        return value

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._store),
        }