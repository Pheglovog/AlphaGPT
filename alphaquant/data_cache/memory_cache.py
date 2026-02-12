"""
内存缓存实现

使用 Python 字典实现高速内存缓存
适合小数据量、高频访问的场景
"""

import time
from typing import Any, Optional
from collections import OrderedDict

from .base_cache import BaseCache


class MemoryCache(BaseCache):
    """
    内存缓存

    特点：
    - 最快的缓存（内存访问）
    - 适合小数据量（<1GB）
    - 进程重启后丢失
    - 支持 LRU 淘汰策略
    """

    def __init__(
        self,
        default_ttl: Optional[int] = 3600,
        max_size: Optional[int] = 1000,
        enable_lru: bool = True
    ):
        """
        初始化内存缓存

        Args:
            default_ttl: 默认过期时间（秒）
            max_size: 最大缓存条目数，None 表示无限制
            enable_lru: 是否启用 LRU 淘汰策略
        """
        super().__init__(default_ttl)
        self.max_size = max_size
        self.enable_lru = enable_lru and max_size is not None

        # 使用 OrderedDict 实现 LRU
        if self.enable_lru:
            self._cache: OrderedDict = OrderedDict()
        else:
            self._cache = {}

        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存数据，不存在或过期返回 None
        """
        if key not in self._cache:
            self._stats['misses'] += 1
            return None

        entry = self._cache[key]

        # 检查是否过期
        if entry['expires_at'] is not None and time.time() > entry['expires_at']:
            self._delete_internal(key)
            self._stats['misses'] += 1
            return None

        # LRU: 移动到最新位置
        if self.enable_lru:
            self._cache.move_to_end(key)

        self._stats['hits'] += 1
        return entry['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存数据

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None 使用默认值

        Returns:
            是否成功
        """
        # 计算 TTL
        if ttl is None:
            ttl = self.default_ttl

        # 计算过期时间
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        # LRU 淘汰：如果已满，删除最旧的
        if self.enable_lru and len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = next(iter(self._cache))
            self._delete_internal(oldest_key)
            self._stats['evictions'] += 1

        # 设置缓存
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time(),
        }

        # LRU: 移动到最新位置
        if self.enable_lru:
            self._cache.move_to_end(key)

        self._stats['sets'] += 1
        return True

    def delete(self, key: str) -> bool:
        """
        删除缓存数据

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        if key in self._cache:
            self._delete_internal(key)
            return True
        return False

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        return self.get(key) is not None

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            是否成功
        """
        self._cache.clear()
        return True

    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0.0

        return {
            'type': 'memory',
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': f"{hit_rate * 100:.2f}%",
            'sets': self._stats['sets'],
            'deletes': self._stats['deletes'],
            'evictions': self._stats['evictions'],
        }

    def _delete_internal(self, key: str) -> None:
        """
        内部删除方法（不更新统计）

        Args:
            key: 缓存键
        """
        if self.enable_lru:
            del self._cache[key]
        else:
            self._cache.pop(key, None)

    def cleanup_expired(self) -> int:
        """
        清理过期缓存

        Returns:
            清理数量
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry['expires_at'] is not None and current_time > entry['expires_at']
        ]

        for key in expired_keys:
            self._delete_internal(key)

        return len(expired_keys)

    def get_keys(self) -> list:
        """
        获取所有缓存键

        Returns:
            键列表
        """
        return list(self._cache.keys())

    def get_size_bytes(self) -> int:
        """
        估算缓存占用内存（字节数）

        Returns:
            估算的内存大小
        """
        import sys
        return sum(sys.getsizeof(entry['value']) for entry in self._cache.values())
