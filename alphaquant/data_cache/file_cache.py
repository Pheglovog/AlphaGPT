"""
文件缓存实现

使用本地文件系统持久化缓存
适合大数据量、低频访问的场景
"""

import os
import time
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from loguru import logger

from .base_cache import BaseCache


class FileCache(BaseCache):
    """
    文件缓存

    特点：
    - 持久化存储（重启不丢失）
    - 适合大数据量（TB 级别）
    - 访问速度较慢（磁盘 I/O）
    - 支持过期时间
    """

    def __init__(
        self,
        cache_dir: str = './data_cache',
        default_ttl: Optional[int] = 86400,
        subdirs: bool = True
    ):
        """
        初始化文件缓存

        Args:
            cache_dir: 缓存目录
            default_ttl: 默认过期时间（秒），86400 = 24 小时
            subdirs: 是否使用子目录组织（hash 前缀）
        """
        super().__init__(default_ttl)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.subdirs = subdirs

        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
        }

    def _get_cache_path(self, key: str) -> Path:
        """
        获取缓存文件路径

        Args:
            key: 缓存键

        Returns:
            缓存文件路径
        """
        # 对键进行 hash，避免特殊字符问题
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        if self.subdirs:
            # 使用前两级 hash 作为子目录，避免单目录文件过多
            subdir = self.cache_dir / key_hash[:2] / key_hash[2:4]
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / f"{key_hash}.cache"
        else:
            return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存数据，不存在或过期返回 None
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self._stats['misses'] += 1
            return None

        try:
            with open(cache_path, 'rb') as f:
                entry = pickle.load(f)

            # 检查是否过期
            if entry['expires_at'] is not None and time.time() > entry['expires_at']:
                self.delete(key)
                self._stats['misses'] += 1
                return None

            self._stats['hits'] += 1
            return entry['value']

        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
            self._stats['errors'] += 1
            return None

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
        cache_path = self._get_cache_path(key)

        # 计算 TTL
        if ttl is None:
            ttl = self.default_ttl

        # 计算过期时间
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        entry = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time(),
        }

        try:
            # 创建临时文件，原子性写入
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 重命名到最终路径（原子操作）
            temp_path.rename(cache_path)

            self._stats['sets'] += 1
            return True

        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_path}: {e}")
            self._stats['errors'] += 1
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存数据

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                cache_path.unlink()
                self._stats['deletes'] += 1
                return True
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_path}: {e}")
                self._stats['errors'] += 1
                return False

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
        try:
            # 递归删除所有缓存文件
            for item in self.cache_dir.rglob('*.cache'):
                item.unlink()

            # 删除空子目录
            for subdir in sorted(self.cache_dir.rglob('*'), reverse=True):
                if subdir.is_dir() and not any(subdir.iterdir()):
                    subdir.rmdir()

            return True

        except Exception as e:
            logger.warning(f"Failed to clear cache directory: {e}")
            return False

    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0.0

        # 计算缓存大小
        total_size = 0
        file_count = 0
        for cache_file in self.cache_dir.rglob('*.cache'):
            total_size += cache_file.stat().st_size
            file_count += 1

        return {
            'type': 'file',
            'cache_dir': str(self.cache_dir),
            'file_count': file_count,
            'total_size': self._format_size(total_size),
            'total_size_bytes': total_size,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': f"{hit_rate * 100:.2f}%",
            'sets': self._stats['sets'],
            'deletes': self._stats['deletes'],
            'errors': self._stats['errors'],
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化字节大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def cleanup_expired(self) -> int:
        """
        清理过期缓存

        Returns:
            清理数量
        """
        current_time = time.time()
        cleaned = 0

        for cache_file in self.cache_dir.rglob('*.cache'):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)

                if entry['expires_at'] is not None and current_time > entry['expires_at']:
                    cache_file.unlink()
                    cleaned += 1

            except Exception as e:
                logger.warning(f"Failed to check cache file {cache_file}: {e}")

        return cleaned

    def get_keys(self) -> list:
        """
        获取所有缓存键（不推荐用于大量缓存）

        Returns:
            键列表
        """
        keys = []
        for cache_file in self.cache_dir.rglob('*.cache'):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    keys.append(entry.get('key', 'unknown'))
            except Exception:
                pass
        return keys
