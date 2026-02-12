"""
基础缓存接口

定义所有缓存实现的通用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import timedelta


class BaseCache(ABC):
    """缓存基类"""

    def __init__(self, default_ttl: Optional[int] = None):
        """
        初始化缓存

        Args:
            default_ttl: 默认过期时间（秒），None 表示永不过期
        """
        self.default_ttl = default_ttl

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存数据，不存在返回 None
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存数据

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        pass

    def get_or_set(
        self,
        key: str,
        value_factory: callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        获取缓存，如果不存在则设置

        Args:
            key: 缓存键
            value_factory: 值生成函数
            ttl: 过期时间（秒）

        Returns:
            缓存值
        """
        value = self.get(key)
        if value is not None:
            return value

        value = value_factory()
        self.set(key, value, ttl)
        return value

    def set_many(self, items: dict, ttl: Optional[int] = None) -> int:
        """
        批量设置缓存

        Args:
            items: {key: value} 字典
            ttl: 过期时间（秒）

        Returns:
            成功数量
        """
        success = 0
        for key, value in items.items():
            if self.set(key, value, ttl):
                success += 1
        return success

    def get_many(self, keys: list) -> dict:
        """
        批量获取缓存

        Args:
            keys: 缓存键列表

        Returns:
            {key: value} 字典
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def delete_many(self, keys: list) -> int:
        """
        批量删除缓存

        Args:
            keys: 缓存键列表

        Returns:
            成功数量
        """
        success = 0
        for key in keys:
            if self.delete(key):
                success += 1
        return success
