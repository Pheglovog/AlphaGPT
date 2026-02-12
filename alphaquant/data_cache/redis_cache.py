"""
Redis 缓存实现

使用 Redis 提供高性能分布式缓存
适合中等数据量、高频访问的场景
"""

import time
from typing import Any, Optional
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .base_cache import BaseCache


def is_redis_available() -> bool:
    """检查 Redis 模块是否可用"""
    return REDIS_AVAILABLE


class RedisCache(BaseCache):
    """
    Redis 缓存

    特点：
    - 高性能（内存 + 网络延迟）
    - 持久化（可选）
    - 分布式支持
    - 适合中等数据量（1-10GB）
    - 需要 Redis 服务器

    安装：pip install redis
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = 3600,
        key_prefix: str = 'alphagpt:',
        **kwargs
    ):
        """
        初始化 Redis 缓存

        Args:
            host: Redis 主机
            port: Redis 端口
            db: Redis 数据库编号
            password: Redis 密码
            default_ttl: 默认过期时间（秒），3600 = 1 小时
            key_prefix: 键前缀
            **kwargs: 其他 redis.Redis 参数
        """
        super().__init__(default_ttl)

        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis module not available. "
                "Install with: pip install redis"
            )

        self.key_prefix = key_prefix

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # 使用二进制模式支持 pickle
                **kwargs
            )
            # 测试连接
            self.client.ping()
            logger.info(f"Redis cache connected: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
        }

    def _make_key(self, key: str) -> str:
        """
        生成完整的 Redis 键

        Args:
            key: 原始键

        Returns:
            完整键
        """
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存数据，不存在返回 None
        """
        redis_key = self._make_key(key)

        try:
            data = self.client.get(redis_key)
            if data is None:
                self._stats['misses'] += 1
                return None

            # 反序列化
            import pickle
            value = pickle.loads(data)
            self._stats['hits'] += 1
            return value

        except Exception as e:
            logger.warning(f"Failed to get from Redis: {e}")
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
        redis_key = self._make_key(key)

        # 计算 TTL
        if ttl is None:
            ttl = self.default_ttl

        try:
            # 序列化
            import pickle
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # 设置缓存
            if ttl is not None:
                self.client.setex(redis_key, ttl, data)
            else:
                self.client.set(redis_key, data)

            self._stats['sets'] += 1
            return True

        except Exception as e:
            logger.warning(f"Failed to set to Redis: {e}")
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
        redis_key = self._make_key(key)

        try:
            result = self.client.delete(redis_key)
            self._stats['deletes'] += result
            return result > 0

        except Exception as e:
            logger.warning(f"Failed to delete from Redis: {e}")
            self._stats['errors'] += 1
            return False

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        redis_key = self._make_key(key)

        try:
            return bool(self.client.exists(redis_key))
        except Exception as e:
            logger.warning(f"Failed to check existence in Redis: {e}")
            return False

    def clear(self) -> bool:
        """
        清空所有缓存（带前缀）

        Returns:
            是否成功
        """
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)

            if keys:
                self.client.delete(*keys)

            return True

        except Exception as e:
            logger.warning(f"Failed to clear Redis cache: {e}")
            return False

    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0.0

        # 获取 Redis 服务器信息
        info = {}
        try:
            info = self.client.info('stats')
            key_count = self.client.dbsize()
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {e}")
            key_count = 0

        return {
            'type': 'redis',
            'key_prefix': self.key_prefix,
            'key_count': key_count,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': f"{hit_rate * 100:.2f}%",
            'sets': self._stats['sets'],
            'deletes': self._stats['deletes'],
            'errors': self._stats['errors'],
            'redis_info': info,
        }

    def set_many(self, items: dict, ttl: Optional[int] = None) -> int:
        """
        批量设置缓存（使用 Pipeline 提高性能）

        Args:
            items: {key: value} 字典
            ttl: 过期时间（秒）

        Returns:
            成功数量
        """
        import pickle

        # 计算 TTL
        if ttl is None:
            ttl = self.default_ttl

        try:
            pipe = self.client.pipeline()

            for key, value in items.items():
                redis_key = self._make_key(key)
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

                if ttl is not None:
                    pipe.setex(redis_key, ttl, data)
                else:
                    pipe.set(redis_key, data)

            pipe.execute()

            self._stats['sets'] += len(items)
            return len(items)

        except Exception as e:
            logger.warning(f"Failed to set many to Redis: {e}")
            self._stats['errors'] += len(items)
            return 0

    def get_many(self, keys: list) -> dict:
        """
        批量获取缓存（使用 Pipeline 提高性能）

        Args:
            keys: 缓存键列表

        Returns:
            {key: value} 字典
        """
        import pickle

        redis_keys = [self._make_key(key) for key in keys]

        try:
            pipe = self.client.pipeline()
            for redis_key in redis_keys:
                pipe.get(redis_key)

            values = pipe.execute()

            result = {}
            for key, data in zip(keys, values):
                if data is not None:
                    value = pickle.loads(data)
                    result[key] = value
                    self._stats['hits'] += 1
                else:
                    self._stats['misses'] += 1

            return result

        except Exception as e:
            logger.warning(f"Failed to get many from Redis: {e}")
            return {}

    def delete_many(self, keys: list) -> int:
        """
        批量删除缓存

        Args:
            keys: 缓存键列表

        Returns:
            成功数量
        """
        redis_keys = [self._make_key(key) for key in keys]

        try:
            result = self.client.delete(*redis_keys)
            self._stats['deletes'] += result
            return result

        except Exception as e:
            logger.warning(f"Failed to delete many from Redis: {e}")
            return 0

    def expire(self, key: str, ttl: int) -> bool:
        """
        更新缓存过期时间

        Args:
            key: 缓存键
            ttl: 新的过期时间（秒）

        Returns:
            是否成功
        """
        redis_key = self._make_key(key)

        try:
            return bool(self.client.expire(redis_key, ttl))
        except Exception as e:
            logger.warning(f"Failed to expire Redis key: {e}")
            return False

    def ttl(self, key: str) -> Optional[int]:
        """
        获取缓存剩余时间

        Args:
            key: 缓存键

        Returns:
            剩余秒数，-1 表示永不过期，-2 表示不存在
        """
        redis_key = self._make_key(key)

        try:
            return self.client.ttl(redis_key)
        except Exception as e:
            logger.warning(f"Failed to get TTL from Redis: {e}")
            return None

    def get_keys(self, pattern: str = '*') -> list:
        """
        获取匹配的键

        Args:
            pattern: 键模式（默认所有键）

        Returns:
            键列表
        """
        try:
            redis_pattern = f"{self.key_prefix}{pattern}"
            redis_keys = self.client.keys(redis_pattern)
            # 移除前缀
            return [key[len(self.key_prefix):] for key in redis_keys]
        except Exception as e:
            logger.warning(f"Failed to get keys from Redis: {e}")
            return []
