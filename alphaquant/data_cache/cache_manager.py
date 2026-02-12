"""
多级缓存管理器

提供多级缓存系统，自动从 L1 → L2 → L3 查找
支持缓存预热和智能缓存策略
"""

from typing import Any, Optional, List
from loguru import logger

from .base_cache import BaseCache
from .memory_cache import MemoryCache
from .file_cache import FileCache
from .redis_cache import RedisCache, is_redis_available


class CacheManager:
    """
    多级缓存管理器

    缓存层级：
    - L1: 内存缓存（最快）
    - L2: Redis 缓存（快速，可选）
    - L3: 文件缓存（慢速，持久化）

    策略：
    - 查询：L1 → L2 → L3
    - 写入：同时写入所有层级
    - 回填：L3 → L2 → L1
    """

    def __init__(
        self,
        enable_memory: bool = True,
        enable_redis: bool = False,
        enable_file: bool = True,
        memory_config: Optional[dict] = None,
        redis_config: Optional[dict] = None,
        file_config: Optional[dict] = None,
        auto_warmup: bool = False
    ):
        """
        初始化多级缓存管理器

        Args:
            enable_memory: 启用内存缓存
            enable_redis: 启用 Redis 缓存
            enable_file: 启用文件缓存
            memory_config: 内存缓存配置
            redis_config: Redis 缓存配置
            file_config: 文件缓存配置
            auto_warmup: 自动预热缓存（启动时从 L3 加载到 L1）
        """
        self.caches = []
        self.cache_names = []

        # L1: 内存缓存
        if enable_memory:
            memory_config = memory_config or {
                'default_ttl': 3600,
                'max_size': 1000,
                'enable_lru': True,
            }
            self.l1_cache = MemoryCache(**memory_config)
            self.caches.append(self.l1_cache)
            self.cache_names.append('L1(Memory)')
            logger.info("L1 Memory cache enabled")

        # L2: Redis 缓存
        if enable_redis:
            if is_redis_available():
                redis_config = redis_config or {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'default_ttl': 3600,
                }
                try:
                    self.l2_cache = RedisCache(**redis_config)
                    self.caches.append(self.l2_cache)
                    self.cache_names.append('L2(Redis)')
                    logger.info("L2 Redis cache enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis cache: {e}")
            else:
                logger.warning("Redis not available, install with: pip install redis")

        # L3: 文件缓存
        if enable_file:
            file_config = file_config or {
                'cache_dir': './data_cache',
                'default_ttl': 86400,
                'subdirs': True,
            }
            self.l3_cache = FileCache(**file_config)
            self.caches.append(self.l3_cache)
            self.cache_names.append('L3(File)')
            logger.info("L3 File cache enabled")

        if not self.caches:
            raise ValueError("At least one cache layer must be enabled")

        self.level_count = len(self.caches)

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据（多级查找）

        Args:
            key: 缓存键

        Returns:
            缓存数据，不存在返回 None
        """
        # 从 L1 开始查找
        for i, cache in enumerate(self.caches):
            value = cache.get(key)
            if value is not None:
                # 如果不是 L1，回填到更高层级
                if i > 0:
                    self._promote(key, value, level=i)
                    logger.debug(f"Cache hit at L{i+1}, promoted to higher levels")
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存数据（写入所有层级）

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否成功（所有层级都成功才算成功）
        """
        success = True
        for cache in self.caches:
            if not cache.set(key, value, ttl):
                success = False
        return success

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

    def delete(self, key: str) -> bool:
        """
        删除缓存数据（从所有层级删除）

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        success = True
        for cache in self.caches:
            if not cache.delete(key):
                success = False
        return success

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
        success = True
        for cache in self.caches:
            if not cache.clear():
                success = False
        return success

    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        return {
            'levels': self.level_count,
            'cache_names': self.cache_names,
            'stats': [cache.get_stats() for cache in self.caches],
        }

    def _promote(self, key: str, value: Any, level: int) -> None:
        """
        将数据回填到更高层级

        Args:
            key: 缓存键
            value: 缓存值
            level: 当前层级（从 0 开始）
        """
        for i in range(level):
            self.caches[i].set(key, value)

    def warmup(self, keys: List[str]) -> dict:
        """
        缓存预热：从 L3 加载到 L1

        Args:
            keys: 预热的键列表

        Returns:
            预热结果统计
        """
        stats = {
            'total': len(keys),
            'hit': 0,
            'miss': 0,
            'errors': 0,
        }

        for key in keys:
            try:
                value = self.get(key)
                if value is not None:
                    stats['hit'] += 1
                else:
                    stats['miss'] += 1
            except Exception as e:
                logger.warning(f"Failed to warmup key {key}: {e}")
                stats['errors'] += 1

        logger.info(
            f"Cache warmup completed: "
            f"{stats['hit']} hits, {stats['miss']} misses, {stats['errors']} errors"
        )

        return stats

    def cleanup_expired(self) -> dict:
        """
        清理所有层级的过期缓存

        Returns:
            清理结果统计
        """
        result = {}

        for i, cache in enumerate(self.caches):
            if hasattr(cache, 'cleanup_expired'):
                cleaned = cache.cleanup_expired()
                result[f"L{i+1}"] = cleaned

        return result

    def get_memory_usage(self) -> dict:
        """
        获取内存使用情况

        Returns:
            内存使用统计
        """
        result = {}

        for i, cache in enumerate(self.caches):
            if hasattr(cache, 'get_size_bytes'):
                result[f"L{i+1}"] = cache.get_size_bytes()
            else:
                result[f"L{i+1}"] = None

        return result

    # 便捷方法

    def cache_stock_data(self, stock_id: str, data: Any, ttl: int = 3600) -> bool:
        """缓存股票数据"""
        key = f"stock_data:{stock_id}"
        return self.set(key, data, ttl)

    def get_stock_data(self, stock_id: str) -> Optional[Any]:
        """获取股票数据"""
        key = f"stock_data:{stock_id}"
        return self.get(key)

    def cache_factor_data(self, factor_name: str, data: Any, ttl: int = 3600) -> bool:
        """缓存因子数据"""
        key = f"factor_data:{factor_name}"
        return self.set(key, data, ttl)

    def get_factor_data(self, factor_name: str) -> Optional[Any]:
        """获取因子数据"""
        key = f"factor_data:{factor_name}"
        return self.get(key)

    def cache_backtest_result(self, strategy_id: str, result: Any, ttl: int = 86400) -> bool:
        """缓存回测结果"""
        key = f"backtest_result:{strategy_id}"
        return self.set(key, result, ttl)

    def get_backtest_result(self, strategy_id: str) -> Optional[Any]:
        """获取回测结果"""
        key = f"backtest_result:{strategy_id}"
        return self.get(key)


# 便捷工厂函数

def create_cache(
    use_redis: bool = False,
    cache_dir: str = './data_cache'
) -> CacheManager:
    """
    创建缓存管理器（便捷工厂函数）

    Args:
        use_redis: 是否使用 Redis
        cache_dir: 文件缓存目录

    Returns:
        缓存管理器实例
    """
    return CacheManager(
        enable_memory=True,
        enable_redis=use_redis,
        enable_file=True,
        file_config={'cache_dir': cache_dir}
    )
