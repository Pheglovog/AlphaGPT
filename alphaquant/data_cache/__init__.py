"""
AlphaGPT 数据缓存模块

提供多级缓存系统：
1. 内存缓存（最快，小数据量）
2. Redis 缓存（快速，中等数据量，可选）
3. 文件缓存（慢速，大数据量）

使用示例：
    from alphaquant.data_cache import CacheManager

    # 初始化缓存管理器
    cache = CacheManager()

    # 缓存数据
    cache.set('stock_data:600000', df, ttl=3600)

    # 获取缓存
    data = cache.get('stock_data:600000')

    # 清除缓存
    cache.delete('stock_data:600000')
"""

from .cache_manager import CacheManager
from .file_cache import FileCache
from .memory_cache import MemoryCache
from .redis_cache import RedisCache, is_redis_available

# 为了向后兼容，DataCache 是 CacheManager 的别名
DataCache = CacheManager

__all__ = [
    'CacheManager',
    'DataCache',  # 向后兼容
    'FileCache',
    'MemoryCache',
    'RedisCache',
    'is_redis_available',
]

__version__ = '1.0.0'
