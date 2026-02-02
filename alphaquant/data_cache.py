"""
数据缓存管理器
避免重复请求 Tushare API，提升性能
"""

import hashlib
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path


class CacheConfig:
    """缓存配置"""

    def __init__(
        self,
        cache_dir: str = "./cache",
        max_age_hours: int = 24,  # 缓存过期时间（小时）
        max_size_mb: int = 1024,  # 缓存最大大小（MB）
        enable_compression: bool = True  # 是否启用压缩
    ):
        self.cache_dir = Path(cache_dir)
        self.max_age_hours = max_age_hours
        self.max_size_mb = max_size_mb
        self.enable_compression = enable_compression

        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存统计
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_updates": 0,
            "total_size_bytes": 0
        }

    def get_cache_file_path(self, cache_key: str, data_type: str = "data") -> Path:
        """
        获取缓存文件路径

        Args:
            cache_key: 缓存键
            data_type: 数据类型（data, metadata）

        Returns:
            缓存文件路径
        """
        filename = f"{cache_key}.{data_type}"
        if self.enable_compression:
            filename += ".pkl.gz"
        else:
            filename += ".pkl"

        return self.cache_dir / filename

    def is_cache_valid(self, cache_file: Path) -> bool:
        """
        检查缓存是否有效

        Args:
            cache_file: 缓存文件路径

        Returns:
            缓存是否有效
        """
        if not cache_file.exists():
            return False

        # 检查文件年龄
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.total_seconds() > self.max_age_hours * 3600:
            logger.info(f"缓存已过期：{cache_file.name}（年龄：{file_age.total_seconds()/3600:.1f} 小时）")
            return False

        # 检查文件大小
        file_size_mb = cache_file.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            logger.warning(f"缓存文件过大：{cache_file.name}（{file_size_mb:.1f} MB）")
            return False

        return True

    def clean_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        清理过期缓存

        Args:
            older_than_hours: 清理多少小时前的缓存（None = 使用配置的 max_age_hours）

        Returns:
            清理的文件数
        """
        if older_than_hours is None:
            older_than_hours = self.max_age_hours

        count = 0
        for cache_file in self.cache_dir.glob("*.pkl*"):
            if not self.is_cache_valid(cache_file):
                cache_file.unlink()
                count += 1
                logger.info(f"删除缓存文件：{cache_file.name}")

        return count

    def get_cache_size(self) -> int:
        """获取缓存大小（字节）"""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.pkl*"):
            if cache_file.exists():
                total_size += cache_file.stat().st_size

        return total_size


class DataCache:
    """数据缓存管理器"""

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化数据缓存

        Args:
            config: 缓存配置（如果不提供，使用默认配置）
        """
        self.config = config or CacheConfig()

        # 缓存索引（记录所有缓存键）
        self.cache_index: Dict[str, Dict[str, Any]] = {}

        # 加载缓存索引
        self._load_cache_index()

    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """
        生成缓存键

        Args:
            params: 请求参数字典

        Returns:
            缓存键（MD5 哈希）
        """
        # 将参数转换为排序后的字符串
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))

        # 生成 MD5 哈希
        cache_key = hashlib.md5(param_str.encode()).hexdigest()

        return cache_key

    def _load_cache_index(self):
        """加载缓存索引"""
        index_file = self.config.cache_dir / "cache_index.pkl"

        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    self.cache_index = pickle.load(f)

                logger.info(f"缓存索引已加载（{len(self.cache_index)} 个缓存）")
            except Exception as e:
                logger.warning(f"加载缓存索引失败：{e}")
                self.cache_index = {}
        else:
            logger.info("缓存索引不存在，创建新的")

    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = self.config.cache_dir / "cache_index.pkl"

        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)

            logger.debug(f"缓存索引已保存（{len(self.cache_index)} 个缓存）")
        except Exception as e:
            logger.error(f"保存缓存索引失败：{e}")

    def get(
        self,
        params: Dict[str, Any],
        data_type: str = "data"
    ) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
        """
        从缓存获取数据

        Args:
            params: 请求参数字典
            data_type: 数据类型（data, metadata）

        Returns:
            缓存的数据（如果存在且有效），否则返回 None
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(params)
        cache_file = self.config.get_cache_file_path(cache_key, data_type)

        # 检查缓存是否存在
        if not self.is_cache_available(cache_key):
            logger.debug(f"缓存不存在：{cache_key}")
            return None

        # 检查缓存是否有效
        if not self.config.is_cache_valid(cache_file):
            logger.debug(f"缓存已失效：{cache_key}")
            return None

        # 加载缓存
        try:
            if self.config.enable_compression and cache_file.suffix == ".gz":
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)

            # 更新缓存统计
            self.config.cache_stats["total_requests"] += 1
            self.config.cache_stats["cache_hits"] += 1

            logger.info(f"缓存命中：{cache_key}（{cache_file.name}）")

            return data

        except Exception as e:
            logger.error(f"加载缓存失败：{e}")
            return None

    def set(
        self,
        params: Dict[str, Any],
        data: Union[pd.DataFrame, Dict[str, Any]],
        data_type: str = "data",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        设置缓存

        Args:
            params: 请求参数字典
            data: 要缓存的数据
            data_type: 数据类型（data, metadata）
            metadata: 额外的元数据

        Returns:
            是否成功
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(params)
        cache_file = self.config.get_cache_file_path(cache_key, data_type)
        metadata_file = self.config.get_cache_file_path(cache_key, "metadata")

        try:
            # 保存数据
            if self.config.enable_compression:
                import gzip
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)

            # 保存元数据
            cache_metadata = {
                "created_at": datetime.now().isoformat(),
                "params": params,
                "data_type": data_type,
                "cache_key": cache_key,
                "metadata": metadata or {},
                "cache_file": str(cache_file)
            }

            if self.config.enable_compression:
                with gzip.open(metadata_file, 'wb') as f:
                    pickle.dump(cache_metadata, f)
            else:
                with open(metadata_file, 'wb') as f:
                    pickle.dump(cache_metadata, f)

            # 更新缓存索引
            self.cache_index[cache_key] = cache_metadata
            self._save_cache_index()

            # 更新缓存统计
            self.config.cache_stats["total_requests"] += 1
            self.config.cache_stats["cache_misses"] += 1
            self.config.cache_stats["cache_updates"] += 1

            logger.info(f"缓存已保存：{cache_key}（{cache_file.name}）")

            # 检查缓存大小
            cache_size = self.config.get_cache_size()
            self.config.cache_stats["total_size_bytes"] = cache_size
            cache_size_mb = cache_size / (1024 * 1024)

            logger.debug(f"缓存大小：{cache_size_mb:.2f} MB")

            return True

        except Exception as e:
            logger.error(f"保存缓存失败：{e}")
            return False

    def invalidate(self, params: Dict[str, Any]) -> bool:
        """
        使缓存失效

        Args:
            params: 请求参数字典

        Returns:
            是否成功
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(params)

        if cache_key not in self.cache_index:
            logger.warning(f"缓存不存在：{cache_key}")
            return False

        # 删除缓存文件
        cache_metadata = self.cache_index[cache_key]
        cache_file = Path(cache_metadata["cache_file"])
        metadata_file = self.config.get_cache_file_path(cache_key, "metadata")

        try:
            if cache_file.exists():
                cache_file.unlink()

            if metadata_file.exists():
                metadata_file.unlink()

            # 从索引中删除
            del self.cache_index[cache_key]
            self._save_cache_index()

            logger.info(f"缓存已失效：{cache_key}")
            return True

        except Exception as e:
            logger.error(f"使缓存失效失败：{e}")
            return False

    def is_cache_available(self, cache_key: str) -> bool:
        """
        检查缓存是否可用

        Args:
            cache_key: 缓存键

        Returns:
            是否可用
        """
        return cache_key in self.cache_index

    def clear(self):
        """清空所有缓存"""
        count = 0
        for cache_file in self.config.cache_dir.glob("*.pkl*"):
            if cache_file.exists():
                cache_file.unlink()
                count += 1

        # 清空索引
        self.cache_index = {}
        self._save_cache_index()

        logger.info(f"已清空所有缓存（{count} 个文件）")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计

        Returns:
            缓存统计信息
        """
        cache_hit_rate = 0
        if self.config.cache_stats["total_requests"] > 0:
            cache_hit_rate = (self.config.cache_stats["cache_hits"] /
                          self.config.cache_stats["total_requests"]) * 100

        cache_size_bytes = self.config.get_cache_size()
        cache_size_mb = cache_size_bytes / (1024 * 1024)

        return {
            "total_requests": self.config.cache_stats["total_requests"],
            "cache_hits": self.config.cache_stats["cache_hits"],
            "cache_misses": self.config.cache_stats["cache_misses"],
            "cache_updates": self.config.cache_stats["cache_updates"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size_bytes": cache_size_bytes,
            "cache_size_mb": cache_size_mb,
            "cache_count": len(self.cache_index)
        }

    def get_cache_report(self) -> str:
        """
        获取缓存报告

        Returns:
            缓存报告字符串
        """
        stats = self.get_cache_stats()

        report = f"""
        === 数据缓存报告 ===

        缓存目录：{self.config.cache_dir}
        缓存配置：
          - 最大年龄：{self.config.max_age_hours} 小时
          - 最大大小：{self.config.max_size_mb} MB
          - 启用压缩：{self.config.enable_compression}

        缓存统计：
          - 总请求数：{stats['total_requests']}
          - 缓存命中数：{stats['cache_hits']}
          - 缓存未命中数：{stats['cache_misses']}
          - 缓存更新数：{stats['cache_updates']}
          - 缓存命中率：{stats['cache_hit_rate']:.2f}%
          - 缓存文件数：{stats['cache_count']}
          - 缓存大小：{stats['cache_size_mb']:.2f} MB

        缓存文件：
        """

        # 列出所有缓存文件
        for i, (cache_key, metadata) in enumerate(self.cache_index.items(), 1):
            report += f"  {i}. {cache_key} - {metadata['params']}\n"

        return report


# 使用示例
def example_usage():
    """使用示例"""
    import pandas as pd
    import numpy as np

    # 创建数据缓存
    cache = DataCache()

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    data = {
        'trade_date': dates,
        'open': 10 + np.random.randn(len(dates)) * 2,
        'high': 12 + np.random.randn(len(dates)) * 2,
        'low': 8 + np.random.randn(len(dates)) * 2,
        'close': 11 + np.random.randn(len(dates)) * 2,
        'vol': np.random.randint(100000, 1000000, len(dates))
    }

    df = pd.DataFrame(data)

    # 测试参数
    params1 = {
        'ts_code': '000001.SZ',
        'start_date': '20240101',
        'end_date': '20240131'
    }

    params2 = {
        'ts_code': '000002.SZ',
        'start_date': '20240101',
        'end_date': '20240131'
    }

    # 1. 第一次请求（缓存未命中）
    print("\n=== 第一次请求（缓存未命中）===")
    cached_data = cache.get(params1, data_type="data")

    if cached_data is None:
        print("缓存未命中，从 API 获取数据...")
        # 这里应该调用 Tushare API
        cache.set(params1, df, data_type="data", metadata={"source": "Tushare"})
    else:
        print("缓存命中！")

    # 2. 第二次请求（缓存命中）
    print("\n=== 第二次请求（缓存命中）===")
    cached_data = cache.get(params1, data_type="data")

    if cached_data is not None:
        print(f"缓存命中！获取 {len(cached_data)} 行数据")
        print(cached_data.head())
    else:
        print("缓存未命中")

    # 3. 获取缓存统计
    print("\n=== 缓存统计 ===")
    stats = cache.get_cache_stats()
    print(f"缓存命中率：{stats['cache_hit_rate']:.2f}%")
    print(f"缓存文件数：{stats['cache_count']}")

    # 4. 获取缓存报告
    print("\n=== 缓存报告 ===")
    report = cache.get_cache_report()
    print(report)

    # 5. 使缓存失效
    print("\n=== 使缓存失效 ===")
    cache.invalidate(params1)

    # 6. 再次请求（缓存未命中）
    print("\n=== 使缓存失效后的请求（缓存未命中）===")
    cached_data = cache.get(params1, data_type="data")

    if cached_data is None:
        print("缓存未命中，需要重新获取数据...")
    else:
        print("缓存命中！")


if __name__ == "__main__":
    example_usage()
