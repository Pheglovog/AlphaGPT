"""
内存分析工具

监控和优化内存使用情况
"""

import gc
import psutil
import os
import time
import sys
import pandas as pd
from typing import Optional, Dict, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from loguru import logger


@dataclass
class MemoryStats:
    """内存统计信息"""
    rss_mb: float  # 常驻内存（MB）
    vms_mb: float  # 虚拟内存（MB）
    percent: float  # 占总内存百分比
    available_mb: float  # 可用内存（MB）
    timestamp: float  # 时间戳


class MemoryProfiler:
    """
    内存分析器

    功能：
    - 实时监控内存使用
    - 内存泄漏检测
    - 对象大小分析
    - 性能基准测试
    """

    def __init__(self):
        """初始化内存分析器"""
        self.process = psutil.Process(os.getpid())
        self.baseline = None
        self.snapshots = []

    def get_memory_stats(self) -> MemoryStats:
        """
        获取当前内存统计

        Returns:
            内存统计信息
        """
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=psutil.virtual_memory().available / 1024 / 1024,
            timestamp=time.time()
        )

    def set_baseline(self) -> MemoryStats:
        """
        设置基线内存

        Returns:
            基线内存统计
        """
        self.baseline = self.get_memory_stats()
        logger.info(f"Baseline memory: {self.baseline.rss_mb:.2f} MB")
        return self.baseline

    def snapshot(self, label: str = "") -> MemoryStats:
        """
        创建内存快照

        Args:
            label: 快照标签

        Returns:
            内存统计
        """
        stats = self.get_memory_stats()
        stats_dict = {
            'label': label,
            'stats': stats,
            'delta': None
        }

        # 计算相对于基线的变化
        if self.baseline:
            stats_dict['delta'] = {
                'rss_mb': stats.rss_mb - self.baseline.rss_mb,
                'vms_mb': stats.vms_mb - self.baseline.vms_mb,
            }

        self.snapshots.append(stats_dict)
        return stats

    def get_summary(self) -> Dict:
        """
        获取内存使用摘要

        Returns:
            摘要字典
        """
        if not self.snapshots:
            return {}

        # 找到内存峰值
        max_rss = max(s['stats'].rss_mb for s in self.snapshots)
        min_rss = min(s['stats'].rss_mb for s in self.snapshots)
        avg_rss = sum(s['stats'].rss_mb for s in self.snapshots) / len(self.snapshots)

        # 计算总增长
        if self.baseline and len(self.snapshots) > 0:
            total_growth = self.snapshots[-1]['stats'].rss_mb - self.baseline.rss_mb
        else:
            total_growth = 0.0

        return {
            'baseline': self.baseline.rss_mb if self.baseline else None,
            'peak_rss_mb': max_rss,
            'min_rss_mb': min_rss,
            'avg_rss_mb': avg_rss,
            'total_growth_mb': total_growth,
            'snapshots_count': len(self.snapshots),
        }

    def print_summary(self) -> None:
        """打印内存使用摘要"""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("Memory Usage Summary")
        print("=" * 70)
        print(f"Baseline: {summary['baseline']:.2f} MB" if summary['baseline'] else "Baseline: Not set")
        print(f"Peak: {summary['peak_rss_mb']:.2f} MB")
        print(f"Minimum: {summary['min_rss_mb']:.2f} MB")
        print(f"Average: {summary['avg_rss_mb']:.2f} MB")
        print(f"Total Growth: {summary['total_growth_mb']:+.2f} MB")
        print(f"Snapshots: {summary['snapshots_count']}")
        print("=" * 70 + "\n")

    def clear_snapshots(self) -> None:
        """清除所有快照"""
        self.snapshots.clear()
        self.baseline = None

    @staticmethod
    def get_object_size(obj, seen: Optional[set] = None) -> int:
        """
        获取对象的内存大小

        Args:
            obj: 对象
            seen: 已见对象的集合（避免循环引用）

        Returns:
            内存大小（字节）
        """
        size = sys.getsizeof(obj)

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)

        if isinstance(obj, dict):
            size += sum(MemoryProfiler.get_object_size(k, seen) for k in obj.keys())
            size += sum(MemoryProfiler.get_object_size(v, seen) for v in obj.values())
        elif hasattr(obj, '__dict__'):
            size += MemoryProfiler.get_object_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum(MemoryProfiler.get_object_size(i, seen) for i in obj)
            except:
                pass

        return size

    @staticmethod
    def force_gc() -> None:
        """强制执行垃圾回收"""
        collected = gc.collect()
        logger.info(f"Garbage collected {collected} objects")


# 导入 sys
import sys


@contextmanager
def memory_profile(label: str = "", print_summary: bool = True):
    """
    内存分析上下文管理器

    Args:
        label: 标签
        print_summary: 是否打印摘要

    Example:
        with memory_profile("Load data"):
            data = load_large_dataset()
    """
    profiler = MemoryProfiler()
    profiler.set_baseline()

    try:
        yield profiler
    finally:
        profiler.snapshot(label)
        if print_summary:
            profiler.print_summary()


def track_memory(func: Callable) -> Callable:
    """
    装饰器：跟踪函数的内存使用

    Args:
        func: 要跟踪的函数

    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.set_baseline()

        logger.info(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)

            profiler.snapshot(func.__name__)
            summary = profiler.get_summary()

            logger.info(
                f"{func.__name__} completed. "
                f"Memory: {summary['peak_rss_mb']:.2f} MB (peak), "
                f"+{summary['total_growth_mb']:.2f} MB (growth)"
            )

            return result
        except Exception as e:
            profiler.snapshot(f"{func.__name__}_failed")
            logger.error(f"{func.__name__} failed: {e}")
            raise

    return wrapper


class MemoryLeakDetector:
    """
    内存泄漏检测器

    检测是否存在内存泄漏
    """

    def __init__(self, iterations: int = 10, tolerance: float = 0.1):
        """
        初始化检测器

        Args:
            iterations: 测试迭代次数
            tolerance: 容忍的增长比例（超过此值认为有泄漏）
        """
        self.iterations = iterations
        self.tolerance = tolerance
        self.profiler = MemoryProfiler()

    def detect(self, func: Callable, *args, **kwargs) -> Dict:
        """
        检测内存泄漏

        Args:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            检测结果
        """
        logger.info(
            f"Starting memory leak detection for {func.__name__} "
            f"({self.iterations} iterations)"
        )

        self.profiler.clear_snapshots()
        self.profiler.set_baseline()

        memory_usage = []

        for i in range(self.iterations):
            # 执行函数
            func(*args, **kwargs)

            # 强制 GC
            self.profiler.force_gc()

            # 记录内存
            stats = self.profiler.snapshot(f"Iteration {i+1}")
            memory_usage.append(stats.rss_mb)

            logger.debug(f"Iteration {i+1}: {stats.rss_mb:.2f} MB")

        # 分析结果
        first_mb = memory_usage[0]
        last_mb = memory_usage[-1]
        growth = last_mb - first_mb
        growth_ratio = growth / first_mb if first_mb > 0 else 0

        is_leaking = growth_ratio > self.tolerance

        result = {
            'is_leaking': is_leaking,
            'first_mb': first_mb,
            'last_mb': last_mb,
            'growth_mb': growth,
            'growth_ratio': growth_ratio,
            'tolerance': self.tolerance,
            'memory_usage': memory_usage,
        }

        if is_leaking:
            logger.warning(
                f"Memory leak detected in {func.__name__}! "
                f"Growth: {growth:.2f} MB ({growth_ratio*100:.1f}%)"
            )
        else:
            logger.info(
                f"No memory leak detected in {func.__name__}. "
                f"Growth: {growth:.2f} MB ({growth_ratio*100:.1f}%)"
            )

        return result


class DataFrameMemoryOptimizer:
    """
    DataFrame 内存优化器

    优化 DataFrame 的内存使用
    """

    @staticmethod
    def reduce_memory(df: pd.DataFrame, category_threshold: int = 50) -> pd.DataFrame:
        """
        减少 DataFrame 内存占用

        Args:
            df: 待优化的 DataFrame
            category_threshold: 转换为 category 的唯一值阈值

        Returns:
            优化后的 DataFrame
        """
        before_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        logger.info(f"Optimizing DataFrame memory. Before: {before_memory:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtype

            # 处理数值类型
            if pd.api.types.is_numeric_dtype(col_type):
                if pd.api.types.is_integer_dtype(col_type):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(col_type):
                    df[col] = pd.to_numeric(df[col], downcast='float')

            # 处理字符串类型
            elif pd.api.types.is_string_dtype(col_type) or pd.api.types.is_object_dtype(col_type):
                unique_count = df[col].nunique()
                total_count = len(df[col])

                # 如果唯一值比例小，转换为 category
                if unique_count / total_count < 0.5 and unique_count < category_threshold:
                    df[col] = df[col].astype('category')

        after_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (before_memory - after_memory) / before_memory * 100

        logger.info(
            f"Optimization complete. After: {after_memory:.2f} MB, "
            f"Reduction: {reduction:.1f}%"
        )

        return df

    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict:
        """
        获取 DataFrame 内存使用情况

        Args:
            df: DataFrame

        Returns:
            内存使用信息
        """
        total = df.memory_usage(deep=True).sum() / 1024 / 1024

        return {
            'total_mb': total,
            'columns': {
                col: {
                    'type': str(df[col].dtype),
                    'size_mb': df[col].memory_usage(deep=True) / 1024 / 1024,
                    'percentage': df[col].memory_usage(deep=True) / df.memory_usage(deep=True).sum() * 100
                }
                for col in df.columns
            }
        }
