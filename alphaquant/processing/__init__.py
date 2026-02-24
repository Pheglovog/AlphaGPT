"""
并行处理模块

提供多进程、多线程、异步 I/O 等并行处理能力
"""

from .parallel_processor import (
    ParallelFactorCalculator,
    AsyncDataLoader,
    ThreadPoolManager,
    ParallelProcessor,
    TaskQueue,
    create_parallel_processor,
    parallel_map,
)

__all__ = [
    'ParallelFactorCalculator',
    'AsyncDataLoader',
    'ThreadPoolManager',
    'ParallelProcessor',
    'TaskQueue',
    'create_parallel_processor',
    'parallel_map',
]
