"""
AlphaGPT 并行处理器

功能：
1. 多进程因子计算
2. 多线程数据加载
3. 异步 I/O 操作
4. 任务队列管理
5. 资源池控制

作者：吕布（上等兵•甘的 AI 助手）
日期：2026-02-21
"""

import multiprocessing as mp
import asyncio
import concurrent.futures
from typing import Callable, List, Any, Optional, Dict
from functools import partial
from loguru import logger
import pandas as pd
import numpy as np


class ParallelFactorCalculator:
    """
    并行因子计算器

    使用多进程并行计算因子，充分利用多核 CPU
    """

    def __init__(self, n_processes: Optional[int] = None):
        """
        初始化并行因子计算器

        Args:
            n_processes: 进程数，None 表示使用 CPU 核心数
        """
        self.n_processes = n_processes or mp.cpu_count()
        logger.info(f"ParallelFactorCalculator initialized with {self.n_processes} processes")

    def calculate_factor(
        self,
        data: pd.DataFrame,
        factor_func: Callable,
        factor_name: str,
        **kwargs
    ) -> pd.Series:
        """
        并行计算单个因子

        Args:
            data: 数据 DataFrame
            factor_func: 因子计算函数
            factor_name: 因子名称
            **kwargs: 因子函数的额外参数

        Returns:
            因子 Series
        """
        logger.info(f"Calculating factor: {factor_name}")

        # 分块
        chunk_size = len(data) // self.n_processes
        chunks = []

        for i in range(self.n_processes):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.n_processes - 1 else len(data)
            chunk = data.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)

        # 并行计算
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # 部分函数
            func = partial(factor_func, **kwargs)

            # 提交任务
            futures = [executor.submit(func, chunk) for chunk in chunks]

            # 等待完成
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 合并结果
        result = pd.concat(results, ignore_index=True)

        logger.info(f"Factor {factor_name} calculation completed")

        return result

    def calculate_multiple_factors(
        self,
        data: pd.DataFrame,
        factor_configs: List[Dict]
    ) -> pd.DataFrame:
        """
        并行计算多个因子

        Args:
            data: 数据 DataFrame
            factor_configs: 因子配置列表
                [
                    {
                        'name': 'SMA_5',
                        'func': calculate_sma,
                        'params': {'window': 5}
                    },
                    ...
                ]

        Returns:
            包含所有因子的 DataFrame
        """
        logger.info(f"Calculating {len(factor_configs)} factors in parallel")

        result_df = data.copy()

        # 并行计算每个因子
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_processes) as executor:
            futures = []

            for config in factor_configs:
                future = executor.submit(
                    self.calculate_factor,
                    data=data,
                    factor_func=config['func'],
                    factor_name=config['name'],
                    **config.get('params', {})
                )
                futures.append(future)

            # 等待完成并添加到结果
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                factor_result = future.result()
                factor_name = factor_configs[i]['name']
                result_df[factor_name] = factor_result
                logger.info(f"Factor {factor_name} completed")

        logger.info("All factors calculated")

        return result_df


class AsyncDataLoader:
    """
    异步数据加载器

    使用异步 I/O 并行加载数据
    """

    def __init__(self, max_concurrent: int = 10):
        """
        初始化异步数据加载器

        Args:
            max_concurrent: 最大并发请求数
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def load_single(
        self,
        load_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        异步加载单个数据源

        Args:
            load_func: 加载函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            加载的数据
        """
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, load_func, *args, **kwargs)

    async def load_multiple(
        self,
        load_tasks: List[Dict]
    ) -> List[Any]:
        """
        异步加载多个数据源

        Args:
            load_tasks: 加载任务列表
                [
                    {
                        'func': load_stock_data,
                        'args': ('000001',),
                        'kwargs': {'start_date': '20200101'}
                    },
                    ...
                ]

        Returns:
            加载的数据列表
        """
        logger.info(f"Loading {len(load_tasks)} data sources concurrently")

        # 创建任务
        tasks = [
            self.load_single(
                task['func'],
                *task.get('args', []),
                **task.get('kwargs', {})
            )
            for task in load_tasks
        ]

        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error loading {load_tasks[i]}: {result}")

        logger.info("All data loaded")

        return results

    @staticmethod
    def run_async(coro):
        """
        运行异步协程的辅助函数

        Args:
            coro: 协程对象

        Returns:
            协程结果
        """
        return asyncio.run(coro)


class ThreadPoolManager:
    """
    线程池管理器

    管理共享线程池，避免创建过多线程
    """

    _instance = None
    _executor = None

    def __new__(cls, max_workers: Optional[int] = None):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._max_workers = max_workers or (mp.cpu_count() * 2)
        return cls._instance

    @property
    def executor(self):
        """获取线程池"""
        if self._executor is None or self._executor._shutdown:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers
            )
        return self._executor

    def submit(self, func: Callable, *args, **kwargs):
        """
        提交任务到线程池

        Args:
            func: 函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            Future 对象
        """
        return self.executor.submit(func, *args, **kwargs)

    def shutdown(self):
        """关闭线程池"""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None


class ParallelProcessor:
    """
    并行处理器

    统一接口，支持多种并行处理模式
    """

    def __init__(
        self,
        mode: str = 'process',
        n_workers: Optional[int] = None
    ):
        """
        初始化并行处理器

        Args:
            mode: 并行模式
                - 'process': 多进程（CPU 密集型）
                - 'thread': 多线程（I/O 密集型）
                - 'async': 异步 I/O
            n_workers: 工作进程/线程数
        """
        self.mode = mode
        self.n_workers = n_workers or (mp.cpu_count() if mode == 'process' else mp.cpu_count() * 2)

        logger.info(f"ParallelProcessor initialized: mode={mode}, workers={self.n_workers}")

    def map(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        并行映射

        Args:
            func: 处理函数
            items: 待处理项列表
            **kwargs: 函数的额外参数

        Returns:
            结果列表
        """
        logger.info(f"Processing {len(items)} items in {self.mode} mode")

        if self.mode == 'process':
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                func_with_args = partial(func, **kwargs)
                results = list(executor.map(func_with_args, items))

        elif self.mode == 'thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                func_with_args = partial(func, **kwargs)
                results = list(executor.map(func_with_args, items))

        elif self.mode == 'async':
            async_loader = AsyncDataLoader(max_concurrent=self.n_workers)

            async def async_process():
                tasks = [
                    async_loader.load_single(func, item, **kwargs)
                    for item in items
                ]
                return await asyncio.gather(*tasks)

            results = AsyncDataLoader.run_async(async_process())

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        logger.info(f"Processing completed")

        return results

    def parallel_backtest(
        self,
        strategies: List[Any],
        data: pd.DataFrame,
        **kwargs
    ) -> List[Dict]:
        """
        并行回测多个策略

        Args:
            strategies: 策略列表
            data: 回测数据
            **kwargs: 回测参数

        Returns:
            回测结果列表
        """
        logger.info(f"Running parallel backtests for {len(strategies)} strategies")

        def run_backtest(strategy):
            return strategy.run(data, **kwargs)

        results = self.map(run_backtest, strategies)

        return results


class TaskQueue:
    """
    任务队列

    支持任务优先级、重试和依赖
    """

    def __init__(self, max_workers: int = 4):
        """
        初始化任务队列

        Args:
            max_workers: 最大工作进程数
        """
        self.max_workers = max_workers
        self.tasks = []
        self.results = {}
        self.failed = {}

    def add_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        priority: int = 0,
        dependencies: Optional[List[str]] = None
    ):
        """
        添加任务到队列

        Args:
            task_id: 任务 ID
            func: 任务函数
            args: 函数参数
            kwargs: 函数关键字参数
            priority: 优先级（数字越大优先级越高）
            dependencies: 依赖的任务 ID 列表
        """
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs or {},
            'priority': priority,
            'dependencies': dependencies or [],
            'status': 'pending'
        }

        self.tasks.append(task)

    def run(self):
        """
        运行任务队列

        Returns:
            任务结果字典
        """
        logger.info(f"Running task queue with {len(self.tasks)} tasks")

        # 按优先级排序
        self.tasks.sort(key=lambda t: t['priority'], reverse=True)

        # 运行任务
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            pending_tasks = set(self.tasks)

            while pending_tasks:
                # 检查可以运行的任务
                ready_tasks = []
                for task in pending_tasks:
                    # 检查依赖是否完成
                    deps_satisfied = all(
                        dep_id in self.results or dep_id in self.failed
                        for dep_id in task['dependencies']
                    )

                    # 检查依赖是否成功
                    deps_success = all(
                        dep_id not in self.failed
                        for dep_id in task['dependencies']
                    )

                    if deps_satisfied and deps_success:
                        ready_tasks.append(task)

                if not ready_tasks:
                    logger.warning("No tasks ready to run, possible cycle")
                    break

                # 提交任务
                futures = {}
                for task in ready_tasks:
                    future = executor.submit(task['func'], *task['args'], **task['kwargs'])
                    futures[future] = task
                    task['status'] = 'running'

                # 等待完成
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]

                    try:
                        result = future.result()
                        self.results[task['id']] = result
                        task['status'] = 'completed'
                        logger.info(f"Task {task['id']} completed")
                    except Exception as e:
                        self.failed[task['id']] = e
                        task['status'] = 'failed'
                        logger.error(f"Task {task['id']} failed: {e}")

                    # 从待处理中移除
                    pending_tasks.discard(task)

        logger.info(f"Task queue completed: {len(self.results)} succeeded, {len(self.failed)} failed")

        return self.results


# 便捷工厂函数

def create_parallel_processor(
    mode: str = 'process',
    n_workers: Optional[int] = None
) -> ParallelProcessor:
    """
    创建并行处理器

    Args:
        mode: 并行模式
        n_workers: 工作进程/线程数

    Returns:
        并行处理器实例
    """
    return ParallelProcessor(mode=mode, n_workers=n_workers)


def parallel_map(
    func: Callable,
    items: List[Any],
    mode: str = 'process',
    n_workers: Optional[int] = None,
    **kwargs
) -> List[Any]:
    """
    并行映射（便捷函数）

    Args:
        func: 处理函数
        items: 待处理项列表
        mode: 并行模式
        n_workers: 工作进程/线程数
        **kwargs: 函数的额外参数

    Returns:
        结果列表
    """
    processor = ParallelProcessor(mode=mode, n_workers=n_workers)
    return processor.map(func, items, **kwargs)
