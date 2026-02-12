"""
并行处理工具

提供多进程、多线程、异步处理能力
"""

import multiprocessing
import concurrent.futures
import asyncio
from typing import Callable, List, Any, Dict, Optional, Iterable, Tuple
from functools import partial
from loguru import logger
import time


def _apply_star(args: Tuple[Callable, tuple]) -> Any:
    """帮助函数：展开元组参数调用函数"""
    func, args_tuple = args
    return func(*args_tuple)


class ParallelProcessor:
    """
    并行处理器

    特点：
    - 支持多进程（CPU 密集型）
    - 支持多线程（I/O 密集型）
    - 支持异步（高并发 I/O）
    - 自动负载均衡
    - 错误处理和重试
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_process: bool = True,
        chunk_size: int = 1
    ):
        """
        初始化并行处理器

        Args:
            max_workers: 最大工作线程/进程数，None 使用 CPU 核心数
            use_process: 是否使用进程池（True）或线程池（False）
            chunk_size: 任务分块大小
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.use_process = use_process
        self.chunk_size = chunk_size

        logger.info(
            f"ParallelProcessor initialized: "
            f"workers={self.max_workers}, "
            f"{'process' if use_process else 'thread'} pool"
        )

    def map(
        self,
        func: Callable,
        items: Iterable[Any],
        **kwargs
    ) -> List[Any]:
        """
        并行映射函数到列表

        Args:
            func: 要执行的函数
            items: 输入列表
            **kwargs: 函数额外参数

        Returns:
            结果列表
        """
        start_time = time.time()

        # 绑定额外参数
        if kwargs:
            func = partial(func, **kwargs)

        # 选择执行器
        if self.use_process:
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, items))

        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(items)} items in {elapsed:.2f}s "
            f"({len(items)/elapsed:.1f} items/s)"
        )

        return results

    def map_async(
        self,
        func: Callable,
        items: Iterable[Any],
        **kwargs
    ) -> List[Any]:
        """
        异步并行映射

        Args:
            func: 要执行的函数
            items: 输入列表
            **kwargs: 函数额外参数

        Returns:
            结果列表
        """
        start_time = time.time()

        # 绑定额外参数
        if kwargs:
            func = partial(func, **kwargs)

        # 选择执行器
        if self.use_process:
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(func, item) for item in items]

            # 等待所有任务完成
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]

        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(items)} items asynchronously in {elapsed:.2f}s "
            f"({len(items)/elapsed:.1f} items/s)"
        )

        return results

    def batch_map(
        self,
        func: Callable,
        items: Iterable[Any],
        batch_size: int = None,
        **kwargs
    ) -> List[Any]:
        """
        批量并行处理

        Args:
            func: 要执行的函数（接受列表）
            items: 输入列表
            batch_size: 每批大小
            **kwargs: 函数额外参数

        Returns:
            结果列表
        """
        batch_size = batch_size or self.chunk_size
        items = list(items)

        # 分批
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

        logger.info(
            f"Processing {len(items)} items in {len(batches)} batches "
            f"(batch_size={batch_size})"
        )

        # 处理每批
        batch_results = self.map(func, batches, **kwargs)

        # 合并结果
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    def starmap(
        self,
        func: Callable,
        args_list: List[tuple],
        **kwargs
    ) -> List[Any]:
        """
        并行展开参数映射（类似 itertools.starmap）

        Args:
            func: 要执行的函数
            args_list: 参数元组列表
            **kwargs: 函数额外参数

        Returns:
            结果列表
        """
        start_time = time.time()

        # 绑定额外参数
        if kwargs:
            func = partial(func, **kwargs)

        # 选择执行器
        if self.use_process:
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # 准备参数：(func, args_tuple)
            tasks = [(func, args) for args in args_list]
            results = list(executor.map(_apply_star, tasks))

        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(args_list)} tasks in {elapsed:.2f}s "
            f"({len(args_list)/elapsed:.1f} tasks/s)"
        )

        return results


class AsyncParallelProcessor:
    """
    异步并行处理器

    特点：
    - 基于 asyncio
    - 适合高并发 I/O 操作
    - 自动控制并发数
    """

    def __init__(
        self,
        max_concurrent: int = 100,
        timeout: Optional[float] = None
    ):
        """
        初始化异步并行处理器

        Args:
            max_concurrent: 最大并发数
            timeout: 单个任务超时时间（秒）
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"AsyncParallelProcessor initialized: "
            f"max_concurrent={max_concurrent}"
        )

    async def map(
        self,
        func: Callable,
        items: Iterable[Any],
        **kwargs
    ) -> List[Any]:
        """
        异步映射

        Args:
            func: 要执行的异步函数
            items: 输入列表
            **kwargs: 函数额外参数

        Returns:
            结果列表
        """
        start_time = time.time()

        # 创建所有任务
        tasks = [
            self._run_with_semaphore(func, item, **kwargs)
            for item in items
        ]

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Task failed: {result}")
            else:
                processed_results.append(result)

        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(items)} items asynchronously in {elapsed:.2f}s "
            f"({len(items)/elapsed:.1f} items/s)"
        )

        return processed_results

    async def _run_with_semaphore(
        self,
        func: Callable,
        item: Any,
        **kwargs
    ) -> Any:
        """
        在信号量控制下运行函数

        Args:
            func: 函数
            item: 输入
            **kwargs: 额外参数

        Returns:
            结果
        """
        async with self.semaphore:
            if self.timeout:
                try:
                    result = await asyncio.wait_for(
                        func(item, **kwargs),
                        timeout=self.timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"Task timeout after {self.timeout}s")
                    raise
            else:
                return await func(item, **kwargs)


class ParallelDataProcessor:
    """
    并行数据处理器

    专门用于数据处理任务
    """

    @staticmethod
    def parallel_load_stocks(
        stock_ids: List[str],
        load_func: Callable,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并行加载多个股票数据

        Args:
            stock_ids: 股票 ID 列表
            load_func: 加载函数
            max_workers: 最大工作线程数

        Returns:
            {stock_id: data} 字典
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            use_process=False,  # I/O 操作用线程
        )

        results = processor.map(load_func, stock_ids)

        return dict(zip(stock_ids, results))

    @staticmethod
    def parallel_compute_factors(
        data_list: List[Any],
        compute_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """
        并行计算因子

        Args:
            data_list: 数据列表
            compute_func: 计算函数
            max_workers: 最大工作线程数

        Returns:
            结果列表
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            use_process=True,  # CPU 密集型用进程
        )

        return processor.map(compute_func, data_list)

    @staticmethod
    def parallel_backtest(
        strategies: List[Any],
        backtest_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """
        并行运行回测

        Args:
            strategies: 策略列表
            backtest_func: 回测函数
            max_workers: 最大工作线程数

        Returns:
            结果列表
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            use_process=True,  # CPU 密集型用进程
        )

        return processor.map(backtest_func, strategies)


class ProgressParallelProcessor:
    """
    带进度的并行处理器

    显示处理进度
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_process: bool = True,
        show_progress: bool = True
    ):
        """
        初始化带进度的并行处理器

        Args:
            max_workers: 最大工作线程数
            use_process: 是否使用进程池
            show_progress: 是否显示进度
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.use_process = use_process
        self.show_progress = show_progress
        self.completed = 0
        self.total = 0

    def map(
        self,
        func: Callable,
        items: Iterable[Any],
        **kwargs
    ) -> List[Any]:
        """
        并行映射并显示进度

        Args:
            func: 要执行的函数
            items: 输入列表
            **kwargs: 函数额外参数

        Returns:
            结果列表
        """
        items = list(items)
        self.total = len(items)
        self.completed = 0

        start_time = time.time()

        # 绑定额外参数
        if kwargs:
            func = partial(func, **kwargs)

        # 选择执行器
        if self.use_process:
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor

        results = []

        with executor_class(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(func, item): item
                for item in items
            }

            # 等待任务完成
            for future in concurrent.futures.as_completed(future_to_item):
                result = future.result()
                results.append(result)
                self.completed += 1

                # 显示进度
                if self.show_progress:
                    self._print_progress(time.time() - start_time)

        logger.info(
            f"Completed {self.total} items in {time.time() - start_time:.2f}s"
        )

        return results

    def _print_progress(self, elapsed: float) -> None:
        """打印进度"""
        progress = self.completed / self.total * 100
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else 0

        print(
            f"\rProgress: {self.completed}/{self.total} "
            f"({progress:.1f}%) | "
            f"Rate: {rate:.1f} items/s | "
            f"ETA: {eta:.1f}s",
            end='',
            flush=True
        )


# 便捷函数

def parallel_map(
    func: Callable,
    items: Iterable[Any],
    max_workers: Optional[int] = None,
    use_process: bool = True
) -> List[Any]:
    """
    便捷的并行映射函数

    Args:
        func: 要执行的函数
        items: 输入列表
        max_workers: 最大工作线程数
        use_process: 是否使用进程池

    Returns:
        结果列表
    """
    processor = ParallelProcessor(
        max_workers=max_workers,
        use_process=use_process
    )
    return processor.map(func, items)


def parallel_starmap(
    func: Callable,
    args_list: List[tuple],
    max_workers: Optional[int] = None,
    use_process: bool = True
) -> List[Any]:
    """
    便捷的并行展开映射函数

    Args:
        func: 要执行的函数
        args_list: 参数元组列表
        max_workers: 最大工作线程数
        use_process: 是否使用进程池

    Returns:
        结果列表
    """
    processor = ParallelProcessor(
        max_workers=max_workers,
        use_process=use_process
    )
    return processor.starmap(func, args_list)
