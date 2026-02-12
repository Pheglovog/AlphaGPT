"""
分块数据加载器

支持分批加载数据，减少内存占用
"""

import pandas as pd
import numpy as np
from typing import Iterator, Optional, List, Tuple
from pathlib import Path
from loguru import logger


class ChunkedDataLoader:
    """
    分块数据加载器

    特点：
    - 分批加载大数据集
    - 使用生成器减少内存占用
    - 支持多种数据源（CSV, Parquet）
    - 自动分块大小调整
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        max_memory_mb: Optional[int] = 1024,
        overlap: int = 0
    ):
        """
        初始化分块加载器

        Args:
            chunk_size: 每块的行数
            max_memory_mb: 最大内存限制（MB），None 表示不限制
            overlap: 块之间重叠的行数（用于窗口计算）
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.overlap = overlap

    def load_csv_chunks(
        self,
        file_path: str,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        分块加载 CSV 文件

        Args:
            file_path: 文件路径
            **kwargs: pd.read_csv 的参数

        Yields:
            数据块
        """
        logger.info(f"Loading CSV in chunks: {file_path}, chunk_size={self.chunk_size}")

        # 处理重叠
        if self.overlap > 0:
            previous_tail = None

        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, **kwargs):
            # 添加重叠部分
            if self.overlap > 0 and previous_tail is not None:
                chunk = pd.concat([previous_tail, chunk], ignore_index=True)
                # 保留当前块的尾部用于下一次迭代
                previous_tail = chunk.iloc[-self.overlap:].copy()
                # 移除当前块末尾的重叠部分
                chunk = chunk.iloc[:-self.overlap].copy()
            elif self.overlap > 0:
                previous_tail = chunk.iloc[-self.overlap:].copy()
                chunk = chunk.iloc[:-self.overlap].copy()

            # 检查内存使用
            if self.max_memory_mb is not None:
                mem_mb = chunk.memory_usage(deep=True).sum() / 1024 / 1024
                if mem_mb > self.max_memory_mb:
                    logger.warning(
                        f"Chunk memory ({mem_mb:.2f} MB) exceeds limit ({self.max_memory_mb} MB), "
                        f"consider reducing chunk_size"
                    )

            yield chunk

    def load_parquet_chunks(
        self,
        file_path: str,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        分块加载 Parquet 文件

        Args:
            file_path: 文件路径
            **kwargs: pd.read_parquet 的参数

        Yields:
            数据块
        """
        logger.info(f"Loading Parquet in chunks: {file_path}, chunk_size={self.chunk_size}")

        # Parquet 不原生支持分块，需要先读取总行数
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows

        # 分块读取
        offset = 0
        while offset < total_rows:
            # 计算读取范围
            end = min(offset + self.chunk_size, total_rows)

            # 读取指定范围
            chunk = pd.read_parquet(
                file_path,
                **kwargs
            ).iloc[offset:end].copy()

            # 检查内存使用
            if self.max_memory_mb is not None:
                mem_mb = chunk.memory_usage(deep=True).sum() / 1024 / 1024
                if mem_mb > self.max_memory_mb:
                    logger.warning(
                        f"Chunk memory ({mem_mb:.2f} MB) exceeds limit ({self.max_memory_mb} MB)"
                    )

            yield chunk

            offset = end

    def load_tushare_chunks(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        tushare_pro,
        chunk_days: int = 365,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        分块加载 Tushare 数据

        Args:
            stock_id: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            tushare_pro: Tushare Pro 实例
            chunk_days: 每块的天数
            **kwargs: Tushare API 参数

        Yields:
            数据块
        """
        logger.info(
            f"Loading Tushare data in chunks: {stock_id}, "
            f"{start_date} to {end_date}, chunk_days={chunk_days}"
        )

        # 分日期分块
        from datetime import datetime, timedelta

        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')

        current_start = start
        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_days), end)

            chunk_start = current_start.strftime('%Y%m%d')
            chunk_end = current_end.strftime('%Y%m%d')

            # 获取数据
            chunk = tushare_pro.daily(
                ts_code=stock_id,
                start_date=chunk_start,
                end_date=chunk_end,
                **kwargs
            )

            if not chunk.empty:
                # 检查内存使用
                if self.max_memory_mb is not None:
                    mem_mb = chunk.memory_usage(deep=True).sum() / 1024 / 1024
                    if mem_mb > self.max_memory_mb:
                        logger.warning(
                            f"Chunk memory ({mem_mb:.2f} MB) exceeds limit "
                            f"({self.max_memory_mb} MB)"
                        )

                yield chunk

            current_start = current_end + timedelta(days=1)


class ChunkedProcessor:
    """
    分块处理器

    支持对大数据集进行分块处理，减少内存占用
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        aggregation_func: Optional[callable] = None
    ):
        """
        初始化分块处理器

        Args:
            chunk_size: 每块的处理大小
            aggregation_func: 聚合函数，用于合并各块结果
        """
        self.chunk_size = chunk_size
        self.aggregation_func = aggregation_func or self._default_aggregation

    def process_dataframe(
        self,
        df: pd.DataFrame,
        process_func: callable,
        **kwargs
    ) -> pd.DataFrame:
        """
        分块处理 DataFrame

        Args:
            df: 待处理的 DataFrame
            process_func: 处理函数，接受一个 DataFrame，返回处理后的结果
            **kwargs: process_func 的额外参数

        Returns:
            处理后的 DataFrame（合并所有块的结果）
        """
        logger.info(f"Processing DataFrame in chunks: {len(df)} rows, chunk_size={self.chunk_size}")

        results = []

        # 分块处理
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size].copy()

            # 处理当前块
            result = process_func(chunk, **kwargs)

            if result is not None:
                results.append(result)

        # 合并结果
        if results:
            return self.aggregation_func(results)

        return pd.DataFrame()

    @staticmethod
    def _default_aggregation(results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        默认聚合函数：连接所有结果

        Args:
            results: 结果列表

        Returns:
            合并后的 DataFrame
        """
        return pd.concat(results, ignore_index=True)

    def process_with_memory_limit(
        self,
        df: pd.DataFrame,
        process_func: callable,
        max_memory_mb: int = 1024,
        **kwargs
    ) -> pd.DataFrame:
        """
        在内存限制下分块处理

        Args:
            df: 待处理的 DataFrame
            process_func: 处理函数
            max_memory_mb: 最大内存限制（MB）
            **kwargs: process_func 的额外参数

        Returns:
            处理后的 DataFrame
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())

        logger.info(
            f"Processing DataFrame with memory limit: {max_memory_mb} MB"
        )

        results = []
        current_chunk_size = self.chunk_size

        for i in range(0, len(df), current_chunk_size):
            chunk = df.iloc[i:i + current_chunk_size].copy()

            # 处理当前块
            result = process_func(chunk, **kwargs)

            if result is not None:
                results.append(result)

            # 检查内存使用
            mem_mb = process.memory_info().rss / 1024 / 1024
            if mem_mb > max_memory_mb:
                logger.warning(
                    f"Memory usage ({mem_mb:.2f} MB) exceeds limit ({max_memory_mb} MB), "
                    f"reducing chunk_size"
                )
                # 减小块大小
                current_chunk_size = max(current_chunk_size // 2, 100)

        # 合并结果
        if results:
            return self.aggregation_func(results)

        return pd.DataFrame()


class MemoryEfficientLoader:
    """
    内存高效数据加载器

    结合多种优化技术：
    - 分块加载
    - 生成器
    - 数据类型优化
    - 及时释放
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        optimize_dtypes: bool = True,
        max_memory_mb: Optional[int] = 1024
    ):
        """
        初始化内存高效加载器

        Args:
            chunk_size: 每块的行数
            optimize_dtypes: 是否优化数据类型
            max_memory_mb: 最大内存限制
        """
        self.chunk_size = chunk_size
        self.optimize_dtypes = optimize_dtypes
        self.max_memory_mb = max_memory_mb

        self.chunked_loader = ChunkedDataLoader(
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb
        )

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        优化 DataFrame 的数据类型以减少内存占用

        Args:
            df: 待优化的 DataFrame

        Returns:
            优化后的 DataFrame
        """
        for col in df.columns:
            col_type = df[col].dtype

            # 整数类型优化
            if col_type == 'int64':
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min >= 0:
                    if col_max < 256:
                        df[col] = df[col].astype('uint8')
                    elif col_max < 65536:
                        df[col] = df[col].astype('uint16')
                    elif col_max < 4294967296:
                        df[col] = df[col].astype('uint32')
                else:
                    if col_min >= -128 and col_max < 128:
                        df[col] = df[col].astype('int8')
                    elif col_min >= -32768 and col_max < 32768:
                        df[col] = df[col].astype('int16')
                    elif col_min >= -2147483648 and col_max < 2147483648:
                        df[col] = df[col].astype('int32')

            # 浮点类型优化
            elif col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    def load_csv_memory_efficient(
        self,
        file_path: str,
        optimize: bool = None,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        内存高效加载 CSV

        Args:
            file_path: 文件路径
            optimize: 是否优化数据类型
            **kwargs: pd.read_csv 的参数

        Yields:
            优化后的数据块
        """
        if optimize is None:
            optimize = self.optimize_dtypes

        for chunk in self.chunked_loader.load_csv_chunks(file_path, **kwargs):
            if optimize:
                chunk = MemoryEfficientLoader.optimize_dtypes(chunk)
            yield chunk

    def load_parquet_memory_efficient(
        self,
        file_path: str,
        optimize: bool = None,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        内存高效加载 Parquet

        Args:
            file_path: 文件路径
            optimize: 是否优化数据类型
            **kwargs: pd.read_parquet 的参数

        Yields:
            优化后的数据块
        """
        if optimize is None:
            optimize = self.optimize_dtypes

        for chunk in self.chunked_loader.load_parquet_chunks(file_path, **kwargs):
            if optimize:
                chunk = MemoryEfficientLoader.optimize_dtypes(chunk)
            yield chunk
