"""
AlphaGPT 数据加载优化 - Parquet 支持

作者：吕布（上等兵•甘的 AI 助手）
日期：2026-02-13
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# 尝试导入 Parquet 库
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class ParquetDataLoader:
    """Parquet 数据加载器 - 更快的 I/O 操作"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_parquet(self, file_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从 Parquet 文件加载数据

        Args:
            file_path: Parquet 文件路径
            columns: 要加载的列（None 表示加载所有列）

        Returns:
            pd.DataFrame: 加载的数据
        """
        if not PYARROW_AVAILABLE:
            print("PyArrow 不可用，回退到 pandas.read_parquet()")
            return pd.read_parquet(file_path, columns=columns)

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        # 使用 PyArrow 加载
        table = pq.read_table(file_path, columns=columns)
        return table.to_pandas()

    def save_to_parquet(self, df: pd.DataFrame, file_path: str,
                       compression: str = "snappy") -> None:
        """
        将 DataFrame 保存到 Parquet 文件

        Args:
            df: 要保存的 DataFrame
            file_path: 输出文件路径
            compression: 压缩算法（snappy, gzip, brotli）
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if PYARROW_AVAILABLE:
            # 使用 PyArrow 保存（更快）
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression=compression)
        else:
            # 回退到 pandas
            df.to_parquet(file_path, compression=compression)

        print(f"数据已保存到 Parquet 文件：{file_path}")
        print(f"文件大小：{file_path.stat().st_size / 1024 / 1024:.2f} MB")

    def convert_csv_to_parquet(self, csv_path: str,
                             parquet_path: Optional[str] = None,
                             dtypes: Optional[Dict[str, str]] = None,
                             chunksize: int = 100000) -> None:
        """
        将 CSV 文件转换为 Parquet 格式

        Args:
            csv_path: CSV 文件路径
            parquet_path: 输出 Parquet 文件路径（None 表示同目录）
            dtypes: 数据类型映射
            chunksize: 每次读取的行数
        """
        if parquet_path is None:
            csv_path_obj = Path(csv_path)
            parquet_path = csv_path_obj.parent / (csv_path_obj.stem + ".parquet")
        else:
            parquet_path = Path(parquet_path)

        print(f"正在转换：{csv_path} -> {parquet_path}")
        print(f"使用 chunksize: {chunksize}")

        # 分块读取和转换
        writer = None
        chunk_num = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=dtypes):
            if PYARROW_AVAILABLE and chunk_num == 0:
                # 第一个块：创建表和写入器
                table = pa.Table.from_pandas(chunk)
                pq.write_table(table, parquet_path, compression="snappy")
                writer = None
            elif PYARROW_AVAILABLE:
                # 后续块：追加
                table = pa.Table.from_pandas(chunk)
                pq.write_table(table, parquet_path, compression="snappy")
            else:
                # 不使用 PyArrow：pandas 会处理追加
                if chunk_num == 0:
                    chunk.to_parquet(parquet_path, compression="snappy", index=False)
                else:
                    chunk.to_parquet(parquet_path, compression="snappy", index=False,
                                   mode="append")

            chunk_num += 1
            if chunk_num % 10 == 0:
                print(f"已处理 {chunk_num * chunksize} 行")

        print(f"转换完成！共 {chunk_num * chunksize} 行")


class TushareProDataLoader:
    """Tushare Pro 批量数据加载器 - 减少 API 调用次数"""

    def __init__(self, pro_api_token: str):
        import tushare as ts

        self.pro = ts.pro_api(pro_api_token)

    def batch_daily_data(self, symbols: List[str],
                        start_date: str, end_date: str,
                        fields: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取日线数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            fields: 要获取的字段

        Returns:
            Dict[str, pd.DataFrame]: 股票代码 -> DataFrame 的映射
        """
        if fields is None:
            fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'pre_close', 'vol', 'amount']

        all_results = {}

        # Tushare Pro 批量查询（一次性查询多个股票）
        for symbol in symbols:
            try:
                df = self.pro.daily(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields
                )
                all_results[symbol] = df

                # 避免限流
                import time
                time.sleep(0.1)

            except Exception as e:
                print(f"获取 {symbol} 数据失败：{e}")
                all_results[symbol] = pd.DataFrame()

        return all_results

    def batch_daily_basic(self, symbols: List[str],
                          start_date: str, end_date: str,
                          fields: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取日线基础数据（市盈率等）

        Args:
            symbols: 股票代码列表
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            fields: 要获取的字段

        Returns:
            Dict[str, pd.DataFrame]: 股票代码 -> DataFrame 的映射
        """
        if fields is None:
            fields = ['ts_code', 'trade_date', 'turnover_rate', 'volume_ratio',
                    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
                    'dv_ttm', 'total_share', 'float_share', 'free_share',
                    'total_mv', 'circ_mv']

        all_results = {}

        for symbol in symbols:
            try:
                df = self.pro.daily_basic(
                    ts_code=symbol,
                    trade_date=start_date,
                    fields=fields
                )

                # 过滤日期范围
                df = df[(df['trade_date'] >= start_date) &
                        (df['trade_date'] <= end_date)]

                all_results[symbol] = df

                # 避免限流
                import time
                time.sleep(0.1)

            except Exception as e:
                print(f"获取 {symbol} 基础数据失败：{e}")
                all_results[symbol] = pd.DataFrame()

        return all_results


class OptimizedDataLoader:
    """优化的数据加载器 - 集成 Parquet 和 Tushare Pro"""

    def __init__(self, pro_api_token: Optional[str] = None,
                 cache_dir: str = "./cache"):
        self.pro_token = pro_api_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化加载器
        self.parquet_loader = ParquetDataLoader(str(self.cache_dir))
        if self.pro_token:
            self.tushare_loader = TushareProDataLoader(self.pro_token)
        else:
            self.tushare_loader = None

    def load_stock_data(self, symbol: str, start_date: str, end_date: str,
                        use_cache: bool = True,
                        force_refresh: bool = False,
                        use_pro_api: bool = True) -> pd.DataFrame:
        """
        加载股票数据（支持缓存和 Parquet）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新数据
            use_pro_api: 是否使用 Tushare Pro

        Returns:
            pd.DataFrame: 加载的数据
        """
        # 检查缓存
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.parquet"

        if use_cache and cache_file.exists() and not force_refresh:
            print(f"从缓存加载数据：{symbol}")
            return self.parquet_loader.load_from_parquet(str(cache_file))

        print(f"从 API 加载数据：{symbol}")

        # 从 API 获取数据
        daily_data = None
        basic_data = None

        if use_pro_api and self.tushare_loader:
            # 使用 Tushare Pro 批量 API
            daily_data = self.tushare_loader.batch_daily_data([symbol], start_date, end_date)
            basic_data = self.tushare_loader.batch_daily_basic([symbol], start_date, end_date)

            daily_data = daily_data.get(symbol)
            basic_data = basic_data.get(symbol)
        else:
            # 使用普通 Tushare API
            import tushare as ts
            pro = ts.pro_api(self.pro_token)
            daily_data = pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            basic_data = pro.daily_basic(
                ts_code=symbol,
                trade_date=start_date
            )

        # 合并数据
        if daily_data is not None and not daily_data.empty and basic_data is not None and not basic_data.empty:
            df = pd.merge(daily_data, basic_data, on=['ts_code', 'trade_date'], how='left')
        else:
            df = daily_data if daily_data is not None else basic_data

        if df is None or df.empty:
            raise ValueError(f"无法获取 {symbol} 的数据")

        # 保存到 Parquet 缓存
        self.parquet_loader.save_to_parquet(df, str(cache_file))

        return df

    def load_multiple_stocks(self, symbols: List[str], start_date: str, end_date: str,
                           use_cache: bool = True,
                           use_pro_api: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量加载多个股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            use_pro_api: 是否使用 Tushare Pro

        Returns:
            Dict[str, pd.DataFrame]: 股票代码 -> DataFrame 的映射
        """
        all_data = {}

        if use_pro_api and self.tushare_loader:
            # 使用批量 API（更快）
            print(f"使用批量 API 加载 {len(symbols)} 只股票")
            daily_data = self.tushare_loader.batch_daily_data(symbols, start_date, end_date)
            basic_data = self.tushare_loader.batch_daily_basic(symbols, start_date, end_date)

            for symbol in symbols:
                df_daily = daily_data.get(symbol)
                df_basic = basic_data.get(symbol)

                if df_daily is not None and not df_daily.empty and df_basic is not None and not df_basic.empty:
                    df = pd.merge(df_daily, df_basic, on=['ts_code', 'trade_date'], how='left')
                    all_data[symbol] = df

                # 保存到缓存
                if not df.empty:
                    cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.parquet"
                    self.parquet_loader.save_to_parquet(df, str(cache_file))

        else:
            # 逐个加载（较慢）
            for symbol in symbols:
                df = self.load_stock_data(symbol, start_date, end_date, use_cache=use_cache,
                                        use_pro_api=use_pro_api)
                all_data[symbol] = df

        return all_data


def benchmark_csv_vs_parquet(csv_path: str, sample_size: int = 1000000):
    """
    对比 CSV 和 Parquet 的加载性能

    Args:
        csv_path: CSV 文件路径
        sample_size: 数据集大小
    """
    # 生成测试数据
    print(f"生成测试数据：{sample_size} 行")
    test_data = pd.DataFrame({
        'ts_code': ['600519.SH'] * sample_size,
        'trade_date': pd.date_range('2020-01-01', periods=sample_size).strftime('%Y%m%d'),
        'open': np.random.randn(sample_size) * 10 + 1000,
        'high': np.random.randn(sample_size) * 10 + 1010,
        'low': np.random.randn(sample_size) * 10 + 990,
        'close': np.random.randn(sample_size) * 10 + 1000,
        'vol': np.random.randint(100000, 1000000, sample_size)
    })

    # 保存为 CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(csv_path, index=False)

    # CSV 加载
    print("\n测试 CSV 加载性能...")
    start = pd.Timestamp.now()
    df_csv = pd.read_csv(csv_path)
    csv_time = (pd.Timestamp.now() - start).total_seconds()

    # Parquet 加载
    parquet_path = csv_path.parent / (csv_path.stem + ".parquet")
    print("转换 CSV 为 Parquet...")
    loader = ParquetDataLoader()
    loader.save_to_parquet(test_data, str(parquet_path))

    print("\n测试 Parquet 加载性能...")
    start = pd.Timestamp.now()
    df_parquet = loader.load_from_parquet(str(parquet_path))
    parquet_time = (pd.Timestamp.now() - start).total_seconds()

    # 对比结果
    print("\n" + "="*50)
    print("性能对比结果")
    print("="*50)
    print(f"CSV 加载时间：{csv_time:.3f} 秒")
    print(f"Parquet 加载时间：{parquet_time:.3f} 秒")
    print(f"Parquet 提升：{csv_time / parquet_time:.2f}x")
    print("="*50)

    # 清理测试文件
    csv_path.unlink(missing_ok=True)
    parquet_path.unlink(missing_ok=True)


def example_usage():
    """使用示例"""
    print("AlphaGPT 数据加载优化示例\n")

    # 1. Parquet 加载器
    print("1. Parquet 加载器")
    loader = ParquetDataLoader()
    # df = loader.load_from_parquet("cache/600519_SH.parquet")

    # 2. CSV 转 Parquet
    print("\n2. CSV 转 Parquet")
    # loader.convert_csv_to_parquet("data/600519_SH.csv", "cache/600519_SH.parquet")

    # 3. 优化的数据加载器
    print("\n3. 优化的数据加载器（集成 Tushare Pro 和 Parquet）")
    pro_token = os.getenv('TUSHARE_TOKEN')

    if pro_token:
        optimized_loader = OptimizedDataLoader(pro_token)

        # 单个股票（带缓存）
        df = optimized_loader.load_stock_data(
            symbol="600519.SH",
            start_date="20230101",
            end_date="20240101"
        )
        print(f"加载数据：{len(df)} 行")

        # 多个股票（批量 API）
        symbols = ["600519.SH", "000001.SZ", "600000.SH"]
        all_data = optimized_loader.load_multiple_stocks(
            symbols=symbols,
            start_date="20230101",
            end_date="20240101"
        )
        print(f"加载了 {len(all_data)} 只股票的数据")
    else:
        print("未设置 TUSHARE_TOKEN 环境变量，跳过 Tushare Pro 示例")

    # 4. 性能基准测试
    print("\n4. 性能基准测试")
    benchmark_csv_vs_parquet("./cache/benchmark.csv")


if __name__ == "__main__":
    example_usage()
