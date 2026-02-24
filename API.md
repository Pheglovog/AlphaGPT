# AlphaGPT API 文档

本文档详细说明 AlphaGPT 系统各模块的 API 接口和使用方法。

---

## 目录

- [数据提供者 (Data Providers)](#数据提供者-data-providers)
  - [TushareProProvider](#tushareproprovider)
  - [ParquetDataLoader](#parquetdataloader)
  - [ChunkedDataLoader](#chunkeddataloader)
- [数据缓存 (Data Cache)](#数据缓存-data-cache)
  - [CacheManager](#cachemanager)
  - [MemoryCache](#memorycache)
  - [RedisCache](#rediscache)
  - [FileCache](#filecache)
- [因子引擎 (Factor Engine)](#因子引擎-factor-engine)
  - [VectorizedFactors](#vectorizedfactors)
- [回测引擎 (Backtest Engine)](#回测引擎-backtest-engine)
- [策略管理器 (Strategy Manager)](#策略管理器-strategy-manager)
- [交易执行 (Execution)](#交易执行-execution)
  - [SolanaTrader](#solanatrader)
  - [QuickNodeClient](#quicknodeclient)
  - [JupiterAggregator](#jupiteraggregator)
- [并行处理 (Parallel Processing)](#并行处理-parallel-processing)
  - [ParallelProcessor](#parallelprocessor)
  - [ParallelFactorCalculator](#parallelfactorcalculator)
  - [AsyncDataLoader](#asyncdataloader)
  - [TaskQueue](#taskqueue)
- [内存优化 (Memory Optimization)](#内存优化-memory-optimization)
  - [MemoryProfiler](#memoryprofiler)
  - [MemoryLeakDetector](#memoryleakdetector)
  - [DataFrameMemoryOptimizer](#dataframememoryoptimizer)

---

## 数据提供者 (Data Providers)

### TushareProProvider

Tushare Pro 数据接口，提供中国股市历史行情、财务数据、市场信息等。

**模块**: `alphaquant.data_providers.tushare.TushareProProvider`

#### 初始化

```python
from alphaquant.data_providers.tushare import TushareProProvider

async with TushareProProvider(token="your_token", concurrent_limit=5) as provider:
    # 使用 provider
    pass
```

**参数**:
- `token` (str): Tushare Pro API Token
- `concurrent_limit` (Optional[int]): 并发限制，默认根据账号等级（免费版=2, 2000积分=5, 5000积分=10）

#### API 方法

##### `get_stock_list(exchange: str = 'SSE')`

获取股票列表

**参数**:
- `exchange` (str): 交易所（SSE=上交所, SZSE=深交所）

**返回**:
- `pd.DataFrame`: 股票列表

**示例**:
```python
stocks = await provider.get_stock_list(exchange='SSE')
print(f"上交所股票数量: {len(stocks)}")
```

##### `get_daily_quotes(ts_code: str = '', trade_date: str = '', start_date: str = '', end_date: str = '')`

获取日线行情

**参数**:
- `ts_code` (str): 股票代码（如: 000001.SZ）
- `trade_date` (str): 交易日期（YYYYMMDD）
- `start_date` (str): 开始日期
- `end_date` (str): 结束日期

**返回**:
- `pd.DataFrame`: 日线行情

**示例**:
```python
quotes = await provider.get_daily_quotes(
    ts_code='600519.SH',
    start_date='20240101',
    end_date='20240131'
)
```

##### `get_daily_quotes_batch(ts_codes: List[str], start_date: str = '', end_date: str = '')`

批量获取多只股票的日线行情（自动控制并发）

**参数**:
- `ts_codes` (List[str]): 股票代码列表
- `start_date` (str): 开始日期
- `end_date` (str): 结束日期

**返回**:
- `Dict[str, pd.DataFrame]`: {股票代码: DataFrame} 字典

**示例**:
```python
quotes_dict = await provider.get_daily_quotes_batch(
    ts_codes=['600519.SH', '000001.SZ'],
    start_date='20240101',
    end_date='20240131'
)
```

##### `get_index_quotes(index_code: str, start_date: str = '', end_date: str = '')`

获取指数行情

**参数**:
- `index_code` (str): 指数代码（如: 000001.SH=上证指数）
- `start_date` (str): 开始日期
- `end_date` (str): 结束日期

**返回**:
- `pd.DataFrame`: 指数行情

**示例**:
```python
index_quotes = await provider.get_index_quotes(
    index_code='000001.SH',
    start_date='20240101',
    end_date='20240131'
)
```

##### `get_stock_basic(ts_code: str = '')`

获取股票基本信息

**参数**:
- `ts_code` (str): 股票代码

**返回**:
- `pd.DataFrame`: 股票基本信息

**示例**:
```python
basic = await provider.get_stock_basic(ts_code='600519.SH')
```

##### `get_limit_list(trade_date: str = '')`

获取涨跌停列表

**参数**:
- `trade_date` (str): 交易日期（YYYYMMDD）

**返回**:
- `pd.DataFrame`: 涨跌停列表

**示例**:
```python
limits = await provider.get_limit_list(trade_date='20240131')
```

##### `get_money_flow(trade_date: str = '')`

获取资金流向数据

**参数**:
- `trade_date` (str): 交易日期（YYYYMMDD）

**返回**:
- `pd.DataFrame`: 资金流向数据

**示例**:
```python
money_flow = await provider.get_money_flow(trade_date='20240131')
```

### ParquetDataLoader

Parquet 数据加载器，提供更快的 I/O 操作。

**模块**: `alphaquant.data_providers.parquet_loader.ParquetDataLoader`

#### 初始化

```python
from alphaquant.data_providers.parquet_loader import ParquetDataLoader

loader = ParquetDataLoader(cache_dir="./cache")
```

**参数**:
- `cache_dir` (str): 缓存目录，默认 "./cache"

#### API 方法

##### `load_from_parquet(file_path: str, columns: Optional[List[str]] = None)`

从 Parquet 文件加载数据

**参数**:
- `file_path` (str): Parquet 文件路径
- `columns` (Optional[List[str]]): 要加载的列，None 表示加载所有列

**返回**:
- `pd.DataFrame`: 加载的数据

**示例**:
```python
data = loader.load_from_parquet('data/stock_data.parquet', columns=['close', 'volume'])
```

##### `save_to_parquet(df: pd.DataFrame, file_path: str, compression: str = 'snappy')`

保存数据到 Parquet 文件

**参数**:
- `df` (pd.DataFrame): 要保存的数据
- `file_path` (str): 输出文件路径
- `compression` (str): 压缩算法，默认 'snappy'

**返回**:
- `bool`: 是否成功

**示例**:
```python
loader.save_to_parquet(df, 'data/stock_data.parquet')
```

### ChunkedDataLoader

分块数据加载器，支持分批加载大数据集，减少内存占用。

**模块**: `alphaquant.data_providers.chunked_loader.ChunkedDataLoader`

#### 初始化

```python
from alphaquant.data_providers.chunked_loader import ChunkedDataLoader

loader = ChunkedDataLoader(
    chunk_size=10000,
    max_memory_mb=1024,
    overlap=100
)
```

**参数**:
- `chunk_size` (int): 每块的行数，默认 10000
- `max_memory_mb` (Optional[int]): 最大内存限制（MB），None 表示不限制
- `overlap` (int): 块之间重叠的行数，用于窗口计算，默认 0

#### API 方法

##### `load_csv_chunks(file_path: str, **kwargs)`

分块加载 CSV 文件

**参数**:
- `file_path` (str): 文件路径
- `**kwargs`: pd.read_csv 的参数

**返回**:
- `Iterator[pd.DataFrame]`: 数据块生成器

**示例**:
```python
for chunk in loader.load_csv_chunks('large_data.csv'):
    process_chunk(chunk)
    # chunk 会在下一次迭代时自动释放
```

##### `load_parquet_chunks(file_path: str, **kwargs)`

分块加载 Parquet 文件

**参数**:
- `file_path` (str): 文件路径
- `**kwargs`: pd.read_parquet 的参数

**返回**:
- `Iterator[pd.DataFrame]`: 数据块生成器

**示例**:
```python
for chunk in loader.load_parquet_chunks('large_data.parquet'):
    process_chunk(chunk)
```

---

## 数据缓存 (Data Cache)

### CacheConfig

缓存配置类

**模块**: `alphaquant.data_cache.CacheConfig`

#### 初始化

```python
from alphaquant.data_cache import CacheConfig

config = CacheConfig(
    cache_dir="./cache",
    max_age_hours=24,
    max_size_mb=1024,
    enable_compression=True
)
```

**参数**:
- `cache_dir` (str): 缓存目录路径
- `max_age_hours` (int): 缓存过期时间（小时），默认 24
- `max_size_mb` (int): 缓存最大大小（MB），默认 1024
- `enable_compression` (bool): 是否启用压缩，默认 True

### DataCache

数据缓存管理器

**模块**: `alphaquant.data_cache.DataCache`

#### 初始化

```python
from alphaquant.data_cache import DataCache, CacheConfig

config = CacheConfig(cache_dir="./cache")
cache = DataCache(config)
```

#### API 方法

##### `get(cache_key: str, data_type: str = "data")`

从缓存获取数据

**参数**:
- `cache_key` (str): 缓存键
- `data_type` (str): 数据类型（data, metadata），默认 "data"

**返回**:
- `Any`: 缓存的数据，如果不存在或过期则返回 None

##### `set(cache_key: str, data: Any, data_type: str = "data", metadata: Optional[Dict] = None)`

将数据存入缓存

**参数**:
- `cache_key` (str): 缓存键
- `data` (Any): 要缓存的数据
- `data_type` (str): 数据类型，默认 "data"
- `metadata` (Optional[Dict]): 元数据

##### `exists(cache_key: str, data_type: str = "data")`

检查缓存是否存在

**参数**:
- `cache_key` (str): 缓存键
- `data_type` (str): 数据类型，默认 "data"

**返回**:
- `bool`: 缓存是否存在且未过期

##### `delete(cache_key: str, data_type: str = "data")`

删除缓存

**参数**:
- `cache_key` (str): 缓存键
- `data_type` (str): 数据类型，默认 "data"

##### `clear()`

清空所有缓存

##### `get_stats()`

获取缓存统计信息

**返回**:
- `Dict`: 缓存统计信息

**示例**:
```python
stats = cache.get_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")
print(f"缓存大小: {stats['size_mb']:.2f} MB")
```

---

## 因子引擎 (Factor Engine)

### ChinaFactorEngine

中国市场因子计算引擎

**模块**: `alphaquant.factors.china_factors.ChinaFactorEngine`

#### 初始化

```python
from alphaquant.factors.china_factors import ChinaFactorEngine

engine = ChinaFactorEngine()
```

#### API 方法

##### `compute_basic_factors(raw_data: Dict[str, torch.Tensor])`

计算基础因子（6 维）

**参数**:
- `raw_data` (Dict[str, torch.Tensor]): 原始数据字典，包含：
  - `close`: 收盘价 [batch_size, sequence_length]
  - `open`: 开盘价 [batch_size, sequence_length]
  - `high`: 最高价 [batch_size, sequence_length]
  - `low`: 最低价 [batch_size, sequence_length]
  - `volume`: 成交量 [batch_size, sequence_length]

**返回**:
- `torch.Tensor`: 基础因子 [batch_size, 6, sequence_length]

**基础因子**:
1. 收益率 (Return)
2. 波动率 (Volatility)
3. 成交量变化 (Volume Change)
4. 价格动量 (Price Momentum)
5. RSI (相对强弱指标)
6. MACD

**示例**:
```python
import torch

raw_data = {
    'close': torch.rand(5, 60) * 10 + 100,
    'open': torch.rand(5, 60) * 10 + 100,
    'high': torch.rand(5, 60) * 10 + 105,
    'low': torch.rand(5, 60) * 10 + 95,
    'volume': torch.rand(5, 60) * 1000000,
}

basic_factors = engine.compute_basic_factors(raw_data)
print(f"基础因子形状: {basic_factors.shape}")  # [5, 6, 60]
```

##### `compute_advanced_factors(raw_data: Dict[str, torch.Tensor])`

计算高级因子（18 维）

**参数**:
- `raw_data` (Dict[str, torch.Tensor]): 原始数据字典

**返回**:
- `torch.Tensor`: 高级因子 [batch_size, 18, sequence_length]

**高级因子**:
1. 布林带上下轨
2. ATR (平均真实波幅)
3. KDJ 指标
4. 成交量加权平均价 (VWAP)
5. 量价相关性
6. 涨跌幅分布
7. 买卖压力
8. 趋势强度
9. 相对强度
10. 北向资金流入
11. 融资融券余额
12. 涨跌停统计
13. 板块相对强度
14. 市场情绪指标
15. 风险偏好
16. 资金流向比率
17. 换手率
18. 振幅

**示例**:
```python
advanced_factors = engine.compute_advanced_factors(raw_data)
print(f"高级因子形状: {advanced_factors.shape}")  # [5, 18, 60]
```

##### `compute_all_factors(raw_data: Dict[str, torch.Tensor])`

计算所有因子（24 维：6 基础 + 18 高级）

**参数**:
- `raw_data` (Dict[str, torch.Tensor]): 原始数据字典

**返回**:
- `torch.Tensor`: 所有因子 [batch_size, 24, sequence_length]

**示例**:
```python
all_factors = engine.compute_all_factors(raw_data)
print(f"所有因子形状: {all_factors.shape}")  # [5, 24, 60]
```

---

## 性能优化 (Performance Optimization)

### VectorizedFactors

向量化因子计算器，使用 NumPy 和 Numba 实现高性能因子计算，比传统 pandas 方法快 10x-100x。

**模块**: `alphaquant.factors.vectorized_factors.VectorizedFactors`

#### 初始化

```python
from alphaquant.factors.vectorized_factors import VectorizedFactors

# 所有方法都是静态方法，无需实例化
# 直接调用类方法即可
```

#### API 方法

##### `rolling_mean_numba(data: np.ndarray, window: int) -> np.ndarray`

使用 Numba 加速的滚动均值计算

**参数**:
- `data` (np.ndarray): 输入数组
- `window` (int): 滚动窗口大小

**返回**:
- `np.ndarray`: 滚动均值数组

**性能**: 比 pandas.rolling().mean() 快约 20-50x

**示例**:
```python
import numpy as np
from alphaquant.factors.vectorized_factors import VectorizedFactors

data = np.random.randn(10000)
ma20 = VectorizedFactors.rolling_mean_numba(data, window=20)
```

##### `rolling_std_numba(data: np.ndarray, window: int) -> np.ndarray`

使用 Numba 加速的滚动标准差计算

**参数**:
- `data` (np.ndarray): 输入数组
- `window` (int): 滚动窗口大小

**返回**:
- `np.ndarray`: 滚动标准差数组

**性能**: 比 pandas.rolling().std() 快约 15-30x

**示例**:
```python
std20 = VectorizedFactors.rolling_std_numba(data, window=20)
```

##### `ema_numba(data: np.ndarray, period: int) -> np.ndarray`

使用 Numba 加速的指数移动平均计算

**参数**:
- `data` (np.ndarray): 输入数组
- `period` (int): EMA 周期

**返回**:
- `np.ndarray`: EMA 数组

**性能**: 比 pandas.ewm().mean() 快约 10-20x

**示例**:
```python
ema12 = VectorizedFactors.ema_numba(data, period=12)
```

##### `rsi_numba(data: np.ndarray, period: int = 14) -> np.ndarray`

使用 Numba 加速的 RSI（相对强弱指标）计算

**参数**:
- `data` (np.ndarray): 价格数据
- `period` (int): RSI 周期，默认 14

**返回**:
- `np.ndarray`: RSI 数组

**性能**: 比传统实现快约 30-50x

**示例**:
```python
rsi = VectorizedFactors.rsi_numba(data, period=14)
print(f"当前 RSI: {rsi[-1]:.2f}")
```

##### `bollinger_bands_numba(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

使用 Numba 加速的布林带计算

**参数**:
- `data` (np.ndarray): 价格数据
- `period` (int): 移动平均周期，默认 20
- `std_dev` (float): 标准差倍数，默认 2.0

**返回**:
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`: (上轨, 中轨, 下轨)

**性能**: 比传统实现快约 20-40x

**示例**:
```python
upper, middle, lower = VectorizedFactors.bollinger_bands_numba(
    data, period=20, std_dev=2.0
)
```

##### `macd_numba(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]`

使用 Numba 加速的 MACD（指数平滑异同移动平均线）计算

**参数**:
- `data` (np.ndarray): 价格数据
- `fast` (int): 快速 EMA 周期，默认 12
- `slow` (int): 慢速 EMA 周期，默认 26
- `signal` (int): 信号线 EMA 周期，默认 9

**返回**:
- `Tuple[np.ndarray, np.ndarray]`: (MACD, 信号线)

**性能**: 比传统实现快约 20-40x

**示例**:
```python
macd, signal = VectorizedFactors.macd_numba(data, fast=12, slow=26, signal=9)
```

##### `momentum_numba(data: np.ndarray, period: int = 10) -> np.ndarray`

使用 Numba 加速的动量指标计算

**参数**:
- `data` (np.ndarray): 价格数据
- `period` (int): 周期，默认 10

**返回**:
- `np.ndarray`: 动量数组

**性能**: 比传统实现快约 50-100x

**示例**:
```python
momentum = VectorizedFactors.momentum_numba(data, period=10)
```

##### `batch_compute_factors(prices: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame`

批量计算多个窗口的技术指标

**参数**:
- `prices` (pd.DataFrame): 价格数据（包含 close, volume 列）
- `windows` (List[int]): 窗口大小列表，默认 [5, 10, 20, 60]

**返回**:
- `pd.DataFrame`: 包含所有技术指标的 DataFrame

**示例**:
```python
import pandas as pd

df = pd.DataFrame({
    'close': np.random.randn(1000).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 1000)
})

factors_df = VectorizedFactors.batch_compute_factors(df, windows=[5, 10, 20])
print(f"因子形状: {factors_df.shape}")  # (1000, 20+)
```

**性能对比**:
```python
import time

# Pandas 原生实现
start = time.time()
for _ in range(100):
    df['close'].rolling(20).mean()
pandas_time = time.time() - start

# Numba 加速实现
start = time.time()
for _ in range(100):
    VectorizedFactors.rolling_mean_numba(data, 20)
numba_time = time.time() - start

print(f"Pandas: {pandas_time:.2f}s")
print(f"Numba: {numba_time:.2f}s")
print(f"加速比: {pandas_time/numba_time:.1f}x")
```

---

### ParallelProcessor

并行处理器，提供多进程、多线程、异步处理能力。

**模块**: `alphaquant.utils.parallel_processor.ParallelProcessor`

#### 初始化

```python
from alphaquant.utils.parallel_processor import ParallelProcessor

# 使用进程池（CPU 密集型）
processor = ParallelProcessor(
    max_workers=4,      # 最大工作进程数，None 使用 CPU 核心数
    use_process=True,   # 使用进程池
    chunk_size=1        # 任务分块大小
)

# 使用线程池（I/O 密集型）
io_processor = ParallelProcessor(
    max_workers=8,
    use_process=False   # 使用线程池
)
```

**参数**:
- `max_workers` (Optional[int]): 最大工作线程/进程数，None 使用 CPU 核心数
- `use_process` (bool): 是否使用进程池（True）或线程池（False）
- `chunk_size` (int): 任务分块大小

#### API 方法

##### `map(func: Callable, items: Iterable[Any], **kwargs) -> List[Any]`

并行映射函数到列表

**参数**:
- `func` (Callable): 要执行的函数
- `items` (Iterable[Any]): 输入列表
- `**kwargs`: 函数额外参数

**返回**:
- `List[Any]`: 结果列表

**示例**:
```python
def process_stock(code: str, days: int = 10):
    # 模拟处理单个股票
    import time
    time.sleep(0.1)
    return f"{code}_processed_{days}"

stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']

results = processor.map(process_stock, stocks, days=10)
print(results)
# ['000001.SZ_processed_10', '000002.SZ_processed_10', ...]
```

##### `map_async(func: Callable, items: Iterable[Any], **kwargs) -> List[Any]`

异步并行映射，任务完成即返回结果

**参数**:
- `func` (Callable): 要执行的函数
- `items` (Iterable[Any]): 输入列表
- `**kwargs`: 函数额外参数

**返回**:
- `List[Any]`: 结果列表（顺序可能不同）

**示例**:
```python
results = processor.map_async(process_stock, stocks)
print(results)
```

##### `batch_map(func: Callable, items: Iterable[Any], batch_size: int = None, **kwargs) -> List[Any]`

批量并行映射，将任务分批处理以提高效率

**参数**:
- `func` (Callable): 要执行的函数
- `items` (Iterable[Any]): 输入列表
- `batch_size` (int): 批次大小，None 自动计算
- `**kwargs`: 函数额外参数

**返回**:
- `List[Any]`: 结果列表

**示例**:
```python
def process_batch(batch: list):
    # 批量处理
    return [item * 2 for item in batch]

items = list(range(100))
results = processor.batch_map(process_batch, items, batch_size=10)
```

##### `starmap(func: Callable, args_list: List[Tuple], **kwargs) -> List[Any]`

并行映射函数到参数元组列表（支持多参数）

**参数**:
- `func` (Callable): 要执行的函数
- `args_list` (List[Tuple]): 参数元组列表
- `**kwargs`: 函数额外参数

**返回**:
- `List[Any]`: 结果列表

**示例**:
```python
def calculate_sma(data: np.ndarray, window: int):
    return np.mean(data[-window:])

args_list = [
    (np.random.randn(100), 10),
    (np.random.randn(100), 20),
    (np.random.randn(100), 30),
]

results = processor.starmap(calculate_sma, args_list)
print(results)
```

#### 性能优化示例

**场景 1: 并行计算多个股票的技术指标**
```python
import numpy as np
import pandas as pd

# 模拟数据
stocks_data = {
    '000001.SZ': pd.DataFrame({'close': np.random.randn(500).cumsum() + 10}),
    '000002.SZ': pd.DataFrame({'close': np.random.randn(500).cumsum() + 20}),
    '600000.SH': pd.DataFrame({'close': np.random.randn(500).cumsum() + 5}),
    '600519.SH': pd.DataFrame({'close': np.random.randn(500).cumsum() + 1800}),
}

def calculate_indicators(stock_code: str, df: pd.DataFrame):
    """计算单只股票的技术指标"""
    close = df['close'].values
    return {
        'code': stock_code,
        'ma20': VectorizedFactors.rolling_mean_numba(close, 20)[-1],
        'rsi': VectorizedFactors.rsi_numba(close, 14)[-1],
    }

# 并行处理
processor = ParallelProcessor(max_workers=4, use_process=True)
results = processor.map(calculate_indicators, stocks_data.items())

for result in results:
    print(f"{result['code']}: MA20={result['ma20']:.2f}, RSI={result['rsi']:.2f}")
```

**场景 2: 并行下载多个股票数据**
```python
from alphaquant.data_providers.tushare import TushareProProvider

async def fetch_stock_data(code: str, provider: TushareProProvider):
    """下载单只股票数据"""
    return await provider.get_daily_quotes(ts_code=code, start_date='20240101', end_date='20240131')

async def main():
    stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']
    
    async with TushareProProvider(token="your_token") as provider:
        # 使用异步处理器
        processor = ParallelProcessor(max_workers=4, use_process=False)
        
        # 将异步函数包装为同步
        import asyncio
        async def async_wrapper(code):
            return await fetch_stock_data(code, provider)
        
        # 并行下载
        results = await asyncio.gather(*[async_wrapper(code) for code in stock_codes])
        
        for code, data in zip(stock_codes, results):
            print(f"{code}: {len(data)} 条数据")

# asyncio.run(main())
```

**场景 3: 批量回测多个策略**
```python
def backtest_strategy(strategy_params: dict, data: pd.DataFrame):
    """回测单个策略"""
    # 模拟回测
    return {
        'params': strategy_params,
        'return': np.random.uniform(-0.2, 0.5),
        'sharpe': np.random.uniform(0.5, 2.5),
    }

# 定义策略参数组合
strategy_configs = [
    {'window': 5, 'threshold': 0.02},
    {'window': 10, 'threshold': 0.03},
    {'window': 20, 'threshold': 0.05},
    {'window': 5, 'threshold': 0.05},
    {'window': 10, 'threshold': 0.08},
]

# 并行回测
results = processor.map(backtest_strategy, strategy_configs, data=test_data)

for result in results:
    print(f"{result['params']}: Return={result['return']:.2%}, Sharpe={result['sharpe']:.2f}")
```

#### 性能对比

```python
import time

# 测试数据
items = list(range(1000))

def compute_factor(n: int):
    """模拟复杂计算"""
    import math
    total = 0
    for i in range(1000):
        total += math.sqrt(n + i) * math.log(n + i + 1)
    return total

# 串行处理
start = time.time()
serial_results = [compute_factor(n) for n in items]
serial_time = time.time() - start

# 并行处理（4 进程）
processor = ParallelProcessor(max_workers=4, use_process=True)
start = time.time()
parallel_results = processor.map(compute_factor, items)
parallel_time = time.time() - start

print(f"串行: {serial_time:.2f}s")
print(f"并行: {parallel_time:.2f}s")
print(f"加速比: {serial_time/parallel_time:.1f}x")
print(f"CPU 利用率: {100 * serial_time/parallel_time / 4:.1f}%")
```

#### 注意事项

**多进程 vs 多线程**:
- **多进程** (`use_process=True`): 适合 CPU 密集型任务（数学计算、数据处理）
- **多线程** (`use_process=False`): 适合 I/O 密集型任务（网络请求、文件读写）

**数据共享**:
- 多进程之间数据不共享，需要通过返回值传递结果
- 避免传递大量数据，会影响性能

**错误处理**:
- 单个任务失败不会影响其他任务
- 失败的任务会抛出异常

---

## 并行处理 (Parallel Processing)

### ParallelProcessor

统一并行处理接口，支持多进程、多线程、异步 I/O。

**模块**: `alphaquant.processing.parallel_processor.ParallelProcessor`

#### 初始化

```python
from alphaquant.processing import ParallelProcessor

# 多进程模式（CPU 密集型）
processor = ParallelProcessor(mode='process', n_workers=4)

# 多线程模式（I/O 密集型）
processor = ParallelProcessor(mode='thread', n_workers=8)

# 异步模式（高并发网络请求）
processor = ParallelProcessor(mode='async', n_workers=10)
```

**参数**:
- `mode` (str): 并行模式
  - `'process'`: 多进程（CPU 密集型）
  - `'thread'`: 多线程（I/O 密集型）
  - `'async'`: 异步 I/O
- `n_workers` (Optional[int]): 工作进程/线程数，None 表示自动选择

#### API 方法

##### `map(func: Callable, items: List[Any], **kwargs) -> List[Any]`

并行映射

**参数**:
- `func` (Callable): 处理函数
- `items` (List[Any]): 待处理项列表
- `**kwargs`: 函数的额外参数

**返回**:
- `List[Any]`: 结果列表

**示例**:
```python
# 多进程计算
results = processor.map(calculate_factor, data_chunks, window=20)

# 多线程加载
results = processor.map(load_file, file_paths)

# 异步网络请求
results = processor.map(fetch_api, urls)
```

##### `parallel_backtest(strategies: List[Any], data: pd.DataFrame, **kwargs) -> List[Dict]`

并行回测多个策略

**参数**:
- `strategies` (List[Any]): 策略列表
- `data` (pd.DataFrame): 回测数据
- `**kwargs`: 回测参数

**返回**:
- `List[Dict]`: 回测结果列表

**示例**:
```python
strategies = [
    Strategy(param1=10),
    Strategy(param1=20),
    Strategy(param1=30),
]

results = processor.parallel_backtest(strategies, backtest_data)
best_result = max(results, key=lambda r: r['sharpe_ratio'])
```

### ParallelFactorCalculator

并行因子计算器，使用多进程并行计算因子。

**模块**: `alphaquant.processing.parallel_processor.ParallelFactorCalculator`

#### 初始化

```python
from alphaquant.processing import ParallelFactorCalculator

calculator = ParallelFactorCalculator(n_processes=4)
```

**参数**:
- `n_processes` (Optional[int]): 进程数，None 表示使用 CPU 核心数

#### API 方法

##### `calculate_factor(data: pd.DataFrame, factor_func: Callable, factor_name: str, **kwargs) -> pd.Series`

并行计算单个因子

**参数**:
- `data` (pd.DataFrame): 数据 DataFrame
- `factor_func` (Callable): 因子计算函数
- `factor_name` (str): 因子名称
- `**kwargs`: 因子函数的额外参数

**返回**:
- `pd.Series`: 因子 Series

**示例**:
```python
from alphaquant.factors import VectorizedFactors

factors = VectorizedFactors()
data['SMA_5'] = calculator.calculate_factor(
    data=data,
    factor_func=factors.calculate_sma,
    factor_name='SMA_5',
    window=5
)
```

##### `calculate_multiple_factors(data: pd.DataFrame, factor_configs: List[Dict]) -> pd.DataFrame`

并行计算多个因子

**参数**:
- `data` (pd.DataFrame): 数据 DataFrame
- `factor_configs` (List[Dict]): 因子配置列表
  ```python
  [
      {
          'name': 'SMA_5',
          'func': factors.calculate_sma,
          'params': {'window': 5}
      },
      {
          'name': 'RSI',
          'func': factors.calculate_rsi,
          'params': {}
      }
  ]
  ```

**返回**:
- `pd.DataFrame`: 包含所有因子的 DataFrame

**示例**:
```python
factor_configs = [
    {
        'name': 'SMA_5',
        'func': factors.calculate_sma,
        'params': {'window': 5}
    },
    {
        'name': 'SMA_20',
        'func': factors.calculate_sma,
        'params': {'window': 20}
    },
    {
        'name': 'RSI',
        'func': factors.calculate_rsi,
        'params': {}
    }
]

result_df = calculator.calculate_multiple_factors(data, factor_configs)
```

### AsyncDataLoader

异步数据加载器，使用异步 I/O 并行加载数据。

**模块**: `alphaquant.processing.parallel_processor.AsyncDataLoader`

#### 初始化

```python
from alphaquant.processing import AsyncDataLoader

loader = AsyncDataLoader(max_concurrent=10)
```

**参数**:
- `max_concurrent` (int): 最大并发请求数，默认 10

#### API 方法

##### `async load_multiple(load_tasks: List[Dict]) -> List[Any]`

异步加载多个数据源

**参数**:
- `load_tasks` (List[Dict]): 加载任务列表
  ```python
  [
      {
          'func': load_stock_data,
          'args': ('000001',),
          'kwargs': {'start_date': '20200101'}
      },
      ...
  ]
  ```

**返回**:
- `List[Any]`: 加载的数据列表

**示例**:
```python
import asyncio
import pandas as pd

# 定义加载任务
load_tasks = [
    {
        'func': pd.read_csv,
        'args': (f'data/stock_{i}.csv',),
        'kwargs': {}
    }
    for i in range(100)
]

# 异步加载
async def load_all():
    return await loader.load_multiple(load_tasks)

data_list = asyncio.run(load_all())
```

### TaskQueue

任务队列，支持任务优先级、依赖和重试。

**模块**: `alphaquant.processing.parallel_processor.TaskQueue`

#### 初始化

```python
from alphaquant.processing import TaskQueue

queue = TaskQueue(max_workers=4)
```

**参数**:
- `max_workers` (int): 最大工作进程数，默认 4

#### API 方法

##### `add_task(task_id: str, func: Callable, args: tuple = (), kwargs: Optional[Dict] = None, priority: int = 0, dependencies: Optional[List[str]] = None)`

添加任务到队列

**参数**:
- `task_id` (str): 任务 ID
- `func` (Callable): 任务函数
- `args` (tuple): 函数参数
- `kwargs` (Optional[Dict]): 函数关键字参数
- `priority` (int): 优先级（数字越大优先级越高），默认 0
- `dependencies` (Optional[List[str]]): 依赖的任务 ID 列表

**示例**:
```python
# 添加任务
queue.add_task(
    task_id='load_data',
    func=load_data,
    args=('20200101', '20201231'),
    priority=10
)

queue.add_task(
    task_id='calculate_factors',
    func=calculate_factors,
    dependencies=['load_data'],
    priority=8
)

queue.add_task(
    task_id='run_backtest',
    func=run_backtest,
    dependencies=['calculate_factors'],
    priority=5
)
```

##### `run() -> Dict`

运行任务队列

**返回**:
- `Dict`: 任务结果字典

**示例**:
```python
results = queue.run()
print(f"成功: {len(results)}")
```

---

## 内存优化 (Memory Optimization)

### MemoryProfiler

内存分析器，监控和优化内存使用情况。

**模块**: `alphaquant.utils.memory_profiler.MemoryProfiler`

#### 初始化

```python
from alphaquant.utils.memory_profiler import MemoryProfiler

profiler = MemoryProfiler()
```

#### API 方法

##### `get_memory_stats() -> MemoryStats`

获取当前内存统计

**返回**:
- `MemoryStats`: 内存统计信息

**示例**:
```python
stats = profiler.get_memory_stats()
print(f"内存使用: {stats.rss_mb:.2f} MB")
print(f"内存占比: {stats.percent:.1f}%")
```

##### `set_baseline() -> MemoryStats`

设置基线内存

**示例**:
```python
profiler.set_baseline()
```

##### `snapshot(label: str = "") -> MemoryStats`

创建内存快照

**参数**:
- `label` (str): 快照标签

**返回**:
- `MemoryStats`: 内存统计

**示例**:
```python
profiler.set_baseline()

data = load_data()
profiler.snapshot("After loading")

result = process_data(data)
profiler.snapshot("After processing")

profiler.print_summary()
```

##### `get_summary() -> Dict`

获取内存使用摘要

**返回**:
- `Dict`: 摘要字典

**示例**:
```python
summary = profiler.get_summary()
print(f"峰值内存: {summary['peak_rss_mb']:.2f} MB")
print(f"内存增长: {summary['total_growth_mb']:+.2f} MB")
```

### MemoryLeakDetector

内存泄漏检测器。

**模块**: `alphaquant.utils.memory_profiler.MemoryLeakDetector`

#### 初始化

```python
from alphaquant.utils.memory_profiler import MemoryLeakDetector

detector = MemoryLeakDetector(iterations=10, tolerance=0.1)
```

**参数**:
- `iterations` (int): 测试迭代次数，默认 10
- `tolerance` (float): 容忍的增长比例，超过此值认为有泄漏，默认 0.1

#### API 方法

##### `detect(func: Callable, *args, **kwargs) -> Dict`

检测内存泄漏

**参数**:
- `func` (Callable): 要测试的函数
- `*args`: 函数参数
- `**kwargs`: 函数关键字参数

**返回**:
- `Dict`: 检测结果

**示例**:
```python
result = detector.detect(repeated_function)

if result['is_leaking']:
    print(f"内存泄漏: {result['growth_mb']:.2f} MB")
else:
    print("无内存泄漏")
```

### DataFrameMemoryOptimizer

DataFrame 内存优化器，优化 DataFrame 的内存使用。

**模块**: `alphaquant.utils.memory_profiler.DataFrameMemoryOptimizer`

#### API 方法

##### `reduce_memory(df: pd.DataFrame, category_threshold: int = 50) -> pd.DataFrame`

减少 DataFrame 内存占用

**参数**:
- `df` (pd.DataFrame): 待优化的 DataFrame
- `category_threshold` (int): 转换为 category 的唯一值阈值，默认 50

**返回**:
- `pd.DataFrame`: 优化后的 DataFrame

**示例**:
```python
from alphaquant.utils.memory_profiler import DataFrameMemoryOptimizer

optimizer = DataFrameMemoryOptimizer()
df = optimizer.reduce_memory(df)

print(f"内存减少: {optimizer.get_memory_reduction():.1f}%")
```

---

## 回测引擎 (Backtest Engine)

### BacktestEngine

回测引擎，支持 T+1 交易规则、涨跌停限制、交易成本、滑点模型

**模块**: `alphaquant.backtest.backtester.BacktestEngine`

#### 初始化

```python
from alphaquant.backtest.backtester import BacktestEngine

engine = BacktestEngine(
    initial_capital=1000000.0,
    commission_rate=0.0003,
    slippage_rate=0.001
)
```

**参数**:
- `initial_capital` (float): 初始资金，默认 1000000
- `commission_rate` (float): 佣金率，默认 0.0003
- `slippage_rate` (float): 滑点率，默认 0.001

#### API 方法

##### `add_data(symbol: str, data: pd.DataFrame)`

添加股票数据

**参数**:
- `symbol` (str): 股票代码
- `data` (pd.DataFrame): 股票数据，需包含列：open, high, low, close, volume

**示例**:
```python
data = pd.read_csv('600519.SH.csv')
engine.add_data('600519.SH', data)
```

##### `run(strategy: Callable, start_date: str, end_date: str)`

运行回测

**参数**:
- `strategy` (Callable): 策略函数，签名为 `strategy(symbol: str, history: pd.DataFrame) -> List[Order]`
- `start_date` (str): 开始日期
- `end_date` (str): 结束日期

**返回**:
- `Dict`: 回测结果，包含：
  - `total_return`: 总收益率
  - `annual_return`: 年化收益率
  - `sharpe_ratio`: 夏普比率
  - `max_drawdown`: 最大回撤
  - `win_rate`: 胜率
  - `profit_loss_ratio`: 盈亏比
  - `trades`: 交易记录

**示例**:
```python
def my_strategy(symbol, history):
    orders = []
    # 简单的策略逻辑
    if len(history) > 0:
        latest = history.iloc[-1]
        if latest['close'] > latest['ma20']:
            orders.append({
                'symbol': symbol,
                'action': 'buy',
                'quantity': 100,
                'price': latest['close']
            })
    return orders

results = engine.run(my_strategy, '2024-01-01', '2024-12-31')
print(f"总收益: {results['total_return']:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2f}%")
```

##### `get_positions()`

获取当前持仓

**返回**:
- `Dict[str, int]`: {股票代码: 持仓数量}

##### `get_cash()`

获取当前现金

**返回**:
- `float`: 当前现金

##### `get_total_value()`

获取总资产

**返回**:
- `float`: 总资产（现金 + 持仓市值）

---

## 策略管理器 (Strategy Manager)

### StrategyManager

策略管理器，用于实盘模拟和策略执行

**模块**: `alphaquant.strategy.manager.StrategyManager`

#### 初始化

```python
from alphaquant.strategy.manager import StrategyManager

manager = StrategyManager(
    config={
        'max_positions': 5,
        'buy_threshold': 0.7,
        'sell_threshold': 0.3,
        'stop_loss': 0.05,
        'take_profit': 0.10
    }
)
```

**参数**:
- `config` (Dict): 策略配置
  - `max_positions` (int): 最大持仓数
  - `buy_threshold` (float): 买入阈值
  - `sell_threshold` (float): 卖出阈值
  - `stop_loss` (float): 止损比例
  - `take_profit` (float): 止盈比例

#### API 方法

##### `generate_signals(factor_scores: Dict[str, float])`

根据因子分数生成交易信号

**参数**:
- `factor_scores` (Dict[str, float]): {股票代码: 因子分数}

**返回**:
- `Dict[str, str]`: {股票代码: 信号}（buy/hold/sell）

**示例**:
```python
factor_scores = {
    '600519.SH': 0.8,
    '000001.SZ': 0.6,
    '000002.SZ': 0.2
}

signals = manager.generate_signals(factor_scores)
print(signals)
# 输出: {'600519.SH': 'buy', '000001.SZ': 'hold', '000002.SZ': 'sell'}
```

##### `calculate_position_size(signal: str, score: float, total_capital: float)`

计算仓位大小

**参数**:
- `signal` (str): 交易信号（buy/hold/sell）
- `score` (float): 因子分数
- `total_capital` (float): 总资金

**返回**:
- `float`: 仓位大小（资金金额）

##### `update_position(symbol: str, action: str, quantity: int, price: float)`

更新持仓

**参数**:
- `symbol` (str): 股票代码
- `action` (str): 动作（buy/sell）
- `quantity` (int): 数量
- `price` (float): 价格

---

## 交易执行 (Execution)

### SolanaTrader

Solana 交易器，支持 Jupiter 聚合器进行代币交换

**模块**: `execution.trader.SolanaTrader`

#### 初始化

```python
from execution.trader import SolanaTrader

trader = SolanaTrader()
```

#### API 方法

##### `async buy(token_address: str, amount_sol: float, slippage_bps=500)`

买入代币

**参数**:
- `token_address` (str): 目标代币地址
- `amount_sol` (float): 使用的 SOL 数量
- `slippage_bps` (int): 滑点容忍度（基点），默认 500 (5%)

**返回**:
- `bool`: 交易是否成功

**示例**:
```python
success = await trader.buy(
    token_address="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    amount_sol=0.1,
    slippage_bps=500
)
```

##### `async sell(token_address: str, percentage: float = 1.0, slippage_bps=500)`

卖出代币

**参数**:
- `token_address` (str): 目标代币地址
- `percentage` (float): 卖出比例（0-1），默认 1.0（全部）
- `slippage_bps` (int): 滑点容忍度（基点），默认 500

**返回**:
- `bool`: 交易是否成功

**示例**:
```python
success = await trader.sell(
    token_address="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    percentage=0.5,
    slippage_bps=500
)
```

##### `async close()`

关闭交易器，清理资源

### QuickNodeClient

QuickNode RPC 客户端

**模块**: `execution.rpc_handler.QuickNodeClient`

#### 初始化

```python
from execution.rpc_handler import QuickNodeClient

client = QuickNodeClient()
```

#### API 方法

##### `async get_balance()`

获取钱包余额

**返回**:
- `float`: SOL 余额

##### `async send_and_confirm(transaction)`

发送并确认交易

**参数**:
- `transaction`: Solana 交易对象

**返回**:
- `str`: 交易签名

### JupiterAggregator

Jupiter 聚合器客户端

**模块**: `execution.jupiter.JupiterAggregator`

#### API 方法

##### `async get_quote(input_mint: str, output_mint: str, amount_integer: int, slippage_bps: int)`

获取报价

**参数**:
- `input_mint` (str): 输入代币 Mint 地址
- `output_mint` (str): 输出代币 Mint 地址
- `amount_integer` (int): 数量（整数）
- `slippage_bps` (int): 滑点容忍度

**返回**:
- `Dict`: 报价信息

##### `async get_swap_tx(quote: Dict)`

获取交换交易

**参数**:
- `quote` (Dict): 报价信息

**返回**:
- `str`: Base64 编码的交易

##### `deserialize_and_sign(b64_tx: str)`

反序列化并签名交易

**参数**:
- `b64_tx` (str): Base64 编码的交易

**返回**:
- `Transaction`: 已签名的交易对象

---

## 错误处理

所有模块都使用 `loguru` 进行日志记录，并实现了重试机制：

- **网络请求失败**: 自动重试 3 次，指数退避
- **API 限流 (429)**: 自动等待后重试
- **交易失败**: 记录错误日志

**示例日志**:
```
2024-02-08 06:00:00 | INFO     | Tushare stats: 100 requests in 5.0s (20.00 req/s)
2024-02-08 06:00:01 | WARNING  | Tushare rate limit hit (429), retrying in 2s (attempt 1/4)
2024-02-08 06:00:03 | SUCCESS  | BUY Successful: DezXAZ... | Tx: 5xK7s...
```

---

## 完整示例

```python
import asyncio
import torch
import pandas as pd
from alphaquant.data_providers.tushare import TushareProProvider
from alphaquant.factors.china_factors import ChinaFactorEngine
from alphaquant.backtest.backtester import BacktestEngine

async def main():
    # 1. 获取数据
    async with TushareProProvider(token="your_token", concurrent_limit=5) as provider:
        quotes = await provider.get_daily_quotes(
            ts_code='600519.SH',
            start_date='20240101',
            end_date='20240131'
        )
    
    # 2. 计算因子
    engine = ChinaFactorEngine()
    raw_data = {
        'close': torch.tensor([quotes['close'].values]).T,
        'open': torch.tensor([quotes['open'].values]).T,
        'high': torch.tensor([quotes['high'].values]).T,
        'low': torch.tensor([quotes['low'].values]).T,
        'volume': torch.tensor([quotes['volume'].values]).T,
    }
    factors = engine.compute_all_factors(raw_data)
    
    # 3. 回测
    backtest_engine = BacktestEngine(initial_capital=1000000.0)
    backtest_engine.add_data('600519.SH', quotes)
    
    def simple_strategy(symbol, history):
        orders = []
        if len(history) > 20:
            latest = history.iloc[-1]
            ma20 = history['close'].tail(20).mean()
            if latest['close'] > ma20:
                orders.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': 100,
                    'price': latest['close']
                })
            elif latest['close'] < ma20:
                orders.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': 100,
                    'price': latest['close']
                })
        return orders
    
    results = backtest_engine.run(simple_strategy, '2024-01-01', '2024-01-31')
    print(f"总收益: {results['total_return']:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 更多信息

- [项目主页](https://github.com/Pheglovog/AlphaGPT)
- [中文文档](README_CN.md)
- [OpenSpec 规范](OPENSPEC.md)
- [Tushare 文档](https://tushare.pro/document/2)

---

**最后更新**: 2026-02-21
**版本**: v1.1.0

**更新日志**:
- v1.1.0 (2026-02-21):
  - 新增 ParquetDataLoader - Parquet 数据加载器
  - 新增 ChunkedDataLoader - 分块数据加载器
  - 新增 VectorizedFactors - 向量化因子计算器
  - 新增 ParallelProcessor - 并行处理接口
  - 新增 ParallelFactorCalculator - 并行因子计算器
  - 新增 AsyncDataLoader - 异步数据加载器
  - 新增 TaskQueue - 任务队列
  - 新增 MemoryProfiler - 内存分析器
  - 新增 MemoryLeakDetector - 内存泄漏检测器
  - 新增 DataFrameMemoryOptimizer - DataFrame 优化器
- v1.0.0 (2026-02-08):
  - 初始版本
