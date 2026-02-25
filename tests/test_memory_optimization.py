"""内存优化测试

测试内存优化功能
"""

import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaquant.data_providers.chunked_loader import (
    ChunkedDataLoader,
    ChunkedProcessor,
    MemoryEfficientLoader
)
from alphaquant.utils.memory_profiler import (
    MemoryProfiler,
    memory_profile,
    track_memory,
    MemoryLeakDetector,
    DataFrameMemoryOptimizer
)


def generate_test_dataframe(rows: int = 100000) -> pd.DataFrame:
    """
    生成测试 DataFrame

    Args:
        rows: 行数

    Returns:
        测试 DataFrame
    """
    np.random.seed(42)

    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=rows),
        'stock_id': np.random.choice(['600000', '600001', '600002'], rows),
        'open': np.random.rand(rows) * 100 + 50,
        'high': np.random.rand(rows) * 100 + 50,
        'low': np.random.rand(rows) * 100 + 50,
        'close': np.random.rand(rows) * 100 + 50,
        'volume': np.random.randint(1000000, 10000000, rows),
        'amount': np.random.rand(rows) * 1000000000,
    })


def test_chunked_loader():
    """测试分块加载器"""
    print("\n" + "=" * 70)
    print("Test 1: ChunkedDataLoader")
    print("=" * 70)

    # 生成测试数据
    df = generate_test_dataframe(rows=50000)

    # 保存到 CSV
    csv_path = './test_chunk_data.csv'
    df.to_csv(csv_path, index=False)

    # 分块加载
    loader = ChunkedDataLoader(chunk_size=10000)

    with memory_profile("Chunked CSV Loading"):
        chunks = []
        for chunk in loader.load_csv_chunks(csv_path):
            chunks.append(chunk)

        # 合并
        loaded_df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(loaded_df)} rows")

    # 清理
    import os
    os.remove(csv_path)


def test_memory_efficient_loader():
    """测试内存高效加载器"""
    print("\n" + "=" * 70)
    print("Test 2: MemoryEfficientLoader")
    print("=" * 70)

    # 生成测试数据
    df = generate_test_dataframe(rows=50000)

    # 保存到 CSV
    csv_path = './test_memory_efficient.csv'
    df.to_csv(csv_path, index=False)

    # 原始加载
    print("\nOriginal loading:")
    profiler1 = MemoryProfiler()
    profiler1.set_baseline()

    df_original = pd.read_csv(csv_path)
    profiler1.snapshot("Original Loading")

    print(f"Memory: {profiler1.get_summary()['total_growth_mb']:.2f} MB")
    print(f"Dtypes:\n{df_original.dtypes}")

    # 内存优化加载
    print("\nMemory efficient loading:")
    profiler2 = MemoryProfiler()
    profiler2.set_baseline()

    loader = MemoryEfficientLoader(chunk_size=10000, optimize_dtypes=True)

    chunks = []
    for chunk in loader.load_csv_memory_efficient(csv_path):
        chunks.append(chunk)

    df_optimized = pd.concat(chunks, ignore_index=True)
    profiler2.snapshot("Memory Efficient Loading")

    print(f"Memory: {profiler2.get_summary()['total_growth_mb']:.2f} MB")
    print(f"Dtypes:\n{df_optimized.dtypes}")

    # 对比
    print(f"\nMemory reduction: "
          f"{(1 - profiler2.get_summary()['total_growth_mb'] / profiler1.get_summary()['total_growth_mb']) * 100:.1f}%")

    # 清理
    import os
    os.remove(csv_path)


def test_dataframe_memory_optimizer():
    """测试 DataFrame 内存优化器"""
    print("\n" + "=" * 70)
    print("Test 3: DataFrameMemoryOptimizer")
    print("=" * 70)

    # 生成测试数据
    df = generate_test_dataframe(rows=50000)

    # 获取优化前的内存
    optimizer = DataFrameMemoryOptimizer()
    memory_before = optimizer.get_memory_usage(df)

    print(f"\nMemory Before Optimization:")
    print(f"  Total: {memory_before['total_mb']:.2f} MB")
    print(f"  Columns:")
    for col, info in memory_before['columns'].items():
        print(f"    {col}: {info['type']}, {info['size_mb']:.2f} MB ({info['percentage']:.1f}%)")

    # 优化
    df_optimized = optimizer.reduce_memory(df)

    # 获取优化后的内存
    memory_after = optimizer.get_memory_usage(df_optimized)

    print(f"\nMemory After Optimization:")
    print(f"  Total: {memory_after['total_mb']:.2f} MB")
    print(f"  Columns:")
    for col, info in memory_after['columns'].items():
        print(f"    {col}: {info['type']}, {info['size_mb']:.2f} MB ({info['percentage']:.1f}%)")

    # 计算节省
    reduction = (memory_before['total_mb'] - memory_after['total_mb']) / memory_before['total_mb'] * 100
    print(f"\nMemory Reduction: {reduction:.1f}%")


def test_memory_profiler():
    """测试内存分析器"""
    print("\n" + "=" * 70)
    print("Test 4: MemoryProfiler")
    print("=" * 70)

    profiler = MemoryProfiler()
    profiler.set_baseline()

    # 操作 1
    df1 = generate_test_dataframe(rows=10000)
    profiler.snapshot("After loading 10K rows")

    # 操作 2
    df2 = generate_test_dataframe(rows=20000)
    profiler.snapshot("After loading 20K rows")

    # 操作 3
    df3 = pd.concat([df1, df2], ignore_index=True)
    profiler.snapshot("After concatenation")

    # 打印摘要
    profiler.print_summary()


@track_memory
def test_track_memory_decorator():
    """测试内存跟踪装饰器"""
    print("\n" + "=" * 70)
    print("Test 5: @track_memory Decorator")
    print("=" * 70)

    # 加载数据
    df = generate_test_dataframe(rows=50000)

    # 计算因子
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['momentum'] = df['close'] / df['close'].shift(10) - 1

    # 测试通过验证
    assert df is not None
    assert len(df) > 0


def test_memory_leak_detector():
    """测试内存泄漏检测器"""
    print("\n" + "=" * 70)
    print("Test 6: MemoryLeakDetector")
    print("=" * 70)

    # 定义一个没有泄漏的函数
    def no_leak_function():
        df = generate_test_dataframe(rows=1000)
        result = df['close'].mean()
        del df
        return result

    # 检测
    detector = MemoryLeakDetector(iterations=10, tolerance=0.1)
    result = detector.detect(no_leak_function)

    print(f"\nMemory Leak Detection Result:")
    print(f"  Is Leaking: {result['is_leaking']}")
    print(f"  First: {result['first_mb']:.2f} MB")
    print(f"  Last: {result['last_mb']:.2f} MB")
    print(f"  Growth: {result['growth_mb']:.2f} MB ({result['growth_ratio']*100:.1f}%)")


def test_chunked_processor():
    """测试分块处理器"""
    print("\n" + "=" * 70)
    print("Test 7: ChunkedProcessor")
    print("=" * 70)

    # 生成测试数据
    df = generate_test_dataframe(rows=50000)

    # 定义处理函数
    def compute_moving_average(chunk, window=5):
        chunk['ma'] = chunk['close'].rolling(window).mean()
        return chunk

    # 原始处理
    print("\nOriginal processing:")
    profiler1 = MemoryProfiler()
    profiler1.set_baseline()

    result1 = compute_moving_average(df.copy())
    profiler1.snapshot("Original")

    print(f"Memory: {profiler1.get_summary()['total_growth_mb']:.2f} MB")

    # 分块处理
    print("\nChunked processing:")
    profiler2 = MemoryProfiler()
    profiler2.set_baseline()

    processor = ChunkedProcessor(chunk_size=10000)
    result2 = processor.process_dataframe(df.copy(), compute_moving_average)
    profiler2.snapshot("Chunked")

    print(f"Memory: {profiler2.get_summary()['total_growth_mb']:.2f} MB")

    # 对比
    original_growth = profiler1.get_summary()['total_growth_mb']
    chunked_growth = profiler2.get_summary()['total_growth_mb']

    if original_growth > 0:
        reduction = (1 - chunked_growth / original_growth) * 100
        print(f"\nMemory Reduction: {reduction:.1f}%")
    else:
        print(f"\nBoth methods used minimal memory")
        print(f"Original: {original_growth:.2f} MB, Chunked: {chunked_growth:.2f} MB")


def main():
    """主测试函数"""
    print("=" * 70)
    print("AlphaGPT Memory Optimization Tests")
    print("=" * 70)

    # 运行测试
    test_chunked_loader()
    test_memory_efficient_loader()
    test_dataframe_memory_optimizer()
    test_memory_profiler()
    test_track_memory_decorator()
    test_memory_leak_detector()
    test_chunked_processor()

    print("\n" + "=" * 70)
    print("All Memory Optimization Tests Completed")
    print("=" * 70)


if __name__ == '__main__':
    main()
