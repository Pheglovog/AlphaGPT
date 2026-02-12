"""
并行处理性能测试

测试并行处理工具的性能
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaquant.utils.parallel_processor import (
    ParallelProcessor,
    ParallelDataProcessor,
    ProgressParallelProcessor,
    parallel_map,
)


def generate_test_data(rows: int = 10000) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=rows),
        'close': np.random.rand(rows) * 100 + 50,
        'volume': np.random.randint(1000000, 10000000, rows),
    })


def cpu_intensive_task(x: int) -> int:
    """CPU 密集型任务"""
    result = 0
    for i in range(100000):
        result += x * i
    return result


def io_intensive_task(stock_id: str) -> dict:
    """I/O 密集型任务（模拟）"""
    time.sleep(0.01)  # 模拟 I/O 延迟
    return {
        'stock_id': stock_id,
        'close': np.random.rand() * 100,
    }


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算因子（CPU 密集型）"""
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    return df


def test_cpu_parallel():
    """测试 CPU 并行处理"""
    print("\n" + "=" * 70)
    print("Test 1: CPU Parallel Processing")
    print("=" * 70)

    n_tasks = 100

    # 串行处理
    print("\nSerial processing:")
    start = time.time()
    serial_results = [cpu_intensive_task(i) for i in range(n_tasks)]
    serial_time = time.time() - start
    print(f"Time: {serial_time:.2f}s")

    # 并行处理（进程）
    print("\nParallel processing (Process):")
    processor = ParallelProcessor(max_workers=None, use_process=True)
    start = time.time()
    parallel_results = processor.map(cpu_intensive_task, range(n_tasks))
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")

    # 并行处理（线程）
    print("\nParallel processing (Thread):")
    processor = ParallelProcessor(max_workers=None, use_process=False)
    start = time.time()
    thread_results = processor.map(cpu_intensive_task, range(n_tasks))
    thread_time = time.time() - start
    print(f"Time: {thread_time:.2f}s")

    # 验证结果
    assert serial_results == parallel_results
    assert serial_results == thread_results

    # 计算加速比
    speedup = serial_time / parallel_time
    print(f"\nSpeedup (Process): {speedup:.2f}x")

    speedup_thread = serial_time / thread_time
    print(f"Speedup (Thread): {speedup_thread:.2f}x")


def test_io_parallel():
    """测试 I/O 并行处理"""
    print("\n" + "=" * 70)
    print("Test 2: I/O Parallel Processing")
    print("=" * 70)

    stock_ids = [f'60000{i}' for i in range(50)]

    # 串行处理
    print("\nSerial processing:")
    start = time.time()
    serial_results = [io_intensive_task(sid) for sid in stock_ids]
    serial_time = time.time() - start
    print(f"Time: {serial_time:.2f}s")

    # 并行处理（线程）
    print("\nParallel processing (Thread):")
    processor = ParallelProcessor(max_workers=10, use_process=False)
    start = time.time()
    parallel_results = processor.map(io_intensive_task, stock_ids)
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")

    # 计算加速比
    speedup = serial_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 70)
    print("Test 3: Batch Processing")
    print("=" * 70)

    n_items = 1000

    def process_batch(items: list) -> list:
        """处理一批数据"""
        return [x * 2 for x in items]

    # 批量并行处理
    processor = ParallelProcessor(max_workers=4, use_process=False)
    start = time.time()
    results = processor.batch_map(process_batch, range(n_items), batch_size=100)
    batch_time = time.time() - start

    print(f"\nProcessed {n_items} items in {batch_time:.2f}s")
    print(f"Rate: {n_items/batch_time:.1f} items/s")

    # 验证结果（并行结果可能乱序，需要排序）
    assert sorted(results) == sorted([x * 2 for x in range(n_items)])


def test_parallel_data_processing():
    """测试并行数据处理"""
    print("\n" + "=" * 70)
    print("Test 4: Parallel Data Processing")
    print("=" * 70)

    # 生成测试数据
    data_list = [generate_test_data(1000) for _ in range(10)]

    # 串行处理
    print("\nSerial processing:")
    start = time.time()
    serial_results = [compute_factors(df.copy()) for df in data_list]
    serial_time = time.time() - start
    print(f"Time: {serial_time:.2f}s")

    # 并行处理
    print("\nParallel processing:")
    start = time.time()
    parallel_results = ParallelDataProcessor.parallel_compute_factors(
        [df.copy() for df in data_list],
        compute_factors,
        max_workers=4
    )
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")

    # 验证结果
    for sr, pr in zip(serial_results, parallel_results):
        pd.testing.assert_frame_equal(sr, pr)

    # 计算加速比
    speedup = serial_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")


def test_progress_processor():
    """测试带进度的处理器"""
    print("\n" + "=" * 70)
    print("Test 5: Progress Parallel Processor")
    print("=" * 70)

    def slow_task(x: int) -> int:
        """慢任务"""
        time.sleep(0.05)
        return x * 2

    processor = ProgressParallelProcessor(
        max_workers=4,
        use_process=False,
        show_progress=True
    )

    print("\nProcessing with progress:")
    start = time.time()
    results = processor.map(slow_task, range(20))
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Rate: {len(results)/elapsed:.1f} items/s")

    # 验证结果（并行结果可能乱序，需要排序）
    assert sorted(results) == sorted([x * 2 for x in range(20)])


def test_parallel_load_stocks():
    """测试并行加载股票"""
    print("\n" + "=" * 70)
    print("Test 6: Parallel Load Stocks")
    print("=" * 70)

    def load_stock(stock_id: str) -> pd.DataFrame:
        """加载股票数据（模拟）"""
        time.sleep(0.02)  # 模拟 I/O
        return generate_test_data(500)

    stock_ids = [f'60000{i}' for i in range(20)]

    # 串行加载
    print("\nSerial loading:")
    start = time.time()
    serial_data = {sid: load_stock(sid) for sid in stock_ids}
    serial_time = time.time() - start
    print(f"Time: {serial_time:.2f}s")

    # 并行加载
    print("\nParallel loading:")
    start = time.time()
    parallel_data = ParallelDataProcessor.parallel_load_stocks(
        stock_ids,
        load_stock,
        max_workers=10
    )
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")

    # 计算加速比
    speedup = serial_time / parallel_time
    print(f"\nSpeedup: {speedup:.2f}x")


def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "=" * 70)
    print("Test 7: Convenience Functions")
    print("=" * 70)

    # 定义函数（不能用 lambda，因为进程池需要 pickle）
    def square(x: int) -> int:
        return x ** 2

    # 测试 parallel_map（使用线程池）
    print("\nTesting parallel_map (thread pool):")
    results = parallel_map(square, range(100), max_workers=4, use_process=False)
    assert sorted(results) == sorted([x ** 2 for x in range(100)])
    print("✓ parallel_map works")

    # 测试 parallel_starmap（使用线程池，因为局部函数不能 pickle）
    print("\nTesting parallel_starmap (thread pool):")
    from alphaquant.utils.parallel_processor import parallel_starmap

    def add(a, b, c):
        return a + b + c

    args_list = [(i, i*2, i*3) for i in range(10)]
    results = parallel_starmap(add, args_list, max_workers=4, use_process=False)
    expected = [a + b + c for a, b, c in args_list]
    assert results == expected
    print("✓ parallel_starmap works")


def benchmark_scalability():
    """基准测试：可扩展性"""
    print("\n" + "=" * 70)
    print("Test 8: Scalability Benchmark")
    print("=" * 70)

    task_counts = [10, 50, 100, 200]

    print("\nCPU Tasks:")
    for n_tasks in task_counts:
        processor = ParallelProcessor(max_workers=4, use_process=True)
        start = time.time()
        processor.map(cpu_intensive_task, range(n_tasks))
        elapsed = time.time() - start
        print(f"  {n_tasks} tasks: {elapsed:.2f}s ({n_tasks/elapsed:.1f} tasks/s)")

    print("\nI/O Tasks:")
    for n_tasks in task_counts:
        processor = ParallelProcessor(max_workers=10, use_process=False)
        start = time.time()
        processor.map(io_intensive_task, [f'60000{i}' for i in range(n_tasks)])
        elapsed = time.time() - start
        print(f"  {n_tasks} tasks: {elapsed:.2f}s ({n_tasks/elapsed:.1f} tasks/s)")


def main():
    """主测试函数"""
    print("=" * 70)
    print("AlphaGPT Parallel Processing Tests")
    print("=" * 70)

    # 运行测试
    test_cpu_parallel()
    test_io_parallel()
    test_batch_processing()
    test_parallel_data_processing()
    test_progress_processor()
    test_parallel_load_stocks()
    test_convenience_functions()
    benchmark_scalability()

    print("\n" + "=" * 70)
    print("All Parallel Processing Tests Completed")
    print("=" * 70)


if __name__ == '__main__':
    main()
