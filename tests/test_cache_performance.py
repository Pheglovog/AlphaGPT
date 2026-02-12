"""
缓存系统性能基准测试

测试不同缓存实现的性能对比
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaquant.data_cache import CacheManager, MemoryCache, FileCache


def generate_test_data(size: int = 10000) -> pd.DataFrame:
    """
    生成测试数据

    Args:
        size: 数据行数

    Returns:
        测试 DataFrame
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=size)

    return pd.DataFrame({
        'date': dates,
        'open': np.random.rand(size) * 100 + 50,
        'high': np.random.rand(size) * 100 + 50,
        'low': np.random.rand(size) * 100 + 50,
        'close': np.random.rand(size) * 100 + 50,
        'volume': np.random.randint(1000000, 10000000, size),
    })


def benchmark_cache(cache, name: str, data: pd.DataFrame, operations: int = 1000):
    """
    基准测试单个缓存

    Args:
        cache: 缓存实例
        name: 缓存名称
        data: 测试数据
        operations: 操作次数

    Returns:
        性能结果
    """
    results = {
        'name': name,
        'operations': operations,
    }

    # 写入测试
    start_time = time.time()
    for i in range(operations):
        cache.set(f'test_key_{i}', data, ttl=3600)
    write_time = time.time() - start_time
    results['write_time'] = write_time
    results['write_ops_per_sec'] = operations / write_time

    # 读取测试
    start_time = time.time()
    for i in range(operations):
        cache.get(f'test_key_{i}')
    read_time = time.time() - start_time
    results['read_time'] = read_time
    results['read_ops_per_sec'] = operations / read_time

    # 删除测试
    start_time = time.time()
    for i in range(operations):
        cache.delete(f'test_key_{i}')
    delete_time = time.time() - start_time
    results['delete_time'] = delete_time
    results['delete_ops_per_sec'] = operations / delete_time

    return results


def print_results(results: dict) -> None:
    """
    打印性能测试结果

    Args:
        results: 性能结果
    """
    print(f"\n{'=' * 70}")
    print(f"{results['name']} 性能测试结果")
    print(f"{'=' * 70}")
    print(f"操作次数: {results['operations']}")
    print(f"\n写入性能:")
    print(f"  总耗时: {results['write_time']:.4f} 秒")
    print(f"  吞吐量: {results['write_ops_per_sec']:.2f} ops/sec")
    print(f"\n读取性能:")
    print(f"  总耗时: {results['read_time']:.4f} 秒")
    print(f"  吞吐量: {results['read_ops_per_sec']:.2f} ops/sec")
    print(f"\n删除性能:")
    print(f"  总耗时: {results['delete_time']:.4f} 秒")
    print(f"  吞吐量: {results['delete_ops_per_sec']:.2f} ops/sec")
    print(f"{'=' * 70}\n")


def compare_results(results_list: list) -> None:
    """
    对比性能测试结果

    Args:
        results_list: 性能结果列表
    """
    print(f"\n{'=' * 70}")
    print(f"性能对比")
    print(f"{'=' * 70}")
    print(f"{'缓存类型':<15} {'写入 ops/s':<15} {'读取 ops/s':<15} {'删除 ops/s':<15}")
    print(f"{'-' * 70}")

    for result in results_list:
        print(
            f"{result['name']:<15} "
            f"{result['write_ops_per_sec']:<15.2f} "
            f"{result['read_ops_per_sec']:<15.2f} "
            f"{result['delete_ops_per_sec']:<15.2f}"
        )

    print(f"{'=' * 70}\n")

    # 计算加速比（相对于文件缓存）
    file_cache_result = next((r for r in results_list if 'File' in r['name']), None)
    if file_cache_result:
        print("\n加速比（相对于文件缓存）：")
        print(f"{'缓存类型':<15} {'写入':<15} {'读取':<15} {'删除':<15}")
        print(f"{'-' * 70}")

        for result in results_list:
            if result != file_cache_result:
                write_speedup = result['write_ops_per_sec'] / file_cache_result['write_ops_per_sec']
                read_speedup = result['read_ops_per_sec'] / file_cache_result['read_ops_per_sec']
                delete_speedup = result['delete_ops_per_sec'] / file_cache_result['delete_ops_per_sec']

                print(
                    f"{result['name']:<15} "
                    f"{write_speedup:<15.2f}x "
                    f"{read_speedup:<15.2f}x "
                    f"{delete_speedup:<15.2f}x"
                )

        print(f"{'=' * 70}\n")


def main():
    """主测试函数"""
    print("AlphaGPT 缓存系统性能基准测试")
    print("=" * 70)

    # 生成测试数据
    print("\n生成测试数据...")
    data = generate_test_data(size=1000)
    print(f"测试数据大小: {data.memory_usage(deep=True).sum() / 1024:.2f} KB")

    operations = 100
    results_list = []

    # 测试内存缓存
    print("\n测试内存缓存...")
    memory_cache = MemoryCache(default_ttl=3600, max_size=2000)
    memory_result = benchmark_cache(memory_cache, 'Memory Cache', data, operations)
    print_results(memory_result)
    results_list.append(memory_result)

    # 打印内存缓存统计
    print("内存缓存统计:")
    stats = memory_cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试文件缓存
    print("\n测试文件缓存...")
    file_cache = FileCache(cache_dir='./test_cache', default_ttl=3600)
    file_result = benchmark_cache(file_cache, 'File Cache', data, operations)
    print_results(file_result)
    results_list.append(file_result)

    # 打印文件缓存统计
    print("文件缓存统计:")
    stats = file_cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 清理文件缓存
    file_cache.clear()

    # 测试多级缓存
    print("\n测试多级缓存...")
    cache_manager = CacheManager(
        enable_memory=True,
        enable_redis=False,
        enable_file=True,
        file_config={'cache_dir': './test_cache'}
    )

    # 写入测试
    start_time = time.time()
    for i in range(operations):
        cache_manager.set(f'test_key_{i}', data, ttl=3600)
    write_time = time.time() - start_time

    # 读取测试（应该命中 L1）
    start_time = time.time()
    for i in range(operations):
        cache_manager.get(f'test_key_{i}')
    read_time = time.time() - start_time

    # 删除测试
    start_time = time.time()
    for i in range(operations):
        cache_manager.delete(f'test_key_{i}')
    delete_time = time.time() - start_time

    multi_result = {
        'name': 'Multi-Level Cache (L1)',
        'operations': operations,
        'write_time': write_time,
        'write_ops_per_sec': operations / write_time,
        'read_time': read_time,
        'read_ops_per_sec': operations / read_time,
        'delete_time': delete_time,
        'delete_ops_per_sec': operations / delete_time,
    }
    print_results(multi_result)
    results_list.append(multi_result)

    # 清理
    cache_manager.clear()

    # 性能对比
    compare_results(results_list)

    print("✅ 所有性能测试完成")


if __name__ == '__main__':
    main()
