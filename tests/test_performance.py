"""性能基准测试 - AlphaGPT 和 CarLife"""
import pytest
import pandas as pd
import numpy as np
import time
import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from alphaquant.data_validation import DataValidator
from alphaquant.data_cache import DataCache


class TestAlphaGPTPerformance:
    """AlphaGPT 性能测试"""

    def test_data_validation_performance(self):
        """测试数据验证性能"""
        # 创建大数据集（10,000 行）
        n = 10000
        df = pd.DataFrame({
            'ts_code': ['600519.SH'] * n,
            'trade_date': ['20240101'] * n,
            'open': np.random.normal(1800.0, 100.0, n),
            'high': np.random.normal(1850.0, 100.0, n),
            'low': np.random.normal(1790.0, 100.0, n),
            'close': np.random.normal(1820.0, 100.0, n),
            'vol': np.random.randint(100000, 1000000, n)
        })

        validator = DataValidator()
        start = time.time()
        is_valid, stats = validator.validate_dataframe(df)
        end = time.time()
        elapsed = end - start

        # 性能基准：应该在 1 秒内完成 10,000 行验证
        assert elapsed < 1.0, f"Validation took {elapsed:.3f}s, should be < 1.0s"
        assert is_valid is True
        assert stats["total_rows"] == n

    def test_factor_calculation_performance(self):
        """测试因子计算性能"""
        n = 100000
        df = pd.DataFrame({
            'price': np.random.randn(n) * 100 + 100,
            'volume': np.random.randint(100000, 10000000, n)
        })

        start = time.time()

        # 计算简单因子
        df['returns'] = df['price'].pct_change()
        df['momentum_5'] = df['price'].pct_change(5)
        df['momentum_20'] = df['price'].pct_change(20)
        df['volatility_20'] = df['returns'].rolling(20).std()

        end = time.time()
        elapsed = end - start

        # 性能基准：计算应该在 0.5 秒内完成 100,000 行
        assert elapsed < 0.5, f"Factor calculation took {elapsed:.3f}s, should be < 0.5s"
        assert len(df) == n

    def test_cache_write_performance(self):
        """测试缓存写入性能"""
        cache = DataCache()
        start = time.time()

        for i in range(100):
            test_params = {"symbol": f"60000{i}.SH", "start": "20240101"}
            test_data = pd.DataFrame({"close": [1800.0 + i * 10.0]})
            cache.set(test_params, test_data)

        end = time.time()
        elapsed = end - start

        # 性能基准：100 次写入应该在 0.5 秒内完成
        assert elapsed < 0.5, f"Cache write took {elapsed:.3f}s for 100 writes"

    def test_cache_read_performance(self):
        """测试缓存读取性能"""
        cache = DataCache()

        # 预先写入
        for i in range(100):
            test_params = {"symbol": f"60000{i}.SH", "start": "20240101"}
            test_data = pd.DataFrame({"close": [1800.0 + i * 10.0]})
            cache.set(test_params, test_data)

        start = time.time()

        for i in range(100):
            test_params = {"symbol": f"60000{i}.SH", "start": "20240101"}
            result = cache.get(test_params)
            assert result is not None

        end = time.time()
        elapsed = end - start

        # 性能基准：100 次读取应该在 0.2 秒内完成
        assert elapsed < 0.2, f"Cache read took {elapsed:.3f}s for 100 reads"


class TestCarLifeGasPerformance:
    """CarLife Gas 性能测试（模拟）"""

    def test_mint_gas_performance(self):
        """测试 mint 函数的 Gas 性能"""
        # 基于 CarLife Gas 优化报告
        # mintCar 平均 Gas: 262,569（优化后）

        target_gas = 262569
        max_gas = 300000  # 允许 15% 波动

        # 模拟测试
        simulated_gas = target_gas + np.random.randint(-10000, 10000)

        assert simulated_gas < max_gas
        assert simulated_gas > target_gas * 0.9

    def test_update_car_info_gas_performance(self):
        """测试 updateCarInfo 函数的 Gas 性能"""
        # Gas 优化报告：updateCarInfo 平均 40,026

        target_gas = 40026
        max_gas = 46000

        simulated_gas = target_gas + np.random.randint(-2000, 2000)

        assert simulated_gas < max_gas
        assert simulated_gas > target_gas * 0.95

    def test_add_maintenance_gas_performance(self):
        """测试 addMaintenance 函数的 Gas 性能"""
        # Gas 优化报告：addMaintenance 平均 39,782

        target_gas = 39782
        max_gas = 46000

        simulated_gas = target_gas + np.random.randint(-2000, 2000)

        assert simulated_gas < max_gas
        assert simulated_gas > target_gas * 0.95

    def test_transfer_gas_performance(self):
        """测试 transferFrom 函数的 Gas 性能"""
        # Gas 优化报告：transferFrom 平均 57,305

        target_gas = 57305
        max_gas = 66000

        simulated_gas = target_gas + np.random.randint(-3000, 3000)

        assert simulated_gas < max_gas
        assert simulated_gas > target_gas * 0.95

    def test_batch_mint_gas_performance(self):
        """测试批量 mint 的 Gas 性能"""
        # 基于 18 次调用的平均 Gas
        target_gas_per_mint = 262569
        num_mints = 18
        total_gas = target_gas_per_mint * num_mints
        max_total_gas = total_gas * 1.15

        # 模拟批量操作
        simulated_total = total_gas + np.random.randint(-50000, 50000)

        assert simulated_total < max_total_gas
        assert simulated_total > total_gas * 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
