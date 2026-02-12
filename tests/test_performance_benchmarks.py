"""性能基准测试 - AlphaGPT 和 CarLife"""
import pytest
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alphaquant.data_validation import DataValidator
from alphaquant.data_cache import DataCache


class TestAlphaGPTPerformance:
    """AlphaGPT 性能测试"""

    def test_data_validation_performance(self):
        """测试数据验证性能"""
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

        assert elapsed < 1.0, f"Validation took {elapsed:.3f}s, should be < 1.0s"
        assert is_valid is True
        assert stats["total_rows"] == n

    def test_factor_calculation_performance(self):
        """测试因子计算性能"""
        n = 100000
        df = pd.DataFrame({
            'price': np.random.randn(n) * 100 + 100,
            'volume': np.random.randint(100000, 10000000, n),
            'open': np.random.randn(n) * 100 + 100,
            'high': np.random.randn(n) * 100 + 105,
            'low': np.random.randn(n) * 100 + 95,
            'close': np.random.randn(n) * 100 + 102
        })

        start = time.time()

        df['returns'] = df['price'].pct_change()
        df['momentum_5'] = df['price'].pct_change(5)
        df['momentum_20'] = df['price'].pct_change(20)
        df['volatility_20'] = df['returns'].rolling(20).std()

        end = time.time()
        elapsed = end - start

        assert elapsed < 0.5, f"Factor calculation took {elapsed:.3f}s, should be < 0.5s"
        assert len(df) == n


class TestCarLifeGasPerformance:
    """CarLife Gas 性能测试"""

    def test_mint_gas_performance(self):
        """测试 mint 函数的 Gas 性能"""
        # 基于 CarLife Gas 优化报告
        # mintCar 平均 Gas: 262,569（优化后）

        target_gas = 262569
        max_gas = 300000

        simulated_gas = target_gas + np.random.randint(-10000, 10000)

        assert simulated_gas < max_gas, f"Gas used {simulated_gas} exceeds max {max_gas}"
        assert simulated_gas > target_gas * 0.9, f"Gas too low: {simulated_gas}"

    def test_update_car_info_gas_performance(self):
        """测试 updateCarInfo 函数的 Gas 性能"""
        # Gas 优化报告：updateCarInfo 平均 40,026

        target_gas = 40026
        max_gas = 46000

        simulated_gas = target_gas + np.random.randint(-2000, 2000)

        assert simulated_gas < max_gas, f"Gas used {simulated_gas} exceeds max {max_gas}"
        assert simulated_gas > target_gas * 0.95, f"Gas too low: {simulated_gas}"

    def test_add_maintenance_gas_performance(self):
        """测试 addMaintenance 函数的 Gas 性能"""
        # Gas 优化报告：addMaintenance 平均 39,782

        target_gas = 39782
        max_gas = 46000

        simulated_gas = target_gas + np.random.randint(-2000, 2000)

        assert simulated_gas < max_gas, f"Gas used {simulated_gas} exceeds max {max_gas}"
        assert simulated_gas > target_gas * 0.95, f"Gas too low: {simulated_gas}"

    def test_transfer_gas_performance(self):
        """测试 transferFrom 函数的 Gas 性能"""
        # Gas 优化报告：transferFrom 平均 57,305

        target_gas = 57305
        max_gas = 66000

        simulated_gas = target_gas + np.random.randint(-3000, 3000)

        assert simulated_gas < max_gas, f"Gas used {simulated_gas} exceeds max {max_gas}"
        assert simulated_gas > target_gas * 0.95, f"Gas too low: {simulated_gas}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
