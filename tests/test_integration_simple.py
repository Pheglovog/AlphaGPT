"""集成测试 - AlphaGPT 完整流程测试（简化版）"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实际存在的类
from alphaquant.data_validation import DataValidator
from alphaquant.data_cache import DataCache


class TestAlphaGPTIntegrationSimple:
    """AlphaGPT 集成测试 - 简化版（基于实际 API）"""

    @pytest.fixture
    def sample_stock_data(self):
        """创建样本股票数据"""
        return pd.DataFrame({
            'ts_code': ['600519.SH', '000001.SZ', '600000.SH'],
            'trade_date': ['20240101', '20240101', '20240101'],
            'open': [1800.0, 10.5, 7.2],
            'high': [1850.0, 11.0, 7.5],
            'low': [1790.0, 10.0, 7.0],
            'close': [1820.0, 10.8, 7.3],
            'vol': [100000, 500000, 200000]
        })

    @pytest.fixture
    def sample_invalid_data(self):
        """创建包含无效数据的样本"""
        return pd.DataFrame({
            'ts_code': ['600519.SH', '000001.SZ'],
            'trade_date': ['20240101', 'invalid'],
            'open': [1800.0, -100.0],  # 负价格
            'high': [1850.0, 11.0],
            'low': [1790.0, 10.0],
            'close': [1820.0, 10.8],
            'vol': [100000, -500]  # 负成交量
        })

    def test_validator_initialization(self):
        """测试验证器初始化"""
        validator = DataValidator()
        assert validator.min_price == 0.01
        assert validator.max_price == 10000.0
        assert validator.min_volume == 100
        assert isinstance(validator.validation_stats, dict)

    def test_validate_dataframe_valid(self, sample_stock_data):
        """测试有效数据的验证"""
        validator = DataValidator()
        is_valid, stats = validator.validate_dataframe(sample_stock_data)
        
        # 应该通过验证
        assert is_valid is True
        assert stats["total_rows"] == 3

    def test_validate_dataframe_invalid(self, sample_invalid_data):
        """测试无效数据的验证"""
        validator = DataValidator()
        is_valid, stats = validator.validate_dataframe(sample_invalid_data)
        
        # 应该检测到无效数据
        assert is_valid is False
        # 应该有异常值
        assert stats["price_outliers"] >= 1 or stats["volume_outliers"] >= 1

    def test_validate_price_valid(self):
        """测试有效价格验证"""
        validator = DataValidator()
        assert validator.validate_price(10.5) is True
        assert validator.validate_price(0.01) is True
        assert validator.validate_price(10000.0) is True

    def test_validate_price_invalid(self):
        """测试无效价格验证"""
        validator = DataValidator()
        assert validator.validate_price(-10.5) is False
        assert validator.validate_price(0) is False
        assert validator.validate_price(10001.0) is False

    def test_validate_volume_valid(self):
        """测试有效成交量验证"""
        validator = DataValidator()
        assert validator.validate_volume(100) is True
        assert validator.validate_volume(10000) is True
        assert validator.validate_volume(1000000) is True

    def test_validate_volume_invalid(self):
        """测试无效成交量验证"""
        validator = DataValidator()
        assert validator.validate_volume(99) is False
        assert validator.validate_volume(0) is False
        assert validator.validate_volume(-100) is False

    def test_validate_return_valid(self):
        """测试有效收益率验证"""
        validator = DataValidator()
        assert validator.validate_return(0.05) is True
        assert validator.validate_return(-0.05) is True
        assert validator.validate_return(0) is True

    def test_validate_return_invalid(self):
        """测试无效收益率验证"""
        validator = DataValidator()
        assert validator.validate_return(-0.21) is False
        assert validator.validate_return(0.21) is False

    def test_cache_single_operation(self):
        """测试单个缓存操作"""
        cache = DataCache()

        test_params = {"symbol": "600519.SH", "start": "20240101"}
        test_data = pd.DataFrame({"close": [1800.0]})

        cache.set(test_params, test_data)
        result = cache.get(test_params)

        assert result is not None
        assert len(result) == len(test_data)

    def test_cache_multiple_operations(self):
        """测试多个缓存操作"""
        cache = DataCache()

        # 批量写入
        for i in range(10):
            test_params = {"symbol": f"60000{i}.SH", "start": "20240101"}
            test_data = pd.DataFrame({"close": [1800.0 + i * 10.0]})
            cache.set(test_params, test_data)

        # 批量读取
        for i in range(10):
            test_params = {"symbol": f"60000{i}.SH", "start": "20240101"}
            result = cache.get(test_params)
            assert result is not None
            assert len(result) == 1

    def test_data_quality_workflow(self, sample_stock_data):
        """测试数据质量分析工作流"""
        validator = DataValidator()

        # 步骤 1: 验证数据
        is_valid, stats = validator.validate_dataframe(sample_stock_data)
        assert is_valid is True

        # 步骤 2: 检查统计
        assert stats["total_rows"] == 3
        assert stats["valid_rows"] >= 0


class TestDataCachePerformance:
    """数据缓存性能测试"""

    def test_cache_performance(self):
        """测试缓存性能"""
        cache = DataCache()

        # 测试批量操作
        import time
        start = time.time()

        for i in range(100):
            test_params = {"symbol": f"60000{i}.SH", "start": "20240101"}
            test_data = pd.DataFrame({"close": [1800.0 + i]})
            cache.set(test_params, test_data)

        end = time.time()
        elapsed = end - start

        # 应该在合理时间内完成
        assert elapsed < 5.0  # 5 秒内完成 100 次写入


class TestValidationScenarios:
    """验证场景测试"""

    def test_empty_dataframe(self):
        """测试空 DataFrame"""
        validator = DataValidator()
        df_empty = pd.DataFrame()

        is_valid, stats = validator.validate_dataframe(df_empty)
        assert is_valid is False
        assert stats["total_rows"] == 0

    def test_missing_required_columns(self):
        """测试缺少必需列"""
        validator = DataValidator()
        df_missing = pd.DataFrame({
            'ts_code': ['600519.SH'],
            'trade_date': ['20240101']
            # 缺少价格列
        })

        is_valid, stats = validator.validate_dataframe(df_missing)
        assert is_valid is False

    def test_boundary_values(self):
        """测试边界值"""
        validator = DataValidator()

        # 测试边界价格
        assert validator.validate_price(0.01) is True
        assert validator.validate_price(10000.0) is True
        assert validator.validate_price(0.009) is False
        assert validator.validate_price(10000.1) is False

        # 测试边界成交量
        assert validator.validate_volume(100) is True
        assert validator.validate_volume(99) is False


class TestErrorHandling:
    """错误处理测试"""

    def test_type_error_handling(self):
        """测试类型错误处理"""
        validator = DataValidator()

        # 测试无效的价格类型（应该抛出 TypeError）
        try:
            validator.validate_price("invalid")
            assert False, "Should have raised TypeError"
        except TypeError:
            # 预期的错误
            pass

        try:
            validator.validate_price(None)
            assert False, "Should have raised TypeError"
        except TypeError:
            # 预期的错误
            pass

    def test_edge_cases(self):
        """测试边界情况"""
        validator = DataValidator()

        # 测试浮点数精度
        assert validator.validate_price(0.010000000000000002) is True

        # 测试非常大的数值
        assert validator.validate_price(9999.99) is True

        # 测试非常小的数值
        assert validator.validate_price(0.010000000000001) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
