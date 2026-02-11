"""数据验证模块单元测试（基于类 API）"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataValidator:
    """DataValidator 类测试"""

    def test_validator_initialization(self):
        """测试验证器初始化"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.min_price == 0.01
        assert validator.max_price == 10000.0
        assert validator.min_volume == 100
        assert isinstance(validator.validation_stats, dict)

    def test_validate_price_valid(self):
        """测试有效价格验证"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.validate_price(10.5) is True
        assert validator.validate_price(0.01) is True
        assert validator.validate_price(10000.0) is True

    def test_validate_price_invalid(self):
        """测试无效价格验证"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.validate_price(-10.5) is False
        assert validator.validate_price(0) is False
        assert validator.validate_price(10001.0) is False

    def test_validate_volume_valid(self):
        """测试有效成交量验证"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.validate_volume(100) is True
        assert validator.validate_volume(10000) is True
        assert validator.validate_volume(1000000) is True

    def test_validate_volume_invalid(self):
        """测试无效成交量验证"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.validate_volume(99) is False
        assert validator.validate_volume(0) is False
        assert validator.validate_volume(-100) is False

    def test_validate_return_valid(self):
        """测试有效收益率验证"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.validate_return(0.05) is True
        assert validator.validate_return(-0.05) is True
        assert validator.validate_return(0) is True

    def test_validate_return_invalid(self):
        """测试无效收益率验证"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        assert validator.validate_return(-0.21) is False
        assert validator.validate_return(0.21) is False

    def test_validate_dataframe_empty(self):
        """测试空 DataFrame"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        df = pd.DataFrame()
        is_valid, stats = validator.validate_dataframe(df)

        assert is_valid is False
        assert stats["total_rows"] == 0

    def test_validate_dataframe_valid(self):
        """测试有效 DataFrame"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        df = pd.DataFrame({
            'trade_date': ['20240101', '20240102'],
            'open': [1800.0, 1810.0],
            'high': [1850.0, 1860.0],
            'low': [1790.0, 1800.0],
            'close': [1820.0, 1830.0],
            'vol': [100000, 110000]
        })

        is_valid, stats = validator.validate_dataframe(df)
        assert is_valid is True
        assert stats["total_rows"] == 2

    def test_validate_dataframe_missing_columns(self):
        """测试缺少必需列"""
        from alphaquant.data_validation import DataValidator

        validator = DataValidator()
        df = pd.DataFrame({
            'trade_date': ['20240101'],
            'open': [1800.0]
        })

        is_valid, stats = validator.validate_dataframe(df)
        assert is_valid is False


class TestDataCleaner:
    """DataCleaner 类测试"""

    def test_cleaner_initialization(self):
        """测试清洗器初始化"""
        from alphaquant.data_validation import DataCleaner

        cleaner = DataCleaner()
        assert cleaner is not None

    def test_remove_duplicates(self):
        """测试移除重复数据"""
        from alphaquant.data_validation import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            'ts_code': ['600519.SH', '600519.SH', '000001.SZ'],
            'trade_date': ['20240101', '20240101', '20240101'],
            'close': [1820.0, 1820.0, 10.5]
        })

        df_cleaned = cleaner.remove_duplicates(df, subset=['ts_code', 'trade_date'])
        assert len(df_cleaned) <= len(df)



class TestDataQualityAnalyzer:
    """DataQualityAnalyzer 类测试"""

    def test_analyzer_initialization(self):
        """测试分析器初始化"""
        from alphaquant.data_validation import DataQualityAnalyzer

        analyzer = DataQualityAnalyzer()
        assert analyzer is not None

    def test_calculate_completeness(self):
        """测试计算完整性"""
        from alphaquant.data_validation import DataQualityAnalyzer

        analyzer = DataQualityAnalyzer()
        df = pd.DataFrame({
            'col1': [1, 2, 3, None, 5],
            'col2': [1, None, 3, 4, 5]
        })

        # analyze_completeness 返回整体完整性
        completeness = analyzer.analyze_completeness(df)
        assert 'completeness' in completeness
        assert 0 <= completeness['completeness'] <= 100



class TestIntegration:
    """集成测试"""

    def test_full_validation_pipeline(self):
        """测试完整的验证流程"""
        from alphaquant.data_validation import DataValidator, DataCleaner, DataQualityAnalyzer

        # 创建测试数据
        df = pd.DataFrame({
            'trade_date': ['20240101', '20240102', '20240103', '20240104'],
            'open': [1800.0, 1810.0, -100.0, 1830.0],  # 第3行有异常
            'high': [1850.0, 1860.0, 1800.0, 1880.0],
            'low': [1790.0, 1800.0, 1700.0, 1820.0],
            'close': [1820.0, 1830.0, 1750.0, 1850.0],
            'vol': [100000, 110000, 0, 120000]  # 第3行成交量为0
        })

        # 验证
        validator = DataValidator()
        is_valid, stats = validator.validate_dataframe(df)

        # 清洗 - 移除异常值
        cleaner = DataCleaner()
        df_cleaned = cleaner.remove_outliers(df, 'open')

        # 分析
        analyzer = DataQualityAnalyzer()
        completeness = analyzer.analyze_completeness(df_cleaned)

        assert isinstance(is_valid, bool)
        assert isinstance(stats, dict)
        assert len(df_cleaned) <= len(df)
        assert isinstance(completeness, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
