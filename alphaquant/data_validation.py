"""
数据验证和清洗模块
确保数据质量和一致性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta


class DataValidator:
    """数据验证器"""

    def __init__(
        self,
        min_price: float = 0.01,      # 最低价格
        max_price: float = 10000.0,    # 最高价格
        min_volume: int = 100,          # 最低成交量
        min_return: float = -0.20,      # 最小日收益率
        max_return: float = 0.20,       # 最大日收益率
        min_sharpe: float = -5.0,       # 最小夏普比率
        max_sharpe: float = 5.0,        # 最大夏普比率
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_return = min_return
        self.max_return = max_return
        self.min_sharpe = min_sharpe
        self.max_sharpe = max_sharpe

        # 验证统计
        self.validation_stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "price_outliers": 0,
            "volume_outliers": 0,
            "return_outliers": 0,
            "sharpe_outliers": 0
        }

    def validate_price(self, price: float) -> bool:
        """验证价格是否合理"""
        return self.min_price <= price <= self.max_price

    def validate_volume(self, volume: int) -> bool:
        """验证成交量是否合理"""
        return volume >= self.min_volume

    def validate_return(self, return_rate: float) -> bool:
        """验证收益率是否合理"""
        return self.min_return <= return_rate <= self.max_return

    def validate_sharpe(self, sharpe_ratio: float) -> bool:
        """验证夏普比率是否合理"""
        return self.min_sharpe <= sharpe_ratio <= self.max_sharpe

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        验证整个 DataFrame

        Args:
            df: 输入 DataFrame

        Returns:
            (is_valid: bool, stats: Dict) 是否有效及统计信息
        """
        self.validation_stats["total_rows"] = len(df)

        if df.empty:
            logger.warning("DataFrame 为空")
            return False, self.validation_stats

        # 检查必需列
        required_columns = ["trade_date", "open", "high", "low", "close", "vol"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"缺少必需列: {missing_columns}")
            return False, self.validation_stats

        # 复制 DataFrame 以避免修改原始数据
        df_clean = df.copy()

        # 验证价格
        if "open" in df_clean.columns:
            price_outliers = []
            for col in ["open", "high", "low", "close"]:
                if col in df_clean.columns:
                    invalid_mask = ~df_clean[col].apply(self.validate_price)
                    price_outliers.append(invalid_mask.sum())

                    # 记录异常价格
                    df_clean.loc[invalid_mask, col] = np.nan

            self.validation_stats["price_outliers"] = sum(price_outliers)
            logger.info(f"价格异常值: {self.validation_stats['price_outliers']}")

        # 验证成交量
        if "vol" in df_clean.columns:
            invalid_mask = ~df_clean["vol"].apply(self.validate_volume)
            self.validation_stats["volume_outliers"] = invalid_mask.sum()

            # 记录异常成交量
            df_clean.loc[invalid_mask, "vol"] = np.nan

            logger.info(f"成交量异常值: {self.validation_stats['volume_outliers']}")

        # 验证收益率（如果存在）
        if "pct_chg" in df_clean.columns:
            invalid_mask = ~df_clean["pct_chg"].apply(self.validate_return)
            self.validation_stats["return_outliers"] = invalid_mask.sum()

            # 记录异常收益率
            df_clean.loc[invalid_mask, "pct_chg"] = np.nan

            logger.info(f"收益率异常值: {self.validation_stats['return_outliers']}")

        # 计算有效行数
        valid_rows = df_clean.dropna(subset=["open", "high", "low", "close", "vol"]).shape[0]
        self.validation_stats["valid_rows"] = valid_rows
        self.validation_stats["invalid_rows"] = len(df) - valid_rows

        logger.info(f"验证完成：{valid_rows}/{len(df)} 行有效")

        is_valid = (self.validation_stats["invalid_rows"] == 0)

        if not is_valid:
            logger.warning(f"DataFrame 验证失败：{self.validation_stats}")

        return is_valid, self.validation_stats

    def get_validation_report(self) -> str:
        """获取验证报告"""
        report = f"""
        === 数据验证报告 ===
        总行数：{self.validation_stats["total_rows"]}
        有效行数：{self.validation_stats["valid_rows"]}
        无效行数：{self.validation_stats["invalid_rows"]}
        异常价格：{self.validation_stats["price_outliers"]}
        异常成交量：{self.validation_stats["volume_outliers"]}
        异常收益率：{self.validation_stats["return_outliers"]}
        异常夏普比率：{self.validation_stats["sharpe_outliers"]}
        """
        return report


class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        self.cleaning_stats = {
            "duplicates_removed": 0,
            "nulls_filled": 0,
            "outliers_removed": 0,
            "rows_cleaned": 0
        }

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        移除重复行

        Args:
            df: 输入 DataFrame
            subset: 判断重复的列（默认：所有列）

        Returns:
            清洗后的 DataFrame
        """
        original_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep="first")
        removed = original_rows - len(df_clean)

        self.cleaning_stats["duplicates_removed"] = removed
        self.cleaning_stats["rows_cleaned"] += removed

        logger.info(f"移除 {removed} 个重复行")

        return df_clean.reset_index(drop=True)

    def fill_nulls(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """
        填充空值

        Args:
            df: 输入 DataFrame
            method: 填充方法（ffill=前向填充，bfill=后向填充，mean=均值填充）

        Returns:
            填充后的 DataFrame
        """
        original_nulls = df.isnull().sum().sum()

        if method == "ffill":
            df_clean = df.fillna(method="ffill")
        elif method == "bfill":
            df_clean = df.fillna(method="bfill")
        elif method == "mean":
            # 对数值列使用均值填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean_value = df[col].mean()
                df_clean[col].fillna(mean_value, inplace=True)
        else:
            logger.warning(f"未知的填充方法：{method}，使用 ffill")
            df_clean = df.fillna(method="ffill")

        filled_nulls = original_nulls - df_clean.isnull().sum().sum()
        self.cleaning_stats["nulls_filled"] = filled_nulls
        self.cleaning_stats["rows_cleaned"] += filled_nulls

        logger.info(f"填充 {filled_nulls} 个空值（方法：{method})")

        return df_clean

    def remove_outliers(self, df: pd.DataFrame, column: str, method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame:
        """
        移除异常值

        Args:
            df: 输入 DataFrame
            column: 列名
            method: 异常值检测方法（iqr=四分位数，zscore=Z分数）
            threshold: 阈值（IQR 的倍数或 Z 分数的绝对值）

        Returns:
            清洗后的 DataFrame
        """
        if column not in df.columns:
            logger.warning(f"列 '{column}' 不存在于 DataFrame 中")
            return df

        original_rows = len(df)

        if method == "iqr":
            # IQR 方法（四分位数）
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            outliers_removed = (~mask).sum()

        elif method == "zscore":
            # Z 分数方法
            mean = df[column].mean()
            std = df[column].std()

            z_scores = np.abs((df[column] - mean) / std)
            mask = z_scores <= threshold
            outliers_removed = (~mask).sum()

        else:
            logger.warning(f"未知的异常值检测方法：{method}，使用 iqr")
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            outliers_removed = (~mask).sum()

        df_clean = df[mask].reset_index(drop=True)

        self.cleaning_stats["outliers_removed"] = outliers_removed
        self.cleaning_stats["rows_cleaned"] += outliers_removed

        logger.info(f"移除 {outliers_removed} 个异常值（列：{column}，方法：{method}）")

        return df_clean

    def normalize_prices(self, df: pd.DataFrame, reference_date: str) -> pd.DataFrame:
        """
        标准化价格（使用调整后开盘价）

        Args:
            df: 输入 DataFrame
            reference_date: 参考日期（用于计算调整因子）

        Returns:
            标准化后的 DataFrame
        """
        if "open" not in df.columns or "pre_close" not in df.columns:
            logger.warning("缺少 'open' 或 'pre_close' 列，跳过价格标准化")
            return df

        # 检查参考日期是否存在
        if reference_date not in df["trade_date"].values:
            logger.warning(f"参考日期 {reference_date} 不存在，跳过价格标准化")
            return df

        # 计算参考日的调整因子
        ref_row = df[df["trade_date"] == reference_date].iloc[0]
        if ref_row["pre_close"] == 0:
            logger.warning("参考日的 pre_close 为 0，跳过价格标准化")
            return df

        adjustment_factor = ref_row["pre_close"] / ref_row["open"]

        # 应用调整因子
        df_clean = df.copy()
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col] * adjustment_factor

        logger.info(f"价格标准化完成，调整因子：{adjustment_factor:.4f}")

        return df_clean

    def get_cleaning_report(self) -> str:
        """获取清洗报告"""
        report = f"""
        === 数据清洗报告 ===
        移除的重复行：{self.cleaning_stats["duplicates_removed"]}
        填充的空值：{self.cleaning_stats["nulls_filled"]}
        移除的异常值：{self.cleaning_stats["outliers_removed"]}
        清洗的总行数：{self.cleaning_stats["rows_cleaned"]}
        """
        return report


class DataQualityAnalyzer:
    """数据质量分析器"""

    def __init__(self):
        pass

    def analyze_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        分析数据完整性

        Args:
            df: 输入 DataFrame

        Returns:
            完整性统计
        """
        stats = {
            "total_cells": df.size,
            "total_cells_with_nulls": df.isnull().sum().sum(),
            "completeness": (1 - df.isnull().sum().sum() / df.size) * 100,
            "total_rows": len(df),
            "rows_with_nulls": df.isnull().any(axis=1).sum(),
            "total_columns": len(df.columns),
            "columns_with_nulls": df.isnull().any(axis=0).sum()
        }

        return stats

    def analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据一致性

        Args:
            df: 输入 DataFrame

        Returns:
            一致性统计
        """
        stats = {}

        # 检查价格一致性（low <= open <= high, close <= high）
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            valid_price = (df["low"] <= df["open"]) & (df["open"] <= df["high"]) & (df["close"] <= df["high"])
            stats["price_consistency"] = (valid_price.sum() / len(df)) * 100
            stats["price_inconsistencies"] = (~valid_price).sum()
        else:
            stats["price_consistency"] = None
            stats["price_inconsistencies"] = None

        # 检查成交量一致性（必须是正数）
        if "vol" in df.columns:
            valid_volume = (df["vol"] > 0)
            stats["volume_consistency"] = (valid_volume.sum() / len(df)) * 100
            stats["volume_inconsistencies"] = (~valid_volume).sum()
        else:
            stats["volume_consistency"] = None
            stats["volume_inconsistencies"] = None

        return stats

    def analyze_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        分析数据分布

        Args:
            df: 输入 DataFrame
            column: 列名

        Returns:
            分布统计
        """
        if column not in df.columns:
            logger.warning(f"列 '{column}' 不存在")
            return {}

        stats = {
            "count": len(df[column].dropna()),
            "mean": df[column].mean(),
            "std": df[column].std(),
            "min": df[column].min(),
            "max": df[column].max(),
            "q25": df[column].quantile(0.25),
            "q50": df[column].median(),
            "q75": df[column].quantile(0.75),
            "skewness": df[column].skew(),
            "kurtosis": df[column].kurtosis()
        }

        return stats

    def generate_quality_report(self, df: pd.DataFrame) -> str:
        """
        生成数据质量报告

        Args:
            df: 输入 DataFrame

        Returns:
            质量报告
        """
        completeness = self.analyze_completeness(df)
        consistency = self.analyze_consistency(df)

        report = f"""
        === 数据质量报告 ===

        数据维度：{df.shape[0]} 行 × {df.shape[1]} 列

        --- 完整性 ---
        总单元格数：{completeness['total_cells']}
        空单元格数：{completeness['total_cells_with_nulls']}
        完整度：{completeness['completeness']:.2f}%
        含空值的行数：{completeness['rows_with_nulls']}
        含空值的列数：{completeness['columns_with_nulls']}

        --- 一致性 ---
        价格一致性：{consistency.get('price_consistency', 'N/A'):.2f}%
        价格不一致数：{consistency.get('price_inconsistencies', 'N/A')}
        成交量一致性：{consistency.get('volume_consistency', 'N/A'):.2f}%
        成交量不一致数：{consistency.get('volume_inconsistencies', 'N/A')}

        --- 数据列 ---
        列名：{', '.join(df.columns)}

        --- 数据类型 ---
        {df.dtypes.to_string()}
        """

        return report


# 使用示例
def example_usage():
    """使用示例"""
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    data = {
        'trade_date': dates,
        'open': 10 + np.random.randn(len(dates)) * 2,
        'high': 12 + np.random.randn(len(dates)) * 2,
        'low': 8 + np.random.randn(len(dates)) * 2,
        'close': 11 + np.random.randn(len(dates)) * 2,
        'vol': np.random.randint(100000, 1000000, len(dates))
    }

    df = pd.DataFrame(data)

    # 添加一些异常值
    df.loc[10:15, 'vol'] = 0  # 异常成交量
    df.loc[20:25, 'open'] = 1000  # 异常价格
    df.loc[30:35, 'close'] = -5  # 异常价格
    df.loc[40:45, 'close'] = np.nan  # 空值
    df.loc[50:55, 'close'] = df.loc[45:50, 'close']  # 重复值

    print("=== 原始数据 ===")
    print(df.head(10))

    # 1. 数据验证
    print("\n=== 数据验证 ===")
    validator = DataValidator()
    is_valid, stats = validator.validate_dataframe(df)
    print(validator.get_validation_report())

    # 2. 数据清洗
    print("\n=== 数据清洗 ===")
    cleaner = DataCleaner()

    # 移除重复行
    df_clean = cleaner.remove_duplicates(df)

    # 填充空值
    df_clean = cleaner.fill_nulls(df_clean, method="ffill")

    # 移除价格异常值
    df_clean = cleaner.remove_outliers(df_clean, column="open", method="iqr")

    print(cleaner.get_cleaning_report())

    print("\n=== 清洗后的数据 ===")
    print(df_clean.head(10))

    # 3. 数据质量分析
    print("\n=== 数据质量分析 ===")
    analyzer = DataQualityAnalyzer()
    report = analyzer.generate_quality_report(df_clean)
    print(report)


if __name__ == "__main__":
    example_usage()
