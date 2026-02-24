"""
向量化因子计算模块
使用 NumPy 和 Numba 实现高性能因子计算
比传统 pandas 方法快 10x-100x
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from loguru import logger
from numba import njit, prange


class VectorizedFactors:
    """向量化因子计算器"""

    @staticmethod
    @njit(parallel=True)
    def rolling_mean_numba(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 Numba 加速的滚动均值计算

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动均值数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        for i in prange(window - 1, n):
            result[i] = np.mean(data[i - window + 1 : i + 1])

        # 填充前面无法计算的值
        for i in range(window - 1):
            result[i] = np.nan

        return result

    @staticmethod
    @njit(parallel=True)
    def rolling_std_numba(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 Numba 加速的滚动标准差计算

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动标准差数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        for i in prange(window - 1, n):
            window_data = data[i - window + 1 : i + 1]
            mean = np.mean(window_data)
            result[i] = np.sqrt(np.mean((window_data - mean) ** 2))

        for i in range(window - 1):
            result[i] = np.nan

        return result

    @staticmethod
    @njit(parallel=True)
    def rolling_max_numba(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 Numba 加速的滚动最大值计算

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动最大值数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        for i in prange(window - 1, n):
            result[i] = np.max(data[i - window + 1 : i + 1])

        for i in range(window - 1):
            result[i] = np.nan

        return result

    @staticmethod
    @njit(parallel=True)
    def rolling_min_numba(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 Numba 加速的滚动最小值计算

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动最小值数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        for i in prange(window - 1, n):
            result[i] = np.min(data[i - window + 1 : i + 1])

        for i in range(window - 1):
            result[i] = np.nan

        return result

    @staticmethod
    @njit(parallel=True)
    def ema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """
        使用 Numba 加速的指数移动平均计算

        Args:
            data: 输入数组
            period: 周期

        Returns:
            EMA 数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        multiplier = 2.0 / (period + 1)
        result[0] = data[0]

        for i in prange(1, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]

        return result

    @staticmethod
    @njit(parallel=True)
    def rsi_numba(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        使用 Numba 加速的 RSI 计算

        Args:
            data: 价格数据
            period: RSI 周期

        Returns:
            RSI 数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        # 计算价格变化
        delta = np.empty(n - 1, dtype=np.float64)
        for i in prange(1, n):
            delta[i - 1] = data[i] - data[i - 1]

        # 分离上涨和下跌
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # 计算平均上涨和下跌
        avg_gain = np.empty(n, dtype=np.float64)
        avg_loss = np.empty(n, dtype=np.float64)

        avg_gain[:period] = np.nan
        avg_loss[:period] = np.nan

        if n > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])

            for i in prange(period + 1, n):
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        # 计算 RSI
        rs = avg_gain / (avg_loss + 1e-10)
        result = 100 - (100 / (1 + rs))

        return result

    @staticmethod
    @njit(parallel=True)
    def bollinger_bands_numba(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用 Numba 加速的布林带计算

        Args:
            data: 价格数据
            period: 移动平均周期
            std_dev: 标准差倍数

        Returns:
            (上轨, 中轨, 下轨)
        """
        middle = VectorizedFactors.rolling_mean_numba(data, period)
        std = VectorizedFactors.rolling_std_numba(data, period)

        upper = middle + std * std_dev
        lower = middle - std * std_dev

        return upper, middle, lower

    @staticmethod
    @njit(parallel=True)
    def macd_numba(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Numba 加速的 MACD 计算

        Args:
            data: 价格数据
            fast: 快速 EMA 周期
            slow: 慢速 EMA 周期
            signal: 信号线 EMA 周期

        Returns:
            (MACD, 信号线)
        """
        ema_fast = VectorizedFactors.ema_numba(data, fast)
        ema_slow = VectorizedFactors.ema_numba(data, slow)

        macd = ema_fast - ema_slow
        signal_line = VectorizedFactors.ema_numba(macd, signal)

        return macd, signal_line

    @staticmethod
    @njit(parallel=True)
    def momentum_numba(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        使用 Numba 加速的动量指标计算

        Args:
            data: 价格数据
            period: 周期

        Returns:
            动量数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        for i in prange(period, n):
            result[i] = data[i] - data[i - period]

        for i in range(period):
            result[i] = np.nan

        return result

    @staticmethod
    @njit(parallel=True)
    def volatility_numba(data: np.ndarray, window: int = 20) -> np.ndarray:
        """
        使用 Numba 加速的波动率计算

        Args:
            data: 价格数据
            window: 窗口大小

        Returns:
            波动率数组（年化）
        """
        returns = np.diff(data) / (data[:-1] + 1e-10)
        rolling_std = VectorizedFactors.rolling_std_numba(returns, window)

        # 年化波动率（假设 252 个交易日）
        annualized_vol = rolling_std * np.sqrt(252)

        return annualized_vol

    @staticmethod
    @njit(parallel=True)
    def sharpe_ratio_numba(data: np.ndarray, risk_free_rate: float = 0.03, window: int = 60) -> np.ndarray:
        """
        使用 Numba 加速的夏普比率计算

        Args:
            data: 价格数据
            risk_free_rate: 无风险利率
            window: 窗口大小

        Returns:
            夏普比率数组
        """
        returns = np.diff(data) / (data[:-1] + 1e-10)

        n = len(returns)
        result = np.empty(n, dtype=np.float64)

        daily_rf = risk_free_rate / 252

        for i in prange(window - 1, n):
            window_returns = returns[i - window + 1 : i + 1]
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)

            if std_return > 1e-10:
                result[i] = (mean_return - daily_rf) / std_return * np.sqrt(252)
            else:
                result[i] = np.nan

        for i in range(window - 1):
            result[i] = np.nan

        return result

    @staticmethod
    @njit(parallel=True)
    def max_drawdown_numba(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Numba 加速的最大回撤计算

        Args:
            data: 价格数据

        Returns:
            (回撤数组, 最大回撤)
        """
        n = len(data)
        cumulative_max = np.empty(n, dtype=np.float64)
        drawdown = np.empty(n, dtype=np.float64)

        cumulative_max[0] = data[0]
        drawdown[0] = 0.0

        for i in prange(1, n):
            cumulative_max[i] = max(cumulative_max[i - 1], data[i])
            drawdown[i] = (data[i] - cumulative_max[i]) / cumulative_max[i]

        max_dd = np.min(drawdown)

        return drawdown, max_dd

    @staticmethod
    def batch_compute_factors(df: pd.DataFrame, windows: list = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        批量计算技术因子

        Args:
            df: 输入 DataFrame（包含 close, high, low, volume）
            windows: 计算窗口列表

        Returns:
            包含所有因子的 DataFrame
        """
        result = df.copy()

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 计算各种窗口的移动平均
        for window in windows:
            result[f'close_sma_{window}'] = VectorizedFactors.rolling_mean_numba(close, window)
            result[f'close_ema_{window}'] = VectorizedFactors.ema_numba(close, window)
            result[f'close_std_{window}'] = VectorizedFactors.rolling_std_numba(close, window)

        # RSI
        result['rsi_14'] = VectorizedFactors.rsi_numba(close, 14)

        # 布林带
        bb_upper, bb_middle, bb_lower = VectorizedFactors.bollinger_bands_numba(close, 20, 2)
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        result['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # MACD
        macd, signal_line = VectorizedFactors.macd_numba(close)
        result['macd'] = macd
        result['macd_signal'] = signal_line
        result['macd_hist'] = macd - signal_line

        # 动量
        result['momentum_10'] = VectorizedFactors.momentum_numba(close, 10)
        result['momentum_20'] = VectorizedFactors.momentum_numba(close, 20)

        # 波动率
        result['volatility_20'] = VectorizedFactors.volatility_numba(close, 20)

        # 夏普比率
        result['sharpe_60'] = VectorizedFactors.sharpe_ratio_numba(close, 0.03, 60)

        # 最大回撤
        drawdown, max_dd = VectorizedFactors.max_drawdown_numba(close)
        result['drawdown'] = drawdown
        result['max_drawdown'] = max_dd

        return result


# 性能测试
def benchmark_performance():
    """基准测试：对比 pandas vs Numba 性能"""
    import time

    # 生成测试数据
    np.random.seed(42)
    n = 100_000
    data = np.cumprod(1 + np.random.randn(n) * 0.01) * 100

    logger.info(f"测试数据量: {n:,} 行")

    # Pandas 方法
    start = time.time()
    pandas_ema = pd.Series(data).ewm(span=20).mean().values
    pandas_time = time.time() - start
    logger.info(f"Pandas EMA (20): {pandas_time:.4f} 秒")

    # Numba 方法
    start = time.time()
    numba_ema = VectorizedFactors.ema_numba(data, 20)
    numba_time = time.time() - start
    logger.info(f"Numba EMA (20): {numba_time:.4f} 秒")

    speedup = pandas_time / numba_time
    logger.info(f"加速倍数: {speedup:.2f}x")

    # 验证结果一致性
    diff = np.abs(pandas_ema - numba_ema).max()
    logger.info(f"最大差异: {diff:.10f}")


if __name__ == "__main__":
    # 运行基准测试
    benchmark_performance()
