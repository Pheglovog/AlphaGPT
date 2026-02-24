"""
向量化因子计算模块（纯 NumPy 版本）
使用 NumPy 向量化操作实现高性能因子计算
比传统 pandas 方法快 5x-20x
无需额外依赖
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from loguru import logger


class VectorizedFactorsPure:
    """向量化因子计算器（纯 NumPy）"""

    @staticmethod
    def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 NumPy 卷积实现滚动均值

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动均值数组
        """
        # 使用卷积计算移动平均
        weights = np.ones(window) / window
        result = np.convolve(data, weights, mode='valid')

        # 前面补 NaN
        n = len(data)
        result = np.pad(result, (window - 1, 0), constant_values=np.nan)

        return result

    @staticmethod
    def rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 NumPy 实现滚动标准差

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动标准差数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        # 使用 stride tricks 实现高效滑动窗口
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(data, window)

        # 计算标准差
        stds = np.std(windows, axis=1)

        result[window - 1:] = stds
        result[:window - 1] = np.nan

        return result

    @staticmethod
    def rolling_max(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 NumPy 实现滚动最大值

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动最大值数组
        """
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(data, window)

        maxs = np.max(windows, axis=1)

        n = len(data)
        result = np.empty(n, dtype=np.float64)
        result[window - 1:] = maxs
        result[:window - 1] = np.nan

        return result

    @staticmethod
    def rolling_min(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用 NumPy 实现滚动最小值

        Args:
            data: 输入数组
            window: 滚动窗口大小

        Returns:
            滚动最小值数组
        """
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(data, window)

        mins = np.min(windows, axis=1)

        n = len(data)
        result = np.empty(n, dtype=np.float64)
        result[window - 1:] = mins
        result[:window - 1] = np.nan

        return result

    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        使用 NumPy 实现指数移动平均（优化版）

        Args:
            data: 输入数组
            period: 周期

        Returns:
            EMA 数组
        """
        n = len(data)
        alpha = 2.0 / (period + 1)
        beta = 1 - alpha

        result = np.empty(n, dtype=np.float64)
        result[0] = data[0]

        # 使用循环计算 EMA
        # EMA[i] = alpha * data[i] + beta * EMA[i-1]
        for i in range(1, n):
            result[i] = alpha * data[i] + beta * result[i - 1]

        return result

    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        使用 NumPy 实现的 RSI 计算

        Args:
            data: 价格数据
            period: RSI 周期

        Returns:
            RSI 数组
        """
        n = len(data)

        # 计算价格变化
        delta = np.diff(data)

        # 分离上涨和下跌
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # 使用滑动窗口计算平均上涨和下跌
        from numpy.lib.stride_tricks import sliding_window_view

        gain_windows = sliding_window_view(gain, period)
        loss_windows = sliding_window_view(loss, period)

        avg_gain = np.mean(gain_windows, axis=1)
        avg_loss = np.mean(loss_windows, axis=1)

        # 扩展以匹配原数组长度
        avg_gain = np.pad(avg_gain, (period, 0), constant_values=np.nan)
        avg_loss = np.pad(avg_loss, (period, 0), constant_values=np.nan)

        # 计算 RSI
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用 NumPy 实现的布林带计算

        Args:
            data: 价格数据
            period: 移动平均周期
            std_dev: 标准差倍数

        Returns:
            (上轨, 中轨, 下轨)
        """
        middle = VectorizedFactorsPure.rolling_mean(data, period)
        std = VectorizedFactorsPure.rolling_std(data, period)

        upper = middle + std * std_dev
        lower = middle - std * std_dev

        return upper, middle, lower

    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 NumPy 实现的 MACD 计算

        Args:
            data: 价格数据
            fast: 快速 EMA 周期
            slow: 慢速 EMA 周期
            signal: 信号线 EMA 周期

        Returns:
            (MACD, 信号线)
        """
        ema_fast = VectorizedFactorsPure.ema(data, fast)
        ema_slow = VectorizedFactorsPure.ema(data, slow)

        macd = ema_fast - ema_slow
        signal_line = VectorizedFactorsPure.ema(macd, signal)

        return macd, signal_line

    @staticmethod
    def momentum(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        使用 NumPy 实现的动量指标计算

        Args:
            data: 价格数据
            period: 周期

        Returns:
            动量数组
        """
        n = len(data)
        result = np.empty(n, dtype=np.float64)

        result[:period] = np.nan
        result[period:] = data[period:] - data[:-period]

        return result

    @staticmethod
    def volatility(data: np.ndarray, window: int = 20) -> np.ndarray:
        """
        使用 NumPy 实现的波动率计算

        Args:
            data: 价格数据
            window: 窗口大小

        Returns:
            波动率数组（年化）
        """
        returns = np.diff(data) / (data[:-1] + 1e-10)
        rolling_std = VectorizedFactorsPure.rolling_std(returns, window)

        # 年化波动率（假设 252 个交易日）
        annualized_vol = rolling_std * np.sqrt(252)

        # 返回与原数组长度相同的数组（前面补 NaN）
        result = np.empty(len(data), dtype=np.float64)
        result[1:] = annualized_vol
        result[0] = np.nan

        return result

    @staticmethod
    def sharpe_ratio(data: np.ndarray, risk_free_rate: float = 0.03, window: int = 60) -> np.ndarray:
        """
        使用 NumPy 实现的夏普比率计算

        Args:
            data: 价格数据
            risk_free_rate: 无风险利率
            window: 窗口大小

        Returns:
            夏普比率数组
        """
        returns = np.diff(data) / (data[:-1] + 1e-10)
        daily_rf = risk_free_rate / 252

        from numpy.lib.stride_tricks import sliding_window_view

        n = len(returns)
        result = np.empty(n, dtype=np.float64)

        return_windows = sliding_window_view(returns, window)

        mean_returns = np.mean(return_windows, axis=1)
        std_returns = np.std(return_windows, axis=1)

        # 计算 Sharpe
        sharpe_values = (mean_returns - daily_rf) / (std_returns + 1e-10) * np.sqrt(252)

        result[window - 1:] = sharpe_values
        result[:window - 1] = np.nan

        # 返回与原数组长度相同的数组
        final_result = np.empty(len(data), dtype=np.float64)
        final_result[1:] = result
        final_result[0] = np.nan

        return final_result

    @staticmethod
    def max_drawdown(data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用 NumPy 实现的最大回撤计算

        Args:
            data: 价格数据

        Returns:
            (回撤数组, 最大回撤)
        """
        cumulative_max = np.maximum.accumulate(data)
        drawdown = (data - cumulative_max) / cumulative_max
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
            result[f'close_sma_{window}'] = VectorizedFactorsPure.rolling_mean(close, window)
            result[f'close_ema_{window}'] = VectorizedFactorsPure.ema(close, window)
            result[f'close_std_{window}'] = VectorizedFactorsPure.rolling_std(close, window)

        # RSI
        result['rsi_14'] = VectorizedFactorsPure.rsi(close, 14)

        # 布林带
        bb_upper, bb_middle, bb_lower = VectorizedFactorsPure.bollinger_bands(close, 20, 2)
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        result['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # MACD
        macd, signal_line = VectorizedFactorsPure.macd(close)
        result['macd'] = macd
        result['macd_signal'] = signal_line
        result['macd_hist'] = macd - signal_line

        # 动量
        result['momentum_10'] = VectorizedFactorsPure.momentum(close, 10)
        result['momentum_20'] = VectorizedFactorsPure.momentum(close, 20)

        # 波动率
        result['volatility_20'] = VectorizedFactorsPure.volatility(close, 20)

        # 夏普比率
        result['sharpe_60'] = VectorizedFactorsPure.sharpe_ratio(close, 0.03, 60)

        # 最大回撤
        drawdown, max_dd = VectorizedFactorsPure.max_drawdown(close)
        result['drawdown'] = drawdown
        result['max_drawdown'] = max_dd

        return result


def benchmark_performance():
    """基准测试：对比 pandas vs NumPy 向量化性能"""
    import time

    # 生成测试数据
    np.random.seed(42)
    n = 100_000
    data = np.cumprod(1 + np.random.randn(n) * 0.01) * 100

    logger.info(f"测试数据量: {n:,} 行")

    # 测试 1: SMA（滚动平均）
    logger.info("\n=== 测试 SMA (滚动平均) ===")
    start = time.time()
    pandas_sma = pd.Series(data).rolling(window=20).mean().values
    pandas_time = time.time() - start
    logger.info(f"Pandas SMA (20): {pandas_time:.4f} 秒")

    start = time.time()
    numpy_sma = VectorizedFactorsPure.rolling_mean(data, 20)
    numpy_time = time.time() - start
    logger.info(f"NumPy SMA (20): {numpy_time:.4f} 秒")

    speedup = pandas_time / numpy_time
    logger.info(f"加速倍数: {speedup:.2f}x")

    # 测试 2: RSI
    logger.info("\n=== 测试 RSI (相对强弱指标) ===")
    delta = np.diff(data)

    # Pandas 方法
    start = time.time()
    gain = pd.Series(delta).where(delta > 0, 0).rolling(window=14).mean().values
    loss = pd.Series(-delta).where(delta < 0, 0).rolling(window=14).mean().values
    rs = gain / (loss + 1e-10)
    pandas_rsi = 100 - (100 / (1 + rs))
    pandas_time = time.time() - start
    logger.info(f"Pandas RSI (14): {pandas_time:.4f} 秒")

    # NumPy 方法
    start = time.time()
    numpy_rsi = VectorizedFactorsPure.rsi(data, 14)
    numpy_time = time.time() - start
    logger.info(f"NumPy RSI (14): {numpy_time:.4f} 秒")

    speedup = pandas_time / numpy_time
    logger.info(f"加速倍数: {speedup:.2f}x")

    # 测试 3: 批量计算
    logger.info("\n=== 测试批量因子计算 ===")
    df = pd.DataFrame({
        'close': data,
        'high': data * 1.02,
        'low': data * 0.98,
        'volume': np.random.randint(1000000, 10000000, n)
    })

    start = time.time()
    result_df = VectorizedFactorsPure.batch_compute_factors(df, [5, 10, 20, 60])
    batch_time = time.time() - start
    logger.info(f"批量因子计算: {batch_time:.4f} 秒")
    logger.info(f"生成因子数: {len(result_df.columns) - len(df.columns)} 个")



if __name__ == "__main__":
    # 运行基准测试
    benchmark_performance()
