"""
中国股市技术因子
包含适合中国市场的技术指标和因子计算

参考：
- AlphaGPT 因子设计
- 传统的技术分析指标
- 中国市场特有的因子
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from loguru import logger


class ChinaMarketFactors:
    """中国市场特定因子"""

    @staticmethod
    def northbound_flow(flow_data: pd.DataFrame, window: int = 5) -> torch.Tensor:
        """
        北向资金流向因子

        Args:
            flow_data: 北向资金数据
            window: 均值窗口

        Returns:
            北向资金流向因子
        """
        if flow_data.empty:
            return torch.zeros(1)

        flow = flow_data['north_money'].values
        flow_ma = pd.Series(flow).rolling(window=window).mean().values

        # 标准化
        flow_std = (flow_ma - np.nanmean(flow_ma)) / (np.nanstd(flow_ma) + 1e-6)

        return torch.clamp(torch.tensor(flow_std), -5, 5)

    @staticmethod
    def margin_trading(margin_data: pd.DataFrame) -> torch.Tensor:
        """
        融资融券余额因子

        Args:
            margin_data: 融资融券数据

        Returns:
            融资融券因子
        """
        if margin_data.empty:
            return torch.zeros(1)

        fin_balance = margin_data['fin_balance'].values
        fin_balance_chg = np.diff(fin_balance) / (fin_balance[:-1] + 1e-6)

        return torch.clamp(torch.tensor(fin_balance_chg), -5, 5)

    @staticmethod
    def limit_up_down(quote_data: pd.DataFrame, window: int = 20) -> torch.Tensor:
        """
        涨跌停统计因子

        Args:
            quote_data: 行情数据
            window: 统计窗口

        Returns:
            涨跌停因子
        """
        if len(quote_data) < window:
            return torch.zeros(1)

        pct_chg = quote_data['pct_chg'].values
        limit_count = np.sum((pct_chg[-window:] >= 9.8) | (pct_chg[-window:] <= -9.8))

        # 涨跌停频率
        limit_freq = limit_count / window

        return torch.clamp(torch.tensor(limit_freq * 10 - 5), -5, 5)


class TechnicalIndicators:
    """传统技术指标"""

    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        return pd.Series(data).rolling(period).mean().values

    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """指数移动平均"""
        return pd.Series(data).ewm(span=period).mean().values

    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """相对强弱指标 (RSI)"""
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values

        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        # 填充 nan 为 50
        rsi = np.nan_to_num(rsi, nan=50)
        return rsi

    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD 指标

        Returns:
            (MACD, Signal, Histogram)
        """
        ema_fast = pd.Series(data).ewm(span=fast).mean().values
        ema_slow = pd.Series(data).ewm(span=slow).mean().values

        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        布林带

        Returns:
            (Upper, Middle, Lower)
        """
        middle = pd.Series(data).rolling(period).mean().values
        std = pd.Series(data).rolling(period).std().values

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """平均真实波幅 (ATR)"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(period).mean().values

        return atr

    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        KDJ 指标

        Returns:
            (K, D, J)
        """
        lowest_low = pd.Series(low).rolling(k_period).min().values
        highest_high = pd.Series(high).rolling(k_period).max().values

        rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-6) * 100
        k = pd.Series(rsv).rolling(d_period).mean().values
        d = pd.Series(k).rolling(d_period).mean().values
        j = 3 * k - 2 * d

        k = np.nan_to_num(k, nan=50)
        d = np.nan_to_num(d, nan=50)
        j = np.nan_to_num(j, nan=50)

        return k, d, j

    @staticmethod
    def volume_price_trend(close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """
        成交量价格趋势 (VPT)

        量价关系的重要指标
        """
        close_change = np.diff(close) / (close[:-1] + 1e-6)
        vpt = np.cumsum(close_change * volume[1:])

        return vpt

    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """能量潮 (OBV)"""
        obv = np.zeros_like(close)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        return obv


class ChinaFactorEngine:
    """
    中国市场因子引擎
    整合 AlphaGPT 因子和中国市场特定因子
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.indicators = TechnicalIndicators()
        self.china_factors = ChinaMarketFactors()

    def robust_norm(self, data: torch.Tensor) -> torch.Tensor:
        """
        鲁棒标准化（基于中位数）

        Args:
            data: 输入数据

        Returns:
            标准化后的数据
        """
        median = torch.nanmedian(data, dim=-1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(data - median), dim=-1, keepdim=True)[0] + 1e-6
        norm = (data - median) / mad
        return torch.clamp(norm, -5.0, 5.0)

    def compute_basic_factors(self, raw_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算基础因子（6维）

        Args:
            raw_data: 原始数据字典，包含:
                - close: 收盘价
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - volume: 成交量
                - amount: 成交额

        Returns:
            因子张量 [batch, 6, time]
        """
        c = raw_data['close']
        o = raw_data['open']
        h = raw_data['high']
        l = raw_data['low']
        v = raw_data['volume']

        # 1. 对数收益
        ret = torch.log(c / (torch.roll(c, 1, dims=-1) + 1e-9))

        # 2. 买卖压力（类似 K 线实体强度）
        hl_range = h - l + 1e-9
        body = c - o
        pressure = body / hl_range
        pressure = torch.tanh(pressure * 3.0)

        # 3. 成交量加速度
        vol_prev = torch.roll(v, 1, dims=-1)
        vol_chg = (v - vol_prev) / (vol_prev + 1.0)
        vol_acc = vol_chg - torch.roll(vol_chg, 1, dims=-1)
        fomo = torch.clamp(vol_acc, -5.0, 5.0)

        # 4. 价格偏离（布林带位置）
        pad = torch.zeros_like(c)[..., :20]
        c_pad = torch.cat([pad, c], dim=-1)
        ma = c_pad.unfold(1, 20, 1).mean(dim=-1)
        dev = (c - ma) / (ma + 1e-9)
        deviation = torch.clamp(dev, -5.0, 5.0)

        # 5. 对数成交量
        log_vol = torch.log1p(v)

        # 6. 振幅因子
        amplitude = (h - l) / (c + 1e-9)

        features = torch.stack([
            self.robust_norm(ret),
            pressure,
            self.robust_norm(fomo),
            self.robust_norm(deviation),
            self.robust_norm(log_vol),
            self.robust_norm(amplitude)
        ], dim=1)

        return features

    def compute_advanced_factors(self, raw_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算高级因子（18维）
        包含传统技术指标和中国市场特定因子

        Args:
            raw_data: 原始数据字典

        Returns:
            因子张量 [batch, 18, time]
        """
        # 转换为 numpy 计算
        c = raw_data['close'].cpu().numpy()
        o = raw_data['open'].cpu().numpy()
        h = raw_data['high'].cpu().numpy()
        l = raw_data['low'].cpu().numpy()
        v = raw_data['volume'].cpu().numpy()

        batch_size, seq_len = c.shape
        factors = []

        # 批量计算每个样本的因子
        for i in range(batch_size):
            sample_factors = []

            # 传统技术指标
            # RSI
            rsi = self.indicators.rsi(c[i], 14)
            sample_factors.append((rsi - 50) / 50)

            # MACD
            macd, signal, hist = self.indicators.macd(c[i])
            sample_factors.append(hist)

            # 布林带位置
            bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(c[i])
            bb_position = (c[i] - bb_lower) / (bb_upper - bb_lower + 1e-6) - 0.5
            sample_factors.append(bb_position * 2)

            # ATR（归一化）
            atr = self.indicators.atr(h[i], l[i], c[i])
            atr_norm = atr / (c[i] + 1e-6)
            sample_factors.append(atr_norm * 100)

            # KDJ
            k, d, j = self.indicators.stochastic(h[i], l[i], c[i])
            sample_factors.append((k - 50) / 50)
            sample_factors.append((j - 50) / 50)

            # VPT（量价趋势）
            vpt = self.indicators.volume_price_trend(c[i], v[i])
            vpt_norm = (vpt - np.mean(vpt)) / (np.std(vpt) + 1e-6)
            sample_factors.append(np.clip(vpt_norm, -5, 5))

            # OBV
            obv = self.indicators.obv(c[i], v[i])
            obv_chg = np.diff(obv) / (np.abs(obv[:-1]) + 1e-6)
            obv_norm = np.clip(obv_chg, -5, 5)
            sample_factors.append(np.pad(obv_norm, (1, 0)))

            # AlphaGPT 风格因子
            ret = np.log(c[i] / (np.roll(c[i], 1) + 1e-9))
            sample_factors.append(self._normalize(ret))

            # 波动率
            ret_sq = ret ** 2
            vol = np.sqrt(pd.Series(ret_sq).rolling(10).mean().values + 1e-6)
            sample_factors.append(self._normalize(vol))

            # 动量（不同周期）
            mom_5 = np.roll(c[i], -5) / c[i] - 1
            mom_10 = np.roll(c[i], -10) / c[i] - 1
            mom_20 = np.roll(c[i], -20) / c[i] - 1
            sample_factors.append(np.nan_to_num(mom_5, nan=0) * 100)
            sample_factors.append(np.nan_to_num(mom_10, nan=0) * 100)
            sample_factors.append(np.nan_to_num(mom_20, nan=0) * 100)

            # 价格相对位置（在最近N天中）
            for window in [5, 10, 20]:
                c_roll = np.roll(c[i], window)
                min_val = np.minimum.reduce([c[i], c_roll])
                max_val = np.maximum.reduce([c[i], c_roll])
                position = (c[i] - min_val) / (max_val - min_val + 1e-6) - 0.5
                sample_factors.append(np.nan_to_num(position * 2, nan=0))

            # 堆叠成单个样本 [18, seq_len]
            sample_tensor = torch.tensor(np.array(sample_factors), dtype=torch.float32)
            factors.append(sample_tensor)

        # 批量堆叠 [batch, 18, seq_len]
        batch_tensor = torch.stack(factors, dim=0)
        return batch_tensor.to(self.device)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """标准化 numpy 数组"""
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median)) + 1e-6
        norm = (data - median) / mad
        return np.clip(norm, -5.0, 5.0)


def factor_example():
    """因子计算示例"""
    engine = ChinaFactorEngine()

    # 模拟数据 [batch=10, time=100]
    batch_size = 10
    seq_len = 100

    raw_data = {
        'close': torch.rand(batch_size, seq_len) * 10 + 100,
        'open': torch.rand(batch_size, seq_len) * 10 + 100,
        'high': torch.rand(batch_size, seq_len) * 10 + 105,
        'low': torch.rand(batch_size, seq_len) * 10 + 95,
        'volume': torch.rand(batch_size, seq_len) * 1000000,
        'amount': torch.rand(batch_size, seq_len) * 100000000
    }

    # 计算基础因子
    basic_factors = engine.compute_basic_factors(raw_data)
    print(f"基础因子形状: {basic_factors.shape}")  # [10, 6, 100]

    # 计算高级因子
    advanced_factors = engine.compute_advanced_factors(raw_data)
    print(f"高级因子形状: {advanced_factors.shape}")  # [10, 18, 100]


if __name__ == "__main__":
    factor_example()
