"""
因子计算引擎

支持的因子：
- 基础因子: 收益率、波动率、成交量变化、价格动量、RSI、MACD
- 高级因子: 布林带、ATR、KDJ、VWAP 等
"""

from .china_factors import ChinaFactorEngine

__all__ = ["ChinaFactorEngine"]
