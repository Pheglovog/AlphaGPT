"""
回测引擎模块

功能：
- 支持中国股市 T+1 交易规则
- 支持涨跌停限制
- 支持交易成本和滑点模型
"""

from .backtester import BacktestEngine

__all__ = ["BacktestEngine"]
