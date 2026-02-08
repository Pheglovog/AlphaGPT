"""
AlphaQuant - 中国股市量化交易系统

核心模块：
- data_providers: 数据提供者（Tushare Pro 等）
- data_cache: 数据缓存管理
- data_validation: 数据验证
- factors: 因子计算引擎
- model: 量化模型
- backtest: 回测引擎
- strategy: 策略管理
- metrics: 性能指标
- dashboard: 可视化面板
- execution: Solana 交易执行
"""

__version__ = "1.0.0"
__author__ = "Pheglovog"

__all__ = [
    "data_providers",
    "data_cache",
    "data_validation",
    "factors",
    "model",
    "backtest",
    "strategy",
    "metrics",
    "dashboard",
    "execution",
]
